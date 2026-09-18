#ifndef FASTLLM_LONG_PREFILL_CHUNK_H
#define FASTLLM_LONG_PREFILL_CHUNK_H

#include "basellm.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>

namespace fastllm {

inline bool LongPrefillChunkEnabled(int prefillChunkSize) {
    if (prefillChunkSize <= 0) {
        return false;
    }
    const char *env = std::getenv("FASTLLM_LONG_PREFILL_CHUNK");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::atoi(env) != 0;
}

struct RequestRole {
    bool stillPrefilling = false;
    bool isPromptEligible = false;
    bool isDecodeActive = false;
    // Already started a chunked prompt. pagesLimit throttles *new*
    // prefills; blocking this would stall the remaining chunks forever.
    bool inFlightPrefill = false;
};

inline RequestRole ClassifyRequest(const ResponseContext *ctx,
                                   bool longPrefillChunk) {
    RequestRole role;
    if (ctx == nullptr) {
        return role;
    }
    role.stillPrefilling = longPrefillChunk && ctx->prefillRemaining > 0;
    role.isPromptEligible = ctx->preTokens == 0 || role.stillPrefilling;
    role.isDecodeActive = ctx->preTokens > 0 && !role.stillPrefilling;
    role.inFlightPrefill = role.stillPrefilling && ctx->preTokens > 0;
    return role;
}

inline void ArmChunkedPrefill(ResponseContext *ctx, int chunkSize) {
    if (ctx == nullptr || ctx->prefillRemaining > 0 || chunkSize <= 0) {
        return;
    }
    const int n = (int)ctx->currentTokens.size();
    if (n > chunkSize) {
        ctx->prefillRemaining = n;
    }
}

inline int SelectPrefillChunkLen(const ResponseContext *ctx,
                                 bool longPrefillChunk,
                                 int prefillChunkSize,
                                 int alreadyInBatch,
                                 int prefillTokenCount,
                                 int currentPrefillTokenLimit) {
    if (ctx == nullptr) {
        return 0;
    }
    const int remaining = ctx->prefillRemaining > 0 ?
        ctx->prefillRemaining : (int)ctx->currentTokens.size();
    if (!longPrefillChunk) {
        return remaining;
    }
    int curLen = remaining;
    if (prefillChunkSize > 0) {
        curLen = std::min(curLen, prefillChunkSize);
    }
    if (alreadyInBatch > 0) {
        curLen = std::min(curLen, currentPrefillTokenLimit - prefillTokenCount);
    }
    if (curLen <= 0) {
        return 0;
    }
    // seqLen==1 is the decode / CUDA-graph path. A leftover-1 last
    // chunk, or a 1-token leftover budget, must not look like decode.
    // 线性注意力快照只能落在页边界上，所以让"最后一块"停在提示词的最后一个整页边界 A，
    // 而不是停在总长上。否则 A 之后最多 pageLen-1 个 token 的尾巴每轮都要重算，
    // 而且它只能以小分块跑（实测 880 tokens/s 对 2400+），正好压在首字延迟上。
    // 必须限定 prefillRemaining > 0：只有被分块的请求才有"下一轮"去吃掉尾巴。
    // 未分块的一次性 prefill（prefillRemaining == 0）里 remaining 就是 currentTokens.size()，
    // 照截不误会把 A..total 的尾巴静默丢掉，模型看到的是被截断的提示词（实测过）。
    if (curLen == remaining && ctx->prefillRemaining > 0) {
        const int total = (int)ctx->allTokens.size();
        const int fed = total - remaining;
        const int pageLen = fastllm::GetPageLen();
        const int lastBoundary = (total / pageLen) * pageLen;
        const int alignedLen = lastBoundary - fed;
        const int tailLen = remaining - alignedLen;
        // 尾巴必须 >= 2：剩下 1 个 token 的尾巴会撞上下面两道 seqLen==1 守卫，
        // 要么被丢掉要么被当成解码步。alignedLen <= 1 同理。
        if (pageLen > 1 && total > 0 && fed >= 0 && (total % pageLen) != 0 &&
            alignedLen > 1 && alignedLen < curLen && tailLen >= 2) {
            curLen = alignedLen;
        }
    }
    if (curLen == 1 && remaining > 1) {
        return 0;
    }
    if (remaining > curLen && remaining - curLen == 1) {
        const int leftover = alreadyInBatch > 0 ?
            currentPrefillTokenLimit - prefillTokenCount : remaining;
        if (leftover >= remaining) {
            curLen = remaining;
        } else {
            return 0;
        }
    }
    return curLen;
}

// Round-robin ticket for in-flight prefills. Opt-in (FASTLLM_PREFILL_ROTATE=1)
// because it reverses the deliberate starvation in PrefillOrderSortKey below:
// with two long prompts, the default lets one run to completion and only then
// starts the peer, so the later request waits for the whole first prefill.
// Rotating at *chunk* granularity gives each request one chunk in turn, which
// keeps both in flight; it does not split a chunk between requests, so it is
// not the slice-level ping-pong that comment warns about.
inline unsigned long long NextPrefillTicket() {
    static std::atomic<unsigned long long> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

// Test seam. The env lookup below caches on first call, so a test cannot rely
// on setenv; it sets this instead. Unset (-1) means "follow the environment".
inline int &PrefillRotationOverride() {
    static int override = -1;
    return override;
}

inline bool PrefillRotationEnabled() {
    if (PrefillRotationOverride() >= 0) {
        return PrefillRotationOverride() != 0;
    }
    static const bool enabled = [] {
        const char *env = std::getenv("FASTLLM_PREFILL_ROTATE");
        return env != nullptr && env[0] != '0';
    }();
    return enabled;
}

inline int PrefillOrderSortKey(const ResponseContext *ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    const int remaining = (int)ctx->currentTokens.size();
    // Only a request that has already fed tokens is in-flight. Arming a
    // peer that then loses the token-budget skip must not let it jump
    // the queue (two 80K jobs would otherwise ping-pong 2048 slices).
    if (ctx->prefillRemaining > 0 && ctx->preTokens > 0) {
        if (PrefillRotationEnabled()) {
            return -(remaining + 1000000000) + ctx->prefillTicket;
        }
        return -(remaining + 1000000000);
    }
    return -remaining;
}

inline bool PrefixRestoreAllowed(const ResponseContext *ctx) {
    return ctx != nullptr &&
        ctx->cacheLen == 0 &&
        ctx->preTokens == 0 &&
        ctx->prefillRemaining == 0;
}

// Decode yield between outer prefill chunks. Chunking without this
// turns one long stall into many short stalls: MTPLoop prefers any
// pending prefill and never mixed-batches.
inline bool ShouldForceDecodeThisIteration(int decodeActiveCount,
                                           bool hasPrefill,
                                           bool activePrefillNeedsDecode) {
    return decodeActiveCount > 0 && hasPrefill && activePrefillNeedsDecode;
}

inline bool ShouldArmPrefillYield(int decodeActiveBeforeSelection,
                                  int selectedCount,
                                  int orderCount) {
    if (decodeActiveBeforeSelection > 0) {
        return true;
    }
    return selectedCount > 0 && selectedCount < orderCount;
}

inline bool CommitIntermediatePrefillChunk(ResponseContext *ctx, int fed) {
    if (ctx == nullptr || ctx->prefillRemaining <= fed) {
        if (ctx != nullptr && ctx->prefillRemaining > 0) {
            ctx->prefillRemaining = 0;
        }
        return false;
    }
    const int curLen = std::min(fed, (int)ctx->currentTokens.size());
    if (curLen > 0) {
        ctx->currentTokens.erase(
            ctx->currentTokens.begin(),
            ctx->currentTokens.begin() + curLen);
        ctx->prefillRemaining -= curLen;
    }
    if (ctx->prefillRemaining < 0) {
        ctx->prefillRemaining = 0;
    }
    return true;
}

}  // namespace fastllm

#endif
