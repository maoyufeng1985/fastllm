#ifndef FASTLLM_LONG_PREFILL_CHUNK_H
#define FASTLLM_LONG_PREFILL_CHUNK_H

#include "basellm.h"

#include <algorithm>
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

inline int PrefillOrderSortKey(const ResponseContext *ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    const int remaining = (int)ctx->currentTokens.size();
    // Only a request that has already fed tokens is in-flight. Arming a
    // peer that then loses the token-budget skip must not let it jump
    // the queue (two 80K jobs would otherwise ping-pong 2048 slices).
    if (ctx->prefillRemaining > 0 && ctx->preTokens > 0) {
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
