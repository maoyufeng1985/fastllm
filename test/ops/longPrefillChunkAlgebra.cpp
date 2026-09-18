#include "longPrefillChunk.h"

#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;

namespace {

void Check(bool ok, const std::string &message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

ResponseContext MakePrompt(int n, int preTokens = 0, int remaining = 0) {
    ResponseContext ctx;
    ctx.currentTokens.assign(n, 1);
    ctx.preTokens = preTokens;
    ctx.prefillRemaining = remaining;
    ctx.cacheLen = 0;
    return ctx;
}

// allTokens is the one field the real engine keeps at the prompt's full length
// for the whole prefill, so the alignment rule has to be exercised through it.
// fed is how much of the prompt has already been fed (prefillRemaining is what
// is left), which is all SelectPrefillChunkLen may look at to find the page
// boundary the last chunk should stop on.
ResponseContext MakeChunkedPrefill(int total, int fed) {
    ResponseContext ctx;
    const int remaining = total - fed;
    ctx.allTokens.assign(total, 1);
    ctx.currentTokens.assign(remaining, 1);
    ctx.preTokens = fed;
    ctx.prefillRemaining = remaining;
    ctx.cacheLen = 0;
    return ctx;
}

} // namespace

int main() {
    try {
        {
            auto ctx = MakePrompt(80);
            auto role = ClassifyRequest(&ctx, true);
            Check(role.isPromptEligible && !role.isDecodeActive &&
                      !role.stillPrefilling,
                  "fresh prompt is prefill, not decode");
        }

        {
            auto ctx = MakePrompt(80000 - 2048, 2048, 80000 - 2048);
            auto role = ClassifyRequest(&ctx, true);
            Check(role.stillPrefilling && role.isPromptEligible &&
                      !role.isDecodeActive && role.inFlightPrefill,
                  "mid-prompt must not look decode-active");
            auto off = ClassifyRequest(&ctx, false);
            Check(!off.stillPrefilling && off.isDecodeActive &&
                      !off.isPromptEligible && !off.inFlightPrefill,
                  "switch off restores preTokens classification");
        }

        {
            auto fresh = MakePrompt(80000);
            auto role = ClassifyRequest(&fresh, true);
            Check(!role.inFlightPrefill, "fresh prompt is not in-flight");
        }

        {
            auto ctx = MakePrompt(1, 80000, 0);
            auto role = ClassifyRequest(&ctx, true);
            Check(role.isDecodeActive && !role.isPromptEligible,
                  "last chunk committed is decode");
        }

        {
            auto exact = MakePrompt(2048);
            ArmChunkedPrefill(&exact, 2048);
            Check(exact.prefillRemaining == 0, "exact chunk size is one-shot");
            auto over = MakePrompt(2049);
            ArmChunkedPrefill(&over, 2048);
            Check(over.prefillRemaining == 2049, "oversize prompt is armed");
        }

        {
            auto ctx = MakePrompt(80000, 0, 80000);
            Check(SelectPrefillChunkLen(&ctx, true, 2048, 0, 0, 2048) == 2048,
                  "first in batch is one chunk");
            Check(SelectPrefillChunkLen(&ctx, true, 2048, 1, 2048, 2048) == 0,
                  "2048 token budget cannot take a second 2048");
            Check(SelectPrefillChunkLen(&ctx, true, 2048, 1, 2048, 4096) == 2048,
                  "4096 budget admits two 2048 chunks");
            Check(SelectPrefillChunkLen(&ctx, false, 2048, 0, 0, 2048) == 80000,
                  "switch off submits the whole prompt");
            auto odd = MakePrompt(2049, 0, 2049);
            Check(SelectPrefillChunkLen(&odd, true, 2048, 0, 0, 4096) == 2049,
                  "absorb remainder-1 instead of a seqLen=1 last chunk");
            Check(SelectPrefillChunkLen(&odd, true, 2048, 1, 2048, 4096) == 0,
                  "later-in-batch leftover 2048 of 2049 skips instead of seqLen=1");
            Check(SelectPrefillChunkLen(&odd, true, 2048, 1, 4095, 4096) == 0,
                  "1-token leftover of a longer prompt is not a prefill chunk");
        }

        {
            auto ctx = MakePrompt(4096, 0, 4096);
            Check(CommitIntermediatePrefillChunk(&ctx, 2048),
                  "intermediate chunk stays in prefill");
            Check((int)ctx.currentTokens.size() == 2048, "erase fed prefix");
            Check(ctx.prefillRemaining == 2048, "remaining shrinks");
            Check(!CommitIntermediatePrefillChunk(&ctx, 2048),
                  "last chunk falls through to assign");
            Check(ctx.prefillRemaining == 0, "last chunk clears remaining");
        }

        // 末块停在提示词的最后一个页边界 A = floor(total/pageLen)*pageLen。
        // 这样 A 之后那点尾巴（<= pageLen-1）留给下一轮，A 本身能成为快照点。
        // pageLen 在 CPU 单测里是默认值 128（fastllm.cpp 的 defaultPageLen）。
        {
            const int pageLen = 128;
            // 冷启动 15866：喂到 14336 后，末块应从 1530 缩到 15744-14336=1408。
            auto cold = MakeChunkedPrefill(15866, 14336);
            Check(SelectPrefillChunkLen(&cold, true, 2048, 0, 0, 8192) ==
                      15866 / pageLen * pageLen - 14336,
                  "last chunk stops at the prompt's last page boundary");

            // 缩完是 1408，剩下 122 的尾巴（<= pageLen-1），下一轮继续吃。
            Check(SelectPrefillChunkLen(&cold, true, 2048, 0, 0, 8192) == 1408,
                  "aligned last chunk is 1408 for a 15866 prompt fed to 14336");

            // 尾巴那一轮：fed=15744，remaining=122，已经没有页边界可停，不能再缩。
            auto tail = MakeChunkedPrefill(15866, 15744);
            Check(SelectPrefillChunkLen(&tail, true, 2048, 0, 0, 8192) == 122,
                  "the leftover tail is fed whole, not shrunk to nothing");

            // 提示词总长正好是页的整数倍：没有尾巴，末块长度不变。
            auto exact = MakeChunkedPrefill(15872, 14336);
            Check(SelectPrefillChunkLen(&exact, true, 2048, 0, 0, 8192) == 1536,
                  "a prompt already on the page grid is left alone");

            // 尾巴只剩 1 个 token 时不许缩：total=15873 -> A=15872, fed=15744,
            // alignedLen=128, tailLen=129-128=1。留 1 个 token 的末块会被当成解码步。
            auto oneTail = MakeChunkedPrefill(15873, 15744);
            Check(SelectPrefillChunkLen(&oneTail, true, 2048, 0, 0, 8192) == 129,
                  "a 1-token tail is not created by shrinking");

            // 未分块的一次性 prefill：prefillRemaining == 0，绝不能缩，否则尾巴被静默丢弃。
            // 短提示词（<= chunkSize）走的就是这条路径。
            auto shortPrompt = MakePrompt(266);
            Check(SelectPrefillChunkLen(&shortPrompt, true, 2048, 0, 0, 8192) == 266,
                  "an un-chunked short prompt is never truncated");
            auto shortOdd = MakePrompt(176);
            Check(SelectPrefillChunkLen(&shortOdd, true, 2048, 0, 0, 8192) == 176,
                  "an un-chunked 176-token prompt is fed whole");
        }

        {
            auto inflight = MakePrompt(80000 - 2048, 2048, 80000 - 2048);
            auto fresh = MakePrompt(80000);
            Check(PrefillOrderSortKey(&inflight) < PrefillOrderSortKey(&fresh),
                  "in-flight long prefill sorts ahead of a new 80K");
            auto armedUnfed = MakePrompt(80000, 0, 80000);
            Check(PrefillOrderSortKey(&inflight) < PrefillOrderSortKey(&armedUnfed),
                  "armed but unfed peer must not jump a real in-flight");
        }

        {
            auto fresh = MakePrompt(80000);
            Check(PrefixRestoreAllowed(&fresh),
                  "fresh prompt may restore a prefix");
            auto mid = MakePrompt(80000 - 2048, 2048, 80000 - 2048);
            Check(!PrefixRestoreAllowed(&mid),
                  "mid-prompt must not Query the unfed suffix");
        }

        {
            Check(!ShouldForceDecodeThisIteration(0, true, true),
                  "first 80K must drain its own chunks with no decode-active peer");
            Check(ShouldForceDecodeThisIteration(1, true, true),
                  "after #0's last chunk, #1's chunks yield one decode token");
            Check(!ShouldForceDecodeThisIteration(1, true, false),
                  "yield is latched only after a prefill select");
            Check(ShouldArmPrefillYield(1, 1, 2),
                  "decode-active peer arms yield after a prefill select");
            Check(!ShouldArmPrefillYield(0, 1, 1),
                  "lone first prompt does not arm yield");
            Check(ShouldArmPrefillYield(0, 1, 2),
                  "idle burst that did not fit arms yield for later");
        }

        {
            Check(!LongPrefillChunkEnabled(0), "chunk size 0 disables");
            Check(LongPrefillChunkEnabled(2048), "unset env enables");
        }

        {
            // Two in-flight prefills with equal progress. Under the default
            // ordering the ticket is ignored, so one request prefill all the
            // way through before its peer starts.
            auto a = MakePrompt(20000 - 2048, 2048, 20000 - 2048);
            auto b = MakePrompt(20000 - 2048, 2048, 20000 - 2048);
            a.prefillTicket = 7;
            b.prefillTicket = 3;
            PrefillRotationOverride() = 0;
            Check(PrefillOrderSortKey(&a) == PrefillOrderSortKey(&b),
                  "rotation off: equal progress ignores the ticket");

            // Rotation on: the older ticket wins even at equal progress, so
            // the peer gets a chunk instead of waiting out the whole prefill.
            PrefillRotationOverride() = 1;
            Check(PrefillOrderSortKey(&b) < PrefillOrderSortKey(&a),
                  "rotation on: the older ticket sorts first");
            b.prefillTicket = 8;
            Check(PrefillOrderSortKey(&a) < PrefillOrderSortKey(&b),
                  "rotation on: the order flips as tickets advance");

            // Rotation must not touch the fresh-prompt branch.
            auto freshLong = MakePrompt(90000);
            auto freshShort = MakePrompt(100);
            freshLong.prefillTicket = 9;
            freshShort.prefillTicket = 1;
            Check(PrefillOrderSortKey(&freshLong) < PrefillOrderSortKey(&freshShort),
                  "fresh prompts still order by length under rotation");

            // A lone in-flight prefill cannot be reordered by its ticket,
            // because there is no peer to prefer over it.
            auto solo = MakePrompt(80000 - 2048, 2048, 80000 - 2048);
            solo.prefillTicket = 42;
            Check(PrefillOrderSortKey(&solo) < PrefillOrderSortKey(&freshShort),
                  "a lone in-flight prefill still leads every fresh prompt");

            unsigned long long first = NextPrefillTicket();
            unsigned long long second = NextPrefillTicket();
            Check(second == first + 1, "tickets advance by one per stamp");
            PrefillRotationOverride() = -1;
        }

        printf("PASS: longPrefillChunkAlgebra\n");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
