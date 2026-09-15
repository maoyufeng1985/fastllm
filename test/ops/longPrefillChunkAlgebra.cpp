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
