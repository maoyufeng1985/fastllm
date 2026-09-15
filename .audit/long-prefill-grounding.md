# Grounding: RunNewMainLoop long-prefill chunk (PR-A)

Source of truth: `docs/sm70_long_prefill_chunk_plan.md` plus the live code below.

## What the scheduler does today

`basellm::RunNewMainLoop` (`src/models/basellm.cpp:1296`) is the qwen3_5 no-MTP path (`use_new_engine=true`, `defaultChunkedPrefillSize=2048`).

Each iteration:

1. Sort requests by `-currentTokens.size()` (longest first) at 1727.
2. `hasPrefill` / `currentActivate` / idle-burst membership all key off `preTokens == 0` (1721-1727, 1788, 1799).
3. Two-phase select: `isPrompt=1` then `isPrompt=0`. Decode is skipped if any prefill was already selected (1837). `forceDecodeThisIteration` skips the prefill phase after an add-prefill so decode can run (1834).
4. Prefill `thisLen = currentTokens.size()` (2210). If `thisLen > prefillChunkSize` the request is admitted only when it is the first in the batch, **whole**. No slice.
5. `FillLLMInputs` gets the entire `currentTokens`. `promptLen = cacheLen + currentTokens.size()`, `index = 0` only when `preTokens == 0`; otherwise `index++` (2288-2294).
6. `preTokens += seqLens.back()` immediately after fill (2324), before Forward.
7. Inner loop at 2500 splits a *single already-built* 80K `inputIds` under `forwardLocker`. Interleave cannot run between those splits.
8. After Forward, every request does `currentTokens.assign(1, curRet)`, push to `resultTokenQueue` / `allTokens`, `curTokens++`, eos checks (2633-2674). Intermediate lm_head output is treated as a generated token.

`FillLLMInputs` (`3405`): for `seqLen > 1`, `pid[i] = promptLen - seqLen + i`. For `seqLen == 1`, `pid = promptLen + index - 1`. qwen3_5 `NeedAttentionMask` is always false.

`isIntermediateChunkedPrefill` is a process-wide bool on `basellm`. qwen3_5 skips lm_head when it is set and greedy (`qwen3_5.cpp:15100`).

Request create (`3184`): `currentTokens = inputTokens`. Prefix restore may raise `cacheLen` and shrink `currentTokens`.

`releaseAndReinitRequest` (1342) clears `intParams`, `preTokens`, `cacheLen` and restores `currentTokens = allTokens`.

## Load-bearing constraints the plan under-stated

1. **`preTokens` is not only `isPrompt`.** After the first 2048 chunk, `preTokens > 0` so:
   - the request counts as `currentActivate` (decode-alive)
   - it drops out of `hasPrefill`
   - it is erased from `idleBurstPrefillHandles`
   Combined with `forceDecodeThisIteration`, a still-prefilling request can be skipped in both phases and stall. C=2 both-long-prompt is the failure mode.
   Fix: "still prefilling" must count as prefill, not as decode-active.

2. **Do not increment `index` on prefill chunks.** A remainder-1 last chunk takes the `seqLen==1` FillLLMInputs branch. If `index` was bumped per chunk, first-token position is wrong. Keep `index = 0` and `promptLen = cacheLen + consumedBefore + curLen` through the last prefill chunk. First real decode then does `index++` against the full prompt length, matching today.

3. **Last-chunk commit vs intermediate.** Intermediate: erase `currentTokens[0, curLen)`, do not push/assign/eos/`curTokens++`. Last chunk: existing assign(1, curRet) path. `allTokens` already holds the prompt from create; do not push prompt pieces.

4. **Page accounting uses `appendTokens`.** Pass `curLen`, not remaining full prompt. Sliding-window layers already cap at `retained + prefillChunkSize`.

5. **Scope.** Only `RunNewMainLoop`. Do not touch `Qwen35MTPLoop`. Leave the 2500 inner loop as the `FASTLLM_LONG_PREFILL_CHUNK=0` fallback. No mixed batch. CUDA graph stays `dims[1]==1`.

6. **Switch.** Unset / non-zero enables selection chunking. `0` or `prefillChunkSize <= 0` restores today.

## Existing types

```
ResponseContext {
  currentTokens;   // remaining prompt during prefill; one decode token after
  allTokens;       // prompt at create, then generated tokens appended
  preTokens;       // tokens fed to the model (prompt chunks + decode)
  curTokens;       // generated count
  cacheLen;        // restored prefix
  intParams;       // promptLen, index, add_special_tokens, ...
  resultTokenQueue;
}
```

No `prefillRemaining` today.

## What "still prefilling" must mean

```
stillPrefilling <=> longPrefillChunk enabled
                    AND remaining unfed prompt tokens > 0
hasPrefill      <=> any request with stillPrefilling OR preTokens==0
currentActivate <=> preTokens>0 AND NOT stillPrefilling   // real decode
isPrompt select <=> stillPrefilling OR preTokens==0
```

`promptLen` for a chunk = `cacheLen + (originalRemaining - remaining) + curLen`
= absolute position of the last token in this chunk.

## Non-goals

Mixed prefill/decode batch. MTP loop. Changing graph capture. New chunk-size knob (reuse `GetChunkedPrefillSize()`).
