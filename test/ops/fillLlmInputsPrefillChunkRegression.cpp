#include "models/llama.h"

#include <cmath>
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

std::vector<float> ReadFloats(const Data &data) {
    Check(data.cpuData != nullptr, "empty cpuData");
    const int n = data.Count(0);
    std::vector<float> out(n);
    const float *p = (const float *)data.cpuData;
    for (int i = 0; i < n; i++) {
        out[i] = p[i];
    }
    return out;
}

void FillChunk(LlamaModel &model,
               const std::vector<float> &tokens,
               int promptLen,
               int index,
               Data &inputIds,
               Data &attentionMask,
               Data &positionIds) {
    std::vector<std::vector<float>> inputTokens = {tokens};
    std::map<std::string, int> params;
    params["promptLen"] = promptLen;
    params["index"] = index;
    model.FillLLMInputs(inputTokens, params, inputIds, attentionMask, positionIds);
}

void ExpectPositions(const Data &positionIds,
                     int seqLen,
                     int start,
                     const std::string &label) {
    auto pids = ReadFloats(positionIds);
    Check((int)pids.size() == seqLen, label + ": pid count");
    for (int i = 0; i < seqLen; i++) {
        Check(pids[i] == (float)(start + i),
              label + ": pid[" + std::to_string(i) + "]");
    }
}

} // namespace

int main() {
    try {
        LlamaModel model;

        // First 2048 of a 4096 prompt, no prefix restore.
        // promptLen must be the absolute end of this chunk, not remaining.
        {
            const int cacheLen = 0;
            const int curLen = 2048;
            const int already = 0;
            const int promptLen = cacheLen + already + curLen;
            std::vector<float> tokens(curLen);
            for (int i = 0; i < curLen; i++) {
                tokens[i] = (float)i;
            }
            Data inputIds, attentionMask, positionIds;
            FillChunk(model, tokens, promptLen, 0, inputIds, attentionMask, positionIds);
            Check(inputIds.Count(0) == curLen, "first chunk seqLen");
            ExpectPositions(positionIds, curLen, cacheLen + already, "first chunk");
            Check(attentionMask.dims.size() == 0, "first chunk mask empty (qlen>=1024)");
        }

        // Second 2048 of the same prompt. If promptLen were cacheLen+remaining
        // (=2048) positions would restart at 0.
        {
            const int cacheLen = 0;
            const int curLen = 2048;
            const int already = 2048;
            const int promptLen = cacheLen + already + curLen;
            std::vector<float> tokens(curLen);
            for (int i = 0; i < curLen; i++) {
                tokens[i] = (float)(already + i);
            }
            Data inputIds, attentionMask, positionIds;
            FillChunk(model, tokens, promptLen, 0, inputIds, attentionMask, positionIds);
            ExpectPositions(positionIds, curLen, cacheLen + already, "second chunk");
            Check(ReadFloats(positionIds)[0] == 2048.f,
                  "second chunk must not restart at 0");
        }

        // Prefix restore of 128 then first remaining chunk.
        {
            const int cacheLen = 128;
            const int curLen = 2048;
            const int already = 0;
            const int promptLen = cacheLen + already + curLen;
            std::vector<float> tokens(curLen, 1.f);
            Data inputIds, attentionMask, positionIds;
            FillChunk(model, tokens, promptLen, 0, inputIds, attentionMask, positionIds);
            ExpectPositions(positionIds, curLen, 128, "prefix first chunk");
        }

        // Remainder-1 last chunk. seqLen==1 uses promptLen+index-1.
        // index must stay 0 or the last prompt position is wrong.
        {
            const int cacheLen = 0;
            const int already = 2048;
            const int curLen = 1;
            const int promptLen = cacheLen + already + curLen;
            std::vector<float> tokens = {9.f};
            Data inputIds, attentionMask, positionIds;
            FillChunk(model, tokens, promptLen, 0, inputIds, attentionMask, positionIds);
            Check(inputIds.Count(0) == 1, "remainder-1 seqLen");
            Check(ReadFloats(positionIds)[0] == (float)(promptLen - 1),
                  "remainder-1 last prompt position");
        }

        // First generated token after that last chunk: index becomes 1,
        // promptLen stays the full prompt end.
        {
            const int promptLen = 2049;
            std::vector<float> tokens = {3.f};
            Data inputIds, attentionMask, positionIds;
            FillChunk(model, tokens, promptLen, 1, inputIds, attentionMask, positionIds);
            Check(ReadFloats(positionIds)[0] == (float)promptLen,
                  "first decode position");
        }

        // Small chunk with a mask: promptLen is the mask width, not seqLen.
        {
            const int cacheLen = 10;
            const int already = 4;
            const int curLen = 8;
            const int promptLen = cacheLen + already + curLen;
            std::vector<float> tokens(curLen, 2.f);
            Data inputIds, attentionMask, positionIds;
            FillChunk(model, tokens, promptLen, 0, inputIds, attentionMask, positionIds);
            ExpectPositions(positionIds, curLen, cacheLen + already, "masked chunk");
            Check(attentionMask.dims.size() == 2, "masked chunk rank");
            Check(attentionMask.dims[0] == curLen && attentionMask.dims[1] == promptLen,
                  "mask is seqLen x promptLen (absolute end)");
        }

        printf("PASS: fillLlmInputsPrefillChunkRegression\n");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
