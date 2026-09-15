#pragma once

namespace fastllm {
class Data;
}

// Prepare an already materialized FP8 Linear weight for repeated small-row
// SM70 GEMM calls. This uses the existing in-place layout/FP16-scale conversion.
// The caller must select the weight's CUDA device and ensure serving is idle.
// Returns false for ineligible weights/row counts or a deferred conversion,
// without changing the weight storage.
bool FastllmCudaWarmupFp8E4M3Sm70(fastllm::Data &weight, int rows);

// Build the SM70 NVFP4 QPN2 sidecar for one Linear weight ahead of any CUDA
// Graph capture. The sidecar is a plain cudaMalloc, so building it lazily on
// the first decode call would run inside capture. The caller must select the
// weight's CUDA device and ensure serving is idle. Returns true when the
// sidecar already exists or was just built, false for ineligible weights or a
// failed build; storage and the native layout are left untouched either way.
bool FastllmCudaWarmupNvfp4Qpn2Sm70(fastllm::Data &weight);
