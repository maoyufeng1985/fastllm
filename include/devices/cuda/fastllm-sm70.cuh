//
// SM70 (V100) QPN dense GEMM bridge, torch-free.
//
// The QPN8 execution layout is derived from dnv2003/v100-skinny (MIT) and its
// block-scale adaptation in haohervchb/sglang-V100, ported from 1Cat-vLLM's
// csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu. The kernels are unchanged; only the
// host wrappers are rewritten to use raw CUDA pointers so they can be driven
// directly from FastLLM's FP8_E4M3 block-128 weights. See
// src/devices/cuda/sm70/LICENSE.v100-skinny for the retained MIT notice.
//
// This is the PR1 kernel foundation for the SM70 concurrency port. It is
// self-contained and unit-testable on a single V100. It is only meaningful on
// compute capability 7.0; on other architectures every entry point reports
// unsupported and callers keep their existing path.
//
// Rollback switches (all default ON on SM70, set to 0 to disable):
//   FASTLLM_SM70            master switch
//   FASTLLM_SM70_QPN        QPN family
//   FASTLLM_SM70_FP8_QPN8   FP8 QPN8 dense GEMM
//   FASTLLM_SM70_NVFP4_QPN2 NVFP4 QPN2 dense GEMM
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastllm {
namespace sm70 {

// True only on SM70 devices where the QPN8 kernels are compiled in and the
// FASTLLM_SM70 / FASTLLM_SM70_QPN / FASTLLM_SM70_FP8_QPN8 switches are not
// explicitly disabled.
bool Fp8QpnSupported();

// Shape gate for the FP8 QPN8 dense GEMM. M is the token/row count, K the
// input dim, N the output dim. Requires block-128 scaling (K%128==0),
// N%32==0, and 1 <= M <= 32.
bool Fp8QpnCanRun(int m, int k, int n);

// Weight preparation. qweight is the source [N, K] row-major FP8_E4M3 byte
// matrix. codes receives the QPN8 packed layout [K, N] uint8 and groupScales
// receives the packed FP16 scales. The caller owns and must allocate codes
// (K*N bytes) and groupScales before calling. All pointers are device
// pointers.
//
// The QPN8 256x scale convention is applied here and cancels internally
// against the fp8 decoder (which yields value/256), so the GEMM below has the
// plain dequantized semantics out = in @ (fp8_value * scale).
//
// Two scale layouts are supported, selected by channelScales:
//   channelScales == false (block-128, FastLLM native):
//     blockScales is [N/128, K/128] row-major FP32; groupScales is
//     [K/128, N/32] halves.
//   channelScales == true:
//     blockScales is [N, 1] row-major FP32; groupScales is [1, N] halves.
bool Fp8QpnPrepare(const uint8_t *qweight, const float *scales,
                   uint8_t *codes, half *groupScales,
                   int k, int n, bool channelScales, cudaStream_t stream);

// Dense GEMM: out[m, n] = in[m, k] @ dequant(W). in/out are row-major half.
// codes/groupScales are the buffers produced by Fp8QpnPrepare; channelScales
// must match the layout used during preparation. Returns false (without
// writing out) when the shape is not supported.
bool Fp8QpnGemm(const uint8_t *codes, const half *groupScales,
                const half *in, half *out,
                int m, int k, int n, bool channelScales, cudaStream_t stream);

// NVFP4 QPN2 dense GEMM. True only on SM70 when FASTLLM_SM70 /
// FASTLLM_SM70_QPN / FASTLLM_SM70_NVFP4_QPN2 are not explicitly disabled.
bool Nvfp4QpnSupported();

// Shape gate: 1 <= M <= 32, K % 64 == 0, N % 32 == 0, and K/16 is divisible
// by a supported split-K in {8, 16, 32}.
bool Nvfp4QpnCanRun(int m, int k, int n);

// Weight preparation. weight is the source [N, K/2] row-major packed NVFP4
// bytes (even K in the low nibble). scales is [N, K/16] row-major FP8_E4M3
// group scales. codes and packedScales receive the QPN2 fragment layout;
// the caller owns and must allocate codes (N*K/2 bytes) and packedScales
// (N*K/16 bytes). All pointers are device pointers.
bool Nvfp4QpnPrepare(const uint8_t *weight, const uint8_t *scales,
                     uint8_t *codes, uint8_t *packedScales,
                     int k, int n, cudaStream_t stream);

// FastLLM native NVFP4_BLOCK_16 is interleaved [N, (K/16)*12]: eight packed
// E2M1 bytes plus one fused FP32 group scale (checkpoint E4M3 * globalScale).
// This converts that layout into the QPN2 fragment layout (codes then raw
// E4M3) by unfusing `fp32 / globalScale`, matching 1Cat-vLLM's raw-E4M3 +
// weight_global_scale contract. GEMM must be called with the same globalScale.
// Failure leaves the source untouched. Persistent size is N*K*9/16, which
// fits in the native N*K*12/16 allocation. `dest == nullptr` overwrites
// `storage` in place; otherwise packed codes/scales are written to `dest`
// and the native source is preserved for the large-M path.
bool Nvfp4QpnPrepareFromNative(uint8_t *storage, size_t storageBytes,
                               int k, int n, float globalScale,
                               cudaStream_t stream, uint8_t *dest = nullptr,
                               size_t destBytes = 0);

// QPN2 persistent bytes: packed codes (N*K/2) followed by E4M3 scales (N*K/16).
inline size_t Nvfp4QpnPackedBytes(int k, int n) {
  return static_cast<size_t>(n) * k / 2 + static_cast<size_t>(n) * k / 16;
}

// Dense GEMM: out[m, n] = in[m, k] @ dequant(W) * globalScale.
// codes/packedScales are the buffers produced by Nvfp4QpnPrepare.
// Returns false (without writing out) when the shape is not supported.
bool Nvfp4QpnGemm(const uint8_t *codes, const uint8_t *packedScales,
                  const half *in, half *out,
                  int m, int k, int n, float globalScale, cudaStream_t stream);

}  // namespace sm70
}  // namespace fastllm
