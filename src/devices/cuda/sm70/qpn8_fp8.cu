//
// SM70 (V100) QPN8 FP8 dense GEMM, torch-free.
//
// The QPN8 execution layout is derived from dnv2003/v100-skinny (MIT) and its
// block-scale adaptation in haohervchb/sglang-V100, ported from 1Cat-vLLM's
// csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu. The device kernels are unchanged;
// only the host wrappers are rewritten to use raw CUDA pointers so they can be
// driven directly from FastLLM's FP8_E4M3 block-128 weights. See
// LICENSE.v100-skinny in this directory for the retained MIT notice.
//
// This is the PR1 kernel foundation for the SM70 concurrency port. It is
// self-contained and unit-testable on a single V100. On non-SM70 devices every
// entry point reports unsupported and callers keep their existing path.
//

#include "devices/cuda/fastllm-sm70.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastllm {
namespace sm70 {

namespace {

__device__ __forceinline__ void fp8x8_to_half2x4(uint2 q, half2 out[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned b0 = (q.x >> (8 * i)) & 0xffu;
    const unsigned b1 = (q.y >> (8 * i)) & 0xffu;
    const unsigned h0 = ((b0 & 0x80u) << 8) | ((b0 & 0x7fu) << 7);
    const unsigned h1 = ((b1 & 0x80u) << 8) | ((b1 & 0x7fu) << 7);
    const unsigned packed = h0 | (h1 << 16);
    out[i] = *reinterpret_cast<const half2*>(&packed);
  }
}

__device__ __forceinline__ void fp8x8_to_half2x4_fast(uint2 q, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x3f803f80u;
  unsigned permuted[4];
  permuted[0] = __byte_perm(q.x, q.y, 0x0400);
  permuted[1] = __byte_perm(q.x, q.y, 0x0501);
  permuted[2] = __byte_perm(q.x, q.y, 0x0602);
  permuted[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned value =
        ((permuted[i] << 8) & kSign) | ((permuted[i] << 7) & kExponentMantissa);
    out[i] = *reinterpret_cast<const half2*>(&value);
  }
}

constexpr int kQpn8PrepareThreads = 256;

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn8_lane_from_col(int col) {
  return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}

__device__ __forceinline__ int qpn8_physical_k(int logical_k) {
  const int local = logical_k & 7;
  return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

__global__ void fp8_qpn8_prepack_sm70_kernel(
    uint8_t* __restrict__ codes, const uint8_t* __restrict__ qweight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k;
  if (index >= numel) {
    return;
  }

  const int row = static_cast<int>(index / k);
  const int logical_k = static_cast<int>(index % k);
  const int tile = row >> 5;
  const int lane = qpn8_lane_from_col(row & 31);
  const int group = logical_k >> 4;
  const int physical_k = qpn8_physical_k(logical_k & 15);
  const int groups_k16 = k >> 4;
  const size_t packed_index =
      (((static_cast<size_t>(tile) * groups_k16 + group) * 32 + lane) * 16 +
       physical_k);
  codes[packed_index] = qweight[index];
}

__global__ void fp8_qpn8_scale_sm70_kernel(half* __restrict__ group_scales,
                                           const float* __restrict__ scales,
                                           int n_blocks, int k_blocks) {
  const int n_tiles = n_blocks * 4;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int numel = k_blocks * n_tiles;
  if (index >= numel) {
    return;
  }

  const int k_block = index / n_tiles;
  const int n_tile = index - k_block * n_tiles;
  group_scales[index] =
      __float2half(scales[(n_tile >> 2) * k_blocks + k_block] * 256.0f);
}

__global__ void fp8_qpn8_channel_scale_sm70_kernel(
    half* __restrict__ channel_scales, const float* __restrict__ scales,
    int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n) {
    channel_scales[col] = __float2half(scales[col] * 256.0f);
  }
}

__global__ void fp8_qpn8_dequantize_sm70_kernel(
    half* __restrict__ output, const uint8_t* __restrict__ codes,
    const half* __restrict__ group_scales, int n, int k, bool channel_scales) {
  const size_t word_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t word_count = static_cast<size_t>(n) * k / 16;
  if (word_index >= word_count) {
    return;
  }

  const int groups_k16 = k >> 4;
  const int lane = static_cast<int>(word_index & 31);
  const size_t tile_word = word_index >> 5;
  const int group = static_cast<int>(tile_word % groups_k16);
  const int tile = static_cast<int>(tile_word / groups_k16);
  const int col = tile * 32 + qpn8_col_from_lane(lane);
  const int tiles_n32 = n >> 5;
  const half scale = channel_scales
                         ? group_scales[col]
                         : group_scales[(group >> 3) * tiles_n32 + tile];
  const half2 scale2 = __halves2half2(scale, scale);
  const uint4 packed = reinterpret_cast<const uint4*>(codes)[word_index];
  half2 weights[8];
  fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
  fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const half2 value = __hmul2(weights[pair], scale2);
    const int k_base = group * 16 + pair * 2;
    output[static_cast<size_t>(k_base) * n + col] = __low2half(value);
    output[static_cast<size_t>(k_base + 1) * n + col] = __high2half(value);
  }
}

__global__ void fp8_qpn8_silu_and_mul_sm70_kernel(
    half* __restrict__ output, const half* __restrict__ gate_up, int rows,
    int hidden) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const half* row_input = gate_up + static_cast<size_t>(row) * hidden * 2;
  half* row_output = output + static_cast<size_t>(row) * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float gate = __half2float(row_input[col]);
    const float silu = gate / (1.0f + __expf(-gate));
    row_output[col] = __hmul(__float2half(silu), row_input[hidden + col]);
  }
}

#define FASTLLM_SM70_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false, bool FusedBA = false, bool SplitOutputs = false,
          int RowTiles = 1>
__global__ void fp8_qpn8_sm70_kernel(
    const uint8_t* __restrict__ codes, const half* __restrict__ group_scales,
    const half* __restrict__ input, half* __restrict__ output,
    half* __restrict__ z_output, const half* __restrict__ ba_weight,
    half* __restrict__ ba_output, half* __restrict__ b_output,
    half* __restrict__ a_output, int ba_n, int qkv_n, int n, int k, int m,
    bool channel_scales) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "QPN8 supports one or two 8-row tiles");
  static_assert(!M1Only || RowTiles == 1,
                "QPN8 M=1 specialization uses one row tile");
  __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  if constexpr (FusedBA) {
    const int qpn_tiles = n >> 5;
    if (tile >= qpn_tiles) {
      constexpr int kBARowsPerBlock = 2;
      constexpr int kBAThreadsPerRow = 256;
      const int ba_block = tile - qpn_tiles;
      const int ba_group = threadIdx.x / kBAThreadsPerRow;
      const int ba_thread = threadIdx.x % kBAThreadsPerRow;
      const int ba_warp = ba_thread >> 5;
      const int ba_row = ba_block * kBARowsPerBlock + ba_group;
      float value = 0.0f;
      const half2* input2 = reinterpret_cast<const half2*>(input);
      const half2* weight2 = reinterpret_cast<const half2*>(
          ba_weight + static_cast<size_t>(ba_row) * k);
      for (int pair = ba_thread; pair < k / 2; pair += kBAThreadsPerRow) {
        const float2 x = __half22float2(__ldg(input2 + pair));
        const float2 weight = __half22float2(__ldg(weight2 + pair));
        value = fmaf(x.x, weight.x, value);
        value = fmaf(x.y, weight.y, value);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
      }
      if (lane == 0) {
        partials[ba_group][ba_warp] = value;
      }
      __syncthreads();
      if (ba_warp == 0) {
        value = lane < 8 ? partials[ba_group][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0 && ba_row < ba_n) {
          if constexpr (SplitOutputs) {
            if (ba_row < ba_n / 2) {
              b_output[ba_row] = __float2half(value);
            } else {
              a_output[ba_row - ba_n / 2] = __float2half(value);
            }
          } else {
            ba_output[ba_row] = __float2half(value);
          }
        }
      }
      return;
    }
  }
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const int tiles_n32 = n >> 5;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr = group_scales + tile;

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][chain][i] = 0.0f;
      }
    }
  }
  int loaded_scale_group = -1;
  half loaded_scale =
      channel_scales
          ? __ldg(group_scales + tile * 32 + qpn8_col_from_lane(lane))
          : __float2half(0.0f);
  uint4 prefetched = make_uint4(0, 0, 0, 0);
  if constexpr (PrefetchCodes) {
    prefetched = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const int scale_group = group >> 3;
    if (!channel_scales && scale_group != loaded_scale_group) {
      loaded_scale =
          __ldg(scale_ptr + static_cast<size_t>(scale_group) * tiles_n32);
      loaded_scale_group = scale_group;
    }

    const uint4 packed =
        PrefetchCodes ? prefetched
                      : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    uint4 next = make_uint4(0, 0, 0, 0);
    if constexpr (PrefetchCodes) {
      if (group + 1 < group_begin + groups_per_warp) {
        next = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
      }
    }
    half2 weights[8];
    if constexpr (FastDecoder) {
      fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
    } else {
      fp8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);
    }

    const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      weights[i] = __hmul2(weights[i], scale2);
    }

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int input_row_idx = row_tile * 8 + row;
      if (input_row_idx < m) {
        const half* input_row = input + static_cast<size_t>(input_row_idx) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }

      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      FASTLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      FASTLLM_SM70_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      FASTLLM_SM70_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      FASTLLM_SM70_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
    if constexpr (PrefetchCodes) {
      prefetched = next;
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][0][i] += accum[row_tile][chain][i];
      }
    }
  }

  if constexpr (M1Only) {
    if ((lane & 17) == 0) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
        for (int offset = 0; offset < 2; ++offset) {
          const int i = pair * 4 + offset;
          const int output_col =
              offset | (((lane >> 1) & 1) << 1) | (pair << 2);
          partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
        }
      }
    }
  } else {
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int output_row =
            row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col =
            (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partials[warp][output_row * 32 + quadpair * 8 + output_col] =
            accum[row_tile][0][i];
      }
    }
  }
  __syncthreads();

  constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
  for (int element = threadIdx.x; element < kOutputElements;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    if constexpr (M1Only) {
      const int output_col = tile * 32 + element;
      if constexpr (SplitOutputs) {
        if (output_col < qkv_n) {
          output[output_col] = __float2half(value);
        } else {
          z_output[output_col - qkv_n] = __float2half(value);
        }
      } else {
        output[output_col] = __float2half(value);
      }
    } else {
      const int output_row = element >> 5;
      const int output_col = element & 31;
      if (output_row < m) {
        output[static_cast<size_t>(output_row) * n + tile * 32 + output_col] =
            __float2half(value);
      }
    }
  }
}

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false, int RowTiles = 1>
void launch_fp8_qpn8_sm70(const uint8_t* codes, const half* group_scales,
                          const half* input, half* output, int n, int k, int m,
                          bool channel_scales, cudaStream_t stream) {
  fp8_qpn8_sm70_kernel<SplitK, NAcc, FastDecoder, PrefetchCodes, M1Only, false,
                       false, RowTiles><<<(n / 32), (32 * SplitK), 0, stream>>>(
      codes, group_scales, input, output, nullptr, nullptr, nullptr, nullptr,
      nullptr, 0, n, n, k, m, channel_scales);
}

// M32 needs four independent 8-row accumulator tiles. Keeping all logical
// split-K warps resident would require 64 KiB of static reduction storage for
// split-16. Instead, half as many physical warps execute the original logical
// warp ranges in two ordered phases. The compact first-half sum lets the final
// reduction retain the exact p0 + ... + p(SplitK-1) order while using only
// 28 KiB (split-12) or 36 KiB (split-16) of shared memory. Each output CTA
// streams its packed weight tile once and reuses it across all 32 rows.
template <int SplitK>
__global__ void fp8_qpn8_m32_twophase_sm70_kernel(
    const uint8_t* __restrict__ codes, const half* __restrict__ channel_scales,
    const half* __restrict__ input, half* __restrict__ output, int n, int k,
    int m) {
  static_assert(SplitK == 12 || SplitK == 16,
                "M32 two-phase QPN8 supports split-12 or split-16");
  constexpr int kPhysicalWarps = SplitK / 2;
  constexpr int kRowTiles = 4;
  constexpr int kOutputElements = kRowTiles * 256;
  __shared__ float reduction_storage[kPhysicalWarps + 1][kOutputElements];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half scale =
      __ldg(channel_scales + tile * 32 + qpn8_col_from_lane(lane));
  const half2 scale2 = __halves2half2(scale, scale);

#pragma unroll
  for (int phase = 0; phase < 2; ++phase) {
    float accum[kRowTiles][2][8];
#pragma unroll
    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
      for (int chain = 0; chain < 2; ++chain) {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          accum[row_tile][chain][index] = 0.0f;
        }
      }
    }

    const int logical_warp = warp + phase * kPhysicalWarps;
    const int group_begin = logical_warp * groups_per_warp;
#pragma unroll 4
    for (int group = group_begin; group < group_begin + groups_per_warp;
         ++group) {
      const uint4 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
      half2 weights[8];
      fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        weights[index] = __hmul2(weights[index], scale2);
      }

      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
      for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
        uint4 input01 = make_uint4(0, 0, 0, 0);
        uint4 input23 = make_uint4(0, 0, 0, 0);
        const int input_row_idx = row_tile * 8 + row;
        if (input_row_idx < m) {
          const half* input_row =
              input + static_cast<size_t>(input_row_idx) * k;
          input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
          input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        FASTLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
        FASTLLM_SM70_MMA_8N8K4(accum[row_tile][1], a0[2], a0[3], b[2], b[3]);
        FASTLLM_SM70_MMA_8N8K4(accum[row_tile][0], a1[0], a1[1], b[4], b[5]);
        FASTLLM_SM70_MMA_8N8K4(accum[row_tile][1], a1[2], a1[3], b[6], b[7]);
      }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][1][index];
        const int output_row =
            row_tile * 8 + (index & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col =
            (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
        reduction_storage[warp][output_row * 32 + quadpair * 8 + output_col] =
            accum[row_tile][0][index];
      }
    }
    __syncthreads();

    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
      float value =
          phase == 0 ? 0.0f : reduction_storage[kPhysicalWarps][element];
#pragma unroll
      for (int k_warp = 0; k_warp < kPhysicalWarps; ++k_warp) {
        value += reduction_storage[k_warp][element];
      }
      if (phase == 0) {
        reduction_storage[kPhysicalWarps][element] = value;
      } else {
        const int output_row = element >> 5;
        const int output_col = element & 31;
        if (output_row < m) {
          output[static_cast<size_t>(output_row) * n + tile * 32 + output_col] =
              __float2half(value);
        }
      }
    }
    __syncthreads();
  }
}

template <int SplitK>
void launch_fp8_qpn8_m32_twophase_sm70(const uint8_t* codes,
                                       const half* channel_scales,
                                       const half* input, half* output, int n,
                                       int k, int m, cudaStream_t stream) {
  constexpr int kPhysicalWarps = SplitK / 2;
  fp8_qpn8_m32_twophase_sm70_kernel<SplitK>
      <<<(n / 32), (32 * kPhysicalWarps), 0, stream>>>(codes, channel_scales,
                                                       input, output, n, k, m);
}

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr || std::strcmp(value, "0") != 0;
}

// Pick a split-K factor for the given shape, or -1 when no supported factor
// divides K/16. Mirrors 1Cat's accepted split set {4, 8, 12, 16, 32} with the
// M-dependent restrictions.
int ChooseSplitK(int m, int k) {
  const int groups = k / 16;
  if (groups <= 0) {
    return -1;
  }
  if (m > 16) {
    if (groups % 16 == 0) {
      return 16;
    }
    if (groups % 12 == 0) {
      return 12;
    }
    return -1;
  }
  if (groups % 16 == 0) {
    return 16;
  }
  if (groups % 8 == 0) {
    return 8;
  }
  if (groups % 4 == 0) {
    return 4;
  }
  return -1;
}

}  // namespace

bool Fp8QpnSupported() {
  if (!EnvEnabled("FASTLLM_SM70") || !EnvEnabled("FASTLLM_SM70_QPN") ||
      !EnvEnabled("FASTLLM_SM70_FP8_QPN8")) {
    return false;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    return false;
  }
  static thread_local bool supported = false;
  static thread_local bool initialized = false;
  if (!initialized) {
    int major = 0, minor = 0;
    supported =
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) ==
            cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) ==
            cudaSuccess &&
        major == 7 && minor == 0;
    initialized = true;
  }
  return supported;
}

bool Fp8QpnCanRun(int m, int k, int n) {
  if (m < 1 || m > 32 || n <= 0 || n % 32 != 0 || k <= 0 || k % 16 != 0) {
    return false;
  }
  return ChooseSplitK(m, k) != -1;
}

bool Fp8QpnPrepare(const uint8_t* qweight, const float* scales,
                   uint8_t* codes, half* groupScales,
                   int k, int n, bool channelScales, cudaStream_t stream) {
  if (qweight == nullptr || scales == nullptr || codes == nullptr ||
      groupScales == nullptr || k <= 0 || n <= 0 || k % 16 != 0 ||
      n % 32 != 0) {
    return false;
  }
  if (!channelScales && (k % 128 != 0 || n % 128 != 0)) {
    return false;
  }

  const size_t weight_numel = static_cast<size_t>(n) * k;
  const int weight_blocks = static_cast<int>(
      (weight_numel + kQpn8PrepareThreads - 1) / kQpn8PrepareThreads);
  fp8_qpn8_prepack_sm70_kernel<<<weight_blocks, kQpn8PrepareThreads, 0,
                                 stream>>>(
      codes, qweight, n, k);

  const size_t scale_numel = channelScales
                                 ? static_cast<size_t>(n)
                                 : static_cast<size_t>(k / 128) * (n / 32);
  const int scale_blocks = static_cast<int>(
      (scale_numel + kQpn8PrepareThreads - 1) / kQpn8PrepareThreads);
  if (channelScales) {
    fp8_qpn8_channel_scale_sm70_kernel<<<scale_blocks, kQpn8PrepareThreads, 0,
                                         stream>>>(
        groupScales, scales, n);
  } else {
    fp8_qpn8_scale_sm70_kernel<<<scale_blocks, kQpn8PrepareThreads, 0,
                                 stream>>>(
        groupScales, scales, n / 128, k / 128);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool Fp8QpnGemm(const uint8_t* codes, const half* groupScales,
                const half* in, half* out,
                int m, int k, int n, bool channelScales, cudaStream_t stream) {
  if (codes == nullptr || groupScales == nullptr || in == nullptr ||
      out == nullptr) {
    return false;
  }
  if (!Fp8QpnSupported() || !Fp8QpnCanRun(m, k, n)) {
    return false;
  }
  // M=9..32 use the two-row / two-phase kernels, which read channel scales.
  if (m > 8 && !channelScales) {
    return false;
  }
  const int split = ChooseSplitK(m, k);
  if (split < 0) {
    return false;
  }

  if (m > 16) {
    if (split == 12) {
      launch_fp8_qpn8_m32_twophase_sm70<12>(
          codes, groupScales, in, out, n, k, m, stream);
    } else {
      launch_fp8_qpn8_m32_twophase_sm70<16>(
          codes, groupScales, in, out, n, k, m, stream);
    }
    return cudaGetLastError() == cudaSuccess;
  }

  if (m <= 8) {
    if (split == 4) {
      launch_fp8_qpn8_sm70<4, 2, true, false, false, 1>(
          codes, groupScales, in, out, n, k, m, channelScales, stream);
    } else if (split == 8) {
      launch_fp8_qpn8_sm70<8, 2, true, false, false, 1>(
          codes, groupScales, in, out, n, k, m, channelScales, stream);
    } else if (split == 16) {
      launch_fp8_qpn8_sm70<16, 2, true, false, false, 1>(
          codes, groupScales, in, out, n, k, m, channelScales, stream);
    } else {
      return false;
    }
  } else {
    // m in 9..16: two 8-row tiles, channel scales.
    if (split == 4) {
      launch_fp8_qpn8_sm70<4, 2, true, false, false, 2>(
          codes, groupScales, in, out, n, k, m, true, stream);
    } else if (split == 8) {
      launch_fp8_qpn8_sm70<8, 2, true, false, false, 2>(
          codes, groupScales, in, out, n, k, m, true, stream);
    } else if (split == 16) {
      launch_fp8_qpn8_sm70<16, 2, true, false, false, 2>(
          codes, groupScales, in, out, n, k, m, true, stream);
    } else {
      return false;
    }
  }
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace sm70
}  // namespace fastllm
