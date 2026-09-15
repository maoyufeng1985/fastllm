//
// SM70 (V100) QPN2 NVFP4 dense GEMM, torch-free.
//
// The QPN2 execution layout is derived from dnv2003/v100-skinny (MIT), ported
// from 1Cat-vLLM's csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu. The device
// kernels are unchanged; only the host wrappers are rewritten to use raw CUDA
// pointers. See LICENSE.v100-skinny in this directory for the retained MIT
// notice.
//
// This is the PR1 NVFP4 kernel foundation for the SM70 concurrency port. It is
// self-contained and unit-testable on a single V100. On non-SM70 devices every
// entry point reports unsupported and callers keep their existing path.
//

#include "devices/cuda/fastllm-sm70.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastllm {
namespace sm70 {

namespace {

constexpr int kPrepareThreads = 256;
constexpr int kQpn2RowsPerCta = 8;

__device__ __forceinline__ int qpn2_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn2_logical_k(int physical_k) {
  const int local = physical_k & 7;
  return (physical_k & 8) + ((local & 3) << 1) + (local >> 2);
}

__global__ void nvfp4_qpn2_prepack_codes_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ weight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 2;
  if (index >= numel) {
    return;
  }

  const int physical_byte = static_cast<int>(index & 7);
  size_t outer = index >> 3;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  const int logical_k0 = group * 16 + qpn2_logical_k(physical_byte * 2);
  const int logical_k1 = group * 16 + qpn2_logical_k(physical_byte * 2 + 1);
  const int k_bytes = k >> 1;
  const uint8_t packed0 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k0 >> 1)];
  const uint8_t packed1 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k1 >> 1)];
  const uint8_t code0 =
      static_cast<uint8_t>((packed0 >> ((logical_k0 & 1) * 4)) & 0x0f);
  const uint8_t code1 =
      static_cast<uint8_t>((packed1 >> ((logical_k1 & 1) * 4)) & 0x0f);
  output[index] = static_cast<uint8_t>(code0 | (code1 << 4));
}

// Inverse of fp8e4m3_to_half2: round a positive magnitude through the same
// half*256 construction the GEMM decoder uses, then restore the sign bit.
__device__ __forceinline__ uint8_t float_to_e4m3(float value) {
  const unsigned sign = (__float_as_uint(value) >> 24) & 0x80u;
  const float absValue = fabsf(value);
  if (!(absValue > 0.0f)) {
    return static_cast<uint8_t>(sign);
  }
  if (absValue >= 448.0f) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }
  const unsigned short halfBits =
      __half_as_ushort(__float2half_rn(absValue * 0.00390625f));
  return static_cast<uint8_t>(sign | ((halfBits >> 7) & 0x7fu));
}

// Native interleaved [source_n, (K/16)*12] → QPN2 fragment codes over
// Nvfp4QpnPackedRows(n) columns. Same index mapping as
// nvfp4_qpn2_prepack_codes_kernel, but the source is eight packed E2M1 bytes
// plus a trailing FP32 scale per group of 16. Columns at or past source_n
// have no source row, so they are written as zero.
__global__ void nvfp4_qpn2_prepack_codes_from_native_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ source, int n,
    int source_n, int k, int source_row_bytes) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 2;
  if (index >= numel) {
    return;
  }

  const int physical_byte = static_cast<int>(index & 7);
  size_t outer = index >> 3;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  if (column >= source_n) {
    output[index] = 0;
    return;
  }
  const int local_k0 = qpn2_logical_k(physical_byte * 2);
  const int local_k1 = qpn2_logical_k(physical_byte * 2 + 1);
  const uint8_t* block =
      source + static_cast<size_t>(column) * source_row_bytes +
      static_cast<size_t>(group) * 12;
  const uint8_t packed0 = block[local_k0 >> 1];
  const uint8_t packed1 = block[local_k1 >> 1];
  const uint8_t code0 =
      static_cast<uint8_t>((packed0 >> ((local_k0 & 1) * 4)) & 0x0f);
  const uint8_t code1 =
      static_cast<uint8_t>((packed1 >> ((local_k1 & 1) * 4)) & 0x0f);
  output[index] = static_cast<uint8_t>(code0 | (code1 << 4));
}

__global__ void nvfp4_qpn2_prepack_scales_from_native_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ source, int n,
    int source_n, int k, int source_row_bytes, float global_scale) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 16;
  if (index >= numel) {
    return;
  }

  const int lane = static_cast<int>(index & 31);
  size_t outer = index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  if (column >= source_n) {
    output[index] = 0;
    return;
  }
  const uint8_t* block =
      source + static_cast<size_t>(column) * source_row_bytes +
      static_cast<size_t>(group) * 12;
  // Native layout stores E4M3 * globalScale as FP32. Recover the checkpoint
  // E4M3 so GEMM can apply globalScale once, matching 1Cat QPN2.
  const float fused = *reinterpret_cast<const float*>(block + 8);
  const float unfused =
      (global_scale == 0.0f) ? fused : (fused / global_scale);
  output[index] = float_to_e4m3(unfused);
}

__global__ void nvfp4_qpn2_prepack_scales_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ scales, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 16;
  if (index >= numel) {
    return;
  }

  const int lane = static_cast<int>(index & 31);
  size_t outer = index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  output[index] = scales[static_cast<size_t>(column) * groups_k16 + group];
}

__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    output[index] = __hmul2(*reinterpret_cast<half2*>(&values[index]), scale);
  }
}

#define FASTLLM_SM70_QPN2_MMA(C, A0, A1, B0, B1)                    \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int SplitK, int NAcc, int RowTiles = 1>
__global__ void nvfp4_qpn2_sm70_kernel(const uint8_t* __restrict__ codes,
                                       const uint8_t* __restrict__ group_scales,
                                       const half* __restrict__ input,
                                       half* __restrict__ output, int n, int k,
                                       int m, float global_scale) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "NVFP4 QPN2 supports one or two 8-row tiles");
  __shared__ float partials[SplitK][RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.y * kQpn2RowsPerCta * RowTiles;
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const uint2* code_ptr = reinterpret_cast<const uint2*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const uint8_t* scale_ptr =
      group_scales + static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half2 global_scale2 = __float2half2_rn(global_scale * 16384.0f);

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][chain][index] = 0.0f;
      }
    }
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint2 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    const half2 scale = __hmul2(
        fp8e4m3_to_half2(__ldg(scale_ptr + static_cast<size_t>(group) * 32)),
        global_scale2);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int row = row_base + row_tile * kQpn2RowsPerCta + local_row;
      if (row < m) {
        const half* input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      FASTLLM_SM70_QPN2_MMA(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      FASTLLM_SM70_QPN2_MMA(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      FASTLLM_SM70_QPN2_MMA(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      FASTLLM_SM70_QPN2_MMA(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][chain][index];
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int output_row = row_tile * kQpn2RowsPerCta + (index & 2) +
                             ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[row_tile][0][index];
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 256;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    const int output_row = row_base + (element >> 5);
    const int output_col = element & 31;
    // n is the logical output dim, so the last tile may be partial. Those
    // columns read the sidecar's zero padding and are not stored.
    if (output_row < m && tile * 32 + output_col < n) {
      output[static_cast<size_t>(output_row) * n + tile * 32 + output_col] =
          __float2half(value);
    }
  }
}

template <int SplitK, int NAcc, int RowTiles = 1>
void launch_qpn2(const uint8_t* codes, const uint8_t* scales, const half* input,
                 half* output, int n, int k, int m, float global_scale,
                 cudaStream_t stream) {
  constexpr int kRowsPerCta = kQpn2RowsPerCta * RowTiles;
  const dim3 grid((n + 31) / 32, (m + kRowsPerCta - 1) / kRowsPerCta);
  nvfp4_qpn2_sm70_kernel<SplitK, NAcc, RowTiles>
      <<<grid, (32 * SplitK), 0, stream>>>(codes, scales, input, output, n, k,
                                           m, global_scale);
}

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr || std::strcmp(value, "0") != 0;
}

// Prefer split-8 (1Cat's common decode configs), then 16, then 32.
int ChooseSplitK(int k) {
  const int groups = k / 16;
  if (groups <= 0) {
    return -1;
  }
  if (groups % 8 == 0) {
    return 8;
  }
  if (groups % 16 == 0) {
    return 16;
  }
  if (groups % 32 == 0) {
    return 32;
  }
  return -1;
}

}  // namespace

bool Nvfp4QpnSupported() {
  if (!EnvEnabled("FASTLLM_SM70") || !EnvEnabled("FASTLLM_SM70_QPN") ||
      !EnvEnabled("FASTLLM_SM70_NVFP4_QPN2")) {
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

bool Nvfp4QpnCanRun(int m, int k, int n) {
  if (m < 1 || m > 32 || n <= 0 || k <= 0 || k % 64 != 0) {
    return false;
  }
  return ChooseSplitK(k) != -1;
}

bool Nvfp4QpnPrepare(const uint8_t* weight, const uint8_t* scales,
                     uint8_t* codes, uint8_t* packedScales,
                     int k, int n, cudaStream_t stream) {
  if (weight == nullptr || scales == nullptr || codes == nullptr ||
      packedScales == nullptr || k <= 0 || n <= 0 || k % 64 != 0 ||
      n % 32 != 0) {
    return false;
  }

  const size_t code_numel = static_cast<size_t>(n) * k / 2;
  const int code_blocks = static_cast<int>(
      (code_numel + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn2_prepack_codes_kernel<<<code_blocks, kPrepareThreads, 0, stream>>>(
      codes, weight, n, k);

  const size_t scale_numel = static_cast<size_t>(n) * k / 16;
  const int scale_blocks = static_cast<int>(
      (scale_numel + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn2_prepack_scales_kernel<<<scale_blocks, kPrepareThreads, 0,
                                     stream>>>(
      packedScales, scales, n, k);
  return cudaGetLastError() == cudaSuccess;
}

bool Nvfp4QpnPrepareFromNative(uint8_t* storage, size_t storageBytes,
                               int k, int n, float globalScale,
                               cudaStream_t stream, uint8_t* dest,
                               size_t destBytes) {
  if (storage == nullptr || k <= 0 || n <= 0 || k % 64 != 0) {
    return false;
  }
  const size_t sourceRowBytes = static_cast<size_t>(k / 16) * 12;
  const size_t sourceBytes = static_cast<size_t>(n) * sourceRowBytes;
  // Columns at or past n have no source row, so the sidecar is sized for the
  // padded row count while the source bound stays at n.
  const int packedN = Nvfp4QpnPackedRows(n);
  const size_t codeBytes = static_cast<size_t>(packedN) * k / 2;
  const size_t scaleBytes = static_cast<size_t>(packedN) * k / 16;
  const size_t packedBytes = codeBytes + scaleBytes;
  uint8_t* out = dest != nullptr ? dest : storage;
  const size_t outBytes = dest != nullptr ? destBytes : storageBytes;
  if (storageBytes < sourceBytes || outBytes < packedBytes) {
    return false;
  }

  int storageDevice = -1;
  int originalDevice = -1;
  cudaPointerAttributes attributes;
  if (cudaPointerGetAttributes(&attributes, storage) == cudaSuccess) {
#if (CUDART_VERSION < 10000) && !(defined(USE_ROCM))
    if (attributes.memoryType == cudaMemoryTypeDevice) {
#else
    if (attributes.type == cudaMemoryTypeDevice ||
        attributes.type == cudaMemoryTypeManaged) {
#endif
      storageDevice = attributes.device;
    }
  } else {
    cudaGetLastError();
  }
  if (cudaGetDevice(&originalDevice) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  cudaStream_t workStream = stream;
  if (storageDevice >= 0 && storageDevice != originalDevice) {
    if (cudaSetDevice(storageDevice) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    workStream = cudaStreamPerThread;
  }

  uint8_t* scratch = nullptr;
  if (cudaMalloc(&scratch, packedBytes) != cudaSuccess || scratch == nullptr) {
    if (scratch != nullptr) {
      cudaFree(scratch);
    }
    cudaGetLastError();
    if (storageDevice >= 0 && storageDevice != originalDevice) {
      cudaSetDevice(originalDevice);
    }
    return false;
  }

  uint8_t* codes = scratch;
  uint8_t* packedScales = scratch + codeBytes;
  const int codeBlocks = static_cast<int>(
      (codeBytes + kPrepareThreads - 1) / kPrepareThreads);
  const int scaleBlocks = static_cast<int>(
      (scaleBytes + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn2_prepack_codes_from_native_kernel<<<codeBlocks, kPrepareThreads, 0,
                                                workStream>>>(
      codes, storage, packedN, n, k, static_cast<int>(sourceRowBytes));
  nvfp4_qpn2_prepack_scales_from_native_kernel<<<scaleBlocks, kPrepareThreads,
                                                 0, workStream>>>(
      packedScales, storage, packedN, n, k, static_cast<int>(sourceRowBytes),
      globalScale);

  cudaError_t operationState = cudaPeekAtLastError();
  bool converted = operationState == cudaSuccess;
  bool overwriteStarted = false;
  if (converted) {
    operationState =
        cudaMemcpyAsync(out, scratch, packedBytes, cudaMemcpyDeviceToDevice,
                        workStream);
    overwriteStarted = operationState == cudaSuccess;
    converted = overwriteStarted;
  }

  const cudaError_t syncState = cudaStreamSynchronize(workStream);
  cudaFree(scratch);
  if (storageDevice >= 0 && storageDevice != originalDevice) {
    cudaSetDevice(originalDevice);
  }
  if (!converted || syncState != cudaSuccess) {
    if (overwriteStarted && out == storage) {
      std::printf(
          "Fastllm NVFP4 QPN2 conversion failed after in-place copy began.\n");
      throw("nvfp4 qpn2 in-place conversion error");
    }
    cudaGetLastError();
    return false;
  }
  return true;
}

bool Nvfp4QpnGemm(const uint8_t* codes, const uint8_t* packedScales,
                  const half* in, half* out,
                  int m, int k, int n, float globalScale, cudaStream_t stream) {
  if (codes == nullptr || packedScales == nullptr || in == nullptr ||
      out == nullptr) {
    return false;
  }
  if (!Nvfp4QpnSupported() || !Nvfp4QpnCanRun(m, k, n)) {
    return false;
  }
  const int split = ChooseSplitK(k);
  if (split < 0) {
    return false;
  }

  const bool twoTile = m > kQpn2RowsPerCta && m <= 16 && split != 32;
  if (twoTile) {
    if (split == 8) {
      launch_qpn2<8, 2, 2>(codes, packedScales, in, out, n, k, m, globalScale,
                           stream);
    } else if (split == 16) {
      launch_qpn2<16, 2, 2>(codes, packedScales, in, out, n, k, m, globalScale,
                            stream);
    } else {
      return false;
    }
  } else if (split == 8) {
    launch_qpn2<8, 2, 1>(codes, packedScales, in, out, n, k, m, globalScale,
                         stream);
  } else if (split == 16) {
    launch_qpn2<16, 2, 1>(codes, packedScales, in, out, n, k, m, globalScale,
                          stream);
  } else if (split == 32) {
    launch_qpn2<32, 2, 1>(codes, packedScales, in, out, n, k, m, globalScale,
                          stream);
  } else {
    return false;
  }
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace sm70
}  // namespace fastllm
