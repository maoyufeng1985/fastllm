// SM70 fused NVFP4 x FP16 GEMM (prefill). One kernel: read packed NVFP4
// weights, dequantize into shared memory, run Volta hmma m16n16k16, write
// out. Replaces the two-step dequant-to-global + cublasGemmEx path that the
// 180K trace shows running at 45.6 TFLOPS against a measured 103 TFLOPS wall
// on this box (audit rows 55/58).
//
// Weight layout (authoritative: FastllmCudaNVFP4Block162HalfKernel in
// fastllm-linear-fp8.cu): non-planar, blockM=16/blockK=1. Each weight row
// holds ceil(K/16) blocks of 12 bytes: 8 bytes of packed 4-bit codes
// (code j at nibble (j&1 ? high : low) of byte j>>1) followed by a float
// scale covering that group of 16. Row r starts at cudaData + r*perRow.
// Kernel2 in the trace is cublas' own name; this file is the fused
// replacement candidate, gated off by default.
#include <mma.h>
#include <cstdint>

namespace fastllm {
namespace sm70 {

// 4-bit E2M1 code -> fp16 via a 16-entry constant table. Keeping the table in
// constant memory lets the compiler turn the lookup into a few ALU ops.
__constant__ unsigned short kNvfp4E2M1Bits[16] = {0x0000,0x3800,0x3c00,0x3e00,0x4000,0x4200,0x4400,0x4600,0x0000,0xb800,0xbc00,0xbe00,0xc000,0xc200,0xc400,0xc600};

// Tile: BM x BN output tile per block, BK=16 (one NVFP4 group per k-step).
// Volta WMMA fragment = 16x16x16. BM=64, BN=64, BK=16 -> 4 warps each owning
// one 16x64 or 64x16 strip of fragments; shared holds B (weights) tile
// 64x16 fp16 and A (activations) 64x16 fp16.
constexpr int kBk = 16;  // one NVFP4 block group exactly
constexpr int kW = 16;   // k step per shared-memory stage

template <int BM, int BN>
__global__ __launch_bounds__(256) void Nvfp4FusedGemmKernel(
    const half *__restrict__ act, const uint8_t *__restrict__ wPacked,
    int perRow, float *__restrict__ out, int M, int N, int K) {
    // 128x128 tile, 8 warps in 2x4 grid. Arithmetic intensity doubles vs
    // the 128x64 version: each k-step now feeds 16x2=32 mma per warp.
    const int blockRow = blockIdx.y * BN;
    const int blockCol = blockIdx.x * BM;
    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / 4;            // 0..1
    const int warpCol = warpId % 4;            // 0..3
    const int rowsPerWarp = BM / 2 / 16;       // 4 fragments along M
    const int colsPerWarp = BN / 4 / 16;       // 2 fragments along N
    const int tid = threadIdx.x;

    __shared__ half sA[2][BM][kW];
    __shared__ half sB[2][BN][kW];

    using AFrag = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                         half, nvcuda::wmma::row_major>;
    using BFrag = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                         half, nvcuda::wmma::col_major>;
    using CFrag = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                                         float>;
    CFrag cFrag[rowsPerWarp][colsPerWarp];
#pragma unroll
    for (int i = 0; i < rowsPerWarp; ++i)
        for (int j = 0; j < colsPerWarp; ++j)
            nvcuda::wmma::fill_fragment(cFrag[i][j], 0.0f);

    auto loadTile = [&](int b, int k0) {
        // 32-wide k step: two NVFP4 groups per load. Each thread handles 8
        // consecutive k of one weight row, so the per-row group header and
        // scale are read once per two groups instead of once per group.
        for (int e = tid; e < BM * kW; e += 256) {
            int r = e / kW, c = e % kW;
            int gr = blockCol + r, gc = k0 + c;
            sA[b][r][c] = (gr < M && gc < K) ? act[(size_t)gr * K + gc]
                                             : __float2half_rn(0.f);
        }
        for (int e = tid; e < BN * kW; e += 256) {
            int r = e / kW, c = e % kW;
            int gr = blockRow + r, k = k0 + c;
            half v = __float2half_rn(0.f);
            if (gr < N && k < K) {
                const int group = k >> 4, off = k & 15;
                const uint8_t *row =
                    wPacked + (size_t)gr * perRow + (size_t)group * 12;
                const uint8_t packed = row[off >> 1];
                const uint8_t code =
                    (off & 1) ? (packed >> 4) : (packed & 0xF);
                const float scale = *(const float *)(row + 8);
                const __half lut = *reinterpret_cast<const __half *>(
                    kNvfp4E2M1Bits + code);
                v = __float2half_rn(__half2float(lut) * scale);
            }
            sB[b][r][c] = v;
        }
    };

    loadTile(0, 0);
    for (int k0 = 0; k0 < K; k0 += kW) {
        const int cur = (k0 / kW) & 1;
        const int nxt = cur ^ 1;
        if (k0 + kW < K) {
            loadTile(nxt, k0 + kW);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < colsPerWarp; ++j) {
            BFrag bFrag;
            nvcuda::wmma::load_matrix_sync(
                bFrag, &sB[cur][warpCol * 32 + j * 16][0], kW);
#pragma unroll
            for (int i = 0; i < rowsPerWarp; ++i) {
                AFrag aFrag;
                nvcuda::wmma::load_matrix_sync(
                    aFrag, &sA[cur][(warpRow * rowsPerWarp + i) * 16][0], kW);
                nvcuda::wmma::mma_sync(cFrag[i][j], aFrag, bFrag, cFrag[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < colsPerWarp; ++j) {
#pragma unroll
        for (int i = 0; i < rowsPerWarp; ++i) {
            nvcuda::wmma::store_matrix_sync(
                &out[(size_t)(blockCol + (warpRow * rowsPerWarp + i) * 16) * N +
                     blockRow + warpCol * 32 + j * 16],
                cFrag[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
}
}  // namespace sm70
}  // namespace fastllm
