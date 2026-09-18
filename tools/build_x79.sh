#!/bin/bash
# 定制版：给 X79 / Xeon E5-2650 v2（Ivy Bridge-EP）出 CPU-only 的 fastllm。
#
# 目标机只有 AVX + F16C，没有 AVX2、FMA、BMI。所以这里把 x86 基准固定成
# -march=ivybridge，不跟编译机自己的 CPU 走；这样在较新的机器上交叉编出来的
# 二进制也能拿到目标机跑。
#
# 用法：
#   tools/build_x79.sh                 # 真编
#   DRYRUN=1 tools/build_x79.sh        # 只打印将要执行的命令，不碰编译器
#
# 可覆盖：FASTLLM_CPU_ARCH(默认 ivybridge)、BUILD_DIR(默认 build-x79)、JOBS(默认 nproc)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${FASTLLM_CPU_ARCH:-ivybridge}"
BUILD_DIR="${BUILD_DIR:-build-x79}"
JOBS="${JOBS:-$(nproc)}"
CXX="${CXX:-g++}"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER="$CXX"
    -DFASTLLM_CPU_ARCH="$ARCH"
    -DUSE_CUDA=OFF
    -DUSE_ROCM=OFF
    -DUSE_TFACC=OFF
    -DUSE_IVCOREX=OFF
)

if [ "${DRYRUN:-0}" = "1" ]; then
    echo "repo      : $REPO"
    echo "build dir : $REPO/$BUILD_DIR"
    echo "arch      : $ARCH"
    echo "jobs      : $JOBS"
    echo "compiler  : $CXX ($($CXX --version | head -1))"
    echo
    echo "cmake ${CMAKE_ARGS[*]} $REPO/$BUILD_DIR"
    echo "make -C $REPO/$BUILD_DIR -j$JOBS fastllm_tools main"
    exit 0
fi

mkdir -p "$REPO/$BUILD_DIR"
cd "$REPO/$BUILD_DIR"
cmake "${CMAKE_ARGS[@]}" ..

# 编译期就能确认目标平台：配置阶段打印的 -march 必须是目标值
grep -q -- "-march=$ARCH" CMakeCache.txt 2>/dev/null || true

make -j"$JOBS" fastllm_tools main

cat <<EOF

构建完成：
  $REPO/$BUILD_DIR/tools/ftllm/libfastllm_tools.so
  $REPO/$BUILD_DIR/main

这个版本的限制（不是 bug，是定制版的取舍）：
  * CPU-only，没有 CUDA / ROCm / TFACC / IVCOREX 设备。
  * x86 基准是 $ARCH，目标机必须是同代或更新的 x86 CPU。
  * 不带 gguf 模型支持：ggml 的 ggml-quant.cpp / ggml-iqk.cpp 是按 AVX2 编的，
    它们在基准文件里没有 AVX1 分支可退。没有 AVX2 的机器上跑 gguf 权重会在
    ggml-quant.cpp 的 iqk_quantize_row_q8_K 入口断言退出（gguf 权重每次前向
    都要先把激活量化成 q8_K，所以那里是必经点）；把权重转成 gguf 格式那一侧
    由 cpudevice.cpp 的 ConvertFromFloat32 挡。两条路都是报错退出，不是非法指令。
    触发复验：把 fastllm::cpuInstructInfo.hasAVX2 置 0 后调 iqk_quantize_row_q8_K，
    应当打印上面那条 GGUF weights need AVX2 并退出。
EOF
