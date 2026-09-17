#ifndef FASTLLM_PCIE_IPC_AR_H
#define FASTLLM_PCIE_IPC_AR_H

// FlashInfer's PCIe one-shot all-reduce, wired for FastLLM's single-process
// multi-GPU model.
//
// FlashInfer's own entry point assumes one process per GPU and shares its
// workspace through CUDA IPC. FastLLM runs every rank as a thread inside one
// process, so the peer slabs are ordinary cudaMalloc allocations that all rank
// threads can already address. That removes the IPC bootstrap entirely.
//
// Measured on this box (4x V100-SXM2, PCIe gen3 x16, all four behind one PIX
// switch), FlashInfer vs NCCL, FP16, eager, in one process:
//
//   4 KiB   0.52x      160 KiB  0.78x      1 MiB   1.63x
//   40 KiB  0.63x      512 KiB  1.32x      8 MiB   2.35x
//
// (ratio = FlashInfer / NCCL, below 1.0 means FlashInfer is faster, bit-identical
// results on both paths). So the crossover sits between 160 KiB and 512 KiB and
// the kernel is only offered below that. Off by default.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fastllm {
namespace pcieipc {

// Builds the per-rank workspaces. Called once from FastllmInitNccl, where the
// device list is known and only one thread is running. Returns false and leaves
// the feature off when the environment does not ask for it or anything fails.
bool Init(const std::vector<int> &devices);

// True only when Init succeeded for the currently live device group.
bool Ready();

// Largest message, in bytes, for which this route is offered. Beyond it the
// caller must stay on NCCL.
size_t MaxBytes();

// True when this call was actually served by the PCIe kernel. False means the
// caller must fall back, and this function must then have had no side effects.
bool TryAllReduce(void *data, void *dest, int count, int dataType, int deviceId);

// Drops the workspaces. Safe to call when nothing was built.
void Shutdown();

// Number of calls served, for the run report.
long Uses();

// When set, the PCIe kernel is offered BEFORE the engine's built-in custom
// all-reduce instead of after it. Needed to measure the two head to head on the
// small (decode-sized) messages the built-in one normally serves: by default the
// built-in path is tried first and returns true, so the PCIe kernel never sees
// them. Off by default; opt in per measurement.
bool PreferBeforeCustom();

}  // namespace pcieipc
}  // namespace fastllm


#endif  // FASTLLM_PCIE_IPC_AR_H
