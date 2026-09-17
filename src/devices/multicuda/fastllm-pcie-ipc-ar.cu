#include "devices/multicuda/pcie_ipc_ar.h"

#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "fastllm.h"
#include "comm/pcie_ipc_all_reduce.cuh"

namespace fi = flashinfer::comm::pcie_ipc;

namespace fastllm {
namespace pcieipc {
namespace {

constexpr size_t kDefaultMaxBytes = 160ULL * 1024ULL;
constexpr int kMaxBlocks = 8;
constexpr int kBlocks = 8;
constexpr int kThreads = 256;
constexpr int kMaxWorld = 8;

bool EnabledByEnv() {
    static const bool on = []() {
        const char *v = std::getenv("FASTLLM_PCIE_IPC_AR");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

bool PreferBeforeCustomEnvy() {
    static const bool on = []() {
        const char *v = std::getenv("FASTLLM_PCIE_IPC_AR_BEFORE_CUSTOM");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

size_t MaxBytesByEnv() {
    static const size_t n = []() {
        const char *v = std::getenv("FASTLLM_PCIE_IPC_AR_MAX_BYTES");
        if (v == nullptr || v[0] == '\0') {
            return kDefaultMaxBytes;
        }
        long long x = std::atoll(v);
        return x > 0 ? (size_t)x : kDefaultMaxBytes;
    }();
    return n;
}

// Everything the ranks share. Init() only records parameters: it must not make a
// single CUDA call, because it runs while holding the engine's NCCL init mutex and
// every other rank thread is parked on that same mutex. Touching device r's context
// from here would wait on a thread that cannot make progress.
struct Shared {
    std::mutex mutex;
    bool enabled = false;
    int world = 0;
    int64_t maxNumel = 0;
    int64_t slabBytes = 0;
    int devices[kMaxWorld] = {0};
    std::atomic<uintptr_t> slab[kMaxWorld];     // the IPC-style workspace
    std::atomic<uintptr_t> scratch[kMaxWorld];  // in-place staging, separate buffer
    std::atomic<int> published{0};
    std::atomic<bool> failed{false};
    // Absolute deadline for the "every rank has published" wait. Derived from the
    // shared Init timestamp, not from each rank's own entry time: if each rank used
    // its own start, one rank could time out and fall back to NCCL while a peer
    // reached the barrier just under its own (later) deadline and launched the PCIe
    // kernel. The two ranks would then be in different collectives and hang. A
    // shared deadline makes the outcome identical on every rank.
    std::chrono::steady_clock::time_point publishDeadline{};
    // Fixed-size, never reallocated. A std::vector here was a real bug: Init/Shutdown
    // assign/clear it under the mutex while TryAllReduce read s.views[rank] outside
    // the mutex, so a re-group could hand a rank pointers into freed storage and push
    // them into a kernel. A plain array removes the reallocation entirely.
    fi::PeerViews views[kMaxWorld];
    std::atomic<bool> viewsBuilt[kMaxWorld];
    // Bumped by Init on every (re)arming. A built view is only valid for the
    // generation it was built in; this is what makes a stale view detectable.
    std::atomic<uint64_t> generation{0};
    // Ranks currently between "committed to launch" and "launch returned". Init
    // drains this before freeing old slabs. Host-side only: it does not wait for
    // the kernel itself to finish (the engine re-groups only at NCCL setup, when
    // no collective is running).
    std::atomic<int> inflight{0};
    std::atomic<bool> deadlineArmed{false};
    // Generation each rank's view was built for; -1 means "not built".
    std::atomic<uint64_t> viewGeneration[kMaxWorld];
};

Shared &S() {
    static Shared s;
    return s;
}

std::mutex g_rankMutex[kMaxWorld];
std::atomic<long> g_uses{0};

bool TraceOn() {
    static const bool on = []() {
        const char *v = std::getenv("FASTLLM_PCIE_IPC_AR_TRACE");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

// One line per event, rendered into a local buffer and written with a single
// fwrite. Four rank threads share stderr, and the earlier three-fprintf version
// interleaved: a line for dev=3 and one for dev=0 could come out glued together,
// which made a per-rank call count look asymmetric when it was not.
void Trace(const char *fmt, ...) {
    if (!TraceOn()) {
        return;
    }
    char body[256];
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(body, sizeof body, fmt, ap);
    va_end(ap);
    if (n < 0) {
        return;
    }
    char line[320];
    int total = std::snprintf(line, sizeof line, "[pcie_ipc trace] %s\n", body);
    if (total > 0) {
        static std::mutex traceMutex;
        std::lock_guard<std::mutex> lock(traceMutex);
        std::fwrite(line, 1, (size_t)(total < (int)sizeof line ? total : (int)sizeof line - 1),
                    stderr);
        std::fflush(stderr);
    }
}

void ReportUses() {
    if (!EnabledByEnv()) {
        return;
    }
    std::fprintf(stderr,
                 "[Fastllm] pcie_ipc AR: %ld collective(s) served by the PCIe kernel, "
                 "ceiling %zu bytes/msg.\n",
                 g_uses.load(), MaxBytes());
    std::fflush(stderr);
}

void NoteOnce(const char *why) {
    static std::atomic<int> budget{6};
    if (budget.fetch_sub(1) > 0) {
        std::fprintf(stderr, "[Fastllm] pcie_ipc AR declined: %s\n", why);
        std::fflush(stderr);
    }
}

int RankOf(const Shared &s, int deviceId) {
    for (int i = 0; i < s.world; ++i) {
        if (s.devices[i] == deviceId) {
            return i;
        }
    }
    return -1;
}

// Builds this rank's slab on this rank's own thread, waits for the peers to
// publish theirs, then builds the views. Every CUDA call here happens on the
// thread that owns the device, which is what Init() must not do.
bool EnsureLocal(Shared &s, int rank) {
    std::lock_guard<std::mutex> rankGuard(g_rankMutex[rank]);
    if (s.failed.load(std::memory_order_acquire)) {
        return false;
    }
    // Arm the shared deadline once, on the first rank to reach first use, so every
    // rank compares against the same absolute instant. The deadline value is written
    // BEFORE the flag that publishes it: the earlier version set the flag first, so a
    // second rank could observe "armed" and then read a not-yet-written (default,
    // epoch) timestamp, decide it was already past, and permanently disable the
    // feature. Store-then-flag with a release/acquire pair makes the order explicit.
    if (!s.deadlineArmed.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> deadlineLock(s.mutex);
        if (!s.deadlineArmed.load(std::memory_order_relaxed)) {
            s.publishDeadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(120);
            s.deadlineArmed.store(true, std::memory_order_release);
        }
    }
    const int device = s.devices[rank];
    if (s.slab[rank].load(std::memory_order_acquire) == 0) {
        int previous = 0;
        cudaGetDevice(&previous);
        if (cudaSetDevice(device) != cudaSuccess) {
            cudaGetLastError();
            cudaSetDevice(previous);
            // Sticky + shared: without this the peers sit in the publish-wait loop
            // until the 120 s deadline before they all fall back together, which
            // turns a local failure into a two-minute stall.
            s.failed.store(true, std::memory_order_release);
            NoteOnce("cudaSetDevice failed");
            return false;
        }
        cudaFree(0);  // make sure this device's context exists on this thread
        void *p = nullptr;
        if (cudaMalloc(&p, (size_t)s.slabBytes) != cudaSuccess) {
            cudaGetLastError();
            cudaSetDevice(previous);
            s.failed.store(true, std::memory_order_release);
            NoteOnce("slab cudaMalloc failed");
            return false;
        }
        if (cudaMemset(p, 0, (size_t)s.slabBytes) != cudaSuccess) {
            // The first bytes of the slab are the kernel's signal slots; if the
            // memset fails they hold garbage and the collective can hang.
            cudaGetLastError();
            cudaFree(p);
            cudaSetDevice(previous);
            s.failed.store(true, std::memory_order_release);
            NoteOnce("slab cudaMemset failed");
            return false;
        }
        // Separate staging buffer for in-place calls. It must NOT be the workspace:
        // the workspace's first bytes are the signal slots the kernel uses to
        // synchronise, so writing activations there would corrupt the protocol.
        void *sc = nullptr;
        if (cudaMalloc(&sc, MaxBytes()) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(p);
            cudaSetDevice(previous);
            s.failed.store(true, std::memory_order_release);
            NoteOnce("staging cudaMalloc failed");
            return false;
        }
        s.scratch[rank].store((uintptr_t)sc, std::memory_order_release);
        // P2P so the kernel's peer writes go direct rather than through host
        // staging. The engine normally enables this already; a failure here is not
        // fatal, because an already-enabled pair reports AlreadyEnabled.
        for (int p2 = 0; p2 < s.world; ++p2) {
            if (p2 == rank) {
                continue;
            }
            cudaError_t st = cudaDeviceEnablePeerAccess(s.devices[p2], 0);
            if (st != cudaSuccess) {
                cudaGetLastError();
            }
        }
        cudaSetDevice(previous);
        s.slab[rank].store((uintptr_t)p, std::memory_order_release);
        s.published.fetch_add(1, std::memory_order_acq_rel);
    }
    // Wait until every rank has published, or until the SHARED deadline passes. The
    // bound exists only so a genuinely missing rank cannot park a thread forever; it
    // is 120 s because that is unreachable in practice (peers publish within
    // milliseconds of each other). Because the deadline is shared, all ranks observe
    // the same outcome and therefore all fall back together or all proceed together.
    while (s.published.load(std::memory_order_acquire) < s.world) {
        if (s.failed.load(std::memory_order_acquire)) {
            return false;
        }
        if (std::chrono::steady_clock::now() > s.publishDeadline) {
            s.failed.store(true, std::memory_order_release);
            NoteOnce("timed out waiting for peer slabs");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (s.failed.load(std::memory_order_acquire)) {
        return false;
    }
    if (!s.viewsBuilt[rank].load(std::memory_order_acquire)) {
        int64_t ptrs[kMaxWorld] = {0};
        for (int i = 0; i < s.world; ++i) {
            ptrs[i] = (int64_t)s.slab[i].load(std::memory_order_acquire);
        }
        s.views[rank] = fi::make_peer_views(ptrs, s.world, rank,
                                           fi::compute_workspace_layout(
                                               s.world, (int)s.maxNumel, 2, kMaxBlocks));
        s.viewGeneration[rank].store(s.generation.load(std::memory_order_acquire),
                                     std::memory_order_release);
        s.viewsBuilt[rank].store(true, std::memory_order_release);
    }
    return true;
}

}  // namespace

bool Enabled() {
    return EnabledByEnv();
}

bool PreferBeforeCustom() {
    return PreferBeforeCustomEnvy();
}

size_t MaxBytes() {
    return MaxBytesByEnv();
}

long Uses() {
    return g_uses.load();
}

bool Init(const std::vector<int> &devices) {
    if (!EnabledByEnv()) {
        return false;
    }
    Shared &s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    // FastllmInitNccl runs its init path again on later calls. Resetting the slab
    // pointers or the published counter while a peer rank is already inside
    // EnsureLocal would send the ranks down different paths: some launch the PCIe
    // kernel, some fall back to NCCL, and the collective never matches. So a repeat
    // Init for the same live device group is a no-op.
    if (s.enabled && s.world == (int)devices.size()) {
        bool same = true;
        for (int i = 0; i < s.world; ++i) {
            if (s.devices[i] != devices[i]) {
                same = false;
                break;
            }
        }
        if (same) {
            return true;
        }
    }
    // Invalidate views FIRST, then drain in-flight launches, then free. The old
    // order (free, then bump) left a window where a rank that had passed the
    // generation check was about to hand freed pointers to the kernel.
    s.generation.fetch_add(1, std::memory_order_acq_rel);
    for (int spin = 0; spin < 10000; ++spin) {
        if (s.inflight.load(std::memory_order_acquire) == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Release whatever a previous arming left behind. cudaFree does not need this
    // thread to own the device context, unlike the allocations in EnsureLocal.
    for (int i = 0; i < kMaxWorld; ++i) {
        uintptr_t oldSlab = s.slab[i].exchange(0, std::memory_order_acq_rel);
        uintptr_t oldScratch = s.scratch[i].exchange(0, std::memory_order_acq_rel);
        if (oldScratch != 0) {
            cudaFree((void *)oldScratch);
        }
        if (oldSlab != 0) {
            cudaFree((void *)oldSlab);
        }
    }
    cudaGetLastError();
    if (devices.size() < 2 || devices.size() > kMaxWorld) {
        NoteOnce("world size outside 2..8");
        return false;
    }
    if (devices.size() != 2 && devices.size() != 4 && devices.size() != 8) {
        NoteOnce("world size not 2/4/8");
        return false;
    }
    s.world = (int)devices.size();
    s.maxNumel = (int64_t)(MaxBytes() / 2);
    s.slabBytes = (int64_t)fi::workspace_size(s.world, (int)s.maxNumel, 2, kMaxBlocks);
    for (int i = 0; i < s.world; ++i) {
        s.devices[i] = devices[i];
        s.slab[i].store(0, std::memory_order_release);
        s.scratch[i].store(0, std::memory_order_release);
        s.viewsBuilt[i].store(false, std::memory_order_release);
        s.views[i] = fi::PeerViews{};
    }
    s.published.store(0, std::memory_order_release);
    s.failed.store(false, std::memory_order_release);
    // The publish deadline is armed on FIRST USE, not here. Init runs at NCCL setup,
    // and model load plus warmup can easily exceed any fixed budget; a deadline started
    // here would already be in the past by the first collective, so every rank would
    // bail together and the feature would go silently unused.
    s.deadlineArmed.store(false, std::memory_order_release);
    s.enabled = true;
    static const bool once = []() {
        std::atexit(ReportUses);
        return true;
    }();
    (void)once;
    std::fprintf(stderr,
                 "[Fastllm] pcie_ipc AR armed: %d ranks, max %zu bytes/msg, "
                 "slab %lld B/rank (allocated on first use, per rank thread).\n",
                 s.world, MaxBytes(), (long long)s.slabBytes);
    std::fflush(stderr);
    return true;
}

bool Ready() {
    Shared &s = S();
    if (!s.enabled || !EnabledByEnv() || s.failed.load(std::memory_order_acquire)) {
        return false;
    }
    return s.published.load(std::memory_order_acquire) >= s.world &&
           s.world > 0;
}

void Shutdown() {
    Shared &s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.enabled = false;
    // The slab and staging pointers are deliberately NOT cleared here. Clearing them
    // would lose the only record of those allocations and leak one slab plus one
    // staging buffer per NCCL re-initialisation. They are released by the next Init
    // (which runs on the init thread and only frees, never touches the device
    // context) or, if the feature is not re-armed, left for process exit.
    s.world = 0;
    // views is a fixed array; nothing to release. The generation bump in the next
    // Init invalidates any view a straggler might still be holding.
    s.generation.fetch_add(1, std::memory_order_acq_rel);
}

bool TryAllReduce(void *data, void *dest, int count, int dataType, int deviceId) {
    if (!EnabledByEnv()) {
        return false;
    }
    if (data == nullptr || dest == nullptr || count <= 0) {
        return false;
    }
    if (dataType != (int)fastllm::DataType::FLOAT16) {
        Trace("dev=%d DECLINE dtype=%d count=%d", deviceId, dataType, count);
        NoteOnce("dtype is not FLOAT16");
        return false;
    }
    size_t bytes = (size_t)count * 2;
    if (bytes > MaxBytes()) {
        Trace("dev=%d DECLINE bytes=%zu > %zu", deviceId, bytes, MaxBytes());
        return false;  // above the measured crossover: the normal path, stay quiet
    }
    // Upstream requires numel to be a whole number of 16-byte packs (see the
    // all_reduce contract: "numel and max_numel both divisible by the 16-byte pack
    // width"). fp16 means 8 elements per pack. Without this check a ragged count
    // would make the kernel read and write past the end of the tensor.
    if (count % 8 != 0) {
        Trace("dev=%d DECLINE count=%d not a multiple of 8 fp16 elements", deviceId, count);
        NoteOnce("count is not a whole number of 16-byte packs");
        return false;
    }
    Trace("dev=%d TRY bytes=%zu", deviceId, bytes);
    Shared &s = S();
    int world = 0;
    int rank = -1;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.enabled) {
            return false;
        }
        world = s.world;
        rank = RankOf(s, deviceId);
    }
    if (rank < 0) {
        Trace("dev=%d DECLINE not in group (world=%d)", deviceId, world);
        NoteOnce("device not part of the live group");
        return false;
    }
    Trace("dev=%d rank=%d ENTER", deviceId, rank);
    if ((int64_t)count > s.maxNumel) {
        return false;
    }
    if (!EnsureLocal(s, rank)) {
        Trace("dev=%d rank=%d ENSURE FAILED published=%d/%d", deviceId, rank,
              s.published.load(std::memory_order_acquire), world);
        return false;
    }
    // Reject a view built for an earlier arming: the slabs it points at are gone.
    if (s.viewGeneration[rank].load(std::memory_order_acquire) !=
        s.generation.load(std::memory_order_acquire)) {
        Trace("dev=%d rank=%d DECLINE stale view generation", deviceId, rank);
        NoteOnce("view belongs to an earlier init generation");
        return false;
    }
    Trace("dev=%d rank=%d READY published=%d/%d, launching", deviceId, rank,
          s.published.load(std::memory_order_acquire), world);
    // From here on this rank has COMMITTED to the PCIe kernel for this collective.
    // Every failure below is sticky and shared: without the flag, this rank would
    // fall back to NCCL while the peers launch the PCIe kernel, and the two
    // collectives would never match (hang). Making the flag sticky does not save
    // the current collective -- these failures only happen when the device is
    // already broken -- but it keeps every LATER collective consistent.
    s.inflight.fetch_add(1, std::memory_order_acq_rel);
    struct Decrement {
        std::atomic<int> &c;
        ~Decrement() { c.fetch_sub(1, std::memory_order_acq_rel); }
    } dec{s.inflight};
    // Re-check after taking the in-flight slot: Init bumps the generation BEFORE
    // draining in-flight launches, so a bump that happened between the check above
    // and this line is caught here rather than feeding freed pointers to the kernel.
    if (s.viewGeneration[rank].load(std::memory_order_acquire) !=
        s.generation.load(std::memory_order_acquire)) {
        s.failed.store(true, std::memory_order_release);
        NoteOnce("view went stale between check and launch");
        return false;
    }
    int previous = 0;
    cudaGetDevice(&previous);
    if (cudaSetDevice(deviceId) != cudaSuccess) {
        cudaGetLastError();
        cudaSetDevice(previous);
        s.failed.store(true, std::memory_order_release);
        NoteOnce("cudaSetDevice failed at launch");
        return false;
    }
    const half *input = (const half *)data;
    if (data == dest) {
        // The kernel reads the input while writing the output, so an in-place call
        // stages through a slab-local buffer first. Bounded by MaxBytes (160 KiB).
        void *stage = (void *)s.scratch[rank].load(std::memory_order_acquire);
        cudaError_t st = cudaMemcpyAsync(stage, data, bytes,
                                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        if (st != cudaSuccess) {
            cudaGetLastError();
            cudaSetDevice(previous);
            s.failed.store(true, std::memory_order_release);
            NoteOnce("staging memcpy failed");
            return false;
        }
        input = (const half *)stage;
    }
    cudaError_t res = fi::all_reduce<half>(input, (half *)dest, (int64_t)count,
                                          s.views[rank], rank, world, kMaxBlocks,
                                          s.maxNumel, kBlocks, kThreads,
                                          fi::Variant::kUnstaged, false,
                                          cudaStreamPerThread);
    cudaSetDevice(previous);
    if (res != cudaSuccess) {
        cudaGetLastError();
        s.failed.store(true, std::memory_order_release);
        NoteOnce("kernel launch failed");
        return false;
    }
    Trace("dev=%d rank=%d LAUNCHED ok", deviceId, rank);
    g_uses.fetch_add(1, std::memory_order_relaxed);
    return true;
}

}  // namespace pcieipc
}  // namespace fastllm
