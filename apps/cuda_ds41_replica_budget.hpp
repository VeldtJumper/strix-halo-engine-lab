#pragma once

// --vram-total-gib / --ram-total-gib / --budget-receipt: what the replica measures (the planner is tierstream's
// ss_budget_plan, glue in strix/ds41_rtx_budget.hpp; the measured peaks go to an ss_budget_ledger).
//   device: cudaMemGetInfo (total - free, the whole device) at stages; NVML (nvml.dll / libnvidia-ml.so.1, loaded at
//           run time, optional) for the device's use BEFORE this process made its context (other processes: never
//           charged) and for a background sampler that catches peaks between the stages. charged = used - before.
//   host:   the process's resident memory (Windows working set, peak = PeakWorkingSetSize; Linux VmRSS / VmHWM). Not
//           Windows PrivateUsage: under WDDM it also counts the system commit backing device allocations (measured
//           at 24+32: private 38.5 GiB vs working set 16.8 GiB = pinned tier 15.3 GiB + the process; the excess
//           was trunk + drafter + arena on the PRO), which charged VRAM to the RAM cap a second time. Pinned tiers are
//           locked, so they always count.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "psapi.lib")
#endif
#else
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>
#endif

namespace replica_budget {

constexpr std::uint64_t kNone = UINT64_MAX;

struct HostMem {
    std::uint64_t commit = kNone;  // now
    std::uint64_t peak = kNone;    // the process's peak so far (the OS keeps it)
};

inline HostMem host_mem() {
    HostMem h;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pc{};
    pc.cb = sizeof pc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pc), sizeof pc)) {
        h.commit = pc.WorkingSetSize;
        h.peak = pc.PeakWorkingSetSize;
    }
#else
    std::ifstream in("/proc/self/status");
    std::string k;
    while (in >> k) {
        std::uint64_t v = 0;
        if (k == "VmRSS:" && in >> v) h.commit = v * 1024U;
        else if (k == "VmHWM:" && in >> v) h.peak = v * 1024U;
    }
#endif
    return h;
}

// NVML by run-time loading (no link dependency; absent = stage samples only)
class Nvml {
public:
    Nvml() = default;
    Nvml(const Nvml&) = delete;
    Nvml& operator=(const Nvml&) = delete;
    ~Nvml() {
        if (shutdown_ && dev_) shutdown_();
#ifdef _WIN32
        if (lib_) FreeLibrary(static_cast<HMODULE>(lib_));
#else
        if (lib_) dlclose(lib_);
#endif
    }
    // pci: "dddd:bb:dd.f" (cudaDeviceGetPCIBusId)
    bool open(const char* pci) {
#ifdef _WIN32
        lib_ = LoadLibraryA("nvml.dll");
        if (!lib_) lib_ = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        const auto sym = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib_), n)); };
#else
        lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        const auto sym = [&](const char* n) { return dlsym(lib_, n); };
#endif
        if (!lib_) return false;
        auto init = reinterpret_cast<int (*)()>(sym("nvmlInit_v2"));
        shutdown_ = reinterpret_cast<int (*)()>(sym("nvmlShutdown"));
        auto by_pci = reinterpret_cast<int (*)(const char*, void**)>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
        mem_ = reinterpret_cast<int (*)(void*, Memory*)>(sym("nvmlDeviceGetMemoryInfo"));
        procs_[0] = reinterpret_cast<ProcFn>(sym("nvmlDeviceGetComputeRunningProcesses_v2"));
        procs_[1] = reinterpret_cast<ProcFn>(sym("nvmlDeviceGetGraphicsRunningProcesses_v2"));
        if (!init || !shutdown_ || !by_pci || !mem_ || init() != 0) {
            shutdown_ = nullptr;
            return false;
        }
        if (by_pci(pci, &dev_) != 0) dev_ = nullptr;
        if (!dev_) shutdown_();
        return dev_ != nullptr;
    }
    [[nodiscard]] bool ok() const { return dev_ != nullptr; }
    [[nodiscard]] std::uint64_t used() const {  // the whole device
        Memory m{};
        return dev_ && mem_(dev_, &m) == 0 ? m.used : kNone;
    }
    // this process's device memory (compute + graphics lists; kNone when the driver does not report it, e.g. WDDM)
    [[nodiscard]] std::uint64_t process_used(unsigned pid) const {
        if (!dev_) return kNone;
        std::uint64_t total = 0;
        bool found = false;
        for (const ProcFn f : procs_) {
            if (!f) continue;
            ProcInfo info[64];
            unsigned n = 64;
            if (f(dev_, &n, info) != 0) continue;
            for (unsigned i = 0; i < n && i < 64; ++i)
                if (info[i].pid == pid && info[i].used != kNone) {
                    total = std::max<std::uint64_t>(total, info[i].used);  // the same allocation may be in both lists
                    found = true;
                }
        }
        return found ? total : kNone;
    }

private:
    struct Memory {
        unsigned long long total, free, used;
    };
    struct ProcInfo {  // nvmlProcessInfo_v2_t
        unsigned int pid;
        unsigned long long used;
        unsigned int gpu_instance, compute_instance;
    };
    using ProcFn = int (*)(void*, unsigned*, ProcInfo*);
    void* lib_ = nullptr;
    void* dev_ = nullptr;
    int (*shutdown_)() = nullptr;
    int (*mem_)(void*, Memory*) = nullptr;
    ProcFn procs_[2] = {nullptr, nullptr};
};

inline unsigned current_pid() {
#ifdef _WIN32
    return static_cast<unsigned>(GetCurrentProcessId());
#else
    return static_cast<unsigned>(getpid());
#endif
}

// One load's measurements. begin() BEFORE the first CUDA context of the process; stage() after each fixed part (the
// caller synchronizes first); sample() at any point. The NVML sampler (sample_ms > 0) runs from begin() to the end.
class Meter {
public:
    struct Point {
        std::uint64_t device_used = 0, device_total = 0;  // cudaMemGetInfo
        std::uint64_t charged = 0;                        // device_used - before (this process)
        std::uint64_t host_commit = kNone, host_peak = kNone;
        std::uint64_t nvml_used = kNone, nvml_process = kNone;
    };
    explicit Meter(std::uint32_t sample_ms) : sample_ms_(sample_ms) {}
    Meter(const Meter&) = delete;
    Meter& operator=(const Meter&) = delete;
    ~Meter() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (sampler_.joinable()) sampler_.join();
    }

    // device = the CUDA device the replica runs on (the current one); creates its context
    void begin(int device) {
        char pci[32] = {0};
        if (cudaDeviceGetPCIBusId(pci, sizeof pci, device) == cudaSuccess && nvml_.open(pci)) {
            const auto u = nvml_.used();
            if (u != kNone) {
                before_ = u;
                baseline_ = "nvml";
            }
        }
        pid_ = current_pid();
        if (cudaSetDevice(device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) throw std::runtime_error("budget: cannot create the CUDA context");
        last_ = sample();
        if (before_ > last_.device_used) before_ = last_.device_used;  // NVML and cudaMemGetInfo disagree: charge more, never less
        last_.charged = last_.device_used - before_;
        stages_.push_back({"context", last_.charged});
        if (nvml_.ok() && sample_ms_) sampler_ = std::thread([this] { run_sampler(); });
    }
    [[nodiscard]] Point sample() const {
        Point p;
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            p.device_used = total_b - free_b;
            p.device_total = total_b;
        }
        p.charged = p.device_used > before_ ? p.device_used - before_ : 0U;
        const auto h = host_mem();
        p.host_commit = h.commit;
        p.host_peak = h.peak;
        if (nvml_.ok()) {
            p.nvml_used = nvml_.used();
            p.nvml_process = nvml_.process_used(pid_);
        }
        return p;
    }
    // a fixed part ends: its bytes = charged now - charged at the previous stage
    Point stage(const std::string& name) {
        const Point p = sample();
        const std::uint64_t prev = stages_.empty() ? 0U : total_charged();
        stages_.push_back({name, p.charged > prev ? p.charged - prev : 0U});
        last_ = p;
        return p;
    }
    [[nodiscard]] std::uint64_t total_charged() const {
        std::uint64_t s = 0;
        for (const auto& st : stages_) s += st.second;
        return s;
    }
    [[nodiscard]] const std::vector<std::pair<std::string, std::uint64_t>>& stages() const { return stages_; }
    [[nodiscard]] std::uint64_t before() const { return before_; }
    [[nodiscard]] const char* baseline() const { return baseline_; }
    [[nodiscard]] bool nvml() const { return nvml_.ok(); }
    [[nodiscard]] std::uint32_t sample_ms() const { return sampler_.joinable() ? sample_ms_ : 0U; }
    // the background sampler's peaks (kNone: none): device used - before, and this process's NVML figure
    [[nodiscard]] std::uint64_t sampler_peak() const { const auto v = peak_charged_.load(); return v ? v : kNone; }
    [[nodiscard]] std::uint64_t sampler_process_peak() const { const auto v = peak_process_.load(); return v ? v : kNone; }
    [[nodiscard]] std::uint64_t sampler_samples() const { return samples_.load(); }

private:
    void run_sampler() {
        std::unique_lock<std::mutex> lk(mu_);
        while (!stop_) {
            lk.unlock();
            const auto u = nvml_.used();
            if (u != kNone) {
                const std::uint64_t c = u > before_ ? u - before_ : 0U;
                auto cur = peak_charged_.load();
                while (c > cur && !peak_charged_.compare_exchange_weak(cur, c)) {
                }
            }
            const auto pu = nvml_.process_used(pid_);
            if (pu != kNone) {
                auto cur = peak_process_.load();
                while (pu > cur && !peak_process_.compare_exchange_weak(cur, pu)) {
                }
            }
            samples_.fetch_add(1);
            lk.lock();
            cv_.wait_for(lk, std::chrono::milliseconds(sample_ms_), [this] { return stop_; });
        }
    }
    std::uint32_t sample_ms_ = 0;
    Nvml nvml_;
    unsigned pid_ = 0;
    std::uint64_t before_ = 0;
    const char* baseline_ = "none";  // none: everything used on the device is charged (conservative)
    Point last_;
    std::vector<std::pair<std::string, std::uint64_t>> stages_;
    std::thread sampler_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::atomic<std::uint64_t> peak_charged_{0}, peak_process_{0}, samples_{0};
};

}  // namespace replica_budget
