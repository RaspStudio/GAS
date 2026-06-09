// perf_scope.h - per-thread perf counters with enable/disable gates
#pragma once
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>

namespace perfmini {

inline long perf_event_open(perf_event_attr* hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct EventFD {
    int fd = -1;
    uint64_t id = 0;
    std::string name;
};

class Group {
public:
    void add_hw(const char* name, uint64_t config) {
        perf_event_attr pea{};
        pea.type = PERF_TYPE_HARDWARE;
        pea.size = sizeof(pea);
        pea.config = config;
        common_attr(pea);
        entries_.push_back({pea, name});
    }

    void add_cache(const char* name, uint64_t cache, uint64_t op, uint64_t result) {
        perf_event_attr pea{};
        pea.type = PERF_TYPE_HW_CACHE;
        pea.size = sizeof(pea);
        pea.config = cache | (op << 8) | (result << 16);
        common_attr(pea);
        entries_.push_back({pea, name});
    }

    bool open_current_thread() {
        if (entries_.empty()) return false;
        // leader
        int group_fd = -1;
        for (size_t i = 0; i < entries_.size(); ++i) {
            auto& e = entries_[i];
            int fd = (int)perf_event_open(&e.attr, /*pid=*/0, /*cpu=*/-1, group_fd, /*flags=*/0);
            if (fd < 0) {
                std::fprintf(stderr, "[perf] open %s failed: %s\n", e.name.c_str(), std::strerror(errno));
                close_all();
                return false;
            }
            uint64_t id = 0;
            if (ioctl(fd, PERF_EVENT_IOC_ID, &id) != 0) {
                std::fprintf(stderr, "[perf] ioctl ID %s failed: %s\n", e.name.c_str(), std::strerror(errno));
                ::close(fd); close_all(); return false;
            }
            fds_.push_back({fd, id, e.name});
            if (group_fd == -1) group_fd = fd; // first one is leader
        }
        leader_fd_ = fds_.front().fd;
        return true;
    }

    void reset() {
        if (leader_fd_ >= 0) {
            int r = ioctl(leader_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
            if (r) {
                std::fprintf(stderr, "[perf] reset failed: %s\n", std::strerror(errno));
            }
        }
    }
    void enable() {
        if (leader_fd_ >= 0) {
            int r = ioctl(leader_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
            if (r) {
                std::fprintf(stderr, "[perf] enable failed: %s\n", std::strerror(errno));
            }
        }
    }
    void disable() {
        if (leader_fd_ >= 0) {
            int r = ioctl(leader_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
            if (r) {
                std::fprintf(stderr, "[perf] disable failed: %s\n", std::strerror(errno));
            }
        }
    }

    void read_and_print(FILE* out = stderr) {
        if (leader_fd_ < 0) return;
        struct ReadValue { uint64_t value; uint64_t id; };
        struct ReadFormatHeader {
            uint64_t nr;
            uint64_t time_enabled;
            uint64_t time_running;
            ReadValue values[]; // flexible
        };

        const size_t n = fds_.size();
        const size_t sz = sizeof(ReadFormatHeader) + n * sizeof(ReadValue);
        std::vector<char> buf(sz);
        ssize_t got = ::read(leader_fd_, buf.data(), buf.size());
        if (got != (ssize_t)sz) {
            std::fprintf(stderr, "[perf] read failed: got %zd, want %zu (%s)\n",
                         got, sz, std::strerror(errno));
            return;
        }
        auto* rf = reinterpret_cast<ReadFormatHeader*>(buf.data());
        const double scale = rf->time_running ? (double)rf->time_enabled / (double)rf->time_running : 1.0;

        // map by id
        for (size_t i = 0; i < n; ++i) {
            uint64_t id = rf->values[i].id;
            uint64_t raw = rf->values[i].value;
            auto it = std::find_if(fds_.begin(), fds_.end(), [&](const EventFD& e){ return e.id == id; });
            const char* name = (it==fds_.end()) ? "unknown" : it->name.c_str();
            double val = (double)raw * scale;
            std::fprintf(out, "[perf] %-22s : %.0f\n", name, val);
        }

        fprintf(out, "[perf] time_enabled=%llu ns, time_running=%llu ns\n",
        (unsigned long long)rf->time_enabled, (unsigned long long)rf->time_running);
        if (rf->time_running == 0) {
            fprintf(out, "[perf] WARNING: time_running==0 (event was never scheduled), readings will all be 0\n");
        }
    }

    ~Group() { close_all(); }

private:
    struct Entry { perf_event_attr attr; std::string name; };
    std::vector<Entry> entries_;
    std::vector<EventFD> fds_;
    int leader_fd_ = -1;

    static void common_attr(perf_event_attr& pea) {
        pea.disabled = 1;
        pea.exclude_kernel = 1;
        pea.exclude_hv = 1;
        pea.exclude_idle = 1;
        pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID
                        | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        pea.sample_type = 0;
        pea.pinned = 0;
    }

    void close_all() {
        for (auto& e : fds_) if (e.fd >= 0) ::close(e.fd);
        fds_.clear();
        leader_fd_ = -1;
    }
};

inline Group make_default_group() {
    Group g;
    g.add_hw("cycles",        PERF_COUNT_HW_CPU_CYCLES);
    g.add_hw("instructions",  PERF_COUNT_HW_INSTRUCTIONS);
    // g.add_hw("branches",      PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    // g.add_hw("branch-misses", PERF_COUNT_HW_BRANCH_MISSES);

    g.add_cache("L1D-loads",        PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    g.add_cache("L1D-load-misses",  PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_MISS);

    // g.add_cache("LLC-loads",        PERF_COUNT_HW_CACHE_LL,  PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    // g.add_cache("LLC-load-misses",  PERF_COUNT_HW_CACHE_LL,  PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_MISS);

    // g.add_cache("dTLB-loads",       PERF_COUNT_HW_CACHE_DTLB,PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    // g.add_cache("dTLB-load-misses", PERF_COUNT_HW_CACHE_DTLB,PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_MISS);
    return g;
}

} // namespace perfmini
