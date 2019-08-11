#include "fuzzing_scheduler.h"
#include <rtl/tsan_rtl.h>
#include <sanitizer_common/sanitizer_allocator_internal.h>
#include <sanitizer_common/sanitizer_placement_new.h>
#include <interception/interception.h>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

namespace __interception {
  extern int (*real_pthread_create)(void*, void*, void *(*)(void*), void*);
  extern int (*real_pthread_detach)(void*);
}

namespace __tsan {

namespace {

struct NullFuzzingScheduler : IFuzzingScheduler {
  void SynchronizationPoint() override {
  }
};

thread_local u64 tid = 0;
u64 max_tid = 0;

struct RandomFuzzingScheduler : IFuzzingScheduler {
  RandomFuzzingScheduler() {
    srand(NanoTime());
    pthread_t t;
    REAL(pthread_create)(&t, NULL, reinterpret_cast<void*(*)(void*)>(&RandomFuzzingScheduler::WatchDog), this);
    REAL(pthread_detach)(&t);
  }


private:
  class Stats {
    u64 count_running = 0;
    u64 count_wait = 0;
  public:
    void IncRunning() {
      __atomic_add_fetch(&count_running, 1, __ATOMIC_SEQ_CST);
    }

    void DecRunning() {
      __atomic_add_fetch(&count_running, -1, __ATOMIC_SEQ_CST);
    }

    u64 CountRunning() {
      return __atomic_load_n(&count_running, __ATOMIC_SEQ_CST);
    }

    void IncWait() {
      __atomic_add_fetch(&count_wait, 1, __ATOMIC_SEQ_CST);
    }

    void DecWait() {
      __atomic_add_fetch(&count_wait, -1, __ATOMIC_SEQ_CST);
    }

    u64 CountWait() {
      return __atomic_load_n(&count_wait, __ATOMIC_SEQ_CST);
    }
  };

  enum class ThreadState {
    UNKNOWN,
    RUNNING,
    WAIT,
    OUT_TIME
  };

  struct ThreadContext {
    ThreadState state = ThreadState::UNKNOWN;
    u64 start_time = 0;
  };

  ThreadContext contexts[65536] = {};
  Stats stats;

  u64 GetTid() {
    if (tid == 0) {
      tid = __atomic_add_fetch(&max_tid, 1, __ATOMIC_RELAXED);
    }
    if (tid > 65535) {
      Printf("FATAL: ThreadSanitizer The maximum number of threads created during the program should not exceed 65535");
      Die();
    }
    return tid;
  }

  void SynchronizationPoint() override {
    auto tid = GetTid();
    auto old_state = __atomic_load_n(&contexts[tid].state, __ATOMIC_SEQ_CST);
    __atomic_store_n(&contexts[tid].state, ThreadState::WAIT, __ATOMIC_SEQ_CST);
    if (old_state == ThreadState::RUNNING) {
      auto next_tid = GetNextTid();
      __atomic_store_n(&contexts[next_tid].start_time, NanoTime(), __ATOMIC_SEQ_CST);
      __atomic_store_n(&contexts[next_tid].state, ThreadState::RUNNING, __ATOMIC_SEQ_CST);
    }
    //PrintStates();
    while (__atomic_load_n(&contexts[tid].state, __ATOMIC_SEQ_CST) == ThreadState::WAIT) {
      internal_sched_yield();
    }
  }

  u64 GetNextTid() {
      const u64 local_max_tid = __atomic_load_n(&max_tid, __ATOMIC_SEQ_CST);
      const u64 next_tid = rand() % local_max_tid + 1;
      for (u64 i = 0; i < local_max_tid; i++) {
        if (__atomic_load_n(&contexts[(next_tid + i) % max_tid + 1].state, __ATOMIC_SEQ_CST) == ThreadState::WAIT) {
          return (next_tid + i) % max_tid + 1;
        }
      }
      return next_tid;
  }

  void PrintStates() {
      static bool flag = false;
      while (__sync_val_compare_and_swap(&flag, false, true) != false) {
        internal_sched_yield();
      }

      const u64 local_max_tid = __atomic_load_n(&max_tid, __ATOMIC_SEQ_CST);
      for (u64 i = 1; i <= local_max_tid; i++) {
        auto state = __atomic_load_n(&contexts[i].state, __ATOMIC_SEQ_CST);
        Printf("(%d,%d) ", i, state);
      }
      Printf("\n");
      __atomic_store_n(&flag, false, __ATOMIC_SEQ_CST);
  }

  void* WatchDog() {
    while (true) {
      usleep(200 * 1000);
      u64 local_max_tid = __atomic_load_n(&max_tid, __ATOMIC_SEQ_CST);
      for (u64 i = 1; i <= local_max_tid; i++) {
        if (__atomic_load_n(&contexts[i].state, __ATOMIC_SEQ_CST) == ThreadState::RUNNING && __atomic_load_n(&contexts[i].start_time, __ATOMIC_SEQ_CST) + 200 * 1000 * 1000ULL <= NanoTime()) {
          __atomic_store_n(&contexts[i].state, ThreadState::OUT_TIME, __ATOMIC_SEQ_CST);
        }
      }
      bool exists_running = false;
      for (u64 i = 1; i <= local_max_tid; i++) {
        if (__atomic_load_n(&contexts[i].state, __ATOMIC_SEQ_CST) == ThreadState::RUNNING) {
          exists_running = true;
        }
      }
      if (!exists_running) {
        auto next_tid = GetNextTid();
        __atomic_store_n(&contexts[next_tid].start_time, NanoTime(), __ATOMIC_SEQ_CST);
        __atomic_store_n(&contexts[next_tid].state, ThreadState::RUNNING, __ATOMIC_SEQ_CST);
      }
      //PrintStates();
    }
    return nullptr;
  }

};

IFuzzingScheduler& FuzzingSchedulerDispatcher() {
  if (!strcmp(flags()->fuzzing_scheduler, "")) {
    auto* scheduler = static_cast<NullFuzzingScheduler *>(InternalCalloc(1, sizeof(NullFuzzingScheduler)));
    new (scheduler) NullFuzzingScheduler;
    return *scheduler;
  } else if (!strcmp(flags()->fuzzing_scheduler, "random")) {
    auto* scheduler = static_cast<RandomFuzzingScheduler *>(InternalCalloc(1, sizeof(RandomFuzzingScheduler)));
    new (scheduler) RandomFuzzingScheduler;
    Printf("WARNING! ThreadSanitizer lunched under the management of a random fuzzing scheduler new\n");
    return *scheduler;
  } else {
    Printf("FATAL: ThreadSanitizer invalid fuzzing scheduler. Please check TSAN_OPTIONS!\n");
    Die();
  }
}

}

IFuzzingScheduler& GetFuzzingScheduler() {
  static IFuzzingScheduler& scheduler = FuzzingSchedulerDispatcher();
  return scheduler;
}

}  // namespace __tsan
