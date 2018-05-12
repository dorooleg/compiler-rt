#include <ucontext.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <rtl/relacy/tsan_scheduler_engine.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include "rtl/relacy/schedulers/tsan_random_generator.h"
#include "rtl/tsan_rtl.h"
#include <linux/unistd.h>
#include <asm/ldt.h>
#include <sys/syscall.h>

//schedulers
#include "rtl/relacy/schedulers/tsan_all_states_scheduler.h"
#include "rtl/relacy/schedulers/tsan_fixed_window_scheduler.h"
#include "rtl/relacy/schedulers/tsan_full_path_scheduler.h"
#include "rtl/relacy/schedulers/tsan_parallel_full_path_scheduler.h"
#include "rtl/relacy/schedulers/tsan_random_scheduler.h"
#include "rtl/relacy/schedulers/tsan_random_with_different_distributions_scheduler.h"

//platforms
#include "rtl/relacy/platforms/tsan_fiber_tls_swap_platfrom.h"
#include "rtl/relacy/platforms/tsan_fiber_tls_copy_platform.h"
#include "rtl/relacy/platforms/tsan_pthread_platform.h"

namespace __tsan {
namespace __relacy {

SchedulerEngine::SchedulerEngine() {
  if (!strcmp(flags()->scheduler_type, "")) {
    scheduler_ = nullptr;
  } else if (!strcmp(flags()->scheduler_type, "random")) {
    scheduler_ = static_cast<RandomScheduler *>(InternalCalloc(1, sizeof(RandomScheduler)));
    new (scheduler_) RandomScheduler{threads_box_};
  } else if (!strcmp(flags()->scheduler_type, "all_states")) {
    scheduler_ = static_cast<AllStatesScheduler *>(InternalCalloc(1, sizeof(AllStatesScheduler)));
    new (scheduler_) AllStatesScheduler{threads_box_};
  } else if (!strcmp(flags()->scheduler_type, "full_path")) {
    scheduler_ = static_cast<FullPathScheduler *>(InternalCalloc(1, sizeof(FullPathScheduler)));
    new (scheduler_) FullPathScheduler{threads_box_};
  } else if (!strcmp(flags()->scheduler_type, "parallel_full_path")) {
    scheduler_ = static_cast<ParallelFullPathScheduler *>(InternalCalloc(1, sizeof(ParallelFullPathScheduler)));
    new (scheduler_) ParallelFullPathScheduler{};
  } else if (!strcmp(flags()->scheduler_type, "random_with_different_distributions")) {
    scheduler_ = static_cast<RandomWithDifferentDistributionsScheduler *>(InternalCalloc(1, sizeof(RandomWithDifferentDistributionsScheduler)));
    new (scheduler_) RandomWithDifferentDistributionsScheduler{threads_box_};
  } else if (!strcmp(flags()->scheduler_type, "fixed_window")) {
    scheduler_ = static_cast<FixedWindowScheduler *>(InternalCalloc(1, sizeof(FixedWindowScheduler)));
    new (scheduler_) FixedWindowScheduler{};
  } else {
    Printf("FATAL: ThreadSanitizer invalid scheduler type. Please check TSAN_OPTIONS!\n");
    Die();
  }

  if (!strcmp(flags()->scheduler_platform, "")) {
    platform_ = nullptr;
  } else if (!strcmp(flags()->scheduler_platform, "fiber_tls_swap")) {
    platform_ = static_cast<FiberTlsSwapPlatform *>(InternalCalloc(1, sizeof(FiberTlsSwapPlatform)));
    new (platform_) FiberTlsSwapPlatform{threads_box_};
  } else if (!strcmp(flags()->scheduler_platform, "fiber_tls_copy")) {
    platform_ = static_cast<FiberTlsCopyPlatform *>(InternalCalloc(1, sizeof(FiberTlsCopyPlatform)));
    new (platform_) FiberTlsCopyPlatform{};
  } else if (!strcmp(flags()->scheduler_platform, "pthread")) {
    platform_ = static_cast<PthreadPlatform *>(InternalCalloc(1, sizeof(PthreadPlatform)));
    new (platform_) PthreadPlatform{};
  } else {
    Printf("FATAL: ThreadSanitizer invalid platform type. Please check TSAN_OPTIONS!\n");
    Die();
  }

  if (scheduler_ == nullptr || platform_ == nullptr) {
    if (scheduler_ != nullptr || platform_ != nullptr) {
      Printf("FATAL: ThreadSanitizer platform + scheduler invalid combination\n");
      Die();
    }

    return;
  }

  Printf("Platform %s Type %s\n", flags()->scheduler_platform, flags()->scheduler_type);

  Start();
}

ThreadContext *SchedulerEngine::CreateFiber(void *th, void *attr, void (*callback)(), void *param) {
  return GetPlatformType() == PlatformType::OS ? nullptr : platform_->Create(th, attr, callback, param);
}

void SchedulerEngine::Yield(ThreadContext *context) {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }

  platform_->Yield(context);
}

void SchedulerEngine::AddFiberContext(int tid, ThreadContext *context) {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }
  context->SetTid(tid);
  threads_box_.AddRunning(context);
}

void SchedulerEngine::Yield() {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }
  Yield(scheduler_->Yield());
}

void SchedulerEngine::Join(int wait_tid) {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }

  if (threads_box_.ContainsStoppedByTid(wait_tid)) {
    return;
  }

  if (threads_box_.GetCountRunning() == 0) {
    Printf("FATAL: ThreadSanitizer joining last thread\n");
    Die();
  }

  ThreadContext *context = threads_box_.ExtractRunningByTid(threads_box_.GetCurrentThread()->GetTid());

  if (context == nullptr) {
    Printf("FATAL: ThreadSanitizer is not existing thread\n");
    Die();
  }

  ThreadContext* wait_context = threads_box_.GetRunningByTid(wait_tid);

  if (wait_context == nullptr) {
    wait_context = threads_box_.GetWaitingByTid(wait_tid);
  }

  if (wait_context == nullptr) {
    wait_context = threads_box_.GetSlepingByTid(wait_tid);
  }

  if (wait_context == nullptr) {
    wait_context = threads_box_.GetStoppedByTid(wait_tid);
  }

  if (wait_context == nullptr) {
    wait_context = threads_box_.GetJoiningByTid(wait_tid).GetCurrentThread();
  }

  threads_box_.AddJoining(JoinContext { context, wait_context });
}

void SchedulerEngine::StopThread() {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }

  threads_box_.AddStopped(threads_box_.ExtractRunningByTid(threads_box_.GetCurrentThread()->GetTid()));
  threads_box_.WakeupJoiningByWaitTid(threads_box_.GetCurrentThread()->GetTid());
}


void SchedulerEngine::Start() {
 if (GetPlatformType() == PlatformType::OS) {
   return;
 }
 scheduler_->Initialize();
 while (!scheduler_->IsEnd()) {
    pid_t pid = fork();
    if (pid < 0) {
      Printf("FATAL: ThreadSanitizer fork error\n");
      Die();
    }
    if (pid != 0) {
      scheduler_->Start();
      int status;
      if (waitpid(pid, &status, WUNTRACED | WCONTINUED) == -1) {
        Printf("FATAL: ThreadSanitizer waitpid error\n");
        Die();
      }
      if (WEXITSTATUS(status) != 0) {
        Printf("FATAL: ThreadSanitizer invalid status code\n");
        Die();
      }
      scheduler_->Finish();
    } else {
      break;
    }
  }
  if (scheduler_->IsEnd()) {
     scheduler_->Start();
 }
}

ThreadContext* SchedulerEngine::GetParent() {
  return GetSchedulerType() == SchedulerType::OS ? nullptr : threads_box_.GetCurrentThread()->GetParent();
}

SchedulerType SchedulerEngine::GetSchedulerType() {
  return scheduler_ ? scheduler_->GetType() : SchedulerType::OS;
}

PlatformType SchedulerEngine::GetPlatformType() {
  return platform_ ? platform_->GetType() : PlatformType::OS;
}

void SchedulerEngine::Initialize() {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }

  platform_->Initialize();
}

}

#if SANITIZER_RELACY_SCHEDULER
__relacy::SchedulerEngine _fiber_manager;
#endif

}
