#include <ucontext.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <rtl/relacy/tsan_scheduler_engine.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include "random_generator.h"
#include "rtl/tsan_rtl.h"
#include <linux/unistd.h>
#include <asm/ldt.h>
#include <sys/syscall.h>

namespace __tsan {
namespace __relacy {

FiberContext::FiberContext(void *fiber_context, char *tls, FiberContext *parent, int tid)
    : ThreadContext(tid), ctx_(fiber_context), tls_(tls), parent_(parent) {
}

void *FiberContext::GetFiberContext() {
  return ctx_;
}

void FiberContext::SetFiberContext(void *fiber_context) {
  ctx_ = fiber_context;
}

char *FiberContext::GetTls() {
  return tls_;
}

void FiberContext::SetTls(char *tls) {
  tls_ = tls;
}

FiberContext *FiberContext::GetParent() {
  return parent_;
}

void FiberContext::SetParent(FiberContext *parent) {
  parent_ = parent;
}

unsigned long get_tls_addr() {
  unsigned long addr;
  asm("mov %%fs:0, %0" : "=r"(addr));
  return addr;
}

void set_tls_addr(unsigned long addr) {
  asm("mov %0, %%fs:0" : "+r"(addr));
}


SchedulerEngine::SchedulerEngine() {

  if (!strcmp(flags()->scheduler_type, "")) {
    scheduler_ = nullptr;
  } else {
    Printf("FATAL: ThreadSanitizer invalid scheduler type. Please check TSAN_OPTIONS!\n");
    Die();
  }

  if (!strcmp(flags()->scheduler_platform, "")) {
    platform_ = nullptr;
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

  uptr stk_addr = 0;
  uptr stk_size = 0;
  InitTlsSize();
  GetThreadStackAndTls(true, &stk_addr, &stk_size, &tls_addr_, &tls_size_);
  tls_addr_ = get_tls_addr();

  FiberContext *current_thread = static_cast<FiberContext *>(InternalCalloc(1, sizeof(FiberContext)));
  new(current_thread) FiberContext{InternalCalloc(1, sizeof(ucontext_t)),
                                   reinterpret_cast<char *>(tls_addr_) - tls_size_, current_thread, 0};
  ucontext_t &context = *static_cast<ucontext_t *>(current_thread->GetFiberContext());
  context.uc_stack.ss_flags = 0;
  context.uc_link = nullptr;

  tls_base_ = static_cast<char *>(InternalCalloc(tls_size_, 1));

  internal_memcpy(tls_base_, reinterpret_cast<const char *>(tls_addr_) - tls_size_, tls_size_);
  uptr offset = (uptr) cur_thread() - (tls_addr_ - tls_size_);
  internal_memset(tls_base_ + offset, 0, sizeof(ThreadState));

  threads_box_.AddRunning(current_thread);
  threads_box_.SetCurrentThread(current_thread);

  Start();
}

FiberContext *SchedulerEngine::CreateFiber(void *th, void *attr, void (*callback)(), void *param) {
  (void) th;
  (void) attr;

  if (GetPlatformType() == PlatformType::OS) {
    return nullptr;
  }

  FiberContext *fiber_context = static_cast<FiberContext *>(InternalCalloc(1, sizeof(FiberContext)));
  new(fiber_context) FiberContext{static_cast<struct ucontext_t *>(InternalCalloc(1, sizeof(ucontext_t)))};
  if (getcontext(static_cast<ucontext_t *>(fiber_context->GetFiberContext())) == -1) {
    Printf("FATAL: ThreadSanitizer getcontext error in the moment creating fiber\n");
    Die();
  }

  ucontext_t &context = *static_cast<ucontext_t *>(fiber_context->GetFiberContext());
  context.uc_stack.ss_sp = InternalCalloc(FIBER_STACK_SIZE, sizeof(char));
  context.uc_stack.ss_size = FIBER_STACK_SIZE;
  context.uc_stack.ss_flags = 0;
  context.uc_link = static_cast<struct ucontext_t *>(static_cast<FiberContext*>(threads_box_.GetCurrentThread())->GetFiberContext());
  fiber_context->SetParent(static_cast<FiberContext*>(threads_box_.GetCurrentThread()));
  fiber_context->SetTls(static_cast<char *>(InternalCalloc(tls_size_, 1)));
  internal_memcpy(fiber_context->GetTls(), tls_base_, tls_size_);
  makecontext(static_cast<ucontext_t *>(fiber_context->GetFiberContext()), callback, 1, param);
  return fiber_context;
}

void SchedulerEngine::Yield(FiberContext *context) {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }
  //current_thread_->ApplyChanges(tls_addr_);
  FiberContext *old_thread = static_cast<FiberContext*>(threads_box_.GetCurrentThread());
  threads_box_.SetCurrentThread(context);


  //uptr offset = (uptr)cur_thread() - tls_addr_;
  //internal_memcpy(old_thread->tls + offset, reinterpret_cast<const void *>(tls_addr_ + offset), sizeof(ThreadState));
  //internal_memcpy(reinterpret_cast<char *>(tls_addr_  + offset), current_thread_->tls + offset, sizeof(ThreadState));
  int res = swapcontext(static_cast<struct ucontext_t *>(old_thread->GetFiberContext()),
                        static_cast<const struct ucontext_t *>(static_cast<FiberContext*>(threads_box_.GetCurrentThread())->GetFiberContext()));

  InitializeTLS();

  if (res != 0) {
    Printf("FATAL: ThreadSanitizer swapcontext error in the moment yield fiber\n");
    Die();
  }
}

void SchedulerEngine::AddFiberContext(int tid, FiberContext *context) {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }
  context->SetTid(tid);
  threads_box_.AddRunning(context);
}

void SchedulerEngine::YieldByTid(int tid) {
  Yield(static_cast<FiberContext*>(threads_box_.GetRunningByTid(tid)));
}

void SchedulerEngine::YieldByIndex(uptr index) {
  Yield(static_cast<FiberContext*>(threads_box_.GetRunningByIndex(index)));
}

int SchedulerEngine::MaxRunningTid() {
  return threads_box_.MaxRunningTid();
}

bool SchedulerEngine::IsRunningTid(int tid) {
  return threads_box_.ContainsRunningByTid(tid);
}

RandomGenerator *generator_;

void SchedulerEngine::Yield() {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }

  if (threads_box_.GetCountRunning() == 0) {
    Printf("FATAL: ThreadSanitizer yield count threads == 0\n");
    Die();
  }

  /*int tid = static_cast<int>(paths_->Yield(static_cast<unsigned long>(MaxRunningTid())));

  if (!IsRunningTid(tid)) {
    paths_->InvalidateThread();
    YieldByIndex(rand() % running_.Size());
    return;
  }

  YieldByTid(tid);*/
  YieldByIndex(static_cast<uptr>(generator_->Rand(static_cast<int>(threads_box_.GetCountRunning()))));
}


FiberContext *SchedulerEngine::GetParent() {
  return GetPlatformType() == PlatformType::OS ? nullptr : static_cast<FiberContext*>(threads_box_.GetCurrentThread())->GetParent();
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

  ThreadContext *context = threads_box_.ExtractRunningByTid(static_cast<FiberContext*>(threads_box_.GetCurrentThread())->GetTid());

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
  if (paths_ == nullptr) {
    paths_ = new GeneratorPaths(static_cast<unsigned long *>(CreateSharedMemory(1024 * sizeof(unsigned long))),
                                static_cast<unsigned long *>(CreateSharedMemory(1024 * sizeof(unsigned long))), 1024);
  }
  if (generator_ == nullptr) {
    generator_ = new RandomGenerator();
  }
  while (true) {
    //paths_->Start();
    //uptr offset = (uptr)&cur_thread_placeholder - tls_addr_;
    //internal_memcpy(reinterpret_cast<char *>(tls_addr_  + offset), current_thread_->tls + offset, sizeof(ThreadState));
    pid_t pid = fork();
    if (pid < 0) {
      Printf("FATAL: ThreadSanitizer fork error\n");
      Die();
    }
    if (pid != 0) {
      int status;
      if (waitpid(pid, &status, WUNTRACED | WCONTINUED) == -1) {
        Printf("FATAL: ThreadSanitizer waitpid error\n");
        Die();
      }
      if (WEXITSTATUS(status) != 0) {
        Printf("FATAL: ThreadSanitizer invalid status code\n");
        Die();
      }
      //paths_->Finish();
      generator_->NextGenerator();
    } else {
      break;
    }
  }
}

void *SchedulerEngine::CreateSharedMemory(uptr size) {
  // Our memory buffer will be readable and writable:
  int protection = PROT_READ | PROT_WRITE;

  // The buffer will be shared (meaning other processes can access it), but
  // anonymous (meaning third-party processes cannot obtain an address for it),
  // so only this process and its children will be able to use it:
  int visibility = MAP_ANONYMOUS | MAP_SHARED;

  // The remaining parameters to `mmap()` are not important for this use case,
  // but the manpage for `mmap` explains their purpose.
  return mmap(NULL, size, protection, visibility, 0, 0);
}

void SchedulerEngine::InitializeTLS() {
  if (GetPlatformType() == PlatformType::OS) {
    return;
  }
  uptr descr_addr = (uptr) static_cast<FiberContext*>(threads_box_.GetCurrentThread())->GetTls() + tls_size_;
  set_tls_addr(descr_addr);
}

SchedulerType SchedulerEngine::GetSchedulerType() {
  return scheduler_ ? scheduler_->GetType() : SchedulerType::OS;
}

PlatformType SchedulerEngine::GetPlatformType() {
  return platform_ ? platform_->GetType() : PlatformType::OS;
}

}

#if SANITIZER_RELACY_SCHEDULER
__relacy::SchedulerEngine _fiber_manager;
#endif

}
