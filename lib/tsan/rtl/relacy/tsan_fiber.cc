#include <ucontext.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <rtl/relacy/tsan_fiber.h>
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

unsigned long get_tls_addr()
{
  unsigned long addr;
  asm("mov %%fs:0, %0" : "=r"(addr));
  return addr;
}

void set_tls_addr(unsigned long addr)
{
  asm("mov %0, %%fs:0" : "+r"(addr));
}


FiberManager::FiberManager() {
  FiberContext* current_thread = static_cast<FiberContext*>(InternalCalloc(1, sizeof(FiberContext)));
  current_thread->ctx = InternalCalloc(1, sizeof(ucontext_t));
  current_thread->parent = current_thread;
  current_thread->tid = 0;
  ucontext_t& context = *static_cast<ucontext_t*>(current_thread->ctx);
  context.uc_stack.ss_flags = 0;
  context.uc_link = nullptr;

  uptr stk_addr = 0;
  uptr stk_size = 0;

  InitTlsSize();
  GetThreadStackAndTls(true, &stk_addr, &stk_size, &tls_addr_, &tls_size_);

  current_thread->tls = static_cast<char*>(InternalCalloc(tls_size_, 1));
  tls_addr_ = get_tls_addr();
  current_thread->tls = reinterpret_cast<char *>(get_tls_addr());
  //internal_memcpy(current_thread->tls, reinterpret_cast<const void *>(tls_addr_), tls_size_);

  tls_base_ = static_cast<char*>(InternalCalloc(tls_size_, 1));

  internal_memcpy(tls_base_, reinterpret_cast<const char *>(tls_addr_) - tls_size_, tls_size_);
  uptr offset = (uptr)cur_thread() - (tls_addr_ - tls_size_);
  internal_memset(tls_base_ + offset, 0, sizeof(ThreadState));

  running_.PushBack(current_thread);
  current_thread_ = current_thread;
  current_thread_->tls -= tls_size_;
  //srand(static_cast<unsigned int>(time(nullptr)));

  Start();
}

FiberContext* FiberManager::CreateFiber(void *th, void *attr, void (*callback)(), void * param) {
  (void)th;
  (void)attr;

  FiberContext* fiber_context = static_cast<FiberContext*>(InternalCalloc(1, sizeof(FiberContext)));
  fiber_context->ctx = static_cast<struct ucontext_t*>(InternalCalloc(1, sizeof(ucontext_t)));
  if (getcontext((ucontext_t*)fiber_context->ctx) == -1) {
    Printf("FATAL: ThreadSanitizer getcontext error in the moment creating fiber\n");
    Die();
  }

  ucontext_t& context = *static_cast<ucontext_t*>(fiber_context->ctx);
  context.uc_stack.ss_sp = InternalCalloc(FIBER_STACK_SIZE, sizeof(char));
  context.uc_stack.ss_size = FIBER_STACK_SIZE;
  context.uc_stack.ss_flags = 0;
  context.uc_link = static_cast<struct ucontext_t *>(current_thread_->ctx);
  fiber_context->parent = current_thread_;
  fiber_context->tls = static_cast<char *>(InternalCalloc(tls_size_, 1));
  internal_memcpy(fiber_context->tls, tls_base_, tls_size_);
  makecontext(static_cast<ucontext_t *>(fiber_context->ctx), callback, 1, param);
  return fiber_context;
}

void FiberManager::Yield(FiberContext* context) {
  //current_thread_->ApplyChanges(tls_addr_);
  FiberContext* old_thread = current_thread_;
  current_thread_ = context;


  //uptr offset = (uptr)cur_thread() - tls_addr_;
  //internal_memcpy(old_thread->tls + offset, reinterpret_cast<const void *>(tls_addr_ + offset), sizeof(ThreadState));
  //internal_memcpy(reinterpret_cast<char *>(tls_addr_  + offset), current_thread_->tls + offset, sizeof(ThreadState));
  int res = swapcontext(static_cast<struct ucontext_t *>(old_thread->ctx),
                        static_cast<const struct ucontext_t *>(current_thread_->ctx));

  InitializeTLS();

  if (res != 0) {
    Printf("FATAL: ThreadSanitizer swapcontext error in the moment yield fiber\n");
    Die();
  }
}

void FiberManager::AddFiberContext(int tid, FiberContext* context) {
  context->tid = tid;
  running_.PushBack(context);
}

void FiberManager::YieldByTid(int tid) {
  for (uptr i = 0; i < running_.Size(); i++) {
    if (running_[i]->tid == tid)
    {
      Yield(running_[i]);
      return;
    }
  }

  Printf("FATAL: ThreadSanitizer yield error because tid is not found\n");
  Die();
}

void FiberManager::YieldByIndex(uptr index) {
  if (index >= running_.Size()) {
    Printf("FATAL: ThreadSanitizer index out of bound in yield by index\n");
    Die();
  }

  Yield(running_[index]);
}

int FiberManager::MaxRunningTid() {
  int m = 0;
  for (int i = 0; i < running_.Size(); i++) {
    m = max(running_[i]->tid, m);
  }
  return m;
}

bool FiberManager::IsRunningTid(int tid) {
  for (int i = 0; i < running_.Size(); i++) {
    if (running_[i]->tid == tid) {
      return true;
    }
  }
  return false;
}

  RandomGenerator *generator_;

void FiberManager::Yield() {
  if (running_.Size() == 0) {
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
  YieldByIndex(static_cast<uptr>(generator_->Rand(static_cast<int>(running_.Size()))));
}


FiberContext* FiberManager::GetParent() {
  return current_thread_->parent;
}

void FiberManager::Join(int wait_tid) {
  for (uptr i = 0; i < stoped_.Size(); i++) {
    if (stoped_[i]->tid == wait_tid) {
      return;
    }
  }

  if (running_.Size() == 0) {
    Printf("FATAL: ThreadSanitizer joining last thread\n");
    Die();
  }

  FiberContext* context = nullptr;

  for (uptr i = 0; i < running_.Size(); i++) {
    if (current_thread_->tid == running_[i]->tid) {
      context = running_[i];
      running_[i] = running_[running_.Size() - 1];
      running_.PopBack();
      break;
    }
  }

  if (context == nullptr) {
    Printf("FATAL: ThreadSanitizer is not existing thread\n");
    Die();
  }

  joining_.PushBack(JoinContext { wait_tid, context });
}

void FiberManager::StopThread() {
  for (uptr i = 0; i < running_.Size(); i++) {
    if (current_thread_->tid == running_[i]->tid) {
      stoped_.PushBack(current_thread_);
      running_[i] = running_[running_.Size() - 1];
      running_.PopBack();
      break;
    }
  }

  for (int i = 0; i < (int)joining_.Size(); i++) {
    if (joining_[i].waiting_tid == current_thread_->tid) {
      running_.PushBack(joining_[i].thread_info);
      joining_[i] = joining_[joining_.Size() - 1];
      joining_.PopBack();
      i--;
    }
  }
}


void FiberManager::Start() {
  if (paths_ == nullptr) {
    paths_ = new GeneratorPaths(static_cast<unsigned long *>(CreateSharedMemory(1024 * sizeof(unsigned long))),
                                static_cast<unsigned long *>(CreateSharedMemory(1024 * sizeof(unsigned long))), 1024);
  }
  if (generator_ == nullptr) {
    generator_ = new RandomGenerator();
  }
  while(true) {
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

FiberContext* FiberManager::GetCurrent() {
  return current_thread_;
}

void* FiberManager::CreateSharedMemory(uptr size) {
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

void FiberManager::InitializeTLS() {
  uptr descr_addr = (uptr)current_thread_->tls + tls_size_;
  set_tls_addr(descr_addr);
}

#if SANITIZER_RELACY_SCHEDULER
FiberManager _fiber_manager;
#endif

}
