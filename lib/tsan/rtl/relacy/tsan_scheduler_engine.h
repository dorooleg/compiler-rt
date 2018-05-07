#ifndef TSAN_FIBER_H
#define TSAN_FIBER_H

/*
 * Only Linux support
 */

#include "sanitizer_common/sanitizer_vector.h"
#include "tsan_generator_paths.h"
#include "tsan_threads_box.h"
#include "tsan_scheduler.h"
#include "tsan_platform.h"

#define FIBER_STACK_SIZE 16384
extern THREADLOCAL char cur_thread_placeholder[];

namespace __tsan {
namespace __relacy {

class FiberContext : public ThreadContext {
  public:
   FiberContext(void* fiber_context = nullptr, char* tls = nullptr, FiberContext* parent = nullptr, int tid = 0);

   void* GetFiberContext();

   void SetFiberContext(void* fiber_context);

   char* GetTls();

   void SetTls(char* tls);

   FiberContext* GetParent();

   void SetParent(FiberContext* parent);

  private:
   void *ctx_;
   char *tls_;
   FiberContext *parent_;
};

class SchedulerEngine {
  public:
   SchedulerEngine();

   FiberContext *CreateFiber(void *th, void *attr, void (*callback)(), void *param);

   void Yield(FiberContext *context);

   void AddFiberContext(int tid, FiberContext *context);

   void YieldByTid(int tid);

   void YieldByIndex(uptr index);

   void Yield();

   void Join(int wait_tid);

   FiberContext *GetParent();

   void StopThread();

   void Start();

   int MaxRunningTid();

   bool IsRunningTid(int tid);

   void InitializeTLS();

   SchedulerType GetSchedulerType();

   PlatformType GetPlatformType();

  private:
   static void *CreateSharedMemory(uptr size);

  private:
   GeneratorPaths *paths_;
   ThreadsBox threads_box_;
   Scheduler *scheduler_;
   Platform *platform_;
  public:
   uptr tls_addr_;
   char *tls_base_;
   uptr tls_size_;
};

}
#if SANITIZER_RELACY_SCHEDULER
extern __relacy::SchedulerEngine _fiber_manager;
#endif

}

#endif // TSAN_FIBER_H
