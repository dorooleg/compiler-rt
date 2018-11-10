#ifndef TSAN_FIBER_H
#define TSAN_FIBER_H

/*
 * Only Linux support
 */

#include "sanitizer_common/sanitizer_vector.h"
#include "tsan/rtl/relacy/schedulers/tsan_generator_paths.h"
#include "tsan_threads_box.h"
#include "tsan_scheduler.h"
#include "tsan_platform.h"

namespace __tsan {
namespace __relacy {

class SchedulerEngine {
  public:
   SchedulerEngine();

   ThreadContext *CreateFiber(void *th, void *attr, void (*callback)(), void *param);

   void Yield(ThreadContext *context);

   void Yield();

   void AddFiberContext(int tid, ThreadContext *context);

   void Join(int wait_tid);

   void StopThread();

   int Lock(void *m);

   int Unlock(void *m);

   void Initialize();

   ThreadContext* GetParent();

   SchedulerType GetSchedulerType();

   PlatformType GetPlatformType();

   int CondWait(void *c, void *m);

   int Signal(void *c);

   int Broadcast(void *c);

  private:
   void Start();

  private:
   ThreadsBox threads_box_;
   Scheduler *scheduler_;
   Platform *platform_;
};


}
}

#if SANITIZER_RELACY_SCHEDULER
extern ::__tsan::__relacy::SchedulerEngine _scheduler_engine;
#endif


#endif // TSAN_FIBER_H
