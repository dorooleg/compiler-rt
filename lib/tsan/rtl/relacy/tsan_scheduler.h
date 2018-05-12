#ifndef TSAN_SCHEDULER_H
#define TSAN_SCHEDULER_H

#include "schedulers/tsan_scheduler_type.h"
#include "tsan_thread_context.h"

namespace __tsan {
namespace __relacy {

class Scheduler {
  public:
   virtual ThreadContext* Yield() = 0;

   virtual void Start() = 0;

   virtual void Finish() = 0;

   virtual void Initialize() = 0;

   virtual bool IsEnd() = 0;

   virtual SchedulerType GetType() = 0;

   virtual ~Scheduler() = default;
};

}
}
#endif //TSAN_SCHEDULER_H
