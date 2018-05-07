#ifndef TSAN_SCHEDULER_H
#define TSAN_SCHEDULER_H

#include "schedulers/tsan_scheduler_type.h"

namespace __tsan {
namespace __relacy {

class Scheduler {
  public:
   virtual void Yield() = 0;

   virtual void Start() = 0;

   virtual void Finish() = 0;

   virtual void Initialize() = 0;

   virtual SchedulerType GetType() = 0;

   virtual ~Scheduler() = default;
};

}
}
#endif //TSAN_SCHEDULER_H
