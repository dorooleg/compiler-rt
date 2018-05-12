#ifndef TSAN_FIXED_WINDOW_SCHEDULER_H
#define TSAN_FIXED_WINDOW_SCHEDULER_H

#include "rtl/relacy/tsan_scheduler.h"

namespace __tsan {
namespace __relacy {

class FixedWindowScheduler : public Scheduler {
  public:
   ThreadContext* Yield() override;

   void Start() override;

   void Finish() override;

   bool IsEnd() override;

   void Initialize() override;

   SchedulerType GetType() override;
};

}
}

#endif //TSAN_FIXED_WINDOW_SCHEDULER_H
