#ifndef TSAN_ALL_STATES_SCHEDULER_H
#define TSAN_ALL_STATES_SCHEDULER_H

#include "rtl/relacy/tsan_scheduler.h"

namespace __tsan {
namespace __relacy {

class AllStatesScheduler : public Scheduler {
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

#endif //TSAN_ALL_STATES_SCHEDULER_H
