#ifndef TSAN_RANDOM_SCHEDULER_WITH_DIFFERENT_DISTRIBUTIONS_H
#define TSAN_RANDOM_SCHEDULER_WITH_DIFFERENT_DISTRIBUTIONS_H

#include "rtl/relacy/tsan_scheduler.h"

namespace __tsan {
namespace __relacy {

class RandomWithDifferentDistributionsScheduler : public Scheduler {
  public:
   ThreadContext* Yield() override;

   void Start() override;

   void Finish() override;

   void Initialize() override;

   SchedulerType GetType() override;
};

}
}

#endif //TSAN_RANDOM_SCHEDULER_WITH_DIFFERENT_DISTRIBUTIONS_H
