#ifndef TSAN_PARALLEL_FULL_PATH_SCHEDULER_H
#define TSAN_PARALLEL_FULL_PATH_SCHEDULER_H


#include "rtl/relacy/tsan_scheduler.h"

namespace __tsan {
namespace __relacy {

class ParallelFullPathScheduler : public Scheduler {
  public:
   ThreadContext* Yield() override;

   void Start() override;

   void Finish() override;

   void Initialize() override;

   SchedulerType GetType() override;
};

}
}

#endif //TSAN_PARALLEL_FULL_PATH_SCHEDULER_H
