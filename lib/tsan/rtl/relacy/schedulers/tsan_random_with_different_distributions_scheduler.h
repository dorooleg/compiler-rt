#ifndef TSAN_RANDOM_SCHEDULER_WITH_DIFFERENT_DISTRIBUTIONS_H
#define TSAN_RANDOM_SCHEDULER_WITH_DIFFERENT_DISTRIBUTIONS_H

#include "rtl/relacy/tsan_scheduler.h"
#include "rtl/relacy/tsan_threads_box.h"
#include "rtl/relacy/schedulers/tsan_random_generator.h"

namespace __tsan {
namespace __relacy {

class RandomWithDifferentDistributionsScheduler : public Scheduler {
  public:
   RandomWithDifferentDistributionsScheduler(ThreadsBox& threads_box);

   ThreadContext* Yield() override;

   void Start() override;

   void Finish() override;

   bool IsEnd() override;

   void Initialize() override;

   SchedulerType GetType() override;

  private:
   ThreadsBox& threads_box_;
   RandomGenerator generator_;
};

}
}

#endif //TSAN_RANDOM_SCHEDULER_WITH_DIFFERENT_DISTRIBUTIONS_H
