#ifndef TSAN_RANDOM_SCHEDULER_H
#define TSAN_RANDOM_SCHEDULER_H

#include "rtl/relacy/tsan_scheduler.h"
#include "rtl/relacy/tsan_threads_box.h"

namespace __tsan {
namespace __relacy {

class RandomScheduler : public Scheduler {
  public:
   RandomScheduler(ThreadsBox& threads_box);

   ThreadContext* Yield() override;

   void Start() override;

   void Finish() override;

   bool IsEnd() override;

   void Initialize() override;

   SchedulerType GetType() override;

  private:
   ThreadsBox& threads_box_;
   uptr iteration_;
};

}
}

#endif //TSAN_RANDOM_SCHEDULER_H
