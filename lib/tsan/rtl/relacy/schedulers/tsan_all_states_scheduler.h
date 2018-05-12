#ifndef TSAN_ALL_STATES_SCHEDULER_H
#define TSAN_ALL_STATES_SCHEDULER_H

#include "rtl/relacy/tsan_scheduler.h"
#include "rtl/relacy/tsan_shared_vector.h"
#include "rtl/relacy/tsan_threads_box.h"

namespace __tsan {
namespace __relacy {

class AllStatesScheduler : public Scheduler {
  public:
   explicit AllStatesScheduler(ThreadsBox& threads_box);

   ThreadContext* Yield() override;

   void Start() override;

   void Finish() override;

   bool IsEnd() override;

   void Initialize() override;

   SchedulerType GetType() override;

  private:
   SharedVector<unsigned long> variants_;
   SharedVector<unsigned long> used_;
   ThreadsBox& threads_box_;
   uptr depth_;
   uptr iteration_;
};

}
}

#endif //TSAN_ALL_STATES_SCHEDULER_H
