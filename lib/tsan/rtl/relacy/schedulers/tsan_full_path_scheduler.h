#ifndef TSAN_FULL_PATH_SCHEDULER_H
#define TSAN_FULL_PATH_SCHEDULER_H

#include "rtl/relacy/tsan_scheduler.h"
#include "rtl/relacy/tsan_threads_box.h"
#include "rtl/relacy/schedulers/tsan_generator_paths.h"

namespace __tsan {
namespace __relacy {

class FullPathScheduler : public Scheduler {
  public:
   explicit FullPathScheduler(ThreadsBox& threads_box);

   ThreadContext* Yield() override;

   void Start() override;

   void Finish() override;

   bool IsEnd() override;

   void Initialize() override;

   SchedulerType GetType() override;

  private:
   ThreadsBox& threads_box_;
   GeneratorPaths generator_paths_;
   uptr iteration_;
};

}
}

#endif //TSAN_FULL_PATH_SCHEDULER_H
