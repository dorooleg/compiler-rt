#ifndef TSAN_FIXED_WINDOW_SCHEDULER_H
#define TSAN_FIXED_WINDOW_SCHEDULER_H

#include "rtl/relacy/tsan_scheduler.h"
#include "rtl/relacy/tsan_shared_vector.h"
#include "rtl/relacy/tsan_threads_box.h"

namespace __tsan {
namespace __relacy {

class FixedWindowScheduler : public Scheduler {
  public:
   FixedWindowScheduler(ThreadsBox& threads_box, int window_size);

   ThreadContext* Yield() override;

   void Start() override;

   void Finish() override;

   bool IsEnd() override;

   void Initialize() override;

   SchedulerType GetType() override;
  private:
   ThreadsBox& threads_box_;
   SharedVector<int> window_paths_;
   SharedVector<int> window_border_;
   uptr offset_;
   SharedValue<uptr> depth_;
   SharedValue<int> invalidate_pos_;
   int window_size_;
   bool is_end_;
   uptr iteration_;
};

}
}

#endif //TSAN_FIXED_WINDOW_SCHEDULER_H
