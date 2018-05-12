#include <cstdlib>
#include "tsan_all_states_scheduler.h"
#include "tsan_scheduler_type.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __tsan {
namespace __relacy {

static int bsr(unsigned long number) {
    if (number == 0) {
        return -1;
    }
    long position = 0;
    asm ("bsrq %1, %0" : "=r" (position) : "r" (number));
    return static_cast<int>(position);
}

AllStatesScheduler::AllStatesScheduler(ThreadsBox& threads_box)
    : variants_("variants")
    , used_("used")
    , threads_box_(threads_box)
    , depth_(0)
    , iteration_(0) {

}

ThreadContext* AllStatesScheduler::Yield() {
    if (threads_box_.GetCountRunning() == 0) {
        Printf("FATAL: ThreadSanitizer running threads is not exists\n");
        Die();
    }

    if (variants_.Size() == depth_) {
        variants_.PushBack(0);
        used_.PushBack(0);
    }

    variants_[depth_] |= threads_box_.GetRunningBitSet();
    int tid = bsr(~used_[depth_] & variants_[depth_]);
    used_[depth_] |= 1UL << (unsigned long)tid;
    ++depth_;
    if (threads_box_.ContainsRunningByTid(tid)) {
        return threads_box_.GetRunningByTid(tid);
    }

    return threads_box_.GetRunningByIndex(rand() % threads_box_.GetCountRunning());
}

void AllStatesScheduler::Start() {
    srand(iteration_);
}

void AllStatesScheduler::Finish() {
    iteration_++;
    variants_.Revalidate();
    used_.Revalidate();
}

bool AllStatesScheduler::IsEnd() {
    if (iteration_ == 0) {
        return false;
    }

    for (uptr i = 0; i < variants_.Size(); i++) {
        if ((~used_[i] & variants_[i]) != 0) {
            return false;
        }
    }

    return true;
}


void AllStatesScheduler::Initialize() {

}

SchedulerType AllStatesScheduler::GetType() {
    return SchedulerType::ALL_STATES;
}

}
}