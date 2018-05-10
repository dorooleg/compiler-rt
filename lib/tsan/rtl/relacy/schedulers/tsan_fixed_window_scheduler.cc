#include "tsan_fixed_window_scheduler.h"
#include "tsan_scheduler_type.h"

namespace __tsan {
namespace __relacy {

ThreadContext *FixedWindowScheduler::Yield() {
    return nullptr;
}

void FixedWindowScheduler::Start() {

}

void FixedWindowScheduler::Finish() {

}

void FixedWindowScheduler::Initialize() {

}

SchedulerType FixedWindowScheduler::GetType() {
    return SchedulerType::FIXED_WINDOW;
}

}
}