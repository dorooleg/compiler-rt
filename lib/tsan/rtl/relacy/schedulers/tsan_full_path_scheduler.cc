#include "tsan_full_path_scheduler.h"
#include "tsan_scheduler_type.h"

namespace __tsan {
namespace __relacy {

ThreadContext* FullPathScheduler::Yield() {
    return nullptr;
}

void FullPathScheduler::Start() {

}

void FullPathScheduler::Finish() {

}

void FullPathScheduler::Initialize() {

}

SchedulerType FullPathScheduler::GetType() {
    return SchedulerType::FULL_PATH;
}

}
}