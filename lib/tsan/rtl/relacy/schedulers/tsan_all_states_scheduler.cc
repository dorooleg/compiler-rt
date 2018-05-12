#include "tsan_all_states_scheduler.h"
#include "tsan_scheduler_type.h"

namespace __tsan {
namespace __relacy {

ThreadContext* AllStatesScheduler::Yield() {
    return nullptr;
}

void AllStatesScheduler::Start() {

}

void AllStatesScheduler::Finish() {

}

bool AllStatesScheduler::IsEnd() {
    return false;
}


void AllStatesScheduler::Initialize() {

}

SchedulerType AllStatesScheduler::GetType() {
    return SchedulerType::ALL_STATES;
}

}
}