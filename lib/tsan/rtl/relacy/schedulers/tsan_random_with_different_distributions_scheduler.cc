#include "tsan_random_with_different_distributions_scheduler.h"
#include "tsan_scheduler_type.h"

namespace __tsan {
namespace __relacy {

ThreadContext* RandomWithDifferentDistributionsScheduler::Yield() {
    return nullptr;
}

void RandomWithDifferentDistributionsScheduler::Start() {

}

void RandomWithDifferentDistributionsScheduler::Finish() {

}

void RandomWithDifferentDistributionsScheduler::Initialize() {

}

SchedulerType RandomWithDifferentDistributionsScheduler::GetType() {
    return SchedulerType::RANDOM_WITH_DIFFERENT_DISTRIBUTIONS;
}

}
}