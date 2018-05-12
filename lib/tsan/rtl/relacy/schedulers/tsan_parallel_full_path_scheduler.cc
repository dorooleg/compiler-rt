#include "tsan_parallel_full_path_scheduler.h"
#include "tsan_scheduler_type.h"

namespace __tsan {
namespace __relacy {

ThreadContext* ParallelFullPathScheduler::Yield() {
    return nullptr;
}

void ParallelFullPathScheduler::Start() {

}

void ParallelFullPathScheduler::Finish() {

}

bool ParallelFullPathScheduler::IsEnd() {
    return false;
}

void ParallelFullPathScheduler::Initialize() {

}

SchedulerType ParallelFullPathScheduler::GetType() {
    return SchedulerType::PARALLEL_FULL_PATH;
}

}
}