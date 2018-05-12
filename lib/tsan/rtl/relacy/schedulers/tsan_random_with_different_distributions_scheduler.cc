#include "tsan_random_with_different_distributions_scheduler.h"
#include "tsan_scheduler_type.h"

namespace __tsan {
namespace __relacy {

RandomWithDifferentDistributionsScheduler::RandomWithDifferentDistributionsScheduler(ThreadsBox& thread_box)
        : threads_box_(thread_box) {

}

ThreadContext* RandomWithDifferentDistributionsScheduler::Yield() {
    if (threads_box_.GetCountRunning() == 0) {
        Printf("FATAL: ThreadSanitizer running threads is not exists\n");
        Die();
    }

    return threads_box_.GetRunningByIndex(static_cast<uptr>(generator_.Rand(static_cast<int>(threads_box_.GetCountRunning()))));
}

void RandomWithDifferentDistributionsScheduler::Start() {

}

void RandomWithDifferentDistributionsScheduler::Finish() {
    generator_.NextGenerator();
}

bool RandomWithDifferentDistributionsScheduler::IsEnd() {
    return false;
}

void RandomWithDifferentDistributionsScheduler::Initialize() {

}

SchedulerType RandomWithDifferentDistributionsScheduler::GetType() {
    return SchedulerType::RANDOM_WITH_DIFFERENT_DISTRIBUTIONS;
}

}
}