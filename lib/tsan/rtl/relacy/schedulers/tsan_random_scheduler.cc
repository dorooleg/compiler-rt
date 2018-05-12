#include <cstdlib>
#include "tsan_random_scheduler.h"
#include "tsan_scheduler_type.h"

namespace __tsan {
namespace __relacy {

RandomScheduler::RandomScheduler(ThreadsBox& threads_box)
        : threads_box_(threads_box)
        , iteration_(0) {

}

ThreadContext* RandomScheduler::Yield() {
    if (threads_box_.GetCountRunning() == 0) {
        Printf("FATAL: ThreadSanitizer running threads is not exists\n");
        Die();
    }

    return threads_box_.GetRunningByIndex(rand() % threads_box_.GetCountRunning());
}

void RandomScheduler::Start() {
    srand(iteration_);
}

void RandomScheduler::Finish() {
    iteration_++;
}

bool RandomScheduler::IsEnd() {
    return false;
}

void RandomScheduler::Initialize() {

}

SchedulerType RandomScheduler::GetType() {
    return SchedulerType::RANDOM;
}

}
}