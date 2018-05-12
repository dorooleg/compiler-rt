#include <cstdlib>
#include "tsan_full_path_scheduler.h"
#include "tsan_scheduler_type.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __tsan {
namespace __relacy {

FullPathScheduler::FullPathScheduler(ThreadsBox& threads_box)
        : threads_box_(threads_box)
        , iteration_(0) {

}

ThreadContext* FullPathScheduler::Yield() {
    if (threads_box_.GetCountRunning() == 0) {
        Printf("FATAL: ThreadSanitizer running threads is not exists\n");
        Die();
    }

    int tid = generator_paths_.Yield(static_cast<int>(threads_box_.GetCountRunning()));
    if (!threads_box_.ContainsRunningByTid(tid)) {
        generator_paths_.InvalidateThread();
        return threads_box_.GetRunningByIndex(rand() % threads_box_.GetCountRunning());
    }

    return threads_box_.GetRunningByTid(tid);
}

void FullPathScheduler::Start() {
    srand(iteration_);
    generator_paths_.Start();
}

void FullPathScheduler::Finish() {
    iteration_++;
    generator_paths_.Finish();
}

bool FullPathScheduler::IsEnd() {
    return generator_paths_.IsEnd();
}

void FullPathScheduler::Initialize() {
}

SchedulerType FullPathScheduler::GetType() {
    return SchedulerType::FULL_PATH;
}

}
}