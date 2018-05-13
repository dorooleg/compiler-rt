#include <cstdlib>
#include "tsan_fixed_window_scheduler.h"
#include "tsan_scheduler_type.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __tsan {
namespace __relacy {

FixedWindowScheduler::FixedWindowScheduler(ThreadsBox& threads_box, int window_size)
        : threads_box_(threads_box)
        , window_paths_("window_paths")
        , window_border_("window_border")
        , window_size_(window_size)
        , is_end_(false)
        , iteration_(0) {
    invalidate_pos_ = -1;
}

ThreadContext *FixedWindowScheduler::Yield() {
    if (offset_ <= depth_ && depth_ < offset_ + window_size_) {
        if (window_border_.Size() == depth_ - offset_) {
            window_border_.PushBack(static_cast<const int &>(threads_box_.GetCountRunning()));
            window_paths_.PushBack(0);
        }

        int tid = window_paths_[depth_ - offset_];
        if (!threads_box_.ContainsRunningByTid(tid)) {
            if (invalidate_pos_ == -1) {
                invalidate_pos_ = depth_ - offset_;
            }
            depth_++;
            return threads_box_.GetRunningByIndex(rand() % threads_box_.GetCountRunning());
        }

        depth_++;
        return threads_box_.GetRunningByTid(tid);
    }
    depth_++;
    return threads_box_.GetRunningByIndex(rand() % threads_box_.GetCountRunning());
}

void FixedWindowScheduler::Start() {
    srand(iteration_);
    depth_ = 0;
    invalidate_pos_ = -1;
}

void FixedWindowScheduler::Finish() {
    window_paths_.Revalidate();
    window_border_.Revalidate();
    if (depth_ < offset_ + window_size_) {
        is_end_ = true;
    }

    int p = 1;
    if (window_paths_.Size() > 0) {
        for (int i = static_cast<int>(invalidate_pos_ == -1 ? window_paths_.Size() - 1 : min(window_paths_.Size() - 1,
                                                                                      (uptr) invalidate_pos_));
             p != 0 && i >= 0; i--) {
            p += window_paths_[i];
            window_paths_[i] = p % ((unsigned int) window_border_[i] + 1);
            p = p / ((unsigned int) window_border_[i] + 1);
        }
    } else {
        p = 0;
    }

    if (invalidate_pos_ != -1) {
        for (uptr i = invalidate_pos_ + 1; i < window_paths_.Size(); i++) {
            window_paths_[i] = 0;
        }
    }

    if (p != 0) {
        window_border_.Resize(0);
        window_paths_.Resize(0);
        ++offset_;
    }

    ++iteration_;
}

bool FixedWindowScheduler::IsEnd() {
    return is_end_;
}

void FixedWindowScheduler::Initialize() {

}

SchedulerType FixedWindowScheduler::GetType() {
    return SchedulerType::FIXED_WINDOW;
}

}
}