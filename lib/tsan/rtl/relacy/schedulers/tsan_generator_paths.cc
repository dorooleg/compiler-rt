#include "tsan_generator_paths.h"
#include "rtl/tsan_rtl.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __tsan {
namespace __relacy {

GeneratorPaths::GeneratorPaths() : paths_("paths"), border_("border") {

}

void GeneratorPaths::Start() {
    invalidate_pos_ = -1;
    depth_ = 0;
}

int GeneratorPaths::Yield(int max_tid) {
    if (border_.Size() == depth_) {
        border_.PushBack(max_tid);
    }

    if (paths_.Size() == depth_) {
        paths_.PushBack(0);
    }

    border_[depth_] = max(border_[depth_], max_tid);
    return paths_[depth_++];
}

void GeneratorPaths::InvalidateThread() {
    if (invalidate_pos_ == -1) {
        invalidate_pos_ = depth_ - 1;
    }
}

bool GeneratorPaths::IsEnd() {
    return is_end_;
}

void GeneratorPaths::Finish() {
    paths_.Revalidate();
    border_.Revalidate();

    paths_.Resize(depth_);
    border_.Resize(depth_);

    Next();
}

void GeneratorPaths::Next() {
    int p = 1;
    if (paths_.Size() > 0) {
        for (int i = static_cast<int>(invalidate_pos_ == -1 ? paths_.Size() - 1 : min(paths_.Size() - 1,
                                                                                      (uptr) invalidate_pos_));
             p != 0 && i >= 0; i--) {
            p += paths_[i];
            paths_[i] = p % ((unsigned int) border_[i] + 1);
            p = p / ((unsigned int) border_[i] + 1);
        }
    } else {
        p = 0;
    }

    if (invalidate_pos_ != -1) {
        for (uptr i = invalidate_pos_ + 1; i < paths_.Size(); i++) {
            paths_[i] = 0;
        }
    }

    if (p != 0) {
        is_end_ = true;
    }
}

}
}