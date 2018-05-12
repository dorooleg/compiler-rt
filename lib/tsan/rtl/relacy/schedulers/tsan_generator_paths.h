#ifndef TSAN_GENERATOR_PATHS_H
#define TSAN_GENERATOR_PATHS_H

#include "sanitizer_common/sanitizer_libc.h"
#include "rtl/relacy/tsan_shared_vector.h"

namespace __tsan {
namespace __relacy {

class GeneratorPaths {
  public:
   GeneratorPaths();

   void Start();

   int Yield(int max_tid);

   void InvalidateThread();

   bool IsEnd();

   void Finish();

  private:
   void Next();

  private:
   SharedValue<int> invalidate_pos_;
   SharedValue<uptr> depth_;
   SharedValue<bool> is_end_;
   SharedVector<int> paths_;
   SharedVector<int> border_;
};

}
}

#endif //TSAN_GENERATOR_PATHS_H
