#ifndef TSAN_GENERATOR_PATHS_H
#define TSAN_GENERATOR_PATHS_H

#include "sanitizer_common/sanitizer_libc.h"

namespace __tsan {

class GeneratorPaths {
public:
  void Start();

  unsigned long Yield(unsigned long max_tid);

  void InvalidateThread();

  bool IsEnd();

  void Finish();

  GeneratorPaths(unsigned long *paths, unsigned long *border, unsigned long size);
private:
  void Next();

private:
  long &invalidate_pos_;
  bool is_end_;
  unsigned long size_;
  unsigned long &depth_;
  unsigned long &max_depth_;
  unsigned long *paths_;
  unsigned long *border_;
};

}

#endif //TSAN_GENERATOR_PATHS_H
