#include "tsan_generator_paths.h"
#include "rtl/tsan_rtl.h"

namespace __tsan {

void GeneratorPaths::Start() {
  invalidate_pos_ = -1;
  depth_ = 0;
}

unsigned long GeneratorPaths::Yield(unsigned long max_tid) {
  border_[depth_] = max(border_[depth_], max_tid);
  return paths_[(depth_)++];
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
  if (depth_ > max_depth_) {
    max_depth_ = depth_;
    internal_memset(paths_, 0, max_depth_ * sizeof(unsigned long));
    long double all_p = 1;
    long double all = 0;
    long double current = 0;
    for (int i = max_depth_; i >= 0; i--) {
      all += border_[i] * all_p;
      current += paths_[i] * all_p;
      all_p *= (border_[i] + (long double)1);
    }
    Printf("%ull.%08ull \n", (unsigned long)((current / all) * 1e2), (unsigned long)( ((current / all) * 1e2 - (unsigned long)((current / all) * 1e2)) * 1e8) );
    return;
  }
  long double all_p = 1;
  long double all = 0;
  long double current = 0;
  for (int i = max_depth_; i >= 0; i--) {
    if (paths_[i] > border_[i] || current > all) {
      Printf("Problem\n");
      Die();
    }
    all += border_[i] * all_p;
    current += paths_[i] * all_p;
    all_p *= (border_[i] + (long double)1);
  }
  Printf("%ull.%08ull \n", (unsigned long)((current / all) * 1e2), (unsigned long)( ((current / all) * 1e2 - (unsigned long)((current / all) * 1e2)) * 1e8) );
  Next();
}


GeneratorPaths::GeneratorPaths(unsigned long *paths, unsigned long *border, unsigned long size)
    : invalidate_pos_(reinterpret_cast<long &>(*(paths + 1))), is_end_(false), size_(size), depth_(*border), max_depth_(*paths), paths_(paths), border_(border) {
  invalidate_pos_ = -1;
  internal_memset(paths, 0, size * sizeof(unsigned long));
  internal_memset(border, 0, size * sizeof(unsigned long));
  paths_ += 2;
  border_ += 2;
  size_ -= 2;
}

void GeneratorPaths::Next() {

  unsigned long p = 1;
  for (long i = invalidate_pos_ == -1 ? max_depth_ : invalidate_pos_; p != 0 && i >= 0; i--) {
    p += paths_[i];
    paths_[i] = p % (border_[i] + 1);
    p = p / (border_[i] + 1);
  }

  if (invalidate_pos_ != -1) {
    internal_memset(paths_ + invalidate_pos_ + 1, 0, (max_depth_ - invalidate_pos_ - 1) * sizeof(unsigned long));
  }

  if (p != 0) {
    is_end_ = true;
  }
}
}