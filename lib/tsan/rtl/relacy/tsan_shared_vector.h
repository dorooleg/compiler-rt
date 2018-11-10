#ifndef TSAN_SHARED_VECTOR_H
#define TSAN_SHARED_VECTOR_H

#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_shared_value.h"
#include "tsan_shared_memory.h"
#include "tsan/rtl/tsan_rtl.h"

namespace __tsan {
namespace __relacy {

template<typename T>
class SharedVector {
  public:
   explicit SharedVector(const char* name)
           : begin_()
           , end_()
           , last_()
           , fd_(SharedMemoryOpen(name)) {
   }

   ~SharedVector() {
       //if (begin_)
       //    InternalFree(begin_, (end_ - begin_) * sizeof(T));
       //SharedMemoryClose(fd_, "Physical");
   }

   void Reset() {
       if (begin_)
           InternalFree(begin_, (end_ - begin_) * sizeof(T));
       begin_ = 0;
       end_ = 0;
       last_ = 0;
   }

   uptr Size() const {
       return end_ - begin_;
   }

   T &operator[](uptr i) {
       DCHECK_LT(i, end_ - begin_);
       return begin_[i];
   }

   const T &operator[](uptr i) const {
       DCHECK_LT(i, end_ - begin_);
       return begin_[i];
   }

   T *PushBack() {
       EnsureSize(Size() + 1);
       T *p = &end_[-1];
       internal_memset(p, 0, sizeof(*p));
       return p;
   }

   T *PushBack(const T& v) {
       EnsureSize(Size() + 1);
       T *p = &end_[-1];
       internal_memcpy(p, &v, sizeof(*p));
       return p;
   }

   void PopBack() {
       DCHECK_GT(end_, begin_);
       end_--;
   }

   void Resize(uptr size) {
       if (size == 0) {
           end_ = begin_;
           return;
       }
       uptr old_size = Size();
       if (size <= old_size) {
           end_ = begin_ + size;
           return;
       }
       EnsureSize(size);
       if (old_size < size) {
           for (uptr i = old_size; i < size; i++)
               internal_memset(&begin_[i], 0, sizeof(begin_[i]));
       }
   }

   void Revalidate() {
       int size = end_ - begin_;
       int capacity = last_ - begin_;
       begin_ = (T*)InternalAlloc(capacity * sizeof(T));
       end_ = begin_ + size;
       last_ = begin_ + capacity;
   }

  private:
   SharedValue<T*> begin_;
   SharedValue<T*> end_;
   SharedValue<T*> last_;
   int fd_;

   void *InternalAlloc(unsigned int size) {
       Truncate(fd_, size);
       return CreateSharedMemory(size, fd_, 0);
   }

   void InternalFree(void* value, unsigned int size) {
       FreeSharedMemory(value, size);
   }

   void EnsureSize(uptr size) {
       if (size <= Size())
           return;
       if (size <= (uptr)(last_ - begin_)) {
           end_ = begin_ + size;
           return;
       }
       uptr cap0 = last_ - begin_;
       uptr cap = cap0 * 5 / 4;  // 25% growth
       if (cap == 0)
           cap = 16;
       if (cap < size)
           cap = size;
       T *p = (T*)InternalAlloc(cap * sizeof(T));
       if (cap0) {
           internal_memcpy(p, begin_, cap0 * sizeof(T));
           InternalFree(begin_, cap0 * sizeof(T));
       }
       begin_ = p;
       end_ = begin_ + size;
       last_ = begin_ + cap;
   }

   SharedVector(const SharedVector&);
   void operator=(const SharedVector&);
};

}
}

#endif //TSAN_SHARED_VECTOR_H
