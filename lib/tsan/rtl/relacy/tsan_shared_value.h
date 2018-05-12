#ifndef TSAN_SHARED_VALUE_H
#define TSAN_SHARED_VALUE_H

#include "tsan_shared_memory.h"

namespace __tsan {
namespace __relacy {

template<typename T>
class SharedValue {
  public:
   SharedValue()
           : value_(CreateSharedMemory(sizeof(T))) {
       new (value_) T{};
   }

   explicit SharedValue(const T& value)
           : value_(CreateSharedMemory(sizeof(T))) {
       new (value_) T{};
       *static_cast<T*>(value_) = value;
   }

   SharedValue(const SharedValue& other) : value_(CreateSharedMemory(sizeof(T))) {
       new (value_) T{};
       *static_cast<T*>(value_) = *static_cast<T*>(other.value_);
   }

   SharedValue(SharedValue&& other) {
       value_ = other.value_;
       other.value_ = nullptr;
   }

   SharedValue& operator=(SharedValue other) {
       void* tmp = value_;
       value_ = other.value_;
       other.value_ = tmp;
       return *this;
   }

   SharedValue& operator=(SharedValue&& other) {
       ~SharedValue();
       value_ = other.value_;
   }

   SharedValue& operator=(const T& value) {
       *static_cast<T*>(value_) = value;
       return *this;
   }

   operator T&() {
       return *static_cast<T*>(value_);
   }

   operator const T&() const {
       return *static_cast<T*>(value_);
   }

   T& operator ++() {
       ++static_cast<T&>(*this);
       return *this;
   }

   T operator ++(int) {
       SharedValue old(*this);
       ++static_cast<T&>(*this);
       return old;
   }

   T& operator --() {
       ++static_cast<T&>(*this);
       return *this;
   }

   T operator --(int) {
       SharedValue old(value_);
       ++static_cast<T&>(*this);
       return old;
   }

   ~SharedValue() {
       //FreeSharedMemory(value_, sizeof(T));
       *this = T {};
   }

  private:
   void *value_;
};

}
}

#endif //TSAN_SHARED_VALUE_H
