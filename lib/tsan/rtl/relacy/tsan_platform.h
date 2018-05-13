#ifndef TSAN_RELACY_PLATFORM_H
#define TSAN_RELACY_PLATFORM_H

#include "tsan_thread_context.h"
#include "platforms/tsan_platform_type.h"

namespace __tsan {
namespace __relacy {

class Platform {
  public:
   virtual ThreadContext* Create(void *th, void *attr, void (*callback)(), void *param) = 0;

   virtual void Initialize() = 0;

   virtual PlatformType GetType() = 0;

   virtual void Yield(ThreadContext *context) = 0;

   virtual void Start() = 0;

   virtual ~Platform() = default;
};

}
}
#endif //TSAN_RELACY_PLATFORM_H
