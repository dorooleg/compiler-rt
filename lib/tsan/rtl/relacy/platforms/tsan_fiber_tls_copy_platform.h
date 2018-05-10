#ifndef TSAN_FIBER_TLS_COPY_PLATFORM_H
#define TSAN_FIBER_TLS_COPY_PLATFORM_H

#include "rtl/relacy/tsan_platform.h"

namespace __tsan {
namespace __relacy {

class FiberTlsCopyPlatform : public Platform {
  public:
   ThreadContext* Create(void *th, void *attr, void (*callback)(), void *param) override;

   virtual void Initialize() override;

   PlatformType GetType() override;

   void Yield(ThreadContext *context) override;
};

}
}

#endif //TSAN_FIBER_TLS_COPY_PLATFORM_H
