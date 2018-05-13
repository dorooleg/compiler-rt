#ifndef TSAN_FIBER_TLS_COPY_PLATFORM_H
#define TSAN_FIBER_TLS_COPY_PLATFORM_H

#include "rtl/relacy/tsan_platform.h"
#include "rtl/relacy/tsan_threads_box.h"

namespace __tsan {
namespace __relacy {

class FiberTlsCopyPlatform : public Platform {
  public:
   explicit FiberTlsCopyPlatform(ThreadsBox& threads_box);

   ThreadContext* Create(void *th, void *attr, void (*callback)(), void *param) override;

   void Initialize() override;

   PlatformType GetType() override;

   void Yield(ThreadContext *context) override;

   void Start() override;

  private:
   static constexpr uptr FIBER_STACK_SIZE = 64 * 1024;
   char *tls_base_;
   uptr tls_size_;
   uptr tls_addr_;
   ThreadsBox& threads_box_;
};

}
}

#endif //TSAN_FIBER_TLS_COPY_PLATFORM_H
