#ifndef TSAN_PTHREAD_PLATFORM_H
#define TSAN_PTHREAD_PLATFORM_H

#include "rtl/relacy/tsan_platform.h"
#include "rtl/relacy/tsan_threads_box.h"

namespace __tsan {
namespace __relacy {


class PthreadContext : public ThreadContext {
  public:
   PthreadContext() : m_wait(true) {

   }

   void SetWait(bool wait) {
       m_wait = wait;
   }

   bool GetWait() {
       return m_wait;
   }

  private:
   bool m_wait{};
};

class PthreadPlatform : public Platform {
  public:
   PthreadPlatform(ThreadsBox& threads_box);

   ThreadContext* Create(void *th, void *attr, void (*callback)(), void *param) override;

   void Initialize() override;

   PlatformType GetType() override;

   void Yield(ThreadContext *context) override;

   void Start() override;

  private:
   ThreadsBox& threads_box_;
   ThreadContext* last_created_;
};

}
}

#endif //TSAN_PTHREAD_PLATFORM_H
