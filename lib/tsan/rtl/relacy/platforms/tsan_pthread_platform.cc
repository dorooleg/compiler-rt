#include <interception/interception.h>
#include <rtl/tsan_rtl.h>
#include <zconf.h>
#include <ucontext.h>
#include "tsan_pthread_platform.h"
#include "sanitizer_common/sanitizer_placement_new.h"

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

PthreadPlatform::PthreadPlatform(ThreadsBox& threads_box)
        : threads_box_(threads_box) {
    PthreadContext *fiber_context = static_cast<PthreadContext *>(InternalCalloc(1, sizeof(PthreadContext)));
    new(fiber_context) PthreadContext{};
    fiber_context->SetParent(threads_box_.GetCurrentThread());
    fiber_context->SetWait(false);
    threads_box_.AddRunning(fiber_context);
    threads_box_.SetCurrentThread(fiber_context);
}

ThreadContext *PthreadPlatform::Create(void *th, void *attr, void (*callback)(), void *param) {
    PthreadContext *fiber_context = static_cast<PthreadContext *>(InternalCalloc(1, sizeof(PthreadContext)));
    new(fiber_context) PthreadContext{};
    fiber_context->SetParent(threads_box_.GetCurrentThread());
    last_created_ = fiber_context;
    REAL(pthread_create)(th, attr, reinterpret_cast<void *(*)(void *)>(callback), param);
    return fiber_context;
}

void PthreadPlatform::Initialize() {
    PthreadContext* thread = static_cast<PthreadContext*>(last_created_);
    while(thread->GetWait()) {
        internal_sched_yield();
    }
}

PlatformType PthreadPlatform::GetType() {
    return PlatformType::PTHREAD;
}

void PthreadPlatform::Yield(ThreadContext *context) {
    if (context == nullptr) {
        Printf("FATAL: ThreadSanitizer context is nullptr\n");
        Die();
    }

    if (threads_box_.GetCurrentThread() == nullptr) {
        Printf("FATAL: ThreadSanitizer current thread is nullptr\n");
        Die();
    }
    PthreadContext *new_thread = static_cast<PthreadContext *>(context);
    PthreadContext *old_thread = static_cast<PthreadContext *>(threads_box_.GetCurrentThread());
    threads_box_.SetCurrentThread(context);
    if (!threads_box_.ContainsStoppedByTid(old_thread->GetTid())) {
        old_thread->SetWait(true);
    }
    new_thread->SetWait(false);
    while(old_thread->GetWait()) {
        internal_sched_yield();
    }
}



void PthreadPlatform::Start() {
}

}
}