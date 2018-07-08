#include <interception/interception.h>
#include <rtl/tsan_rtl.h>
#include <zconf.h>
#include <ucontext.h>
#include "tsan_pthread_platform.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __tsan {
namespace __relacy {


PthreadPlatform::PthreadPlatform(ThreadsBox& threads_box)
        : threads_box_(threads_box) {
    PthreadContext *fiber_context = static_cast<PthreadContext *>(InternalCalloc(1, sizeof(PthreadContext)));
    new(fiber_context) PthreadContext{};
    fiber_context->SetParent(threads_box_.GetCurrentThread());
    fiber_context->SetWait(false);
    threads_box_.AddRunning(fiber_context);
    threads_box_.SetCurrentThread(fiber_context);
}

volatile int exclusion_create = 0;

ThreadContext *PthreadPlatform::Create(void *th, void *attr, void (*callback)(), void *param) {
    while (__sync_lock_test_and_set(&exclusion_create, 1)) {
        Printf("FATAL: Double threads in critical section create %d \n", threads_box_.GetCurrentThread()->GetTid());
        threads_box_.PrintDebugInfo();
        Die();
        // Do nothing. This GCC builtin instruction
        // ensures memory barrier.
    }

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

    __sync_synchronize(); // Memory barrier.
    exclusion_create = 0;
}

PlatformType PthreadPlatform::GetType() {
    return PlatformType::PTHREAD;
}

volatile int exclusion = 0;

void PthreadPlatform::Yield(ThreadContext *context) {


    while (__sync_lock_test_and_set(&exclusion, 1)) {
        //Printf("FATAL: Double threads in critical section %d \n", threads_box_.GetCurrentThread()->GetTid());
        threads_box_.PrintDebugInfo();
        //Die();
        // Do nothing. This GCC builtin instruction
        // ensures memory barrier.
    }

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
    } else {
        if (old_thread->GetTid() == new_thread->GetTid()) {
            Printf("FATAL: tids are equals\n");
            threads_box_.PrintDebugInfo();
            Die();
        }
    }
    old_thread->SetWait(true);
    new_thread->SetWait(false);

    __sync_synchronize(); // Memory barrier.
    exclusion = 0;

    while(old_thread->GetWait()) {
        internal_sched_yield();
    }
}



void PthreadPlatform::Start() {
}

}
}