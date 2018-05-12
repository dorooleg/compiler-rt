#include "tsan_fiber_tls_swap_platfrom.h"
#include <ucontext.h>
#include <rtl/tsan_rtl.h>
#include <interception/interception.h>
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __tsan {
namespace __relacy {

static unsigned long get_tls_addr() {
    unsigned long addr;
    asm("mov %%fs:0, %0" : "=r"(addr));
    return addr;
}

static void set_tls_addr(unsigned long addr) {
    asm("mov %0, %%fs:0" : "+r"(addr));
}

class FiberContext : public ThreadContext {
  public:
   explicit FiberContext(ucontext_t *fiber_context = nullptr, char *tls = nullptr, FiberContext *parent = nullptr, int tid = 0)
           : ThreadContext(tid), ctx_(fiber_context), tls_(tls) {
       SetParent(parent);
   }

   ucontext_t *GetFiberContext() {
       return ctx_;
   }

   void SetFiberContext(ucontext_t *fiber_context) {
       ctx_ = fiber_context;
   }

   char *GetTls() {
       return tls_;
   }

   void SetTls(char *tls) {
       tls_ = tls;
   }

  private:
   ucontext_t *ctx_;
   char *tls_;
};

FiberTlsSwapPlatform::FiberTlsSwapPlatform(ThreadsBox &threads_box)
        : threads_box_(threads_box) {
    uptr stk_addr = 0;
    uptr stk_size = 0;
    uptr tls_addr = 0;
    InitTlsSize();
    GetThreadStackAndTls(true, &stk_addr, &stk_size, &tls_addr, &tls_size_);
    tls_addr = get_tls_addr();

    FiberContext *current_thread = static_cast<FiberContext *>(InternalCalloc(1, sizeof(FiberContext)));
    new(current_thread) FiberContext{static_cast<struct ucontext_t *>(InternalCalloc(1, sizeof(ucontext_t))),
                                     reinterpret_cast<char *>(tls_addr) - tls_size_,
                                     current_thread,
                                     0};

    ucontext_t &context = *current_thread->GetFiberContext();
    context.uc_stack.ss_flags = 0;
    context.uc_link = nullptr;

    tls_base_ = static_cast<char *>(InternalCalloc(tls_size_, 1));

    internal_memcpy(tls_base_, reinterpret_cast<const char *>(tls_addr) - tls_size_, tls_size_);
    uptr offset = (uptr) cur_thread() - (tls_addr - tls_size_);
    internal_memset(tls_base_ + offset, 0, sizeof(ThreadState));

    threads_box_.AddRunning(current_thread);
    threads_box_.SetCurrentThread(current_thread);
}

static void *empty_call(void *) {
    return nullptr;
}

ThreadContext *FiberTlsSwapPlatform::Create(void *th, void *attr, void (*callback)(), void *param) {
    (void) th;
    (void) attr;

    FiberContext *fiber_context = static_cast<FiberContext *>(InternalCalloc(1, sizeof(FiberContext)));
    new(fiber_context) FiberContext{static_cast<struct ucontext_t *>(InternalCalloc(1, sizeof(ucontext_t)))};

    if (getcontext(fiber_context->GetFiberContext()) == -1) {
        Printf("FATAL: ThreadSanitizer getcontext error in the moment creating fiber\n");
        Die();
    }

    ucontext_t &context = *fiber_context->GetFiberContext();
    context.uc_stack.ss_sp = InternalCalloc(FIBER_STACK_SIZE, sizeof(char));
    context.uc_stack.ss_size = FIBER_STACK_SIZE;
    context.uc_stack.ss_flags = 0;
    context.uc_link = static_cast<FiberContext*>(threads_box_.GetCurrentThread())->GetFiberContext();;
    fiber_context->SetParent(threads_box_.GetCurrentThread());
    fiber_context->SetTls(static_cast<char *>(InternalCalloc(tls_size_, 1)));
    internal_memcpy(fiber_context->GetTls(), tls_base_, tls_size_);
    makecontext(fiber_context->GetFiberContext(), callback, 1, param);
    REAL(pthread_create)(th, attr, empty_call, param);
    return fiber_context;
}

void FiberTlsSwapPlatform::Initialize() {
    uptr descr_addr = (uptr) static_cast<FiberContext *>(threads_box_.GetCurrentThread())->GetTls() + tls_size_;
    set_tls_addr(descr_addr);
}

PlatformType FiberTlsSwapPlatform::GetType() {
    return PlatformType::FIBER_TLS_SWAP;
}

void FiberTlsSwapPlatform::Yield(ThreadContext *context) {
    FiberContext *old_thread = static_cast<FiberContext *>(threads_box_.GetCurrentThread());
    FiberContext *new_thread = static_cast<FiberContext *>(context);
    threads_box_.SetCurrentThread(context);

    int res = swapcontext(old_thread->GetFiberContext(), new_thread->GetFiberContext());

    Initialize();

    if (res != 0) {
        Printf("FATAL: ThreadSanitizer swapcontext error in the moment yield fiber\n");
        Die();
    }
}

}
}