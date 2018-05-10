#include "tsan_fiber_tls_copy_platform.h"

namespace __tsan {
namespace __relacy {

ThreadContext* FiberTlsCopyPlatform::Create(void *th, void *attr, void (*callback)(), void *param) {
    return nullptr;
}

void FiberTlsCopyPlatform::Initialize() {

}

PlatformType FiberTlsCopyPlatform::GetType() {
    return PlatformType::FIBER_TLS_COPY;
}

void FiberTlsCopyPlatform::Yield(ThreadContext *context) {

}

}
}