#include "tsan_pthread_platform.h"

namespace __tsan {
namespace __relacy {

ThreadContext *PthreadPlatform::Create(void *th, void *attr, void (*callback)(), void *param) {
    return nullptr;
}

void PthreadPlatform::Initialize() {

}

PlatformType PthreadPlatform::GetType() {
    return PlatformType::PTHREAD;
}

void PthreadPlatform::Yield(ThreadContext *context) {

}

}
}