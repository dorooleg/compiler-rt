#ifndef TSAN_PLATFORM_TYPE_H
#define TSAN_PLATFORM_TYPE_H


namespace __tsan {
namespace __relacy {

enum class PlatformType {
   OS,
   FIBER_TLS_COPY,
   FIBER_TLS_SWAP,
   PTHREAD
};

}
}

#endif //TSAN_PLATFORM_TYPE_H
