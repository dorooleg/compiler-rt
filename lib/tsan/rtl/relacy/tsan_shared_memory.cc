#include "rtl/relacy/tsan_shared_memory.h"
#include <sys/mman.h>
#include <zconf.h>
#include <fcntl.h>
#include "rtl/tsan_rtl.h"

namespace __tsan {
namespace __relacy {

void *CreateSharedMemory(unsigned int size, int fd) {
    constexpr int protection = PROT_READ | PROT_WRITE;
    constexpr int visibility = MAP_ANONYMOUS | MAP_SHARED;
    return mmap(nullptr, size, protection, visibility, fd, 0);
}

void *CreateSharedMemory(unsigned int size, int fd, int visibility) {
    constexpr int protection = PROT_READ | PROT_WRITE;
    return mmap(nullptr, size, protection, MAP_SHARED | visibility, fd, 0);
}

void FreeSharedMemory(void *value, unsigned int size) {
    munmap(value, size);
}

int SharedMemoryOpen(const char* name) {
    return shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
}

void SharedMemoryClose(int fd, const char* name) {
    close(fd);
    shm_unlink(name);
}

int Truncate(int fd, unsigned int size) {
    return ftruncate(fd, size);
}

}
}