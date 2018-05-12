#ifndef TSAN_SHARED_MEMORY_H
#define TSAN_SHARED_MEMORY_H

namespace __tsan {
namespace __relacy {

void *CreateSharedMemory(unsigned int size, int fd = 0);

void *CreateSharedMemory(unsigned int size, int fd, int visibility);

void FreeSharedMemory(void *value, unsigned int size);

int SharedMemoryOpen(const char* name);

void SharedMemoryClose(int fd, const char* name);

int Truncate(int fd, unsigned int size);

}
}


#endif //TSAN_SHARED_MEMORY_H
