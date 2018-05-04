#ifndef TSAN_FIBER_H
#define TSAN_FIBER_H

/*
 * Only Linux support
 */

#include "sanitizer_common/sanitizer_vector.h"
#include "tsan_generator_paths.h"

#define FIBER_STACK_SIZE 16384
extern THREADLOCAL char cur_thread_placeholder[];

namespace __tsan {

template <typename F, typename S>
struct Pair
{
  F first;
  S second;
};

struct FiberContext
{
  int tid;
  void* ctx;
  char* tls;
  FiberContext* parent;

  void Read(void* addr, uptr size, uptr tls_addr) {
    for (uptr i = 0; i < size; i++) {
      if (!Find(addr)) {
        Add(addr, tls_addr);
      }
    }
  }

  void Write(void* addr, uptr size, uptr tls_addr) {
    Read(addr, size, tls_addr);
    tls_writes.PushBack(Pair<void*, uptr> { addr, size });
  }

  void Add(void* addr, uptr tls_addr) {
    internal_memcpy(addr, tls + (uptr)addr - tls_addr, 1);
    tls_mapping.PushBack(addr);
  }

  bool Find(void* addr) {
    for (int i = 0; i < tls_mapping.Size(); i++) {
      if (tls_mapping[i] <= addr && addr <= tls_mapping[i]) {
        return true;
      }
    }
    return false;
  }

  void ApplyChanges(uptr tls_addr) {
    for (int i = 0; i < tls_writes.Size(); i++) {
      internal_memcpy(tls + (uptr)tls_writes[i].first - tls_addr, tls_writes[i].first, tls_writes[i].second);
    }
    tls_mapping.Resize(0);
    tls_writes.Resize(0);
  }

private:
  Vector<void*> tls_mapping;
  Vector<Pair<void*, uptr>> tls_writes;
};

struct JoinContext
{
  int waiting_tid;
  FiberContext* thread_info;
};

class FiberManager
{
public:
  FiberManager();
  FiberContext* CreateFiber(void *th, void *attr, void (*callback)(), void * param);
  void Yield(FiberContext* context);
  void AddFiberContext(int tid, FiberContext* context);
  void YieldByTid(int tid);
  void YieldByIndex(uptr index);
  void Yield();
  void Join(int wait_tid);
  FiberContext* GetParent();
  void StopThread();
  void Start();
  int MaxRunningTid();
  bool IsRunningTid(int tid);
  FiberContext* GetCurrent();
  void InitializeTLS();
private:
  static void* CreateSharedMemory(uptr size);

private:
  GeneratorPaths *paths_;
  FiberContext* current_thread_;
  Vector<FiberContext*> running_;
  Vector<JoinContext> joining_;
  Vector<FiberContext*> stoped_;
public:
  uptr tls_addr_;
  char* tls_base_;
  uptr tls_size_;
};

#if SANITIZER_RELACY_SCHEDULER
extern FiberManager _fiber_manager;
#endif

}

#endif // TSAN_FIBER_H
