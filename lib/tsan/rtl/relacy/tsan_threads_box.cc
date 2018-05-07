#include "tsan_threads_box.h"

namespace __tsan {
namespace __relacy {

ThreadContext *ThreadsBox::GetCurrentThread() {
  return current_thread_;
}

void ThreadsBox::SetCurrentThread(ThreadContext *context) {
  current_thread_ = context;
}

#define THREADS_INFO(Type, ReturnType, Threads) \
bool ThreadsBox::Contains##Type##ByTid(int tid) const { \
  return ContainsByTid(tid, Threads); \
} \
 \
int ThreadsBox::Max##Type##Tid() const { \
  return MaxTid(Threads); \
} \
ReturnType ThreadsBox::Extract##Type##ByTid(int tid) { \
  return ExtractByTid(tid, Threads); \
} \
 \
ReturnType ThreadsBox::Get##Type##ByTid(int tid) { \
  return GetByTid(tid, Threads); \
} \
 \
ReturnType ThreadsBox::Extract##Type##ByIndex(uptr idx) { \
  return ExtractByIndex(idx, Threads); \
} \
 \
ReturnType ThreadsBox::Get##Type##ByIndex(uptr idx) { \
  return GetByIndex(idx, Threads); \
} \
 \
void ThreadsBox::Add##Type(ReturnType context) { \
  Add(context, Threads); \
} \
 \
uptr ThreadsBox::GetCount##Type() { \
  return Threads.Size(); \
}

THREADS_INFO(Running, ThreadContext*, running_threads_)

THREADS_INFO(Joining, JoinContext, joining_threads_)

THREADS_INFO(Stopped, ThreadContext*, stopped_threads_)

THREADS_INFO(Waiting, ThreadContext*, waiting_threads_)

THREADS_INFO(Sleping, ThreadContext*, sleping_threads_)

#undef THREADS_INFO

void ThreadsBox::WakeupJoiningByWaitTid(int wait_tid) {
  for (int i = 0; i < (int) joining_threads_.Size(); i++) {
    if (joining_threads_[i].GetWaitTid() == wait_tid) {
      AddRunning(ExtractJoiningByIndex(i).GetCurrentThread());
      i--;
    }
  }
}

template<typename T>
T ThreadsBox::ExtractByIndex(uptr idx, Vector<T> &threads) {
  DCHECK_GE(threads.Size(), idx + 1);
  T context = threads[idx];
  threads[idx] = threads[threads.Size() - 1];
  threads.PopBack();
  return context;
}

template<typename T>
T ThreadsBox::GetByIndex(uptr idx, Vector<T> &threads) {
  DCHECK_GE(threads.Size(), idx + 1);
  return threads[idx];
}

template<typename T>
void ThreadsBox::Add(T context, Vector<T> &threads) {
  threads.PushBack(context);
}

}
}