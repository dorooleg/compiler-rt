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

THREADS_INFO(Waiting, MutexContext, waiting_threads_)

THREADS_INFO(Sleping, ThreadContext*, sleping_threads_)

#undef THREADS_INFO

void ThreadsBox::WakeupJoiningByWaitTid(int wait_tid) {
  for (int i = 0; i < (int) joining_threads_.Size(); i++) {
    if (joining_threads_[i].GetWaitTid() == wait_tid) {
        ThreadContext* ctx = ExtractJoiningByIndex(i).GetCurrentThread();
        if (!ContainsWaitingByTid(ctx->GetTid())) {
            AddRunning(ctx);
        }
      i--;
    }
  }
}

unsigned long ThreadsBox::GetRunningBitSet() {
    unsigned long bit_set = 0;
    for (uptr i = 0; i < running_threads_.Size(); i++) {
        bit_set |= 1UL << (unsigned long)running_threads_[i]->GetTid();
    }
    return bit_set;
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

void ThreadsBox::AddMutex(void* mutex) {
    if (ExistsMutex(mutex))
        return;
    locked_mutexes_.PushBack(mutex);
}

void ThreadsBox::ExtractMutex(void* mutex) {
    for (uptr i = 0; i < locked_mutexes_.Size(); i++) {
        if (locked_mutexes_[i] == mutex) {
            locked_mutexes_[i] = locked_mutexes_[locked_mutexes_.Size() - 1];
            locked_mutexes_.PopBack();
            return;
        }
    }
}

bool ThreadsBox::ExistsMutex(void* mutex) {
    for (uptr i = 0; i < locked_mutexes_.Size(); i++) {
        if (locked_mutexes_[i] == mutex)
            return true;
    }
    return false;
}

ThreadContext* ThreadsBox::ExtractWaitByMutex(void* mutex) {
    for (uptr i = 0; i < waiting_threads_.Size(); i++) {
        if (waiting_threads_[i].GetMutex() == mutex) {
            ThreadContext *context = waiting_threads_[i].GetCurrentThread();
            waiting_threads_[i] = waiting_threads_[waiting_threads_.Size() - 1];
            waiting_threads_.PopBack();
            return context;
        }
    }
    return nullptr;
}

void ThreadsBox::AddConditionVariable(void *c, ThreadContext* context) {
    ConditionVariableContext* ctx = GetConditionVariable(c);
    if (ctx == nullptr) {
        condition_variables_.PushBack(ConditionVariableContext { c });
        ctx = &condition_variables_[condition_variables_.Size() - 1];
    }
    ctx->PushBack(context);
}

ThreadContext* ThreadsBox::ExtractWaitByConditionVariable(void *c) {
    ConditionVariableContext* context = GetConditionVariable(c);
    if (context == nullptr || context->CountThreads() == 0) {
        return nullptr;
    }

    ThreadContext* thread_context = context->ExtractBack();
    return thread_context;
}

bool ThreadsBox::ExistsConditionVariable(void *c) {
    return GetConditionVariable(c) != nullptr;
}

ThreadContext* ThreadsBox::GetConditionVariableThreadByTid(int tid) {
    for (uptr i = 0; i < condition_variables_.Size(); i++) {
        if (ThreadContext* context = condition_variables_[i].GetByTid(tid)) {
            return context;
        }
    }
    return nullptr;
}

ConditionVariableContext* ThreadsBox::GetConditionVariable(void *c) {
    for (uptr i = 0; i < condition_variables_.Size(); i++) {
        if (condition_variables_[i].GetConditionVariable() == c && condition_variables_[i].CountThreads() != 0) {
            return &condition_variables_[i];
        }
    }
    return nullptr;
}

void ThreadsBox::PrintDebugInfo() {
    Printf("Current thread: %d\n", current_thread_->GetTid());

    Printf("Running threads [%d]: ", running_threads_.Size());
    for (uptr i = 0; i < running_threads_.Size(); i++) {
        Printf("%d ", running_threads_[i]->GetTid());
    }
    Printf("\n");

    Printf("Joining threads [%d]: ", joining_threads_.Size());
    for (uptr i = 0; i < joining_threads_.Size(); i++) {
        Printf("(%d, %d) ", joining_threads_[i].GetTid(), joining_threads_[i].GetWaitTid());
    }
    Printf("\n");

    Printf("Waiting threads [%d]: ", waiting_threads_.Size());
    for (uptr i = 0; i < waiting_threads_.Size(); i++) {
        Printf("(%d, %p) ", waiting_threads_[i].GetTid(), waiting_threads_[i].GetMutex());
    }
    Printf("\n");
}

}
}