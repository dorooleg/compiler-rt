#ifndef TSAN_THREAD_CONTEXT_H
#define TSAN_THREAD_CONTEXT_H

#include <sanitizer_common/sanitizer_vector.h>

namespace __tsan {
namespace __relacy {

class ThreadContext {
  public:
   explicit ThreadContext(int tid = 0);

   int GetTid() const;

   void SetTid(int tid);

   ThreadContext* GetParent();

   void SetParent(ThreadContext *parent);

  private:
   int tid_;
   ThreadContext* parent_;
};

class JoinContext {
  public:
   JoinContext(ThreadContext *current_thread, ThreadContext *wait_thread);

   int GetTid() const;

   int GetWaitTid() const;

   ThreadContext* GetCurrentThread();

   ThreadContext* GetWaitThread();

  private:
   ThreadContext *wait_thread_;
   ThreadContext *current_thread_;
};

class MutexContext {
  public:
   MutexContext(ThreadContext* thread, void* mutex);

   int GetTid() const;

   ThreadContext* GetCurrentThread();

   void* GetMutex();

  private:
   ThreadContext* thread_;
   void* mutex_;
};

class ConditionVariableContext {
  public:
   ConditionVariableContext(void* cond_var);

   ThreadContext* ExtractByTid(int tid);

   ThreadContext* ExtractBack();

   ThreadContext* GetByTid(int tid);

   int CountThreads() const;

   void PushBack(ThreadContext* context);

   void* GetConditionVariable();

  private:
   Vector<ThreadContext*> threads_;
   void* cond_var_;
};

}
}

#endif //TSAN_THREAD_CONTEXT_H
