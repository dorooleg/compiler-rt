#ifndef TSAN_THREAD_CONTEXT_H
#define TSAN_THREAD_CONTEXT_H

namespace __tsan {
namespace __relacy {

class ThreadContext {
  public:
   explicit ThreadContext(int tid = 0);

   int GetTid() const;

   void SetTid(int tid);

  private:
   int tid_;
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

}
}

#endif //TSAN_THREAD_CONTEXT_H
