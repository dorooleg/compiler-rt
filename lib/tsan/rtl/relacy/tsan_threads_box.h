#ifndef TSAN_THREADS_BOX_H
#define TSAN_THREADS_BOX_H

#include <rtl/tsan_defs.h>
#include "sanitizer_common/sanitizer_vector.h"
#include "tsan_thread_context.h"
#include "tsan_type_traits.h"

namespace __tsan {
namespace __relacy {

class ThreadsBox {
  public:
   ThreadContext *GetCurrentThread();

   void SetCurrentThread(ThreadContext *context);

#define THREADS_INFO(Type, ReturnType) \
    bool Contains##Type##ByTid(int tid) const; \
    int Max##Type##Tid() const; \
    ReturnType Extract##Type##ByTid(int tid); \
    ReturnType Get##Type##ByTid(int tid); \
    ReturnType Extract##Type##ByIndex(uptr idx); \
    ReturnType Get##Type##ByIndex(uptr idx); \
    void Add##Type(ReturnType context); \
    uptr GetCount##Type();

   THREADS_INFO(Running, ThreadContext*)

   THREADS_INFO(Joining, JoinContext)

   THREADS_INFO(Stopped, ThreadContext*)

   THREADS_INFO(Waiting, ThreadContext*)

   THREADS_INFO(Sleping, ThreadContext*)

#undef THREADS_INFO

   void WakeupJoiningByWaitTid(int wait_tid);

   unsigned long GetRunningBitSet();

  private:
   template<typename T>
   typename enable_if<!is_pointer<T>::value, int>::type MaxTid(const Vector<T> &threads) const {
     int m = 0;
     for (uptr i = 0; i < threads.Size(); i++) {
       m = max(threads[i].GetTid(), m);
     }
     return m;
   }

   template<typename T>
   typename enable_if<is_pointer<T>::value, int>::type MaxTid(const Vector<T> &threads) const {
     int m = 0;
     for (uptr i = 0; i < threads.Size(); i++) {
       m = max(threads[i]->GetTid(), m);
     }
     return m;
   }

   template<typename T>
   typename enable_if<!is_pointer<T>::value, bool>::type ContainsByTid(int tid, const Vector<T> &threads) const {
     for (uptr i = 0; i < threads.Size(); i++) {
       if (threads[i].GetTid() == tid) {
         return true;
       }
     }
     return false;
   }

   template<typename T>
   typename enable_if<is_pointer<T>::value, bool>::type ContainsByTid(int tid, const Vector<T> &threads) const {
     for (uptr i = 0; i < threads.Size(); i++) {
       if (threads[i]->GetTid() == tid) {
         return true;
       }
     }
     return false;
   }

   template<typename T>
   typename enable_if<!is_pointer<T>::value, T>::type GetByTid(int tid, Vector<T> &threads) {
     for (uptr i = 0; i < threads.Size(); i++) {
       if (threads[i].GetTid() == tid) {
         return threads[i];
       }
     }
     Printf("FATAL: ThreadSanitizer invalid tid for GetByTid\n");
     Die();
   }

   template<typename T>
   typename enable_if<!is_pointer<T>::value, T>::type ExtractByTid(int tid, Vector<T> &threads) {
     for (uptr i = 0; i < threads.Size(); i++) {
       if (threads[i].GetTid() == tid) {
         T context = threads[i];
         threads[i] = threads[threads.Size() - 1];
         threads.PopBack();
         return context;
       }
     }
     Printf("FATAL: ThreadSanitizer invalid tid for ExtractByTid\n");
     Die();
   }

   template<typename T>
   typename enable_if<is_pointer<T>::value, T>::type ExtractByTid(int tid, Vector<T> &threads) {
     for (uptr i = 0; i < threads.Size(); i++) {
       if (threads[i]->GetTid() == tid) {
         T context = threads[i];
         threads[i] = threads[threads.Size() - 1];
         threads.PopBack();
         return context;
       }
     }
     return nullptr;
   }

   template<typename T>
   typename enable_if<is_pointer<T>::value, T>::type GetByTid(int tid, Vector<T> &threads) {
     for (uptr i = 0; i < threads.Size(); i++) {
       if (threads[i]->GetTid() == tid) {
         return threads[i];
       }
     }
     return nullptr;
   }

   template<typename T>
   T ExtractByIndex(uptr idx, Vector<T> &threads);

   template<typename T>
   T GetByIndex(uptr idx, Vector<T> &threads);

   template<typename T>
   void Add(T context, Vector<T> &threads);

  private:
   ThreadContext *current_thread_;
   Vector<ThreadContext *> running_threads_;
   Vector<JoinContext> joining_threads_;
   Vector<ThreadContext *> stopped_threads_;
   Vector<ThreadContext *> waiting_threads_;
   Vector<ThreadContext *> sleping_threads_;
};

}
}

#endif // TSAN_THREADS_BOX_H
