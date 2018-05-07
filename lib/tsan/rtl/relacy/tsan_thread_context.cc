#include "tsan_thread_context.h"

namespace __tsan {
namespace __relacy {

ThreadContext::ThreadContext(int tid)
    : tid_(tid) {}

int ThreadContext::GetTid() const {
  return tid_;
}

void ThreadContext::SetTid(int tid) {
  tid_ = tid;
}

JoinContext::JoinContext(ThreadContext *current_thread, ThreadContext *wait_thread)
    : wait_thread_(wait_thread), current_thread_(current_thread) {}

int JoinContext::GetTid() const {
  return current_thread_->GetTid();
}

int JoinContext::GetWaitTid() const {
  return wait_thread_->GetTid();
}

ThreadContext* JoinContext::GetCurrentThread() {
  return current_thread_;
}

ThreadContext* JoinContext::GetWaitThread() {
  return wait_thread_;
}

}
}