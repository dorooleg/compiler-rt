#include "tsan_thread_context.h"

namespace __tsan {
namespace __relacy {

ThreadContext::ThreadContext(int tid)
    : tid_(tid)
    , parent_(nullptr)
{}

int ThreadContext::GetTid() const {
  return tid_;
}

void ThreadContext::SetTid(int tid) {
  tid_ = tid;
}

ThreadContext* ThreadContext::GetParent() {
  return parent_;
}

void ThreadContext::SetParent(ThreadContext *parent) {
  parent_ = parent;
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

MutexContext::MutexContext(ThreadContext* thread, void* mutex) : thread_(thread), mutex_(mutex) {
}

int MutexContext::GetTid() const {
  return thread_->GetTid();
}

ThreadContext* MutexContext::GetCurrentThread() {
  return thread_;
}

void* MutexContext::GetMutex() {
  return mutex_;
}

ConditionVariableContext::ConditionVariableContext(void* cond_var) : cond_var_(cond_var) {

}

ThreadContext* ConditionVariableContext::ExtractByTid(int tid) {
  for (uptr i = 0; i < threads_.Size(); i++) {
    if (threads_[i]->GetTid() == tid) {
      ThreadContext* context = threads_[i];
      threads_[i] = threads_[threads_.Size() - 1];
      threads_.PopBack();
      return context;
    }
  }
  return nullptr;
}

ThreadContext* ConditionVariableContext::ExtractBack() {
  ThreadContext* context = threads_[threads_.Size() - 1];
  threads_.PopBack();
  return context;
}

ThreadContext* ConditionVariableContext::GetByTid(int tid) {
    for (uptr i = 0; i < threads_.Size(); i++) {
      if (threads_[i]->GetTid() == tid) {
        return threads_[i];
      }
    }
    return nullptr;
}

int ConditionVariableContext::CountThreads() const {
  return static_cast<int>(threads_.Size());
}

void ConditionVariableContext::PushBack(ThreadContext* context) {
  threads_.PushBack(context);
}

void* ConditionVariableContext::GetConditionVariable() {
  return cond_var_;
}

}
}