//===-- tsan_go.cc --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ThreadSanitizer runtime for Go language.
//
//===----------------------------------------------------------------------===//

#include "tsan_rtl.h"
#include "tsan_symbolize.h"
#include "sanitizer_common/sanitizer_common.h"
#include <stdlib.h>

namespace __tsan {

void InitializeInterceptors() {
}

void InitializeDynamicAnnotations() {
}

bool IsExpectedReport(uptr addr, uptr size) {
  return false;
}

void *internal_alloc(MBlockType typ, uptr sz) {
  return InternalAlloc(sz);
}

void internal_free(void *p) {
  InternalFree(p);
}

// Callback into Go.
static void (*go_runtime_cb)(uptr cmd, void *ctx);

enum {
  CallbackGetProc = 0,
  CallbackSymbolizeCode = 1,
  CallbackSymbolizeData = 2,
};

struct SymbolizeCodeContext {
  uptr pc;
  char *func;
  char *file;
  uptr line;
  uptr off;
  uptr res;
};

SymbolizedStack *SymbolizeCode(uptr addr) {
  SymbolizedStack *s = SymbolizedStack::New(addr);
  SymbolizeCodeContext cbctx;
  internal_memset(&cbctx, 0, sizeof(cbctx));
  cbctx.pc = addr;
  go_runtime_cb(CallbackSymbolizeCode, &cbctx);
  if (cbctx.res) {
    AddressInfo &info = s->info;
    info.module_offset = cbctx.off;
    info.function = internal_strdup(cbctx.func ? cbctx.func : "??");
    info.file = internal_strdup(cbctx.file ? cbctx.file : "-");
    info.line = cbctx.line;
    info.column = 0;
  }
  return s;
}

struct SymbolizeDataContext {
  uptr addr;
  uptr heap;
  uptr start;
  uptr size;
  char *name;
  char *file;
  uptr line;
  uptr res;
};

ReportLocation *SymbolizeData(uptr addr) {
  SymbolizeDataContext cbctx;
  internal_memset(&cbctx, 0, sizeof(cbctx));
  cbctx.addr = addr;
  go_runtime_cb(CallbackSymbolizeData, &cbctx);
  if (!cbctx.res)
    return 0;
  if (cbctx.heap) {
    MBlock *b = ctx->metamap.GetBlock(cbctx.start);
    if (!b)
      return 0;
    ReportLocation *loc = ReportLocation::New(ReportLocationHeap);
    loc->heap_chunk_start = cbctx.start;
    loc->heap_chunk_size = b->siz;
    loc->tid = b->tid;
    loc->stack = SymbolizeStackId(b->stk);
    return loc;
  } else {
    ReportLocation *loc = ReportLocation::New(ReportLocationGlobal);
    loc->global.name = internal_strdup(cbctx.name ? cbctx.name : "??");
    loc->global.file = internal_strdup(cbctx.file ? cbctx.file : "??");
    loc->global.line = cbctx.line;
    loc->global.start = cbctx.start;
    loc->global.size = cbctx.size;
    return loc;
  }
}

static ThreadState *main_thr;
static bool inited;

static Processor* get_cur_proc() {
  if (UNLIKELY(!inited)) {
    // Running Initialize().
    // We have not yet returned the Processor to Go, so we cannot ask it back.
    // Currently, Initialize() does not use the Processor, so return nullptr.
    return nullptr;
  }
  Processor *proc;
  go_runtime_cb(CallbackGetProc, &proc);
  return proc;
}

Processor *ThreadState::proc() {
  return get_cur_proc();
}

extern "C" {

static ThreadState *AllocGoroutine() {
  ThreadState *thr = (ThreadState*)internal_alloc(MBlockThreadContex,
      sizeof(ThreadState));
  internal_memset(thr, 0, sizeof(*thr));
  return thr;
}

void __tsan_init(ThreadState **thrp, Processor **procp,
                 void (*cb)(uptr cmd, void *cb)) {
  go_runtime_cb = cb;
  ThreadState *thr = AllocGoroutine();
  main_thr = *thrp = thr;
  Initialize(thr);
  *procp = thr->proc1;
  inited = true;
}

void __tsan_fini() {
  // FIXME: Not necessary thread 0.
  ThreadState *thr = main_thr;
  int res = Finalize(thr);
  exit(res);
}

void __tsan_map_shadow(uptr addr, uptr size) {
  MapShadow(addr, size);
}

void __tsan_read(ThreadState *thr, void *addr, void *pc) {
  MemoryRead(thr, (uptr)pc, (uptr)addr, kSizeLog1);
}

void __tsan_read_pc(ThreadState *thr, void *addr, uptr callpc, uptr pc) {
  if (callpc != 0)
    FuncEntry(thr, callpc);
  MemoryRead(thr, (uptr)pc, (uptr)addr, kSizeLog1);
  if (callpc != 0)
    FuncExit(thr);
}

void __tsan_write(ThreadState *thr, void *addr, void *pc) {
  MemoryWrite(thr, (uptr)pc, (uptr)addr, kSizeLog1);
}

void __tsan_write_pc(ThreadState *thr, void *addr, uptr callpc, uptr pc) {
  if (callpc != 0)
    FuncEntry(thr, callpc);
  MemoryWrite(thr, (uptr)pc, (uptr)addr, kSizeLog1);
  if (callpc != 0)
    FuncExit(thr);
}

void __tsan_read_range(ThreadState *thr, void *addr, uptr size, uptr pc) {
  MemoryAccessRange(thr, (uptr)pc, (uptr)addr, size, false);
}

void __tsan_write_range(ThreadState *thr, void *addr, uptr size, uptr pc) {
  MemoryAccessRange(thr, (uptr)pc, (uptr)addr, size, true);
}

void __tsan_func_enter(ThreadState *thr, void *pc) {
  FuncEntry(thr, (uptr)pc);
}

void __tsan_func_exit(ThreadState *thr) {
  FuncExit(thr);
}

void __tsan_malloc(ThreadState *thr, uptr pc, uptr p, uptr sz) {
  CHECK(inited);
  if (thr && pc)
    ctx->metamap.AllocBlock(thr, pc, p, sz);
  MemoryResetRange(0, 0, (uptr)p, sz);
}

void __tsan_free(uptr p, uptr sz) {
  ctx->metamap.FreeRange(get_cur_proc(), p, sz);
}

void __tsan_go_start(ThreadState *parent, ThreadState **pthr, void *pc) {
  ThreadState *thr = AllocGoroutine();
  *pthr = thr;
  int goid = ThreadCreate(parent, (uptr)pc, 0, true);
  ThreadStart(thr, goid, 0, /*workerthread*/ false);
}

void __tsan_go_end(ThreadState *thr) {
  ThreadFinish(thr);
  internal_free(thr);
}

void __tsan_proc_create(Processor **pproc) {
  *pproc = ProcCreate();
}

void __tsan_proc_destroy(Processor *proc) {
  ProcDestroy(proc);
}

void __tsan_acquire(ThreadState *thr, void *addr) {
  Acquire(thr, 0, (uptr)addr);
}

void __tsan_release(ThreadState *thr, void *addr) {
  ReleaseStore(thr, 0, (uptr)addr);
}

void __tsan_release_merge(ThreadState *thr, void *addr) {
  Release(thr, 0, (uptr)addr);
}

void __tsan_finalizer_goroutine(ThreadState *thr) {
  AcquireGlobal(thr, 0);
}

void __tsan_mutex_before_lock(ThreadState *thr, uptr addr, uptr write) {
  if (write)
    MutexPreLock(thr, 0, addr);
  else
    MutexPreReadLock(thr, 0, addr);
}

void __tsan_mutex_after_lock(ThreadState *thr, uptr addr, uptr write) {
  if (write)
    MutexPostLock(thr, 0, addr);
  else
    MutexPostReadLock(thr, 0, addr);
}

void __tsan_mutex_before_unlock(ThreadState *thr, uptr addr, uptr write) {
  if (write)
    MutexUnlock(thr, 0, addr);
  else
    MutexReadUnlock(thr, 0, addr);
}

void __tsan_go_ignore_sync_begin(ThreadState *thr) {
  ThreadIgnoreSyncBegin(thr, 0);
}

void __tsan_go_ignore_sync_end(ThreadState *thr) {
  ThreadIgnoreSyncEnd(thr, 0);
}

void __tsan_report_count(u64 *pn) {
  Lock lock(&ctx->report_mtx);
  *pn = ctx->nreported;
}

}  // extern "C"
}  // namespace __tsan
//===-- tsan_clock.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_clock.h"
#include "tsan_rtl.h"
#include "sanitizer_common/sanitizer_placement_new.h"

// SyncClock and ThreadClock implement vector clocks for sync variables
// (mutexes, atomic variables, file descriptors, etc) and threads, respectively.
// ThreadClock contains fixed-size vector clock for maximum number of threads.
// SyncClock contains growable vector clock for currently necessary number of
// threads.
// Together they implement very simple model of operations, namely:
//
//   void ThreadClock::acquire(const SyncClock *src) {
//     for (int i = 0; i < kMaxThreads; i++)
//       clock[i] = max(clock[i], src->clock[i]);
//   }
//
//   void ThreadClock::release(SyncClock *dst) const {
//     for (int i = 0; i < kMaxThreads; i++)
//       dst->clock[i] = max(dst->clock[i], clock[i]);
//   }
//
//   void ThreadClock::ReleaseStore(SyncClock *dst) const {
//     for (int i = 0; i < kMaxThreads; i++)
//       dst->clock[i] = clock[i];
//   }
//
//   void ThreadClock::acq_rel(SyncClock *dst) {
//     acquire(dst);
//     release(dst);
//   }
//
// Conformance to this model is extensively verified in tsan_clock_test.cc.
// However, the implementation is significantly more complex. The complexity
// allows to implement important classes of use cases in O(1) instead of O(N).
//
// The use cases are:
// 1. Singleton/once atomic that has a single release-store operation followed
//    by zillions of acquire-loads (the acquire-load is O(1)).
// 2. Thread-local mutex (both lock and unlock can be O(1)).
// 3. Leaf mutex (unlock is O(1)).
// 4. A mutex shared by 2 threads (both lock and unlock can be O(1)).
// 5. An atomic with a single writer (writes can be O(1)).
// The implementation dynamically adopts to workload. So if an atomic is in
// read-only phase, these reads will be O(1); if it later switches to read/write
// phase, the implementation will correctly handle that by switching to O(N).
//
// Thread-safety note: all const operations on SyncClock's are conducted under
// a shared lock; all non-const operations on SyncClock's are conducted under
// an exclusive lock; ThreadClock's are private to respective threads and so
// do not need any protection.
//
// Description of SyncClock state:
// clk_ - variable size vector clock, low kClkBits hold timestamp,
//   the remaining bits hold "acquired" flag (the actual value is thread's
//   reused counter);
//   if acquried == thr->reused_, then the respective thread has already
//   acquired this clock (except possibly for dirty elements).
// dirty_ - holds up to two indeces in the vector clock that other threads
//   need to acquire regardless of "acquired" flag value;
// release_store_tid_ - denotes that the clock state is a result of
//   release-store operation by the thread with release_store_tid_ index.
// release_store_reused_ - reuse count of release_store_tid_.

// We don't have ThreadState in these methods, so this is an ugly hack that
// works only in C++.
#if !SANITIZER_GO
# define CPP_STAT_INC(typ) StatInc(cur_thread(), typ)
#else
# define CPP_STAT_INC(typ) (void)0
#endif

namespace __tsan {

static atomic_uint32_t *ref_ptr(ClockBlock *cb) {
  return reinterpret_cast<atomic_uint32_t *>(&cb->table[ClockBlock::kRefIdx]);
}

// Drop reference to the first level block idx.
static void UnrefClockBlock(ClockCache *c, u32 idx, uptr blocks) {
  ClockBlock *cb = ctx->clock_alloc.Map(idx);
  atomic_uint32_t *ref = ref_ptr(cb);
  u32 v = atomic_load(ref, memory_order_acquire);
  for (;;) {
    CHECK_GT(v, 0);
    if (v == 1)
      break;
    if (atomic_compare_exchange_strong(ref, &v, v - 1, memory_order_acq_rel))
      return;
  }
  // First level block owns second level blocks, so them as well.
  for (uptr i = 0; i < blocks; i++)
    ctx->clock_alloc.Free(c, cb->table[ClockBlock::kBlockIdx - i]);
  ctx->clock_alloc.Free(c, idx);
}

ThreadClock::ThreadClock(unsigned tid, unsigned reused)
    : tid_(tid)
    , reused_(reused + 1)  // 0 has special meaning
    , cached_idx_()
    , cached_size_()
    , cached_blocks_() {
  CHECK_LT(tid, kMaxTidInClock);
  CHECK_EQ(reused_, ((u64)reused_ << kClkBits) >> kClkBits);
  nclk_ = tid_ + 1;
  last_acquire_ = 0;
  internal_memset(clk_, 0, sizeof(clk_));
}

void ThreadClock::ResetCached(ClockCache *c) {
  if (cached_idx_) {
    UnrefClockBlock(c, cached_idx_, cached_blocks_);
    cached_idx_ = 0;
    cached_size_ = 0;
    cached_blocks_ = 0;
  }
}

void ThreadClock::acquire(ClockCache *c, SyncClock *src) {
  DCHECK_LE(nclk_, kMaxTid);
  DCHECK_LE(src->size_, kMaxTid);
  CPP_STAT_INC(StatClockAcquire);

  // Check if it's empty -> no need to do anything.
  const uptr nclk = src->size_;
  if (nclk == 0) {
    CPP_STAT_INC(StatClockAcquireEmpty);
    return;
  }

  bool acquired = false;
  for (unsigned i = 0; i < kDirtyTids; i++) {
    SyncClock::Dirty dirty = src->dirty_[i];
    unsigned tid = dirty.tid;
    if (tid != kInvalidTid) {
      if (clk_[tid] < dirty.epoch) {
        clk_[tid] = dirty.epoch;
        acquired = true;
      }
    }
  }

  // Check if we've already acquired src after the last release operation on src
  if (tid_ >= nclk || src->elem(tid_).reused != reused_) {
    // O(N) acquire.
    CPP_STAT_INC(StatClockAcquireFull);
    nclk_ = max(nclk_, nclk);
    u64 *dst_pos = &clk_[0];
    for (ClockElem &src_elem : *src) {
      u64 epoch = src_elem.epoch;
      if (*dst_pos < epoch) {
        *dst_pos = epoch;
        acquired = true;
      }
      dst_pos++;
    }

    // Remember that this thread has acquired this clock.
    if (nclk > tid_)
      src->elem(tid_).reused = reused_;
  }

  if (acquired) {
    CPP_STAT_INC(StatClockAcquiredSomething);
    last_acquire_ = clk_[tid_];
    ResetCached(c);
  }
}

void ThreadClock::release(ClockCache *c, SyncClock *dst) {
  DCHECK_LE(nclk_, kMaxTid);
  DCHECK_LE(dst->size_, kMaxTid);

  if (dst->size_ == 0) {
    // ReleaseStore will correctly set release_store_tid_,
    // which can be important for future operations.
    ReleaseStore(c, dst);
    return;
  }

  CPP_STAT_INC(StatClockRelease);
  // Check if we need to resize dst.
  if (dst->size_ < nclk_)
    dst->Resize(c, nclk_);

  // Check if we had not acquired anything from other threads
  // since the last release on dst. If so, we need to update
  // only dst->elem(tid_).
  if (dst->elem(tid_).epoch > last_acquire_) {
    UpdateCurrentThread(c, dst);
    if (dst->release_store_tid_ != tid_ ||
        dst->release_store_reused_ != reused_)
      dst->release_store_tid_ = kInvalidTid;
    return;
  }

  // O(N) release.
  CPP_STAT_INC(StatClockReleaseFull);
  dst->Unshare(c);
  // First, remember whether we've acquired dst.
  bool acquired = IsAlreadyAcquired(dst);
  if (acquired)
    CPP_STAT_INC(StatClockReleaseAcquired);
  // Update dst->clk_.
  dst->FlushDirty();
  uptr i = 0;
  for (ClockElem &ce : *dst) {
    ce.epoch = max(ce.epoch, clk_[i]);
    ce.reused = 0;
    i++;
  }
  // Clear 'acquired' flag in the remaining elements.
  if (nclk_ < dst->size_)
    CPP_STAT_INC(StatClockReleaseClearTail);
  for (uptr i = nclk_; i < dst->size_; i++)
    dst->elem(i).reused = 0;
  dst->release_store_tid_ = kInvalidTid;
  dst->release_store_reused_ = 0;
  // If we've acquired dst, remember this fact,
  // so that we don't need to acquire it on next acquire.
  if (acquired)
    dst->elem(tid_).reused = reused_;
}

void ThreadClock::ReleaseStore(ClockCache *c, SyncClock *dst) {
  DCHECK_LE(nclk_, kMaxTid);
  DCHECK_LE(dst->size_, kMaxTid);
  CPP_STAT_INC(StatClockStore);

  if (dst->size_ == 0 && cached_idx_ != 0) {
    // Reuse the cached clock.
    // Note: we could reuse/cache the cached clock in more cases:
    // we could update the existing clock and cache it, or replace it with the
    // currently cached clock and release the old one. And for a shared
    // existing clock, we could replace it with the currently cached;
    // or unshare, update and cache. But, for simplicity, we currnetly reuse
    // cached clock only when the target clock is empty.
    dst->tab_ = ctx->clock_alloc.Map(cached_idx_);
    dst->tab_idx_ = cached_idx_;
    dst->size_ = cached_size_;
    dst->blocks_ = cached_blocks_;
    CHECK_EQ(dst->dirty_[0].tid, kInvalidTid);
    // The cached clock is shared (immutable),
    // so this is where we store the current clock.
    dst->dirty_[0].tid = tid_;
    dst->dirty_[0].epoch = clk_[tid_];
    dst->release_store_tid_ = tid_;
    dst->release_store_reused_ = reused_;
    // Rememeber that we don't need to acquire it in future.
    dst->elem(tid_).reused = reused_;
    // Grab a reference.
    atomic_fetch_add(ref_ptr(dst->tab_), 1, memory_order_relaxed);
    return;
  }

  // Check if we need to resize dst.
  if (dst->size_ < nclk_)
    dst->Resize(c, nclk_);

  if (dst->release_store_tid_ == tid_ &&
      dst->release_store_reused_ == reused_ &&
      dst->elem(tid_).epoch > last_acquire_) {
    CPP_STAT_INC(StatClockStoreFast);
    UpdateCurrentThread(c, dst);
    return;
  }

  // O(N) release-store.
  CPP_STAT_INC(StatClockStoreFull);
  dst->Unshare(c);
  // Note: dst can be larger than this ThreadClock.
  // This is fine since clk_ beyond size is all zeros.
  uptr i = 0;
  for (ClockElem &ce : *dst) {
    ce.epoch = clk_[i];
    ce.reused = 0;
    i++;
  }
  for (uptr i = 0; i < kDirtyTids; i++)
    dst->dirty_[i].tid = kInvalidTid;
  dst->release_store_tid_ = tid_;
  dst->release_store_reused_ = reused_;
  // Rememeber that we don't need to acquire it in future.
  dst->elem(tid_).reused = reused_;

  // If the resulting clock is cachable, cache it for future release operations.
  // The clock is always cachable if we released to an empty sync object.
  if (cached_idx_ == 0 && dst->Cachable()) {
    // Grab a reference to the ClockBlock.
    atomic_uint32_t *ref = ref_ptr(dst->tab_);
    if (atomic_load(ref, memory_order_acquire) == 1)
      atomic_store_relaxed(ref, 2);
    else
      atomic_fetch_add(ref_ptr(dst->tab_), 1, memory_order_relaxed);
    cached_idx_ = dst->tab_idx_;
    cached_size_ = dst->size_;
    cached_blocks_ = dst->blocks_;
  }
}

void ThreadClock::acq_rel(ClockCache *c, SyncClock *dst) {
  CPP_STAT_INC(StatClockAcquireRelease);
  acquire(c, dst);
  ReleaseStore(c, dst);
}

// Updates only single element related to the current thread in dst->clk_.
void ThreadClock::UpdateCurrentThread(ClockCache *c, SyncClock *dst) const {
  // Update the threads time, but preserve 'acquired' flag.
  for (unsigned i = 0; i < kDirtyTids; i++) {
    SyncClock::Dirty *dirty = &dst->dirty_[i];
    const unsigned tid = dirty->tid;
    if (tid == tid_ || tid == kInvalidTid) {
      CPP_STAT_INC(StatClockReleaseFast);
      dirty->tid = tid_;
      dirty->epoch = clk_[tid_];
      return;
    }
  }
  // Reset all 'acquired' flags, O(N).
  // We are going to touch dst elements, so we need to unshare it.
  dst->Unshare(c);
  CPP_STAT_INC(StatClockReleaseSlow);
  dst->elem(tid_).epoch = clk_[tid_];
  for (uptr i = 0; i < dst->size_; i++)
    dst->elem(i).reused = 0;
  dst->FlushDirty();
}

// Checks whether the current thread has already acquired src.
bool ThreadClock::IsAlreadyAcquired(const SyncClock *src) const {
  if (src->elem(tid_).reused != reused_)
    return false;
  for (unsigned i = 0; i < kDirtyTids; i++) {
    SyncClock::Dirty dirty = src->dirty_[i];
    if (dirty.tid != kInvalidTid) {
      if (clk_[dirty.tid] < dirty.epoch)
        return false;
    }
  }
  return true;
}

// Sets a single element in the vector clock.
// This function is called only from weird places like AcquireGlobal.
void ThreadClock::set(ClockCache *c, unsigned tid, u64 v) {
  DCHECK_LT(tid, kMaxTid);
  DCHECK_GE(v, clk_[tid]);
  clk_[tid] = v;
  if (nclk_ <= tid)
    nclk_ = tid + 1;
  last_acquire_ = clk_[tid_];
  ResetCached(c);
}

void ThreadClock::DebugDump(int(*printf)(const char *s, ...)) {
  printf("clock=[");
  for (uptr i = 0; i < nclk_; i++)
    printf("%s%llu", i == 0 ? "" : ",", clk_[i]);
  printf("] tid=%u/%u last_acq=%llu", tid_, reused_, last_acquire_);
}

SyncClock::SyncClock() {
  ResetImpl();
}

SyncClock::~SyncClock() {
  // Reset must be called before dtor.
  CHECK_EQ(size_, 0);
  CHECK_EQ(blocks_, 0);
  CHECK_EQ(tab_, 0);
  CHECK_EQ(tab_idx_, 0);
}

void SyncClock::Reset(ClockCache *c) {
  if (size_)
    UnrefClockBlock(c, tab_idx_, blocks_);
  ResetImpl();
}

void SyncClock::ResetImpl() {
  tab_ = 0;
  tab_idx_ = 0;
  size_ = 0;
  blocks_ = 0;
  release_store_tid_ = kInvalidTid;
  release_store_reused_ = 0;
  for (uptr i = 0; i < kDirtyTids; i++)
    dirty_[i].tid = kInvalidTid;
}

void SyncClock::Resize(ClockCache *c, uptr nclk) {
  CPP_STAT_INC(StatClockReleaseResize);
  Unshare(c);
  if (nclk <= capacity()) {
    // Memory is already allocated, just increase the size.
    size_ = nclk;
    return;
  }
  if (size_ == 0) {
    // Grow from 0 to one-level table.
    CHECK_EQ(size_, 0);
    CHECK_EQ(blocks_, 0);
    CHECK_EQ(tab_, 0);
    CHECK_EQ(tab_idx_, 0);
    tab_idx_ = ctx->clock_alloc.Alloc(c);
    tab_ = ctx->clock_alloc.Map(tab_idx_);
    internal_memset(tab_, 0, sizeof(*tab_));
    atomic_store_relaxed(ref_ptr(tab_), 1);
    size_ = 1;
  } else if (size_ > blocks_ * ClockBlock::kClockCount) {
    u32 idx = ctx->clock_alloc.Alloc(c);
    ClockBlock *new_cb = ctx->clock_alloc.Map(idx);
    uptr top = size_ - blocks_ * ClockBlock::kClockCount;
    CHECK_LT(top, ClockBlock::kClockCount);
    const uptr move = top * sizeof(tab_->clock[0]);
    internal_memcpy(&new_cb->clock[0], tab_->clock, move);
    internal_memset(&new_cb->clock[top], 0, sizeof(*new_cb) - move);
    internal_memset(tab_->clock, 0, move);
    append_block(idx);
  }
  // At this point we have first level table allocated and all clock elements
  // are evacuated from it to a second level block.
  // Add second level tables as necessary.
  while (nclk > capacity()) {
    u32 idx = ctx->clock_alloc.Alloc(c);
    ClockBlock *cb = ctx->clock_alloc.Map(idx);
    internal_memset(cb, 0, sizeof(*cb));
    append_block(idx);
  }
  size_ = nclk;
}

// Flushes all dirty elements into the main clock array.
void SyncClock::FlushDirty() {
  for (unsigned i = 0; i < kDirtyTids; i++) {
    Dirty *dirty = &dirty_[i];
    if (dirty->tid != kInvalidTid) {
      CHECK_LT(dirty->tid, size_);
      elem(dirty->tid).epoch = dirty->epoch;
      dirty->tid = kInvalidTid;
    }
  }
}

bool SyncClock::IsShared() const {
  if (size_ == 0)
    return false;
  atomic_uint32_t *ref = ref_ptr(tab_);
  u32 v = atomic_load(ref, memory_order_acquire);
  CHECK_GT(v, 0);
  return v > 1;
}

// Unshares the current clock if it's shared.
// Shared clocks are immutable, so they need to be unshared before any updates.
// Note: this does not apply to dirty entries as they are not shared.
void SyncClock::Unshare(ClockCache *c) {
  if (!IsShared())
    return;
  // First, copy current state into old.
  SyncClock old;
  old.tab_ = tab_;
  old.tab_idx_ = tab_idx_;
  old.size_ = size_;
  old.blocks_ = blocks_;
  old.release_store_tid_ = release_store_tid_;
  old.release_store_reused_ = release_store_reused_;
  for (unsigned i = 0; i < kDirtyTids; i++)
    old.dirty_[i] = dirty_[i];
  // Then, clear current object.
  ResetImpl();
  // Allocate brand new clock in the current object.
  Resize(c, old.size_);
  // Now copy state back into this object.
  Iter old_iter(&old);
  for (ClockElem &ce : *this) {
    ce = *old_iter;
    ++old_iter;
  }
  release_store_tid_ = old.release_store_tid_;
  release_store_reused_ = old.release_store_reused_;
  for (unsigned i = 0; i < kDirtyTids; i++)
    dirty_[i] = old.dirty_[i];
  // Drop reference to old and delete if necessary.
  old.Reset(c);
}

// Can we cache this clock for future release operations?
ALWAYS_INLINE bool SyncClock::Cachable() const {
  if (size_ == 0)
    return false;
  for (unsigned i = 0; i < kDirtyTids; i++) {
    if (dirty_[i].tid != kInvalidTid)
      return false;
  }
  return atomic_load_relaxed(ref_ptr(tab_)) == 1;
}

// elem linearizes the two-level structure into linear array.
// Note: this is used only for one time accesses, vector operations use
// the iterator as it is much faster.
ALWAYS_INLINE ClockElem &SyncClock::elem(unsigned tid) const {
  DCHECK_LT(tid, size_);
  const uptr block = tid / ClockBlock::kClockCount;
  DCHECK_LE(block, blocks_);
  tid %= ClockBlock::kClockCount;
  if (block == blocks_)
    return tab_->clock[tid];
  u32 idx = get_block(block);
  ClockBlock *cb = ctx->clock_alloc.Map(idx);
  return cb->clock[tid];
}

ALWAYS_INLINE uptr SyncClock::capacity() const {
  if (size_ == 0)
    return 0;
  uptr ratio = sizeof(ClockBlock::clock[0]) / sizeof(ClockBlock::table[0]);
  // How many clock elements we can fit into the first level block.
  // +1 for ref counter.
  uptr top = ClockBlock::kClockCount - RoundUpTo(blocks_ + 1, ratio) / ratio;
  return blocks_ * ClockBlock::kClockCount + top;
}

ALWAYS_INLINE u32 SyncClock::get_block(uptr bi) const {
  DCHECK(size_);
  DCHECK_LT(bi, blocks_);
  return tab_->table[ClockBlock::kBlockIdx - bi];
}

ALWAYS_INLINE void SyncClock::append_block(u32 idx) {
  uptr bi = blocks_++;
  CHECK_EQ(get_block(bi), 0);
  tab_->table[ClockBlock::kBlockIdx - bi] = idx;
}

// Used only by tests.
u64 SyncClock::get(unsigned tid) const {
  for (unsigned i = 0; i < kDirtyTids; i++) {
    Dirty dirty = dirty_[i];
    if (dirty.tid == tid)
      return dirty.epoch;
  }
  return elem(tid).epoch;
}

// Used only by Iter test.
u64 SyncClock::get_clean(unsigned tid) const {
  return elem(tid).epoch;
}

void SyncClock::DebugDump(int(*printf)(const char *s, ...)) {
  printf("clock=[");
  for (uptr i = 0; i < size_; i++)
    printf("%s%llu", i == 0 ? "" : ",", elem(i).epoch);
  printf("] reused=[");
  for (uptr i = 0; i < size_; i++)
    printf("%s%llu", i == 0 ? "" : ",", elem(i).reused);
  printf("] release_store_tid=%d/%d dirty_tids=%d[%llu]/%d[%llu]",
      release_store_tid_, release_store_reused_,
      dirty_[0].tid, dirty_[0].epoch,
      dirty_[1].tid, dirty_[1].epoch);
}

void SyncClock::Iter::Next() {
  // Finished with the current block, move on to the next one.
  block_++;
  if (block_ < parent_->blocks_) {
    // Iterate over the next second level block.
    u32 idx = parent_->get_block(block_);
    ClockBlock *cb = ctx->clock_alloc.Map(idx);
    pos_ = &cb->clock[0];
    end_ = pos_ + min(parent_->size_ - block_ * ClockBlock::kClockCount,
        ClockBlock::kClockCount);
    return;
  }
  if (block_ == parent_->blocks_ &&
      parent_->size_ > parent_->blocks_ * ClockBlock::kClockCount) {
    // Iterate over elements in the first level block.
    pos_ = &parent_->tab_->clock[0];
    end_ = pos_ + min(parent_->size_ - block_ * ClockBlock::kClockCount,
        ClockBlock::kClockCount);
    return;
  }
  parent_ = nullptr;  // denotes end
}
}  // namespace __tsan
//===-- tsan_external.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_rtl.h"
#include "tsan_interceptors.h"

namespace __tsan {

#define CALLERPC ((uptr)__builtin_return_address(0))

struct TagData {
  const char *object_type;
  const char *header;
};

static TagData registered_tags[kExternalTagMax] = {
  {},
  {"Swift variable", "Swift access race"},
};
static atomic_uint32_t used_tags{kExternalTagFirstUserAvailable};  // NOLINT.
static TagData *GetTagData(uptr tag) {
  // Invalid/corrupted tag?  Better return NULL and let the caller deal with it.
  if (tag >= atomic_load(&used_tags, memory_order_relaxed)) return nullptr;
  return &registered_tags[tag];
}

const char *GetObjectTypeFromTag(uptr tag) {
  TagData *tag_data = GetTagData(tag);
  return tag_data ? tag_data->object_type : nullptr;
}

const char *GetReportHeaderFromTag(uptr tag) {
  TagData *tag_data = GetTagData(tag);
  return tag_data ? tag_data->header : nullptr;
}

void InsertShadowStackFrameForTag(ThreadState *thr, uptr tag) {
  FuncEntry(thr, (uptr)&registered_tags[tag]);
}

uptr TagFromShadowStackFrame(uptr pc) {
  uptr tag_count = atomic_load(&used_tags, memory_order_relaxed);
  void *pc_ptr = (void *)pc;
  if (pc_ptr < GetTagData(0) || pc_ptr > GetTagData(tag_count - 1))
    return 0;
  return (TagData *)pc_ptr - GetTagData(0);
}

#if !SANITIZER_GO

typedef void(*AccessFunc)(ThreadState *, uptr, uptr, int);
void ExternalAccess(void *addr, void *caller_pc, void *tag, AccessFunc access) {
  CHECK_LT(tag, atomic_load(&used_tags, memory_order_relaxed));
  ThreadState *thr = cur_thread();
  if (caller_pc) FuncEntry(thr, (uptr)caller_pc);
  InsertShadowStackFrameForTag(thr, (uptr)tag);
  bool in_ignored_lib;
  if (!caller_pc || !libignore()->IsIgnored((uptr)caller_pc, &in_ignored_lib)) {
    access(thr, CALLERPC, (uptr)addr, kSizeLog1);
  }
  FuncExit(thr);
  if (caller_pc) FuncExit(thr);
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void *__tsan_external_register_tag(const char *object_type) {
  uptr new_tag = atomic_fetch_add(&used_tags, 1, memory_order_relaxed);
  CHECK_LT(new_tag, kExternalTagMax);
  GetTagData(new_tag)->object_type = internal_strdup(object_type);
  char header[127] = {0};
  internal_snprintf(header, sizeof(header), "race on %s", object_type);
  GetTagData(new_tag)->header = internal_strdup(header);
  return (void *)new_tag;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_external_register_header(void *tag, const char *header) {
  CHECK_GE((uptr)tag, kExternalTagFirstUserAvailable);
  CHECK_LT((uptr)tag, kExternalTagMax);
  atomic_uintptr_t *header_ptr =
      (atomic_uintptr_t *)&GetTagData((uptr)tag)->header;
  header = internal_strdup(header);
  char *old_header =
      (char *)atomic_exchange(header_ptr, (uptr)header, memory_order_seq_cst);
  if (old_header) internal_free(old_header);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_external_assign_tag(void *addr, void *tag) {
  CHECK_LT(tag, atomic_load(&used_tags, memory_order_relaxed));
  Allocator *a = allocator();
  MBlock *b = nullptr;
  if (a->PointerIsMine((void *)addr)) {
    void *block_begin = a->GetBlockBegin((void *)addr);
    if (block_begin) b = ctx->metamap.GetBlock((uptr)block_begin);
  }
  if (b) {
    b->tag = (uptr)tag;
  }
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_external_read(void *addr, void *caller_pc, void *tag) {
  ExternalAccess(addr, caller_pc, tag, MemoryRead);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_external_write(void *addr, void *caller_pc, void *tag) {
  ExternalAccess(addr, caller_pc, tag, MemoryWrite);
}
}  // extern "C"

#endif  // !SANITIZER_GO

}  // namespace __tsan
//===-- tsan_flags.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_flags.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "ubsan/ubsan_flags.h"

namespace __tsan {

// Can be overriden in frontend.
#ifdef TSAN_EXTERNAL_HOOKS
extern "C" const char* __tsan_default_options();
#else
SANITIZER_WEAK_DEFAULT_IMPL
const char *__tsan_default_options() {
  return "";
}
#endif

void Flags::SetDefaults() {
#define TSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "tsan_flags.inc"
#undef TSAN_FLAG
  // DDFlags
  second_deadlock_stack = false;
}

void RegisterTsanFlags(FlagParser *parser, Flags *f) {
#define TSAN_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "tsan_flags.inc"
#undef TSAN_FLAG
  // DDFlags
  RegisterFlag(parser, "second_deadlock_stack",
      "Report where each mutex is locked in deadlock reports",
      &f->second_deadlock_stack);
}

void InitializeFlags(Flags *f, const char *env) {
  SetCommonFlagsDefaults();
  {
    // Override some common flags defaults.
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.allow_addr2line = true;
    if (SANITIZER_GO) {
      // Does not work as expected for Go: runtime handles SIGABRT and crashes.
      cf.abort_on_error = false;
      // Go does not have mutexes.
    } else {
      cf.detect_deadlocks = true;
    }
    cf.print_suppressions = false;
    cf.stack_trace_format = "    #%n %f %S %M";
    cf.exitcode = 66;
    cf.intercept_tls_get_addr = true;
    OverrideCommonFlags(cf);
  }

  f->SetDefaults();

  FlagParser parser;
  RegisterTsanFlags(&parser, f);
  RegisterCommonFlags(&parser);
#if TSAN_CONTAINS_UBSAN
  __ubsan::Flags *uf = __ubsan::flags();
  uf->SetDefaults();

  FlagParser ubsan_parser;
  __ubsan::RegisterUbsanFlags(&ubsan_parser, uf);
  RegisterCommonFlags(&ubsan_parser);
#endif

  // Let a frontend override.
  parser.ParseString(__tsan_default_options());
#if TSAN_CONTAINS_UBSAN
  const char *ubsan_default_options = __ubsan::MaybeCallUbsanDefaultOptions();
  ubsan_parser.ParseString(ubsan_default_options);
#endif
  // Override from command line.
  parser.ParseString(env);
#if TSAN_CONTAINS_UBSAN
  ubsan_parser.ParseString(GetEnv("UBSAN_OPTIONS"));
#endif

  // Sanity check.
  if (!f->report_bugs) {
    f->report_thread_leaks = false;
    f->report_destroy_locked = false;
    f->report_signal_unsafe = false;
  }

  InitializeCommonFlags();

  if (Verbosity()) ReportUnrecognizedFlags();

  if (common_flags()->help) parser.PrintFlagDescriptions();

  if (f->history_size < 0 || f->history_size > 7) {
    Printf("ThreadSanitizer: incorrect value for history_size"
           " (must be [0..7])\n");
    Die();
  }

  if (f->io_sync < 0 || f->io_sync > 2) {
    Printf("ThreadSanitizer: incorrect value for io_sync"
           " (must be [0..2])\n");
    Die();
  }
}

}  // namespace __tsan
//===-- tsan_interface_atomic.cc ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

// ThreadSanitizer atomic operations are based on C++11/C1x standards.
// For background see C++11 standard.  A slightly older, publicly
// available draft of the standard (not entirely up-to-date, but close enough
// for casual browsing) is available here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3242.pdf
// The following page contains more background information:
// http://www.hpl.hp.com/personal/Hans_Boehm/c++mm/

#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "tsan_flags.h"
#include "tsan_interface.h"
#include "tsan_rtl.h"
#include "rtl/relacy/tsan_scheduler_engine.h"

using namespace __tsan;  // NOLINT

#if !SANITIZER_GO && __TSAN_HAS_INT128
// Protects emulation of 128-bit atomic operations.
static StaticSpinMutex mutex128;
#endif

static bool IsLoadOrder(morder mo) {
  return mo == mo_relaxed || mo == mo_consume
      || mo == mo_acquire || mo == mo_seq_cst;
}

static bool IsStoreOrder(morder mo) {
  return mo == mo_relaxed || mo == mo_release || mo == mo_seq_cst;
}

static bool IsReleaseOrder(morder mo) {
  return mo == mo_release || mo == mo_acq_rel || mo == mo_seq_cst;
}

static bool IsAcquireOrder(morder mo) {
  return mo == mo_consume || mo == mo_acquire
      || mo == mo_acq_rel || mo == mo_seq_cst;
}

static bool IsAcqRelOrder(morder mo) {
  return mo == mo_acq_rel || mo == mo_seq_cst;
}

template<typename T> T func_xchg(volatile T *v, T op) {
  T res = __sync_lock_test_and_set(v, op);
  // __sync_lock_test_and_set does not contain full barrier.
  __sync_synchronize();
  return res;
}

template<typename T> T func_add(volatile T *v, T op) {
  return __sync_fetch_and_add(v, op);
}

template<typename T> T func_sub(volatile T *v, T op) {
  return __sync_fetch_and_sub(v, op);
}

template<typename T> T func_and(volatile T *v, T op) {
  return __sync_fetch_and_and(v, op);
}

template<typename T> T func_or(volatile T *v, T op) {
  return __sync_fetch_and_or(v, op);
}

template<typename T> T func_xor(volatile T *v, T op) {
  return __sync_fetch_and_xor(v, op);
}

template<typename T> T func_nand(volatile T *v, T op) {
  // clang does not support __sync_fetch_and_nand.
  T cmp = *v;
  for (;;) {
    T newv = ~(cmp & op);
    T cur = __sync_val_compare_and_swap(v, cmp, newv);
    if (cmp == cur)
      return cmp;
    cmp = cur;
  }
}

template<typename T> T func_cas(volatile T *v, T cmp, T xch) {
  return __sync_val_compare_and_swap(v, cmp, xch);
}

// clang does not support 128-bit atomic ops.
// Atomic ops are executed under tsan internal mutex,
// here we assume that the atomic variables are not accessed
// from non-instrumented code.
#if !defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16) && !SANITIZER_GO \
    && __TSAN_HAS_INT128
a128 func_xchg(volatile a128 *v, a128 op) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = op;
  return cmp;
}

a128 func_add(volatile a128 *v, a128 op) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp + op;
  return cmp;
}

a128 func_sub(volatile a128 *v, a128 op) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp - op;
  return cmp;
}

a128 func_and(volatile a128 *v, a128 op) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp & op;
  return cmp;
}

a128 func_or(volatile a128 *v, a128 op) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp | op;
  return cmp;
}

a128 func_xor(volatile a128 *v, a128 op) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp ^ op;
  return cmp;
}

a128 func_nand(volatile a128 *v, a128 op) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = ~(cmp & op);
  return cmp;
}

a128 func_cas(volatile a128 *v, a128 cmp, a128 xch) {
  SpinMutexLock lock(&mutex128);
  a128 cur = *v;
  if (cur == cmp)
    *v = xch;
  return cur;
}
#endif

template<typename T>
static int SizeLog() {
  if (sizeof(T) <= 1)
    return kSizeLog1;
  else if (sizeof(T) <= 2)
    return kSizeLog2;
  else if (sizeof(T) <= 4)
    return kSizeLog4;
  else
    return kSizeLog8;
  // For 16-byte atomics we also use 8-byte memory access,
  // this leads to false negatives only in very obscure cases.
}

#if !SANITIZER_GO
static atomic_uint8_t *to_atomic(const volatile a8 *a) {
  return reinterpret_cast<atomic_uint8_t *>(const_cast<a8 *>(a));
}

static atomic_uint16_t *to_atomic(const volatile a16 *a) {
  return reinterpret_cast<atomic_uint16_t *>(const_cast<a16 *>(a));
}
#endif

static atomic_uint32_t *to_atomic(const volatile a32 *a) {
  return reinterpret_cast<atomic_uint32_t *>(const_cast<a32 *>(a));
}

static atomic_uint64_t *to_atomic(const volatile a64 *a) {
  return reinterpret_cast<atomic_uint64_t *>(const_cast<a64 *>(a));
}

static memory_order to_mo(morder mo) {
  switch (mo) {
  case mo_relaxed: return memory_order_relaxed;
  case mo_consume: return memory_order_consume;
  case mo_acquire: return memory_order_acquire;
  case mo_release: return memory_order_release;
  case mo_acq_rel: return memory_order_acq_rel;
  case mo_seq_cst: return memory_order_seq_cst;
  }
  CHECK(0);
  return memory_order_seq_cst;
}

template<typename T>
static T NoTsanAtomicLoad(const volatile T *a, morder mo) {
  return atomic_load(to_atomic(a), to_mo(mo));
}

#if __TSAN_HAS_INT128 && !SANITIZER_GO
static a128 NoTsanAtomicLoad(const volatile a128 *a, morder mo) {
  SpinMutexLock lock(&mutex128);
  return *a;
}
#endif

template<typename T>
static T AtomicLoad(ThreadState *thr, uptr pc, const volatile T *a, morder mo) {
  CHECK(IsLoadOrder(mo));
  // This fast-path is critical for performance.
  // Assume the access is atomic.
  if (!IsAcquireOrder(mo)) {
    MemoryReadAtomic(thr, pc, (uptr)a, SizeLog<T>());
    return NoTsanAtomicLoad(a, mo);
  }
  // Don't create sync object if it does not exist yet. For example, an atomic
  // pointer is initialized to nullptr and then periodically acquire-loaded.
  T v = NoTsanAtomicLoad(a, mo);
  SyncVar *s = ctx->metamap.GetIfExistsAndLock((uptr)a, false);
  if (s) {
    AcquireImpl(thr, pc, &s->clock);
    // Re-read under sync mutex because we need a consistent snapshot
    // of the value and the clock we acquire.
    v = NoTsanAtomicLoad(a, mo);
    s->mtx.ReadUnlock();
  }
  MemoryReadAtomic(thr, pc, (uptr)a, SizeLog<T>());
  return v;
}

template<typename T>
static void NoTsanAtomicStore(volatile T *a, T v, morder mo) {
  atomic_store(to_atomic(a), v, to_mo(mo));
}

#if __TSAN_HAS_INT128 && !SANITIZER_GO
static void NoTsanAtomicStore(volatile a128 *a, a128 v, morder mo) {
  SpinMutexLock lock(&mutex128);
  *a = v;
}
#endif

template<typename T>
static void AtomicStore(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  CHECK(IsStoreOrder(mo));
  MemoryWriteAtomic(thr, pc, (uptr)a, SizeLog<T>());
  // This fast-path is critical for performance.
  // Assume the access is atomic.
  // Strictly saying even relaxed store cuts off release sequence,
  // so must reset the clock.
  if (!IsReleaseOrder(mo)) {
    NoTsanAtomicStore(a, v, mo);
    return;
  }
  __sync_synchronize();
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, (uptr)a, true);
  thr->fast_state.IncrementEpoch();
  // Can't increment epoch w/o writing to the trace as well.
  TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
  ReleaseStoreImpl(thr, pc, &s->clock);
  NoTsanAtomicStore(a, v, mo);
  s->mtx.Unlock();
}

template<typename T, T (*F)(volatile T *v, T op)>
static T AtomicRMW(ThreadState *thr, uptr pc, volatile T *a, T v, morder mo) {
  MemoryWriteAtomic(thr, pc, (uptr)a, SizeLog<T>());
  SyncVar *s = 0;
  if (mo != mo_relaxed) {
    s = ctx->metamap.GetOrCreateAndLock(thr, pc, (uptr)a, true);
    thr->fast_state.IncrementEpoch();
    // Can't increment epoch w/o writing to the trace as well.
    TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
    if (IsAcqRelOrder(mo))
      AcquireReleaseImpl(thr, pc, &s->clock);
    else if (IsReleaseOrder(mo))
      ReleaseImpl(thr, pc, &s->clock);
    else if (IsAcquireOrder(mo))
      AcquireImpl(thr, pc, &s->clock);
  }
  v = F(a, v);
  if (s)
    s->mtx.Unlock();
  return v;
}

template<typename T>
static T NoTsanAtomicExchange(volatile T *a, T v, morder mo) {
  return func_xchg(a, v);
}

template<typename T>
static T NoTsanAtomicFetchAdd(volatile T *a, T v, morder mo) {
  return func_add(a, v);
}

template<typename T>
static T NoTsanAtomicFetchSub(volatile T *a, T v, morder mo) {
  return func_sub(a, v);
}

template<typename T>
static T NoTsanAtomicFetchAnd(volatile T *a, T v, morder mo) {
  return func_and(a, v);
}

template<typename T>
static T NoTsanAtomicFetchOr(volatile T *a, T v, morder mo) {
  return func_or(a, v);
}

template<typename T>
static T NoTsanAtomicFetchXor(volatile T *a, T v, morder mo) {
  return func_xor(a, v);
}

template<typename T>
static T NoTsanAtomicFetchNand(volatile T *a, T v, morder mo) {
  return func_nand(a, v);
}

template<typename T>
static T AtomicExchange(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_xchg>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchAdd(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_add>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchSub(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_sub>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchAnd(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_and>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchOr(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_or>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchXor(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_xor>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchNand(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_nand>(thr, pc, a, v, mo);
}

template<typename T>
static bool NoTsanAtomicCAS(volatile T *a, T *c, T v, morder mo, morder fmo) {
  return atomic_compare_exchange_strong(to_atomic(a), c, v, to_mo(mo));
}

#if __TSAN_HAS_INT128
static bool NoTsanAtomicCAS(volatile a128 *a, a128 *c, a128 v,
    morder mo, morder fmo) {
  a128 old = *c;
  a128 cur = func_cas(a, old, v);
  if (cur == old)
    return true;
  *c = cur;
  return false;
}
#endif

template<typename T>
static T NoTsanAtomicCAS(volatile T *a, T c, T v, morder mo, morder fmo) {
  NoTsanAtomicCAS(a, &c, v, mo, fmo);
  return c;
}

template<typename T>
static bool AtomicCAS(ThreadState *thr, uptr pc,
    volatile T *a, T *c, T v, morder mo, morder fmo) {
  (void)fmo;  // Unused because llvm does not pass it yet.
  MemoryWriteAtomic(thr, pc, (uptr)a, SizeLog<T>());
  SyncVar *s = 0;
  bool write_lock = mo != mo_acquire && mo != mo_consume;
  if (mo != mo_relaxed) {
    s = ctx->metamap.GetOrCreateAndLock(thr, pc, (uptr)a, write_lock);
    thr->fast_state.IncrementEpoch();
    // Can't increment epoch w/o writing to the trace as well.
    TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
    if (IsAcqRelOrder(mo))
      AcquireReleaseImpl(thr, pc, &s->clock);
    else if (IsReleaseOrder(mo))
      ReleaseImpl(thr, pc, &s->clock);
    else if (IsAcquireOrder(mo))
      AcquireImpl(thr, pc, &s->clock);
  }
  T cc = *c;
  T pr = func_cas(a, cc, v);
  if (s) {
    if (write_lock)
      s->mtx.Unlock();
    else
      s->mtx.ReadUnlock();
  }
  if (pr == cc)
    return true;
  *c = pr;
  return false;
}

template<typename T>
static T AtomicCAS(ThreadState *thr, uptr pc,
    volatile T *a, T c, T v, morder mo, morder fmo) {
  AtomicCAS(thr, pc, a, &c, v, mo, fmo);
  return c;
}

#if !SANITIZER_GO
static void NoTsanAtomicFence(morder mo) {
  __sync_synchronize();
}

static void AtomicFence(ThreadState *thr, uptr pc, morder mo) {
  // FIXME(dvyukov): not implemented.
  __sync_synchronize();
}
#endif

// Interface functions follow.
#if !SANITIZER_GO

// C/C++

static morder convert_morder(morder mo) {
  if (flags()->force_seq_cst_atomics)
    return (morder)mo_seq_cst;

  // Filter out additional memory order flags:
  // MEMMODEL_SYNC        = 1 << 15
  // __ATOMIC_HLE_ACQUIRE = 1 << 16
  // __ATOMIC_HLE_RELEASE = 1 << 17
  //
  // HLE is an optimization, and we pretend that elision always fails.
  // MEMMODEL_SYNC is used when lowering __sync_ atomics,
  // since we use __sync_ atomics for actual atomic operations,
  // we can safely ignore it as well. It also subtly affects semantics,
  // but we don't model the difference.
  return (morder)(mo & 0x7fff);
}

#define SCOPED_ATOMIC(func, ...) \
    ThreadState *const thr = cur_thread(); \
    if (thr->ignore_sync || thr->ignore_interceptors) { \
      ProcessPendingSignals(thr); \
      return NoTsanAtomic##func(__VA_ARGS__); \
    } \
    const uptr callpc = (uptr)__builtin_return_address(0); \
    uptr pc = StackTrace::GetCurrentPc(); \
    mo = convert_morder(mo); \
    AtomicStatInc(thr, sizeof(*a), mo, StatAtomic##func); \
    ScopedAtomic sa(thr, callpc, a, mo, __func__); \
    return Atomic##func(thr, pc, __VA_ARGS__); \
/**/

class ScopedAtomic {
 public:
  ScopedAtomic(ThreadState *thr, uptr pc, const volatile void *a,
               morder mo, const char *func)
      : thr_(thr) {
    FuncEntry(thr_, pc);
    DPrintf("#%d: %s(%p, %d)\n", thr_->tid, func, a, mo);
  }
  ~ScopedAtomic() {
    ProcessPendingSignals(thr_);
    FuncExit(thr_);
  }
 private:
  ThreadState *thr_;
};

static void AtomicStatInc(ThreadState *thr, uptr size, morder mo, StatType t) {
  StatInc(thr, StatAtomic);
  StatInc(thr, t);
  StatInc(thr, size == 1 ? StatAtomic1
             : size == 2 ? StatAtomic2
             : size == 4 ? StatAtomic4
             : size == 8 ? StatAtomic8
             :             StatAtomic16);
  StatInc(thr, mo == mo_relaxed ? StatAtomicRelaxed
             : mo == mo_consume ? StatAtomicConsume
             : mo == mo_acquire ? StatAtomicAcquire
             : mo == mo_release ? StatAtomicRelease
             : mo == mo_acq_rel ? StatAtomicAcq_Rel
             :                    StatAtomicSeq_Cst);
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_load(const volatile a8 *a, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Load, a, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_load(const volatile a16 *a, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Load, a, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_load(const volatile a32 *a, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Load, a, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_load(const volatile a64 *a, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Load, a, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_load(const volatile a128 *a, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Load, a, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_atomic8_store(volatile a8 *a, a8 v, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Store, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_atomic16_store(volatile a16 *a, a16 v, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Store, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_atomic32_store(volatile a32 *a, a32 v, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Store, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_atomic64_store(volatile a64 *a, a64 v, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Store, a, v, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_atomic128_store(volatile a128 *a, a128 v, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(Store, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_exchange(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_exchange(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_exchange(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_exchange(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_exchange(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_fetch_add(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_fetch_add(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_fetch_add(volatile a32 *a, a32 v, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_fetch_add(volatile a64 *a, a64 v, morder mo) {
#if SANITIZER_RELACY_SCHEDULER
  _fiber_manager.Yield();
#endif
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_fetch_add(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_fetch_sub(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_fetch_sub(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_fetch_sub(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_fetch_sub(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_fetch_sub(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_fetch_and(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_fetch_and(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_fetch_and(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_fetch_and(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_fetch_and(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_fetch_or(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_fetch_or(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_fetch_or(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_fetch_or(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_fetch_or(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_fetch_xor(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_fetch_xor(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_fetch_xor(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_fetch_xor(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_fetch_xor(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_fetch_nand(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_fetch_nand(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_fetch_nand(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_fetch_nand(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_fetch_nand(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic8_compare_exchange_strong(volatile a8 *a, a8 *c, a8 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic16_compare_exchange_strong(volatile a16 *a, a16 *c, a16 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic32_compare_exchange_strong(volatile a32 *a, a32 *c, a32 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic64_compare_exchange_strong(volatile a64 *a, a64 *c, a64 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic128_compare_exchange_strong(volatile a128 *a, a128 *c, a128 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic8_compare_exchange_weak(volatile a8 *a, a8 *c, a8 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic16_compare_exchange_weak(volatile a16 *a, a16 *c, a16 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic32_compare_exchange_weak(volatile a32 *a, a32 *c, a32 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic64_compare_exchange_weak(volatile a64 *a, a64 *c, a64 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_atomic128_compare_exchange_weak(volatile a128 *a, a128 *c, a128 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __tsan_atomic8_compare_exchange_val(volatile a8 *a, a8 c, a8 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __tsan_atomic16_compare_exchange_val(volatile a16 *a, a16 c, a16 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __tsan_atomic32_compare_exchange_val(volatile a32 *a, a32 c, a32 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __tsan_atomic64_compare_exchange_val(volatile a64 *a, a64 c, a64 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

#if __TSAN_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __tsan_atomic128_compare_exchange_val(volatile a128 *a, a128 c, a128 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_atomic_thread_fence(morder mo) {
  char* a = 0;
  SCOPED_ATOMIC(Fence, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_atomic_signal_fence(morder mo) {
}
}  // extern "C"

#else  // #if !SANITIZER_GO

// Go

#define ATOMIC(func, ...) \
    if (thr->ignore_sync) { \
      NoTsanAtomic##func(__VA_ARGS__); \
    } else { \
      FuncEntry(thr, cpc); \
      Atomic##func(thr, pc, __VA_ARGS__); \
      FuncExit(thr); \
    } \
/**/

#define ATOMIC_RET(func, ret, ...) \
    if (thr->ignore_sync) { \
      (ret) = NoTsanAtomic##func(__VA_ARGS__); \
    } else { \
      FuncEntry(thr, cpc); \
      (ret) = Atomic##func(thr, pc, __VA_ARGS__); \
      FuncExit(thr); \
    } \
/**/

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic32_load(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(Load, *(a32*)(a+8), *(a32**)a, mo_acquire);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic64_load(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(Load, *(a64*)(a+8), *(a64**)a, mo_acquire);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic32_store(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC(Store, *(a32**)a, *(a32*)(a+8), mo_release);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic64_store(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC(Store, *(a64**)a, *(a64*)(a+8), mo_release);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic32_fetch_add(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(FetchAdd, *(a32*)(a+16), *(a32**)a, *(a32*)(a+8), mo_acq_rel);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic64_fetch_add(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(FetchAdd, *(a64*)(a+16), *(a64**)a, *(a64*)(a+8), mo_acq_rel);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic32_exchange(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(Exchange, *(a32*)(a+16), *(a32**)a, *(a32*)(a+8), mo_acq_rel);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic64_exchange(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(Exchange, *(a64*)(a+16), *(a64**)a, *(a64*)(a+8), mo_acq_rel);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic32_compare_exchange(
    ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  a32 cur = 0;
  a32 cmp = *(a32*)(a+8);
  ATOMIC_RET(CAS, cur, *(a32**)a, cmp, *(a32*)(a+12), mo_acq_rel, mo_acquire);
  *(bool*)(a+16) = (cur == cmp);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_go_atomic64_compare_exchange(
    ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  a64 cur = 0;
  a64 cmp = *(a64*)(a+8);
  ATOMIC_RET(CAS, cur, *(a64**)a, cmp, *(a64*)(a+16), mo_acq_rel, mo_acquire);
  *(bool*)(a+24) = (cur == cmp);
}
}  // extern "C"
#endif  // #if !SANITIZER_GO
//===-- tsan_md5.cc -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_defs.h"

namespace __tsan {

#define F(x, y, z)      ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z)      ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z)      ((x) ^ (y) ^ (z))
#define I(x, y, z)      ((y) ^ ((x) | ~(z)))

#define STEP(f, a, b, c, d, x, t, s) \
  (a) += f((b), (c), (d)) + (x) + (t); \
  (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
  (a) += (b);

#define SET(n) \
  (*(const MD5_u32plus *)&ptr[(n) * 4])
#define GET(n) \
  SET(n)

typedef unsigned int MD5_u32plus;
typedef unsigned long ulong_t;  // NOLINT

typedef struct {
  MD5_u32plus lo, hi;
  MD5_u32plus a, b, c, d;
  unsigned char buffer[64];
  MD5_u32plus block[16];
} MD5_CTX;

static const void *body(MD5_CTX *ctx, const void *data, ulong_t size) {
  const unsigned char *ptr = (const unsigned char *)data;
  MD5_u32plus a, b, c, d;
  MD5_u32plus saved_a, saved_b, saved_c, saved_d;

  a = ctx->a;
  b = ctx->b;
  c = ctx->c;
  d = ctx->d;

  do {
    saved_a = a;
    saved_b = b;
    saved_c = c;
    saved_d = d;

    STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
    STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
    STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
    STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
    STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
    STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
    STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
    STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
    STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
    STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
    STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
    STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
    STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
    STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
    STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
    STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

    STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
    STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
    STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
    STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
    STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
    STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
    STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
    STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
    STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
    STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
    STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
    STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
    STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
    STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

    STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
    STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
    STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
    STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
    STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
    STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
    STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
    STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
    STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
    STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
    STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
    STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
    STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
    STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
    STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)

    STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
    STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
    STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
    STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
    STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
    STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
    STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
    STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
    STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
    STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
    STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
    STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
    STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

    a += saved_a;
    b += saved_b;
    c += saved_c;
    d += saved_d;

    ptr += 64;
  } while (size -= 64);

  ctx->a = a;
  ctx->b = b;
  ctx->c = c;
  ctx->d = d;

  return ptr;
}

void MD5_Init(MD5_CTX *ctx) {
  ctx->a = 0x67452301;
  ctx->b = 0xefcdab89;
  ctx->c = 0x98badcfe;
  ctx->d = 0x10325476;

  ctx->lo = 0;
  ctx->hi = 0;
}

void MD5_Update(MD5_CTX *ctx, const void *data, ulong_t size) {
  MD5_u32plus saved_lo;
  ulong_t used, free;

  saved_lo = ctx->lo;
  if ((ctx->lo = (saved_lo + size) & 0x1fffffff) < saved_lo)
    ctx->hi++;
  ctx->hi += size >> 29;

  used = saved_lo & 0x3f;

  if (used) {
    free = 64 - used;

    if (size < free) {
      internal_memcpy(&ctx->buffer[used], data, size);
      return;
    }

    internal_memcpy(&ctx->buffer[used], data, free);
    data = (const unsigned char *)data + free;
    size -= free;
    body(ctx, ctx->buffer, 64);
  }

  if (size >= 64) {
    data = body(ctx, data, size & ~(ulong_t)0x3f);
    size &= 0x3f;
  }

  internal_memcpy(ctx->buffer, data, size);
}

void MD5_Final(unsigned char *result, MD5_CTX *ctx) {
  ulong_t used, free;

  used = ctx->lo & 0x3f;

  ctx->buffer[used++] = 0x80;

  free = 64 - used;

  if (free < 8) {
    internal_memset(&ctx->buffer[used], 0, free);
    body(ctx, ctx->buffer, 64);
    used = 0;
    free = 64;
  }

  internal_memset(&ctx->buffer[used], 0, free - 8);

  ctx->lo <<= 3;
  ctx->buffer[56] = ctx->lo;
  ctx->buffer[57] = ctx->lo >> 8;
  ctx->buffer[58] = ctx->lo >> 16;
  ctx->buffer[59] = ctx->lo >> 24;
  ctx->buffer[60] = ctx->hi;
  ctx->buffer[61] = ctx->hi >> 8;
  ctx->buffer[62] = ctx->hi >> 16;
  ctx->buffer[63] = ctx->hi >> 24;

  body(ctx, ctx->buffer, 64);

  result[0] = ctx->a;
  result[1] = ctx->a >> 8;
  result[2] = ctx->a >> 16;
  result[3] = ctx->a >> 24;
  result[4] = ctx->b;
  result[5] = ctx->b >> 8;
  result[6] = ctx->b >> 16;
  result[7] = ctx->b >> 24;
  result[8] = ctx->c;
  result[9] = ctx->c >> 8;
  result[10] = ctx->c >> 16;
  result[11] = ctx->c >> 24;
  result[12] = ctx->d;
  result[13] = ctx->d >> 8;
  result[14] = ctx->d >> 16;
  result[15] = ctx->d >> 24;

  internal_memset(ctx, 0, sizeof(*ctx));
}

MD5Hash md5_hash(const void *data, uptr size) {
  MD5Hash res;
  MD5_CTX ctx;
  MD5_Init(&ctx);
  MD5_Update(&ctx, data, size);
  MD5_Final((unsigned char*)&res.hash[0], &ctx);
  return res;
}
}  // namespace __tsan
//===-- tsan_mutex.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_mutex.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"

namespace __tsan {

// Simple reader-writer spin-mutex. Optimized for not-so-contended case.
// Readers have preference, can possibly starvate writers.

// The table fixes what mutexes can be locked under what mutexes.
// E.g. if the row for MutexTypeThreads contains MutexTypeReport,
// then Report mutex can be locked while under Threads mutex.
// The leaf mutexes can be locked under any other mutexes.
// Recursive locking is not supported.
#if SANITIZER_DEBUG && !SANITIZER_GO
const MutexType MutexTypeLeaf = (MutexType)-1;
static MutexType CanLockTab[MutexTypeCount][MutexTypeCount] = {
  /*0  MutexTypeInvalid*/     {},
  /*1  MutexTypeTrace*/       {MutexTypeLeaf},
  /*2  MutexTypeThreads*/     {MutexTypeReport},
  /*3  MutexTypeReport*/      {MutexTypeSyncVar,
                               MutexTypeMBlock, MutexTypeJavaMBlock},
  /*4  MutexTypeSyncVar*/     {MutexTypeDDetector},
  /*5  MutexTypeSyncTab*/     {},  // unused
  /*6  MutexTypeSlab*/        {MutexTypeLeaf},
  /*7  MutexTypeAnnotations*/ {},
  /*8  MutexTypeAtExit*/      {MutexTypeSyncVar},
  /*9  MutexTypeMBlock*/      {MutexTypeSyncVar},
  /*10 MutexTypeJavaMBlock*/  {MutexTypeSyncVar},
  /*11 MutexTypeDDetector*/   {},
  /*12 MutexTypeFired*/       {MutexTypeLeaf},
  /*13 MutexTypeRacy*/        {MutexTypeLeaf},
  /*14 MutexTypeGlobalProc*/  {},
};

static bool CanLockAdj[MutexTypeCount][MutexTypeCount];
#endif

void InitializeMutex() {
#if SANITIZER_DEBUG && !SANITIZER_GO
  // Build the "can lock" adjacency matrix.
  // If [i][j]==true, then one can lock mutex j while under mutex i.
  const int N = MutexTypeCount;
  int cnt[N] = {};
  bool leaf[N] = {};
  for (int i = 1; i < N; i++) {
    for (int j = 0; j < N; j++) {
      MutexType z = CanLockTab[i][j];
      if (z == MutexTypeInvalid)
        continue;
      if (z == MutexTypeLeaf) {
        CHECK(!leaf[i]);
        leaf[i] = true;
        continue;
      }
      CHECK(!CanLockAdj[i][(int)z]);
      CanLockAdj[i][(int)z] = true;
      cnt[i]++;
    }
  }
  for (int i = 0; i < N; i++) {
    CHECK(!leaf[i] || cnt[i] == 0);
  }
  // Add leaf mutexes.
  for (int i = 0; i < N; i++) {
    if (!leaf[i])
      continue;
    for (int j = 0; j < N; j++) {
      if (i == j || leaf[j] || j == MutexTypeInvalid)
        continue;
      CHECK(!CanLockAdj[j][i]);
      CanLockAdj[j][i] = true;
    }
  }
  // Build the transitive closure.
  bool CanLockAdj2[MutexTypeCount][MutexTypeCount];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      CanLockAdj2[i][j] = CanLockAdj[i][j];
    }
  }
  for (int k = 0; k < N; k++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if (CanLockAdj2[i][k] && CanLockAdj2[k][j]) {
          CanLockAdj2[i][j] = true;
        }
      }
    }
  }
#if 0
  Printf("Can lock graph:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Printf("%d ", CanLockAdj[i][j]);
    }
    Printf("\n");
  }
  Printf("Can lock graph closure:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Printf("%d ", CanLockAdj2[i][j]);
    }
    Printf("\n");
  }
#endif
  // Verify that the graph is acyclic.
  for (int i = 0; i < N; i++) {
    if (CanLockAdj2[i][i]) {
      Printf("Mutex %d participates in a cycle\n", i);
      Die();
    }
  }
#endif
}

InternalDeadlockDetector::InternalDeadlockDetector() {
  // Rely on zero initialization because some mutexes can be locked before ctor.
}

#if SANITIZER_DEBUG && !SANITIZER_GO
void InternalDeadlockDetector::Lock(MutexType t) {
  // Printf("LOCK %d @%zu\n", t, seq_ + 1);
  CHECK_GT(t, MutexTypeInvalid);
  CHECK_LT(t, MutexTypeCount);
  u64 max_seq = 0;
  u64 max_idx = MutexTypeInvalid;
  for (int i = 0; i != MutexTypeCount; i++) {
    if (locked_[i] == 0)
      continue;
    CHECK_NE(locked_[i], max_seq);
    if (max_seq < locked_[i]) {
      max_seq = locked_[i];
      max_idx = i;
    }
  }
  locked_[t] = ++seq_;
  if (max_idx == MutexTypeInvalid)
    return;
  // Printf("  last %d @%zu\n", max_idx, max_seq);
  if (!CanLockAdj[max_idx][t]) {
    Printf("ThreadSanitizer: internal deadlock detected\n");
    Printf("ThreadSanitizer: can't lock %d while under %zu\n",
               t, (uptr)max_idx);
    CHECK(0);
  }
}

void InternalDeadlockDetector::Unlock(MutexType t) {
  // Printf("UNLO %d @%zu #%zu\n", t, seq_, locked_[t]);
  CHECK(locked_[t]);
  locked_[t] = 0;
}

void InternalDeadlockDetector::CheckNoLocks() {
  for (int i = 0; i != MutexTypeCount; i++) {
    CHECK_EQ(locked_[i], 0);
  }
}
#endif

void CheckNoLocks(ThreadState *thr) {
#if SANITIZER_DEBUG && !SANITIZER_GO
  thr->internal_deadlock_detector.CheckNoLocks();
#endif
}

const uptr kUnlocked = 0;
const uptr kWriteLock = 1;
const uptr kReadLock = 2;

class Backoff {
 public:
  Backoff()
    : iter_() {
  }

  bool Do() {
    if (iter_++ < kActiveSpinIters)
      proc_yield(kActiveSpinCnt);
    else
      internal_sched_yield();
    return true;
  }

  u64 Contention() const {
    u64 active = iter_ % kActiveSpinIters;
    u64 passive = iter_ - active;
    return active + 10 * passive;
  }

 private:
  int iter_;
  static const int kActiveSpinIters = 10;
  static const int kActiveSpinCnt = 20;
};

Mutex::Mutex(MutexType type, StatType stat_type) {
  CHECK_GT(type, MutexTypeInvalid);
  CHECK_LT(type, MutexTypeCount);
#if SANITIZER_DEBUG
  type_ = type;
#endif
#if TSAN_COLLECT_STATS
  stat_type_ = stat_type;
#endif
  atomic_store(&state_, kUnlocked, memory_order_relaxed);
}

Mutex::~Mutex() {
  CHECK_EQ(atomic_load(&state_, memory_order_relaxed), kUnlocked);
}

void Mutex::Lock() {
#if SANITIZER_DEBUG && !SANITIZER_GO
  cur_thread()->internal_deadlock_detector.Lock(type_);
#endif
  uptr cmp = kUnlocked;
  if (atomic_compare_exchange_strong(&state_, &cmp, kWriteLock,
                                     memory_order_acquire))
    return;
  for (Backoff backoff; backoff.Do();) {
    if (atomic_load(&state_, memory_order_relaxed) == kUnlocked) {
      cmp = kUnlocked;
      if (atomic_compare_exchange_weak(&state_, &cmp, kWriteLock,
                                       memory_order_acquire)) {
#if TSAN_COLLECT_STATS && !SANITIZER_GO
        StatInc(cur_thread(), stat_type_, backoff.Contention());
#endif
        return;
      }
    }
  }
}

void Mutex::Unlock() {
  uptr prev = atomic_fetch_sub(&state_, kWriteLock, memory_order_release);
  (void)prev;
  DCHECK_NE(prev & kWriteLock, 0);
#if SANITIZER_DEBUG && !SANITIZER_GO
  cur_thread()->internal_deadlock_detector.Unlock(type_);
#endif
}

void Mutex::ReadLock() {
#if SANITIZER_DEBUG && !SANITIZER_GO
  cur_thread()->internal_deadlock_detector.Lock(type_);
#endif
  uptr prev = atomic_fetch_add(&state_, kReadLock, memory_order_acquire);
  if ((prev & kWriteLock) == 0)
    return;
  for (Backoff backoff; backoff.Do();) {
    prev = atomic_load(&state_, memory_order_acquire);
    if ((prev & kWriteLock) == 0) {
#if TSAN_COLLECT_STATS && !SANITIZER_GO
      StatInc(cur_thread(), stat_type_, backoff.Contention());
#endif
      return;
    }
  }
}

void Mutex::ReadUnlock() {
  uptr prev = atomic_fetch_sub(&state_, kReadLock, memory_order_release);
  (void)prev;
  DCHECK_EQ(prev & kWriteLock, 0);
  DCHECK_GT(prev & ~kWriteLock, 0);
#if SANITIZER_DEBUG && !SANITIZER_GO
  cur_thread()->internal_deadlock_detector.Unlock(type_);
#endif
}

void Mutex::CheckLocked() {
  CHECK_NE(atomic_load(&state_, memory_order_relaxed), 0);
}

}  // namespace __tsan
//===-- tsan_report.cc ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_report.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"

namespace __tsan {

ReportStack::ReportStack() : frames(nullptr), suppressable(false) {}

ReportStack *ReportStack::New() {
  void *mem = internal_alloc(MBlockReportStack, sizeof(ReportStack));
  return new(mem) ReportStack();
}

ReportLocation::ReportLocation(ReportLocationType type)
    : type(type), global(), heap_chunk_start(0), heap_chunk_size(0), tid(0),
      fd(0), suppressable(false), stack(nullptr) {}

ReportLocation *ReportLocation::New(ReportLocationType type) {
  void *mem = internal_alloc(MBlockReportStack, sizeof(ReportLocation));
  return new(mem) ReportLocation(type);
}

class Decorator: public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() { }
  const char *Access()     { return Blue(); }
  const char *ThreadDescription()    { return Cyan(); }
  const char *Location()   { return Green(); }
  const char *Sleep()   { return Yellow(); }
  const char *Mutex()   { return Magenta(); }
};

ReportDesc::ReportDesc()
    : tag(kExternalTagNone)
    , stacks()
    , mops()
    , locs()
    , mutexes()
    , threads()
    , unique_tids()
    , sleep()
    , count() {
}

ReportMop::ReportMop()
    : mset() {
}

ReportDesc::~ReportDesc() {
  // FIXME(dvyukov): it must be leaking a lot of memory.
}

#if !SANITIZER_GO

const int kThreadBufSize = 32;
const char *thread_name(char *buf, int tid) {
  if (tid == 0)
    return "main thread";
  internal_snprintf(buf, kThreadBufSize, "thread T%d", tid);
  return buf;
}

static const char *ReportTypeString(ReportType typ, uptr tag) {
  if (typ == ReportTypeRace)
    return "data race";
  if (typ == ReportTypeVptrRace)
    return "data race on vptr (ctor/dtor vs virtual call)";
  if (typ == ReportTypeUseAfterFree)
    return "heap-use-after-free";
  if (typ == ReportTypeVptrUseAfterFree)
    return "heap-use-after-free (virtual call vs free)";
  if (typ == ReportTypeExternalRace) {
    const char *str = GetReportHeaderFromTag(tag);
    return str ? str : "race on external object";
  }
  if (typ == ReportTypeThreadLeak)
    return "thread leak";
  if (typ == ReportTypeMutexDestroyLocked)
    return "destroy of a locked mutex";
  if (typ == ReportTypeMutexDoubleLock)
    return "double lock of a mutex";
  if (typ == ReportTypeMutexInvalidAccess)
    return "use of an invalid mutex (e.g. uninitialized or destroyed)";
  if (typ == ReportTypeMutexBadUnlock)
    return "unlock of an unlocked mutex (or by a wrong thread)";
  if (typ == ReportTypeMutexBadReadLock)
    return "read lock of a write locked mutex";
  if (typ == ReportTypeMutexBadReadUnlock)
    return "read unlock of a write locked mutex";
  if (typ == ReportTypeSignalUnsafe)
    return "signal-unsafe call inside of a signal";
  if (typ == ReportTypeErrnoInSignal)
    return "signal handler spoils errno";
  if (typ == ReportTypeDeadlock)
    return "lock-order-inversion (potential deadlock)";
  return "";
}

#if SANITIZER_MAC
static const char *const kInterposedFunctionPrefix = "wrap_";
#else
static const char *const kInterposedFunctionPrefix = "__interceptor_";
#endif

void PrintStack(const ReportStack *ent) {
  if (ent == 0 || ent->frames == 0) {
    Printf("    [failed to restore the stack]\n\n");
    return;
  }
  SymbolizedStack *frame = ent->frames;
  for (int i = 0; frame && frame->info.address; frame = frame->next, i++) {
    InternalScopedString res(2 * GetPageSizeCached());
    RenderFrame(&res, common_flags()->stack_trace_format, i, frame->info,
                common_flags()->symbolize_vs_style,
                common_flags()->strip_path_prefix, kInterposedFunctionPrefix);
    Printf("%s\n", res.data());
  }
  Printf("\n");
}

static void PrintMutexSet(Vector<ReportMopMutex> const& mset) {
  for (uptr i = 0; i < mset.Size(); i++) {
    if (i == 0)
      Printf(" (mutexes:");
    const ReportMopMutex m = mset[i];
    Printf(" %s M%llu", m.write ? "write" : "read", m.id);
    Printf(i == mset.Size() - 1 ? ")" : ",");
  }
}

static const char *MopDesc(bool first, bool write, bool atomic) {
  return atomic ? (first ? (write ? "Atomic write" : "Atomic read")
                : (write ? "Previous atomic write" : "Previous atomic read"))
                : (first ? (write ? "Write" : "Read")
                : (write ? "Previous write" : "Previous read"));
}

static const char *ExternalMopDesc(bool first, bool write) {
  return first ? (write ? "Modifying" : "Read-only")
               : (write ? "Previous modifying" : "Previous read-only");
}

static void PrintMop(const ReportMop *mop, bool first) {
  Decorator d;
  char thrbuf[kThreadBufSize];
  Printf("%s", d.Access());
  if (mop->external_tag == kExternalTagNone) {
    Printf("  %s of size %d at %p by %s",
           MopDesc(first, mop->write, mop->atomic), mop->size,
           (void *)mop->addr, thread_name(thrbuf, mop->tid));
  } else {
    const char *object_type = GetObjectTypeFromTag(mop->external_tag);
    if (object_type == nullptr)
        object_type = "external object";
    Printf("  %s access of %s at %p by %s",
           ExternalMopDesc(first, mop->write), object_type,
           (void *)mop->addr, thread_name(thrbuf, mop->tid));
  }
  PrintMutexSet(mop->mset);
  Printf(":\n");
  Printf("%s", d.Default());
  PrintStack(mop->stack);
}

static void PrintLocation(const ReportLocation *loc) {
  Decorator d;
  char thrbuf[kThreadBufSize];
  bool print_stack = false;
  Printf("%s", d.Location());
  if (loc->type == ReportLocationGlobal) {
    const DataInfo &global = loc->global;
    if (global.size != 0)
      Printf("  Location is global '%s' of size %zu at %p (%s+%p)\n\n",
             global.name, global.size, global.start,
             StripModuleName(global.module), global.module_offset);
    else
      Printf("  Location is global '%s' at %p (%s+%p)\n\n", global.name,
             global.start, StripModuleName(global.module),
             global.module_offset);
  } else if (loc->type == ReportLocationHeap) {
    char thrbuf[kThreadBufSize];
    const char *object_type = GetObjectTypeFromTag(loc->external_tag);
    if (!object_type) {
      Printf("  Location is heap block of size %zu at %p allocated by %s:\n",
             loc->heap_chunk_size, loc->heap_chunk_start,
             thread_name(thrbuf, loc->tid));
    } else {
      Printf("  Location is %s of size %zu at %p allocated by %s:\n",
             object_type, loc->heap_chunk_size, loc->heap_chunk_start,
             thread_name(thrbuf, loc->tid));
    }
    print_stack = true;
  } else if (loc->type == ReportLocationStack) {
    Printf("  Location is stack of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationTLS) {
    Printf("  Location is TLS of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationFD) {
    Printf("  Location is file descriptor %d created by %s at:\n",
        loc->fd, thread_name(thrbuf, loc->tid));
    print_stack = true;
  }
  Printf("%s", d.Default());
  if (print_stack)
    PrintStack(loc->stack);
}

static void PrintMutexShort(const ReportMutex *rm, const char *after) {
  Decorator d;
  Printf("%sM%zd%s%s", d.Mutex(), rm->id, d.Default(), after);
}

static void PrintMutexShortWithAddress(const ReportMutex *rm,
                                       const char *after) {
  Decorator d;
  Printf("%sM%zd (%p)%s%s", d.Mutex(), rm->id, rm->addr, d.Default(), after);
}

static void PrintMutex(const ReportMutex *rm) {
  Decorator d;
  if (rm->destroyed) {
    Printf("%s", d.Mutex());
    Printf("  Mutex M%llu is already destroyed.\n\n", rm->id);
    Printf("%s", d.Default());
  } else {
    Printf("%s", d.Mutex());
    Printf("  Mutex M%llu (%p) created at:\n", rm->id, rm->addr);
    Printf("%s", d.Default());
    PrintStack(rm->stack);
  }
}

static void PrintThread(const ReportThread *rt) {
  Decorator d;
  if (rt->id == 0)  // Little sense in describing the main thread.
    return;
  Printf("%s", d.ThreadDescription());
  Printf("  Thread T%d", rt->id);
  if (rt->name && rt->name[0] != '\0')
    Printf(" '%s'", rt->name);
  char thrbuf[kThreadBufSize];
  const char *thread_status = rt->running ? "running" : "finished";
  if (rt->workerthread) {
    Printf(" (tid=%zu, %s) is a GCD worker thread\n", rt->os_id, thread_status);
    Printf("\n");
    Printf("%s", d.Default());
    return;
  }
  Printf(" (tid=%zu, %s) created by %s", rt->os_id, thread_status,
         thread_name(thrbuf, rt->parent_tid));
  if (rt->stack)
    Printf(" at:");
  Printf("\n");
  Printf("%s", d.Default());
  PrintStack(rt->stack);
}

static void PrintSleep(const ReportStack *s) {
  Decorator d;
  Printf("%s", d.Sleep());
  Printf("  As if synchronized via sleep:\n");
  Printf("%s", d.Default());
  PrintStack(s);
}

static ReportStack *ChooseSummaryStack(const ReportDesc *rep) {
  if (rep->mops.Size())
    return rep->mops[0]->stack;
  if (rep->stacks.Size())
    return rep->stacks[0];
  if (rep->mutexes.Size())
    return rep->mutexes[0]->stack;
  if (rep->threads.Size())
    return rep->threads[0]->stack;
  return 0;
}

static bool FrameIsInternal(const SymbolizedStack *frame) {
  if (frame == 0)
    return false;
  const char *file = frame->info.file;
  const char *module = frame->info.module;
  if (file != 0 &&
      (internal_strstr(file, "tsan_interceptors.cc") ||
       internal_strstr(file, "sanitizer_common_interceptors.inc") ||
       internal_strstr(file, "tsan_interface_")))
    return true;
  if (module != 0 && (internal_strstr(module, "libclang_rt.tsan_")))
    return true;
  return false;
}

static SymbolizedStack *SkipTsanInternalFrames(SymbolizedStack *frames) {
  while (FrameIsInternal(frames) && frames->next)
    frames = frames->next;
  return frames;
}

void PrintReport(const ReportDesc *rep) {
  Decorator d;
  Printf("==================\n");
  const char *rep_typ_str = ReportTypeString(rep->typ, rep->tag);
  Printf("%s", d.Warning());
  Printf("WARNING: ThreadSanitizer: %s (pid=%d)\n", rep_typ_str,
         (int)internal_getpid());
  Printf("%s", d.Default());

  if (rep->typ == ReportTypeDeadlock) {
    char thrbuf[kThreadBufSize];
    Printf("  Cycle in lock order graph: ");
    for (uptr i = 0; i < rep->mutexes.Size(); i++)
      PrintMutexShortWithAddress(rep->mutexes[i], " => ");
    PrintMutexShort(rep->mutexes[0], "\n\n");
    CHECK_GT(rep->mutexes.Size(), 0U);
    CHECK_EQ(rep->mutexes.Size() * (flags()->second_deadlock_stack ? 2 : 1),
             rep->stacks.Size());
    for (uptr i = 0; i < rep->mutexes.Size(); i++) {
      Printf("  Mutex ");
      PrintMutexShort(rep->mutexes[(i + 1) % rep->mutexes.Size()],
                      " acquired here while holding mutex ");
      PrintMutexShort(rep->mutexes[i], " in ");
      Printf("%s", d.ThreadDescription());
      Printf("%s:\n", thread_name(thrbuf, rep->unique_tids[i]));
      Printf("%s", d.Default());
      if (flags()->second_deadlock_stack) {
        PrintStack(rep->stacks[2*i]);
        Printf("  Mutex ");
        PrintMutexShort(rep->mutexes[i],
                        " previously acquired by the same thread here:\n");
        PrintStack(rep->stacks[2*i+1]);
      } else {
        PrintStack(rep->stacks[i]);
        if (i == 0)
          Printf("    Hint: use TSAN_OPTIONS=second_deadlock_stack=1 "
                 "to get more informative warning message\n\n");
      }
    }
  } else {
    for (uptr i = 0; i < rep->stacks.Size(); i++) {
      if (i)
        Printf("  and:\n");
      PrintStack(rep->stacks[i]);
    }
  }

  for (uptr i = 0; i < rep->mops.Size(); i++)
    PrintMop(rep->mops[i], i == 0);

  if (rep->sleep)
    PrintSleep(rep->sleep);

  for (uptr i = 0; i < rep->locs.Size(); i++)
    PrintLocation(rep->locs[i]);

  if (rep->typ != ReportTypeDeadlock) {
    for (uptr i = 0; i < rep->mutexes.Size(); i++)
      PrintMutex(rep->mutexes[i]);
  }

  for (uptr i = 0; i < rep->threads.Size(); i++)
    PrintThread(rep->threads[i]);

  if (rep->typ == ReportTypeThreadLeak && rep->count > 1)
    Printf("  And %d more similar thread leaks.\n\n", rep->count - 1);

  if (ReportStack *stack = ChooseSummaryStack(rep)) {
    if (SymbolizedStack *frame = SkipTsanInternalFrames(stack->frames))
      ReportErrorSummary(rep_typ_str, frame->info);
  }

  if (common_flags()->print_module_map == 2) PrintModuleMap();

  Printf("==================\n");
}

#else  // #if !SANITIZER_GO

const int kMainThreadId = 1;

void PrintStack(const ReportStack *ent) {
  if (ent == 0 || ent->frames == 0) {
    Printf("  [failed to restore the stack]\n");
    return;
  }
  SymbolizedStack *frame = ent->frames;
  for (int i = 0; frame; frame = frame->next, i++) {
    const AddressInfo &info = frame->info;
    Printf("  %s()\n      %s:%d +0x%zx\n", info.function,
        StripPathPrefix(info.file, common_flags()->strip_path_prefix),
        info.line, (void *)info.module_offset);
  }
}

static void PrintMop(const ReportMop *mop, bool first) {
  Printf("\n");
  Printf("%s at %p by ",
      (first ? (mop->write ? "Write" : "Read")
             : (mop->write ? "Previous write" : "Previous read")), mop->addr);
  if (mop->tid == kMainThreadId)
    Printf("main goroutine:\n");
  else
    Printf("goroutine %d:\n", mop->tid);
  PrintStack(mop->stack);
}

static void PrintLocation(const ReportLocation *loc) {
  switch (loc->type) {
  case ReportLocationHeap: {
    Printf("\n");
    Printf("Heap block of size %zu at %p allocated by ",
        loc->heap_chunk_size, loc->heap_chunk_start);
    if (loc->tid == kMainThreadId)
      Printf("main goroutine:\n");
    else
      Printf("goroutine %d:\n", loc->tid);
    PrintStack(loc->stack);
    break;
  }
  case ReportLocationGlobal: {
    Printf("\n");
    Printf("Global var %s of size %zu at %p declared at %s:%zu\n",
        loc->global.name, loc->global.size, loc->global.start,
        loc->global.file, loc->global.line);
    break;
  }
  default:
    break;
  }
}

static void PrintThread(const ReportThread *rt) {
  if (rt->id == kMainThreadId)
    return;
  Printf("\n");
  Printf("Goroutine %d (%s) created at:\n",
    rt->id, rt->running ? "running" : "finished");
  PrintStack(rt->stack);
}

void PrintReport(const ReportDesc *rep) {
  Printf("==================\n");
  if (rep->typ == ReportTypeRace) {
    Printf("WARNING: DATA RACE");
    for (uptr i = 0; i < rep->mops.Size(); i++)
      PrintMop(rep->mops[i], i == 0);
    for (uptr i = 0; i < rep->locs.Size(); i++)
      PrintLocation(rep->locs[i]);
    for (uptr i = 0; i < rep->threads.Size(); i++)
      PrintThread(rep->threads[i]);
  } else if (rep->typ == ReportTypeDeadlock) {
    Printf("WARNING: DEADLOCK\n");
    for (uptr i = 0; i < rep->mutexes.Size(); i++) {
      Printf("Goroutine %d lock mutex %d while holding mutex %d:\n",
          999, rep->mutexes[i]->id,
          rep->mutexes[(i+1) % rep->mutexes.Size()]->id);
      PrintStack(rep->stacks[2*i]);
      Printf("\n");
      Printf("Mutex %d was previously locked here:\n",
          rep->mutexes[(i+1) % rep->mutexes.Size()]->id);
      PrintStack(rep->stacks[2*i + 1]);
      Printf("\n");
    }
  }
  Printf("==================\n");
}

#endif

}  // namespace __tsan
//===-- tsan_rtl.cc -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Main file (entry points) for the TSan run-time.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "tsan_defs.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "tsan_suppressions.h"
#include "tsan_symbolize.h"
#include "ubsan/ubsan_init.h"
#if SANITIZER_RELACY_SCHEDULER
#include "relacy/tsan_scheduler_engine.h"
#endif

#ifdef __SSE3__
// <emmintrin.h> transitively includes <stdlib.h>,
// and it's prohibited to include std headers into tsan runtime.
// So we do this dirty trick.
#define _MM_MALLOC_H_INCLUDED
#define __MM_MALLOC_H
#include <emmintrin.h>
typedef __m128i m128;
#endif

volatile int __tsan_resumed = 0;

extern "C" void __tsan_resume() {
  __tsan_resumed = 1;
}

namespace __tsan {

#if !SANITIZER_GO && !SANITIZER_MAC
__attribute__((tls_model("initial-exec")))
THREADLOCAL char cur_thread_placeholder[sizeof(ThreadState)] ALIGNED(64);
#endif
static char ctx_placeholder[sizeof(Context)] ALIGNED(64);
Context *ctx;

// Can be overriden by a front-end.
#ifdef TSAN_EXTERNAL_HOOKS
bool OnFinalize(bool failed);
void OnInitialize();
#else
SANITIZER_WEAK_CXX_DEFAULT_IMPL
bool OnFinalize(bool failed) {
  return failed;
}
SANITIZER_WEAK_CXX_DEFAULT_IMPL
void OnInitialize() {}
#endif

static char thread_registry_placeholder[sizeof(ThreadRegistry)];

static ThreadContextBase *CreateThreadContext(u32 tid) {
  // Map thread trace when context is created.
  char name[50];
  internal_snprintf(name, sizeof(name), "trace %u", tid);
  MapThreadTrace(GetThreadTrace(tid), TraceSize() * sizeof(Event), name);
  const uptr hdr = GetThreadTraceHeader(tid);
  internal_snprintf(name, sizeof(name), "trace header %u", tid);
  MapThreadTrace(hdr, sizeof(Trace), name);
  new((void*)hdr) Trace();
  // We are going to use only a small part of the trace with the default
  // value of history_size. However, the constructor writes to the whole trace.
  // Unmap the unused part.
  uptr hdr_end = hdr + sizeof(Trace);
  hdr_end -= sizeof(TraceHeader) * (kTraceParts - TraceParts());
  hdr_end = RoundUp(hdr_end, GetPageSizeCached());
  if (hdr_end < hdr + sizeof(Trace))
    UnmapOrDie((void*)hdr_end, hdr + sizeof(Trace) - hdr_end);
  void *mem = internal_alloc(MBlockThreadContex, sizeof(ThreadContext));
  return new(mem) ThreadContext(tid);
}

#if !SANITIZER_GO
static const u32 kThreadQuarantineSize = 16;
#else
static const u32 kThreadQuarantineSize = 64;
#endif

Context::Context()
  : initialized()
  , report_mtx(MutexTypeReport, StatMtxReport)
  , nreported()
  , nmissed_expected()
  , thread_registry(new(thread_registry_placeholder) ThreadRegistry(
      CreateThreadContext, kMaxTid, kThreadQuarantineSize, kMaxTidReuse))
  , racy_mtx(MutexTypeRacy, StatMtxRacy)
  , racy_stacks()
  , racy_addresses()
  , fired_suppressions_mtx(MutexTypeFired, StatMtxFired)
  , fired_suppressions(8)
  , clock_alloc("clock allocator") {
}

// The objects are allocated in TLS, so one may rely on zero-initialization.
ThreadState::ThreadState(Context *ctx, int tid, int unique_id, u64 epoch,
                         unsigned reuse_count,
                         uptr stk_addr, uptr stk_size,
                         uptr tls_addr, uptr tls_size)
  : fast_state(tid, epoch)
  // Do not touch these, rely on zero initialization,
  // they may be accessed before the ctor.
  // , ignore_reads_and_writes()
  // , ignore_interceptors()
  , clock(tid, reuse_count)
#if !SANITIZER_GO
  , jmp_bufs()
#endif
  , tid(tid)
  , unique_id(unique_id)
  , stk_addr(stk_addr)
  , stk_size(stk_size)
  , tls_addr(tls_addr)
  , tls_size(tls_size)
#if !SANITIZER_GO
  , last_sleep_clock(tid)
#endif
{
}

#if !SANITIZER_GO
static void MemoryProfiler(Context *ctx, fd_t fd, int i) {
  uptr n_threads;
  uptr n_running_threads;
  ctx->thread_registry->GetNumberOfThreads(&n_threads, &n_running_threads);
  InternalScopedBuffer<char> buf(4096);
  WriteMemoryProfile(buf.data(), buf.size(), n_threads, n_running_threads);
  WriteToFile(fd, buf.data(), internal_strlen(buf.data()));
}

static void BackgroundThread(void *arg) {
  // This is a non-initialized non-user thread, nothing to see here.
  // We don't use ScopedIgnoreInterceptors, because we want ignores to be
  // enabled even when the thread function exits (e.g. during pthread thread
  // shutdown code).
  cur_thread()->ignore_interceptors++;
  const u64 kMs2Ns = 1000 * 1000;

  fd_t mprof_fd = kInvalidFd;
  if (flags()->profile_memory && flags()->profile_memory[0]) {
    if (internal_strcmp(flags()->profile_memory, "stdout") == 0) {
      mprof_fd = 1;
    } else if (internal_strcmp(flags()->profile_memory, "stderr") == 0) {
      mprof_fd = 2;
    } else {
      InternalScopedString filename(kMaxPathLength);
      filename.append("%s.%d", flags()->profile_memory, (int)internal_getpid());
      fd_t fd = OpenFile(filename.data(), WrOnly);
      if (fd == kInvalidFd) {
        Printf("ThreadSanitizer: failed to open memory profile file '%s'\n",
            &filename[0]);
      } else {
        mprof_fd = fd;
      }
    }
  }

  u64 last_flush = NanoTime();
  uptr last_rss = 0;
  for (int i = 0;
      atomic_load(&ctx->stop_background_thread, memory_order_relaxed) == 0;
      i++) {
    SleepForMillis(100);
    u64 now = NanoTime();

    // Flush memory if requested.
    if (flags()->flush_memory_ms > 0) {
      if (last_flush + flags()->flush_memory_ms * kMs2Ns < now) {
        VPrintf(1, "ThreadSanitizer: periodic memory flush\n");
        FlushShadowMemory();
        last_flush = NanoTime();
      }
    }
    // GetRSS can be expensive on huge programs, so don't do it every 100ms.
    if (flags()->memory_limit_mb > 0) {
      uptr rss = GetRSS();
      uptr limit = uptr(flags()->memory_limit_mb) << 20;
      VPrintf(1, "ThreadSanitizer: memory flush check"
                 " RSS=%llu LAST=%llu LIMIT=%llu\n",
              (u64)rss >> 20, (u64)last_rss >> 20, (u64)limit >> 20);
      if (2 * rss > limit + last_rss) {
        VPrintf(1, "ThreadSanitizer: flushing memory due to RSS\n");
        FlushShadowMemory();
        rss = GetRSS();
        VPrintf(1, "ThreadSanitizer: memory flushed RSS=%llu\n", (u64)rss>>20);
      }
      last_rss = rss;
    }

    // Write memory profile if requested.
    if (mprof_fd != kInvalidFd)
      MemoryProfiler(ctx, mprof_fd, i);

    // Flush symbolizer cache if requested.
    if (flags()->flush_symbolizer_ms > 0) {
      u64 last = atomic_load(&ctx->last_symbolize_time_ns,
                             memory_order_relaxed);
      if (last != 0 && last + flags()->flush_symbolizer_ms * kMs2Ns < now) {
        Lock l(&ctx->report_mtx);
        ScopedErrorReportLock l2;
        SymbolizeFlush();
        atomic_store(&ctx->last_symbolize_time_ns, 0, memory_order_relaxed);
      }
    }
  }
}

static void StartBackgroundThread() {
  ctx->background_thread = internal_start_thread(&BackgroundThread, 0);
}

#ifndef __mips__
static void StopBackgroundThread() {
  atomic_store(&ctx->stop_background_thread, 1, memory_order_relaxed);
  internal_join_thread(ctx->background_thread);
  ctx->background_thread = 0;
}
#endif
#endif

void DontNeedShadowFor(uptr addr, uptr size) {
  ReleaseMemoryPagesToOS(MemToShadow(addr), MemToShadow(addr + size));
}

void MapShadow(uptr addr, uptr size) {
  // Global data is not 64K aligned, but there are no adjacent mappings,
  // so we can get away with unaligned mapping.
  // CHECK_EQ(addr, addr & ~((64 << 10) - 1));  // windows wants 64K alignment
  const uptr kPageSize = GetPageSizeCached();
  uptr shadow_begin = RoundDownTo((uptr)MemToShadow(addr), kPageSize);
  uptr shadow_end = RoundUpTo((uptr)MemToShadow(addr + size), kPageSize);
  MmapFixedNoReserve(shadow_begin, shadow_end - shadow_begin, "shadow");

  // Meta shadow is 2:1, so tread carefully.
  static bool data_mapped = false;
  static uptr mapped_meta_end = 0;
  uptr meta_begin = (uptr)MemToMeta(addr);
  uptr meta_end = (uptr)MemToMeta(addr + size);
  meta_begin = RoundDownTo(meta_begin, 64 << 10);
  meta_end = RoundUpTo(meta_end, 64 << 10);
  if (!data_mapped) {
    // First call maps data+bss.
    data_mapped = true;
    MmapFixedNoReserve(meta_begin, meta_end - meta_begin, "meta shadow");
  } else {
    // Mapping continous heap.
    // Windows wants 64K alignment.
    meta_begin = RoundDownTo(meta_begin, 64 << 10);
    meta_end = RoundUpTo(meta_end, 64 << 10);
    if (meta_end <= mapped_meta_end)
      return;
    if (meta_begin < mapped_meta_end)
      meta_begin = mapped_meta_end;
    MmapFixedNoReserve(meta_begin, meta_end - meta_begin, "meta shadow");
    mapped_meta_end = meta_end;
  }
  VPrintf(2, "mapped meta shadow for (%p-%p) at (%p-%p)\n",
      addr, addr+size, meta_begin, meta_end);
}

void MapThreadTrace(uptr addr, uptr size, const char *name) {
  DPrintf("#0: Mapping trace at %p-%p(0x%zx)\n", addr, addr + size, size);
  CHECK_GE(addr, TraceMemBeg());
  CHECK_LE(addr + size, TraceMemEnd());
  CHECK_EQ(addr, addr & ~((64 << 10) - 1));  // windows wants 64K alignment
  uptr addr1 = (uptr)MmapFixedNoReserve(addr, size, name);
  if (addr1 != addr) {
    Printf("FATAL: ThreadSanitizer can not mmap thread trace (%p/%p->%p)\n",
        addr, size, addr1);
    Die();
  }
}

static void CheckShadowMapping() {
  uptr beg, end;
  for (int i = 0; GetUserRegion(i, &beg, &end); i++) {
    // Skip cases for empty regions (heap definition for architectures that
    // do not use 64-bit allocator).
    if (beg == end)
      continue;
    VPrintf(3, "checking shadow region %p-%p\n", beg, end);
    uptr prev = 0;
    for (uptr p0 = beg; p0 <= end; p0 += (end - beg) / 4) {
      for (int x = -(int)kShadowCell; x <= (int)kShadowCell; x += kShadowCell) {
        const uptr p = RoundDown(p0 + x, kShadowCell);
        if (p < beg || p >= end)
          continue;
        const uptr s = MemToShadow(p);
        const uptr m = (uptr)MemToMeta(p);
        VPrintf(3, "  checking pointer %p: shadow=%p meta=%p\n", p, s, m);
        CHECK(IsAppMem(p));
        CHECK(IsShadowMem(s));
        CHECK_EQ(p, ShadowToMem(s));
        CHECK(IsMetaMem(m));
        if (prev) {
          // Ensure that shadow and meta mappings are linear within a single
          // user range. Lots of code that processes memory ranges assumes it.
          const uptr prev_s = MemToShadow(prev);
          const uptr prev_m = (uptr)MemToMeta(prev);
          CHECK_EQ(s - prev_s, (p - prev) * kShadowMultiplier);
          CHECK_EQ((m - prev_m) / kMetaShadowSize,
                   (p - prev) / kMetaShadowCell);
        }
        prev = p;
      }
    }
  }
}

#if !SANITIZER_GO
static void OnStackUnwind(const SignalContext &sig, const void *,
                          BufferedStackTrace *stack) {
  uptr top = 0;
  uptr bottom = 0;
  bool fast = common_flags()->fast_unwind_on_fatal;
  if (fast) GetThreadStackTopAndBottom(false, &top, &bottom);
  stack->Unwind(kStackTraceMax, sig.pc, sig.bp, sig.context, top, bottom, fast);
}

static void TsanOnDeadlySignal(int signo, void *siginfo, void *context) {
  HandleDeadlySignal(siginfo, context, GetTid(), &OnStackUnwind, nullptr);
}
#endif

void Initialize(ThreadState *thr) {
  // Thread safe because done before all threads exist.
  static bool is_initialized = false;
  if (is_initialized)
    return;
  is_initialized = true;
  // We are not ready to handle interceptors yet.
  ScopedIgnoreInterceptors ignore;
  SanitizerToolName = "ThreadSanitizer";
  // Install tool-specific callbacks in sanitizer_common.
  SetCheckFailedCallback(TsanCheckFailed);

  ctx = new(ctx_placeholder) Context;
  const char *options = GetEnv(SANITIZER_GO ? "GORACE" : "TSAN_OPTIONS");
  CacheBinaryName();
  InitializeFlags(&ctx->flags, options);
  AvoidCVE_2016_2143();
  InitializePlatformEarly();
#if !SANITIZER_GO
  // Re-exec ourselves if we need to set additional env or command line args.
  MaybeReexec();

  InitializeAllocator();
  ReplaceSystemMalloc();
#endif
  if (common_flags()->detect_deadlocks)
    ctx->dd = DDetector::Create(flags());
  Processor *proc = ProcCreate();
  ProcWire(proc, thr);
  InitializeInterceptors();
  CheckShadowMapping();
  InitializePlatform();
  InitializeMutex();
  InitializeDynamicAnnotations();
#if !SANITIZER_GO
  InitializeShadowMemory();
  InitializeAllocatorLate();
  InstallDeadlySignalHandlers(TsanOnDeadlySignal);
#endif
  // Setup correct file descriptor for error reports.
  __sanitizer_set_report_path(common_flags()->log_path);
  InitializeSuppressions();
#if !SANITIZER_GO
  InitializeLibIgnore();
  Symbolizer::GetOrInit()->AddHooks(EnterSymbolizer, ExitSymbolizer);
#endif

  VPrintf(1, "***** Running under ThreadSanitizer v2 (pid %d) *****\n",
          (int)internal_getpid());

  // Initialize thread 0.
  int tid = ThreadCreate(thr, 0, 0, true);
  CHECK_EQ(tid, 0);
  ThreadStart(thr, tid, GetTid(), /*workerthread*/ false);
#if TSAN_CONTAINS_UBSAN
  __ubsan::InitAsPlugin();
#endif
  ctx->initialized = true;

#if !SANITIZER_GO
  Symbolizer::LateInitialize();
#endif

  if (flags()->stop_on_start) {
    Printf("ThreadSanitizer is suspended at startup (pid %d)."
           " Call __tsan_resume().\n",
           (int)internal_getpid());
    while (__tsan_resumed == 0) {}
  }

  OnInitialize();
}

void MaybeSpawnBackgroundThread() {
  // On MIPS, TSan initialization is run before
  // __pthread_initialize_minimal_internal() is finished, so we can not spawn
  // new threads.
#if !SANITIZER_GO && !defined(__mips__)
  static atomic_uint32_t bg_thread = {};
  if (atomic_load(&bg_thread, memory_order_relaxed) == 0 &&
      atomic_exchange(&bg_thread, 1, memory_order_relaxed) == 0) {
    StartBackgroundThread();
    SetSandboxingCallback(StopBackgroundThread);
  }
#endif
}


int Finalize(ThreadState *thr) {
  bool failed = false;

  if (common_flags()->print_module_map == 1) PrintModuleMap();

  if (flags()->atexit_sleep_ms > 0 && ThreadCount(thr) > 1)
    SleepForMillis(flags()->atexit_sleep_ms);

  // Wait for pending reports.
  ctx->report_mtx.Lock();
  { ScopedErrorReportLock l; }
  ctx->report_mtx.Unlock();

#if !SANITIZER_GO
  if (Verbosity()) AllocatorPrintStats();
#endif

  ThreadFinalize(thr);

  if (ctx->nreported) {
    failed = true;
#if !SANITIZER_GO
    Printf("ThreadSanitizer: reported %d warnings\n", ctx->nreported);
#else
    Printf("Found %d data race(s)\n", ctx->nreported);
#endif
  }

  if (ctx->nmissed_expected) {
    failed = true;
    Printf("ThreadSanitizer: missed %d expected races\n",
        ctx->nmissed_expected);
  }

  if (common_flags()->print_suppressions)
    PrintMatchedSuppressions();
#if !SANITIZER_GO
  if (flags()->print_benign)
    PrintMatchedBenignRaces();
#endif

  failed = OnFinalize(failed);

#if TSAN_COLLECT_STATS
  StatAggregate(ctx->stat, thr->stat);
  StatOutput(ctx->stat);
#endif

  return failed ? common_flags()->exitcode : 0;
}

#if !SANITIZER_GO
void ForkBefore(ThreadState *thr, uptr pc) {
  ctx->thread_registry->Lock();
  ctx->report_mtx.Lock();
}

void ForkParentAfter(ThreadState *thr, uptr pc) {
  ctx->report_mtx.Unlock();
  ctx->thread_registry->Unlock();
}

void ForkChildAfter(ThreadState *thr, uptr pc) {
  ctx->report_mtx.Unlock();
  ctx->thread_registry->Unlock();

  uptr nthread = 0;
  ctx->thread_registry->GetNumberOfThreads(0, 0, &nthread /* alive threads */);
  VPrintf(1, "ThreadSanitizer: forked new process with pid %d,"
      " parent had %d threads\n", (int)internal_getpid(), (int)nthread);
  if (nthread == 1) {
    StartBackgroundThread();
  } else {
    // We've just forked a multi-threaded process. We cannot reasonably function
    // after that (some mutexes may be locked before fork). So just enable
    // ignores for everything in the hope that we will exec soon.
    ctx->after_multithreaded_fork = true;
    thr->ignore_interceptors++;
    ThreadIgnoreBegin(thr, pc);
    ThreadIgnoreSyncBegin(thr, pc);
  }
}
#endif

#if SANITIZER_GO
NOINLINE
void GrowShadowStack(ThreadState *thr) {
  const int sz = thr->shadow_stack_end - thr->shadow_stack;
  const int newsz = 2 * sz;
  uptr *newstack = (uptr*)internal_alloc(MBlockShadowStack,
      newsz * sizeof(uptr));
  internal_memcpy(newstack, thr->shadow_stack, sz * sizeof(uptr));
  internal_free(thr->shadow_stack);
  thr->shadow_stack = newstack;
  thr->shadow_stack_pos = newstack + sz;
  thr->shadow_stack_end = newstack + newsz;
}
#endif

u32 CurrentStackId(ThreadState *thr, uptr pc) {
  if (!thr->is_inited)  // May happen during bootstrap.
    return 0;
  if (pc != 0) {
#if !SANITIZER_GO
    DCHECK_LT(thr->shadow_stack_pos, thr->shadow_stack_end);
#else
    if (thr->shadow_stack_pos == thr->shadow_stack_end)
      GrowShadowStack(thr);
#endif
    thr->shadow_stack_pos[0] = pc;
    thr->shadow_stack_pos++;
  }
  u32 id = StackDepotPut(
      StackTrace(thr->shadow_stack, thr->shadow_stack_pos - thr->shadow_stack));
  if (pc != 0)
    thr->shadow_stack_pos--;
  return id;
}

void TraceSwitch(ThreadState *thr) {
#if !SANITIZER_GO
  if (ctx->after_multithreaded_fork)
    return;
#endif
  thr->nomalloc++;
  Trace *thr_trace = ThreadTrace(thr->tid);
  Lock l(&thr_trace->mtx);
  unsigned trace = (thr->fast_state.epoch() / kTracePartSize) % TraceParts();
  TraceHeader *hdr = &thr_trace->headers[trace];
  hdr->epoch0 = thr->fast_state.epoch();
  ObtainCurrentStack(thr, 0, &hdr->stack0);
  hdr->mset0 = thr->mset;
  thr->nomalloc--;
}

Trace *ThreadTrace(int tid) {
  return (Trace*)GetThreadTraceHeader(tid);
}

uptr TraceTopPC(ThreadState *thr) {
  Event *events = (Event*)GetThreadTrace(thr->tid);
  uptr pc = events[thr->fast_state.GetTracePos()];
  return pc;
}

uptr TraceSize() {
  return (uptr)(1ull << (kTracePartSizeBits + flags()->history_size + 1));
}

uptr TraceParts() {
  return TraceSize() / kTracePartSize;
}

#if !SANITIZER_GO
extern "C" void __tsan_trace_switch() {
  TraceSwitch(cur_thread());
}

extern "C" void __tsan_report_race() {
  ReportRace(cur_thread());
}
#endif

ALWAYS_INLINE
Shadow LoadShadow(u64 *p) {
  u64 raw = atomic_load((atomic_uint64_t*)p, memory_order_relaxed);
  return Shadow(raw);
}

ALWAYS_INLINE
void StoreShadow(u64 *sp, u64 s) {
  atomic_store((atomic_uint64_t*)sp, s, memory_order_relaxed);
}

ALWAYS_INLINE
void StoreIfNotYetStored(u64 *sp, u64 *s) {
  StoreShadow(sp, *s);
  *s = 0;
}

ALWAYS_INLINE
void HandleRace(ThreadState *thr, u64 *shadow_mem,
                              Shadow cur, Shadow old) {
  thr->racy_state[0] = cur.raw();
  thr->racy_state[1] = old.raw();
  thr->racy_shadow_addr = shadow_mem;
#if !SANITIZER_GO
  HACKY_CALL(__tsan_report_race);
#else
  ReportRace(thr);
#endif
}

static inline bool HappensBefore(Shadow old, ThreadState *thr) {
  return thr->clock.get(old.TidWithIgnore()) >= old.epoch();
}

ALWAYS_INLINE
void MemoryAccessImpl1(ThreadState *thr, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite, bool kIsAtomic,
    u64 *shadow_mem, Shadow cur) {
  StatInc(thr, StatMop);
  StatInc(thr, kAccessIsWrite ? StatMopWrite : StatMopRead);
  StatInc(thr, (StatType)(StatMop1 + kAccessSizeLog));

  // This potentially can live in an MMX/SSE scratch register.
  // The required intrinsics are:
  // __m128i _mm_move_epi64(__m128i*);
  // _mm_storel_epi64(u64*, __m128i);
  u64 store_word = cur.raw();

  // scan all the shadow values and dispatch to 4 categories:
  // same, replace, candidate and race (see comments below).
  // we consider only 3 cases regarding access sizes:
  // equal, intersect and not intersect. initially I considered
  // larger and smaller as well, it allowed to replace some
  // 'candidates' with 'same' or 'replace', but I think
  // it's just not worth it (performance- and complexity-wise).

  Shadow old(0);

  // It release mode we manually unroll the loop,
  // because empirically gcc generates better code this way.
  // However, we can't afford unrolling in debug mode, because the function
  // consumes almost 4K of stack. Gtest gives only 4K of stack to death test
  // threads, which is not enough for the unrolled loop.
#if SANITIZER_DEBUG
  for (int idx = 0; idx < 4; idx++) {
#include "tsan_update_shadow_word_inl.h"
  }
#else
  int idx = 0;
#include "tsan_update_shadow_word_inl.h"
  idx = 1;
#include "tsan_update_shadow_word_inl.h"
  idx = 2;
#include "tsan_update_shadow_word_inl.h"
  idx = 3;
#include "tsan_update_shadow_word_inl.h"
#endif

  // we did not find any races and had already stored
  // the current access info, so we are done
  if (LIKELY(store_word == 0))
    return;
  // choose a random candidate slot and replace it
  StoreShadow(shadow_mem + (cur.epoch() % kShadowCnt), store_word);
  StatInc(thr, StatShadowReplace);
  return;
 RACE:
  HandleRace(thr, shadow_mem, cur, old);
  return;
}

void UnalignedMemoryAccess(ThreadState *thr, uptr pc, uptr addr,
    int size, bool kAccessIsWrite, bool kIsAtomic) {
  while (size) {
    int size1 = 1;
    int kAccessSizeLog = kSizeLog1;
    if (size >= 8 && (addr & ~7) == ((addr + 7) & ~7)) {
      size1 = 8;
      kAccessSizeLog = kSizeLog8;
    } else if (size >= 4 && (addr & ~7) == ((addr + 3) & ~7)) {
      size1 = 4;
      kAccessSizeLog = kSizeLog4;
    } else if (size >= 2 && (addr & ~7) == ((addr + 1) & ~7)) {
      size1 = 2;
      kAccessSizeLog = kSizeLog2;
    }
    MemoryAccess(thr, pc, addr, kAccessSizeLog, kAccessIsWrite, kIsAtomic);
    addr += size1;
    size -= size1;
  }
}

ALWAYS_INLINE
bool ContainsSameAccessSlow(u64 *s, u64 a, u64 sync_epoch, bool is_write) {
  Shadow cur(a);
  for (uptr i = 0; i < kShadowCnt; i++) {
    Shadow old(LoadShadow(&s[i]));
    if (Shadow::Addr0AndSizeAreEqual(cur, old) &&
        old.TidWithIgnore() == cur.TidWithIgnore() &&
        old.epoch() > sync_epoch &&
        old.IsAtomic() == cur.IsAtomic() &&
        old.IsRead() <= cur.IsRead())
      return true;
  }
  return false;
}

#if defined(__SSE3__)
#define SHUF(v0, v1, i0, i1, i2, i3) _mm_castps_si128(_mm_shuffle_ps( \
    _mm_castsi128_ps(v0), _mm_castsi128_ps(v1), \
    (i0)*1 + (i1)*4 + (i2)*16 + (i3)*64))
ALWAYS_INLINE
bool ContainsSameAccessFast(u64 *s, u64 a, u64 sync_epoch, bool is_write) {
  // This is an optimized version of ContainsSameAccessSlow.
  // load current access into access[0:63]
  const m128 access     = _mm_cvtsi64_si128(a);
  // duplicate high part of access in addr0:
  // addr0[0:31]        = access[32:63]
  // addr0[32:63]       = access[32:63]
  // addr0[64:95]       = access[32:63]
  // addr0[96:127]      = access[32:63]
  const m128 addr0      = SHUF(access, access, 1, 1, 1, 1);
  // load 4 shadow slots
  const m128 shadow0    = _mm_load_si128((__m128i*)s);
  const m128 shadow1    = _mm_load_si128((__m128i*)s + 1);
  // load high parts of 4 shadow slots into addr_vect:
  // addr_vect[0:31]    = shadow0[32:63]
  // addr_vect[32:63]   = shadow0[96:127]
  // addr_vect[64:95]   = shadow1[32:63]
  // addr_vect[96:127]  = shadow1[96:127]
  m128 addr_vect        = SHUF(shadow0, shadow1, 1, 3, 1, 3);
  if (!is_write) {
    // set IsRead bit in addr_vect
    const m128 rw_mask1 = _mm_cvtsi64_si128(1<<15);
    const m128 rw_mask  = SHUF(rw_mask1, rw_mask1, 0, 0, 0, 0);
    addr_vect           = _mm_or_si128(addr_vect, rw_mask);
  }
  // addr0 == addr_vect?
  const m128 addr_res   = _mm_cmpeq_epi32(addr0, addr_vect);
  // epoch1[0:63]       = sync_epoch
  const m128 epoch1     = _mm_cvtsi64_si128(sync_epoch);
  // epoch[0:31]        = sync_epoch[0:31]
  // epoch[32:63]       = sync_epoch[0:31]
  // epoch[64:95]       = sync_epoch[0:31]
  // epoch[96:127]      = sync_epoch[0:31]
  const m128 epoch      = SHUF(epoch1, epoch1, 0, 0, 0, 0);
  // load low parts of shadow cell epochs into epoch_vect:
  // epoch_vect[0:31]   = shadow0[0:31]
  // epoch_vect[32:63]  = shadow0[64:95]
  // epoch_vect[64:95]  = shadow1[0:31]
  // epoch_vect[96:127] = shadow1[64:95]
  const m128 epoch_vect = SHUF(shadow0, shadow1, 0, 2, 0, 2);
  // epoch_vect >= sync_epoch?
  const m128 epoch_res  = _mm_cmpgt_epi32(epoch_vect, epoch);
  // addr_res & epoch_res
  const m128 res        = _mm_and_si128(addr_res, epoch_res);
  // mask[0] = res[7]
  // mask[1] = res[15]
  // ...
  // mask[15] = res[127]
  const int mask        = _mm_movemask_epi8(res);
  return mask != 0;
}
#endif

ALWAYS_INLINE
bool ContainsSameAccess(u64 *s, u64 a, u64 sync_epoch, bool is_write) {
#if defined(__SSE3__)
  bool res = ContainsSameAccessFast(s, a, sync_epoch, is_write);
  // NOTE: this check can fail if the shadow is concurrently mutated
  // by other threads. But it still can be useful if you modify
  // ContainsSameAccessFast and want to ensure that it's not completely broken.
  // DCHECK_EQ(res, ContainsSameAccessSlow(s, a, sync_epoch, is_write));
  return res;
#else
  return ContainsSameAccessSlow(s, a, sync_epoch, is_write);
#endif
}

ALWAYS_INLINE USED
void MemoryAccess(ThreadState *thr, uptr pc, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite, bool kIsAtomic) {
  u64 *shadow_mem = (u64*)MemToShadow(addr);
  DPrintf2("#%d: MemoryAccess: @%p %p size=%d"
      " is_write=%d shadow_mem=%p {%zx, %zx, %zx, %zx}\n",
      (int)thr->fast_state.tid(), (void*)pc, (void*)addr,
      (int)(1 << kAccessSizeLog), kAccessIsWrite, shadow_mem,
      (uptr)shadow_mem[0], (uptr)shadow_mem[1],
      (uptr)shadow_mem[2], (uptr)shadow_mem[3]);
#if SANITIZER_DEBUG
  if (!IsAppMem(addr)) {
    Printf("Access to non app mem %zx\n", addr);
    DCHECK(IsAppMem(addr));
  }
  if (!IsShadowMem((uptr)shadow_mem)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem, addr);
    DCHECK(IsShadowMem((uptr)shadow_mem));
  }
#endif

  if (!SANITIZER_GO && *shadow_mem == kShadowRodata) {
    // Access to .rodata section, no races here.
    // Measurements show that it can be 10-20% of all memory accesses.
    StatInc(thr, StatMop);
    StatInc(thr, kAccessIsWrite ? StatMopWrite : StatMopRead);
    StatInc(thr, (StatType)(StatMop1 + kAccessSizeLog));
    StatInc(thr, StatMopRodata);
    return;
  }

  FastState fast_state = thr->fast_state;
  if (fast_state.GetIgnoreBit()) {
    StatInc(thr, StatMop);
    StatInc(thr, kAccessIsWrite ? StatMopWrite : StatMopRead);
    StatInc(thr, (StatType)(StatMop1 + kAccessSizeLog));
    StatInc(thr, StatMopIgnored);
    return;
  }

  Shadow cur(fast_state);
  cur.SetAddr0AndSizeLog(addr & 7, kAccessSizeLog);
  cur.SetWrite(kAccessIsWrite);
  cur.SetAtomic(kIsAtomic);

  if (LIKELY(ContainsSameAccess(shadow_mem, cur.raw(),
      thr->fast_synch_epoch, kAccessIsWrite))) {
    StatInc(thr, StatMop);
    StatInc(thr, kAccessIsWrite ? StatMopWrite : StatMopRead);
    StatInc(thr, (StatType)(StatMop1 + kAccessSizeLog));
    StatInc(thr, StatMopSame);
    return;
  }

  if (kCollectHistory) {
    fast_state.IncrementEpoch();
    thr->fast_state = fast_state;
    TraceAddEvent(thr, fast_state, EventTypeMop, pc);
    cur.IncrementEpoch();
  }

  MemoryAccessImpl1(thr, addr, kAccessSizeLog, kAccessIsWrite, kIsAtomic,
      shadow_mem, cur);
}

// Called by MemoryAccessRange in tsan_rtl_thread.cc
ALWAYS_INLINE USED
void MemoryAccessImpl(ThreadState *thr, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite, bool kIsAtomic,
    u64 *shadow_mem, Shadow cur) {
  if (LIKELY(ContainsSameAccess(shadow_mem, cur.raw(),
      thr->fast_synch_epoch, kAccessIsWrite))) {
    StatInc(thr, StatMop);
    StatInc(thr, kAccessIsWrite ? StatMopWrite : StatMopRead);
    StatInc(thr, (StatType)(StatMop1 + kAccessSizeLog));
    StatInc(thr, StatMopSame);
    return;
  }

  MemoryAccessImpl1(thr, addr, kAccessSizeLog, kAccessIsWrite, kIsAtomic,
      shadow_mem, cur);
}

static void MemoryRangeSet(ThreadState *thr, uptr pc, uptr addr, uptr size,
                           u64 val) {
  (void)thr;
  (void)pc;
  if (size == 0)
    return;
  // FIXME: fix me.
  uptr offset = addr % kShadowCell;
  if (offset) {
    offset = kShadowCell - offset;
    if (size <= offset)
      return;
    addr += offset;
    size -= offset;
  }
  DCHECK_EQ(addr % 8, 0);
  // If a user passes some insane arguments (memset(0)),
  // let it just crash as usual.
  if (!IsAppMem(addr) || !IsAppMem(addr + size - 1))
    return;
  // Don't want to touch lots of shadow memory.
  // If a program maps 10MB stack, there is no need reset the whole range.
  size = (size + (kShadowCell - 1)) & ~(kShadowCell - 1);
  // UnmapOrDie/MmapFixedNoReserve does not work on Windows.
  if (SANITIZER_WINDOWS || size < common_flags()->clear_shadow_mmap_threshold) {
    u64 *p = (u64*)MemToShadow(addr);
    CHECK(IsShadowMem((uptr)p));
    CHECK(IsShadowMem((uptr)(p + size * kShadowCnt / kShadowCell - 1)));
    // FIXME: may overwrite a part outside the region
    for (uptr i = 0; i < size / kShadowCell * kShadowCnt;) {
      p[i++] = val;
      for (uptr j = 1; j < kShadowCnt; j++)
        p[i++] = 0;
    }
  } else {
    // The region is big, reset only beginning and end.
    const uptr kPageSize = GetPageSizeCached();
    u64 *begin = (u64*)MemToShadow(addr);
    u64 *end = begin + size / kShadowCell * kShadowCnt;
    u64 *p = begin;
    // Set at least first kPageSize/2 to page boundary.
    while ((p < begin + kPageSize / kShadowSize / 2) || ((uptr)p % kPageSize)) {
      *p++ = val;
      for (uptr j = 1; j < kShadowCnt; j++)
        *p++ = 0;
    }
    // Reset middle part.
    u64 *p1 = p;
    p = RoundDown(end, kPageSize);
    UnmapOrDie((void*)p1, (uptr)p - (uptr)p1);
    MmapFixedNoReserve((uptr)p1, (uptr)p - (uptr)p1);
    // Set the ending.
    while (p < end) {
      *p++ = val;
      for (uptr j = 1; j < kShadowCnt; j++)
        *p++ = 0;
    }
  }
}

void MemoryResetRange(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  MemoryRangeSet(thr, pc, addr, size, 0);
}

void MemoryRangeFreed(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  // Processing more than 1k (4k of shadow) is expensive,
  // can cause excessive memory consumption (user does not necessary touch
  // the whole range) and most likely unnecessary.
  if (size > 1024)
    size = 1024;
  CHECK_EQ(thr->is_freeing, false);
  thr->is_freeing = true;
  MemoryAccessRange(thr, pc, addr, size, true);
  thr->is_freeing = false;
  if (kCollectHistory) {
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeMop, pc);
  }
  Shadow s(thr->fast_state);
  s.ClearIgnoreBit();
  s.MarkAsFreed();
  s.SetWrite(true);
  s.SetAddr0AndSizeLog(0, 3);
  MemoryRangeSet(thr, pc, addr, size, s.raw());
}

void MemoryRangeImitateWrite(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  if (kCollectHistory) {
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeMop, pc);
  }
  Shadow s(thr->fast_state);
  s.ClearIgnoreBit();
  s.SetWrite(true);
  s.SetAddr0AndSizeLog(0, 3);
  MemoryRangeSet(thr, pc, addr, size, s.raw());
}

ALWAYS_INLINE USED
void FuncEntry(ThreadState *thr, uptr pc) {
  StatInc(thr, StatFuncEnter);
  DPrintf2("#%d: FuncEntry %p\n", (int)thr->fast_state.tid(), (void*)pc);
  if (kCollectHistory) {
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeFuncEnter, pc);
  }

  // Shadow stack maintenance can be replaced with
  // stack unwinding during trace switch (which presumably must be faster).
  DCHECK_GE(thr->shadow_stack_pos, thr->shadow_stack);
#if !SANITIZER_GO
  DCHECK_LT(thr->shadow_stack_pos, thr->shadow_stack_end);
#else
  if (thr->shadow_stack_pos == thr->shadow_stack_end)
    GrowShadowStack(thr);
#endif
  thr->shadow_stack_pos[0] = pc;
  thr->shadow_stack_pos++;
}

ALWAYS_INLINE USED
void FuncExit(ThreadState *thr) {
  StatInc(thr, StatFuncExit);
  DPrintf2("#%d: FuncExit\n", (int)thr->fast_state.tid());
  if (kCollectHistory) {
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeFuncExit, 0);
  }

  DCHECK_GT(thr->shadow_stack_pos, thr->shadow_stack);
#if !SANITIZER_GO
  DCHECK_LT(thr->shadow_stack_pos, thr->shadow_stack_end);
#endif
  thr->shadow_stack_pos--;
}

void ThreadIgnoreBegin(ThreadState *thr, uptr pc, bool save_stack) {
  DPrintf("#%d: ThreadIgnoreBegin\n", thr->tid);
  thr->ignore_reads_and_writes++;
  CHECK_GT(thr->ignore_reads_and_writes, 0);
  thr->fast_state.SetIgnoreBit();
#if !SANITIZER_GO
  if (save_stack && !ctx->after_multithreaded_fork)
    thr->mop_ignore_set.Add(CurrentStackId(thr, pc));
#endif
}

void ThreadIgnoreEnd(ThreadState *thr, uptr pc) {
  DPrintf("#%d: ThreadIgnoreEnd\n", thr->tid);
  CHECK_GT(thr->ignore_reads_and_writes, 0);
  thr->ignore_reads_and_writes--;
  if (thr->ignore_reads_and_writes == 0) {
    thr->fast_state.ClearIgnoreBit();
#if !SANITIZER_GO
    thr->mop_ignore_set.Reset();
#endif
  }
}

#if !SANITIZER_GO
extern "C" SANITIZER_INTERFACE_ATTRIBUTE
uptr __tsan_testonly_shadow_stack_current_size() {
  ThreadState *thr = cur_thread();
  return thr->shadow_stack_pos - thr->shadow_stack;
}
#endif

void ThreadIgnoreSyncBegin(ThreadState *thr, uptr pc, bool save_stack) {
  DPrintf("#%d: ThreadIgnoreSyncBegin\n", thr->tid);
  thr->ignore_sync++;
  CHECK_GT(thr->ignore_sync, 0);
#if !SANITIZER_GO
  if (save_stack && !ctx->after_multithreaded_fork)
    thr->sync_ignore_set.Add(CurrentStackId(thr, pc));
#endif
}

void ThreadIgnoreSyncEnd(ThreadState *thr, uptr pc) {
  DPrintf("#%d: ThreadIgnoreSyncEnd\n", thr->tid);
  CHECK_GT(thr->ignore_sync, 0);
  thr->ignore_sync--;
#if !SANITIZER_GO
  if (thr->ignore_sync == 0)
    thr->sync_ignore_set.Reset();
#endif
}

bool MD5Hash::operator==(const MD5Hash &other) const {
  return hash[0] == other.hash[0] && hash[1] == other.hash[1];
}

#if SANITIZER_DEBUG
void build_consistency_debug() {}
#else
void build_consistency_release() {}
#endif

#if TSAN_COLLECT_STATS
void build_consistency_stats() {}
#else
void build_consistency_nostats() {}
#endif

}  // namespace __tsan

#if !SANITIZER_GO
// Must be included in this file to make sure everything is inlined.
#include "tsan_interface_inl.h"
#endif
//===-- tsan_rtl_mutex.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include <sanitizer_common/sanitizer_deadlock_detector_interface.h>
#include <sanitizer_common/sanitizer_stackdepot.h>

#include "tsan_rtl.h"
#include "tsan_flags.h"
#include "tsan_sync.h"
#include "tsan_report.h"
#include "tsan_symbolize.h"
#include "tsan_platform.h"

namespace __tsan {

void ReportDeadlock(ThreadState *thr, uptr pc, DDReport *r);

struct Callback : DDCallback {
  ThreadState *thr;
  uptr pc;

  Callback(ThreadState *thr, uptr pc)
      : thr(thr)
      , pc(pc) {
    DDCallback::pt = thr->proc()->dd_pt;
    DDCallback::lt = thr->dd_lt;
  }

  u32 Unwind() override { return CurrentStackId(thr, pc); }
  int UniqueTid() override { return thr->unique_id; }
};

void DDMutexInit(ThreadState *thr, uptr pc, SyncVar *s) {
  Callback cb(thr, pc);
  ctx->dd->MutexInit(&cb, &s->dd);
  s->dd.ctx = s->GetId();
}

static void ReportMutexMisuse(ThreadState *thr, uptr pc, ReportType typ,
    uptr addr, u64 mid) {
  // In Go, these misuses are either impossible, or detected by std lib,
  // or false positives (e.g. unlock in a different thread).
  if (SANITIZER_GO)
    return;
  ThreadRegistryLock l(ctx->thread_registry);
  ScopedReport rep(typ);
  rep.AddMutex(mid);
  VarSizeStackTrace trace;
  ObtainCurrentStack(thr, pc, &trace);
  rep.AddStack(trace, true);
  rep.AddLocation(addr, 1);
  OutputReport(thr, rep);
}

void MutexCreate(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexCreate %zx flagz=0x%x\n", thr->tid, addr, flagz);
  StatInc(thr, StatMutexCreate);
  if (!(flagz & MutexFlagLinkerInit) && IsAppMem(addr)) {
    CHECK(!thr->is_freeing);
    thr->is_freeing = true;
    MemoryWrite(thr, pc, addr, kSizeLog1);
    thr->is_freeing = false;
  }
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  s->SetFlags(flagz & MutexCreationFlagMask);
  if (!SANITIZER_GO && s->creation_stack_id == 0)
    s->creation_stack_id = CurrentStackId(thr, pc);
  s->mtx.Unlock();
}

void MutexDestroy(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexDestroy %zx\n", thr->tid, addr);
  StatInc(thr, StatMutexDestroy);
  SyncVar *s = ctx->metamap.GetIfExistsAndLock(addr, true);
  if (s == 0)
    return;
  if ((flagz & MutexFlagLinkerInit)
      || s->IsFlagSet(MutexFlagLinkerInit)
      || ((flagz & MutexFlagNotStatic) && !s->IsFlagSet(MutexFlagNotStatic))) {
    // Destroy is no-op for linker-initialized mutexes.
    s->mtx.Unlock();
    return;
  }
  if (common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ctx->dd->MutexDestroy(&cb, &s->dd);
    ctx->dd->MutexInit(&cb, &s->dd);
  }
  bool unlock_locked = false;
  if (flags()->report_destroy_locked
      && s->owner_tid != SyncVar::kInvalidTid
      && !s->IsFlagSet(MutexFlagBroken)) {
    s->SetFlags(MutexFlagBroken);
    unlock_locked = true;
  }
  u64 mid = s->GetId();
  u64 last_lock = s->last_lock;
  if (!unlock_locked)
    s->Reset(thr->proc());  // must not reset it before the report is printed
  s->mtx.Unlock();
  if (unlock_locked) {
    ThreadRegistryLock l(ctx->thread_registry);
    ScopedReport rep(ReportTypeMutexDestroyLocked);
    rep.AddMutex(mid);
    VarSizeStackTrace trace;
    ObtainCurrentStack(thr, pc, &trace);
    rep.AddStack(trace, true);
    FastState last(last_lock);
    RestoreStack(last.tid(), last.epoch(), &trace, 0);
    rep.AddStack(trace, true);
    rep.AddLocation(addr, 1);
    OutputReport(thr, rep);

    SyncVar *s = ctx->metamap.GetIfExistsAndLock(addr, true);
    if (s != 0) {
      s->Reset(thr->proc());
      s->mtx.Unlock();
    }
  }
  thr->mset.Remove(mid);
  // Imitate a memory write to catch unlock-destroy races.
  // Do this outside of sync mutex, because it can report a race which locks
  // sync mutexes.
  if (IsAppMem(addr)) {
    CHECK(!thr->is_freeing);
    thr->is_freeing = true;
    MemoryWrite(thr, pc, addr, kSizeLog1);
    thr->is_freeing = false;
  }
  // s will be destroyed and freed in MetaMap::FreeBlock.
}

void MutexPreLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexPreLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (!(flagz & MutexFlagTryLock) && common_flags()->detect_deadlocks) {
    SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, false);
    s->UpdateFlags(flagz);
    if (s->owner_tid != thr->tid) {
      Callback cb(thr, pc);
      ctx->dd->MutexBeforeLock(&cb, &s->dd, true);
      s->mtx.ReadUnlock();
      ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
    } else {
      s->mtx.ReadUnlock();
    }
  }
}

void MutexPostLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz, int rec) {
  DPrintf("#%d: MutexPostLock %zx flag=0x%x rec=%d\n",
      thr->tid, addr, flagz, rec);
  if (flagz & MutexFlagRecursiveLock)
    CHECK_GT(rec, 0);
  else
    rec = 1;
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  s->UpdateFlags(flagz);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeLock, s->GetId());
  bool report_double_lock = false;
  if (s->owner_tid == SyncVar::kInvalidTid) {
    CHECK_EQ(s->recursion, 0);
    s->owner_tid = thr->tid;
    s->last_lock = thr->fast_state.raw();
  } else if (s->owner_tid == thr->tid) {
    CHECK_GT(s->recursion, 0);
  } else if (flags()->report_mutex_bugs && !s->IsFlagSet(MutexFlagBroken)) {
    s->SetFlags(MutexFlagBroken);
    report_double_lock = true;
  }
  const bool first = s->recursion == 0;
  s->recursion += rec;
  if (first) {
    StatInc(thr, StatMutexLock);
    AcquireImpl(thr, pc, &s->clock);
    AcquireImpl(thr, pc, &s->read_clock);
  } else if (!s->IsFlagSet(MutexFlagWriteReentrant)) {
    StatInc(thr, StatMutexRecLock);
  }
  thr->mset.Add(s->GetId(), true, thr->fast_state.epoch());
  bool pre_lock = false;
  if (first && common_flags()->detect_deadlocks) {
    pre_lock = (flagz & MutexFlagDoPreLockOnPostLock) &&
        !(flagz & MutexFlagTryLock);
    Callback cb(thr, pc);
    if (pre_lock)
      ctx->dd->MutexBeforeLock(&cb, &s->dd, true);
    ctx->dd->MutexAfterLock(&cb, &s->dd, true, flagz & MutexFlagTryLock);
  }
  u64 mid = s->GetId();
  s->mtx.Unlock();
  // Can't touch s after this point.
  s = 0;
  if (report_double_lock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexDoubleLock, addr, mid);
  if (first && pre_lock && common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

int MutexUnlock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexUnlock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeUnlock, s->GetId());
  int rec = 0;
  bool report_bad_unlock = false;
  if (!SANITIZER_GO && (s->recursion == 0 || s->owner_tid != thr->tid)) {
    if (flags()->report_mutex_bugs && !s->IsFlagSet(MutexFlagBroken)) {
      s->SetFlags(MutexFlagBroken);
      report_bad_unlock = true;
    }
  } else {
    rec = (flagz & MutexFlagRecursiveUnlock) ? s->recursion : 1;
    s->recursion -= rec;
    if (s->recursion == 0) {
      StatInc(thr, StatMutexUnlock);
      s->owner_tid = SyncVar::kInvalidTid;
      ReleaseStoreImpl(thr, pc, &s->clock);
    } else {
      StatInc(thr, StatMutexRecUnlock);
    }
  }
  thr->mset.Del(s->GetId(), true);
  if (common_flags()->detect_deadlocks && s->recursion == 0 &&
      !report_bad_unlock) {
    Callback cb(thr, pc);
    ctx->dd->MutexBeforeUnlock(&cb, &s->dd, true);
  }
  u64 mid = s->GetId();
  s->mtx.Unlock();
  // Can't touch s after this point.
  if (report_bad_unlock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexBadUnlock, addr, mid);
  if (common_flags()->detect_deadlocks && !report_bad_unlock) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
  return rec;
}

void MutexPreReadLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexPreReadLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (!(flagz & MutexFlagTryLock) && common_flags()->detect_deadlocks) {
    SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, false);
    s->UpdateFlags(flagz);
    Callback cb(thr, pc);
    ctx->dd->MutexBeforeLock(&cb, &s->dd, false);
    s->mtx.ReadUnlock();
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

void MutexPostReadLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexPostReadLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  StatInc(thr, StatMutexReadLock);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, false);
  s->UpdateFlags(flagz);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeRLock, s->GetId());
  bool report_bad_lock = false;
  if (s->owner_tid != SyncVar::kInvalidTid) {
    if (flags()->report_mutex_bugs && !s->IsFlagSet(MutexFlagBroken)) {
      s->SetFlags(MutexFlagBroken);
      report_bad_lock = true;
    }
  }
  AcquireImpl(thr, pc, &s->clock);
  s->last_lock = thr->fast_state.raw();
  thr->mset.Add(s->GetId(), false, thr->fast_state.epoch());
  bool pre_lock = false;
  if (common_flags()->detect_deadlocks) {
    pre_lock = (flagz & MutexFlagDoPreLockOnPostLock) &&
        !(flagz & MutexFlagTryLock);
    Callback cb(thr, pc);
    if (pre_lock)
      ctx->dd->MutexBeforeLock(&cb, &s->dd, false);
    ctx->dd->MutexAfterLock(&cb, &s->dd, false, flagz & MutexFlagTryLock);
  }
  u64 mid = s->GetId();
  s->mtx.ReadUnlock();
  // Can't touch s after this point.
  s = 0;
  if (report_bad_lock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexBadReadLock, addr, mid);
  if (pre_lock  && common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

void MutexReadUnlock(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexReadUnlock %zx\n", thr->tid, addr);
  StatInc(thr, StatMutexReadUnlock);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeRUnlock, s->GetId());
  bool report_bad_unlock = false;
  if (s->owner_tid != SyncVar::kInvalidTid) {
    if (flags()->report_mutex_bugs && !s->IsFlagSet(MutexFlagBroken)) {
      s->SetFlags(MutexFlagBroken);
      report_bad_unlock = true;
    }
  }
  ReleaseImpl(thr, pc, &s->read_clock);
  if (common_flags()->detect_deadlocks && s->recursion == 0) {
    Callback cb(thr, pc);
    ctx->dd->MutexBeforeUnlock(&cb, &s->dd, false);
  }
  u64 mid = s->GetId();
  s->mtx.Unlock();
  // Can't touch s after this point.
  thr->mset.Del(mid, false);
  if (report_bad_unlock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexBadReadUnlock, addr, mid);
  if (common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

void MutexReadOrWriteUnlock(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexReadOrWriteUnlock %zx\n", thr->tid, addr);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  bool write = true;
  bool report_bad_unlock = false;
  if (s->owner_tid == SyncVar::kInvalidTid) {
    // Seems to be read unlock.
    write = false;
    StatInc(thr, StatMutexReadUnlock);
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeRUnlock, s->GetId());
    ReleaseImpl(thr, pc, &s->read_clock);
  } else if (s->owner_tid == thr->tid) {
    // Seems to be write unlock.
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeUnlock, s->GetId());
    CHECK_GT(s->recursion, 0);
    s->recursion--;
    if (s->recursion == 0) {
      StatInc(thr, StatMutexUnlock);
      s->owner_tid = SyncVar::kInvalidTid;
      ReleaseImpl(thr, pc, &s->clock);
    } else {
      StatInc(thr, StatMutexRecUnlock);
    }
  } else if (!s->IsFlagSet(MutexFlagBroken)) {
    s->SetFlags(MutexFlagBroken);
    report_bad_unlock = true;
  }
  thr->mset.Del(s->GetId(), write);
  if (common_flags()->detect_deadlocks && s->recursion == 0) {
    Callback cb(thr, pc);
    ctx->dd->MutexBeforeUnlock(&cb, &s->dd, write);
  }
  u64 mid = s->GetId();
  s->mtx.Unlock();
  // Can't touch s after this point.
  if (report_bad_unlock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexBadUnlock, addr, mid);
  if (common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

void MutexRepair(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexRepair %zx\n", thr->tid, addr);
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  s->owner_tid = SyncVar::kInvalidTid;
  s->recursion = 0;
  s->mtx.Unlock();
}

void MutexInvalidAccess(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexInvalidAccess %zx\n", thr->tid, addr);
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  u64 mid = s->GetId();
  s->mtx.Unlock();
  ReportMutexMisuse(thr, pc, ReportTypeMutexInvalidAccess, addr, mid);
}

void Acquire(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: Acquire %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SyncVar *s = ctx->metamap.GetIfExistsAndLock(addr, false);
  if (!s)
    return;
  AcquireImpl(thr, pc, &s->clock);
  s->mtx.ReadUnlock();
}

static void UpdateClockCallback(ThreadContextBase *tctx_base, void *arg) {
  ThreadState *thr = reinterpret_cast<ThreadState*>(arg);
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  u64 epoch = tctx->epoch1;
  if (tctx->status == ThreadStatusRunning)
    epoch = tctx->thr->fast_state.epoch();
  thr->clock.set(&thr->proc()->clock_cache, tctx->tid, epoch);
}

void AcquireGlobal(ThreadState *thr, uptr pc) {
  DPrintf("#%d: AcquireGlobal\n", thr->tid);
  if (thr->ignore_sync)
    return;
  ThreadRegistryLock l(ctx->thread_registry);
  ctx->thread_registry->RunCallbackForEachThreadLocked(
      UpdateClockCallback, thr);
}

void Release(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: Release %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  // Can't increment epoch w/o writing to the trace as well.
  TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
  ReleaseImpl(thr, pc, &s->clock);
  s->mtx.Unlock();
}

void ReleaseStore(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: ReleaseStore %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SyncVar *s = ctx->metamap.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  // Can't increment epoch w/o writing to the trace as well.
  TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
  ReleaseStoreImpl(thr, pc, &s->clock);
  s->mtx.Unlock();
}

#if !SANITIZER_GO
static void UpdateSleepClockCallback(ThreadContextBase *tctx_base, void *arg) {
  ThreadState *thr = reinterpret_cast<ThreadState*>(arg);
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  u64 epoch = tctx->epoch1;
  if (tctx->status == ThreadStatusRunning)
    epoch = tctx->thr->fast_state.epoch();
  thr->last_sleep_clock.set(&thr->proc()->clock_cache, tctx->tid, epoch);
}

void AfterSleep(ThreadState *thr, uptr pc) {
  DPrintf("#%d: AfterSleep %zx\n", thr->tid);
  if (thr->ignore_sync)
    return;
  thr->last_sleep_stack_id = CurrentStackId(thr, pc);
  ThreadRegistryLock l(ctx->thread_registry);
  ctx->thread_registry->RunCallbackForEachThreadLocked(
      UpdateSleepClockCallback, thr);
}
#endif

void AcquireImpl(ThreadState *thr, uptr pc, SyncClock *c) {
  if (thr->ignore_sync)
    return;
  thr->clock.set(thr->fast_state.epoch());
  thr->clock.acquire(&thr->proc()->clock_cache, c);
  StatInc(thr, StatSyncAcquire);
}

void ReleaseImpl(ThreadState *thr, uptr pc, SyncClock *c) {
  if (thr->ignore_sync)
    return;
  thr->clock.set(thr->fast_state.epoch());
  thr->fast_synch_epoch = thr->fast_state.epoch();
  thr->clock.release(&thr->proc()->clock_cache, c);
  StatInc(thr, StatSyncRelease);
}

void ReleaseStoreImpl(ThreadState *thr, uptr pc, SyncClock *c) {
  if (thr->ignore_sync)
    return;
  thr->clock.set(thr->fast_state.epoch());
  thr->fast_synch_epoch = thr->fast_state.epoch();
  thr->clock.ReleaseStore(&thr->proc()->clock_cache, c);
  StatInc(thr, StatSyncRelease);
}

void AcquireReleaseImpl(ThreadState *thr, uptr pc, SyncClock *c) {
  if (thr->ignore_sync)
    return;
  thr->clock.set(thr->fast_state.epoch());
  thr->fast_synch_epoch = thr->fast_state.epoch();
  thr->clock.acq_rel(&thr->proc()->clock_cache, c);
  StatInc(thr, StatSyncAcquire);
  StatInc(thr, StatSyncRelease);
}

void ReportDeadlock(ThreadState *thr, uptr pc, DDReport *r) {
  if (r == 0)
    return;
  ThreadRegistryLock l(ctx->thread_registry);
  ScopedReport rep(ReportTypeDeadlock);
  for (int i = 0; i < r->n; i++) {
    rep.AddMutex(r->loop[i].mtx_ctx0);
    rep.AddUniqueTid((int)r->loop[i].thr_ctx);
    rep.AddThread((int)r->loop[i].thr_ctx);
  }
  uptr dummy_pc = 0x42;
  for (int i = 0; i < r->n; i++) {
    for (int j = 0; j < (flags()->second_deadlock_stack ? 2 : 1); j++) {
      u32 stk = r->loop[i].stk[j];
      if (stk && stk != 0xffffffff) {
        rep.AddStack(StackDepotGet(stk), true);
      } else {
        // Sometimes we fail to extract the stack trace (FIXME: investigate),
        // but we should still produce some stack trace in the report.
        rep.AddStack(StackTrace(&dummy_pc, 1), true);
      }
    }
  }
  OutputReport(thr, rep);
}

}  // namespace __tsan
//===-- tsan_rtl_report.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_suppressions.h"
#include "tsan_symbolize.h"
#include "tsan_report.h"
#include "tsan_sync.h"
#include "tsan_mman.h"
#include "tsan_flags.h"
#include "tsan_fd.h"

namespace __tsan {

using namespace __sanitizer;  // NOLINT

static ReportStack *SymbolizeStack(StackTrace trace);

void TsanCheckFailed(const char *file, int line, const char *cond,
                     u64 v1, u64 v2) {
  // There is high probability that interceptors will check-fail as well,
  // on the other hand there is no sense in processing interceptors
  // since we are going to die soon.
  ScopedIgnoreInterceptors ignore;
#if !SANITIZER_GO
  cur_thread()->ignore_sync++;
  cur_thread()->ignore_reads_and_writes++;
#endif
  Printf("FATAL: ThreadSanitizer CHECK failed: "
         "%s:%d \"%s\" (0x%zx, 0x%zx)\n",
         file, line, cond, (uptr)v1, (uptr)v2);
  PrintCurrentStackSlow(StackTrace::GetCurrentPc());
  Die();
}

// Can be overriden by an application/test to intercept reports.
#ifdef TSAN_EXTERNAL_HOOKS
bool OnReport(const ReportDesc *rep, bool suppressed);
#else
SANITIZER_WEAK_CXX_DEFAULT_IMPL
bool OnReport(const ReportDesc *rep, bool suppressed) {
  (void)rep;
  return suppressed;
}
#endif

SANITIZER_WEAK_DEFAULT_IMPL
void __tsan_on_report(const ReportDesc *rep) {
  (void)rep;
}

static void StackStripMain(SymbolizedStack *frames) {
  SymbolizedStack *last_frame = nullptr;
  SymbolizedStack *last_frame2 = nullptr;
  for (SymbolizedStack *cur = frames; cur; cur = cur->next) {
    last_frame2 = last_frame;
    last_frame = cur;
  }

  if (last_frame2 == 0)
    return;
#if !SANITIZER_GO
  const char *last = last_frame->info.function;
  const char *last2 = last_frame2->info.function;
  // Strip frame above 'main'
  if (last2 && 0 == internal_strcmp(last2, "main")) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
  // Strip our internal thread start routine.
  } else if (last && 0 == internal_strcmp(last, "__tsan_thread_start_func")) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
  // Strip global ctors init.
  } else if (last && 0 == internal_strcmp(last, "__do_global_ctors_aux")) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
  // If both are 0, then we probably just failed to symbolize.
  } else if (last || last2) {
    // Ensure that we recovered stack completely. Trimmed stack
    // can actually happen if we do not instrument some code,
    // so it's only a debug print. However we must try hard to not miss it
    // due to our fault.
    DPrintf("Bottom stack frame is missed\n");
  }
#else
  // The last frame always point into runtime (gosched0, goexit0, runtime.main).
  last_frame->ClearAll();
  last_frame2->next = nullptr;
#endif
}

ReportStack *SymbolizeStackId(u32 stack_id) {
  if (stack_id == 0)
    return 0;
  StackTrace stack = StackDepotGet(stack_id);
  if (stack.trace == nullptr)
    return nullptr;
  return SymbolizeStack(stack);
}

static ReportStack *SymbolizeStack(StackTrace trace) {
  if (trace.size == 0)
    return 0;
  SymbolizedStack *top = nullptr;
  for (uptr si = 0; si < trace.size; si++) {
    const uptr pc = trace.trace[si];
    uptr pc1 = pc;
    // We obtain the return address, but we're interested in the previous
    // instruction.
    if ((pc & kExternalPCBit) == 0)
      pc1 = StackTrace::GetPreviousInstructionPc(pc);
    SymbolizedStack *ent = SymbolizeCode(pc1);
    CHECK_NE(ent, 0);
    SymbolizedStack *last = ent;
    while (last->next) {
      last->info.address = pc;  // restore original pc for report
      last = last->next;
    }
    last->info.address = pc;  // restore original pc for report
    last->next = top;
    top = ent;
  }
  StackStripMain(top);

  ReportStack *stack = ReportStack::New();
  stack->frames = top;
  return stack;
}

ScopedReportBase::ScopedReportBase(ReportType typ, uptr tag) {
  ctx->thread_registry->CheckLocked();
  void *mem = internal_alloc(MBlockReport, sizeof(ReportDesc));
  rep_ = new(mem) ReportDesc;
  rep_->typ = typ;
  rep_->tag = tag;
  ctx->report_mtx.Lock();
}

ScopedReportBase::~ScopedReportBase() {
  ctx->report_mtx.Unlock();
  DestroyAndFree(rep_);
}

void ScopedReportBase::AddStack(StackTrace stack, bool suppressable) {
  ReportStack **rs = rep_->stacks.PushBack();
  *rs = SymbolizeStack(stack);
  (*rs)->suppressable = suppressable;
}

void ScopedReportBase::AddMemoryAccess(uptr addr, uptr external_tag, Shadow s,
                                       StackTrace stack, const MutexSet *mset) {
  void *mem = internal_alloc(MBlockReportMop, sizeof(ReportMop));
  ReportMop *mop = new(mem) ReportMop;
  rep_->mops.PushBack(mop);
  mop->tid = s.tid();
  mop->addr = addr + s.addr0();
  mop->size = s.size();
  mop->write = s.IsWrite();
  mop->atomic = s.IsAtomic();
  mop->stack = SymbolizeStack(stack);
  mop->external_tag = external_tag;
  if (mop->stack)
    mop->stack->suppressable = true;
  for (uptr i = 0; i < mset->Size(); i++) {
    MutexSet::Desc d = mset->Get(i);
    u64 mid = this->AddMutex(d.id);
    ReportMopMutex mtx = {mid, d.write};
    mop->mset.PushBack(mtx);
  }
}

void ScopedReportBase::AddUniqueTid(int unique_tid) {
  rep_->unique_tids.PushBack(unique_tid);
}

void ScopedReportBase::AddThread(const ThreadContext *tctx, bool suppressable) {
  for (uptr i = 0; i < rep_->threads.Size(); i++) {
    if ((u32)rep_->threads[i]->id == tctx->tid)
      return;
  }
  void *mem = internal_alloc(MBlockReportThread, sizeof(ReportThread));
  ReportThread *rt = new(mem) ReportThread;
  rep_->threads.PushBack(rt);
  rt->id = tctx->tid;
  rt->os_id = tctx->os_id;
  rt->running = (tctx->status == ThreadStatusRunning);
  rt->name = internal_strdup(tctx->name);
  rt->parent_tid = tctx->parent_tid;
  rt->workerthread = tctx->workerthread;
  rt->stack = 0;
  rt->stack = SymbolizeStackId(tctx->creation_stack_id);
  if (rt->stack)
    rt->stack->suppressable = suppressable;
}

#if !SANITIZER_GO
static bool FindThreadByUidLockedCallback(ThreadContextBase *tctx, void *arg) {
  int unique_id = *(int *)arg;
  return tctx->unique_id == (u32)unique_id;
}

static ThreadContext *FindThreadByUidLocked(int unique_id) {
  ctx->thread_registry->CheckLocked();
  return static_cast<ThreadContext *>(
      ctx->thread_registry->FindThreadContextLocked(
          FindThreadByUidLockedCallback, &unique_id));
}

static ThreadContext *FindThreadByTidLocked(int tid) {
  ctx->thread_registry->CheckLocked();
  return static_cast<ThreadContext*>(
      ctx->thread_registry->GetThreadLocked(tid));
}

static bool IsInStackOrTls(ThreadContextBase *tctx_base, void *arg) {
  uptr addr = (uptr)arg;
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  if (tctx->status != ThreadStatusRunning)
    return false;
  ThreadState *thr = tctx->thr;
  CHECK(thr);
  return ((addr >= thr->stk_addr && addr < thr->stk_addr + thr->stk_size) ||
          (addr >= thr->tls_addr && addr < thr->tls_addr + thr->tls_size));
}

ThreadContext *IsThreadStackOrTls(uptr addr, bool *is_stack) {
  ctx->thread_registry->CheckLocked();
  ThreadContext *tctx = static_cast<ThreadContext*>(
      ctx->thread_registry->FindThreadContextLocked(IsInStackOrTls,
                                                    (void*)addr));
  if (!tctx)
    return 0;
  ThreadState *thr = tctx->thr;
  CHECK(thr);
  *is_stack = (addr >= thr->stk_addr && addr < thr->stk_addr + thr->stk_size);
  return tctx;
}
#endif

void ScopedReportBase::AddThread(int unique_tid, bool suppressable) {
#if !SANITIZER_GO
  if (const ThreadContext *tctx = FindThreadByUidLocked(unique_tid))
    AddThread(tctx, suppressable);
#endif
}

void ScopedReportBase::AddMutex(const SyncVar *s) {
  for (uptr i = 0; i < rep_->mutexes.Size(); i++) {
    if (rep_->mutexes[i]->id == s->uid)
      return;
  }
  void *mem = internal_alloc(MBlockReportMutex, sizeof(ReportMutex));
  ReportMutex *rm = new(mem) ReportMutex;
  rep_->mutexes.PushBack(rm);
  rm->id = s->uid;
  rm->addr = s->addr;
  rm->destroyed = false;
  rm->stack = SymbolizeStackId(s->creation_stack_id);
}

u64 ScopedReportBase::AddMutex(u64 id) {
  u64 uid = 0;
  u64 mid = id;
  uptr addr = SyncVar::SplitId(id, &uid);
  SyncVar *s = ctx->metamap.GetIfExistsAndLock(addr, true);
  // Check that the mutex is still alive.
  // Another mutex can be created at the same address,
  // so check uid as well.
  if (s && s->CheckId(uid)) {
    mid = s->uid;
    AddMutex(s);
  } else {
    AddDeadMutex(id);
  }
  if (s)
    s->mtx.Unlock();
  return mid;
}

void ScopedReportBase::AddDeadMutex(u64 id) {
  for (uptr i = 0; i < rep_->mutexes.Size(); i++) {
    if (rep_->mutexes[i]->id == id)
      return;
  }
  void *mem = internal_alloc(MBlockReportMutex, sizeof(ReportMutex));
  ReportMutex *rm = new(mem) ReportMutex;
  rep_->mutexes.PushBack(rm);
  rm->id = id;
  rm->addr = 0;
  rm->destroyed = true;
  rm->stack = 0;
}

void ScopedReportBase::AddLocation(uptr addr, uptr size) {
  if (addr == 0)
    return;
#if !SANITIZER_GO
  int fd = -1;
  int creat_tid = kInvalidTid;
  u32 creat_stack = 0;
  if (FdLocation(addr, &fd, &creat_tid, &creat_stack)) {
    ReportLocation *loc = ReportLocation::New(ReportLocationFD);
    loc->fd = fd;
    loc->tid = creat_tid;
    loc->stack = SymbolizeStackId(creat_stack);
    rep_->locs.PushBack(loc);
    ThreadContext *tctx = FindThreadByUidLocked(creat_tid);
    if (tctx)
      AddThread(tctx);
    return;
  }
  MBlock *b = 0;
  Allocator *a = allocator();
  if (a->PointerIsMine((void*)addr)) {
    void *block_begin = a->GetBlockBegin((void*)addr);
    if (block_begin)
      b = ctx->metamap.GetBlock((uptr)block_begin);
  }
  if (b != 0) {
    ThreadContext *tctx = FindThreadByTidLocked(b->tid);
    ReportLocation *loc = ReportLocation::New(ReportLocationHeap);
    loc->heap_chunk_start = (uptr)allocator()->GetBlockBegin((void *)addr);
    loc->heap_chunk_size = b->siz;
    loc->external_tag = b->tag;
    loc->tid = tctx ? tctx->tid : b->tid;
    loc->stack = SymbolizeStackId(b->stk);
    rep_->locs.PushBack(loc);
    if (tctx)
      AddThread(tctx);
    return;
  }
  bool is_stack = false;
  if (ThreadContext *tctx = IsThreadStackOrTls(addr, &is_stack)) {
    ReportLocation *loc =
        ReportLocation::New(is_stack ? ReportLocationStack : ReportLocationTLS);
    loc->tid = tctx->tid;
    rep_->locs.PushBack(loc);
    AddThread(tctx);
  }
#endif
  if (ReportLocation *loc = SymbolizeData(addr)) {
    loc->suppressable = true;
    rep_->locs.PushBack(loc);
    return;
  }
}

#if !SANITIZER_GO
void ScopedReportBase::AddSleep(u32 stack_id) {
  rep_->sleep = SymbolizeStackId(stack_id);
}
#endif

void ScopedReportBase::SetCount(int count) { rep_->count = count; }

const ReportDesc *ScopedReportBase::GetReport() const { return rep_; }

ScopedReport::ScopedReport(ReportType typ, uptr tag)
    : ScopedReportBase(typ, tag) {}

ScopedReport::~ScopedReport() {}

void RestoreStack(int tid, const u64 epoch, VarSizeStackTrace *stk,
                  MutexSet *mset, uptr *tag) {
  // This function restores stack trace and mutex set for the thread/epoch.
  // It does so by getting stack trace and mutex set at the beginning of
  // trace part, and then replaying the trace till the given epoch.
  Trace* trace = ThreadTrace(tid);
  ReadLock l(&trace->mtx);
  const int partidx = (epoch / kTracePartSize) % TraceParts();
  TraceHeader* hdr = &trace->headers[partidx];
  if (epoch < hdr->epoch0 || epoch >= hdr->epoch0 + kTracePartSize)
    return;
  CHECK_EQ(RoundDown(epoch, kTracePartSize), hdr->epoch0);
  const u64 epoch0 = RoundDown(epoch, TraceSize());
  const u64 eend = epoch % TraceSize();
  const u64 ebegin = RoundDown(eend, kTracePartSize);
  DPrintf("#%d: RestoreStack epoch=%zu ebegin=%zu eend=%zu partidx=%d\n",
          tid, (uptr)epoch, (uptr)ebegin, (uptr)eend, partidx);
  Vector<uptr> stack;
  stack.Resize(hdr->stack0.size + 64);
  for (uptr i = 0; i < hdr->stack0.size; i++) {
    stack[i] = hdr->stack0.trace[i];
    DPrintf2("  #%02zu: pc=%zx\n", i, stack[i]);
  }
  if (mset)
    *mset = hdr->mset0;
  uptr pos = hdr->stack0.size;
  Event *events = (Event*)GetThreadTrace(tid);
  for (uptr i = ebegin; i <= eend; i++) {
    Event ev = events[i];
    EventType typ = (EventType)(ev >> kEventPCBits);
    uptr pc = (uptr)(ev & ((1ull << kEventPCBits) - 1));
    DPrintf2("  %zu typ=%d pc=%zx\n", i, typ, pc);
    if (typ == EventTypeMop) {
      stack[pos] = pc;
    } else if (typ == EventTypeFuncEnter) {
      if (stack.Size() < pos + 2)
        stack.Resize(pos + 2);
      stack[pos++] = pc;
    } else if (typ == EventTypeFuncExit) {
      if (pos > 0)
        pos--;
    }
    if (mset) {
      if (typ == EventTypeLock) {
        mset->Add(pc, true, epoch0 + i);
      } else if (typ == EventTypeUnlock) {
        mset->Del(pc, true);
      } else if (typ == EventTypeRLock) {
        mset->Add(pc, false, epoch0 + i);
      } else if (typ == EventTypeRUnlock) {
        mset->Del(pc, false);
      }
    }
    for (uptr j = 0; j <= pos; j++)
      DPrintf2("      #%zu: %zx\n", j, stack[j]);
  }
  if (pos == 0 && stack[0] == 0)
    return;
  pos++;
  stk->Init(&stack[0], pos);
  ExtractTagFromStack(stk, tag);
}

static bool HandleRacyStacks(ThreadState *thr, VarSizeStackTrace traces[2],
                             uptr addr_min, uptr addr_max) {
  bool equal_stack = false;
  RacyStacks hash;
  bool equal_address = false;
  RacyAddress ra0 = {addr_min, addr_max};
  {
    ReadLock lock(&ctx->racy_mtx);
    if (flags()->suppress_equal_stacks) {
      hash.hash[0] = md5_hash(traces[0].trace, traces[0].size * sizeof(uptr));
      hash.hash[1] = md5_hash(traces[1].trace, traces[1].size * sizeof(uptr));
      for (uptr i = 0; i < ctx->racy_stacks.Size(); i++) {
        if (hash == ctx->racy_stacks[i]) {
          VPrintf(2,
              "ThreadSanitizer: suppressing report as doubled (stack)\n");
          equal_stack = true;
          break;
        }
      }
    }
    if (flags()->suppress_equal_addresses) {
      for (uptr i = 0; i < ctx->racy_addresses.Size(); i++) {
        RacyAddress ra2 = ctx->racy_addresses[i];
        uptr maxbeg = max(ra0.addr_min, ra2.addr_min);
        uptr minend = min(ra0.addr_max, ra2.addr_max);
        if (maxbeg < minend) {
          VPrintf(2, "ThreadSanitizer: suppressing report as doubled (addr)\n");
          equal_address = true;
          break;
        }
      }
    }
  }
  if (!equal_stack && !equal_address)
    return false;
  if (!equal_stack) {
    Lock lock(&ctx->racy_mtx);
    ctx->racy_stacks.PushBack(hash);
  }
  if (!equal_address) {
    Lock lock(&ctx->racy_mtx);
    ctx->racy_addresses.PushBack(ra0);
  }
  return true;
}

static void AddRacyStacks(ThreadState *thr, VarSizeStackTrace traces[2],
                          uptr addr_min, uptr addr_max) {
  Lock lock(&ctx->racy_mtx);
  if (flags()->suppress_equal_stacks) {
    RacyStacks hash;
    hash.hash[0] = md5_hash(traces[0].trace, traces[0].size * sizeof(uptr));
    hash.hash[1] = md5_hash(traces[1].trace, traces[1].size * sizeof(uptr));
    ctx->racy_stacks.PushBack(hash);
  }
  if (flags()->suppress_equal_addresses) {
    RacyAddress ra0 = {addr_min, addr_max};
    ctx->racy_addresses.PushBack(ra0);
  }
}

bool OutputReport(ThreadState *thr, const ScopedReport &srep) {
  if (!flags()->report_bugs || thr->suppress_reports)
    return false;
  atomic_store_relaxed(&ctx->last_symbolize_time_ns, NanoTime());
  const ReportDesc *rep = srep.GetReport();
  CHECK_EQ(thr->current_report, nullptr);
  thr->current_report = rep;
  Suppression *supp = 0;
  uptr pc_or_addr = 0;
  for (uptr i = 0; pc_or_addr == 0 && i < rep->mops.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->mops[i]->stack, &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->stacks.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->stacks[i], &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->threads.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->threads[i]->stack, &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->locs.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->locs[i], &supp);
  if (pc_or_addr != 0) {
    Lock lock(&ctx->fired_suppressions_mtx);
    FiredSuppression s = {srep.GetReport()->typ, pc_or_addr, supp};
    ctx->fired_suppressions.push_back(s);
  }
  {
    bool old_is_freeing = thr->is_freeing;
    thr->is_freeing = false;
    bool suppressed = OnReport(rep, pc_or_addr != 0);
    thr->is_freeing = old_is_freeing;
    if (suppressed) {
      thr->current_report = nullptr;
      return false;
    }
  }
  PrintReport(rep);
  __tsan_on_report(rep);
  ctx->nreported++;
  if (flags()->halt_on_error)
    Die();
  thr->current_report = nullptr;
  return true;
}

bool IsFiredSuppression(Context *ctx, ReportType type, StackTrace trace) {
  ReadLock lock(&ctx->fired_suppressions_mtx);
  for (uptr k = 0; k < ctx->fired_suppressions.size(); k++) {
    if (ctx->fired_suppressions[k].type != type)
      continue;
    for (uptr j = 0; j < trace.size; j++) {
      FiredSuppression *s = &ctx->fired_suppressions[k];
      if (trace.trace[j] == s->pc_or_addr) {
        if (s->supp)
          atomic_fetch_add(&s->supp->hit_count, 1, memory_order_relaxed);
        return true;
      }
    }
  }
  return false;
}

static bool IsFiredSuppression(Context *ctx, ReportType type, uptr addr) {
  ReadLock lock(&ctx->fired_suppressions_mtx);
  for (uptr k = 0; k < ctx->fired_suppressions.size(); k++) {
    if (ctx->fired_suppressions[k].type != type)
      continue;
    FiredSuppression *s = &ctx->fired_suppressions[k];
    if (addr == s->pc_or_addr) {
      if (s->supp)
        atomic_fetch_add(&s->supp->hit_count, 1, memory_order_relaxed);
      return true;
    }
  }
  return false;
}

static bool RaceBetweenAtomicAndFree(ThreadState *thr) {
  Shadow s0(thr->racy_state[0]);
  Shadow s1(thr->racy_state[1]);
  CHECK(!(s0.IsAtomic() && s1.IsAtomic()));
  if (!s0.IsAtomic() && !s1.IsAtomic())
    return true;
  if (s0.IsAtomic() && s1.IsFreed())
    return true;
  if (s1.IsAtomic() && thr->is_freeing)
    return true;
  return false;
}

void ReportRace(ThreadState *thr) {
  CheckNoLocks(thr);

  // Symbolizer makes lots of intercepted calls. If we try to process them,
  // at best it will cause deadlocks on internal mutexes.
  ScopedIgnoreInterceptors ignore;

  if (!flags()->report_bugs)
    return;
  if (!flags()->report_atomic_races && !RaceBetweenAtomicAndFree(thr))
    return;

  bool freed = false;
  {
    Shadow s(thr->racy_state[1]);
    freed = s.GetFreedAndReset();
    thr->racy_state[1] = s.raw();
  }

  uptr addr = ShadowToMem((uptr)thr->racy_shadow_addr);
  uptr addr_min = 0;
  uptr addr_max = 0;
  {
    uptr a0 = addr + Shadow(thr->racy_state[0]).addr0();
    uptr a1 = addr + Shadow(thr->racy_state[1]).addr0();
    uptr e0 = a0 + Shadow(thr->racy_state[0]).size();
    uptr e1 = a1 + Shadow(thr->racy_state[1]).size();
    addr_min = min(a0, a1);
    addr_max = max(e0, e1);
    if (IsExpectedReport(addr_min, addr_max - addr_min))
      return;
  }

  ReportType typ = ReportTypeRace;
  if (thr->is_vptr_access && freed)
    typ = ReportTypeVptrUseAfterFree;
  else if (thr->is_vptr_access)
    typ = ReportTypeVptrRace;
  else if (freed)
    typ = ReportTypeUseAfterFree;

  if (IsFiredSuppression(ctx, typ, addr))
    return;

  const uptr kMop = 2;
  VarSizeStackTrace traces[kMop];
  uptr tags[kMop] = {kExternalTagNone};
  uptr toppc = TraceTopPC(thr);
  if (toppc >> kEventPCBits) {
    // This is a work-around for a known issue.
    // The scenario where this happens is rather elaborate and requires
    // an instrumented __sanitizer_report_error_summary callback and
    // a __tsan_symbolize_external callback and a race during a range memory
    // access larger than 8 bytes. MemoryAccessRange adds the current PC to
    // the trace and starts processing memory accesses. A first memory access
    // triggers a race, we report it and call the instrumented
    // __sanitizer_report_error_summary, which adds more stuff to the trace
    // since it is intrumented. Then a second memory access in MemoryAccessRange
    // also triggers a race and we get here and call TraceTopPC to get the
    // current PC, however now it contains some unrelated events from the
    // callback. Most likely, TraceTopPC will now return a EventTypeFuncExit
    // event. Later we subtract -1 from it (in GetPreviousInstructionPc)
    // and the resulting PC has kExternalPCBit set, so we pass it to
    // __tsan_symbolize_external_ex. __tsan_symbolize_external_ex is within its
    // rights to crash since the PC is completely bogus.
    // test/tsan/double_race.cc contains a test case for this.
    toppc = 0;
  }
  ObtainCurrentStack(thr, toppc, &traces[0], &tags[0]);
  if (IsFiredSuppression(ctx, typ, traces[0]))
    return;

  // MutexSet is too large to live on stack.
  Vector<u64> mset_buffer;
  mset_buffer.Resize(sizeof(MutexSet) / sizeof(u64) + 1);
  MutexSet *mset2 = new(&mset_buffer[0]) MutexSet();

  Shadow s2(thr->racy_state[1]);
  RestoreStack(s2.tid(), s2.epoch(), &traces[1], mset2, &tags[1]);
  if (IsFiredSuppression(ctx, typ, traces[1]))
    return;

  if (HandleRacyStacks(thr, traces, addr_min, addr_max))
    return;

  // If any of the accesses has a tag, treat this as an "external" race.
  uptr tag = kExternalTagNone;
  for (uptr i = 0; i < kMop; i++) {
    if (tags[i] != kExternalTagNone) {
      typ = ReportTypeExternalRace;
      tag = tags[i];
      break;
    }
  }

  ThreadRegistryLock l0(ctx->thread_registry);
  ScopedReport rep(typ, tag);
  for (uptr i = 0; i < kMop; i++) {
    Shadow s(thr->racy_state[i]);
    rep.AddMemoryAccess(addr, tags[i], s, traces[i],
                        i == 0 ? &thr->mset : mset2);
  }

  for (uptr i = 0; i < kMop; i++) {
    FastState s(thr->racy_state[i]);
    ThreadContext *tctx = static_cast<ThreadContext*>(
        ctx->thread_registry->GetThreadLocked(s.tid()));
    if (s.epoch() < tctx->epoch0 || s.epoch() > tctx->epoch1)
      continue;
    rep.AddThread(tctx);
  }

  rep.AddLocation(addr_min, addr_max - addr_min);

#if !SANITIZER_GO
  {  // NOLINT
    Shadow s(thr->racy_state[1]);
    if (s.epoch() <= thr->last_sleep_clock.get(s.tid()))
      rep.AddSleep(thr->last_sleep_stack_id);
  }
#endif

  if (!OutputReport(thr, rep))
    return;

  AddRacyStacks(thr, traces, addr_min, addr_max);
}

void PrintCurrentStack(ThreadState *thr, uptr pc) {
  VarSizeStackTrace trace;
  ObtainCurrentStack(thr, pc, &trace);
  PrintStack(SymbolizeStack(trace));
}

// Always inlining PrintCurrentStackSlow, because LocatePcInTrace assumes
// __sanitizer_print_stack_trace exists in the actual unwinded stack, but
// tail-call to PrintCurrentStackSlow breaks this assumption because
// __sanitizer_print_stack_trace disappears after tail-call.
// However, this solution is not reliable enough, please see dvyukov's comment
// http://reviews.llvm.org/D19148#406208
// Also see PR27280 comment 2 and 3 for breaking examples and analysis.
ALWAYS_INLINE
void PrintCurrentStackSlow(uptr pc) {
#if !SANITIZER_GO
  BufferedStackTrace *ptrace =
      new(internal_alloc(MBlockStackTrace, sizeof(BufferedStackTrace)))
          BufferedStackTrace();
  ptrace->Unwind(kStackTraceMax, pc, 0, 0, 0, 0, false);
  for (uptr i = 0; i < ptrace->size / 2; i++) {
    uptr tmp = ptrace->trace_buffer[i];
    ptrace->trace_buffer[i] = ptrace->trace_buffer[ptrace->size - i - 1];
    ptrace->trace_buffer[ptrace->size - i - 1] = tmp;
  }
  PrintStack(SymbolizeStack(*ptrace));
#endif
}

}  // namespace __tsan

using namespace __tsan;

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  PrintCurrentStackSlow(StackTrace::GetCurrentPc());
}
}  // extern "C"
//===-- tsan_rtl_thread.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_placement_new.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "tsan_platform.h"
#include "tsan_report.h"
#include "tsan_sync.h"

namespace __tsan {

// ThreadContext implementation.

ThreadContext::ThreadContext(int tid)
  : ThreadContextBase(tid)
  , thr()
  , sync()
  , epoch0()
  , epoch1() {
}

#if !SANITIZER_GO
ThreadContext::~ThreadContext() {
}
#endif

void ThreadContext::OnDead() {
  CHECK_EQ(sync.size(), 0);
}

void ThreadContext::OnJoined(void *arg) {
  ThreadState *caller_thr = static_cast<ThreadState *>(arg);
  AcquireImpl(caller_thr, 0, &sync);
  sync.Reset(&caller_thr->proc()->clock_cache);
}

struct OnCreatedArgs {
  ThreadState *thr;
  uptr pc;
};

void ThreadContext::OnCreated(void *arg) {
  thr = 0;
  if (tid == 0)
    return;
  OnCreatedArgs *args = static_cast<OnCreatedArgs *>(arg);
  if (!args->thr)  // GCD workers don't have a parent thread.
    return;
  args->thr->fast_state.IncrementEpoch();
  // Can't increment epoch w/o writing to the trace as well.
  TraceAddEvent(args->thr, args->thr->fast_state, EventTypeMop, 0);
  ReleaseImpl(args->thr, 0, &sync);
  creation_stack_id = CurrentStackId(args->thr, args->pc);
  if (reuse_count == 0)
    StatInc(args->thr, StatThreadMaxTid);
}

void ThreadContext::OnReset() {
  CHECK_EQ(sync.size(), 0);
  uptr trace_p = GetThreadTrace(tid);
  ReleaseMemoryPagesToOS(trace_p, trace_p + TraceSize() * sizeof(Event));
  //!!! ReleaseMemoryToOS(GetThreadTraceHeader(tid), sizeof(Trace));
}

void ThreadContext::OnDetached(void *arg) {
  ThreadState *thr1 = static_cast<ThreadState*>(arg);
  sync.Reset(&thr1->proc()->clock_cache);
}

struct OnStartedArgs {
  ThreadState *thr;
  uptr stk_addr;
  uptr stk_size;
  uptr tls_addr;
  uptr tls_size;
};

void ThreadContext::OnStarted(void *arg) {
  OnStartedArgs *args = static_cast<OnStartedArgs*>(arg);
  thr = args->thr;
  // RoundUp so that one trace part does not contain events
  // from different threads.
  epoch0 = RoundUp(epoch1 + 1, kTracePartSize);
  epoch1 = (u64)-1;
  new(thr) ThreadState(ctx, tid, unique_id, epoch0, reuse_count,
      args->stk_addr, args->stk_size, args->tls_addr, args->tls_size);
#if !SANITIZER_GO
  thr->shadow_stack = &ThreadTrace(thr->tid)->shadow_stack[0];
  thr->shadow_stack_pos = thr->shadow_stack;
  thr->shadow_stack_end = thr->shadow_stack + kShadowStackSize;
#else
  // Setup dynamic shadow stack.
  const int kInitStackSize = 8;
  thr->shadow_stack = (uptr*)internal_alloc(MBlockShadowStack,
      kInitStackSize * sizeof(uptr));
  thr->shadow_stack_pos = thr->shadow_stack;
  thr->shadow_stack_end = thr->shadow_stack + kInitStackSize;
#endif
  if (common_flags()->detect_deadlocks)
    thr->dd_lt = ctx->dd->CreateLogicalThread(unique_id);
  thr->fast_state.SetHistorySize(flags()->history_size);
  // Commit switch to the new part of the trace.
  // TraceAddEvent will reset stack0/mset0 in the new part for us.
  TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);

  thr->fast_synch_epoch = epoch0;
  AcquireImpl(thr, 0, &sync);
  StatInc(thr, StatSyncAcquire);
  sync.Reset(&thr->proc()->clock_cache);
  thr->is_inited = true;
  DPrintf("#%d: ThreadStart epoch=%zu stk_addr=%zx stk_size=%zx "
          "tls_addr=%zx tls_size=%zx\n",
          tid, (uptr)epoch0, args->stk_addr, args->stk_size,
          args->tls_addr, args->tls_size);
}

void ThreadContext::OnFinished() {
#if SANITIZER_GO
  internal_free(thr->shadow_stack);
  thr->shadow_stack = nullptr;
  thr->shadow_stack_pos = nullptr;
  thr->shadow_stack_end = nullptr;
#endif
  if (!detached) {
    thr->fast_state.IncrementEpoch();
    // Can't increment epoch w/o writing to the trace as well.
    TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
    ReleaseImpl(thr, 0, &sync);
  }
  epoch1 = thr->fast_state.epoch();

  if (common_flags()->detect_deadlocks)
    ctx->dd->DestroyLogicalThread(thr->dd_lt);
  thr->clock.ResetCached(&thr->proc()->clock_cache);
#if !SANITIZER_GO
  thr->last_sleep_clock.ResetCached(&thr->proc()->clock_cache);
#endif
  thr->~ThreadState();
#if TSAN_COLLECT_STATS
  StatAggregate(ctx->stat, thr->stat);
#endif
  thr = 0;
}

#if !SANITIZER_GO
struct ThreadLeak {
  ThreadContext *tctx;
  int count;
};

static void MaybeReportThreadLeak(ThreadContextBase *tctx_base, void *arg) {
  Vector<ThreadLeak> &leaks = *(Vector<ThreadLeak>*)arg;
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  if (tctx->detached || tctx->status != ThreadStatusFinished)
    return;
  for (uptr i = 0; i < leaks.Size(); i++) {
    if (leaks[i].tctx->creation_stack_id == tctx->creation_stack_id) {
      leaks[i].count++;
      return;
    }
  }
  ThreadLeak leak = {tctx, 1};
  leaks.PushBack(leak);
}
#endif

#if !SANITIZER_GO
static void ReportIgnoresEnabled(ThreadContext *tctx, IgnoreSet *set) {
  if (tctx->tid == 0) {
    Printf("ThreadSanitizer: main thread finished with ignores enabled\n");
  } else {
    Printf("ThreadSanitizer: thread T%d %s finished with ignores enabled,"
      " created at:\n", tctx->tid, tctx->name);
    PrintStack(SymbolizeStackId(tctx->creation_stack_id));
  }
  Printf("  One of the following ignores was not ended"
      " (in order of probability)\n");
  for (uptr i = 0; i < set->Size(); i++) {
    Printf("  Ignore was enabled at:\n");
    PrintStack(SymbolizeStackId(set->At(i)));
  }
  Die();
}

static void ThreadCheckIgnore(ThreadState *thr) {
  if (ctx->after_multithreaded_fork)
    return;
  if (thr->ignore_reads_and_writes)
    ReportIgnoresEnabled(thr->tctx, &thr->mop_ignore_set);
  if (thr->ignore_sync)
    ReportIgnoresEnabled(thr->tctx, &thr->sync_ignore_set);
}
#else
static void ThreadCheckIgnore(ThreadState *thr) {}
#endif

void ThreadFinalize(ThreadState *thr) {
  ThreadCheckIgnore(thr);
#if !SANITIZER_GO
  if (!flags()->report_thread_leaks)
    return;
  ThreadRegistryLock l(ctx->thread_registry);
  Vector<ThreadLeak> leaks;
  ctx->thread_registry->RunCallbackForEachThreadLocked(
      MaybeReportThreadLeak, &leaks);
  for (uptr i = 0; i < leaks.Size(); i++) {
    ScopedReport rep(ReportTypeThreadLeak);
    rep.AddThread(leaks[i].tctx, true);
    rep.SetCount(leaks[i].count);
    OutputReport(thr, rep);
  }
#endif
}

int ThreadCount(ThreadState *thr) {
  uptr result;
  ctx->thread_registry->GetNumberOfThreads(0, 0, &result);
  return (int)result;
}

int ThreadCreate(ThreadState *thr, uptr pc, uptr uid, bool detached) {
  StatInc(thr, StatThreadCreate);
  OnCreatedArgs args = { thr, pc };
  u32 parent_tid = thr ? thr->tid : kInvalidTid;  // No parent for GCD workers.
  int tid =
      ctx->thread_registry->CreateThread(uid, detached, parent_tid, &args);
  DPrintf("#%d: ThreadCreate tid=%d uid=%zu\n", parent_tid, tid, uid);
  StatSet(thr, StatThreadMaxAlive, ctx->thread_registry->GetMaxAliveThreads());
  return tid;
}

void ThreadStart(ThreadState *thr, int tid, tid_t os_id, bool workerthread) {
  uptr stk_addr = 0;
  uptr stk_size = 0;
  uptr tls_addr = 0;
  uptr tls_size = 0;
#if !SANITIZER_GO
  GetThreadStackAndTls(tid == 0, &stk_addr, &stk_size, &tls_addr, &tls_size);

  if (tid) {
    if (stk_addr && stk_size)
      MemoryRangeImitateWrite(thr, /*pc=*/ 1, stk_addr, stk_size);

    if (tls_addr && tls_size) ImitateTlsWrite(thr, tls_addr, tls_size);
  }
#endif

  ThreadRegistry *tr = ctx->thread_registry;
  OnStartedArgs args = { thr, stk_addr, stk_size, tls_addr, tls_size };
  tr->StartThread(tid, os_id, workerthread, &args);

  tr->Lock();
  thr->tctx = (ThreadContext*)tr->GetThreadLocked(tid);
  tr->Unlock();

#if !SANITIZER_GO
  if (ctx->after_multithreaded_fork) {
    thr->ignore_interceptors++;
    ThreadIgnoreBegin(thr, 0);
    ThreadIgnoreSyncBegin(thr, 0);
  }
#endif
}

void ThreadFinish(ThreadState *thr) {
  ThreadCheckIgnore(thr);
  StatInc(thr, StatThreadFinish);
  if (thr->stk_addr && thr->stk_size)
    DontNeedShadowFor(thr->stk_addr, thr->stk_size);
  if (thr->tls_addr && thr->tls_size)
    DontNeedShadowFor(thr->tls_addr, thr->tls_size);
  thr->is_dead = true;
  ctx->thread_registry->FinishThread(thr->tid);
}

static bool FindThreadByUid(ThreadContextBase *tctx, void *arg) {
  uptr uid = (uptr)arg;
  if (tctx->user_id == uid && tctx->status != ThreadStatusInvalid) {
    tctx->user_id = 0;
    return true;
  }
  return false;
}

int ThreadTid(ThreadState *thr, uptr pc, uptr uid) {
  int res = ctx->thread_registry->FindThread(FindThreadByUid, (void*)uid);
  DPrintf("#%d: ThreadTid uid=%zu tid=%d\n", thr->tid, uid, res);
  return res;
}

void ThreadJoin(ThreadState *thr, uptr pc, int tid) {
  CHECK_GT(tid, 0);
  CHECK_LT(tid, kMaxTid);
  DPrintf("#%d: ThreadJoin tid=%d\n", thr->tid, tid);
  ctx->thread_registry->JoinThread(tid, thr);
}

void ThreadDetach(ThreadState *thr, uptr pc, int tid) {
  CHECK_GT(tid, 0);
  CHECK_LT(tid, kMaxTid);
  ctx->thread_registry->DetachThread(tid, thr);
}

void ThreadSetName(ThreadState *thr, const char *name) {
  ctx->thread_registry->SetThreadName(thr->tid, name);
}

void MemoryAccessRange(ThreadState *thr, uptr pc, uptr addr,
                       uptr size, bool is_write) {
  if (size == 0)
    return;

  u64 *shadow_mem = (u64*)MemToShadow(addr);
  DPrintf2("#%d: MemoryAccessRange: @%p %p size=%d is_write=%d\n",
      thr->tid, (void*)pc, (void*)addr,
      (int)size, is_write);

#if SANITIZER_DEBUG
  if (!IsAppMem(addr)) {
    Printf("Access to non app mem %zx\n", addr);
    DCHECK(IsAppMem(addr));
  }
  if (!IsAppMem(addr + size - 1)) {
    Printf("Access to non app mem %zx\n", addr + size - 1);
    DCHECK(IsAppMem(addr + size - 1));
  }
  if (!IsShadowMem((uptr)shadow_mem)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem, addr);
    DCHECK(IsShadowMem((uptr)shadow_mem));
  }
  if (!IsShadowMem((uptr)(shadow_mem + size * kShadowCnt / 8 - 1))) {
    Printf("Bad shadow addr %p (%zx)\n",
               shadow_mem + size * kShadowCnt / 8 - 1, addr + size - 1);
    DCHECK(IsShadowMem((uptr)(shadow_mem + size * kShadowCnt / 8 - 1)));
  }
#endif

  StatInc(thr, StatMopRange);

  if (*shadow_mem == kShadowRodata) {
    DCHECK(!is_write);
    // Access to .rodata section, no races here.
    // Measurements show that it can be 10-20% of all memory accesses.
    StatInc(thr, StatMopRangeRodata);
    return;
  }

  FastState fast_state = thr->fast_state;
  if (fast_state.GetIgnoreBit())
    return;

  fast_state.IncrementEpoch();
  thr->fast_state = fast_state;
  TraceAddEvent(thr, fast_state, EventTypeMop, pc);

  bool unaligned = (addr % kShadowCell) != 0;

  // Handle unaligned beginning, if any.
  for (; addr % kShadowCell && size; addr++, size--) {
    int const kAccessSizeLog = 0;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(addr & (kShadowCell - 1), kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false,
        shadow_mem, cur);
  }
  if (unaligned)
    shadow_mem += kShadowCnt;
  // Handle middle part, if any.
  for (; size >= kShadowCell; addr += kShadowCell, size -= kShadowCell) {
    int const kAccessSizeLog = 3;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(0, kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false,
        shadow_mem, cur);
    shadow_mem += kShadowCnt;
  }
  // Handle ending, if any.
  for (; size; addr++, size--) {
    int const kAccessSizeLog = 0;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(addr & (kShadowCell - 1), kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false,
        shadow_mem, cur);
  }
}

}  // namespace __tsan
//===-- tsan_rtl_proc.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_placement_new.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "tsan_flags.h"

namespace __tsan {

Processor *ProcCreate() {
  void *mem = InternalAlloc(sizeof(Processor));
  internal_memset(mem, 0, sizeof(Processor));
  Processor *proc = new(mem) Processor;
  proc->thr = nullptr;
#if !SANITIZER_GO
  AllocatorProcStart(proc);
#endif
  if (common_flags()->detect_deadlocks)
    proc->dd_pt = ctx->dd->CreatePhysicalThread();
  return proc;
}

void ProcDestroy(Processor *proc) {
  CHECK_EQ(proc->thr, nullptr);
#if !SANITIZER_GO
  AllocatorProcFinish(proc);
#endif
  ctx->clock_alloc.FlushCache(&proc->clock_cache);
  ctx->metamap.OnProcIdle(proc);
  if (common_flags()->detect_deadlocks)
     ctx->dd->DestroyPhysicalThread(proc->dd_pt);
  proc->~Processor();
  InternalFree(proc);
}

void ProcWire(Processor *proc, ThreadState *thr) {
  CHECK_EQ(thr->proc1, nullptr);
  CHECK_EQ(proc->thr, nullptr);
  thr->proc1 = proc;
  proc->thr = thr;
}

void ProcUnwire(Processor *proc, ThreadState *thr) {
  CHECK_EQ(thr->proc1, proc);
  CHECK_EQ(proc->thr, thr);
  thr->proc1 = nullptr;
  proc->thr = nullptr;
}

}  // namespace __tsan
//===-- tsan_stack_trace.cc -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_stack_trace.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"

namespace __tsan {

VarSizeStackTrace::VarSizeStackTrace()
    : StackTrace(nullptr, 0), trace_buffer(nullptr) {}

VarSizeStackTrace::~VarSizeStackTrace() {
  ResizeBuffer(0);
}

void VarSizeStackTrace::ResizeBuffer(uptr new_size) {
  if (trace_buffer) {
    internal_free(trace_buffer);
  }
  trace_buffer =
      (new_size > 0)
          ? (uptr *)internal_alloc(MBlockStackTrace,
                                   new_size * sizeof(trace_buffer[0]))
          : nullptr;
  trace = trace_buffer;
  size = new_size;
}

void VarSizeStackTrace::Init(const uptr *pcs, uptr cnt, uptr extra_top_pc) {
  ResizeBuffer(cnt + !!extra_top_pc);
  internal_memcpy(trace_buffer, pcs, cnt * sizeof(trace_buffer[0]));
  if (extra_top_pc)
    trace_buffer[cnt] = extra_top_pc;
}

}  // namespace __tsan
//===-- tsan_stat.cc ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_stat.h"
#include "tsan_rtl.h"

namespace __tsan {

#if TSAN_COLLECT_STATS

void StatAggregate(u64 *dst, u64 *src) {
  for (int i = 0; i < StatCnt; i++)
    dst[i] += src[i];
}

void StatOutput(u64 *stat) {
  stat[StatShadowNonZero] = stat[StatShadowProcessed] - stat[StatShadowZero];

  static const char *name[StatCnt] = {};
  name[StatMop]                          = "Memory accesses                   ";
  name[StatMopRead]                      = "  Including reads                 ";
  name[StatMopWrite]                     = "            writes                ";
  name[StatMop1]                         = "  Including size 1                ";
  name[StatMop2]                         = "            size 2                ";
  name[StatMop4]                         = "            size 4                ";
  name[StatMop8]                         = "            size 8                ";
  name[StatMopSame]                      = "  Including same                  ";
  name[StatMopIgnored]                   = "  Including ignored               ";
  name[StatMopRange]                     = "  Including range                 ";
  name[StatMopRodata]                    = "  Including .rodata               ";
  name[StatMopRangeRodata]               = "  Including .rodata range         ";
  name[StatShadowProcessed]              = "Shadow processed                  ";
  name[StatShadowZero]                   = "  Including empty                 ";
  name[StatShadowNonZero]                = "  Including non empty             ";
  name[StatShadowSameSize]               = "  Including same size             ";
  name[StatShadowIntersect]              = "            intersect             ";
  name[StatShadowNotIntersect]           = "            not intersect         ";
  name[StatShadowSameThread]             = "  Including same thread           ";
  name[StatShadowAnotherThread]          = "            another thread        ";
  name[StatShadowReplace]                = "  Including evicted               ";

  name[StatFuncEnter]                    = "Function entries                  ";
  name[StatFuncExit]                     = "Function exits                    ";
  name[StatEvents]                       = "Events collected                  ";

  name[StatThreadCreate]                 = "Total threads created             ";
  name[StatThreadFinish]                 = "  threads finished                ";
  name[StatThreadReuse]                  = "  threads reused                  ";
  name[StatThreadMaxTid]                 = "  max tid                         ";
  name[StatThreadMaxAlive]               = "  max alive threads               ";

  name[StatMutexCreate]                  = "Mutexes created                   ";
  name[StatMutexDestroy]                 = "  destroyed                       ";
  name[StatMutexLock]                    = "  lock                            ";
  name[StatMutexUnlock]                  = "  unlock                          ";
  name[StatMutexRecLock]                 = "  recursive lock                  ";
  name[StatMutexRecUnlock]               = "  recursive unlock                ";
  name[StatMutexReadLock]                = "  read lock                       ";
  name[StatMutexReadUnlock]              = "  read unlock                     ";

  name[StatSyncCreated]                  = "Sync objects created              ";
  name[StatSyncDestroyed]                = "             destroyed            ";
  name[StatSyncAcquire]                  = "             acquired             ";
  name[StatSyncRelease]                  = "             released             ";

  name[StatClockAcquire]                 = "Clock acquire                     ";
  name[StatClockAcquireEmpty]            = "  empty clock                     ";
  name[StatClockAcquireFastRelease]      = "  fast from release-store         ";
  name[StatClockAcquireFull]             = "  full (slow)                     ";
  name[StatClockAcquiredSomething]       = "  acquired something              ";
  name[StatClockRelease]                 = "Clock release                     ";
  name[StatClockReleaseResize]           = "  resize                          ";
  name[StatClockReleaseFast]             = "  fast                            ";
  name[StatClockReleaseSlow]             = "  dirty overflow (slow)           ";
  name[StatClockReleaseFull]             = "  full (slow)                     ";
  name[StatClockReleaseAcquired]         = "  was acquired                    ";
  name[StatClockReleaseClearTail]        = "  clear tail                      ";
  name[StatClockStore]                   = "Clock release store               ";
  name[StatClockStoreResize]             = "  resize                          ";
  name[StatClockStoreFast]               = "  fast                            ";
  name[StatClockStoreFull]               = "  slow                            ";
  name[StatClockStoreTail]               = "  clear tail                      ";
  name[StatClockAcquireRelease]          = "Clock acquire-release             ";

  name[StatAtomic]                       = "Atomic operations                 ";
  name[StatAtomicLoad]                   = "  Including load                  ";
  name[StatAtomicStore]                  = "            store                 ";
  name[StatAtomicExchange]               = "            exchange              ";
  name[StatAtomicFetchAdd]               = "            fetch_add             ";
  name[StatAtomicFetchSub]               = "            fetch_sub             ";
  name[StatAtomicFetchAnd]               = "            fetch_and             ";
  name[StatAtomicFetchOr]                = "            fetch_or              ";
  name[StatAtomicFetchXor]               = "            fetch_xor             ";
  name[StatAtomicFetchNand]              = "            fetch_nand            ";
  name[StatAtomicCAS]                    = "            compare_exchange      ";
  name[StatAtomicFence]                  = "            fence                 ";
  name[StatAtomicRelaxed]                = "  Including relaxed               ";
  name[StatAtomicConsume]                = "            consume               ";
  name[StatAtomicAcquire]                = "            acquire               ";
  name[StatAtomicRelease]                = "            release               ";
  name[StatAtomicAcq_Rel]                = "            acq_rel               ";
  name[StatAtomicSeq_Cst]                = "            seq_cst               ";
  name[StatAtomic1]                      = "  Including size 1                ";
  name[StatAtomic2]                      = "            size 2                ";
  name[StatAtomic4]                      = "            size 4                ";
  name[StatAtomic8]                      = "            size 8                ";
  name[StatAtomic16]                     = "            size 16               ";

  name[StatAnnotation]                   = "Dynamic annotations               ";
  name[StatAnnotateHappensBefore]        = "  HappensBefore                   ";
  name[StatAnnotateHappensAfter]         = "  HappensAfter                    ";
  name[StatAnnotateCondVarSignal]        = "  CondVarSignal                   ";
  name[StatAnnotateCondVarSignalAll]     = "  CondVarSignalAll                ";
  name[StatAnnotateMutexIsNotPHB]        = "  MutexIsNotPHB                   ";
  name[StatAnnotateCondVarWait]          = "  CondVarWait                     ";
  name[StatAnnotateRWLockCreate]         = "  RWLockCreate                    ";
  name[StatAnnotateRWLockCreateStatic]   = "  StatAnnotateRWLockCreateStatic  ";
  name[StatAnnotateRWLockDestroy]        = "  RWLockDestroy                   ";
  name[StatAnnotateRWLockAcquired]       = "  RWLockAcquired                  ";
  name[StatAnnotateRWLockReleased]       = "  RWLockReleased                  ";
  name[StatAnnotateTraceMemory]          = "  TraceMemory                     ";
  name[StatAnnotateFlushState]           = "  FlushState                      ";
  name[StatAnnotateNewMemory]            = "  NewMemory                       ";
  name[StatAnnotateNoOp]                 = "  NoOp                            ";
  name[StatAnnotateFlushExpectedRaces]   = "  FlushExpectedRaces              ";
  name[StatAnnotateEnableRaceDetection]  = "  EnableRaceDetection             ";
  name[StatAnnotateMutexIsUsedAsCondVar] = "  MutexIsUsedAsCondVar            ";
  name[StatAnnotatePCQGet]               = "  PCQGet                          ";
  name[StatAnnotatePCQPut]               = "  PCQPut                          ";
  name[StatAnnotatePCQDestroy]           = "  PCQDestroy                      ";
  name[StatAnnotatePCQCreate]            = "  PCQCreate                       ";
  name[StatAnnotateExpectRace]           = "  ExpectRace                      ";
  name[StatAnnotateBenignRaceSized]      = "  BenignRaceSized                 ";
  name[StatAnnotateBenignRace]           = "  BenignRace                      ";
  name[StatAnnotateIgnoreReadsBegin]     = "  IgnoreReadsBegin                ";
  name[StatAnnotateIgnoreReadsEnd]       = "  IgnoreReadsEnd                  ";
  name[StatAnnotateIgnoreWritesBegin]    = "  IgnoreWritesBegin               ";
  name[StatAnnotateIgnoreWritesEnd]      = "  IgnoreWritesEnd                 ";
  name[StatAnnotateIgnoreSyncBegin]      = "  IgnoreSyncBegin                 ";
  name[StatAnnotateIgnoreSyncEnd]        = "  IgnoreSyncEnd                   ";
  name[StatAnnotatePublishMemoryRange]   = "  PublishMemoryRange              ";
  name[StatAnnotateUnpublishMemoryRange] = "  UnpublishMemoryRange            ";
  name[StatAnnotateThreadName]           = "  ThreadName                      ";
  name[Stat__tsan_mutex_create]          = "  __tsan_mutex_create             ";
  name[Stat__tsan_mutex_destroy]         = "  __tsan_mutex_destroy            ";
  name[Stat__tsan_mutex_pre_lock]        = "  __tsan_mutex_pre_lock           ";
  name[Stat__tsan_mutex_post_lock]       = "  __tsan_mutex_post_lock          ";
  name[Stat__tsan_mutex_pre_unlock]      = "  __tsan_mutex_pre_unlock         ";
  name[Stat__tsan_mutex_post_unlock]     = "  __tsan_mutex_post_unlock        ";
  name[Stat__tsan_mutex_pre_signal]      = "  __tsan_mutex_pre_signal         ";
  name[Stat__tsan_mutex_post_signal]     = "  __tsan_mutex_post_signal        ";
  name[Stat__tsan_mutex_pre_divert]      = "  __tsan_mutex_pre_divert         ";
  name[Stat__tsan_mutex_post_divert]     = "  __tsan_mutex_post_divert        ";

  name[StatMtxTotal]                     = "Contentionz                       ";
  name[StatMtxTrace]                     = "  Trace                           ";
  name[StatMtxThreads]                   = "  Threads                         ";
  name[StatMtxReport]                    = "  Report                          ";
  name[StatMtxSyncVar]                   = "  SyncVar                         ";
  name[StatMtxSyncTab]                   = "  SyncTab                         ";
  name[StatMtxSlab]                      = "  Slab                            ";
  name[StatMtxAtExit]                    = "  Atexit                          ";
  name[StatMtxAnnotations]               = "  Annotations                     ";
  name[StatMtxMBlock]                    = "  MBlock                          ";
  name[StatMtxDeadlockDetector]          = "  DeadlockDetector                ";
  name[StatMtxFired]                     = "  FiredSuppressions               ";
  name[StatMtxRacy]                      = "  RacyStacks                      ";
  name[StatMtxFD]                        = "  FD                              ";
  name[StatMtxGlobalProc]                = "  GlobalProc                      ";

  Printf("Statistics:\n");
  for (int i = 0; i < StatCnt; i++)
    Printf("%s: %16zu\n", name[i], (uptr)stat[i]);
}

#endif

}  // namespace __tsan
//===-- tsan_suppressions.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "tsan_suppressions.h"
#include "tsan_rtl.h"
#include "tsan_flags.h"
#include "tsan_mman.h"
#include "tsan_platform.h"

#if !SANITIZER_GO
// Suppressions for true/false positives in standard libraries.
static const char *const std_suppressions =
// Libstdc++ 4.4 has data races in std::string.
// See http://crbug.com/181502 for an example.
"race:^_M_rep$\n"
"race:^_M_is_leaked$\n"
// False positive when using std <thread>.
// Happens because we miss atomic synchronization in libstdc++.
// See http://llvm.org/bugs/show_bug.cgi?id=17066 for details.
"race:std::_Sp_counted_ptr_inplace<std::thread::_Impl\n";

// Can be overriden in frontend.
SANITIZER_WEAK_DEFAULT_IMPL
const char *__tsan_default_suppressions() {
  return 0;
}
#endif

namespace __tsan {

ALIGNED(64) static char suppression_placeholder[sizeof(SuppressionContext)];
static SuppressionContext *suppression_ctx = nullptr;
static const char *kSuppressionTypes[] = {
    kSuppressionRace,   kSuppressionRaceTop, kSuppressionMutex,
    kSuppressionThread, kSuppressionSignal, kSuppressionLib,
    kSuppressionDeadlock};

void InitializeSuppressions() {
  CHECK_EQ(nullptr, suppression_ctx);
  suppression_ctx = new (suppression_placeholder) // NOLINT
      SuppressionContext(kSuppressionTypes, ARRAY_SIZE(kSuppressionTypes));
  suppression_ctx->ParseFromFile(flags()->suppressions);
#if !SANITIZER_GO
  suppression_ctx->Parse(__tsan_default_suppressions());
  suppression_ctx->Parse(std_suppressions);
#endif
}

SuppressionContext *Suppressions() {
  CHECK(suppression_ctx);
  return suppression_ctx;
}

static const char *conv(ReportType typ) {
  if (typ == ReportTypeRace)
    return kSuppressionRace;
  else if (typ == ReportTypeVptrRace)
    return kSuppressionRace;
  else if (typ == ReportTypeUseAfterFree)
    return kSuppressionRace;
  else if (typ == ReportTypeVptrUseAfterFree)
    return kSuppressionRace;
  else if (typ == ReportTypeExternalRace)
    return kSuppressionRace;
  else if (typ == ReportTypeThreadLeak)
    return kSuppressionThread;
  else if (typ == ReportTypeMutexDestroyLocked)
    return kSuppressionMutex;
  else if (typ == ReportTypeMutexDoubleLock)
    return kSuppressionMutex;
  else if (typ == ReportTypeMutexInvalidAccess)
    return kSuppressionMutex;
  else if (typ == ReportTypeMutexBadUnlock)
    return kSuppressionMutex;
  else if (typ == ReportTypeMutexBadReadLock)
    return kSuppressionMutex;
  else if (typ == ReportTypeMutexBadReadUnlock)
    return kSuppressionMutex;
  else if (typ == ReportTypeSignalUnsafe)
    return kSuppressionSignal;
  else if (typ == ReportTypeErrnoInSignal)
    return kSuppressionNone;
  else if (typ == ReportTypeDeadlock)
    return kSuppressionDeadlock;
  Printf("ThreadSanitizer: unknown report type %d\n", typ);
  Die();
}

static uptr IsSuppressed(const char *stype, const AddressInfo &info,
    Suppression **sp) {
  if (suppression_ctx->Match(info.function, stype, sp) ||
      suppression_ctx->Match(info.file, stype, sp) ||
      suppression_ctx->Match(info.module, stype, sp)) {
    VPrintf(2, "ThreadSanitizer: matched suppression '%s'\n", (*sp)->templ);
    atomic_fetch_add(&(*sp)->hit_count, 1, memory_order_relaxed);
    return info.address;
  }
  return 0;
}

uptr IsSuppressed(ReportType typ, const ReportStack *stack, Suppression **sp) {
  CHECK(suppression_ctx);
  if (!suppression_ctx->SuppressionCount() || stack == 0 ||
      !stack->suppressable)
    return 0;
  const char *stype = conv(typ);
  if (0 == internal_strcmp(stype, kSuppressionNone))
    return 0;
  for (const SymbolizedStack *frame = stack->frames; frame;
      frame = frame->next) {
    uptr pc = IsSuppressed(stype, frame->info, sp);
    if (pc != 0)
      return pc;
  }
  if (0 == internal_strcmp(stype, kSuppressionRace) && stack->frames != nullptr)
    return IsSuppressed(kSuppressionRaceTop, stack->frames->info, sp);
  return 0;
}

uptr IsSuppressed(ReportType typ, const ReportLocation *loc, Suppression **sp) {
  CHECK(suppression_ctx);
  if (!suppression_ctx->SuppressionCount() || loc == 0 ||
      loc->type != ReportLocationGlobal || !loc->suppressable)
    return 0;
  const char *stype = conv(typ);
  if (0 == internal_strcmp(stype, kSuppressionNone))
    return 0;
  Suppression *s;
  const DataInfo &global = loc->global;
  if (suppression_ctx->Match(global.name, stype, &s) ||
      suppression_ctx->Match(global.module, stype, &s)) {
      VPrintf(2, "ThreadSanitizer: matched suppression '%s'\n", s->templ);
      atomic_fetch_add(&s->hit_count, 1, memory_order_relaxed);
      *sp = s;
      return global.start;
  }
  return 0;
}

void PrintMatchedSuppressions() {
  InternalMmapVector<Suppression *> matched(1);
  CHECK(suppression_ctx);
  suppression_ctx->GetMatched(&matched);
  if (!matched.size())
    return;
  int hit_count = 0;
  for (uptr i = 0; i < matched.size(); i++)
    hit_count += atomic_load_relaxed(&matched[i]->hit_count);
  Printf("ThreadSanitizer: Matched %d suppressions (pid=%d):\n", hit_count,
         (int)internal_getpid());
  for (uptr i = 0; i < matched.size(); i++) {
    Printf("%d %s:%s\n", atomic_load_relaxed(&matched[i]->hit_count),
           matched[i]->type, matched[i]->templ);
  }
}
}  // namespace __tsan
//===-- tsan_sync.cc ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_placement_new.h"
#include "tsan_sync.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"

namespace __tsan {

void DDMutexInit(ThreadState *thr, uptr pc, SyncVar *s);

SyncVar::SyncVar()
    : mtx(MutexTypeSyncVar, StatMtxSyncVar) {
  Reset(0);
}

void SyncVar::Init(ThreadState *thr, uptr pc, uptr addr, u64 uid) {
  this->addr = addr;
  this->uid = uid;
  this->next = 0;

  creation_stack_id = 0;
  if (!SANITIZER_GO)  // Go does not use them
    creation_stack_id = CurrentStackId(thr, pc);
  if (common_flags()->detect_deadlocks)
    DDMutexInit(thr, pc, this);
}

void SyncVar::Reset(Processor *proc) {
  uid = 0;
  creation_stack_id = 0;
  owner_tid = kInvalidTid;
  last_lock = 0;
  recursion = 0;
  atomic_store_relaxed(&flags, 0);

  if (proc == 0) {
    CHECK_EQ(clock.size(), 0);
    CHECK_EQ(read_clock.size(), 0);
  } else {
    clock.Reset(&proc->clock_cache);
    read_clock.Reset(&proc->clock_cache);
  }
}

MetaMap::MetaMap()
    : block_alloc_("heap block allocator")
    , sync_alloc_("sync allocator") {
  atomic_store(&uid_gen_, 0, memory_order_relaxed);
}

void MetaMap::AllocBlock(ThreadState *thr, uptr pc, uptr p, uptr sz) {
  u32 idx = block_alloc_.Alloc(&thr->proc()->block_cache);
  MBlock *b = block_alloc_.Map(idx);
  b->siz = sz;
  b->tag = 0;
  b->tid = thr->tid;
  b->stk = CurrentStackId(thr, pc);
  u32 *meta = MemToMeta(p);
  DCHECK_EQ(*meta, 0);
  *meta = idx | kFlagBlock;
}

uptr MetaMap::FreeBlock(Processor *proc, uptr p) {
  MBlock* b = GetBlock(p);
  if (b == 0)
    return 0;
  uptr sz = RoundUpTo(b->siz, kMetaShadowCell);
  FreeRange(proc, p, sz);
  return sz;
}

bool MetaMap::FreeRange(Processor *proc, uptr p, uptr sz) {
  bool has_something = false;
  u32 *meta = MemToMeta(p);
  u32 *end = MemToMeta(p + sz);
  if (end == meta)
    end++;
  for (; meta < end; meta++) {
    u32 idx = *meta;
    if (idx == 0) {
      // Note: don't write to meta in this case -- the block can be huge.
      continue;
    }
    *meta = 0;
    has_something = true;
    while (idx != 0) {
      if (idx & kFlagBlock) {
        block_alloc_.Free(&proc->block_cache, idx & ~kFlagMask);
        break;
      } else if (idx & kFlagSync) {
        DCHECK(idx & kFlagSync);
        SyncVar *s = sync_alloc_.Map(idx & ~kFlagMask);
        u32 next = s->next;
        s->Reset(proc);
        sync_alloc_.Free(&proc->sync_cache, idx & ~kFlagMask);
        idx = next;
      } else {
        CHECK(0);
      }
    }
  }
  return has_something;
}

// ResetRange removes all meta objects from the range.
// It is called for large mmap-ed regions. The function is best-effort wrt
// freeing of meta objects, because we don't want to page in the whole range
// which can be huge. The function probes pages one-by-one until it finds a page
// without meta objects, at this point it stops freeing meta objects. Because
// thread stacks grow top-down, we do the same starting from end as well.
void MetaMap::ResetRange(Processor *proc, uptr p, uptr sz) {
  if (SANITIZER_GO) {
    // UnmapOrDie/MmapFixedNoReserve does not work on Windows,
    // so we do the optimization only for C/C++.
    FreeRange(proc, p, sz);
    return;
  }
  const uptr kMetaRatio = kMetaShadowCell / kMetaShadowSize;
  const uptr kPageSize = GetPageSizeCached() * kMetaRatio;
  if (sz <= 4 * kPageSize) {
    // If the range is small, just do the normal free procedure.
    FreeRange(proc, p, sz);
    return;
  }
  // First, round both ends of the range to page size.
  uptr diff = RoundUp(p, kPageSize) - p;
  if (diff != 0) {
    FreeRange(proc, p, diff);
    p += diff;
    sz -= diff;
  }
  diff = p + sz - RoundDown(p + sz, kPageSize);
  if (diff != 0) {
    FreeRange(proc, p + sz - diff, diff);
    sz -= diff;
  }
  // Now we must have a non-empty page-aligned range.
  CHECK_GT(sz, 0);
  CHECK_EQ(p, RoundUp(p, kPageSize));
  CHECK_EQ(sz, RoundUp(sz, kPageSize));
  const uptr p0 = p;
  const uptr sz0 = sz;
  // Probe start of the range.
  for (uptr checked = 0; sz > 0; checked += kPageSize) {
    bool has_something = FreeRange(proc, p, kPageSize);
    p += kPageSize;
    sz -= kPageSize;
    if (!has_something && checked > (128 << 10))
      break;
  }
  // Probe end of the range.
  for (uptr checked = 0; sz > 0; checked += kPageSize) {
    bool has_something = FreeRange(proc, p + sz - kPageSize, kPageSize);
    sz -= kPageSize;
    // Stacks grow down, so sync object are most likely at the end of the region
    // (if it is a stack). The very end of the stack is TLS and tsan increases
    // TLS by at least 256K, so check at least 512K.
    if (!has_something && checked > (512 << 10))
      break;
  }
  // Finally, page out the whole range (including the parts that we've just
  // freed). Note: we can't simply madvise, because we need to leave a zeroed
  // range (otherwise __tsan_java_move can crash if it encounters a left-over
  // meta objects in java heap).
  uptr metap = (uptr)MemToMeta(p0);
  uptr metasz = sz0 / kMetaRatio;
  UnmapOrDie((void*)metap, metasz);
  MmapFixedNoReserve(metap, metasz);
}

MBlock* MetaMap::GetBlock(uptr p) {
  u32 *meta = MemToMeta(p);
  u32 idx = *meta;
  for (;;) {
    if (idx == 0)
      return 0;
    if (idx & kFlagBlock)
      return block_alloc_.Map(idx & ~kFlagMask);
    DCHECK(idx & kFlagSync);
    SyncVar * s = sync_alloc_.Map(idx & ~kFlagMask);
    idx = s->next;
  }
}

SyncVar* MetaMap::GetOrCreateAndLock(ThreadState *thr, uptr pc,
                              uptr addr, bool write_lock) {
  return GetAndLock(thr, pc, addr, write_lock, true);
}

SyncVar* MetaMap::GetIfExistsAndLock(uptr addr, bool write_lock) {
  return GetAndLock(0, 0, addr, write_lock, false);
}

SyncVar* MetaMap::GetAndLock(ThreadState *thr, uptr pc,
                             uptr addr, bool write_lock, bool create) {
  u32 *meta = MemToMeta(addr);
  u32 idx0 = *meta;
  u32 myidx = 0;
  SyncVar *mys = 0;
  for (;;) {
    u32 idx = idx0;
    for (;;) {
      if (idx == 0)
        break;
      if (idx & kFlagBlock)
        break;
      DCHECK(idx & kFlagSync);
      SyncVar * s = sync_alloc_.Map(idx & ~kFlagMask);
      if (s->addr == addr) {
        if (myidx != 0) {
          mys->Reset(thr->proc());
          sync_alloc_.Free(&thr->proc()->sync_cache, myidx);
        }
        if (write_lock)
          s->mtx.Lock();
        else
          s->mtx.ReadLock();
        return s;
      }
      idx = s->next;
    }
    if (!create)
      return 0;
    if (*meta != idx0) {
      idx0 = *meta;
      continue;
    }

    if (myidx == 0) {
      const u64 uid = atomic_fetch_add(&uid_gen_, 1, memory_order_relaxed);
      myidx = sync_alloc_.Alloc(&thr->proc()->sync_cache);
      mys = sync_alloc_.Map(myidx);
      mys->Init(thr, pc, addr, uid);
    }
    mys->next = idx0;
    if (atomic_compare_exchange_strong((atomic_uint32_t*)meta, &idx0,
        myidx | kFlagSync, memory_order_release)) {
      if (write_lock)
        mys->mtx.Lock();
      else
        mys->mtx.ReadLock();
      return mys;
    }
  }
}

void MetaMap::MoveMemory(uptr src, uptr dst, uptr sz) {
  // src and dst can overlap,
  // there are no concurrent accesses to the regions (e.g. stop-the-world).
  CHECK_NE(src, dst);
  CHECK_NE(sz, 0);
  uptr diff = dst - src;
  u32 *src_meta = MemToMeta(src);
  u32 *dst_meta = MemToMeta(dst);
  u32 *src_meta_end = MemToMeta(src + sz);
  uptr inc = 1;
  if (dst > src) {
    src_meta = MemToMeta(src + sz) - 1;
    dst_meta = MemToMeta(dst + sz) - 1;
    src_meta_end = MemToMeta(src) - 1;
    inc = -1;
  }
  for (; src_meta != src_meta_end; src_meta += inc, dst_meta += inc) {
    CHECK_EQ(*dst_meta, 0);
    u32 idx = *src_meta;
    *src_meta = 0;
    *dst_meta = idx;
    // Patch the addresses in sync objects.
    while (idx != 0) {
      if (idx & kFlagBlock)
        break;
      CHECK(idx & kFlagSync);
      SyncVar *s = sync_alloc_.Map(idx & ~kFlagMask);
      s->addr += diff;
      idx = s->next;
    }
  }
}

void MetaMap::OnProcIdle(Processor *proc) {
  block_alloc_.FlushCache(&proc->block_cache);
  sync_alloc_.FlushCache(&proc->sync_cache);
}

}  // namespace __tsan
//===-- sanitizer_allocator.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
// This allocator is used inside run-times.
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator.h"

#include "sanitizer_allocator_checks.h"
#include "sanitizer_allocator_internal.h"
#include "sanitizer_atomic.h"
#include "sanitizer_common.h"

namespace __sanitizer {

// Default allocator names.
const char *PrimaryAllocatorName = "SizeClassAllocator";
const char *SecondaryAllocatorName = "LargeMmapAllocator";

// ThreadSanitizer for Go uses libc malloc/free.
#if SANITIZER_GO || defined(SANITIZER_USE_MALLOC)
# if SANITIZER_LINUX && !SANITIZER_ANDROID
extern "C" void *__libc_malloc(uptr size);
#  if !SANITIZER_GO
extern "C" void *__libc_memalign(uptr alignment, uptr size);
#  endif
extern "C" void *__libc_realloc(void *ptr, uptr size);
extern "C" void __libc_free(void *ptr);
# else
#  include <stdlib.h>
#  define __libc_malloc malloc
#  if !SANITIZER_GO
static void *__libc_memalign(uptr alignment, uptr size) {
  void *p;
  uptr error = posix_memalign(&p, alignment, size);
  if (error) return nullptr;
  return p;
}
#  endif
#  define __libc_realloc realloc
#  define __libc_free free
# endif

static void *RawInternalAlloc(uptr size, InternalAllocatorCache *cache,
                              uptr alignment) {
  (void)cache;
#if !SANITIZER_GO
  if (alignment == 0)
    return __libc_malloc(size);
  else
    return __libc_memalign(alignment, size);
#else
  // Windows does not provide __libc_memalign/posix_memalign. It provides
  // __aligned_malloc, but the allocated blocks can't be passed to free,
  // they need to be passed to __aligned_free. InternalAlloc interface does
  // not account for such requirement. Alignemnt does not seem to be used
  // anywhere in runtime, so just call __libc_malloc for now.
  DCHECK_EQ(alignment, 0);
  return __libc_malloc(size);
#endif
}

static void *RawInternalRealloc(void *ptr, uptr size,
                                InternalAllocatorCache *cache) {
  (void)cache;
  return __libc_realloc(ptr, size);
}

static void RawInternalFree(void *ptr, InternalAllocatorCache *cache) {
  (void)cache;
  __libc_free(ptr);
}

InternalAllocator *internal_allocator() {
  return 0;
}

#else  // SANITIZER_GO || defined(SANITIZER_USE_MALLOC)

static ALIGNED(64) char internal_alloc_placeholder[sizeof(InternalAllocator)];
static atomic_uint8_t internal_allocator_initialized;
static StaticSpinMutex internal_alloc_init_mu;

static InternalAllocatorCache internal_allocator_cache;
static StaticSpinMutex internal_allocator_cache_mu;

InternalAllocator *internal_allocator() {
  InternalAllocator *internal_allocator_instance =
      reinterpret_cast<InternalAllocator *>(&internal_alloc_placeholder);
  if (atomic_load(&internal_allocator_initialized, memory_order_acquire) == 0) {
    SpinMutexLock l(&internal_alloc_init_mu);
    if (atomic_load(&internal_allocator_initialized, memory_order_relaxed) ==
        0) {
      internal_allocator_instance->Init(kReleaseToOSIntervalNever);
      atomic_store(&internal_allocator_initialized, 1, memory_order_release);
    }
  }
  return internal_allocator_instance;
}

static void *RawInternalAlloc(uptr size, InternalAllocatorCache *cache,
                              uptr alignment) {
  if (alignment == 0) alignment = 8;
  if (cache == 0) {
    SpinMutexLock l(&internal_allocator_cache_mu);
    return internal_allocator()->Allocate(&internal_allocator_cache, size,
                                          alignment);
  }
  return internal_allocator()->Allocate(cache, size, alignment);
}

static void *RawInternalRealloc(void *ptr, uptr size,
                                InternalAllocatorCache *cache) {
  uptr alignment = 8;
  if (cache == 0) {
    SpinMutexLock l(&internal_allocator_cache_mu);
    return internal_allocator()->Reallocate(&internal_allocator_cache, ptr,
                                            size, alignment);
  }
  return internal_allocator()->Reallocate(cache, ptr, size, alignment);
}

static void RawInternalFree(void *ptr, InternalAllocatorCache *cache) {
  if (!cache) {
    SpinMutexLock l(&internal_allocator_cache_mu);
    return internal_allocator()->Deallocate(&internal_allocator_cache, ptr);
  }
  internal_allocator()->Deallocate(cache, ptr);
}

#endif  // SANITIZER_GO || defined(SANITIZER_USE_MALLOC)

const u64 kBlockMagic = 0x6A6CB03ABCEBC041ull;

void *InternalAlloc(uptr size, InternalAllocatorCache *cache, uptr alignment) {
  if (size + sizeof(u64) < size)
    return nullptr;
  void *p = RawInternalAlloc(size + sizeof(u64), cache, alignment);
  if (UNLIKELY(!p))
    return DieOnFailure::OnOOM();
  ((u64*)p)[0] = kBlockMagic;
  return (char*)p + sizeof(u64);
}

void *InternalRealloc(void *addr, uptr size, InternalAllocatorCache *cache) {
  if (!addr)
    return InternalAlloc(size, cache);
  if (size + sizeof(u64) < size)
    return nullptr;
  addr = (char*)addr - sizeof(u64);
  size = size + sizeof(u64);
  CHECK_EQ(kBlockMagic, ((u64*)addr)[0]);
  void *p = RawInternalRealloc(addr, size, cache);
  if (UNLIKELY(!p))
    return DieOnFailure::OnOOM();
  return (char*)p + sizeof(u64);
}

void *InternalCalloc(uptr count, uptr size, InternalAllocatorCache *cache) {
  if (UNLIKELY(CheckForCallocOverflow(count, size)))
    return DieOnFailure::OnBadRequest();
  void *p = InternalAlloc(count * size, cache);
  if (LIKELY(p))
    internal_memset(p, 0, count * size);
  return p;
}

void InternalFree(void *addr, InternalAllocatorCache *cache) {
  if (!addr)
    return;
  addr = (char*)addr - sizeof(u64);
  CHECK_EQ(kBlockMagic, ((u64*)addr)[0]);
  ((u64*)addr)[0] = 0;
  RawInternalFree(addr, cache);
}

// LowLevelAllocator
constexpr uptr kLowLevelAllocatorDefaultAlignment = 8;
static uptr low_level_alloc_min_alignment = kLowLevelAllocatorDefaultAlignment;
static LowLevelAllocateCallback low_level_alloc_callback;

void *LowLevelAllocator::Allocate(uptr size) {
  // Align allocation size.
  size = RoundUpTo(size, low_level_alloc_min_alignment);
  if (allocated_end_ - allocated_current_ < (sptr)size) {
    uptr size_to_allocate = Max(size, GetPageSizeCached());
    allocated_current_ =
        (char*)MmapOrDie(size_to_allocate, __func__);
    allocated_end_ = allocated_current_ + size_to_allocate;
    if (low_level_alloc_callback) {
      low_level_alloc_callback((uptr)allocated_current_,
                               size_to_allocate);
    }
  }
  CHECK(allocated_end_ - allocated_current_ >= (sptr)size);
  void *res = allocated_current_;
  allocated_current_ += size;
  return res;
}

void SetLowLevelAllocateMinAlignment(uptr alignment) {
  CHECK(IsPowerOfTwo(alignment));
  low_level_alloc_min_alignment = Max(alignment, low_level_alloc_min_alignment);
}

void SetLowLevelAllocateCallback(LowLevelAllocateCallback callback) {
  low_level_alloc_callback = callback;
}

static atomic_uint8_t allocator_out_of_memory = {0};
static atomic_uint8_t allocator_may_return_null = {0};

bool IsAllocatorOutOfMemory() {
  return atomic_load_relaxed(&allocator_out_of_memory);
}

void SetAllocatorOutOfMemory() {
  atomic_store_relaxed(&allocator_out_of_memory, 1);
}

// Prints error message and kills the program.
void NORETURN ReportAllocatorCannotReturnNull() {
  Report("%s's allocator is terminating the process instead of returning 0\n",
         SanitizerToolName);
  Report("If you don't like this behavior set allocator_may_return_null=1\n");
  CHECK(0);
  Die();
}

bool AllocatorMayReturnNull() {
  return atomic_load(&allocator_may_return_null, memory_order_relaxed);
}

void SetAllocatorMayReturnNull(bool may_return_null) {
  atomic_store(&allocator_may_return_null, may_return_null,
               memory_order_relaxed);
}

void *ReturnNullOrDieOnFailure::OnBadRequest() {
  if (AllocatorMayReturnNull())
    return nullptr;
  ReportAllocatorCannotReturnNull();
}

void *ReturnNullOrDieOnFailure::OnOOM() {
  atomic_store_relaxed(&allocator_out_of_memory, 1);
  if (AllocatorMayReturnNull())
    return nullptr;
  ReportAllocatorCannotReturnNull();
}

void NORETURN *DieOnFailure::OnBadRequest() {
  ReportAllocatorCannotReturnNull();
}

void NORETURN *DieOnFailure::OnOOM() {
  atomic_store_relaxed(&allocator_out_of_memory, 1);
  ReportAllocatorCannotReturnNull();
}

// Prints hint message.
void PrintHintAllocatorCannotReturnNull(const char *options_name) {
  Report("HINT: if you don't care about these errors you may set "
         "%s=allocator_may_return_null=1\n", options_name);
}

} // namespace __sanitizer
//===-- sanitizer_common.cc -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_allocator_interface.h"
#include "sanitizer_allocator_internal.h"
#include "sanitizer_atomic.h"
#include "sanitizer_flags.h"
#include "sanitizer_libc.h"
#include "sanitizer_placement_new.h"

namespace __sanitizer {

const char *SanitizerToolName = "SanitizerTool";

atomic_uint32_t current_verbosity;
uptr PageSizeCached;
u32 NumberOfCPUsCached;

// PID of the tracer task in StopTheWorld. It shares the address space with the
// main process, but has a different PID and thus requires special handling.
uptr stoptheworld_tracer_pid = 0;
// Cached pid of parent process - if the parent process dies, we want to keep
// writing to the same log file.
uptr stoptheworld_tracer_ppid = 0;

void NORETURN ReportMmapFailureAndDie(uptr size, const char *mem_type,
                                      const char *mmap_type, error_t err,
                                      bool raw_report) {
  static int recursion_count;
  if (raw_report || recursion_count) {
    // If raw report is requested or we went into recursion, just die.
    // The Report() and CHECK calls below may call mmap recursively and fail.
    RawWrite("ERROR: Failed to mmap\n");
    Die();
  }
  recursion_count++;
  Report("ERROR: %s failed to "
         "%s 0x%zx (%zd) bytes of %s (error code: %d)\n",
         SanitizerToolName, mmap_type, size, size, mem_type, err);
#if !SANITIZER_GO
  DumpProcessMap();
#endif
  UNREACHABLE("unable to mmap");
}

typedef bool UptrComparisonFunction(const uptr &a, const uptr &b);
typedef bool U32ComparisonFunction(const u32 &a, const u32 &b);

template<class T>
static inline bool CompareLess(const T &a, const T &b) {
  return a < b;
}

void SortArray(uptr *array, uptr size) {
  InternalSort<uptr*, UptrComparisonFunction>(&array, size, CompareLess);
}

void SortArray(u32 *array, uptr size) {
  InternalSort<u32*, U32ComparisonFunction>(&array, size, CompareLess);
}

const char *StripPathPrefix(const char *filepath,
                            const char *strip_path_prefix) {
  if (!filepath) return nullptr;
  if (!strip_path_prefix) return filepath;
  const char *res = filepath;
  if (const char *pos = internal_strstr(filepath, strip_path_prefix))
    res = pos + internal_strlen(strip_path_prefix);
  if (res[0] == '.' && res[1] == '/')
    res += 2;
  return res;
}

const char *StripModuleName(const char *module) {
  if (!module)
    return nullptr;
  if (SANITIZER_WINDOWS) {
    // On Windows, both slash and backslash are possible.
    // Pick the one that goes last.
    if (const char *bslash_pos = internal_strrchr(module, '\\'))
      return StripModuleName(bslash_pos + 1);
  }
  if (const char *slash_pos = internal_strrchr(module, '/')) {
    return slash_pos + 1;
  }
  return module;
}

void ReportErrorSummary(const char *error_message, const char *alt_tool_name) {
  if (!common_flags()->print_summary)
    return;
  InternalScopedString buff(kMaxSummaryLength);
  buff.append("SUMMARY: %s: %s",
              alt_tool_name ? alt_tool_name : SanitizerToolName, error_message);
  __sanitizer_report_error_summary(buff.data());
}

// Removes the ANSI escape sequences from the input string (in-place).
void RemoveANSIEscapeSequencesFromString(char *str) {
  if (!str)
    return;

  // We are going to remove the escape sequences in place.
  char *s = str;
  char *z = str;
  while (*s != '\0') {
    CHECK_GE(s, z);
    // Skip over ANSI escape sequences with pointer 's'.
    if (*s == '\033' && *(s + 1) == '[') {
      s = internal_strchrnul(s, 'm');
      if (*s == '\0') {
        break;
      }
      s++;
      continue;
    }
    // 's' now points at a character we want to keep. Copy over the buffer
    // content if the escape sequence has been perviously skipped andadvance
    // both pointers.
    if (s != z)
      *z = *s;

    // If we have not seen an escape sequence, just advance both pointers.
    z++;
    s++;
  }

  // Null terminate the string.
  *z = '\0';
}

void LoadedModule::set(const char *module_name, uptr base_address) {
  clear();
  full_name_ = internal_strdup(module_name);
  base_address_ = base_address;
}

void LoadedModule::set(const char *module_name, uptr base_address,
                       ModuleArch arch, u8 uuid[kModuleUUIDSize],
                       bool instrumented) {
  set(module_name, base_address);
  arch_ = arch;
  internal_memcpy(uuid_, uuid, sizeof(uuid_));
  instrumented_ = instrumented;
}

void LoadedModule::clear() {
  InternalFree(full_name_);
  base_address_ = 0;
  max_executable_address_ = 0;
  full_name_ = nullptr;
  arch_ = kModuleArchUnknown;
  internal_memset(uuid_, 0, kModuleUUIDSize);
  instrumented_ = false;
  while (!ranges_.empty()) {
    AddressRange *r = ranges_.front();
    ranges_.pop_front();
    InternalFree(r);
  }
}

void LoadedModule::addAddressRange(uptr beg, uptr end, bool executable,
                                   bool writable, const char *name) {
  void *mem = InternalAlloc(sizeof(AddressRange));
  AddressRange *r =
      new(mem) AddressRange(beg, end, executable, writable, name);
  ranges_.push_back(r);
  if (executable && end > max_executable_address_)
    max_executable_address_ = end;
}

bool LoadedModule::containsAddress(uptr address) const {
  for (const AddressRange &r : ranges()) {
    if (r.beg <= address && address < r.end)
      return true;
  }
  return false;
}

static atomic_uintptr_t g_total_mmaped;

void IncreaseTotalMmap(uptr size) {
  if (!common_flags()->mmap_limit_mb) return;
  uptr total_mmaped =
      atomic_fetch_add(&g_total_mmaped, size, memory_order_relaxed) + size;
  // Since for now mmap_limit_mb is not a user-facing flag, just kill
  // a program. Use RAW_CHECK to avoid extra mmaps in reporting.
  RAW_CHECK((total_mmaped >> 20) < common_flags()->mmap_limit_mb);
}

void DecreaseTotalMmap(uptr size) {
  if (!common_flags()->mmap_limit_mb) return;
  atomic_fetch_sub(&g_total_mmaped, size, memory_order_relaxed);
}

bool TemplateMatch(const char *templ, const char *str) {
  if ((!str) || str[0] == 0)
    return false;
  bool start = false;
  if (templ && templ[0] == '^') {
    start = true;
    templ++;
  }
  bool asterisk = false;
  while (templ && templ[0]) {
    if (templ[0] == '*') {
      templ++;
      start = false;
      asterisk = true;
      continue;
    }
    if (templ[0] == '$')
      return str[0] == 0 || asterisk;
    if (str[0] == 0)
      return false;
    char *tpos = (char*)internal_strchr(templ, '*');
    char *tpos1 = (char*)internal_strchr(templ, '$');
    if ((!tpos) || (tpos1 && tpos1 < tpos))
      tpos = tpos1;
    if (tpos)
      tpos[0] = 0;
    const char *str0 = str;
    const char *spos = internal_strstr(str, templ);
    str = spos + internal_strlen(templ);
    templ = tpos;
    if (tpos)
      tpos[0] = tpos == tpos1 ? '$' : '*';
    if (!spos)
      return false;
    if (start && spos != str0)
      return false;
    start = false;
    asterisk = false;
  }
  return true;
}

static char binary_name_cache_str[kMaxPathLength];
static char process_name_cache_str[kMaxPathLength];

const char *GetProcessName() {
  return process_name_cache_str;
}

static uptr ReadProcessName(/*out*/ char *buf, uptr buf_len) {
  ReadLongProcessName(buf, buf_len);
  char *s = const_cast<char *>(StripModuleName(buf));
  uptr len = internal_strlen(s);
  if (s != buf) {
    internal_memmove(buf, s, len);
    buf[len] = '\0';
  }
  return len;
}

void UpdateProcessName() {
  ReadProcessName(process_name_cache_str, sizeof(process_name_cache_str));
}

// Call once to make sure that binary_name_cache_str is initialized
void CacheBinaryName() {
  if (binary_name_cache_str[0] != '\0')
    return;
  ReadBinaryName(binary_name_cache_str, sizeof(binary_name_cache_str));
  ReadProcessName(process_name_cache_str, sizeof(process_name_cache_str));
}

uptr ReadBinaryNameCached(/*out*/char *buf, uptr buf_len) {
  CacheBinaryName();
  uptr name_len = internal_strlen(binary_name_cache_str);
  name_len = (name_len < buf_len - 1) ? name_len : buf_len - 1;
  if (buf_len == 0)
    return 0;
  internal_memcpy(buf, binary_name_cache_str, name_len);
  buf[name_len] = '\0';
  return name_len;
}

void PrintCmdline() {
  char **argv = GetArgv();
  if (!argv) return;
  Printf("\nCommand: ");
  for (uptr i = 0; argv[i]; ++i)
    Printf("%s ", argv[i]);
  Printf("\n\n");
}

// Malloc hooks.
static const int kMaxMallocFreeHooks = 5;
struct MallocFreeHook {
  void (*malloc_hook)(const void *, uptr);
  void (*free_hook)(const void *);
};

static MallocFreeHook MFHooks[kMaxMallocFreeHooks];

void RunMallocHooks(const void *ptr, uptr size) {
  for (int i = 0; i < kMaxMallocFreeHooks; i++) {
    auto hook = MFHooks[i].malloc_hook;
    if (!hook) return;
    hook(ptr, size);
  }
}

void RunFreeHooks(const void *ptr) {
  for (int i = 0; i < kMaxMallocFreeHooks; i++) {
    auto hook = MFHooks[i].free_hook;
    if (!hook) return;
    hook(ptr);
  }
}

static int InstallMallocFreeHooks(void (*malloc_hook)(const void *, uptr),
                                  void (*free_hook)(const void *)) {
  if (!malloc_hook || !free_hook) return 0;
  for (int i = 0; i < kMaxMallocFreeHooks; i++) {
    if (MFHooks[i].malloc_hook == nullptr) {
      MFHooks[i].malloc_hook = malloc_hook;
      MFHooks[i].free_hook = free_hook;
      return i + 1;
    }
  }
  return 0;
}

} // namespace __sanitizer

using namespace __sanitizer;  // NOLINT

extern "C" {
SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_report_error_summary,
                             const char *error_summary) {
  Printf("%s\n", error_summary);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __sanitizer_acquire_crash_state() {
  static atomic_uint8_t in_crash_state = {};
  return !atomic_exchange(&in_crash_state, 1, memory_order_relaxed);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_set_death_callback(void (*callback)(void)) {
  SetUserDieCallback(callback);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __sanitizer_install_malloc_and_free_hooks(void (*malloc_hook)(const void *,
                                                                  uptr),
                                              void (*free_hook)(const void *)) {
  return InstallMallocFreeHooks(malloc_hook, free_hook);
}
} // extern "C"
//===-- sanitizer_common_libcdep.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator_interface.h"
#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_procmaps.h"


namespace __sanitizer {

static void (*SoftRssLimitExceededCallback)(bool exceeded);
void SetSoftRssLimitExceededCallback(void (*Callback)(bool exceeded)) {
  CHECK_EQ(SoftRssLimitExceededCallback, nullptr);
  SoftRssLimitExceededCallback = Callback;
}

#if SANITIZER_LINUX && !SANITIZER_GO
// Weak default implementation for when sanitizer_stackdepot is not linked in.
SANITIZER_WEAK_ATTRIBUTE StackDepotStats *StackDepotGetStats() {
  return nullptr;
}

void BackgroundThread(void *arg) {
  const uptr hard_rss_limit_mb = common_flags()->hard_rss_limit_mb;
  const uptr soft_rss_limit_mb = common_flags()->soft_rss_limit_mb;
  const bool heap_profile = common_flags()->heap_profile;
  uptr prev_reported_rss = 0;
  uptr prev_reported_stack_depot_size = 0;
  bool reached_soft_rss_limit = false;
  uptr rss_during_last_reported_profile = 0;
  while (true) {
    SleepForMillis(100);
    const uptr current_rss_mb = GetRSS() >> 20;
    if (Verbosity()) {
      // If RSS has grown 10% since last time, print some information.
      if (prev_reported_rss * 11 / 10 < current_rss_mb) {
        Printf("%s: RSS: %zdMb\n", SanitizerToolName, current_rss_mb);
        prev_reported_rss = current_rss_mb;
      }
      // If stack depot has grown 10% since last time, print it too.
      StackDepotStats *stack_depot_stats = StackDepotGetStats();
      if (stack_depot_stats) {
        if (prev_reported_stack_depot_size * 11 / 10 <
            stack_depot_stats->allocated) {
          Printf("%s: StackDepot: %zd ids; %zdM allocated\n",
                 SanitizerToolName,
                 stack_depot_stats->n_uniq_ids,
                 stack_depot_stats->allocated >> 20);
          prev_reported_stack_depot_size = stack_depot_stats->allocated;
        }
      }
    }
    // Check RSS against the limit.
    if (hard_rss_limit_mb && hard_rss_limit_mb < current_rss_mb) {
      Report("%s: hard rss limit exhausted (%zdMb vs %zdMb)\n",
             SanitizerToolName, hard_rss_limit_mb, current_rss_mb);
      DumpProcessMap();
      Die();
    }
    if (soft_rss_limit_mb) {
      if (soft_rss_limit_mb < current_rss_mb && !reached_soft_rss_limit) {
        reached_soft_rss_limit = true;
        Report("%s: soft rss limit exhausted (%zdMb vs %zdMb)\n",
               SanitizerToolName, soft_rss_limit_mb, current_rss_mb);
        if (SoftRssLimitExceededCallback)
          SoftRssLimitExceededCallback(true);
      } else if (soft_rss_limit_mb >= current_rss_mb &&
                 reached_soft_rss_limit) {
        reached_soft_rss_limit = false;
        if (SoftRssLimitExceededCallback)
          SoftRssLimitExceededCallback(false);
      }
    }
    if (heap_profile &&
        current_rss_mb > rss_during_last_reported_profile * 1.1) {
      Printf("\n\nHEAP PROFILE at RSS %zdMb\n", current_rss_mb);
      __sanitizer_print_memory_profile(90, 20);
      rss_during_last_reported_profile = current_rss_mb;
    }
  }
}
#endif

void WriteToSyslog(const char *msg) {
  InternalScopedString msg_copy(kErrorMessageBufferSize);
  msg_copy.append("%s", msg);
  char *p = msg_copy.data();
  char *q;

  // Print one line at a time.
  // syslog, at least on Android, has an implicit message length limit.
  do {
    q = internal_strchr(p, '\n');
    if (q)
      *q = '\0';
    WriteOneLineToSyslog(p);
    if (q)
      p = q + 1;
  } while (q);
}

void MaybeStartBackgroudThread() {
#if SANITIZER_LINUX && \
    !SANITIZER_GO  // Need to implement/test on other platforms.
  // Start the background thread if one of the rss limits is given.
  if (!common_flags()->hard_rss_limit_mb &&
      !common_flags()->soft_rss_limit_mb &&
      !common_flags()->heap_profile) return;
  if (!&real_pthread_create) return;  // Can't spawn the thread anyway.
  internal_start_thread(BackgroundThread, nullptr);
#endif
}

static void (*sandboxing_callback)();
void SetSandboxingCallback(void (*f)()) {
  sandboxing_callback = f;
}

}  // namespace __sanitizer

SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_sandbox_on_notify,
                             __sanitizer_sandbox_arguments *args) {
  __sanitizer::PlatformPrepareForSandboxing(args);
  if (__sanitizer::sandboxing_callback)
    __sanitizer::sandboxing_callback();
}
//===-- sanitizer_deadlock_detector2.cc -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Deadlock detector implementation based on adjacency lists.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_deadlock_detector_interface.h"
#include "sanitizer_common.h"
#include "sanitizer_allocator_internal.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_mutex.h"

#if SANITIZER_DEADLOCK_DETECTOR_VERSION == 2

namespace __sanitizer {

const int kMaxNesting = 64;
const u32 kNoId = -1;
const u32 kEndId = -2;
const int kMaxLink = 8;
const int kL1Size = 1024;
const int kL2Size = 1024;
const int kMaxMutex = kL1Size * kL2Size;

struct Id {
  u32 id;
  u32 seq;

  explicit Id(u32 id = 0, u32 seq = 0)
      : id(id)
      , seq(seq) {
  }
};

struct Link {
  u32 id;
  u32 seq;
  u32 tid;
  u32 stk0;
  u32 stk1;

  explicit Link(u32 id = 0, u32 seq = 0, u32 tid = 0, u32 s0 = 0, u32 s1 = 0)
      : id(id)
      , seq(seq)
      , tid(tid)
      , stk0(s0)
      , stk1(s1) {
  }
};

struct DDPhysicalThread {
  DDReport rep;
  bool report_pending;
  bool visited[kMaxMutex];
  Link pending[kMaxMutex];
  Link path[kMaxMutex];
};

struct ThreadMutex {
  u32 id;
  u32 stk;
};

struct DDLogicalThread {
  u64         ctx;
  ThreadMutex locked[kMaxNesting];
  int         nlocked;
};

struct Mutex {
  StaticSpinMutex mtx;
  u32 seq;
  int nlink;
  Link link[kMaxLink];
};

struct DD : public DDetector {
  explicit DD(const DDFlags *flags);

  DDPhysicalThread* CreatePhysicalThread();
  void DestroyPhysicalThread(DDPhysicalThread *pt);

  DDLogicalThread* CreateLogicalThread(u64 ctx);
  void DestroyLogicalThread(DDLogicalThread *lt);

  void MutexInit(DDCallback *cb, DDMutex *m);
  void MutexBeforeLock(DDCallback *cb, DDMutex *m, bool wlock);
  void MutexAfterLock(DDCallback *cb, DDMutex *m, bool wlock,
      bool trylock);
  void MutexBeforeUnlock(DDCallback *cb, DDMutex *m, bool wlock);
  void MutexDestroy(DDCallback *cb, DDMutex *m);

  DDReport *GetReport(DDCallback *cb);

  void CycleCheck(DDPhysicalThread *pt, DDLogicalThread *lt, DDMutex *mtx);
  void Report(DDPhysicalThread *pt, DDLogicalThread *lt, int npath);
  u32 allocateId(DDCallback *cb);
  Mutex *getMutex(u32 id);
  u32 getMutexId(Mutex *m);

  DDFlags flags;

  Mutex* mutex[kL1Size];

  SpinMutex mtx;
  InternalMmapVector<u32> free_id;
  int id_gen;
};

DDetector *DDetector::Create(const DDFlags *flags) {
  (void)flags;
  void *mem = MmapOrDie(sizeof(DD), "deadlock detector");
  return new(mem) DD(flags);
}

DD::DD(const DDFlags *flags)
    : flags(*flags)
    , free_id(1024) {
  id_gen = 0;
}

DDPhysicalThread* DD::CreatePhysicalThread() {
  DDPhysicalThread *pt = (DDPhysicalThread*)MmapOrDie(sizeof(DDPhysicalThread),
      "deadlock detector (physical thread)");
  return pt;
}

void DD::DestroyPhysicalThread(DDPhysicalThread *pt) {
  pt->~DDPhysicalThread();
  UnmapOrDie(pt, sizeof(DDPhysicalThread));
}

DDLogicalThread* DD::CreateLogicalThread(u64 ctx) {
  DDLogicalThread *lt = (DDLogicalThread*)InternalAlloc(
      sizeof(DDLogicalThread));
  lt->ctx = ctx;
  lt->nlocked = 0;
  return lt;
}

void DD::DestroyLogicalThread(DDLogicalThread *lt) {
  lt->~DDLogicalThread();
  InternalFree(lt);
}

void DD::MutexInit(DDCallback *cb, DDMutex *m) {
  VPrintf(2, "#%llu: DD::MutexInit(%p)\n", cb->lt->ctx, m);
  m->id = kNoId;
  m->recursion = 0;
  atomic_store(&m->owner, 0, memory_order_relaxed);
}

Mutex *DD::getMutex(u32 id) {
  return &mutex[id / kL2Size][id % kL2Size];
}

u32 DD::getMutexId(Mutex *m) {
  for (int i = 0; i < kL1Size; i++) {
    Mutex *tab = mutex[i];
    if (tab == 0)
      break;
    if (m >= tab && m < tab + kL2Size)
      return i * kL2Size + (m - tab);
  }
  return -1;
}

u32 DD::allocateId(DDCallback *cb) {
  u32 id = -1;
  SpinMutexLock l(&mtx);
  if (free_id.size() > 0) {
    id = free_id.back();
    free_id.pop_back();
  } else {
    CHECK_LT(id_gen, kMaxMutex);
    if ((id_gen % kL2Size) == 0) {
      mutex[id_gen / kL2Size] = (Mutex*)MmapOrDie(kL2Size * sizeof(Mutex),
          "deadlock detector (mutex table)");
    }
    id = id_gen++;
  }
  CHECK_LE(id, kMaxMutex);
  VPrintf(3, "#%llu: DD::allocateId assign id %d\n", cb->lt->ctx, id);
  return id;
}

void DD::MutexBeforeLock(DDCallback *cb, DDMutex *m, bool wlock) {
  VPrintf(2, "#%llu: DD::MutexBeforeLock(%p, wlock=%d) nlocked=%d\n",
      cb->lt->ctx, m, wlock, cb->lt->nlocked);
  DDPhysicalThread *pt = cb->pt;
  DDLogicalThread *lt = cb->lt;

  uptr owner = atomic_load(&m->owner, memory_order_relaxed);
  if (owner == (uptr)cb->lt) {
    VPrintf(3, "#%llu: DD::MutexBeforeLock recursive\n",
        cb->lt->ctx);
    return;
  }

  CHECK_LE(lt->nlocked, kMaxNesting);

  // FIXME(dvyukov): don't allocate id if lt->nlocked == 0?
  if (m->id == kNoId)
    m->id = allocateId(cb);

  ThreadMutex *tm = &lt->locked[lt->nlocked++];
  tm->id = m->id;
  if (flags.second_deadlock_stack)
    tm->stk = cb->Unwind();
  if (lt->nlocked == 1) {
    VPrintf(3, "#%llu: DD::MutexBeforeLock first mutex\n",
        cb->lt->ctx);
    return;
  }

  bool added = false;
  Mutex *mtx = getMutex(m->id);
  for (int i = 0; i < lt->nlocked - 1; i++) {
    u32 id1 = lt->locked[i].id;
    u32 stk1 = lt->locked[i].stk;
    Mutex *mtx1 = getMutex(id1);
    SpinMutexLock l(&mtx1->mtx);
    if (mtx1->nlink == kMaxLink) {
      // FIXME(dvyukov): check stale links
      continue;
    }
    int li = 0;
    for (; li < mtx1->nlink; li++) {
      Link *link = &mtx1->link[li];
      if (link->id == m->id) {
        if (link->seq != mtx->seq) {
          link->seq = mtx->seq;
          link->tid = lt->ctx;
          link->stk0 = stk1;
          link->stk1 = cb->Unwind();
          added = true;
          VPrintf(3, "#%llu: DD::MutexBeforeLock added %d->%d link\n",
              cb->lt->ctx, getMutexId(mtx1), m->id);
        }
        break;
      }
    }
    if (li == mtx1->nlink) {
      // FIXME(dvyukov): check stale links
      Link *link = &mtx1->link[mtx1->nlink++];
      link->id = m->id;
      link->seq = mtx->seq;
      link->tid = lt->ctx;
      link->stk0 = stk1;
      link->stk1 = cb->Unwind();
      added = true;
      VPrintf(3, "#%llu: DD::MutexBeforeLock added %d->%d link\n",
          cb->lt->ctx, getMutexId(mtx1), m->id);
    }
  }

  if (!added || mtx->nlink == 0) {
    VPrintf(3, "#%llu: DD::MutexBeforeLock don't check\n",
        cb->lt->ctx);
    return;
  }

  CycleCheck(pt, lt, m);
}

void DD::MutexAfterLock(DDCallback *cb, DDMutex *m, bool wlock,
    bool trylock) {
  VPrintf(2, "#%llu: DD::MutexAfterLock(%p, wlock=%d, try=%d) nlocked=%d\n",
      cb->lt->ctx, m, wlock, trylock, cb->lt->nlocked);
  DDLogicalThread *lt = cb->lt;

  uptr owner = atomic_load(&m->owner, memory_order_relaxed);
  if (owner == (uptr)cb->lt) {
    VPrintf(3, "#%llu: DD::MutexAfterLock recursive\n", cb->lt->ctx);
    CHECK(wlock);
    m->recursion++;
    return;
  }
  CHECK_EQ(owner, 0);
  if (wlock) {
    VPrintf(3, "#%llu: DD::MutexAfterLock set owner\n", cb->lt->ctx);
    CHECK_EQ(m->recursion, 0);
    m->recursion = 1;
    atomic_store(&m->owner, (uptr)cb->lt, memory_order_relaxed);
  }

  if (!trylock)
    return;

  CHECK_LE(lt->nlocked, kMaxNesting);
  if (m->id == kNoId)
    m->id = allocateId(cb);
  ThreadMutex *tm = &lt->locked[lt->nlocked++];
  tm->id = m->id;
  if (flags.second_deadlock_stack)
    tm->stk = cb->Unwind();
}

void DD::MutexBeforeUnlock(DDCallback *cb, DDMutex *m, bool wlock) {
  VPrintf(2, "#%llu: DD::MutexBeforeUnlock(%p, wlock=%d) nlocked=%d\n",
      cb->lt->ctx, m, wlock, cb->lt->nlocked);
  DDLogicalThread *lt = cb->lt;

  uptr owner = atomic_load(&m->owner, memory_order_relaxed);
  if (owner == (uptr)cb->lt) {
    VPrintf(3, "#%llu: DD::MutexBeforeUnlock recursive\n", cb->lt->ctx);
    if (--m->recursion > 0)
      return;
    VPrintf(3, "#%llu: DD::MutexBeforeUnlock reset owner\n", cb->lt->ctx);
    atomic_store(&m->owner, 0, memory_order_relaxed);
  }
  CHECK_NE(m->id, kNoId);
  int last = lt->nlocked - 1;
  for (int i = last; i >= 0; i--) {
    if (cb->lt->locked[i].id == m->id) {
      lt->locked[i] = lt->locked[last];
      lt->nlocked--;
      break;
    }
  }
}

void DD::MutexDestroy(DDCallback *cb, DDMutex *m) {
  VPrintf(2, "#%llu: DD::MutexDestroy(%p)\n",
      cb->lt->ctx, m);
  DDLogicalThread *lt = cb->lt;

  if (m->id == kNoId)
    return;

  // Remove the mutex from lt->locked if there.
  int last = lt->nlocked - 1;
  for (int i = last; i >= 0; i--) {
    if (lt->locked[i].id == m->id) {
      lt->locked[i] = lt->locked[last];
      lt->nlocked--;
      break;
    }
  }

  // Clear and invalidate the mutex descriptor.
  {
    Mutex *mtx = getMutex(m->id);
    SpinMutexLock l(&mtx->mtx);
    mtx->seq++;
    mtx->nlink = 0;
  }

  // Return id to cache.
  {
    SpinMutexLock l(&mtx);
    free_id.push_back(m->id);
  }
}

void DD::CycleCheck(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m) {
  internal_memset(pt->visited, 0, sizeof(pt->visited));
  int npath = 0;
  int npending = 0;
  {
    Mutex *mtx = getMutex(m->id);
    SpinMutexLock l(&mtx->mtx);
    for (int li = 0; li < mtx->nlink; li++)
      pt->pending[npending++] = mtx->link[li];
  }
  while (npending > 0) {
    Link link = pt->pending[--npending];
    if (link.id == kEndId) {
      npath--;
      continue;
    }
    if (pt->visited[link.id])
      continue;
    Mutex *mtx1 = getMutex(link.id);
    SpinMutexLock l(&mtx1->mtx);
    if (mtx1->seq != link.seq)
      continue;
    pt->visited[link.id] = true;
    if (mtx1->nlink == 0)
      continue;
    pt->path[npath++] = link;
    pt->pending[npending++] = Link(kEndId);
    if (link.id == m->id)
      return Report(pt, lt, npath);  // Bingo!
    for (int li = 0; li < mtx1->nlink; li++) {
      Link *link1 = &mtx1->link[li];
      // Mutex *mtx2 = getMutex(link->id);
      // FIXME(dvyukov): fast seq check
      // FIXME(dvyukov): fast nlink != 0 check
      // FIXME(dvyukov): fast pending check?
      // FIXME(dvyukov): npending can be larger than kMaxMutex
      pt->pending[npending++] = *link1;
    }
  }
}

void DD::Report(DDPhysicalThread *pt, DDLogicalThread *lt, int npath) {
  DDReport *rep = &pt->rep;
  rep->n = npath;
  for (int i = 0; i < npath; i++) {
    Link *link = &pt->path[i];
    Link *link0 = &pt->path[i ? i - 1 : npath - 1];
    rep->loop[i].thr_ctx = link->tid;
    rep->loop[i].mtx_ctx0 = link0->id;
    rep->loop[i].mtx_ctx1 = link->id;
    rep->loop[i].stk[0] = flags.second_deadlock_stack ? link->stk0 : 0;
    rep->loop[i].stk[1] = link->stk1;
  }
  pt->report_pending = true;
}

DDReport *DD::GetReport(DDCallback *cb) {
  if (!cb->pt->report_pending)
    return 0;
  cb->pt->report_pending = false;
  return &cb->pt->rep;
}

}  // namespace __sanitizer
#endif  // #if SANITIZER_DEADLOCK_DETECTOR_VERSION == 2
//===-- sanitizer_file.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.  It defines filesystem-related interfaces.  This
// is separate from sanitizer_common.cc so that it's simpler to disable
// all the filesystem support code for a port that doesn't use it.
//
//===---------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if !SANITIZER_FUCHSIA

#include "sanitizer_common.h"
#include "sanitizer_file.h"

namespace __sanitizer {

void CatastrophicErrorWrite(const char *buffer, uptr length) {
  WriteToFile(kStderrFd, buffer, length);
}

StaticSpinMutex report_file_mu;
ReportFile report_file = {&report_file_mu, kStderrFd, "", "", 0};

void RawWrite(const char *buffer) {
  report_file.Write(buffer, internal_strlen(buffer));
}

void ReportFile::ReopenIfNecessary() {
  mu->CheckLocked();
  if (fd == kStdoutFd || fd == kStderrFd) return;

  uptr pid = internal_getpid();
  // If in tracer, use the parent's file.
  if (pid == stoptheworld_tracer_pid)
    pid = stoptheworld_tracer_ppid;
  if (fd != kInvalidFd) {
    // If the report file is already opened by the current process,
    // do nothing. Otherwise the report file was opened by the parent
    // process, close it now.
    if (fd_pid == pid)
      return;
    else
      CloseFile(fd);
  }

  const char *exe_name = GetProcessName();
  if (common_flags()->log_exe_name && exe_name) {
    internal_snprintf(full_path, kMaxPathLength, "%s.%s.%zu", path_prefix,
                      exe_name, pid);
  } else {
    internal_snprintf(full_path, kMaxPathLength, "%s.%zu", path_prefix, pid);
  }
  fd = OpenFile(full_path, WrOnly);
  if (fd == kInvalidFd) {
    const char *ErrorMsgPrefix = "ERROR: Can't open file: ";
    WriteToFile(kStderrFd, ErrorMsgPrefix, internal_strlen(ErrorMsgPrefix));
    WriteToFile(kStderrFd, full_path, internal_strlen(full_path));
    Die();
  }
  fd_pid = pid;
}

void ReportFile::SetReportPath(const char *path) {
  if (!path)
    return;
  uptr len = internal_strlen(path);
  if (len > sizeof(path_prefix) - 100) {
    Report("ERROR: Path is too long: %c%c%c%c%c%c%c%c...\n",
           path[0], path[1], path[2], path[3],
           path[4], path[5], path[6], path[7]);
    Die();
  }

  SpinMutexLock l(mu);
  if (fd != kStdoutFd && fd != kStderrFd && fd != kInvalidFd)
    CloseFile(fd);
  fd = kInvalidFd;
  if (internal_strcmp(path, "stdout") == 0) {
    fd = kStdoutFd;
  } else if (internal_strcmp(path, "stderr") == 0) {
    fd = kStderrFd;
  } else {
    internal_snprintf(path_prefix, kMaxPathLength, "%s", path);
  }
}

bool ReadFileToBuffer(const char *file_name, char **buff, uptr *buff_size,
                      uptr *read_len, uptr max_len, error_t *errno_p) {
  uptr PageSize = GetPageSizeCached();
  uptr kMinFileLen = PageSize;
  *buff = nullptr;
  *buff_size = 0;
  *read_len = 0;
  // The files we usually open are not seekable, so try different buffer sizes.
  for (uptr size = kMinFileLen; size <= max_len; size *= 2) {
    fd_t fd = OpenFile(file_name, RdOnly, errno_p);
    if (fd == kInvalidFd) return false;
    UnmapOrDie(*buff, *buff_size);
    *buff = (char*)MmapOrDie(size, __func__);
    *buff_size = size;
    *read_len = 0;
    // Read up to one page at a time.
    bool reached_eof = false;
    while (*read_len + PageSize <= size) {
      uptr just_read;
      if (!ReadFromFile(fd, *buff + *read_len, PageSize, &just_read, errno_p)) {
        UnmapOrDie(*buff, *buff_size);
        return false;
      }
      if (just_read == 0) {
        reached_eof = true;
        break;
      }
      *read_len += just_read;
    }
    CloseFile(fd);
    if (reached_eof)  // We've read the whole file.
      break;
  }
  return true;
}

static const char kPathSeparator = SANITIZER_WINDOWS ? ';' : ':';

char *FindPathToBinary(const char *name) {
  if (FileExists(name)) {
    return internal_strdup(name);
  }

  const char *path = GetEnv("PATH");
  if (!path)
    return nullptr;
  uptr name_len = internal_strlen(name);
  InternalScopedBuffer<char> buffer(kMaxPathLength);
  const char *beg = path;
  while (true) {
    const char *end = internal_strchrnul(beg, kPathSeparator);
    uptr prefix_len = end - beg;
    if (prefix_len + name_len + 2 <= kMaxPathLength) {
      internal_memcpy(buffer.data(), beg, prefix_len);
      buffer[prefix_len] = '/';
      internal_memcpy(&buffer[prefix_len + 1], name, name_len);
      buffer[prefix_len + 1 + name_len] = '\0';
      if (FileExists(buffer.data()))
        return internal_strdup(buffer.data());
    }
    if (*end == '\0') break;
    beg = end + 1;
  }
  return nullptr;
}

} // namespace __sanitizer

using namespace __sanitizer;  // NOLINT

extern "C" {
void __sanitizer_set_report_path(const char *path) {
  report_file.SetReportPath(path);
}

void __sanitizer_set_report_fd(void *fd) {
  report_file.fd = (fd_t)reinterpret_cast<uptr>(fd);
  report_file.fd_pid = internal_getpid();
}
} // extern "C"

#endif  // !SANITIZER_FUCHSIA
//===-- sanitizer_flag_parser.cc ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_flag_parser.h"

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_flags.h"
#include "sanitizer_flag_parser.h"

namespace __sanitizer {

LowLevelAllocator FlagParser::Alloc;

class UnknownFlags {
  static const int kMaxUnknownFlags = 20;
  const char *unknown_flags_[kMaxUnknownFlags];
  int n_unknown_flags_;

 public:
  void Add(const char *name) {
    CHECK_LT(n_unknown_flags_, kMaxUnknownFlags);
    unknown_flags_[n_unknown_flags_++] = name;
  }

  void Report() {
    if (!n_unknown_flags_) return;
    Printf("WARNING: found %d unrecognized flag(s):\n", n_unknown_flags_);
    for (int i = 0; i < n_unknown_flags_; ++i)
      Printf("    %s\n", unknown_flags_[i]);
    n_unknown_flags_ = 0;
  }
};

UnknownFlags unknown_flags;

void ReportUnrecognizedFlags() {
  unknown_flags.Report();
}

char *FlagParser::ll_strndup(const char *s, uptr n) {
  uptr len = internal_strnlen(s, n);
  char *s2 = (char*)Alloc.Allocate(len + 1);
  internal_memcpy(s2, s, len);
  s2[len] = 0;
  return s2;
}

void FlagParser::PrintFlagDescriptions() {
  Printf("Available flags for %s:\n", SanitizerToolName);
  for (int i = 0; i < n_flags_; ++i)
    Printf("\t%s\n\t\t- %s\n", flags_[i].name, flags_[i].desc);
}

void FlagParser::fatal_error(const char *err) {
  Printf("ERROR: %s\n", err);
  Die();
}

bool FlagParser::is_space(char c) {
  return c == ' ' || c == ',' || c == ':' || c == '\n' || c == '\t' ||
         c == '\r';
}

void FlagParser::skip_whitespace() {
  while (is_space(buf_[pos_])) ++pos_;
}

void FlagParser::parse_flag() {
  uptr name_start = pos_;
  while (buf_[pos_] != 0 && buf_[pos_] != '=' && !is_space(buf_[pos_])) ++pos_;
  if (buf_[pos_] != '=') fatal_error("expected '='");
  char *name = ll_strndup(buf_ + name_start, pos_ - name_start);

  uptr value_start = ++pos_;
  char *value;
  if (buf_[pos_] == '\'' || buf_[pos_] == '"') {
    char quote = buf_[pos_++];
    while (buf_[pos_] != 0 && buf_[pos_] != quote) ++pos_;
    if (buf_[pos_] == 0) fatal_error("unterminated string");
    value = ll_strndup(buf_ + value_start + 1, pos_ - value_start - 1);
    ++pos_; // consume the closing quote
  } else {
    while (buf_[pos_] != 0 && !is_space(buf_[pos_])) ++pos_;
    if (buf_[pos_] != 0 && !is_space(buf_[pos_]))
      fatal_error("expected separator or eol");
    value = ll_strndup(buf_ + value_start, pos_ - value_start);
  }

  bool res = run_handler(name, value);
  if (!res) fatal_error("Flag parsing failed.");
}

void FlagParser::parse_flags() {
  while (true) {
    skip_whitespace();
    if (buf_[pos_] == 0) break;
    parse_flag();
  }

  // Do a sanity check for certain flags.
  if (common_flags_dont_use.malloc_context_size < 1)
    common_flags_dont_use.malloc_context_size = 1;
}

void FlagParser::ParseString(const char *s) {
  if (!s) return;
  // Backup current parser state to allow nested ParseString() calls.
  const char *old_buf_ = buf_;
  uptr old_pos_ = pos_;
  buf_ = s;
  pos_ = 0;

  parse_flags();

  buf_ = old_buf_;
  pos_ = old_pos_;
}

bool FlagParser::ParseFile(const char *path, bool ignore_missing) {
  static const uptr kMaxIncludeSize = 1 << 15;
  char *data;
  uptr data_mapped_size;
  error_t err;
  uptr len;
  if (!ReadFileToBuffer(path, &data, &data_mapped_size, &len,
                        Max(kMaxIncludeSize, GetPageSizeCached()), &err)) {
    if (ignore_missing)
      return true;
    Printf("Failed to read options from '%s': error %d\n", path, err);
    return false;
  }
  ParseString(data);
  UnmapOrDie(data, data_mapped_size);
  return true;
}

bool FlagParser::run_handler(const char *name, const char *value) {
  for (int i = 0; i < n_flags_; ++i) {
    if (internal_strcmp(name, flags_[i].name) == 0)
      return flags_[i].handler->Parse(value);
  }
  // Unrecognized flag. This is not a fatal error, we may print a warning later.
  unknown_flags.Add(name);
  return true;
}

void FlagParser::RegisterHandler(const char *name, FlagHandlerBase *handler,
                                 const char *desc) {
  CHECK_LT(n_flags_, kMaxFlags);
  flags_[n_flags_].name = name;
  flags_[n_flags_].desc = desc;
  flags_[n_flags_].handler = handler;
  ++n_flags_;
}

FlagParser::FlagParser() : n_flags_(0), buf_(nullptr), pos_(0) {
  flags_ = (Flag *)Alloc.Allocate(sizeof(Flag) * kMaxFlags);
}

}  // namespace __sanitizer
//===-- sanitizer_flags.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_flags.h"

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_list.h"
#include "sanitizer_flag_parser.h"

namespace __sanitizer {

CommonFlags common_flags_dont_use;

void CommonFlags::SetDefaults() {
#define COMMON_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "sanitizer_flags.inc"
#undef COMMON_FLAG
}

void CommonFlags::CopyFrom(const CommonFlags &other) {
  internal_memcpy(this, &other, sizeof(*this));
}

// Copy the string from "s" to "out", making the following substitutions:
// %b = binary basename
// %p = pid
void SubstituteForFlagValue(const char *s, char *out, uptr out_size) {
  char *out_end = out + out_size;
  while (*s && out < out_end - 1) {
    if (s[0] != '%') {
      *out++ = *s++;
      continue;
    }
    switch (s[1]) {
      case 'b': {
        const char *base = GetProcessName();
        CHECK(base);
        while (*base && out < out_end - 1)
          *out++ = *base++;
        s += 2; // skip "%b"
        break;
      }
      case 'p': {
        int pid = internal_getpid();
        char buf[32];
        char *buf_pos = buf + 32;
        do {
          *--buf_pos = (pid % 10) + '0';
          pid /= 10;
        } while (pid);
        while (buf_pos < buf + 32 && out < out_end - 1)
          *out++ = *buf_pos++;
        s += 2; // skip "%p"
        break;
      }
      default:
        *out++ = *s++;
        break;
    }
  }
  CHECK(out < out_end - 1);
  *out = '\0';
}

class FlagHandlerInclude : public FlagHandlerBase {
  FlagParser *parser_;
  bool ignore_missing_;

 public:
  explicit FlagHandlerInclude(FlagParser *parser, bool ignore_missing)
      : parser_(parser), ignore_missing_(ignore_missing) {}
  bool Parse(const char *value) final {
    if (internal_strchr(value, '%')) {
      char *buf = (char *)MmapOrDie(kMaxPathLength, "FlagHandlerInclude");
      SubstituteForFlagValue(value, buf, kMaxPathLength);
      bool res = parser_->ParseFile(buf, ignore_missing_);
      UnmapOrDie(buf, kMaxPathLength);
      return res;
    }
    return parser_->ParseFile(value, ignore_missing_);
  }
};

void RegisterIncludeFlags(FlagParser *parser, CommonFlags *cf) {
  FlagHandlerInclude *fh_include = new (FlagParser::Alloc) // NOLINT
      FlagHandlerInclude(parser, /*ignore_missing*/ false);
  parser->RegisterHandler("include", fh_include,
                          "read more options from the given file");
  FlagHandlerInclude *fh_include_if_exists = new (FlagParser::Alloc) // NOLINT
      FlagHandlerInclude(parser, /*ignore_missing*/ true);
  parser->RegisterHandler(
      "include_if_exists", fh_include_if_exists,
      "read more options from the given file (if it exists)");
}

void RegisterCommonFlags(FlagParser *parser, CommonFlags *cf) {
#define COMMON_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &cf->Name);
#include "sanitizer_flags.inc"
#undef COMMON_FLAG

  RegisterIncludeFlags(parser, cf);
}

void InitializeCommonFlags(CommonFlags *cf) {
  // need to record coverage to generate coverage report.
  cf->coverage |= cf->html_cov_report;
  SetVerbosity(cf->verbosity);
}

}  // namespace __sanitizer
//===-- sanitizer_libc.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries. See sanitizer_libc.h for details.
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

s64 internal_atoll(const char *nptr) {
  return internal_simple_strtoll(nptr, nullptr, 10);
}

void *internal_memchr(const void *s, int c, uptr n) {
  const char *t = (const char *)s;
  for (uptr i = 0; i < n; ++i, ++t)
    if (*t == c)
      return reinterpret_cast<void *>(const_cast<char *>(t));
  return nullptr;
}

void *internal_memrchr(const void *s, int c, uptr n) {
  const char *t = (const char *)s;
  void *res = nullptr;
  for (uptr i = 0; i < n; ++i, ++t) {
    if (*t == c) res = reinterpret_cast<void *>(const_cast<char *>(t));
  }
  return res;
}

int internal_memcmp(const void* s1, const void* s2, uptr n) {
  const char *t1 = (const char *)s1;
  const char *t2 = (const char *)s2;
  for (uptr i = 0; i < n; ++i, ++t1, ++t2)
    if (*t1 != *t2)
      return *t1 < *t2 ? -1 : 1;
  return 0;
}

void *internal_memcpy(void *dest, const void *src, uptr n) {
  char *d = (char*)dest;
  const char *s = (const char *)src;
  for (uptr i = 0; i < n; ++i)
    d[i] = s[i];
  return dest;
}

void *internal_memmove(void *dest, const void *src, uptr n) {
  char *d = (char*)dest;
  const char *s = (const char *)src;
  sptr i, signed_n = (sptr)n;
  CHECK_GE(signed_n, 0);
  if (d < s) {
    for (i = 0; i < signed_n; ++i)
      d[i] = s[i];
  } else {
    if (d > s && signed_n > 0)
      for (i = signed_n - 1; i >= 0 ; --i) {
        d[i] = s[i];
      }
  }
  return dest;
}

void *internal_memset(void* s, int c, uptr n) {
  // The next line prevents Clang from making a call to memset() instead of the
  // loop below.
  // FIXME: building the runtime with -ffreestanding is a better idea. However
  // there currently are linktime problems due to PR12396.
  char volatile *t = (char*)s;
  for (uptr i = 0; i < n; ++i, ++t) {
    *t = c;
  }
  return s;
}

uptr internal_strcspn(const char *s, const char *reject) {
  uptr i;
  for (i = 0; s[i]; i++) {
    if (internal_strchr(reject, s[i]))
      return i;
  }
  return i;
}

char* internal_strdup(const char *s) {
  uptr len = internal_strlen(s);
  char *s2 = (char*)InternalAlloc(len + 1);
  internal_memcpy(s2, s, len);
  s2[len] = 0;
  return s2;
}

int internal_strcmp(const char *s1, const char *s2) {
  while (true) {
    unsigned c1 = *s1;
    unsigned c2 = *s2;
    if (c1 != c2) return (c1 < c2) ? -1 : 1;
    if (c1 == 0) break;
    s1++;
    s2++;
  }
  return 0;
}

int internal_strncmp(const char *s1, const char *s2, uptr n) {
  for (uptr i = 0; i < n; i++) {
    unsigned c1 = *s1;
    unsigned c2 = *s2;
    if (c1 != c2) return (c1 < c2) ? -1 : 1;
    if (c1 == 0) break;
    s1++;
    s2++;
  }
  return 0;
}

char* internal_strchr(const char *s, int c) {
  while (true) {
    if (*s == (char)c)
      return const_cast<char *>(s);
    if (*s == 0)
      return nullptr;
    s++;
  }
}

char *internal_strchrnul(const char *s, int c) {
  char *res = internal_strchr(s, c);
  if (!res)
    res = const_cast<char *>(s) + internal_strlen(s);
  return res;
}

char *internal_strrchr(const char *s, int c) {
  const char *res = nullptr;
  for (uptr i = 0; s[i]; i++) {
    if (s[i] == c) res = s + i;
  }
  return const_cast<char *>(res);
}

uptr internal_strlen(const char *s) {
  uptr i = 0;
  while (s[i]) i++;
  return i;
}

uptr internal_strlcat(char *dst, const char *src, uptr maxlen) {
  const uptr srclen = internal_strlen(src);
  const uptr dstlen = internal_strnlen(dst, maxlen);
  if (dstlen == maxlen) return maxlen + srclen;
  if (srclen < maxlen - dstlen) {
    internal_memmove(dst + dstlen, src, srclen + 1);
  } else {
    internal_memmove(dst + dstlen, src, maxlen - dstlen - 1);
    dst[maxlen - 1] = '\0';
  }
  return dstlen + srclen;
}

char *internal_strncat(char *dst, const char *src, uptr n) {
  uptr len = internal_strlen(dst);
  uptr i;
  for (i = 0; i < n && src[i]; i++)
    dst[len + i] = src[i];
  dst[len + i] = 0;
  return dst;
}

uptr internal_strlcpy(char *dst, const char *src, uptr maxlen) {
  const uptr srclen = internal_strlen(src);
  if (srclen < maxlen) {
    internal_memmove(dst, src, srclen + 1);
  } else if (maxlen != 0) {
    internal_memmove(dst, src, maxlen - 1);
    dst[maxlen - 1] = '\0';
  }
  return srclen;
}

char *internal_strncpy(char *dst, const char *src, uptr n) {
  uptr i;
  for (i = 0; i < n && src[i]; i++)
    dst[i] = src[i];
  internal_memset(dst + i, '\0', n - i);
  return dst;
}

uptr internal_strnlen(const char *s, uptr maxlen) {
  uptr i = 0;
  while (i < maxlen && s[i]) i++;
  return i;
}

char *internal_strstr(const char *haystack, const char *needle) {
  // This is O(N^2), but we are not using it in hot places.
  uptr len1 = internal_strlen(haystack);
  uptr len2 = internal_strlen(needle);
  if (len1 < len2) return nullptr;
  for (uptr pos = 0; pos <= len1 - len2; pos++) {
    if (internal_memcmp(haystack + pos, needle, len2) == 0)
      return const_cast<char *>(haystack) + pos;
  }
  return nullptr;
}

s64 internal_simple_strtoll(const char *nptr, char **endptr, int base) {
  CHECK_EQ(base, 10);
  while (IsSpace(*nptr)) nptr++;
  int sgn = 1;
  u64 res = 0;
  bool have_digits = false;
  char *old_nptr = const_cast<char *>(nptr);
  if (*nptr == '+') {
    sgn = 1;
    nptr++;
  } else if (*nptr == '-') {
    sgn = -1;
    nptr++;
  }
  while (IsDigit(*nptr)) {
    res = (res <= UINT64_MAX / 10) ? res * 10 : UINT64_MAX;
    int digit = ((*nptr) - '0');
    res = (res <= UINT64_MAX - digit) ? res + digit : UINT64_MAX;
    have_digits = true;
    nptr++;
  }
  if (endptr) {
    *endptr = (have_digits) ? const_cast<char *>(nptr) : old_nptr;
  }
  if (sgn > 0) {
    return (s64)(Min((u64)INT64_MAX, res));
  } else {
    return (res > INT64_MAX) ? INT64_MIN : ((s64)res * -1);
  }
}

bool mem_is_zero(const char *beg, uptr size) {
  CHECK_LE(size, 1ULL << FIRST_32_SECOND_64(30, 40));  // Sanity check.
  const char *end = beg + size;
  uptr *aligned_beg = (uptr *)RoundUpTo((uptr)beg, sizeof(uptr));
  uptr *aligned_end = (uptr *)RoundDownTo((uptr)end, sizeof(uptr));
  uptr all = 0;
  // Prologue.
  for (const char *mem = beg; mem < (char*)aligned_beg && mem < end; mem++)
    all |= *mem;
  // Aligned loop.
  for (; aligned_beg < aligned_end; aligned_beg++)
    all |= *aligned_beg;
  // Epilogue.
  if ((char*)aligned_end >= beg)
    for (const char *mem = (char*)aligned_end; mem < end; mem++)
      all |= *mem;
  return all == 0;
}

} // namespace __sanitizer
//===-- sanitizer_persistent_allocator.cc -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//
#include "sanitizer_persistent_allocator.h"

namespace __sanitizer {

PersistentAllocator thePersistentAllocator;

}  // namespace __sanitizer
//===-- sanitizer_printf.cc -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
//
// Internal printf function, used inside run-time libraries.
// We can't use libc printf because we intercept some of the functions used
// inside it.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_libc.h"

#include <stdio.h>
#include <stdarg.h>

#if SANITIZER_WINDOWS && defined(_MSC_VER) && _MSC_VER < 1800 &&               \
      !defined(va_copy)
# define va_copy(dst, src) ((dst) = (src))
#endif

namespace __sanitizer {

static int AppendChar(char **buff, const char *buff_end, char c) {
  if (*buff < buff_end) {
    **buff = c;
    (*buff)++;
  }
  return 1;
}

// Appends number in a given base to buffer. If its length is less than
// |minimal_num_length|, it is padded with leading zeroes or spaces, depending
// on the value of |pad_with_zero|.
static int AppendNumber(char **buff, const char *buff_end, u64 absolute_value,
                        u8 base, u8 minimal_num_length, bool pad_with_zero,
                        bool negative, bool uppercase) {
  uptr const kMaxLen = 30;
  RAW_CHECK(base == 10 || base == 16);
  RAW_CHECK(base == 10 || !negative);
  RAW_CHECK(absolute_value || !negative);
  RAW_CHECK(minimal_num_length < kMaxLen);
  int result = 0;
  if (negative && minimal_num_length)
    --minimal_num_length;
  if (negative && pad_with_zero)
    result += AppendChar(buff, buff_end, '-');
  uptr num_buffer[kMaxLen];
  int pos = 0;
  do {
    RAW_CHECK_MSG((uptr)pos < kMaxLen, "AppendNumber buffer overflow");
    num_buffer[pos++] = absolute_value % base;
    absolute_value /= base;
  } while (absolute_value > 0);
  if (pos < minimal_num_length) {
    // Make sure compiler doesn't insert call to memset here.
    internal_memset(&num_buffer[pos], 0,
                    sizeof(num_buffer[0]) * (minimal_num_length - pos));
    pos = minimal_num_length;
  }
  RAW_CHECK(pos > 0);
  pos--;
  for (; pos >= 0 && num_buffer[pos] == 0; pos--) {
    char c = (pad_with_zero || pos == 0) ? '0' : ' ';
    result += AppendChar(buff, buff_end, c);
  }
  if (negative && !pad_with_zero) result += AppendChar(buff, buff_end, '-');
  for (; pos >= 0; pos--) {
    char digit = static_cast<char>(num_buffer[pos]);
    digit = (digit < 10) ? '0' + digit : (uppercase ? 'A' : 'a') + digit - 10;
    result += AppendChar(buff, buff_end, digit);
  }
  return result;
}

static int AppendUnsigned(char **buff, const char *buff_end, u64 num, u8 base,
                          u8 minimal_num_length, bool pad_with_zero,
                          bool uppercase) {
  return AppendNumber(buff, buff_end, num, base, minimal_num_length,
                      pad_with_zero, false /* negative */, uppercase);
}

static int AppendSignedDecimal(char **buff, const char *buff_end, s64 num,
                               u8 minimal_num_length, bool pad_with_zero) {
  bool negative = (num < 0);
  return AppendNumber(buff, buff_end, (u64)(negative ? -num : num), 10,
                      minimal_num_length, pad_with_zero, negative,
                      false /* uppercase */);
}


// Use the fact that explicitly requesting 0 width (%0s) results in UB and
// interpret width == 0 as "no width requested":
// width == 0 - no width requested
// width  < 0 - left-justify s within and pad it to -width chars, if necessary
// width  > 0 - right-justify s, not implemented yet
static int AppendString(char **buff, const char *buff_end, int width,
                        int max_chars, const char *s) {
  if (!s)
    s = "<null>";
  int result = 0;
  for (; *s; s++) {
    if (max_chars >= 0 && result >= max_chars)
      break;
    result += AppendChar(buff, buff_end, *s);
  }
  // Only the left justified strings are supported.
  while (width < -result)
    result += AppendChar(buff, buff_end, ' ');
  return result;
}

static int AppendPointer(char **buff, const char *buff_end, u64 ptr_value) {
  int result = 0;
  result += AppendString(buff, buff_end, 0, -1, "0x");
  result += AppendUnsigned(buff, buff_end, ptr_value, 16,
                           SANITIZER_POINTER_FORMAT_LENGTH,
                           true /* pad_with_zero */, false /* uppercase */);
  return result;
}

int VSNPrintf(char *buff, int buff_length,
              const char *format, va_list args) {
  static const char *kPrintfFormatsHelp =
      "Supported Printf formats: %([0-9]*)?(z|ll)?{d,u,x,X}; %p; "
      "%[-]([0-9]*)?(\\.\\*)?s; %c\n";
  RAW_CHECK(format);
  RAW_CHECK(buff_length > 0);
  const char *buff_end = &buff[buff_length - 1];
  const char *cur = format;
  int result = 0;
  for (; *cur; cur++) {
    if (*cur != '%') {
      result += AppendChar(&buff, buff_end, *cur);
      continue;
    }
    cur++;
    bool left_justified = *cur == '-';
    if (left_justified)
      cur++;
    bool have_width = (*cur >= '0' && *cur <= '9');
    bool pad_with_zero = (*cur == '0');
    int width = 0;
    if (have_width) {
      while (*cur >= '0' && *cur <= '9') {
        width = width * 10 + *cur++ - '0';
      }
    }
    bool have_precision = (cur[0] == '.' && cur[1] == '*');
    int precision = -1;
    if (have_precision) {
      cur += 2;
      precision = va_arg(args, int);
    }
    bool have_z = (*cur == 'z');
    cur += have_z;
    bool have_ll = !have_z && (cur[0] == 'l' && cur[1] == 'l');
    cur += have_ll * 2;
    s64 dval;
    u64 uval;
    const bool have_length = have_z || have_ll;
    const bool have_flags = have_width || have_length;
    // At the moment only %s supports precision and left-justification.
    CHECK(!((precision >= 0 || left_justified) && *cur != 's'));
    switch (*cur) {
      case 'd': {
        dval = have_ll ? va_arg(args, s64)
             : have_z ? va_arg(args, sptr)
             : va_arg(args, int);
        result += AppendSignedDecimal(&buff, buff_end, dval, width,
                                      pad_with_zero);
        break;
      }
      case 'u':
      case 'x':
      case 'X': {
        uval = have_ll ? va_arg(args, u64)
             : have_z ? va_arg(args, uptr)
             : va_arg(args, unsigned);
        bool uppercase = (*cur == 'X');
        result += AppendUnsigned(&buff, buff_end, uval, (*cur == 'u') ? 10 : 16,
                                 width, pad_with_zero, uppercase);
        break;
      }
      case 'p': {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendPointer(&buff, buff_end, va_arg(args, uptr));
        break;
      }
      case 's': {
        RAW_CHECK_MSG(!have_length, kPrintfFormatsHelp);
        // Only left-justified width is supported.
        CHECK(!have_width || left_justified);
        result += AppendString(&buff, buff_end, left_justified ? -width : width,
                               precision, va_arg(args, char*));
        break;
      }
      case 'c': {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendChar(&buff, buff_end, va_arg(args, int));
        break;
      }
      case '%' : {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendChar(&buff, buff_end, '%');
        break;
      }
      default: {
        RAW_CHECK_MSG(false, kPrintfFormatsHelp);
      }
    }
  }
  RAW_CHECK(buff <= buff_end);
  AppendChar(&buff, buff_end + 1, '\0');
  return result;
}

static void (*PrintfAndReportCallback)(const char *);
void SetPrintfAndReportCallback(void (*callback)(const char *)) {
  PrintfAndReportCallback = callback;
}

// Can be overriden in frontend.
#if SANITIZER_GO && defined(TSAN_EXTERNAL_HOOKS)
// Implementation must be defined in frontend.
extern "C" void OnPrint(const char *str);
#else
SANITIZER_INTERFACE_WEAK_DEF(void, OnPrint, const char *str) {
  (void)str;
}
#endif

static void CallPrintfAndReportCallback(const char *str) {
  OnPrint(str);
  if (PrintfAndReportCallback)
    PrintfAndReportCallback(str);
}

static void NOINLINE SharedPrintfCodeNoBuffer(bool append_pid,
                                              char *local_buffer,
                                              int buffer_size,
                                              const char *format,
                                              va_list args) {
  va_list args2;
  va_copy(args2, args);
  const int kLen = 16 * 1024;
  int needed_length;
  char *buffer = local_buffer;
  // First try to print a message using a local buffer, and then fall back to
  // mmaped buffer.
  for (int use_mmap = 0; use_mmap < 2; use_mmap++) {
    if (use_mmap) {
      va_end(args);
      va_copy(args, args2);
      buffer = (char*)MmapOrDie(kLen, "Report");
      buffer_size = kLen;
    }
    needed_length = 0;
    // Check that data fits into the current buffer.
#   define CHECK_NEEDED_LENGTH \
      if (needed_length >= buffer_size) { \
        if (!use_mmap) continue; \
        RAW_CHECK_MSG(needed_length < kLen, \
                      "Buffer in Report is too short!\n"); \
      }
    // Fuchsia's logging infrastructure always keeps track of the logging
    // process, thread, and timestamp, so never prepend such information.
    if (!SANITIZER_FUCHSIA && append_pid) {
      int pid = internal_getpid();
      const char *exe_name = GetProcessName();
      if (common_flags()->log_exe_name && exe_name) {
        needed_length += internal_snprintf(buffer, buffer_size,
                                           "==%s", exe_name);
        CHECK_NEEDED_LENGTH
      }
      needed_length += internal_snprintf(
          buffer + needed_length, buffer_size - needed_length, "==%d==", pid);
      CHECK_NEEDED_LENGTH
    }
    needed_length += VSNPrintf(buffer + needed_length,
                               buffer_size - needed_length, format, args);
    CHECK_NEEDED_LENGTH
    // If the message fit into the buffer, print it and exit.
    break;
#   undef CHECK_NEEDED_LENGTH
  }
  RawWrite(buffer);

  // Remove color sequences from the message.
  RemoveANSIEscapeSequencesFromString(buffer);
  CallPrintfAndReportCallback(buffer);
  LogMessageOnPrintf(buffer);

  // If we had mapped any memory, clean up.
  if (buffer != local_buffer)
    UnmapOrDie((void *)buffer, buffer_size);
  va_end(args2);
}

static void NOINLINE SharedPrintfCode(bool append_pid, const char *format,
                                      va_list args) {
  // |local_buffer| is small enough not to overflow the stack and/or violate
  // the stack limit enforced by TSan (-Wframe-larger-than=512). On the other
  // hand, the bigger the buffer is, the more the chance the error report will
  // fit into it.
  char local_buffer[400];
  SharedPrintfCodeNoBuffer(append_pid, local_buffer, ARRAY_SIZE(local_buffer),
                           format, args);
}

FORMAT(1, 2)
void Printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  SharedPrintfCode(false, format, args);
  va_end(args);
}

// Like Printf, but prints the current PID before the output string.
FORMAT(1, 2)
void Report(const char *format, ...) {
  va_list args;
  va_start(args, format);
  SharedPrintfCode(true, format, args);
  va_end(args);
}

// Writes at most "length" symbols to "buffer" (including trailing '\0').
// Returns the number of symbols that should have been written to buffer
// (not including trailing '\0'). Thus, the string is truncated
// iff return value is not less than "length".
FORMAT(3, 4)
int internal_snprintf(char *buffer, uptr length, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int needed_length = VSNPrintf(buffer, length, format, args);
  va_end(args);
  return needed_length;
}

FORMAT(2, 3)
void InternalScopedString::append(const char *format, ...) {
  CHECK_LT(length_, size());
  va_list args;
  va_start(args, format);
  VSNPrintf(data() + length_, size() - length_, format, args);
  va_end(args);
  length_ += internal_strlen(data() + length_);
  CHECK_LT(length_, size());
}

} // namespace __sanitizer
//===-- sanitizer_suppressions.cc -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Suppression parsing/matching code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_suppressions.h"

#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_file.h"
#include "sanitizer_libc.h"
#include "sanitizer_placement_new.h"

namespace __sanitizer {

SuppressionContext::SuppressionContext(const char *suppression_types[],
                                       int suppression_types_num)
    : suppression_types_(suppression_types),
      suppression_types_num_(suppression_types_num), suppressions_(1),
      can_parse_(true) {
  CHECK_LE(suppression_types_num_, kMaxSuppressionTypes);
  internal_memset(has_suppression_type_, 0, suppression_types_num_);
}

static bool GetPathAssumingFileIsRelativeToExec(const char *file_path,
                                                /*out*/char *new_file_path,
                                                uptr new_file_path_size) {
  InternalScopedString exec(kMaxPathLength);
  if (ReadBinaryNameCached(exec.data(), exec.size())) {
    const char *file_name_pos = StripModuleName(exec.data());
    uptr path_to_exec_len = file_name_pos - exec.data();
    internal_strncat(new_file_path, exec.data(),
                     Min(path_to_exec_len, new_file_path_size - 1));
    internal_strncat(new_file_path, file_path,
                     new_file_path_size - internal_strlen(new_file_path) - 1);
    return true;
  }
  return false;
}

void SuppressionContext::ParseFromFile(const char *filename) {
  if (filename[0] == '\0')
    return;

#if !SANITIZER_FUCHSIA
  // If we cannot find the file, check if its location is relative to
  // the location of the executable.
  InternalScopedString new_file_path(kMaxPathLength);
  if (!FileExists(filename) && !IsAbsolutePath(filename) &&
      GetPathAssumingFileIsRelativeToExec(filename, new_file_path.data(),
                                          new_file_path.size())) {
    filename = new_file_path.data();
  }
#endif  // !SANITIZER_FUCHSIA

  // Read the file.
  VPrintf(1, "%s: reading suppressions file at %s\n",
          SanitizerToolName, filename);
  char *file_contents;
  uptr buffer_size;
  uptr contents_size;
  if (!ReadFileToBuffer(filename, &file_contents, &buffer_size,
                        &contents_size)) {
    Printf("%s: failed to read suppressions file '%s'\n", SanitizerToolName,
           filename);
    Die();
  }

  Parse(file_contents);
}

bool SuppressionContext::Match(const char *str, const char *type,
                               Suppression **s) {
  can_parse_ = false;
  if (!HasSuppressionType(type))
    return false;
  for (uptr i = 0; i < suppressions_.size(); i++) {
    Suppression &cur = suppressions_[i];
    if (0 == internal_strcmp(cur.type, type) && TemplateMatch(cur.templ, str)) {
      *s = &cur;
      return true;
    }
  }
  return false;
}

static const char *StripPrefix(const char *str, const char *prefix) {
  while (str && *str == *prefix) {
    str++;
    prefix++;
  }
  if (!*prefix)
    return str;
  return 0;
}

void SuppressionContext::Parse(const char *str) {
  // Context must not mutate once Match has been called.
  CHECK(can_parse_);
  const char *line = str;
  while (line) {
    while (line[0] == ' ' || line[0] == '\t')
      line++;
    const char *end = internal_strchr(line, '\n');
    if (end == 0)
      end = line + internal_strlen(line);
    if (line != end && line[0] != '#') {
      const char *end2 = end;
      while (line != end2 &&
             (end2[-1] == ' ' || end2[-1] == '\t' || end2[-1] == '\r'))
        end2--;
      int type;
      for (type = 0; type < suppression_types_num_; type++) {
        const char *next_char = StripPrefix(line, suppression_types_[type]);
        if (next_char && *next_char == ':') {
          line = ++next_char;
          break;
        }
      }
      if (type == suppression_types_num_) {
        Printf("%s: failed to parse suppressions\n", SanitizerToolName);
        Die();
      }
      Suppression s;
      s.type = suppression_types_[type];
      s.templ = (char*)InternalAlloc(end2 - line + 1);
      internal_memcpy(s.templ, line, end2 - line);
      s.templ[end2 - line] = 0;
      suppressions_.push_back(s);
      has_suppression_type_[type] = true;
    }
    if (end[0] == 0)
      break;
    line = end + 1;
  }
}

uptr SuppressionContext::SuppressionCount() const {
  return suppressions_.size();
}

bool SuppressionContext::HasSuppressionType(const char *type) const {
  for (int i = 0; i < suppression_types_num_; i++) {
    if (0 == internal_strcmp(type, suppression_types_[i]))
      return has_suppression_type_[i];
  }
  return false;
}

const Suppression *SuppressionContext::SuppressionAt(uptr i) const {
  CHECK_LT(i, suppressions_.size());
  return &suppressions_[i];
}

void SuppressionContext::GetMatched(
    InternalMmapVector<Suppression *> *matched) {
  for (uptr i = 0; i < suppressions_.size(); i++)
    if (atomic_load_relaxed(&suppressions_[i].hit_count))
      matched->push_back(&suppressions_[i]);
}

}  // namespace __sanitizer
//===-- sanitizer_thread_registry.cc --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between sanitizer tools.
//
// General thread bookkeeping functionality.
//===----------------------------------------------------------------------===//

#include "sanitizer_thread_registry.h"

namespace __sanitizer {

ThreadContextBase::ThreadContextBase(u32 tid)
    : tid(tid), unique_id(0), reuse_count(), os_id(0), user_id(0),
      status(ThreadStatusInvalid),
      detached(false), workerthread(false), parent_tid(0), next(0) {
  name[0] = '\0';
  atomic_store(&thread_destroyed, 0, memory_order_release);
}

ThreadContextBase::~ThreadContextBase() {
  // ThreadContextBase should never be deleted.
  CHECK(0);
}

void ThreadContextBase::SetName(const char *new_name) {
  name[0] = '\0';
  if (new_name) {
    internal_strncpy(name, new_name, sizeof(name));
    name[sizeof(name) - 1] = '\0';
  }
}

void ThreadContextBase::SetDead() {
  CHECK(status == ThreadStatusRunning ||
        status == ThreadStatusFinished);
  status = ThreadStatusDead;
  user_id = 0;
  OnDead();
}

void ThreadContextBase::SetDestroyed() {
  atomic_store(&thread_destroyed, 1, memory_order_release);
}

bool ThreadContextBase::GetDestroyed() {
  return !!atomic_load(&thread_destroyed, memory_order_acquire);
}

void ThreadContextBase::SetJoined(void *arg) {
  // FIXME(dvyukov): print message and continue (it's user error).
  CHECK_EQ(false, detached);
  CHECK_EQ(ThreadStatusFinished, status);
  status = ThreadStatusDead;
  user_id = 0;
  OnJoined(arg);
}

void ThreadContextBase::SetFinished() {
  // ThreadRegistry::FinishThread calls here in ThreadStatusCreated state
  // for a thread that never actually started.  In that case the thread
  // should go to ThreadStatusFinished regardless of whether it was created
  // as detached.
  if (!detached || status == ThreadStatusCreated) status = ThreadStatusFinished;
  OnFinished();
}

void ThreadContextBase::SetStarted(tid_t _os_id, bool _workerthread,
                                   void *arg) {
  status = ThreadStatusRunning;
  os_id = _os_id;
  workerthread = _workerthread;
  OnStarted(arg);
}

void ThreadContextBase::SetCreated(uptr _user_id, u64 _unique_id,
                                   bool _detached, u32 _parent_tid, void *arg) {
  status = ThreadStatusCreated;
  user_id = _user_id;
  unique_id = _unique_id;
  detached = _detached;
  // Parent tid makes no sense for the main thread.
  if (tid != 0)
    parent_tid = _parent_tid;
  OnCreated(arg);
}

void ThreadContextBase::Reset() {
  status = ThreadStatusInvalid;
  SetName(0);
  atomic_store(&thread_destroyed, 0, memory_order_release);
  OnReset();
}

// ThreadRegistry implementation.

const u32 ThreadRegistry::kUnknownTid = ~0U;

ThreadRegistry::ThreadRegistry(ThreadContextFactory factory, u32 max_threads,
                               u32 thread_quarantine_size, u32 max_reuse)
    : context_factory_(factory),
      max_threads_(max_threads),
      thread_quarantine_size_(thread_quarantine_size),
      max_reuse_(max_reuse),
      mtx_(),
      n_contexts_(0),
      total_threads_(0),
      alive_threads_(0),
      max_alive_threads_(0),
      running_threads_(0) {
  threads_ = (ThreadContextBase **)MmapOrDie(max_threads_ * sizeof(threads_[0]),
                                             "ThreadRegistry");
  dead_threads_.clear();
  invalid_threads_.clear();
}

void ThreadRegistry::GetNumberOfThreads(uptr *total, uptr *running,
                                        uptr *alive) {
  BlockingMutexLock l(&mtx_);
  if (total) *total = n_contexts_;
  if (running) *running = running_threads_;
  if (alive) *alive = alive_threads_;
}

uptr ThreadRegistry::GetMaxAliveThreads() {
  BlockingMutexLock l(&mtx_);
  return max_alive_threads_;
}

u32 ThreadRegistry::CreateThread(uptr user_id, bool detached, u32 parent_tid,
                                 void *arg) {
  BlockingMutexLock l(&mtx_);
  u32 tid = kUnknownTid;
  ThreadContextBase *tctx = QuarantinePop();
  if (tctx) {
    tid = tctx->tid;
  } else if (n_contexts_ < max_threads_) {
    // Allocate new thread context and tid.
    tid = n_contexts_++;
    tctx = context_factory_(tid);
    threads_[tid] = tctx;
  } else {
#if !SANITIZER_GO
    Report("%s: Thread limit (%u threads) exceeded. Dying.\n",
           SanitizerToolName, max_threads_);
#else
    Printf("race: limit on %u simultaneously alive goroutines is exceeded,"
        " dying\n", max_threads_);
#endif
    Die();
  }
  CHECK_NE(tctx, 0);
  CHECK_NE(tid, kUnknownTid);
  CHECK_LT(tid, max_threads_);
  CHECK_EQ(tctx->status, ThreadStatusInvalid);
  alive_threads_++;
  if (max_alive_threads_ < alive_threads_) {
    max_alive_threads_++;
    CHECK_EQ(alive_threads_, max_alive_threads_);
  }
  tctx->SetCreated(user_id, total_threads_++, detached,
                   parent_tid, arg);
  return tid;
}

void ThreadRegistry::RunCallbackForEachThreadLocked(ThreadCallback cb,
                                                    void *arg) {
  CheckLocked();
  for (u32 tid = 0; tid < n_contexts_; tid++) {
    ThreadContextBase *tctx = threads_[tid];
    if (tctx == 0)
      continue;
    cb(tctx, arg);
  }
}

u32 ThreadRegistry::FindThread(FindThreadCallback cb, void *arg) {
  BlockingMutexLock l(&mtx_);
  for (u32 tid = 0; tid < n_contexts_; tid++) {
    ThreadContextBase *tctx = threads_[tid];
    if (tctx != 0 && cb(tctx, arg))
      return tctx->tid;
  }
  return kUnknownTid;
}

ThreadContextBase *
ThreadRegistry::FindThreadContextLocked(FindThreadCallback cb, void *arg) {
  CheckLocked();
  for (u32 tid = 0; tid < n_contexts_; tid++) {
    ThreadContextBase *tctx = threads_[tid];
    if (tctx != 0 && cb(tctx, arg))
      return tctx;
  }
  return 0;
}

static bool FindThreadContextByOsIdCallback(ThreadContextBase *tctx,
                                            void *arg) {
  return (tctx->os_id == (uptr)arg && tctx->status != ThreadStatusInvalid &&
      tctx->status != ThreadStatusDead);
}

ThreadContextBase *ThreadRegistry::FindThreadContextByOsIDLocked(tid_t os_id) {
  return FindThreadContextLocked(FindThreadContextByOsIdCallback,
                                 (void *)os_id);
}

void ThreadRegistry::SetThreadName(u32 tid, const char *name) {
  BlockingMutexLock l(&mtx_);
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  CHECK_EQ(SANITIZER_FUCHSIA ? ThreadStatusCreated : ThreadStatusRunning,
           tctx->status);
  tctx->SetName(name);
}

void ThreadRegistry::SetThreadNameByUserId(uptr user_id, const char *name) {
  BlockingMutexLock l(&mtx_);
  for (u32 tid = 0; tid < n_contexts_; tid++) {
    ThreadContextBase *tctx = threads_[tid];
    if (tctx != 0 && tctx->user_id == user_id &&
        tctx->status != ThreadStatusInvalid) {
      tctx->SetName(name);
      return;
    }
  }
}

void ThreadRegistry::DetachThread(u32 tid, void *arg) {
  BlockingMutexLock l(&mtx_);
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  if (tctx->status == ThreadStatusInvalid) {
    Report("%s: Detach of non-existent thread\n", SanitizerToolName);
    return;
  }
  tctx->OnDetached(arg);
  if (tctx->status == ThreadStatusFinished) {
    tctx->SetDead();
    QuarantinePush(tctx);
  } else {
    tctx->detached = true;
  }
}

void ThreadRegistry::JoinThread(u32 tid, void *arg) {
  bool destroyed = false;
  do {
    {
      BlockingMutexLock l(&mtx_);
      CHECK_LT(tid, n_contexts_);
      ThreadContextBase *tctx = threads_[tid];
      CHECK_NE(tctx, 0);
      if (tctx->status == ThreadStatusInvalid) {
        Report("%s: Join of non-existent thread\n", SanitizerToolName);
        return;
      }
      if ((destroyed = tctx->GetDestroyed())) {
        tctx->SetJoined(arg);
        QuarantinePush(tctx);
      }
    }
    if (!destroyed)
      internal_sched_yield();
  } while (!destroyed);
}

// Normally this is called when the thread is about to exit.  If
// called in ThreadStatusCreated state, then this thread was never
// really started.  We just did CreateThread for a prospective new
// thread before trying to create it, and then failed to actually
// create it, and so never called StartThread.
void ThreadRegistry::FinishThread(u32 tid) {
  BlockingMutexLock l(&mtx_);
  CHECK_GT(alive_threads_, 0);
  alive_threads_--;
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  bool dead = tctx->detached;
  if (tctx->status == ThreadStatusRunning) {
    CHECK_GT(running_threads_, 0);
    running_threads_--;
  } else {
    // The thread never really existed.
    CHECK_EQ(tctx->status, ThreadStatusCreated);
    dead = true;
  }
  tctx->SetFinished();
  if (dead) {
    tctx->SetDead();
    QuarantinePush(tctx);
  }
  tctx->SetDestroyed();
}

void ThreadRegistry::StartThread(u32 tid, tid_t os_id, bool workerthread,
                                 void *arg) {
  BlockingMutexLock l(&mtx_);
  running_threads_++;
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  CHECK_EQ(ThreadStatusCreated, tctx->status);
  tctx->SetStarted(os_id, workerthread, arg);
}

void ThreadRegistry::QuarantinePush(ThreadContextBase *tctx) {
  if (tctx->tid == 0)
    return;  // Don't reuse the main thread.  It's a special snowflake.
  dead_threads_.push_back(tctx);
  if (dead_threads_.size() <= thread_quarantine_size_)
    return;
  tctx = dead_threads_.front();
  dead_threads_.pop_front();
  CHECK_EQ(tctx->status, ThreadStatusDead);
  tctx->Reset();
  tctx->reuse_count++;
  if (max_reuse_ > 0 && tctx->reuse_count >= max_reuse_)
    return;
  invalid_threads_.push_back(tctx);
}

ThreadContextBase *ThreadRegistry::QuarantinePop() {
  if (invalid_threads_.size() == 0)
    return 0;
  ThreadContextBase *tctx = invalid_threads_.front();
  invalid_threads_.pop_front();
  return tctx;
}

}  // namespace __sanitizer
//===-- sanitizer_stackdepot.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_stackdepot.h"

#include "sanitizer_common.h"
#include "sanitizer_stackdepotbase.h"

namespace __sanitizer {

struct StackDepotNode {
  StackDepotNode *link;
  u32 id;
  atomic_uint32_t hash_and_use_count; // hash_bits : 12; use_count : 20;
  u32 size;
  u32 tag;
  uptr stack[1];  // [size]

  static const u32 kTabSizeLog = 20;
  // Lower kTabSizeLog bits are equal for all items in one bucket.
  // We use these bits to store the per-stack use counter.
  static const u32 kUseCountBits = kTabSizeLog;
  static const u32 kMaxUseCount = 1 << kUseCountBits;
  static const u32 kUseCountMask = (1 << kUseCountBits) - 1;
  static const u32 kHashMask = ~kUseCountMask;

  typedef StackTrace args_type;
  bool eq(u32 hash, const args_type &args) const {
    u32 hash_bits =
        atomic_load(&hash_and_use_count, memory_order_relaxed) & kHashMask;
    if ((hash & kHashMask) != hash_bits || args.size != size || args.tag != tag)
      return false;
    uptr i = 0;
    for (; i < size; i++) {
      if (stack[i] != args.trace[i]) return false;
    }
    return true;
  }
  static uptr storage_size(const args_type &args) {
    return sizeof(StackDepotNode) + (args.size - 1) * sizeof(uptr);
  }
  static u32 hash(const args_type &args) {
    // murmur2
    const u32 m = 0x5bd1e995;
    const u32 seed = 0x9747b28c;
    const u32 r = 24;
    u32 h = seed ^ (args.size * sizeof(uptr));
    for (uptr i = 0; i < args.size; i++) {
      u32 k = args.trace[i];
      k *= m;
      k ^= k >> r;
      k *= m;
      h *= m;
      h ^= k;
    }
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
  }
  static bool is_valid(const args_type &args) {
    return args.size > 0 && args.trace;
  }
  void store(const args_type &args, u32 hash) {
    atomic_store(&hash_and_use_count, hash & kHashMask, memory_order_relaxed);
    size = args.size;
    tag = args.tag;
    internal_memcpy(stack, args.trace, size * sizeof(uptr));
  }
  args_type load() const {
    return args_type(&stack[0], size, tag);
  }
  StackDepotHandle get_handle() { return StackDepotHandle(this); }

  typedef StackDepotHandle handle_type;
};

COMPILER_CHECK(StackDepotNode::kMaxUseCount == (u32)kStackDepotMaxUseCount);

u32 StackDepotHandle::id() { return node_->id; }
int StackDepotHandle::use_count() {
  return atomic_load(&node_->hash_and_use_count, memory_order_relaxed) &
         StackDepotNode::kUseCountMask;
}
void StackDepotHandle::inc_use_count_unsafe() {
  u32 prev =
      atomic_fetch_add(&node_->hash_and_use_count, 1, memory_order_relaxed) &
      StackDepotNode::kUseCountMask;
  CHECK_LT(prev + 1, StackDepotNode::kMaxUseCount);
}

// FIXME(dvyukov): this single reserved bit is used in TSan.
typedef StackDepotBase<StackDepotNode, 1, StackDepotNode::kTabSizeLog>
    StackDepot;
static StackDepot theDepot;

StackDepotStats *StackDepotGetStats() {
  return theDepot.GetStats();
}

u32 StackDepotPut(StackTrace stack) {
  StackDepotHandle h = theDepot.Put(stack);
  return h.valid() ? h.id() : 0;
}

StackDepotHandle StackDepotPut_WithHandle(StackTrace stack) {
  return theDepot.Put(stack);
}

StackTrace StackDepotGet(u32 id) {
  return theDepot.Get(id);
}

void StackDepotLockAll() {
  theDepot.LockAll();
}

void StackDepotUnlockAll() {
  theDepot.UnlockAll();
}

bool StackDepotReverseMap::IdDescPair::IdComparator(
    const StackDepotReverseMap::IdDescPair &a,
    const StackDepotReverseMap::IdDescPair &b) {
  return a.id < b.id;
}

StackDepotReverseMap::StackDepotReverseMap()
    : map_(StackDepotGetStats()->n_uniq_ids + 100) {
  for (int idx = 0; idx < StackDepot::kTabSize; idx++) {
    atomic_uintptr_t *p = &theDepot.tab[idx];
    uptr v = atomic_load(p, memory_order_consume);
    StackDepotNode *s = (StackDepotNode*)(v & ~1);
    for (; s; s = s->link) {
      IdDescPair pair = {s->id, s};
      map_.push_back(pair);
    }
  }
  InternalSort(&map_, map_.size(), IdDescPair::IdComparator);
}

StackTrace StackDepotReverseMap::Get(u32 id) {
  if (!map_.size())
    return StackTrace();
  IdDescPair pair = {id, nullptr};
  uptr idx =
      InternalLowerBound(map_, 0, map_.size(), pair, IdDescPair::IdComparator);
  if (idx > map_.size() || map_[idx].id != id)
    return StackTrace();
  return map_[idx].desc->load();
}

} // namespace __sanitizer
//===-- sanitizer_stacktrace.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_stacktrace.h"

namespace __sanitizer {

uptr StackTrace::GetNextInstructionPc(uptr pc) {
#if defined(__mips__)
  return pc + 8;
#elif defined(__powerpc__)
  return pc + 4;
#else
  return pc + 1;
#endif
}

uptr StackTrace::GetCurrentPc() {
  return GET_CALLER_PC();
}

void BufferedStackTrace::Init(const uptr *pcs, uptr cnt, uptr extra_top_pc) {
  size = cnt + !!extra_top_pc;
  CHECK_LE(size, kStackTraceMax);
  internal_memcpy(trace_buffer, pcs, cnt * sizeof(trace_buffer[0]));
  if (extra_top_pc)
    trace_buffer[cnt] = extra_top_pc;
  top_frame_bp = 0;
}

// In GCC on ARM bp points to saved lr, not fp, so we should check the next
// cell in stack to be a saved frame pointer. GetCanonicFrame returns the
// pointer to saved frame pointer in any case.
static inline uhwptr *GetCanonicFrame(uptr bp,
                                      uptr stack_top,
                                      uptr stack_bottom) {
#ifdef __arm__
  if (!IsValidFrame(bp, stack_top, stack_bottom)) return 0;
  uhwptr *bp_prev = (uhwptr *)bp;
  if (IsValidFrame((uptr)bp_prev[0], stack_top, stack_bottom)) return bp_prev;
  // The next frame pointer does not look right. This could be a GCC frame, step
  // back by 1 word and try again.
  if (IsValidFrame((uptr)bp_prev[-1], stack_top, stack_bottom))
    return bp_prev - 1;
  // Nope, this does not look right either. This means the frame after next does
  // not have a valid frame pointer, but we can still extract the caller PC.
  // Unfortunately, there is no way to decide between GCC and LLVM frame
  // layouts. Assume LLVM.
  return bp_prev;
#else
  return (uhwptr*)bp;
#endif
}

void BufferedStackTrace::FastUnwindStack(uptr pc, uptr bp, uptr stack_top,
                                         uptr stack_bottom, u32 max_depth) {
  const uptr kPageSize = GetPageSizeCached();
  CHECK_GE(max_depth, 2);
  trace_buffer[0] = pc;
  size = 1;
  if (stack_top < 4096) return;  // Sanity check for stack top.
  uhwptr *frame = GetCanonicFrame(bp, stack_top, stack_bottom);
  // Lowest possible address that makes sense as the next frame pointer.
  // Goes up as we walk the stack.
  uptr bottom = stack_bottom;
  // Avoid infinite loop when frame == frame[0] by using frame > prev_frame.
  while (IsValidFrame((uptr)frame, stack_top, bottom) &&
         IsAligned((uptr)frame, sizeof(*frame)) &&
         size < max_depth) {
#ifdef __powerpc__
    // PowerPC ABIs specify that the return address is saved at offset
    // 16 of the *caller's* stack frame.  Thus we must dereference the
    // back chain to find the caller frame before extracting it.
    uhwptr *caller_frame = (uhwptr*)frame[0];
    if (!IsValidFrame((uptr)caller_frame, stack_top, bottom) ||
        !IsAligned((uptr)caller_frame, sizeof(uhwptr)))
      break;
    uhwptr pc1 = caller_frame[2];
#elif defined(__s390__)
    uhwptr pc1 = frame[14];
#else
    uhwptr pc1 = frame[1];
#endif
    // Let's assume that any pointer in the 0th page (i.e. <0x1000 on i386 and
    // x86_64) is invalid and stop unwinding here.  If we're adding support for
    // a platform where this isn't true, we need to reconsider this check.
    if (pc1 < kPageSize)
      break;
    if (pc1 != pc) {
      trace_buffer[size++] = (uptr) pc1;
    }
    bottom = (uptr)frame;
    frame = GetCanonicFrame((uptr)frame[0], stack_top, bottom);
  }
}

void BufferedStackTrace::PopStackFrames(uptr count) {
  CHECK_LT(count, size);
  size -= count;
  for (uptr i = 0; i < size; ++i) {
    trace_buffer[i] = trace_buffer[i + count];
  }
}

static uptr Distance(uptr a, uptr b) { return a < b ? b - a : a - b; }

uptr BufferedStackTrace::LocatePcInTrace(uptr pc) {
  uptr best = 0;
  for (uptr i = 1; i < size; ++i) {
    if (Distance(trace[i], pc) < Distance(trace[best], pc)) best = i;
  }
  return best;
}

}  // namespace __sanitizer
//===-- sanitizer_symbolizer.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator_internal.h"
#include "sanitizer_platform.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_symbolizer_internal.h"

namespace __sanitizer {

AddressInfo::AddressInfo() {
  internal_memset(this, 0, sizeof(AddressInfo));
  function_offset = kUnknown;
}

void AddressInfo::Clear() {
  InternalFree(module);
  InternalFree(function);
  InternalFree(file);
  internal_memset(this, 0, sizeof(AddressInfo));
  function_offset = kUnknown;
}

void AddressInfo::FillModuleInfo(const char *mod_name, uptr mod_offset,
                                 ModuleArch mod_arch) {
  module = internal_strdup(mod_name);
  module_offset = mod_offset;
  module_arch = mod_arch;
}

SymbolizedStack::SymbolizedStack() : next(nullptr), info() {}

SymbolizedStack *SymbolizedStack::New(uptr addr) {
  void *mem = InternalAlloc(sizeof(SymbolizedStack));
  SymbolizedStack *res = new(mem) SymbolizedStack();
  res->info.address = addr;
  return res;
}

void SymbolizedStack::ClearAll() {
  info.Clear();
  if (next)
    next->ClearAll();
  InternalFree(this);
}

DataInfo::DataInfo() {
  internal_memset(this, 0, sizeof(DataInfo));
}

void DataInfo::Clear() {
  InternalFree(module);
  InternalFree(file);
  InternalFree(name);
  internal_memset(this, 0, sizeof(DataInfo));
}

Symbolizer *Symbolizer::symbolizer_;
StaticSpinMutex Symbolizer::init_mu_;
LowLevelAllocator Symbolizer::symbolizer_allocator_;

void Symbolizer::InvalidateModuleList() {
  modules_fresh_ = false;
}

void Symbolizer::AddHooks(Symbolizer::StartSymbolizationHook start_hook,
                          Symbolizer::EndSymbolizationHook end_hook) {
  CHECK(start_hook_ == 0 && end_hook_ == 0);
  start_hook_ = start_hook;
  end_hook_ = end_hook;
}

const char *Symbolizer::ModuleNameOwner::GetOwnedCopy(const char *str) {
  mu_->CheckLocked();

  // 'str' will be the same string multiple times in a row, optimize this case.
  if (last_match_ && !internal_strcmp(last_match_, str))
    return last_match_;

  // FIXME: this is linear search.
  // We should optimize this further if this turns out to be a bottleneck later.
  for (uptr i = 0; i < storage_.size(); ++i) {
    if (!internal_strcmp(storage_[i], str)) {
      last_match_ = storage_[i];
      return last_match_;
    }
  }
  last_match_ = internal_strdup(str);
  storage_.push_back(last_match_);
  return last_match_;
}

Symbolizer::Symbolizer(IntrusiveList<SymbolizerTool> tools)
    : module_names_(&mu_), modules_(), modules_fresh_(false), tools_(tools),
      start_hook_(0), end_hook_(0) {}

Symbolizer::SymbolizerScope::SymbolizerScope(const Symbolizer *sym)
    : sym_(sym) {
  if (sym_->start_hook_)
    sym_->start_hook_();
}

Symbolizer::SymbolizerScope::~SymbolizerScope() {
  if (sym_->end_hook_)
    sym_->end_hook_();
}

}  // namespace __sanitizer
//===-- sanitizer_symbolizer_report.cc ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file is shared between AddressSanitizer and other sanitizer run-time
/// libraries and implements symbolized reports related functions.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_file.h"
#include "sanitizer_flags.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_report_decorator.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_stacktrace_printer.h"
#include "sanitizer_symbolizer.h"

#if SANITIZER_POSIX
# include "sanitizer_posix.h"
# include <sys/mman.h>
#endif

namespace __sanitizer {

#if !SANITIZER_GO
void ReportErrorSummary(const char *error_type, const AddressInfo &info,
                        const char *alt_tool_name) {
  if (!common_flags()->print_summary) return;
  InternalScopedString buff(kMaxSummaryLength);
  buff.append("%s ", error_type);
  RenderFrame(&buff, "%L %F", 0, info, common_flags()->symbolize_vs_style,
              common_flags()->strip_path_prefix);
  ReportErrorSummary(buff.data(), alt_tool_name);
}
#endif

#if !SANITIZER_FUCHSIA

bool ReportFile::SupportsColors() {
  SpinMutexLock l(mu);
  ReopenIfNecessary();
  return SupportsColoredOutput(fd);
}

static INLINE bool ReportSupportsColors() {
  return report_file.SupportsColors();
}

#else  // SANITIZER_FUCHSIA

// Fuchsia's logs always go through post-processing that handles colorization.
static INLINE bool ReportSupportsColors() { return true; }

#endif  // !SANITIZER_FUCHSIA

bool ColorizeReports() {
  // FIXME: Add proper Windows support to AnsiColorDecorator and re-enable color
  // printing on Windows.
  if (SANITIZER_WINDOWS)
    return false;

  const char *flag = common_flags()->color;
  return internal_strcmp(flag, "always") == 0 ||
         (internal_strcmp(flag, "auto") == 0 && ReportSupportsColors());
}

void ReportErrorSummary(const char *error_type, const StackTrace *stack,
                        const char *alt_tool_name) {
#if !SANITIZER_GO
  if (!common_flags()->print_summary)
    return;
  if (stack->size == 0) {
    ReportErrorSummary(error_type);
    return;
  }
  // Currently, we include the first stack frame into the report summary.
  // Maybe sometimes we need to choose another frame (e.g. skip memcpy/etc).
  uptr pc = StackTrace::GetPreviousInstructionPc(stack->trace[0]);
  SymbolizedStack *frame = Symbolizer::GetOrInit()->SymbolizePC(pc);
  ReportErrorSummary(error_type, frame->info, alt_tool_name);
  frame->ClearAll();
#endif
}

void ReportMmapWriteExec(int prot) {
#if SANITIZER_POSIX && (!SANITIZER_GO && !SANITIZER_ANDROID)
  if ((prot & (PROT_WRITE | PROT_EXEC)) != (PROT_WRITE | PROT_EXEC))
    return;

  ScopedErrorReportLock l;
  SanitizerCommonDecorator d;

  InternalScopedBuffer<BufferedStackTrace> stack_buffer(1);
  BufferedStackTrace *stack = stack_buffer.data();
  stack->Reset();
  uptr top = 0;
  uptr bottom = 0;
  GET_CALLER_PC_BP_SP;
  (void)sp;
  bool fast = common_flags()->fast_unwind_on_fatal;
  if (fast)
    GetThreadStackTopAndBottom(false, &top, &bottom);
  stack->Unwind(kStackTraceMax, pc, bp, nullptr, top, bottom, fast);

  Printf("%s", d.Warning());
  Report("WARNING: %s: writable-executable page usage\n", SanitizerToolName);
  Printf("%s", d.Default());

  stack->Print();
  ReportErrorSummary("w-and-x-usage", stack);
#endif
}

#if !SANITIZER_FUCHSIA && !SANITIZER_GO
void StartReportDeadlySignal() {
  // Write the first message using fd=2, just in case.
  // It may actually fail to write in case stderr is closed.
  CatastrophicErrorWrite(SanitizerToolName, internal_strlen(SanitizerToolName));
  static const char kDeadlySignal[] = ":DEADLYSIGNAL\n";
  CatastrophicErrorWrite(kDeadlySignal, sizeof(kDeadlySignal) - 1);
}

static void MaybeReportNonExecRegion(uptr pc) {
#if SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD
  MemoryMappingLayout proc_maps(/*cache_enabled*/ true);
  MemoryMappedSegment segment;
  while (proc_maps.Next(&segment)) {
    if (pc >= segment.start && pc < segment.end && !segment.IsExecutable())
      Report("Hint: PC is at a non-executable region. Maybe a wild jump?\n");
  }
#endif
}

static void PrintMemoryByte(InternalScopedString *str, const char *before,
                            u8 byte) {
  SanitizerCommonDecorator d;
  str->append("%s%s%x%x%s ", before, d.MemoryByte(), byte >> 4, byte & 15,
              d.Default());
}

static void MaybeDumpInstructionBytes(uptr pc) {
  if (!common_flags()->dump_instruction_bytes || (pc < GetPageSizeCached()))
    return;
  InternalScopedString str(1024);
  str.append("First 16 instruction bytes at pc: ");
  if (IsAccessibleMemoryRange(pc, 16)) {
    for (int i = 0; i < 16; ++i) {
      PrintMemoryByte(&str, "", ((u8 *)pc)[i]);
    }
    str.append("\n");
  } else {
    str.append("unaccessible\n");
  }
  Report("%s", str.data());
}

static void MaybeDumpRegisters(void *context) {
  if (!common_flags()->dump_registers) return;
  SignalContext::DumpAllRegisters(context);
}

static void ReportStackOverflowImpl(const SignalContext &sig, u32 tid,
                                    UnwindSignalStackCallbackType unwind,
                                    const void *unwind_context) {
  SanitizerCommonDecorator d;
  Printf("%s", d.Warning());
  static const char kDescription[] = "stack-overflow";
  Report("ERROR: %s: %s on address %p (pc %p bp %p sp %p T%d)\n",
         SanitizerToolName, kDescription, (void *)sig.addr, (void *)sig.pc,
         (void *)sig.bp, (void *)sig.sp, tid);
  Printf("%s", d.Default());
  InternalScopedBuffer<BufferedStackTrace> stack_buffer(1);
  BufferedStackTrace *stack = stack_buffer.data();
  stack->Reset();
  unwind(sig, unwind_context, stack);
  stack->Print();
  ReportErrorSummary(kDescription, stack);
}

static void ReportDeadlySignalImpl(const SignalContext &sig, u32 tid,
                                   UnwindSignalStackCallbackType unwind,
                                   const void *unwind_context) {
  SanitizerCommonDecorator d;
  Printf("%s", d.Warning());
  const char *description = sig.Describe();
  Report("ERROR: %s: %s on unknown address %p (pc %p bp %p sp %p T%d)\n",
         SanitizerToolName, description, (void *)sig.addr, (void *)sig.pc,
         (void *)sig.bp, (void *)sig.sp, tid);
  Printf("%s", d.Default());
  if (sig.pc < GetPageSizeCached())
    Report("Hint: pc points to the zero page.\n");
  if (sig.is_memory_access) {
    const char *access_type =
        sig.write_flag == SignalContext::WRITE
            ? "WRITE"
            : (sig.write_flag == SignalContext::READ ? "READ" : "UNKNOWN");
    Report("The signal is caused by a %s memory access.\n", access_type);
    if (sig.addr < GetPageSizeCached())
      Report("Hint: address points to the zero page.\n");
  }
  MaybeReportNonExecRegion(sig.pc);
  InternalScopedBuffer<BufferedStackTrace> stack_buffer(1);
  BufferedStackTrace *stack = stack_buffer.data();
  stack->Reset();
  unwind(sig, unwind_context, stack);
  stack->Print();
  MaybeDumpInstructionBytes(sig.pc);
  MaybeDumpRegisters(sig.context);
  Printf("%s can not provide additional info.\n", SanitizerToolName);
  ReportErrorSummary(description, stack);
}

void ReportDeadlySignal(const SignalContext &sig, u32 tid,
                        UnwindSignalStackCallbackType unwind,
                        const void *unwind_context) {
  if (sig.IsStackOverflow())
    ReportStackOverflowImpl(sig, tid, unwind, unwind_context);
  else
    ReportDeadlySignalImpl(sig, tid, unwind, unwind_context);
}

void HandleDeadlySignal(void *siginfo, void *context, u32 tid,
                        UnwindSignalStackCallbackType unwind,
                        const void *unwind_context) {
  StartReportDeadlySignal();
  ScopedErrorReportLock rl;
  SignalContext sig(siginfo, context);
  ReportDeadlySignal(sig, tid, unwind, unwind_context);
  Report("ABORTING\n");
  Die();
}

#endif  // !SANITIZER_FUCHSIA && !SANITIZER_GO

static atomic_uintptr_t reporting_thread = {0};
static StaticSpinMutex CommonSanitizerReportMutex;

ScopedErrorReportLock::ScopedErrorReportLock() {
  uptr current = GetThreadSelf();
  for (;;) {
    uptr expected = 0;
    if (atomic_compare_exchange_strong(&reporting_thread, &expected, current,
                                       memory_order_relaxed)) {
      // We've claimed reporting_thread so proceed.
      CommonSanitizerReportMutex.Lock();
      return;
    }

    if (expected == current) {
      // This is either asynch signal or nested error during error reporting.
      // Fail simple to avoid deadlocks in Report().

      // Can't use Report() here because of potential deadlocks in nested
      // signal handlers.
      CatastrophicErrorWrite(SanitizerToolName,
                             internal_strlen(SanitizerToolName));
      static const char msg[] = ": nested bug in the same thread, aborting.\n";
      CatastrophicErrorWrite(msg, sizeof(msg) - 1);

      internal__exit(common_flags()->exitcode);
    }

    internal_sched_yield();
  }
}

ScopedErrorReportLock::~ScopedErrorReportLock() {
  CommonSanitizerReportMutex.Unlock();
  atomic_store_relaxed(&reporting_thread, 0);
}

void ScopedErrorReportLock::CheckLocked() {
  CommonSanitizerReportMutex.CheckLocked();
}

}  // namespace __sanitizer
//===-- sanitizer_termination.cc --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file contains the Sanitizer termination functions CheckFailed and Die,
/// and the callback functionalities associated with them.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

static const int kMaxNumOfInternalDieCallbacks = 5;
static DieCallbackType InternalDieCallbacks[kMaxNumOfInternalDieCallbacks];

bool AddDieCallback(DieCallbackType callback) {
  for (int i = 0; i < kMaxNumOfInternalDieCallbacks; i++) {
    if (InternalDieCallbacks[i] == nullptr) {
      InternalDieCallbacks[i] = callback;
      return true;
    }
  }
  return false;
}

bool RemoveDieCallback(DieCallbackType callback) {
  for (int i = 0; i < kMaxNumOfInternalDieCallbacks; i++) {
    if (InternalDieCallbacks[i] == callback) {
      internal_memmove(&InternalDieCallbacks[i], &InternalDieCallbacks[i + 1],
                       sizeof(InternalDieCallbacks[0]) *
                           (kMaxNumOfInternalDieCallbacks - i - 1));
      InternalDieCallbacks[kMaxNumOfInternalDieCallbacks - 1] = nullptr;
      return true;
    }
  }
  return false;
}

static DieCallbackType UserDieCallback;
void SetUserDieCallback(DieCallbackType callback) {
  UserDieCallback = callback;
}

void NORETURN Die() {
  if (UserDieCallback)
    UserDieCallback();
  for (int i = kMaxNumOfInternalDieCallbacks - 1; i >= 0; i--) {
    if (InternalDieCallbacks[i])
      InternalDieCallbacks[i]();
  }
  if (common_flags()->abort_on_error)
    Abort();
  internal__exit(common_flags()->exitcode);
}

static CheckFailedCallbackType CheckFailedCallback;
void SetCheckFailedCallback(CheckFailedCallbackType callback) {
  CheckFailedCallback = callback;
}

const int kSecondsToSleepWhenRecursiveCheckFailed = 2;

void NORETURN CheckFailed(const char *file, int line, const char *cond,
                          u64 v1, u64 v2) {
  static atomic_uint32_t num_calls;
  if (atomic_fetch_add(&num_calls, 1, memory_order_relaxed) > 10) {
    SleepForSeconds(kSecondsToSleepWhenRecursiveCheckFailed);
    Trap();
  }

  if (CheckFailedCallback) {
    CheckFailedCallback(file, line, cond, v1, v2);
  }
  Report("Sanitizer CHECK failed: %s:%d %s (%lld, %lld)\n", file, line, cond,
                                                            v1, v2);
  Die();
}

} // namespace __sanitizer
//===-- tsan_platform_linux.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Linux- and FreeBSD-specific code.
//===----------------------------------------------------------------------===//


#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX || SANITIZER_FREEBSD || SANITIZER_NETBSD

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_platform_limits_netbsd.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stoptheworld.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_flags.h"

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/mman.h>
#if SANITIZER_LINUX
#include <sys/personality.h>
#include <setjmp.h>
#endif
#include <sys/syscall.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sched.h>
#include <dlfcn.h>
#if SANITIZER_LINUX
#define __need_res_state
#include <resolv.h>
#endif

#ifdef sa_handler
# undef sa_handler
#endif

#ifdef sa_sigaction
# undef sa_sigaction
#endif

#if SANITIZER_FREEBSD
extern "C" void *__libc_stack_end;
void *__libc_stack_end = 0;
#endif

#if SANITIZER_LINUX && defined(__aarch64__)
void InitializeGuardPtr() __attribute__((visibility("hidden")));
#endif

namespace __tsan {

#ifdef TSAN_RUNTIME_VMA
// Runtime detected VMA size.
uptr vmaSize;
#endif

enum {
  MemTotal  = 0,
  MemShadow = 1,
  MemMeta   = 2,
  MemFile   = 3,
  MemMmap   = 4,
  MemTrace  = 5,
  MemHeap   = 6,
  MemOther  = 7,
  MemCount  = 8,
};

void FillProfileCallback(uptr p, uptr rss, bool file,
                         uptr *mem, uptr stats_size) {
  mem[MemTotal] += rss;
  if (p >= ShadowBeg() && p < ShadowEnd())
    mem[MemShadow] += rss;
  else if (p >= MetaShadowBeg() && p < MetaShadowEnd())
    mem[MemMeta] += rss;
#if !SANITIZER_GO
  else if (p >= HeapMemBeg() && p < HeapMemEnd())
    mem[MemHeap] += rss;
  else if (p >= LoAppMemBeg() && p < LoAppMemEnd())
    mem[file ? MemFile : MemMmap] += rss;
  else if (p >= HiAppMemBeg() && p < HiAppMemEnd())
    mem[file ? MemFile : MemMmap] += rss;
#else
  else if (p >= AppMemBeg() && p < AppMemEnd())
    mem[file ? MemFile : MemMmap] += rss;
#endif
  else if (p >= TraceMemBeg() && p < TraceMemEnd())
    mem[MemTrace] += rss;
  else
    mem[MemOther] += rss;
}

void WriteMemoryProfile(char *buf, uptr buf_size, uptr nthread, uptr nlive) {
  uptr mem[MemCount];
  internal_memset(mem, 0, sizeof(mem[0]) * MemCount);
  __sanitizer::GetMemoryProfile(FillProfileCallback, mem, 7);
  StackDepotStats *stacks = StackDepotGetStats();
  internal_snprintf(buf, buf_size,
      "RSS %zd MB: shadow:%zd meta:%zd file:%zd mmap:%zd"
      " trace:%zd heap:%zd other:%zd stacks=%zd[%zd] nthr=%zd/%zd\n",
      mem[MemTotal] >> 20, mem[MemShadow] >> 20, mem[MemMeta] >> 20,
      mem[MemFile] >> 20, mem[MemMmap] >> 20, mem[MemTrace] >> 20,
      mem[MemHeap] >> 20, mem[MemOther] >> 20,
      stacks->allocated >> 20, stacks->n_uniq_ids,
      nlive, nthread);
}

#if SANITIZER_LINUX
void FlushShadowMemoryCallback(
    const SuspendedThreadsList &suspended_threads_list,
    void *argument) {
  ReleaseMemoryPagesToOS(ShadowBeg(), ShadowEnd());
}
#endif

void FlushShadowMemory() {
#if SANITIZER_LINUX
  StopTheWorld(FlushShadowMemoryCallback, 0);
#endif
}

#if !SANITIZER_GO
// Mark shadow for .rodata sections with the special kShadowRodata marker.
// Accesses to .rodata can't race, so this saves time, memory and trace space.
static void MapRodata() {
  // First create temp file.
  const char *tmpdir = GetEnv("TMPDIR");
  if (tmpdir == 0)
    tmpdir = GetEnv("TEST_TMPDIR");
#ifdef P_tmpdir
  if (tmpdir == 0)
    tmpdir = P_tmpdir;
#endif
  if (tmpdir == 0)
    return;
  char name[256];
  internal_snprintf(name, sizeof(name), "%s/tsan.rodata.%d",
                    tmpdir, (int)internal_getpid());
  uptr openrv = internal_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (internal_iserror(openrv))
    return;
  internal_unlink(name);  // Unlink it now, so that we can reuse the buffer.
  fd_t fd = openrv;
  // Fill the file with kShadowRodata.
  const uptr kMarkerSize = 512 * 1024 / sizeof(u64);
  InternalScopedBuffer<u64> marker(kMarkerSize);
  // volatile to prevent insertion of memset
  for (volatile u64 *p = marker.data(); p < marker.data() + kMarkerSize; p++)
    *p = kShadowRodata;
  internal_write(fd, marker.data(), marker.size());
  // Map the file into memory.
  uptr page = internal_mmap(0, GetPageSizeCached(), PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, fd, 0);
  if (internal_iserror(page)) {
    internal_close(fd);
    return;
  }
  // Map the file into shadow of .rodata sections.
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  // Reusing the buffer 'name'.
  MemoryMappedSegment segment(name, ARRAY_SIZE(name));
  while (proc_maps.Next(&segment)) {
    if (segment.filename[0] != 0 && segment.filename[0] != '[' &&
        segment.IsReadable() && segment.IsExecutable() &&
        !segment.IsWritable() && IsAppMem(segment.start)) {
      // Assume it's .rodata
      char *shadow_start = (char *)MemToShadow(segment.start);
      char *shadow_end = (char *)MemToShadow(segment.end);
      for (char *p = shadow_start; p < shadow_end; p += marker.size()) {
        internal_mmap(p, Min<uptr>(marker.size(), shadow_end - p),
                      PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
      }
    }
  }
  internal_close(fd);
}

void InitializeShadowMemoryPlatform() {
  MapRodata();
}

#endif  // #if !SANITIZER_GO

void InitializePlatformEarly() {
#ifdef TSAN_RUNTIME_VMA
  vmaSize =
    (MostSignificantSetBitIndex(GET_CURRENT_FRAME()) + 1);
#if defined(__aarch64__)
  if (vmaSize != 39 && vmaSize != 42 && vmaSize != 48) {
    Printf("FATAL: ThreadSanitizer: unsupported VMA range\n");
    Printf("FATAL: Found %zd - Supported 39, 42 and 48\n", vmaSize);
    Die();
  }
#elif defined(__powerpc64__)
# if !SANITIZER_GO
  if (vmaSize != 44 && vmaSize != 46 && vmaSize != 47) {
    Printf("FATAL: ThreadSanitizer: unsupported VMA range\n");
    Printf("FATAL: Found %zd - Supported 44, 46, and 47\n", vmaSize);
    Die();
  }
# else
  if (vmaSize != 46 && vmaSize != 47) {
    Printf("FATAL: ThreadSanitizer: unsupported VMA range\n");
    Printf("FATAL: Found %zd - Supported 46, and 47\n", vmaSize);
    Die();
  }
# endif
#endif
#endif
}

void InitializePlatform() {
  DisableCoreDumperIfNecessary();

  // Go maps shadow memory lazily and works fine with limited address space.
  // Unlimited stack is not a problem as well, because the executable
  // is not compiled with -pie.
  if (!SANITIZER_GO) {
    bool reexec = false;
    // TSan doesn't play well with unlimited stack size (as stack
    // overlaps with shadow memory). If we detect unlimited stack size,
    // we re-exec the program with limited stack size as a best effort.
    if (StackSizeIsUnlimited()) {
      const uptr kMaxStackSize = 32 * 1024 * 1024;
      VReport(1, "Program is run with unlimited stack size, which wouldn't "
                 "work with ThreadSanitizer.\n"
                 "Re-execing with stack size limited to %zd bytes.\n",
              kMaxStackSize);
      SetStackSizeLimitInBytes(kMaxStackSize);
      reexec = true;
    }

    if (!AddressSpaceIsUnlimited()) {
      Report("WARNING: Program is run with limited virtual address space,"
             " which wouldn't work with ThreadSanitizer.\n");
      Report("Re-execing with unlimited virtual address space.\n");
      SetAddressSpaceUnlimited();
      reexec = true;
    }
#if SANITIZER_LINUX && defined(__aarch64__)
    // After patch "arm64: mm: support ARCH_MMAP_RND_BITS." is introduced in
    // linux kernel, the random gap between stack and mapped area is increased
    // from 128M to 36G on 39-bit aarch64. As it is almost impossible to cover
    // this big range, we should disable randomized virtual space on aarch64.
    int old_personality = personality(0xffffffff);
    if (old_personality != -1 && (old_personality & ADDR_NO_RANDOMIZE) == 0) {
      VReport(1, "WARNING: Program is run with randomized virtual address "
              "space, which wouldn't work with ThreadSanitizer.\n"
              "Re-execing with fixed virtual address space.\n");
      CHECK_NE(personality(old_personality | ADDR_NO_RANDOMIZE), -1);
      reexec = true;
    }
    // Initialize the guard pointer used in {sig}{set,long}jump.
    InitializeGuardPtr();
#endif
    if (reexec)
      ReExec();
  }

#if !SANITIZER_GO
  CheckAndProtect();
  InitTlsSize();
#endif
}

#if !SANITIZER_GO
// Extract file descriptors passed to glibc internal __res_iclose function.
// This is required to properly "close" the fds, because we do not see internal
// closes within glibc. The code is a pure hack.
int ExtractResolvFDs(void *state, int *fds, int nfd) {
#if SANITIZER_LINUX && !SANITIZER_ANDROID
  int cnt = 0;
  struct __res_state *statp = (struct __res_state*)state;
  for (int i = 0; i < MAXNS && cnt < nfd; i++) {
    if (statp->_u._ext.nsaddrs[i] && statp->_u._ext.nssocks[i] != -1)
      fds[cnt++] = statp->_u._ext.nssocks[i];
  }
  return cnt;
#else
  return 0;
#endif
}

// Extract file descriptors passed via UNIX domain sockets.
// This is requried to properly handle "open" of these fds.
// see 'man recvmsg' and 'man 3 cmsg'.
int ExtractRecvmsgFDs(void *msgp, int *fds, int nfd) {
  int res = 0;
  msghdr *msg = (msghdr*)msgp;
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(msg);
  for (; cmsg; cmsg = CMSG_NXTHDR(msg, cmsg)) {
    if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS)
      continue;
    int n = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(fds[0]);
    for (int i = 0; i < n; i++) {
      fds[res++] = ((int*)CMSG_DATA(cmsg))[i];
      if (res == nfd)
        return res;
    }
  }
  return res;
}

void ImitateTlsWrite(ThreadState *thr, uptr tls_addr, uptr tls_size) {
  // Check that the thr object is in tls;
  const uptr thr_beg = (uptr)thr;
  const uptr thr_end = (uptr)thr + sizeof(*thr);
  CHECK_GE(thr_beg, tls_addr);
  CHECK_LE(thr_beg, tls_addr + tls_size);
  CHECK_GE(thr_end, tls_addr);
  CHECK_LE(thr_end, tls_addr + tls_size);
  // Since the thr object is huge, skip it.
  MemoryRangeImitateWrite(thr, /*pc=*/2, tls_addr, thr_beg - tls_addr);
  MemoryRangeImitateWrite(thr, /*pc=*/2, thr_end,
                          tls_addr + tls_size - thr_end);
}

// Note: this function runs with async signals enabled,
// so it must not touch any tsan state.
int call_pthread_cancel_with_cleanup(int(*fn)(void *c, void *m,
    void *abstime), void *c, void *m, void *abstime,
    void(*cleanup)(void *arg), void *arg) {
  // pthread_cleanup_push/pop are hardcore macros mess.
  // We can't intercept nor call them w/o including pthread.h.
  int res;
  pthread_cleanup_push(cleanup, arg);
  res = fn(c, m, abstime);
  pthread_cleanup_pop(0);
  return res;
}
#endif

#if !SANITIZER_GO
void ReplaceSystemMalloc() { }
#endif

#if !SANITIZER_GO
#if SANITIZER_ANDROID
// On Android, one thread can call intercepted functions after
// DestroyThreadState(), so add a fake thread state for "dead" threads.
static ThreadState *dead_thread_state = nullptr;

ThreadState *cur_thread() {
  ThreadState* thr = reinterpret_cast<ThreadState*>(*get_android_tls_ptr());
  if (thr == nullptr) {
    __sanitizer_sigset_t emptyset;
    internal_sigfillset(&emptyset);
    __sanitizer_sigset_t oldset;
    CHECK_EQ(0, internal_sigprocmask(SIG_SETMASK, &emptyset, &oldset));
    thr = reinterpret_cast<ThreadState*>(*get_android_tls_ptr());
    if (thr == nullptr) {
      thr = reinterpret_cast<ThreadState*>(MmapOrDie(sizeof(ThreadState),
                                                     "ThreadState"));
      *get_android_tls_ptr() = reinterpret_cast<uptr>(thr);
      if (dead_thread_state == nullptr) {
        dead_thread_state = reinterpret_cast<ThreadState*>(
            MmapOrDie(sizeof(ThreadState), "ThreadState"));
        dead_thread_state->fast_state.SetIgnoreBit();
        dead_thread_state->ignore_interceptors = 1;
        dead_thread_state->is_dead = true;
        *const_cast<int*>(&dead_thread_state->tid) = -1;
        CHECK_EQ(0, internal_mprotect(dead_thread_state, sizeof(ThreadState),
                                      PROT_READ));
      }
    }
    CHECK_EQ(0, internal_sigprocmask(SIG_SETMASK, &oldset, nullptr));
  }
  return thr;
}

void cur_thread_finalize() {
  __sanitizer_sigset_t emptyset;
  internal_sigfillset(&emptyset);
  __sanitizer_sigset_t oldset;
  CHECK_EQ(0, internal_sigprocmask(SIG_SETMASK, &emptyset, &oldset));
  ThreadState* thr = reinterpret_cast<ThreadState*>(*get_android_tls_ptr());
  if (thr != dead_thread_state) {
    *get_android_tls_ptr() = reinterpret_cast<uptr>(dead_thread_state);
    UnmapOrDie(thr, sizeof(ThreadState));
  }
  CHECK_EQ(0, internal_sigprocmask(SIG_SETMASK, &oldset, nullptr));
}
#endif  // SANITIZER_ANDROID
#endif  // if !SANITIZER_GO

}  // namespace __tsan

#endif  // SANITIZER_LINUX || SANITIZER_FREEBSD || SANITIZER_NETBSD
//===-- sanitizer_posix.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements POSIX-specific functions from
// sanitizer_posix.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_POSIX

#include "sanitizer_common.h"
#include "sanitizer_file.h"
#include "sanitizer_libc.h"
#include "sanitizer_posix.h"
#include "sanitizer_procmaps.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>

#if SANITIZER_FREEBSD
// The MAP_NORESERVE define has been removed in FreeBSD 11.x, and even before
// that, it was never implemented.  So just define it to zero.
#undef  MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

namespace __sanitizer {

// ------------- sanitizer_common.h
uptr GetMmapGranularity() {
  return GetPageSize();
}

void *MmapOrDie(uptr size, const char *mem_type, bool raw_report) {
  size = RoundUpTo(size, GetPageSizeCached());
  uptr res = internal_mmap(nullptr, size,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANON, -1, 0);
  int reserrno;
  if (UNLIKELY(internal_iserror(res, &reserrno)))
    ReportMmapFailureAndDie(size, mem_type, "allocate", reserrno, raw_report);
  IncreaseTotalMmap(size);
  return (void *)res;
}

void UnmapOrDie(void *addr, uptr size) {
  if (!addr || !size) return;
  uptr res = internal_munmap(addr, size);
  if (UNLIKELY(internal_iserror(res))) {
    Report("ERROR: %s failed to deallocate 0x%zx (%zd) bytes at address %p\n",
           SanitizerToolName, size, size, addr);
    CHECK("unable to unmap" && 0);
  }
  DecreaseTotalMmap(size);
}

void *MmapOrDieOnFatalError(uptr size, const char *mem_type) {
  size = RoundUpTo(size, GetPageSizeCached());
  uptr res = internal_mmap(nullptr, size,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANON, -1, 0);
  int reserrno;
  if (UNLIKELY(internal_iserror(res, &reserrno))) {
    if (reserrno == ENOMEM)
      return nullptr;
    ReportMmapFailureAndDie(size, mem_type, "allocate", reserrno);
  }
  IncreaseTotalMmap(size);
  return (void *)res;
}

// We want to map a chunk of address space aligned to 'alignment'.
// We do it by mapping a bit more and then unmapping redundant pieces.
// We probably can do it with fewer syscalls in some OS-dependent way.
void *MmapAlignedOrDieOnFatalError(uptr size, uptr alignment,
                                   const char *mem_type) {
  CHECK(IsPowerOfTwo(size));
  CHECK(IsPowerOfTwo(alignment));
  uptr map_size = size + alignment;
  uptr map_res = (uptr)MmapOrDieOnFatalError(map_size, mem_type);
  if (UNLIKELY(!map_res))
    return nullptr;
  uptr map_end = map_res + map_size;
  uptr res = map_res;
  if (!IsAligned(res, alignment)) {
    res = (map_res + alignment - 1) & ~(alignment - 1);
    UnmapOrDie((void*)map_res, res - map_res);
  }
  uptr end = res + size;
  if (end != map_end)
    UnmapOrDie((void*)end, map_end - end);
  return (void*)res;
}

void *MmapNoReserveOrDie(uptr size, const char *mem_type) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap(nullptr,
                         RoundUpTo(size, PageSize),
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,
                         -1, 0);
  int reserrno;
  if (UNLIKELY(internal_iserror(p, &reserrno)))
    ReportMmapFailureAndDie(size, mem_type, "allocate noreserve", reserrno);
  IncreaseTotalMmap(size);
  return (void *)p;
}

void *MmapFixedImpl(uptr fixed_addr, uptr size, bool tolerate_enomem) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap((void*)(fixed_addr & ~(PageSize - 1)),
      RoundUpTo(size, PageSize),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_FIXED,
      -1, 0);
  int reserrno;
  if (UNLIKELY(internal_iserror(p, &reserrno))) {
    if (tolerate_enomem && reserrno == ENOMEM)
      return nullptr;
    char mem_type[40];
    internal_snprintf(mem_type, sizeof(mem_type), "memory at address 0x%zx",
                      fixed_addr);
    ReportMmapFailureAndDie(size, mem_type, "allocate", reserrno);
  }
  IncreaseTotalMmap(size);
  return (void *)p;
}

void *MmapFixedOrDie(uptr fixed_addr, uptr size) {
  return MmapFixedImpl(fixed_addr, size, false /*tolerate_enomem*/);
}

void *MmapFixedOrDieOnFatalError(uptr fixed_addr, uptr size) {
  return MmapFixedImpl(fixed_addr, size, true /*tolerate_enomem*/);
}

bool MprotectNoAccess(uptr addr, uptr size) {
  return 0 == internal_mprotect((void*)addr, size, PROT_NONE);
}

bool MprotectReadOnly(uptr addr, uptr size) {
  return 0 == internal_mprotect((void *)addr, size, PROT_READ);
}

#if !SANITIZER_MAC
void MprotectMallocZones(void *addr, int prot) {}
#endif

fd_t OpenFile(const char *filename, FileAccessMode mode, error_t *errno_p) {
  int flags;
  switch (mode) {
    case RdOnly: flags = O_RDONLY; break;
    case WrOnly: flags = O_WRONLY | O_CREAT; break;
    case RdWr: flags = O_RDWR | O_CREAT; break;
  }
  fd_t res = internal_open(filename, flags, 0660);
  if (internal_iserror(res, errno_p))
    return kInvalidFd;
  return res;
}

void CloseFile(fd_t fd) {
  internal_close(fd);
}

bool ReadFromFile(fd_t fd, void *buff, uptr buff_size, uptr *bytes_read,
                  error_t *error_p) {
  uptr res = internal_read(fd, buff, buff_size);
  if (internal_iserror(res, error_p))
    return false;
  if (bytes_read)
    *bytes_read = res;
  return true;
}

bool WriteToFile(fd_t fd, const void *buff, uptr buff_size, uptr *bytes_written,
                 error_t *error_p) {
  uptr res = internal_write(fd, buff, buff_size);
  if (internal_iserror(res, error_p))
    return false;
  if (bytes_written)
    *bytes_written = res;
  return true;
}

bool RenameFile(const char *oldpath, const char *newpath, error_t *error_p) {
  uptr res = internal_rename(oldpath, newpath);
  return !internal_iserror(res, error_p);
}

void *MapFileToMemory(const char *file_name, uptr *buff_size) {
  fd_t fd = OpenFile(file_name, RdOnly);
  CHECK(fd != kInvalidFd);
  uptr fsize = internal_filesize(fd);
  CHECK_NE(fsize, (uptr)-1);
  CHECK_GT(fsize, 0);
  *buff_size = RoundUpTo(fsize, GetPageSizeCached());
  uptr map = internal_mmap(nullptr, *buff_size, PROT_READ, MAP_PRIVATE, fd, 0);
  return internal_iserror(map) ? nullptr : (void *)map;
}

void *MapWritableFileToMemory(void *addr, uptr size, fd_t fd, OFF_T offset) {
  uptr flags = MAP_SHARED;
  if (addr) flags |= MAP_FIXED;
  uptr p = internal_mmap(addr, size, PROT_READ | PROT_WRITE, flags, fd, offset);
  int mmap_errno = 0;
  if (internal_iserror(p, &mmap_errno)) {
    Printf("could not map writable file (%d, %lld, %zu): %zd, errno: %d\n",
           fd, (long long)offset, size, p, mmap_errno);
    return nullptr;
  }
  return (void *)p;
}

static inline bool IntervalsAreSeparate(uptr start1, uptr end1,
                                        uptr start2, uptr end2) {
  CHECK(start1 <= end1);
  CHECK(start2 <= end2);
  return (end1 < start2) || (end2 < start1);
}

// FIXME: this is thread-unsafe, but should not cause problems most of the time.
// When the shadow is mapped only a single thread usually exists (plus maybe
// several worker threads on Mac, which aren't expected to map big chunks of
// memory).
bool MemoryRangeIsAvailable(uptr range_start, uptr range_end) {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  MemoryMappedSegment segment;
  while (proc_maps.Next(&segment)) {
    if (segment.start == segment.end) continue;  // Empty range.
    CHECK_NE(0, segment.end);
    if (!IntervalsAreSeparate(segment.start, segment.end - 1, range_start,
                              range_end))
      return false;
  }
  return true;
}

void DumpProcessMap() {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  const sptr kBufSize = 4095;
  char *filename = (char*)MmapOrDie(kBufSize, __func__);
  MemoryMappedSegment segment(filename, kBufSize);
  Report("Process memory map follows:\n");
  while (proc_maps.Next(&segment)) {
    Printf("\t%p-%p\t%s\n", (void *)segment.start, (void *)segment.end,
           segment.filename);
  }
  Report("End of process memory map.\n");
  UnmapOrDie(filename, kBufSize);
}

const char *GetPwd() {
  return GetEnv("PWD");
}

bool IsPathSeparator(const char c) {
  return c == '/';
}

bool IsAbsolutePath(const char *path) {
  return path != nullptr && IsPathSeparator(path[0]);
}

void ReportFile::Write(const char *buffer, uptr length) {
  SpinMutexLock l(mu);
  static const char *kWriteError =
      "ReportFile::Write() can't output requested buffer!\n";
  ReopenIfNecessary();
  if (length != internal_write(fd, buffer, length)) {
    internal_write(fd, kWriteError, internal_strlen(kWriteError));
    Die();
  }
}

bool GetCodeRangeForFile(const char *module, uptr *start, uptr *end) {
  MemoryMappingLayout proc_maps(/*cache_enabled*/false);
  InternalScopedString buff(kMaxPathLength);
  MemoryMappedSegment segment(buff.data(), kMaxPathLength);
  while (proc_maps.Next(&segment)) {
    if (segment.IsExecutable() &&
        internal_strcmp(module, segment.filename) == 0) {
      *start = segment.start;
      *end = segment.end;
      return true;
    }
  }
  return false;
}

uptr SignalContext::GetAddress() const {
  auto si = static_cast<const siginfo_t *>(siginfo);
  return (uptr)si->si_addr;
}

bool SignalContext::IsMemoryAccess() const {
  auto si = static_cast<const siginfo_t *>(siginfo);
  return si->si_signo == SIGSEGV;
}

int SignalContext::GetType() const {
  return static_cast<const siginfo_t *>(siginfo)->si_signo;
}

const char *SignalContext::Describe() const {
  switch (GetType()) {
    case SIGFPE:
      return "FPE";
    case SIGILL:
      return "ILL";
    case SIGABRT:
      return "ABRT";
    case SIGSEGV:
      return "SEGV";
    case SIGBUS:
      return "BUS";
  }
  return "UNKNOWN SIGNAL";
}

} // namespace __sanitizer

#endif // SANITIZER_POSIX
//===-- sanitizer_posix_libcdep.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements libc-dependent POSIX-specific functions
// from sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_POSIX

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_platform_limits_netbsd.h"
#include "sanitizer_platform_limits_openbsd.h"
#include "sanitizer_platform_limits_posix.h"
#include "sanitizer_platform_limits_solaris.h"
#include "sanitizer_posix.h"
#include "sanitizer_procmaps.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if SANITIZER_FREEBSD
// The MAP_NORESERVE define has been removed in FreeBSD 11.x, and even before
// that, it was never implemented.  So just define it to zero.
#undef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

typedef void (*sa_sigaction_t)(int, siginfo_t *, void *);

namespace __sanitizer {

u32 GetUid() {
  return getuid();
}

uptr GetThreadSelf() {
  return (uptr)pthread_self();
}

void ReleaseMemoryPagesToOS(uptr beg, uptr end) {
  uptr page_size = GetPageSizeCached();
  uptr beg_aligned = RoundUpTo(beg, page_size);
  uptr end_aligned = RoundDownTo(end, page_size);
  if (beg_aligned < end_aligned)
    // In the default Solaris compilation environment, madvise() is declared
    // to take a caddr_t arg; casting it to void * results in an invalid
    // conversion error, so use char * instead.
    madvise((char *)beg_aligned, end_aligned - beg_aligned,
            SANITIZER_MADVISE_DONTNEED);
}

void NoHugePagesInRegion(uptr addr, uptr size) {
#ifdef MADV_NOHUGEPAGE  // May not be defined on old systems.
  madvise((void *)addr, size, MADV_NOHUGEPAGE);
#endif  // MADV_NOHUGEPAGE
}

void DontDumpShadowMemory(uptr addr, uptr length) {
#ifdef MADV_DONTDUMP
  madvise((void *)addr, length, MADV_DONTDUMP);
#endif
}

static rlim_t getlim(int res) {
  rlimit rlim;
  CHECK_EQ(0, getrlimit(res, &rlim));
  return rlim.rlim_cur;
}

static void setlim(int res, rlim_t lim) {
  // The following magic is to prevent clang from replacing it with memset.
  volatile struct rlimit rlim;
  rlim.rlim_cur = lim;
  rlim.rlim_max = lim;
  if (setrlimit(res, const_cast<struct rlimit *>(&rlim))) {
    Report("ERROR: %s setrlimit() failed %d\n", SanitizerToolName, errno);
    Die();
  }
}

void DisableCoreDumperIfNecessary() {
  if (common_flags()->disable_coredump) {
    setlim(RLIMIT_CORE, 0);
  }
}

bool StackSizeIsUnlimited() {
  rlim_t stack_size = getlim(RLIMIT_STACK);
  return (stack_size == RLIM_INFINITY);
}

uptr GetStackSizeLimitInBytes() {
  return (uptr)getlim(RLIMIT_STACK);
}

void SetStackSizeLimitInBytes(uptr limit) {
  setlim(RLIMIT_STACK, (rlim_t)limit);
  CHECK(!StackSizeIsUnlimited());
}

bool AddressSpaceIsUnlimited() {
  rlim_t as_size = getlim(RLIMIT_AS);
  return (as_size == RLIM_INFINITY);
}

void SetAddressSpaceUnlimited() {
  setlim(RLIMIT_AS, RLIM_INFINITY);
  CHECK(AddressSpaceIsUnlimited());
}

void SleepForSeconds(int seconds) {
  sleep(seconds);
}

void SleepForMillis(int millis) {
  usleep(millis * 1000);
}

void Abort() {
#if !SANITIZER_GO
  // If we are handling SIGABRT, unhandle it first.
  // TODO(vitalybuka): Check if handler belongs to sanitizer.
  if (GetHandleSignalMode(SIGABRT) != kHandleSignalNo) {
    struct sigaction sigact;
    internal_memset(&sigact, 0, sizeof(sigact));
    sigact.sa_sigaction = (sa_sigaction_t)SIG_DFL;
    internal_sigaction(SIGABRT, &sigact, nullptr);
  }
#endif

  abort();
}

int Atexit(void (*function)(void)) {
#if !SANITIZER_GO
  return atexit(function);
#else
  return 0;
#endif
}

bool SupportsColoredOutput(fd_t fd) {
  return isatty(fd) != 0;
}

#if !SANITIZER_GO
// TODO(glider): different tools may require different altstack size.
static const uptr kAltStackSize = SIGSTKSZ * 4;  // SIGSTKSZ is not enough.

void SetAlternateSignalStack() {
  stack_t altstack, oldstack;
  CHECK_EQ(0, sigaltstack(nullptr, &oldstack));
  // If the alternate stack is already in place, do nothing.
  // Android always sets an alternate stack, but it's too small for us.
  if (!SANITIZER_ANDROID && !(oldstack.ss_flags & SS_DISABLE)) return;
  // TODO(glider): the mapped stack should have the MAP_STACK flag in the
  // future. It is not required by man 2 sigaltstack now (they're using
  // malloc()).
  void* base = MmapOrDie(kAltStackSize, __func__);
  altstack.ss_sp = (char*) base;
  altstack.ss_flags = 0;
  altstack.ss_size = kAltStackSize;
  CHECK_EQ(0, sigaltstack(&altstack, nullptr));
}

void UnsetAlternateSignalStack() {
  stack_t altstack, oldstack;
  altstack.ss_sp = nullptr;
  altstack.ss_flags = SS_DISABLE;
  altstack.ss_size = kAltStackSize;  // Some sane value required on Darwin.
  CHECK_EQ(0, sigaltstack(&altstack, &oldstack));
  UnmapOrDie(oldstack.ss_sp, oldstack.ss_size);
}

static void MaybeInstallSigaction(int signum,
                                  SignalHandlerType handler) {
  if (GetHandleSignalMode(signum) == kHandleSignalNo) return;

  struct sigaction sigact;
  internal_memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = (sa_sigaction_t)handler;
  // Do not block the signal from being received in that signal's handler.
  // Clients are responsible for handling this correctly.
  sigact.sa_flags = SA_SIGINFO | SA_NODEFER;
  if (common_flags()->use_sigaltstack) sigact.sa_flags |= SA_ONSTACK;
  CHECK_EQ(0, internal_sigaction(signum, &sigact, nullptr));
  VReport(1, "Installed the sigaction for signal %d\n", signum);
}

void InstallDeadlySignalHandlers(SignalHandlerType handler) {
  // Set the alternate signal stack for the main thread.
  // This will cause SetAlternateSignalStack to be called twice, but the stack
  // will be actually set only once.
  if (common_flags()->use_sigaltstack) SetAlternateSignalStack();
  MaybeInstallSigaction(SIGSEGV, handler);
  MaybeInstallSigaction(SIGBUS, handler);
  MaybeInstallSigaction(SIGABRT, handler);
  MaybeInstallSigaction(SIGFPE, handler);
  MaybeInstallSigaction(SIGILL, handler);
  MaybeInstallSigaction(SIGTRAP, handler);
}

bool SignalContext::IsStackOverflow() const {
  // Access at a reasonable offset above SP, or slightly below it (to account
  // for x86_64 or PowerPC redzone, ARM push of multiple registers, etc) is
  // probably a stack overflow.
#ifdef __s390__
  // On s390, the fault address in siginfo points to start of the page, not
  // to the precise word that was accessed.  Mask off the low bits of sp to
  // take it into account.
  bool IsStackAccess = addr >= (sp & ~0xFFF) && addr < sp + 0xFFFF;
#else
  // Let's accept up to a page size away from top of stack. Things like stack
  // probing can trigger accesses with such large offsets.
  bool IsStackAccess = addr + GetPageSizeCached() > sp && addr < sp + 0xFFFF;
#endif

#if __powerpc__
  // Large stack frames can be allocated with e.g.
  //   lis r0,-10000
  //   stdux r1,r1,r0 # store sp to [sp-10000] and update sp by -10000
  // If the store faults then sp will not have been updated, so test above
  // will not work, because the fault address will be more than just "slightly"
  // below sp.
  if (!IsStackAccess && IsAccessibleMemoryRange(pc, 4)) {
    u32 inst = *(unsigned *)pc;
    u32 ra = (inst >> 16) & 0x1F;
    u32 opcd = inst >> 26;
    u32 xo = (inst >> 1) & 0x3FF;
    // Check for store-with-update to sp. The instructions we accept are:
    //   stbu rs,d(ra)          stbux rs,ra,rb
    //   sthu rs,d(ra)          sthux rs,ra,rb
    //   stwu rs,d(ra)          stwux rs,ra,rb
    //   stdu rs,ds(ra)         stdux rs,ra,rb
    // where ra is r1 (the stack pointer).
    if (ra == 1 &&
        (opcd == 39 || opcd == 45 || opcd == 37 || opcd == 62 ||
         (opcd == 31 && (xo == 247 || xo == 439 || xo == 183 || xo == 181))))
      IsStackAccess = true;
  }
#endif  // __powerpc__

  // We also check si_code to filter out SEGV caused by something else other
  // then hitting the guard page or unmapped memory, like, for example,
  // unaligned memory access.
  auto si = static_cast<const siginfo_t *>(siginfo);
  return IsStackAccess &&
         (si->si_code == si_SEGV_MAPERR || si->si_code == si_SEGV_ACCERR);
}

#endif  // SANITIZER_GO

bool IsAccessibleMemoryRange(uptr beg, uptr size) {
  uptr page_size = GetPageSizeCached();
  // Checking too large memory ranges is slow.
  CHECK_LT(size, page_size * 10);
  int sock_pair[2];
  if (pipe(sock_pair))
    return false;
  uptr bytes_written =
      internal_write(sock_pair[1], reinterpret_cast<void *>(beg), size);
  int write_errno;
  bool result;
  if (internal_iserror(bytes_written, &write_errno)) {
    CHECK_EQ(EFAULT, write_errno);
    result = false;
  } else {
    result = (bytes_written == size);
  }
  internal_close(sock_pair[0]);
  internal_close(sock_pair[1]);
  return result;
}

void PlatformPrepareForSandboxing(__sanitizer_sandbox_arguments *args) {
  // Some kinds of sandboxes may forbid filesystem access, so we won't be able
  // to read the file mappings from /proc/self/maps. Luckily, neither the
  // process will be able to load additional libraries, so it's fine to use the
  // cached mappings.
  MemoryMappingLayout::CacheMemoryMappings();
}

#if SANITIZER_ANDROID || SANITIZER_GO
int GetNamedMappingFd(const char *name, uptr size) {
  return -1;
}
#else
int GetNamedMappingFd(const char *name, uptr size) {
  if (!common_flags()->decorate_proc_maps)
    return -1;
  char shmname[200];
  CHECK(internal_strlen(name) < sizeof(shmname) - 10);
  internal_snprintf(shmname, sizeof(shmname), "%zu [%s]", internal_getpid(),
                    name);
  int fd = shm_open(shmname, O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
  CHECK_GE(fd, 0);
  int res = internal_ftruncate(fd, size);
  CHECK_EQ(0, res);
  res = shm_unlink(shmname);
  CHECK_EQ(0, res);
  return fd;
}
#endif

void *MmapFixedNoReserve(uptr fixed_addr, uptr size, const char *name) {
  int fd = name ? GetNamedMappingFd(name, size) : -1;
  unsigned flags = MAP_PRIVATE | MAP_FIXED | MAP_NORESERVE;
  if (fd == -1) flags |= MAP_ANON;

  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap((void *)(fixed_addr & ~(PageSize - 1)),
                         RoundUpTo(size, PageSize), PROT_READ | PROT_WRITE,
                         flags, fd, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno))
    Report("ERROR: %s failed to "
           "allocate 0x%zx (%zd) bytes at address %zx (errno: %d)\n",
           SanitizerToolName, size, size, fixed_addr, reserrno);
  IncreaseTotalMmap(size);
  return (void *)p;
}

uptr ReservedAddressRange::Init(uptr size, const char *name, uptr fixed_addr) {
  // We don't pass `name` along because, when you enable `decorate_proc_maps`
  // AND actually use a named mapping AND are using a sanitizer intercepting
  // `open` (e.g. TSAN, ESAN), then you'll get a failure during initialization.
  // TODO(flowerhack): Fix the implementation of GetNamedMappingFd to solve
  // this problem.
  base_ = fixed_addr ? MmapFixedNoAccess(fixed_addr, size) : MmapNoAccess(size);
  size_ = size;
  name_ = name;
  (void)os_handle_;  // unsupported
  return reinterpret_cast<uptr>(base_);
}

// Uses fixed_addr for now.
// Will use offset instead once we've implemented this function for real.
uptr ReservedAddressRange::Map(uptr fixed_addr, uptr size) {
  return reinterpret_cast<uptr>(MmapFixedOrDieOnFatalError(fixed_addr, size));
}

uptr ReservedAddressRange::MapOrDie(uptr fixed_addr, uptr size) {
  return reinterpret_cast<uptr>(MmapFixedOrDie(fixed_addr, size));
}

void ReservedAddressRange::Unmap(uptr addr, uptr size) {
  CHECK_LE(size, size_);
  if (addr == reinterpret_cast<uptr>(base_))
    // If we unmap the whole range, just null out the base.
    base_ = (size == size_) ? nullptr : reinterpret_cast<void*>(addr + size);
  else
    CHECK_EQ(addr + size, reinterpret_cast<uptr>(base_) + size_);
  size_ -= size;
  UnmapOrDie(reinterpret_cast<void*>(addr), size);
}

void *MmapFixedNoAccess(uptr fixed_addr, uptr size, const char *name) {
  int fd = name ? GetNamedMappingFd(name, size) : -1;
  unsigned flags = MAP_PRIVATE | MAP_FIXED | MAP_NORESERVE;
  if (fd == -1) flags |= MAP_ANON;

  return (void *)internal_mmap((void *)fixed_addr, size, PROT_NONE, flags, fd,
                               0);
}

void *MmapNoAccess(uptr size) {
  unsigned flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
  return (void *)internal_mmap(nullptr, size, PROT_NONE, flags, -1, 0);
}

// This function is defined elsewhere if we intercepted pthread_attr_getstack.
extern "C" {
SANITIZER_WEAK_ATTRIBUTE int
real_pthread_attr_getstack(void *attr, void **addr, size_t *size);
} // extern "C"

int my_pthread_attr_getstack(void *attr, void **addr, uptr *size) {
#if !SANITIZER_GO && !SANITIZER_MAC
  if (&real_pthread_attr_getstack)
    return real_pthread_attr_getstack((pthread_attr_t *)attr, addr,
                                      (size_t *)size);
#endif
  return pthread_attr_getstack((pthread_attr_t *)attr, addr, (size_t *)size);
}

#if !SANITIZER_GO
void AdjustStackSize(void *attr_) {
  pthread_attr_t *attr = (pthread_attr_t *)attr_;
  uptr stackaddr = 0;
  uptr stacksize = 0;
  my_pthread_attr_getstack(attr, (void**)&stackaddr, &stacksize);
  // GLibC will return (0 - stacksize) as the stack address in the case when
  // stacksize is set, but stackaddr is not.
  bool stack_set = (stackaddr != 0) && (stackaddr + stacksize != 0);
  // We place a lot of tool data into TLS, account for that.
  const uptr minstacksize = GetTlsSize() + 128*1024;
  if (stacksize < minstacksize) {
    if (!stack_set) {
      if (stacksize != 0) {
        VPrintf(1, "Sanitizer: increasing stacksize %zu->%zu\n", stacksize,
                minstacksize);
        pthread_attr_setstacksize(attr, minstacksize);
      }
    } else {
      Printf("Sanitizer: pre-allocated stack size is insufficient: "
             "%zu < %zu\n", stacksize, minstacksize);
      Printf("Sanitizer: pthread_create is likely to fail.\n");
    }
  }
}
#endif // !SANITIZER_GO

pid_t StartSubprocess(const char *program, const char *const argv[],
                      fd_t stdin_fd, fd_t stdout_fd, fd_t stderr_fd) {
  auto file_closer = at_scope_exit([&] {
    if (stdin_fd != kInvalidFd) {
      internal_close(stdin_fd);
    }
    if (stdout_fd != kInvalidFd) {
      internal_close(stdout_fd);
    }
    if (stderr_fd != kInvalidFd) {
      internal_close(stderr_fd);
    }
  });

  int pid = internal_fork();

  if (pid < 0) {
    int rverrno;
    if (internal_iserror(pid, &rverrno)) {
      Report("WARNING: failed to fork (errno %d)\n", rverrno);
    }
    return pid;
  }

  if (pid == 0) {
    // Child subprocess
    if (stdin_fd != kInvalidFd) {
      internal_close(STDIN_FILENO);
      internal_dup2(stdin_fd, STDIN_FILENO);
      internal_close(stdin_fd);
    }
    if (stdout_fd != kInvalidFd) {
      internal_close(STDOUT_FILENO);
      internal_dup2(stdout_fd, STDOUT_FILENO);
      internal_close(stdout_fd);
    }
    if (stderr_fd != kInvalidFd) {
      internal_close(STDERR_FILENO);
      internal_dup2(stderr_fd, STDERR_FILENO);
      internal_close(stderr_fd);
    }

    for (int fd = sysconf(_SC_OPEN_MAX); fd > 2; fd--) internal_close(fd);

    execv(program, const_cast<char **>(&argv[0]));
    internal__exit(1);
  }

  return pid;
}

bool IsProcessRunning(pid_t pid) {
  int process_status;
  uptr waitpid_status = internal_waitpid(pid, &process_status, WNOHANG);
  int local_errno;
  if (internal_iserror(waitpid_status, &local_errno)) {
    VReport(1, "Waiting on the process failed (errno %d).\n", local_errno);
    return false;
  }
  return waitpid_status == 0;
}

int WaitForProcess(pid_t pid) {
  int process_status;
  uptr waitpid_status = internal_waitpid(pid, &process_status, 0);
  int local_errno;
  if (internal_iserror(waitpid_status, &local_errno)) {
    VReport(1, "Waiting on the process failed (errno %d).\n", local_errno);
    return -1;
  }
  return process_status;
}

bool IsStateDetached(int state) {
  return state == PTHREAD_CREATE_DETACHED;
}

} // namespace __sanitizer

#endif // SANITIZER_POSIX
//===-- sanitizer_procmaps_common.cc --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Information about the process mappings (common parts).
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD ||                \
    SANITIZER_OPENBSD || SANITIZER_SOLARIS

#include "sanitizer_common.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"

namespace __sanitizer {

static ProcSelfMapsBuff cached_proc_self_maps;
static StaticSpinMutex cache_lock;

static int TranslateDigit(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

// Parse a number and promote 'p' up to the first non-digit character.
static uptr ParseNumber(const char **p, int base) {
  uptr n = 0;
  int d;
  CHECK(base >= 2 && base <= 16);
  while ((d = TranslateDigit(**p)) >= 0 && d < base) {
    n = n * base + d;
    (*p)++;
  }
  return n;
}

bool IsDecimal(char c) {
  int d = TranslateDigit(c);
  return d >= 0 && d < 10;
}

uptr ParseDecimal(const char **p) {
  return ParseNumber(p, 10);
}

bool IsHex(char c) {
  int d = TranslateDigit(c);
  return d >= 0 && d < 16;
}

uptr ParseHex(const char **p) {
  return ParseNumber(p, 16);
}

void MemoryMappedSegment::AddAddressRanges(LoadedModule *module) {
  // data_ should be unused on this platform
  CHECK(!data_);
  module->addAddressRange(start, end, IsExecutable(), IsWritable());
}

MemoryMappingLayout::MemoryMappingLayout(bool cache_enabled) {
  // FIXME: in the future we may want to cache the mappings on demand only.
  if (cache_enabled)
    CacheMemoryMappings();

  // Read maps after the cache update to capture the maps/unmaps happening in
  // the process of updating.
  ReadProcMaps(&data_.proc_self_maps);
  if (cache_enabled && data_.proc_self_maps.mmaped_size == 0)
    LoadFromCache();
  CHECK_GT(data_.proc_self_maps.mmaped_size, 0);
  CHECK_GT(data_.proc_self_maps.len, 0);

  Reset();
}

MemoryMappingLayout::~MemoryMappingLayout() {
  // Only unmap the buffer if it is different from the cached one. Otherwise
  // it will be unmapped when the cache is refreshed.
  if (data_.proc_self_maps.data != cached_proc_self_maps.data)
    UnmapOrDie(data_.proc_self_maps.data, data_.proc_self_maps.mmaped_size);
}

void MemoryMappingLayout::Reset() {
  data_.current = data_.proc_self_maps.data;
}

// static
void MemoryMappingLayout::CacheMemoryMappings() {
  ProcSelfMapsBuff new_proc_self_maps;
  ReadProcMaps(&new_proc_self_maps);
  // Don't invalidate the cache if the mappings are unavailable.
  if (new_proc_self_maps.mmaped_size == 0)
    return;
  SpinMutexLock l(&cache_lock);
  if (cached_proc_self_maps.mmaped_size)
    UnmapOrDie(cached_proc_self_maps.data, cached_proc_self_maps.mmaped_size);
  cached_proc_self_maps = new_proc_self_maps;
}

void MemoryMappingLayout::LoadFromCache() {
  SpinMutexLock l(&cache_lock);
  if (cached_proc_self_maps.data)
    data_.proc_self_maps = cached_proc_self_maps;
}

void MemoryMappingLayout::DumpListOfModules(
    InternalMmapVectorNoCtor<LoadedModule> *modules) {
  Reset();
  InternalScopedString module_name(kMaxPathLength);
  MemoryMappedSegment segment(module_name.data(), module_name.size());
  for (uptr i = 0; Next(&segment); i++) {
    const char *cur_name = segment.filename;
    if (cur_name[0] == '\0')
      continue;
    // Don't subtract 'cur_beg' from the first entry:
    // * If a binary is compiled w/o -pie, then the first entry in
    //   process maps is likely the binary itself (all dynamic libs
    //   are mapped higher in address space). For such a binary,
    //   instruction offset in binary coincides with the actual
    //   instruction address in virtual memory (as code section
    //   is mapped to a fixed memory range).
    // * If a binary is compiled with -pie, all the modules are
    //   mapped high at address space (in particular, higher than
    //   shadow memory of the tool), so the module can't be the
    //   first entry.
    uptr base_address = (i ? segment.start : 0) - segment.offset;
    LoadedModule cur_module;
    cur_module.set(cur_name, base_address);
    segment.AddAddressRanges(&cur_module);
    modules->push_back(cur_module);
  }
}

void GetMemoryProfile(fill_profile_f cb, uptr *stats, uptr stats_size) {
  char *smaps = nullptr;
  uptr smaps_cap = 0;
  uptr smaps_len = 0;
  if (!ReadFileToBuffer("/proc/self/smaps", &smaps, &smaps_cap, &smaps_len))
    return;
  uptr start = 0;
  bool file = false;
  const char *pos = smaps;
  while (pos < smaps + smaps_len) {
    if (IsHex(pos[0])) {
      start = ParseHex(&pos);
      for (; *pos != '/' && *pos > '\n'; pos++) {}
      file = *pos == '/';
    } else if (internal_strncmp(pos, "Rss:", 4) == 0) {
      while (!IsDecimal(*pos)) pos++;
      uptr rss = ParseDecimal(&pos) * 1024;
      cb(start, rss, file, stats, stats_size);
    }
    while (*pos++ != '\n') {}
  }
  UnmapOrDie(smaps, smaps_cap);
}

} // namespace __sanitizer

#endif
//===-- sanitizer_procmaps_linux.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Information about the process mappings (Linux-specific parts).
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_LINUX
#include "sanitizer_common.h"
#include "sanitizer_procmaps.h"

namespace __sanitizer {

void ReadProcMaps(ProcSelfMapsBuff *proc_maps) {
  if (!ReadFileToBuffer("/proc/self/maps", &proc_maps->data,
                        &proc_maps->mmaped_size, &proc_maps->len)) {
    proc_maps->data = nullptr;
    proc_maps->mmaped_size = 0;
    proc_maps->len = 0;
  }
}

static bool IsOneOf(char c, char c1, char c2) {
  return c == c1 || c == c2;
}

bool MemoryMappingLayout::Next(MemoryMappedSegment *segment) {
  char *last = data_.proc_self_maps.data + data_.proc_self_maps.len;
  if (data_.current >= last) return false;
  char *next_line =
      (char *)internal_memchr(data_.current, '\n', last - data_.current);
  if (next_line == 0)
    next_line = last;
  // Example: 08048000-08056000 r-xp 00000000 03:0c 64593   /foo/bar
  segment->start = ParseHex(&data_.current);
  CHECK_EQ(*data_.current++, '-');
  segment->end = ParseHex(&data_.current);
  CHECK_EQ(*data_.current++, ' ');
  CHECK(IsOneOf(*data_.current, '-', 'r'));
  segment->protection = 0;
  if (*data_.current++ == 'r') segment->protection |= kProtectionRead;
  CHECK(IsOneOf(*data_.current, '-', 'w'));
  if (*data_.current++ == 'w') segment->protection |= kProtectionWrite;
  CHECK(IsOneOf(*data_.current, '-', 'x'));
  if (*data_.current++ == 'x') segment->protection |= kProtectionExecute;
  CHECK(IsOneOf(*data_.current, 's', 'p'));
  if (*data_.current++ == 's') segment->protection |= kProtectionShared;
  CHECK_EQ(*data_.current++, ' ');
  segment->offset = ParseHex(&data_.current);
  CHECK_EQ(*data_.current++, ' ');
  ParseHex(&data_.current);
  CHECK_EQ(*data_.current++, ':');
  ParseHex(&data_.current);
  CHECK_EQ(*data_.current++, ' ');
  while (IsDecimal(*data_.current)) data_.current++;
  // Qemu may lack the trailing space.
  // https://github.com/google/sanitizers/issues/160
  // CHECK_EQ(*data_.current++, ' ');
  // Skip spaces.
  while (data_.current < next_line && *data_.current == ' ') data_.current++;
  // Fill in the filename.
  if (segment->filename) {
    uptr len =
        Min((uptr)(next_line - data_.current), segment->filename_size - 1);
    internal_strncpy(segment->filename, data_.current, len);
    segment->filename[len] = 0;
  }

  data_.current = next_line + 1;
  return true;
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
//===-- sanitizer_linux.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD || \
    SANITIZER_OPENBSD || SANITIZER_SOLARIS

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_getauxval.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_linux.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"

#if SANITIZER_LINUX
#include <asm/param.h>
#endif

#if SANITIZER_NETBSD
#include <lwp.h>
#endif

// For mips64, syscall(__NR_stat) fills the buffer in the 'struct kernel_stat'
// format. Struct kernel_stat is defined as 'struct stat' in asm/stat.h. To
// access stat from asm/stat.h, without conflicting with definition in
// sys/stat.h, we use this trick.
#if defined(__mips64)
#include <asm/unistd.h>
#include <sys/types.h>
#define stat kernel_stat
#include <asm/stat.h>
#undef stat
#endif

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#if !SANITIZER_SOLARIS
#include <sys/ptrace.h>
#endif
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#if !SANITIZER_OPENBSD
#include <ucontext.h>
#endif
#if SANITIZER_OPENBSD
#include <signal.h>
#include <sys/futex.h>
#endif
#include <unistd.h>

#if SANITIZER_LINUX
#include <sys/utsname.h>
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
#include <sys/personality.h>
#endif

#if SANITIZER_FREEBSD
#include <sys/exec.h>
#include <sys/sysctl.h>
#include <machine/atomic.h>
extern "C" {
// <sys/umtx.h> must be included after <errno.h> and <sys/types.h> on
// FreeBSD 9.2 and 10.0.
#include <sys/umtx.h>
}
#include <sys/thr.h>
#endif  // SANITIZER_FREEBSD

#if SANITIZER_NETBSD
#include <limits.h>  // For NAME_MAX
#include <sys/sysctl.h>
#include <sys/exec.h>
extern struct ps_strings *__ps_strings;
#endif  // SANITIZER_NETBSD

#if SANITIZER_SOLARIS
#include <stdlib.h>
#include <thread.h>
#define environ _environ
#endif

#if !SANITIZER_ANDROID
#include <sys/signal.h>
#endif

extern char **environ;

#if SANITIZER_LINUX
// <linux/time.h>
struct kernel_timeval {
  long tv_sec;
  long tv_usec;
};

// <linux/futex.h> is broken on some linux distributions.
const int FUTEX_WAIT = 0;
const int FUTEX_WAKE = 1;
#endif  // SANITIZER_LINUX

// Are we using 32-bit or 64-bit Linux syscalls?
// x32 (which defines __x86_64__) has SANITIZER_WORDSIZE == 32
// but it still needs to use 64-bit syscalls.
#if SANITIZER_LINUX && (defined(__x86_64__) || defined(__powerpc64__) ||       \
                        SANITIZER_WORDSIZE == 64)
# define SANITIZER_LINUX_USES_64BIT_SYSCALLS 1
#else
# define SANITIZER_LINUX_USES_64BIT_SYSCALLS 0
#endif

#if defined(__x86_64__) || SANITIZER_MIPS64
extern "C" {
extern void internal_sigreturn();
}
#endif

// Note : FreeBSD had implemented both
// Linux and OpenBSD apis, available from
// future 12.x version most likely
#if SANITIZER_LINUX && defined(__NR_getrandom)
# if !defined(GRND_NONBLOCK)
#  define GRND_NONBLOCK 1
# endif
# define SANITIZER_USE_GETRANDOM 1
#else
# define SANITIZER_USE_GETRANDOM 0
#endif  // SANITIZER_LINUX && defined(__NR_getrandom)

#if SANITIZER_OPENBSD
# define SANITIZER_USE_GETENTROPY 1
#else
# define SANITIZER_USE_GETENTROPY 0
#endif // SANITIZER_USE_GETENTROPY

namespace __sanitizer {

#if SANITIZER_LINUX && defined(__x86_64__)
#include "sanitizer_syscall_linux_x86_64.inc"
#elif SANITIZER_LINUX && defined(__aarch64__)
#include "sanitizer_syscall_linux_aarch64.inc"
#elif SANITIZER_LINUX && defined(__arm__)
#include "sanitizer_syscall_linux_arm.inc"
#else
#include "sanitizer_syscall_generic.inc"
#endif

// --------------- sanitizer_libc.h
#if !SANITIZER_SOLARIS
#if !SANITIZER_S390 && !SANITIZER_OPENBSD
uptr internal_mmap(void *addr, uptr length, int prot, int flags, int fd,
                   OFF_T offset) {
#if SANITIZER_NETBSD
  return internal_syscall_ptr(SYSCALL(mmap), addr, length, prot, flags, fd,
                              (long)0, offset);
#elif SANITIZER_FREEBSD || SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(SYSCALL(mmap), (uptr)addr, length, prot, flags, fd,
                          offset);
#else
  // mmap2 specifies file offset in 4096-byte units.
  CHECK(IsAligned(offset, 4096));
  return internal_syscall(SYSCALL(mmap2), addr, length, prot, flags, fd,
                          offset / 4096);
#endif
}
#endif // !SANITIZER_S390 && !SANITIZER_OPENBSD

#if !SANITIZER_OPENBSD
uptr internal_munmap(void *addr, uptr length) {
  return internal_syscall_ptr(SYSCALL(munmap), (uptr)addr, length);
}

int internal_mprotect(void *addr, uptr length, int prot) {
  return internal_syscall_ptr(SYSCALL(mprotect), (uptr)addr, length, prot);
}
#endif

uptr internal_close(fd_t fd) {
  return internal_syscall(SYSCALL(close), fd);
}

uptr internal_open(const char *filename, int flags) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(SYSCALL(openat), AT_FDCWD, (uptr)filename, flags);
#else
  return internal_syscall_ptr(SYSCALL(open), (uptr)filename, flags);
#endif
}

uptr internal_open(const char *filename, int flags, u32 mode) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(SYSCALL(openat), AT_FDCWD, (uptr)filename, flags,
                          mode);
#else
  return internal_syscall_ptr(SYSCALL(open), (uptr)filename, flags, mode);
#endif
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  sptr res;
  HANDLE_EINTR(res, (sptr)internal_syscall_ptr(SYSCALL(read), fd, (uptr)buf,
               count));
  return res;
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  sptr res;
  HANDLE_EINTR(res, (sptr)internal_syscall_ptr(SYSCALL(write), fd, (uptr)buf,
               count));
  return res;
}

uptr internal_ftruncate(fd_t fd, uptr size) {
  sptr res;
#if SANITIZER_NETBSD
  HANDLE_EINTR(res, internal_syscall64(SYSCALL(ftruncate), fd, 0, (s64)size));
#else
  HANDLE_EINTR(res, (sptr)internal_syscall(SYSCALL(ftruncate), fd,
               (OFF_T)size));
#endif
  return res;
}

#if !SANITIZER_LINUX_USES_64BIT_SYSCALLS && SANITIZER_LINUX
static void stat64_to_stat(struct stat64 *in, struct stat *out) {
  internal_memset(out, 0, sizeof(*out));
  out->st_dev = in->st_dev;
  out->st_ino = in->st_ino;
  out->st_mode = in->st_mode;
  out->st_nlink = in->st_nlink;
  out->st_uid = in->st_uid;
  out->st_gid = in->st_gid;
  out->st_rdev = in->st_rdev;
  out->st_size = in->st_size;
  out->st_blksize = in->st_blksize;
  out->st_blocks = in->st_blocks;
  out->st_atime = in->st_atime;
  out->st_mtime = in->st_mtime;
  out->st_ctime = in->st_ctime;
}
#endif

#if defined(__mips64)
// Undefine compatibility macros from <sys/stat.h>
// so that they would not clash with the kernel_stat
// st_[a|m|c]time fields
#undef st_atime
#undef st_mtime
#undef st_ctime
#if defined(SANITIZER_ANDROID)
// Bionic sys/stat.h defines additional macros
// for compatibility with the old NDKs and
// they clash with the kernel_stat structure
// st_[a|m|c]time_nsec fields.
#undef st_atime_nsec
#undef st_mtime_nsec
#undef st_ctime_nsec
#endif
static void kernel_stat_to_stat(struct kernel_stat *in, struct stat *out) {
  internal_memset(out, 0, sizeof(*out));
  out->st_dev = in->st_dev;
  out->st_ino = in->st_ino;
  out->st_mode = in->st_mode;
  out->st_nlink = in->st_nlink;
  out->st_uid = in->st_uid;
  out->st_gid = in->st_gid;
  out->st_rdev = in->st_rdev;
  out->st_size = in->st_size;
  out->st_blksize = in->st_blksize;
  out->st_blocks = in->st_blocks;
#if defined(__USE_MISC)     || \
    defined(__USE_XOPEN2K8) || \
    defined(SANITIZER_ANDROID)
  out->st_atim.tv_sec = in->st_atime;
  out->st_atim.tv_nsec = in->st_atime_nsec;
  out->st_mtim.tv_sec = in->st_mtime;
  out->st_mtim.tv_nsec = in->st_mtime_nsec;
  out->st_ctim.tv_sec = in->st_ctime;
  out->st_ctim.tv_nsec = in->st_ctime_nsec;
#else
  out->st_atime = in->st_atime;
  out->st_atimensec = in->st_atime_nsec;
  out->st_mtime = in->st_mtime;
  out->st_mtimensec = in->st_mtime_nsec;
  out->st_ctime = in->st_ctime;
  out->st_atimensec = in->st_ctime_nsec;
#endif
}
#endif

uptr internal_stat(const char *path, void *buf) {
#if SANITIZER_FREEBSD || SANITIZER_NETBSD || SANITIZER_OPENBSD
  return internal_syscall_ptr(SYSCALL(fstatat), AT_FDCWD, (uptr)path, (uptr)buf,
                              0);
#elif SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(SYSCALL(newfstatat), AT_FDCWD, (uptr)path, (uptr)buf,
                          0);
#elif SANITIZER_LINUX_USES_64BIT_SYSCALLS
# if defined(__mips64)
  // For mips64, stat syscall fills buffer in the format of kernel_stat
  struct kernel_stat kbuf;
  int res = internal_syscall(SYSCALL(stat), path, &kbuf);
  kernel_stat_to_stat(&kbuf, (struct stat *)buf);
  return res;
# else
  return internal_syscall(SYSCALL(stat), (uptr)path, (uptr)buf);
# endif
#else
  struct stat64 buf64;
  int res = internal_syscall(SYSCALL(stat64), path, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_lstat(const char *path, void *buf) {
#if SANITIZER_NETBSD
  return internal_syscall_ptr(SYSCALL(lstat), path, buf);
#elif SANITIZER_FREEBSD || SANITIZER_OPENBSD
  return internal_syscall(SYSCALL(fstatat), AT_FDCWD, (uptr)path, (uptr)buf,
                          AT_SYMLINK_NOFOLLOW);
#elif SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(SYSCALL(newfstatat), AT_FDCWD, (uptr)path, (uptr)buf,
                          AT_SYMLINK_NOFOLLOW);
#elif SANITIZER_LINUX_USES_64BIT_SYSCALLS
# if SANITIZER_MIPS64
  // For mips64, lstat syscall fills buffer in the format of kernel_stat
  struct kernel_stat kbuf;
  int res = internal_syscall(SYSCALL(lstat), path, &kbuf);
  kernel_stat_to_stat(&kbuf, (struct stat *)buf);
  return res;
# else
  return internal_syscall(SYSCALL(lstat), (uptr)path, (uptr)buf);
# endif
#else
  struct stat64 buf64;
  int res = internal_syscall(SYSCALL(lstat64), path, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_fstat(fd_t fd, void *buf) {
#if SANITIZER_FREEBSD || SANITIZER_NETBSD || SANITIZER_OPENBSD ||              \
    SANITIZER_LINUX_USES_64BIT_SYSCALLS
#if SANITIZER_MIPS64 && !SANITIZER_NETBSD && !SANITIZER_OPENBSD
  // For mips64, fstat syscall fills buffer in the format of kernel_stat
  struct kernel_stat kbuf;
  int res = internal_syscall(SYSCALL(fstat), fd, &kbuf);
  kernel_stat_to_stat(&kbuf, (struct stat *)buf);
  return res;
# else
  return internal_syscall_ptr(SYSCALL(fstat), fd, (uptr)buf);
# endif
#else
  struct stat64 buf64;
  int res = internal_syscall(SYSCALL(fstat64), fd, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup2(int oldfd, int newfd) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(SYSCALL(dup3), oldfd, newfd, 0);
#else
  return internal_syscall(SYSCALL(dup2), oldfd, newfd);
#endif
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(SYSCALL(readlinkat), AT_FDCWD, (uptr)path, (uptr)buf,
                          bufsize);
#elif SANITIZER_OPENBSD
  return internal_syscall_ptr(SYSCALL(readlinkat), AT_FDCWD, (uptr)path,
                              (uptr)buf, bufsize);
#else
  return internal_syscall_ptr(SYSCALL(readlink), path, buf, bufsize);
#endif
}

uptr internal_unlink(const char *path) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS || SANITIZER_OPENBSD
  return internal_syscall(SYSCALL(unlinkat), AT_FDCWD, (uptr)path, 0);
#else
  return internal_syscall_ptr(SYSCALL(unlink), (uptr)path);
#endif
}

uptr internal_rename(const char *oldpath, const char *newpath) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS || SANITIZER_OPENBSD
  return internal_syscall(SYSCALL(renameat), AT_FDCWD, (uptr)oldpath, AT_FDCWD,
                          (uptr)newpath);
#else
  return internal_syscall_ptr(SYSCALL(rename), (uptr)oldpath, (uptr)newpath);
#endif
}

uptr internal_sched_yield() {
  return internal_syscall(SYSCALL(sched_yield));
}

void internal__exit(int exitcode) {
#if SANITIZER_FREEBSD || SANITIZER_NETBSD || SANITIZER_OPENBSD
  internal_syscall(SYSCALL(exit), exitcode);
#else
  internal_syscall(SYSCALL(exit_group), exitcode);
#endif
  Die();  // Unreachable.
}

unsigned int internal_sleep(unsigned int seconds) {
  struct timespec ts;
  ts.tv_sec = 1;
  ts.tv_nsec = 0;
  int res = internal_syscall_ptr(SYSCALL(nanosleep), &ts, &ts);
  if (res) return ts.tv_sec;
  return 0;
}

uptr internal_execve(const char *filename, char *const argv[],
                     char *const envp[]) {
  return internal_syscall_ptr(SYSCALL(execve), (uptr)filename, (uptr)argv,
                              (uptr)envp);
}
#endif // !SANITIZER_SOLARIS

// ----------------- sanitizer_common.h
bool FileExists(const char *filename) {
  struct stat st;
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  if (internal_syscall(SYSCALL(newfstatat), AT_FDCWD, filename, &st, 0))
#else
  if (internal_stat(filename, &st))
#endif
    return false;
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
}

tid_t GetTid() {
#if SANITIZER_FREEBSD
  long Tid;
  thr_self(&Tid);
  return Tid;
#elif SANITIZER_OPENBSD
  return internal_syscall(SYSCALL(getthrid));
#elif SANITIZER_NETBSD
  return _lwp_self();
#elif SANITIZER_SOLARIS
  return thr_self();
#else
  return internal_syscall(SYSCALL(gettid));
#endif
}

#if !SANITIZER_SOLARIS
u64 NanoTime() {
#if SANITIZER_FREEBSD || SANITIZER_NETBSD || SANITIZER_OPENBSD
  timeval tv;
#else
  kernel_timeval tv;
#endif
  internal_memset(&tv, 0, sizeof(tv));
  internal_syscall_ptr(SYSCALL(gettimeofday), &tv, 0);
  return (u64)tv.tv_sec * 1000*1000*1000 + tv.tv_usec * 1000;
}

uptr internal_clock_gettime(__sanitizer_clockid_t clk_id, void *tp) {
  return internal_syscall_ptr(SYSCALL(clock_gettime), clk_id, tp);
}
#endif // !SANITIZER_SOLARIS

// Like getenv, but reads env directly from /proc (on Linux) or parses the
// 'environ' array (on some others) and does not use libc. This function
// should be called first inside __asan_init.
const char *GetEnv(const char *name) {
#if SANITIZER_FREEBSD || SANITIZER_NETBSD || SANITIZER_OPENBSD ||              \
    SANITIZER_SOLARIS
  if (::environ != 0) {
    uptr NameLen = internal_strlen(name);
    for (char **Env = ::environ; *Env != 0; Env++) {
      if (internal_strncmp(*Env, name, NameLen) == 0 && (*Env)[NameLen] == '=')
        return (*Env) + NameLen + 1;
    }
  }
  return 0;  // Not found.
#elif SANITIZER_LINUX
  static char *environ;
  static uptr len;
  static bool inited;
  if (!inited) {
    inited = true;
    uptr environ_size;
    if (!ReadFileToBuffer("/proc/self/environ", &environ, &environ_size, &len))
      environ = nullptr;
  }
  if (!environ || len == 0) return nullptr;
  uptr namelen = internal_strlen(name);
  const char *p = environ;
  while (*p != '\0') {  // will happen at the \0\0 that terminates the buffer
    // proc file has the format NAME=value\0NAME=value\0NAME=value\0...
    const char* endp =
        (char*)internal_memchr(p, '\0', len - (p - environ));
    if (!endp)  // this entry isn't NUL terminated
      return nullptr;
    else if (!internal_memcmp(p, name, namelen) && p[namelen] == '=')  // Match.
      return p + namelen + 1;  // point after =
    p = endp + 1;
  }
  return nullptr;  // Not found.
#else
#error "Unsupported platform"
#endif
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD && !SANITIZER_OPENBSD
extern "C" {
SANITIZER_WEAK_ATTRIBUTE extern void *__libc_stack_end;
}
#endif

#if !SANITIZER_GO && !SANITIZER_FREEBSD && !SANITIZER_NETBSD &&                \
    !SANITIZER_OPENBSD
static void ReadNullSepFileToArray(const char *path, char ***arr,
                                   int arr_size) {
  char *buff;
  uptr buff_size;
  uptr buff_len;
  *arr = (char **)MmapOrDie(arr_size * sizeof(char *), "NullSepFileArray");
  if (!ReadFileToBuffer(path, &buff, &buff_size, &buff_len, 1024 * 1024)) {
    (*arr)[0] = nullptr;
    return;
  }
  (*arr)[0] = buff;
  int count, i;
  for (count = 1, i = 1; ; i++) {
    if (buff[i] == 0) {
      if (buff[i+1] == 0) break;
      (*arr)[count] = &buff[i+1];
      CHECK_LE(count, arr_size - 1);  // FIXME: make this more flexible.
      count++;
    }
  }
  (*arr)[count] = nullptr;
}
#endif

#if !SANITIZER_OPENBSD
static void GetArgsAndEnv(char ***argv, char ***envp) {
#if SANITIZER_FREEBSD
  // On FreeBSD, retrieving the argument and environment arrays is done via the
  // kern.ps_strings sysctl, which returns a pointer to a structure containing
  // this information. See also <sys/exec.h>.
  ps_strings *pss;
  size_t sz = sizeof(pss);
  if (sysctlbyname("kern.ps_strings", &pss, &sz, NULL, 0) == -1) {
    Printf("sysctl kern.ps_strings failed\n");
    Die();
  }
  *argv = pss->ps_argvstr;
  *envp = pss->ps_envstr;
#elif SANITIZER_NETBSD
  *argv = __ps_strings->ps_argvstr;
  *envp = __ps_strings->ps_envstr;
#else // SANITIZER_FREEBSD
#if !SANITIZER_GO
  if (&__libc_stack_end) {
#endif // !SANITIZER_GO
    uptr* stack_end = (uptr*)__libc_stack_end;
    int argc = *stack_end;
    *argv = (char**)(stack_end + 1);
    *envp = (char**)(stack_end + argc + 2);
#if !SANITIZER_GO
  } else {
    static const int kMaxArgv = 2000, kMaxEnvp = 2000;
    ReadNullSepFileToArray("/proc/self/cmdline", argv, kMaxArgv);
    ReadNullSepFileToArray("/proc/self/environ", envp, kMaxEnvp);
  }
#endif // !SANITIZER_GO
#endif // SANITIZER_FREEBSD
}

char **GetArgv() {
  char **argv, **envp;
  GetArgsAndEnv(&argv, &envp);
  return argv;
}

void ReExec() {
  char **argv, **envp;
  const char *pathname = "/proc/self/exe";

#if SANITIZER_NETBSD
  static const int name[] = {
    CTL_KERN, KERN_PROC_ARGS, -1, KERN_PROC_PATHNAME,
  };
  char path[400];
  size_t len;

  len = sizeof(path);
  if (sysctl(name, ARRAY_SIZE(name), path, &len, NULL, 0) != -1)
    pathname = path;
#elif SANITIZER_SOLARIS
  pathname = getexecname();
  CHECK_NE(pathname, NULL);
#endif

  GetArgsAndEnv(&argv, &envp);
  uptr rv = internal_execve(pathname, argv, envp);
  int rverrno;
  CHECK_EQ(internal_iserror(rv, &rverrno), true);
  Printf("execve failed, errno %d\n", rverrno);
  Die();
}
#endif

#if !SANITIZER_SOLARIS
enum MutexState {
  MtxUnlocked = 0,
  MtxLocked = 1,
  MtxSleeping = 2
};

BlockingMutex::BlockingMutex() {
  internal_memset(this, 0, sizeof(*this));
}

void BlockingMutex::Lock() {
  CHECK_EQ(owner_, 0);
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  if (atomic_exchange(m, MtxLocked, memory_order_acquire) == MtxUnlocked)
    return;
  while (atomic_exchange(m, MtxSleeping, memory_order_acquire) != MtxUnlocked) {
#if SANITIZER_FREEBSD
    _umtx_op(m, UMTX_OP_WAIT_UINT, MtxSleeping, 0, 0);
#elif SANITIZER_NETBSD
    sched_yield(); /* No userspace futex-like synchronization */
#else
    internal_syscall(SYSCALL(futex), (uptr)m, FUTEX_WAIT, MtxSleeping, 0, 0, 0);
#endif
  }
}

void BlockingMutex::Unlock() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  u32 v = atomic_exchange(m, MtxUnlocked, memory_order_release);
  CHECK_NE(v, MtxUnlocked);
  if (v == MtxSleeping) {
#if SANITIZER_FREEBSD
    _umtx_op(m, UMTX_OP_WAKE, 1, 0, 0);
#elif SANITIZER_NETBSD
                   /* No userspace futex-like synchronization */
#else
    internal_syscall(SYSCALL(futex), (uptr)m, FUTEX_WAKE, 1, 0, 0, 0);
#endif
  }
}

void BlockingMutex::CheckLocked() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  CHECK_NE(MtxUnlocked, atomic_load(m, memory_order_relaxed));
}
#endif // !SANITIZER_SOLARIS

// ----------------- sanitizer_linux.h
// The actual size of this structure is specified by d_reclen.
// Note that getdents64 uses a different structure format. We only provide the
// 32-bit syscall here.
#if SANITIZER_NETBSD || SANITIZER_OPENBSD
// struct dirent is different for Linux and us. At this moment, we use only
// d_fileno (Linux call this d_ino), d_reclen, and d_name.
struct linux_dirent {
  u64 d_ino;  // d_fileno
  u16 d_reclen;
  u16 d_namlen;  // not used
  u8 d_type;     // not used
  char d_name[NAME_MAX + 1];
};
#else
struct linux_dirent {
#if SANITIZER_X32 || defined(__aarch64__)
  u64 d_ino;
  u64 d_off;
#else
  unsigned long      d_ino;
  unsigned long      d_off;
#endif
  unsigned short     d_reclen;
#ifdef __aarch64__
  unsigned char      d_type;
#endif
  char               d_name[256];
};
#endif

#if !SANITIZER_SOLARIS
// Syscall wrappers.
uptr internal_ptrace(int request, int pid, void *addr, void *data) {
#if SANITIZER_NETBSD
  // XXX We need additional work for ptrace:
  //   - for request, we use PT_FOO whereas Linux uses PTRACE_FOO
  //   - data is int for us, but void * for Linux
  //   - Linux sometimes uses data in the case where we use addr instead
  // At this moment, this function is used only within
  // "#if SANITIZER_LINUX && defined(__x86_64__)" block in
  // sanitizer_stoptheworld_linux_libcdep.cc.
  return internal_syscall_ptr(SYSCALL(ptrace), request, pid, (uptr)addr,
                              (uptr)data);
#else
  return internal_syscall(SYSCALL(ptrace), request, pid, (uptr)addr,
                          (uptr)data);
#endif
}

uptr internal_waitpid(int pid, int *status, int options) {
  return internal_syscall_ptr(SYSCALL(wait4), pid, (uptr)status, options,
                          0 /* rusage */);
}

uptr internal_getpid() {
  return internal_syscall(SYSCALL(getpid));
}

uptr internal_getppid() {
  return internal_syscall(SYSCALL(getppid));
}

uptr internal_getdents(fd_t fd, struct linux_dirent *dirp, unsigned int count) {
#if SANITIZER_FREEBSD
  return internal_syscall(SYSCALL(getdirentries), fd, (uptr)dirp, count, NULL);
#elif SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(SYSCALL(getdents64), fd, (uptr)dirp, count);
#else
  return internal_syscall_ptr(SYSCALL(getdents), fd, (uptr)dirp, count);
#endif
}

uptr internal_lseek(fd_t fd, OFF_T offset, int whence) {
#if SANITIZER_NETBSD
  return internal_syscall64(SYSCALL(lseek), fd, 0, offset, whence);
#else
  return internal_syscall(SYSCALL(lseek), fd, offset, whence);
#endif
}

#if SANITIZER_LINUX
uptr internal_prctl(int option, uptr arg2, uptr arg3, uptr arg4, uptr arg5) {
  return internal_syscall(SYSCALL(prctl), option, arg2, arg3, arg4, arg5);
}
#endif

uptr internal_sigaltstack(const void *ss, void *oss) {
  return internal_syscall_ptr(SYSCALL(sigaltstack), (uptr)ss, (uptr)oss);
}

int internal_fork() {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(SYSCALL(clone), SIGCHLD, 0);
#else
  return internal_syscall(SYSCALL(fork));
#endif
}

#if SANITIZER_LINUX
#define SA_RESTORER 0x04000000
// Doesn't set sa_restorer if the caller did not set it, so use with caution
//(see below).
int internal_sigaction_norestorer(int signum, const void *act, void *oldact) {
  __sanitizer_kernel_sigaction_t k_act, k_oldact;
  internal_memset(&k_act, 0, sizeof(__sanitizer_kernel_sigaction_t));
  internal_memset(&k_oldact, 0, sizeof(__sanitizer_kernel_sigaction_t));
  const __sanitizer_sigaction *u_act = (const __sanitizer_sigaction *)act;
  __sanitizer_sigaction *u_oldact = (__sanitizer_sigaction *)oldact;
  if (u_act) {
    k_act.handler = u_act->handler;
    k_act.sigaction = u_act->sigaction;
    internal_memcpy(&k_act.sa_mask, &u_act->sa_mask,
                    sizeof(__sanitizer_kernel_sigset_t));
    // Without SA_RESTORER kernel ignores the calls (probably returns EINVAL).
    k_act.sa_flags = u_act->sa_flags | SA_RESTORER;
    // FIXME: most often sa_restorer is unset, however the kernel requires it
    // to point to a valid signal restorer that calls the rt_sigreturn syscall.
    // If sa_restorer passed to the kernel is NULL, the program may crash upon
    // signal delivery or fail to unwind the stack in the signal handler.
    // libc implementation of sigaction() passes its own restorer to
    // rt_sigaction, so we need to do the same (we'll need to reimplement the
    // restorers; for x86_64 the restorer address can be obtained from
    // oldact->sa_restorer upon a call to sigaction(xxx, NULL, oldact).
#if !SANITIZER_ANDROID || !SANITIZER_MIPS32
    k_act.sa_restorer = u_act->sa_restorer;
#endif
  }

  uptr result = internal_syscall(SYSCALL(rt_sigaction), (uptr)signum,
      (uptr)(u_act ? &k_act : nullptr),
      (uptr)(u_oldact ? &k_oldact : nullptr),
      (uptr)sizeof(__sanitizer_kernel_sigset_t));

  if ((result == 0) && u_oldact) {
    u_oldact->handler = k_oldact.handler;
    u_oldact->sigaction = k_oldact.sigaction;
    internal_memcpy(&u_oldact->sa_mask, &k_oldact.sa_mask,
                    sizeof(__sanitizer_kernel_sigset_t));
    u_oldact->sa_flags = k_oldact.sa_flags;
#if !SANITIZER_ANDROID || !SANITIZER_MIPS32
    u_oldact->sa_restorer = k_oldact.sa_restorer;
#endif
  }
  return result;
}

// Invokes sigaction via a raw syscall with a restorer, but does not support
// all platforms yet.
// We disable for Go simply because we have not yet added to buildgo.sh.
#if (defined(__x86_64__) || SANITIZER_MIPS64) && !SANITIZER_GO
int internal_sigaction_syscall(int signum, const void *act, void *oldact) {
  if (act == nullptr)
    return internal_sigaction_norestorer(signum, act, oldact);
  __sanitizer_sigaction u_adjust;
  internal_memcpy(&u_adjust, act, sizeof(u_adjust));
#if !SANITIZER_ANDROID || !SANITIZER_MIPS32
  if (u_adjust.sa_restorer == nullptr) {
    u_adjust.sa_restorer = internal_sigreturn;
  }
#endif
  return internal_sigaction_norestorer(signum, (const void *)&u_adjust, oldact);
}
#endif  // defined(__x86_64__) && !SANITIZER_GO
#endif  // SANITIZER_LINUX

uptr internal_sigprocmask(int how, __sanitizer_sigset_t *set,
                          __sanitizer_sigset_t *oldset) {
#if SANITIZER_FREEBSD || SANITIZER_NETBSD || SANITIZER_OPENBSD
  return internal_syscall_ptr(SYSCALL(sigprocmask), how, set, oldset);
#else
  __sanitizer_kernel_sigset_t *k_set = (__sanitizer_kernel_sigset_t *)set;
  __sanitizer_kernel_sigset_t *k_oldset = (__sanitizer_kernel_sigset_t *)oldset;
  return internal_syscall(SYSCALL(rt_sigprocmask), (uptr)how,
                          (uptr)&k_set->sig[0], (uptr)&k_oldset->sig[0],
                          sizeof(__sanitizer_kernel_sigset_t));
#endif
}

void internal_sigfillset(__sanitizer_sigset_t *set) {
  internal_memset(set, 0xff, sizeof(*set));
}

void internal_sigemptyset(__sanitizer_sigset_t *set) {
  internal_memset(set, 0, sizeof(*set));
}

#if SANITIZER_LINUX
void internal_sigdelset(__sanitizer_sigset_t *set, int signum) {
  signum -= 1;
  CHECK_GE(signum, 0);
  CHECK_LT(signum, sizeof(*set) * 8);
  __sanitizer_kernel_sigset_t *k_set = (__sanitizer_kernel_sigset_t *)set;
  const uptr idx = signum / (sizeof(k_set->sig[0]) * 8);
  const uptr bit = signum % (sizeof(k_set->sig[0]) * 8);
  k_set->sig[idx] &= ~(1 << bit);
}

bool internal_sigismember(__sanitizer_sigset_t *set, int signum) {
  signum -= 1;
  CHECK_GE(signum, 0);
  CHECK_LT(signum, sizeof(*set) * 8);
  __sanitizer_kernel_sigset_t *k_set = (__sanitizer_kernel_sigset_t *)set;
  const uptr idx = signum / (sizeof(k_set->sig[0]) * 8);
  const uptr bit = signum % (sizeof(k_set->sig[0]) * 8);
  return k_set->sig[idx] & (1 << bit);
}
#endif  // SANITIZER_LINUX
#endif // !SANITIZER_SOLARIS

// ThreadLister implementation.
ThreadLister::ThreadLister(int pid)
  : pid_(pid),
    descriptor_(-1),
    buffer_(4096),
    error_(true),
    entry_((struct linux_dirent *)buffer_.data()),
    bytes_read_(0) {
  char task_directory_path[80];
  internal_snprintf(task_directory_path, sizeof(task_directory_path),
                    "/proc/%d/task/", pid);
  uptr openrv = internal_open(task_directory_path, O_RDONLY | O_DIRECTORY);
  if (internal_iserror(openrv)) {
    error_ = true;
    Report("Can't open /proc/%d/task for reading.\n", pid);
  } else {
    error_ = false;
    descriptor_ = openrv;
  }
}

int ThreadLister::GetNextTID() {
  int tid = -1;
  do {
    if (error_)
      return -1;
    if ((char *)entry_ >= &buffer_[bytes_read_] && !GetDirectoryEntries())
      return -1;
    if (entry_->d_ino != 0 && entry_->d_name[0] >= '0' &&
        entry_->d_name[0] <= '9') {
      // Found a valid tid.
      tid = (int)internal_atoll(entry_->d_name);
    }
    entry_ = (struct linux_dirent *)(((char *)entry_) + entry_->d_reclen);
  } while (tid < 0);
  return tid;
}

void ThreadLister::Reset() {
  if (error_ || descriptor_ < 0)
    return;
  internal_lseek(descriptor_, 0, SEEK_SET);
}

ThreadLister::~ThreadLister() {
  if (descriptor_ >= 0)
    internal_close(descriptor_);
}

bool ThreadLister::error() { return error_; }

bool ThreadLister::GetDirectoryEntries() {
  CHECK_GE(descriptor_, 0);
  CHECK_NE(error_, true);
  bytes_read_ = internal_getdents(descriptor_,
                                  (struct linux_dirent *)buffer_.data(),
                                  buffer_.size());
  if (internal_iserror(bytes_read_)) {
    Report("Can't read directory entries from /proc/%d/task.\n", pid_);
    error_ = true;
    return false;
  } else if (bytes_read_ == 0) {
    return false;
  }
  entry_ = (struct linux_dirent *)buffer_.data();
  return true;
}

#if SANITIZER_WORDSIZE == 32
// Take care of unusable kernel area in top gigabyte.
static uptr GetKernelAreaSize() {
#if SANITIZER_LINUX && !SANITIZER_X32
  const uptr gbyte = 1UL << 30;

  // Firstly check if there are writable segments
  // mapped to top gigabyte (e.g. stack).
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  MemoryMappedSegment segment;
  while (proc_maps.Next(&segment)) {
    if ((segment.end >= 3 * gbyte) && segment.IsWritable()) return 0;
  }

#if !SANITIZER_ANDROID
  // Even if nothing is mapped, top Gb may still be accessible
  // if we are running on 64-bit kernel.
  // Uname may report misleading results if personality type
  // is modified (e.g. under schroot) so check this as well.
  struct utsname uname_info;
  int pers = personality(0xffffffffUL);
  if (!(pers & PER_MASK)
      && uname(&uname_info) == 0
      && internal_strstr(uname_info.machine, "64"))
    return 0;
#endif  // SANITIZER_ANDROID

  // Top gigabyte is reserved for kernel.
  return gbyte;
#else
  return 0;
#endif  // SANITIZER_LINUX && !SANITIZER_X32
}
#endif  // SANITIZER_WORDSIZE == 32

uptr GetMaxVirtualAddress() {
#if (SANITIZER_NETBSD || SANITIZER_OPENBSD) && defined(__x86_64__)
  return 0x7f7ffffff000ULL;  // (0x00007f8000000000 - PAGE_SIZE)
#elif SANITIZER_WORDSIZE == 64
# if defined(__powerpc64__) || defined(__aarch64__)
  // On PowerPC64 we have two different address space layouts: 44- and 46-bit.
  // We somehow need to figure out which one we are using now and choose
  // one of 0x00000fffffffffffUL and 0x00003fffffffffffUL.
  // Note that with 'ulimit -s unlimited' the stack is moved away from the top
  // of the address space, so simply checking the stack address is not enough.
  // This should (does) work for both PowerPC64 Endian modes.
  // Similarly, aarch64 has multiple address space layouts: 39, 42 and 47-bit.
  return (1ULL << (MostSignificantSetBitIndex(GET_CURRENT_FRAME()) + 1)) - 1;
# elif defined(__mips64)
  return (1ULL << 40) - 1;  // 0x000000ffffffffffUL;
# elif defined(__s390x__)
  return (1ULL << 53) - 1;  // 0x001fffffffffffffUL;
# else
  return (1ULL << 47) - 1;  // 0x00007fffffffffffUL;
# endif
#else  // SANITIZER_WORDSIZE == 32
# if defined(__s390__)
  return (1ULL << 31) - 1;  // 0x7fffffff;
# else
  return (1ULL << 32) - 1;  // 0xffffffff;
# endif
#endif  // SANITIZER_WORDSIZE
}

uptr GetMaxUserVirtualAddress() {
  uptr addr = GetMaxVirtualAddress();
#if SANITIZER_WORDSIZE == 32 && !defined(__s390__)
  if (!common_flags()->full_address_space)
    addr -= GetKernelAreaSize();
  CHECK_LT(reinterpret_cast<uptr>(&addr), addr);
#endif
  return addr;
}

uptr GetPageSize() {
// Android post-M sysconf(_SC_PAGESIZE) crashes if called from .preinit_array.
#if SANITIZER_ANDROID
  return 4096;
#elif SANITIZER_LINUX && (defined(__x86_64__) || defined(__i386__))
  return EXEC_PAGESIZE;
#elif SANITIZER_USE_GETAUXVAL
  return getauxval(AT_PAGESZ);
#else
  return sysconf(_SC_PAGESIZE);  // EXEC_PAGESIZE may not be trustworthy.
#endif
}

#if !SANITIZER_OPENBSD
uptr ReadBinaryName(/*out*/char *buf, uptr buf_len) {
#if SANITIZER_SOLARIS
  const char *default_module_name = getexecname();
  CHECK_NE(default_module_name, NULL);
  return internal_snprintf(buf, buf_len, "%s", default_module_name);
#else
#if SANITIZER_FREEBSD || SANITIZER_NETBSD
#if SANITIZER_FREEBSD
  const int Mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
#else
  const int Mib[4] = {CTL_KERN, KERN_PROC_ARGS, -1, KERN_PROC_PATHNAME};
#endif
  const char *default_module_name = "kern.proc.pathname";
  size_t Size = buf_len;
  bool IsErr = (sysctl(Mib, ARRAY_SIZE(Mib), buf, &Size, NULL, 0) != 0);
  int readlink_error = IsErr ? errno : 0;
  uptr module_name_len = Size;
#else
  const char *default_module_name = "/proc/self/exe";
  uptr module_name_len = internal_readlink(
      default_module_name, buf, buf_len);
  int readlink_error;
  bool IsErr = internal_iserror(module_name_len, &readlink_error);
#endif  // SANITIZER_SOLARIS
  if (IsErr) {
    // We can't read binary name for some reason, assume it's unknown.
    Report("WARNING: reading executable name failed with errno %d, "
           "some stack frames may not be symbolized\n", readlink_error);
    module_name_len = internal_snprintf(buf, buf_len, "%s",
                                        default_module_name);
    CHECK_LT(module_name_len, buf_len);
  }
  return module_name_len;
#endif
}
#endif // !SANITIZER_OPENBSD

uptr ReadLongProcessName(/*out*/ char *buf, uptr buf_len) {
#if SANITIZER_LINUX
  char *tmpbuf;
  uptr tmpsize;
  uptr tmplen;
  if (ReadFileToBuffer("/proc/self/cmdline", &tmpbuf, &tmpsize, &tmplen,
                       1024 * 1024)) {
    internal_strncpy(buf, tmpbuf, buf_len);
    UnmapOrDie(tmpbuf, tmpsize);
    return internal_strlen(buf);
  }
#endif
  return ReadBinaryName(buf, buf_len);
}

// Match full names of the form /path/to/base_name{-,.}*
bool LibraryNameIs(const char *full_name, const char *base_name) {
  const char *name = full_name;
  // Strip path.
  while (*name != '\0') name++;
  while (name > full_name && *name != '/') name--;
  if (*name == '/') name++;
  uptr base_name_length = internal_strlen(base_name);
  if (internal_strncmp(name, base_name, base_name_length)) return false;
  return (name[base_name_length] == '-' || name[base_name_length] == '.');
}

#if !SANITIZER_ANDROID
// Call cb for each region mapped by map.
void ForEachMappedRegion(link_map *map, void (*cb)(const void *, uptr)) {
  CHECK_NE(map, nullptr);
#if !SANITIZER_FREEBSD && !SANITIZER_OPENBSD
  typedef ElfW(Phdr) Elf_Phdr;
  typedef ElfW(Ehdr) Elf_Ehdr;
#endif // !SANITIZER_FREEBSD && !SANITIZER_OPENBSD
  char *base = (char *)map->l_addr;
  Elf_Ehdr *ehdr = (Elf_Ehdr *)base;
  char *phdrs = base + ehdr->e_phoff;
  char *phdrs_end = phdrs + ehdr->e_phnum * ehdr->e_phentsize;

  // Find the segment with the minimum base so we can "relocate" the p_vaddr
  // fields.  Typically ET_DYN objects (DSOs) have base of zero and ET_EXEC
  // objects have a non-zero base.
  uptr preferred_base = (uptr)-1;
  for (char *iter = phdrs; iter != phdrs_end; iter += ehdr->e_phentsize) {
    Elf_Phdr *phdr = (Elf_Phdr *)iter;
    if (phdr->p_type == PT_LOAD && preferred_base > (uptr)phdr->p_vaddr)
      preferred_base = (uptr)phdr->p_vaddr;
  }

  // Compute the delta from the real base to get a relocation delta.
  sptr delta = (uptr)base - preferred_base;
  // Now we can figure out what the loader really mapped.
  for (char *iter = phdrs; iter != phdrs_end; iter += ehdr->e_phentsize) {
    Elf_Phdr *phdr = (Elf_Phdr *)iter;
    if (phdr->p_type == PT_LOAD) {
      uptr seg_start = phdr->p_vaddr + delta;
      uptr seg_end = seg_start + phdr->p_memsz;
      // None of these values are aligned.  We consider the ragged edges of the
      // load command as defined, since they are mapped from the file.
      seg_start = RoundDownTo(seg_start, GetPageSizeCached());
      seg_end = RoundUpTo(seg_end, GetPageSizeCached());
      cb((void *)seg_start, seg_end - seg_start);
    }
  }
}
#endif

#if defined(__x86_64__) && SANITIZER_LINUX
// We cannot use glibc's clone wrapper, because it messes with the child
// task's TLS. It writes the PID and TID of the child task to its thread
// descriptor, but in our case the child task shares the thread descriptor with
// the parent (because we don't know how to allocate a new thread
// descriptor to keep glibc happy). So the stock version of clone(), when
// used with CLONE_VM, would end up corrupting the parent's thread descriptor.
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr) {
  long long res;
  if (!fn || !child_stack)
    return -EINVAL;
  CHECK_EQ(0, (uptr)child_stack % 16);
  child_stack = (char *)child_stack - 2 * sizeof(unsigned long long);
  ((unsigned long long *)child_stack)[0] = (uptr)fn;
  ((unsigned long long *)child_stack)[1] = (uptr)arg;
  register void *r8 __asm__("r8") = newtls;
  register int *r10 __asm__("r10") = child_tidptr;
  __asm__ __volatile__(
                       /* %rax = syscall(%rax = SYSCALL(clone),
                        *                %rdi = flags,
                        *                %rsi = child_stack,
                        *                %rdx = parent_tidptr,
                        *                %r8  = new_tls,
                        *                %r10 = child_tidptr)
                        */
                       "syscall\n"

                       /* if (%rax != 0)
                        *   return;
                        */
                       "testq  %%rax,%%rax\n"
                       "jnz    1f\n"

                       /* In the child. Terminate unwind chain. */
                       // XXX: We should also terminate the CFI unwind chain
                       // here. Unfortunately clang 3.2 doesn't support the
                       // necessary CFI directives, so we skip that part.
                       "xorq   %%rbp,%%rbp\n"

                       /* Call "fn(arg)". */
                       "popq   %%rax\n"
                       "popq   %%rdi\n"
                       "call   *%%rax\n"

                       /* Call _exit(%rax). */
                       "movq   %%rax,%%rdi\n"
                       "movq   %2,%%rax\n"
                       "syscall\n"

                       /* Return to parent. */
                     "1:\n"
                       : "=a" (res)
                       : "a"(SYSCALL(clone)), "i"(SYSCALL(exit)),
                         "S"(child_stack),
                         "D"(flags),
                         "d"(parent_tidptr),
                         "r"(r8),
                         "r"(r10)
                       : "rsp", "memory", "r11", "rcx");
  return res;
}
#elif defined(__mips__)
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr) {
  long long res;
  if (!fn || !child_stack)
    return -EINVAL;
  CHECK_EQ(0, (uptr)child_stack % 16);
  child_stack = (char *)child_stack - 2 * sizeof(unsigned long long);
  ((unsigned long long *)child_stack)[0] = (uptr)fn;
  ((unsigned long long *)child_stack)[1] = (uptr)arg;
  register void *a3 __asm__("$7") = newtls;
  register int *a4 __asm__("$8") = child_tidptr;
  // We don't have proper CFI directives here because it requires alot of code
  // for very marginal benefits.
  __asm__ __volatile__(
                       /* $v0 = syscall($v0 = __NR_clone,
                        * $a0 = flags,
                        * $a1 = child_stack,
                        * $a2 = parent_tidptr,
                        * $a3 = new_tls,
                        * $a4 = child_tidptr)
                        */
                       ".cprestore 16;\n"
                       "move $4,%1;\n"
                       "move $5,%2;\n"
                       "move $6,%3;\n"
                       "move $7,%4;\n"
                       /* Store the fifth argument on stack
                        * if we are using 32-bit abi.
                        */
#if SANITIZER_WORDSIZE == 32
                       "lw %5,16($29);\n"
#else
                       "move $8,%5;\n"
#endif
                       "li $2,%6;\n"
                       "syscall;\n"

                       /* if ($v0 != 0)
                        * return;
                        */
                       "bnez $2,1f;\n"

                       /* Call "fn(arg)". */
#if SANITIZER_WORDSIZE == 32
#ifdef __BIG_ENDIAN__
                       "lw $25,4($29);\n"
                       "lw $4,12($29);\n"
#else
                       "lw $25,0($29);\n"
                       "lw $4,8($29);\n"
#endif
#else
                       "ld $25,0($29);\n"
                       "ld $4,8($29);\n"
#endif
                       "jal $25;\n"

                       /* Call _exit($v0). */
                       "move $4,$2;\n"
                       "li $2,%7;\n"
                       "syscall;\n"

                       /* Return to parent. */
                     "1:\n"
                       : "=r" (res)
                       : "r"(flags),
                         "r"(child_stack),
                         "r"(parent_tidptr),
                         "r"(a3),
                         "r"(a4),
                         "i"(__NR_clone),
                         "i"(__NR_exit)
                       : "memory", "$29" );
  return res;
}
#elif defined(__aarch64__)
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr) {
  long long res;
  if (!fn || !child_stack)
    return -EINVAL;
  CHECK_EQ(0, (uptr)child_stack % 16);
  child_stack = (char *)child_stack - 2 * sizeof(unsigned long long);
  ((unsigned long long *)child_stack)[0] = (uptr)fn;
  ((unsigned long long *)child_stack)[1] = (uptr)arg;

  register int (*__fn)(void *)  __asm__("x0") = fn;
  register void *__stack __asm__("x1") = child_stack;
  register int   __flags __asm__("x2") = flags;
  register void *__arg   __asm__("x3") = arg;
  register int  *__ptid  __asm__("x4") = parent_tidptr;
  register void *__tls   __asm__("x5") = newtls;
  register int  *__ctid  __asm__("x6") = child_tidptr;

  __asm__ __volatile__(
                       "mov x0,x2\n" /* flags  */
                       "mov x2,x4\n" /* ptid  */
                       "mov x3,x5\n" /* tls  */
                       "mov x4,x6\n" /* ctid  */
                       "mov x8,%9\n" /* clone  */

                       "svc 0x0\n"

                       /* if (%r0 != 0)
                        *   return %r0;
                        */
                       "cmp x0, #0\n"
                       "bne 1f\n"

                       /* In the child, now. Call "fn(arg)". */
                       "ldp x1, x0, [sp], #16\n"
                       "blr x1\n"

                       /* Call _exit(%r0).  */
                       "mov x8, %10\n"
                       "svc 0x0\n"
                     "1:\n"

                       : "=r" (res)
                       : "i"(-EINVAL),
                         "r"(__fn), "r"(__stack), "r"(__flags), "r"(__arg),
                         "r"(__ptid), "r"(__tls), "r"(__ctid),
                         "i"(__NR_clone), "i"(__NR_exit)
                       : "x30", "memory");
  return res;
}
#elif defined(__powerpc64__)
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                   int *parent_tidptr, void *newtls, int *child_tidptr) {
  long long res;
// Stack frame structure.
#if SANITIZER_PPC64V1
//   Back chain == 0        (SP + 112)
// Frame (112 bytes):
//   Parameter save area    (SP + 48), 8 doublewords
//   TOC save area          (SP + 40)
//   Link editor doubleword (SP + 32)
//   Compiler doubleword    (SP + 24)
//   LR save area           (SP + 16)
//   CR save area           (SP + 8)
//   Back chain             (SP + 0)
# define FRAME_SIZE 112
# define FRAME_TOC_SAVE_OFFSET 40
#elif SANITIZER_PPC64V2
//   Back chain == 0        (SP + 32)
// Frame (32 bytes):
//   TOC save area          (SP + 24)
//   LR save area           (SP + 16)
//   CR save area           (SP + 8)
//   Back chain             (SP + 0)
# define FRAME_SIZE 32
# define FRAME_TOC_SAVE_OFFSET 24
#else
# error "Unsupported PPC64 ABI"
#endif
  if (!fn || !child_stack)
    return -EINVAL;
  CHECK_EQ(0, (uptr)child_stack % 16);

  register int (*__fn)(void *) __asm__("r3") = fn;
  register void *__cstack      __asm__("r4") = child_stack;
  register int __flags         __asm__("r5") = flags;
  register void *__arg         __asm__("r6") = arg;
  register int *__ptidptr      __asm__("r7") = parent_tidptr;
  register void *__newtls      __asm__("r8") = newtls;
  register int *__ctidptr      __asm__("r9") = child_tidptr;

 __asm__ __volatile__(
           /* fn and arg are saved across the syscall */
           "mr 28, %5\n\t"
           "mr 27, %8\n\t"

           /* syscall
             r0 == __NR_clone
             r3 == flags
             r4 == child_stack
             r5 == parent_tidptr
             r6 == newtls
             r7 == child_tidptr */
           "mr 3, %7\n\t"
           "mr 5, %9\n\t"
           "mr 6, %10\n\t"
           "mr 7, %11\n\t"
           "li 0, %3\n\t"
           "sc\n\t"

           /* Test if syscall was successful */
           "cmpdi  cr1, 3, 0\n\t"
           "crandc cr1*4+eq, cr1*4+eq, cr0*4+so\n\t"
           "bne-   cr1, 1f\n\t"

           /* Set up stack frame */
           "li    29, 0\n\t"
           "stdu  29, -8(1)\n\t"
           "stdu  1, -%12(1)\n\t"
           /* Do the function call */
           "std   2, %13(1)\n\t"
#if SANITIZER_PPC64V1
           "ld    0, 0(28)\n\t"
           "ld    2, 8(28)\n\t"
           "mtctr 0\n\t"
#elif SANITIZER_PPC64V2
           "mr    12, 28\n\t"
           "mtctr 12\n\t"
#else
# error "Unsupported PPC64 ABI"
#endif
           "mr    3, 27\n\t"
           "bctrl\n\t"
           "ld    2, %13(1)\n\t"

           /* Call _exit(r3) */
           "li 0, %4\n\t"
           "sc\n\t"

           /* Return to parent */
           "1:\n\t"
           "mr %0, 3\n\t"
             : "=r" (res)
             : "0" (-1),
               "i" (EINVAL),
               "i" (__NR_clone),
               "i" (__NR_exit),
               "r" (__fn),
               "r" (__cstack),
               "r" (__flags),
               "r" (__arg),
               "r" (__ptidptr),
               "r" (__newtls),
               "r" (__ctidptr),
               "i" (FRAME_SIZE),
               "i" (FRAME_TOC_SAVE_OFFSET)
             : "cr0", "cr1", "memory", "ctr", "r0", "r27", "r28", "r29");
  return res;
}
#elif defined(__i386__) && SANITIZER_LINUX
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr) {
  int res;
  if (!fn || !child_stack)
    return -EINVAL;
  CHECK_EQ(0, (uptr)child_stack % 16);
  child_stack = (char *)child_stack - 7 * sizeof(unsigned int);
  ((unsigned int *)child_stack)[0] = (uptr)flags;
  ((unsigned int *)child_stack)[1] = (uptr)0;
  ((unsigned int *)child_stack)[2] = (uptr)fn;
  ((unsigned int *)child_stack)[3] = (uptr)arg;
  __asm__ __volatile__(
                       /* %eax = syscall(%eax = SYSCALL(clone),
                        *                %ebx = flags,
                        *                %ecx = child_stack,
                        *                %edx = parent_tidptr,
                        *                %esi  = new_tls,
                        *                %edi = child_tidptr)
                        */

                        /* Obtain flags */
                        "movl    (%%ecx), %%ebx\n"
                        /* Do the system call */
                        "pushl   %%ebx\n"
                        "pushl   %%esi\n"
                        "pushl   %%edi\n"
                        /* Remember the flag value.  */
                        "movl    %%ebx, (%%ecx)\n"
                        "int     $0x80\n"
                        "popl    %%edi\n"
                        "popl    %%esi\n"
                        "popl    %%ebx\n"

                        /* if (%eax != 0)
                         *   return;
                         */

                        "test    %%eax,%%eax\n"
                        "jnz    1f\n"

                        /* terminate the stack frame */
                        "xorl   %%ebp,%%ebp\n"
                        /* Call FN. */
                        "call    *%%ebx\n"
#ifdef PIC
                        "call    here\n"
                        "here:\n"
                        "popl    %%ebx\n"
                        "addl    $_GLOBAL_OFFSET_TABLE_+[.-here], %%ebx\n"
#endif
                        /* Call exit */
                        "movl    %%eax, %%ebx\n"
                        "movl    %2, %%eax\n"
                        "int     $0x80\n"
                        "1:\n"
                       : "=a" (res)
                       : "a"(SYSCALL(clone)), "i"(SYSCALL(exit)),
                         "c"(child_stack),
                         "d"(parent_tidptr),
                         "S"(newtls),
                         "D"(child_tidptr)
                       : "memory");
  return res;
}
#elif defined(__arm__) && SANITIZER_LINUX
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr) {
  unsigned int res;
  if (!fn || !child_stack)
    return -EINVAL;
  child_stack = (char *)child_stack - 2 * sizeof(unsigned int);
  ((unsigned int *)child_stack)[0] = (uptr)fn;
  ((unsigned int *)child_stack)[1] = (uptr)arg;
  register int r0 __asm__("r0") = flags;
  register void *r1 __asm__("r1") = child_stack;
  register int *r2 __asm__("r2") = parent_tidptr;
  register void *r3 __asm__("r3") = newtls;
  register int *r4 __asm__("r4") = child_tidptr;
  register int r7 __asm__("r7") = __NR_clone;

#if __ARM_ARCH > 4 || defined (__ARM_ARCH_4T__)
# define ARCH_HAS_BX
#endif
#if __ARM_ARCH > 4
# define ARCH_HAS_BLX
#endif

#ifdef ARCH_HAS_BX
# ifdef ARCH_HAS_BLX
#  define BLX(R) "blx "  #R "\n"
# else
#  define BLX(R) "mov lr, pc; bx " #R "\n"
# endif
#else
# define BLX(R)  "mov lr, pc; mov pc," #R "\n"
#endif

  __asm__ __volatile__(
                       /* %r0 = syscall(%r7 = SYSCALL(clone),
                        *               %r0 = flags,
                        *               %r1 = child_stack,
                        *               %r2 = parent_tidptr,
                        *               %r3  = new_tls,
                        *               %r4 = child_tidptr)
                        */

                       /* Do the system call */
                       "swi 0x0\n"

                       /* if (%r0 != 0)
                        *   return %r0;
                        */
                       "cmp r0, #0\n"
                       "bne 1f\n"

                       /* In the child, now. Call "fn(arg)". */
                       "ldr r0, [sp, #4]\n"
                       "ldr ip, [sp], #8\n"
                       BLX(ip)
                       /* Call _exit(%r0). */
                       "mov r7, %7\n"
                       "swi 0x0\n"
                       "1:\n"
                       "mov %0, r0\n"
                       : "=r"(res)
                       : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r7),
                         "i"(__NR_exit)
                       : "memory");
  return res;
}
#endif  // defined(__x86_64__) && SANITIZER_LINUX

#if SANITIZER_ANDROID
#if __ANDROID_API__ < 21
extern "C" __attribute__((weak)) int dl_iterate_phdr(
    int (*)(struct dl_phdr_info *, size_t, void *), void *);
#endif

static int dl_iterate_phdr_test_cb(struct dl_phdr_info *info, size_t size,
                                   void *data) {
  // Any name starting with "lib" indicates a bug in L where library base names
  // are returned instead of paths.
  if (info->dlpi_name && info->dlpi_name[0] == 'l' &&
      info->dlpi_name[1] == 'i' && info->dlpi_name[2] == 'b') {
    *(bool *)data = true;
    return 1;
  }
  return 0;
}

static atomic_uint32_t android_api_level;

static AndroidApiLevel AndroidDetectApiLevel() {
  if (!&dl_iterate_phdr)
    return ANDROID_KITKAT; // K or lower
  bool base_name_seen = false;
  dl_iterate_phdr(dl_iterate_phdr_test_cb, &base_name_seen);
  if (base_name_seen)
    return ANDROID_LOLLIPOP_MR1; // L MR1
  return ANDROID_POST_LOLLIPOP;   // post-L
  // Plain L (API level 21) is completely broken wrt ASan and not very
  // interesting to detect.
}

AndroidApiLevel AndroidGetApiLevel() {
  AndroidApiLevel level =
      (AndroidApiLevel)atomic_load(&android_api_level, memory_order_relaxed);
  if (level) return level;
  level = AndroidDetectApiLevel();
  atomic_store(&android_api_level, level, memory_order_relaxed);
  return level;
}

#endif

static HandleSignalMode GetHandleSignalModeImpl(int signum) {
  switch (signum) {
    case SIGABRT:
      return common_flags()->handle_abort;
    case SIGILL:
      return common_flags()->handle_sigill;
    case SIGTRAP:
      return common_flags()->handle_sigtrap;
    case SIGFPE:
      return common_flags()->handle_sigfpe;
    case SIGSEGV:
      return common_flags()->handle_segv;
    case SIGBUS:
      return common_flags()->handle_sigbus;
  }
  return kHandleSignalNo;
}

HandleSignalMode GetHandleSignalMode(int signum) {
  HandleSignalMode result = GetHandleSignalModeImpl(signum);
  if (result == kHandleSignalYes && !common_flags()->allow_user_segv_handler)
    return kHandleSignalExclusive;
  return result;
}

#if !SANITIZER_GO
void *internal_start_thread(void(*func)(void *arg), void *arg) {
  // Start the thread with signals blocked, otherwise it can steal user signals.
  __sanitizer_sigset_t set, old;
  internal_sigfillset(&set);
#if SANITIZER_LINUX && !SANITIZER_ANDROID
  // Glibc uses SIGSETXID signal during setuid call. If this signal is blocked
  // on any thread, setuid call hangs (see test/tsan/setuid.c).
  internal_sigdelset(&set, 33);
#endif
  internal_sigprocmask(SIG_SETMASK, &set, &old);
  void *th;
  real_pthread_create(&th, nullptr, (void*(*)(void *arg))func, arg);
  internal_sigprocmask(SIG_SETMASK, &old, nullptr);
  return th;
}

void internal_join_thread(void *th) {
  real_pthread_join(th, nullptr);
}
#else
void *internal_start_thread(void (*func)(void *), void *arg) { return 0; }

void internal_join_thread(void *th) {}
#endif

#if defined(__aarch64__)
// Android headers in the older NDK releases miss this definition.
struct __sanitizer_esr_context {
  struct _aarch64_ctx head;
  uint64_t esr;
};

static bool Aarch64GetESR(ucontext_t *ucontext, u64 *esr) {
  static const u32 kEsrMagic = 0x45535201;
  u8 *aux = ucontext->uc_mcontext.__reserved;
  while (true) {
    _aarch64_ctx *ctx = (_aarch64_ctx *)aux;
    if (ctx->size == 0) break;
    if (ctx->magic == kEsrMagic) {
      *esr = ((__sanitizer_esr_context *)ctx)->esr;
      return true;
    }
    aux += ctx->size;
  }
  return false;
}
#endif

#if SANITIZER_OPENBSD
using Context = sigcontext;
#else
using Context = ucontext_t;
#endif

SignalContext::WriteFlag SignalContext::GetWriteFlag() const {
  Context *ucontext = (Context *)context;
#if defined(__x86_64__) || defined(__i386__)
  static const uptr PF_WRITE = 1U << 1;
#if SANITIZER_FREEBSD
  uptr err = ucontext->uc_mcontext.mc_err;
#elif SANITIZER_NETBSD
  uptr err = ucontext->uc_mcontext.__gregs[_REG_ERR];
#elif SANITIZER_OPENBSD
  uptr err = ucontext->sc_err;
#elif SANITIZER_SOLARIS && defined(__i386__)
  const int Err = 13;
  uptr err = ucontext->uc_mcontext.gregs[Err];
#else
  uptr err = ucontext->uc_mcontext.gregs[REG_ERR];
#endif // SANITIZER_FREEBSD
  return err & PF_WRITE ? WRITE : READ;
#elif defined(__mips__)
  uint32_t *exception_source;
  uint32_t faulty_instruction;
  uint32_t op_code;

  exception_source = (uint32_t *)ucontext->uc_mcontext.pc;
  faulty_instruction = (uint32_t)(*exception_source);

  op_code = (faulty_instruction >> 26) & 0x3f;

  // FIXME: Add support for FPU, microMIPS, DSP, MSA memory instructions.
  switch (op_code) {
    case 0x28:  // sb
    case 0x29:  // sh
    case 0x2b:  // sw
    case 0x3f:  // sd
#if __mips_isa_rev < 6
    case 0x2c:  // sdl
    case 0x2d:  // sdr
    case 0x2a:  // swl
    case 0x2e:  // swr
#endif
      return SignalContext::WRITE;

    case 0x20:  // lb
    case 0x24:  // lbu
    case 0x21:  // lh
    case 0x25:  // lhu
    case 0x23:  // lw
    case 0x27:  // lwu
    case 0x37:  // ld
#if __mips_isa_rev < 6
    case 0x1a:  // ldl
    case 0x1b:  // ldr
    case 0x22:  // lwl
    case 0x26:  // lwr
#endif
      return SignalContext::READ;
#if __mips_isa_rev == 6
    case 0x3b:  // pcrel
      op_code = (faulty_instruction >> 19) & 0x3;
      switch (op_code) {
        case 0x1:  // lwpc
        case 0x2:  // lwupc
          return SignalContext::READ;
      }
#endif
  }
  return SignalContext::UNKNOWN;
#elif defined(__arm__)
  static const uptr FSR_WRITE = 1U << 11;
  uptr fsr = ucontext->uc_mcontext.error_code;
  return fsr & FSR_WRITE ? WRITE : READ;
#elif defined(__aarch64__)
  static const u64 ESR_ELx_WNR = 1U << 6;
  u64 esr;
  if (!Aarch64GetESR(ucontext, &esr)) return UNKNOWN;
  return esr & ESR_ELx_WNR ? WRITE : READ;
#elif SANITIZER_SOLARIS && defined(__sparc__)
  // Decode the instruction to determine the access type.
  // From OpenSolaris $SRC/uts/sun4/os/trap.c (get_accesstype).
  uptr pc = ucontext->uc_mcontext.gregs[REG_PC];
  u32 instr = *(u32 *)pc;
  return (instr >> 21) & 1 ? WRITE: READ;
#else
  (void)ucontext;
  return UNKNOWN;  // FIXME: Implement.
#endif
}

void SignalContext::DumpAllRegisters(void *context) {
  // FIXME: Implement this.
}

static void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp) {
#if SANITIZER_NETBSD
  // This covers all NetBSD architectures
  ucontext_t *ucontext = (ucontext_t *)context;
  *pc = _UC_MACHINE_PC(ucontext);
  *bp = _UC_MACHINE_FP(ucontext);
  *sp = _UC_MACHINE_SP(ucontext);
#elif defined(__arm__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.arm_pc;
  *bp = ucontext->uc_mcontext.arm_fp;
  *sp = ucontext->uc_mcontext.arm_sp;
#elif defined(__aarch64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.pc;
  *bp = ucontext->uc_mcontext.regs[29];
  *sp = ucontext->uc_mcontext.sp;
#elif defined(__hppa__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.sc_iaoq[0];
  /* GCC uses %r3 whenever a frame pointer is needed.  */
  *bp = ucontext->uc_mcontext.sc_gr[3];
  *sp = ucontext->uc_mcontext.sc_gr[30];
#elif defined(__x86_64__)
# if SANITIZER_FREEBSD
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.mc_rip;
  *bp = ucontext->uc_mcontext.mc_rbp;
  *sp = ucontext->uc_mcontext.mc_rsp;
#elif SANITIZER_OPENBSD
  sigcontext *ucontext = (sigcontext *)context;
  *pc = ucontext->sc_rip;
  *bp = ucontext->sc_rbp;
  *sp = ucontext->sc_rsp;
# else
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_RIP];
  *bp = ucontext->uc_mcontext.gregs[REG_RBP];
  *sp = ucontext->uc_mcontext.gregs[REG_RSP];
# endif
#elif defined(__i386__)
# if SANITIZER_FREEBSD
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.mc_eip;
  *bp = ucontext->uc_mcontext.mc_ebp;
  *sp = ucontext->uc_mcontext.mc_esp;
#elif SANITIZER_OPENBSD
  sigcontext *ucontext = (sigcontext *)context;
  *pc = ucontext->sc_eip;
  *bp = ucontext->sc_ebp;
  *sp = ucontext->sc_esp;
# else
  ucontext_t *ucontext = (ucontext_t*)context;
# if SANITIZER_SOLARIS
  /* Use the numeric values: the symbolic ones are undefined by llvm
     include/llvm/Support/Solaris.h.  */
# ifndef REG_EIP
#  define REG_EIP 14 // REG_PC
# endif
# ifndef REG_EBP
#  define REG_EBP  6 // REG_FP
# endif
# ifndef REG_ESP
#  define REG_ESP 17 // REG_SP
# endif
# endif
  *pc = ucontext->uc_mcontext.gregs[REG_EIP];
  *bp = ucontext->uc_mcontext.gregs[REG_EBP];
  *sp = ucontext->uc_mcontext.gregs[REG_ESP];
# endif
#elif defined(__powerpc__) || defined(__powerpc64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.regs->nip;
  *sp = ucontext->uc_mcontext.regs->gpr[PT_R1];
  // The powerpc{,64}-linux ABIs do not specify r31 as the frame
  // pointer, but GCC always uses r31 when we need a frame pointer.
  *bp = ucontext->uc_mcontext.regs->gpr[PT_R31];
#elif defined(__sparc__)
  ucontext_t *ucontext = (ucontext_t*)context;
  uptr *stk_ptr;
# if defined (__sparcv9)
# ifndef MC_PC
#  define MC_PC REG_PC
# endif
# ifndef MC_O6
#  define MC_O6 REG_O6
# endif
# ifdef SANITIZER_SOLARIS
#  define mc_gregs gregs
# endif
  *pc = ucontext->uc_mcontext.mc_gregs[MC_PC];
  *sp = ucontext->uc_mcontext.mc_gregs[MC_O6];
  stk_ptr = (uptr *) (*sp + 2047);
  *bp = stk_ptr[15];
# else
  *pc = ucontext->uc_mcontext.gregs[REG_PC];
  *sp = ucontext->uc_mcontext.gregs[REG_O6];
  stk_ptr = (uptr *) *sp;
  *bp = stk_ptr[15];
# endif
#elif defined(__mips__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.pc;
  *bp = ucontext->uc_mcontext.gregs[30];
  *sp = ucontext->uc_mcontext.gregs[29];
#elif defined(__s390__)
  ucontext_t *ucontext = (ucontext_t*)context;
# if defined(__s390x__)
  *pc = ucontext->uc_mcontext.psw.addr;
# else
  *pc = ucontext->uc_mcontext.psw.addr & 0x7fffffff;
# endif
  *bp = ucontext->uc_mcontext.gregs[11];
  *sp = ucontext->uc_mcontext.gregs[15];
#else
# error "Unsupported arch"
#endif
}

void SignalContext::InitPcSpBp() { GetPcSpBp(context, &pc, &sp, &bp); }

void MaybeReexec() {
  // No need to re-exec on Linux.
}

void PrintModuleMap() { }

void CheckNoDeepBind(const char *filename, int flag) {
#ifdef RTLD_DEEPBIND
  if (flag & RTLD_DEEPBIND) {
    Report(
        "You are trying to dlopen a %s shared library with RTLD_DEEPBIND flag"
        " which is incompatibe with sanitizer runtime "
        "(see https://github.com/google/sanitizers/issues/611 for details"
        "). If you want to run %s library under sanitizers please remove "
        "RTLD_DEEPBIND from dlopen flags.\n",
        filename, filename);
    Die();
  }
#endif
}

uptr FindAvailableMemoryRange(uptr size, uptr alignment, uptr left_padding,
                              uptr *largest_gap_found,
                              uptr *max_occupied_addr) {
  UNREACHABLE("FindAvailableMemoryRange is not available");
  return 0;
}

bool GetRandom(void *buffer, uptr length, bool blocking) {
  if (!buffer || !length || length > 256)
    return false;
#if SANITIZER_USE_GETENTROPY
  uptr rnd = getentropy(buffer, length);
  int rverrno = 0;
  if (internal_iserror(rnd, &rverrno) && rverrno == EFAULT)
    return false;
  else if (rnd == 0)
    return true;
#endif // SANITIZER_USE_GETENTROPY

#if SANITIZER_USE_GETRANDOM
  static atomic_uint8_t skip_getrandom_syscall;
  if (!atomic_load_relaxed(&skip_getrandom_syscall)) {
    // Up to 256 bytes, getrandom will not be interrupted.
    uptr res = internal_syscall(SYSCALL(getrandom), buffer, length,
                                blocking ? 0 : GRND_NONBLOCK);
    int rverrno = 0;
    if (internal_iserror(res, &rverrno) && rverrno == ENOSYS)
      atomic_store_relaxed(&skip_getrandom_syscall, 1);
    else if (res == length)
      return true;
  }
#endif // SANITIZER_USE_GETRANDOM
  // Up to 256 bytes, a read off /dev/urandom will not be interrupted.
  // blocking is moot here, O_NONBLOCK has no effect when opening /dev/urandom.
  uptr fd = internal_open("/dev/urandom", O_RDONLY);
  if (internal_iserror(fd))
    return false;
  uptr res = internal_read(fd, buffer, length);
  if (internal_iserror(res))
    return false;
  internal_close(fd);
  return true;
}

} // namespace __sanitizer

#endif
//===-- sanitizer_linux_libcdep.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD ||                \
    SANITIZER_OPENBSD || SANITIZER_SOLARIS

#include "sanitizer_allocator_internal.h"
#include "sanitizer_atomic.h"
#include "sanitizer_common.h"
#include "sanitizer_file.h"
#include "sanitizer_flags.h"
#include "sanitizer_freebsd.h"
#include "sanitizer_linux.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"

#include <dlfcn.h>  // for dlsym()
#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <syslog.h>

#if SANITIZER_FREEBSD
#include <pthread_np.h>
#include <osreldate.h>
#include <sys/sysctl.h>
#define pthread_getattr_np pthread_attr_get_np
#endif

#if SANITIZER_OPENBSD
#include <pthread_np.h>
#include <sys/sysctl.h>
#endif

#if SANITIZER_NETBSD
#include <sys/sysctl.h>
#include <sys/tls.h>
#endif

#if SANITIZER_SOLARIS
#include <thread.h>
#endif

#if SANITIZER_ANDROID
#include <android/api-level.h>
#if !defined(CPU_COUNT) && !defined(__aarch64__)
#include <dirent.h>
#include <fcntl.h>
struct __sanitizer::linux_dirent {
  long           d_ino;
  off_t          d_off;
  unsigned short d_reclen;
  char           d_name[];
};
#endif
#endif

#if !SANITIZER_ANDROID
#include <elf.h>
#include <unistd.h>
#endif

namespace __sanitizer {

SANITIZER_WEAK_ATTRIBUTE int
real_sigaction(int signum, const void *act, void *oldact);

int internal_sigaction(int signum, const void *act, void *oldact) {
#if !SANITIZER_GO
  if (&real_sigaction)
    return real_sigaction(signum, act, oldact);
#endif
  return sigaction(signum, (const struct sigaction *)act,
                   (struct sigaction *)oldact);
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  if (at_initialization) {
    // This is the main thread. Libpthread may not be initialized yet.
    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);

    // Find the mapping that contains a stack variable.
    MemoryMappingLayout proc_maps(/*cache_enabled*/true);
    MemoryMappedSegment segment;
    uptr prev_end = 0;
    while (proc_maps.Next(&segment)) {
      if ((uptr)&rl < segment.end) break;
      prev_end = segment.end;
    }
    CHECK((uptr)&rl >= segment.start && (uptr)&rl < segment.end);

    // Get stacksize from rlimit, but clip it so that it does not overlap
    // with other mappings.
    uptr stacksize = rl.rlim_cur;
    if (stacksize > segment.end - prev_end) stacksize = segment.end - prev_end;
    // When running with unlimited stack size, we still want to set some limit.
    // The unlimited stack size is caused by 'ulimit -s unlimited'.
    // Also, for some reason, GNU make spawns subprocesses with unlimited stack.
    if (stacksize > kMaxThreadStackSize)
      stacksize = kMaxThreadStackSize;
    *stack_top = segment.end;
    *stack_bottom = segment.end - stacksize;
    return;
  }
  uptr stacksize = 0;
  void *stackaddr = nullptr;
#if SANITIZER_SOLARIS
  stack_t ss;
  CHECK_EQ(thr_stksegment(&ss), 0);
  stacksize = ss.ss_size;
  stackaddr = (char *)ss.ss_sp - stacksize;
#elif SANITIZER_OPENBSD
  stack_t sattr;
  CHECK_EQ(pthread_stackseg_np(pthread_self(), &sattr), 0);
  stackaddr = sattr.ss_sp;
  stacksize = sattr.ss_size;
#else  // !SANITIZER_SOLARIS
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  my_pthread_attr_getstack(&attr, &stackaddr, &stacksize);
  pthread_attr_destroy(&attr);
#endif // SANITIZER_SOLARIS

  *stack_top = (uptr)stackaddr + stacksize;
  *stack_bottom = (uptr)stackaddr;
}

#if !SANITIZER_GO
bool SetEnv(const char *name, const char *value) {
  void *f = dlsym(RTLD_NEXT, "setenv");
  if (!f)
    return false;
  typedef int(*setenv_ft)(const char *name, const char *value, int overwrite);
  setenv_ft setenv_f;
  CHECK_EQ(sizeof(setenv_f), sizeof(f));
  internal_memcpy(&setenv_f, &f, sizeof(f));
  return setenv_f(name, value, 1) == 0;
}
#endif

#if !SANITIZER_FREEBSD && !SANITIZER_ANDROID && !SANITIZER_GO &&               \
    !SANITIZER_NETBSD && !SANITIZER_OPENBSD && !SANITIZER_SOLARIS
static uptr g_tls_size;

#ifdef __i386__
# define DL_INTERNAL_FUNCTION __attribute__((regparm(3), stdcall))
#else
# define DL_INTERNAL_FUNCTION
#endif

void InitTlsSize() {
  // all current supported platforms have 16 bytes stack alignment
  const size_t kStackAlign = 16;
  typedef void (*get_tls_func)(size_t*, size_t*) DL_INTERNAL_FUNCTION;
  get_tls_func get_tls;
  void *get_tls_static_info_ptr = dlsym(RTLD_NEXT, "_dl_get_tls_static_info");
  CHECK_EQ(sizeof(get_tls), sizeof(get_tls_static_info_ptr));
  internal_memcpy(&get_tls, &get_tls_static_info_ptr,
                  sizeof(get_tls_static_info_ptr));
  CHECK_NE(get_tls, 0);
  size_t tls_size = 0;
  size_t tls_align = 0;
  get_tls(&tls_size, &tls_align);
  if (tls_align < kStackAlign)
    tls_align = kStackAlign;
  g_tls_size = RoundUpTo(tls_size, tls_align);
}
#else
void InitTlsSize() { }
#endif  // !SANITIZER_FREEBSD && !SANITIZER_ANDROID && !SANITIZER_GO &&
        // !SANITIZER_NETBSD && !SANITIZER_SOLARIS

#if (defined(__x86_64__) || defined(__i386__) || defined(__mips__) ||          \
     defined(__aarch64__) || defined(__powerpc64__) || defined(__s390__) ||    \
     defined(__arm__)) &&                                                      \
    SANITIZER_LINUX && !SANITIZER_ANDROID
// sizeof(struct pthread) from glibc.
static atomic_uintptr_t thread_descriptor_size;

uptr ThreadDescriptorSize() {
  uptr val = atomic_load_relaxed(&thread_descriptor_size);
  if (val)
    return val;
#if defined(__x86_64__) || defined(__i386__) || defined(__arm__)
#ifdef _CS_GNU_LIBC_VERSION
  char buf[64];
  uptr len = confstr(_CS_GNU_LIBC_VERSION, buf, sizeof(buf));
  if (len < sizeof(buf) && internal_strncmp(buf, "glibc 2.", 8) == 0) {
    char *end;
    int minor = internal_simple_strtoll(buf + 8, &end, 10);
    if (end != buf + 8 && (*end == '\0' || *end == '.' || *end == '-')) {
      int patch = 0;
      if (*end == '.')
        // strtoll will return 0 if no valid conversion could be performed
        patch = internal_simple_strtoll(end + 1, nullptr, 10);

      /* sizeof(struct pthread) values from various glibc versions.  */
      if (SANITIZER_X32)
        val = 1728;  // Assume only one particular version for x32.
      // For ARM sizeof(struct pthread) changed in Glibc 2.23.
      else if (SANITIZER_ARM)
        val = minor <= 22 ? 1120 : 1216;
      else if (minor <= 3)
        val = FIRST_32_SECOND_64(1104, 1696);
      else if (minor == 4)
        val = FIRST_32_SECOND_64(1120, 1728);
      else if (minor == 5)
        val = FIRST_32_SECOND_64(1136, 1728);
      else if (minor <= 9)
        val = FIRST_32_SECOND_64(1136, 1712);
      else if (minor == 10)
        val = FIRST_32_SECOND_64(1168, 1776);
      else if (minor == 11 || (minor == 12 && patch == 1))
        val = FIRST_32_SECOND_64(1168, 2288);
      else if (minor <= 13)
        val = FIRST_32_SECOND_64(1168, 2304);
      else
        val = FIRST_32_SECOND_64(1216, 2304);
    }
  }
#endif
#elif defined(__mips__)
  // TODO(sagarthakur): add more values as per different glibc versions.
  val = FIRST_32_SECOND_64(1152, 1776);
#elif defined(__aarch64__)
  // The sizeof (struct pthread) is the same from GLIBC 2.17 to 2.22.
  val = 1776;
#elif defined(__powerpc64__)
  val = 1776; // from glibc.ppc64le 2.20-8.fc21
#elif defined(__s390__)
  val = FIRST_32_SECOND_64(1152, 1776); // valid for glibc 2.22
#endif
  if (val)
    atomic_store_relaxed(&thread_descriptor_size, val);
  return val;
}

// The offset at which pointer to self is located in the thread descriptor.
const uptr kThreadSelfOffset = FIRST_32_SECOND_64(8, 16);

uptr ThreadSelfOffset() {
  return kThreadSelfOffset;
}

#if defined(__mips__) || defined(__powerpc64__)
// TlsPreTcbSize includes size of struct pthread_descr and size of tcb
// head structure. It lies before the static tls blocks.
static uptr TlsPreTcbSize() {
# if defined(__mips__)
  const uptr kTcbHead = 16; // sizeof (tcbhead_t)
# elif defined(__powerpc64__)
  const uptr kTcbHead = 88; // sizeof (tcbhead_t)
# endif
  const uptr kTlsAlign = 16;
  const uptr kTlsPreTcbSize =
      RoundUpTo(ThreadDescriptorSize() + kTcbHead, kTlsAlign);
  return kTlsPreTcbSize;
}
#endif

uptr ThreadSelf() {
  uptr descr_addr;
# if defined(__i386__)
  asm("mov %%gs:%c1,%0" : "=r"(descr_addr) : "i"(kThreadSelfOffset));
# elif defined(__x86_64__)
#if SANITIZER_RELACY_SCHEDULER
  asm("mov %%fs:%c1,%0" : "=r"(descr_addr) : "i"(0)); // kThreadSelfOffset seems are not valid. See ELF TLS documentation
#else
  asm("mov %%fs:%c1,%0" : "=r"(descr_addr) : "i"(kThreadSelfOffset));
#endif
# elif defined(__mips__)
  // MIPS uses TLS variant I. The thread pointer (in hardware register $29)
  // points to the end of the TCB + 0x7000. The pthread_descr structure is
  // immediately in front of the TCB. TlsPreTcbSize() includes the size of the
  // TCB and the size of pthread_descr.
  const uptr kTlsTcbOffset = 0x7000;
  uptr thread_pointer;
  asm volatile(".set push;\
                .set mips64r2;\
                rdhwr %0,$29;\
                .set pop" : "=r" (thread_pointer));
  descr_addr = thread_pointer - kTlsTcbOffset - TlsPreTcbSize();
# elif defined(__aarch64__) || defined(__arm__)
  descr_addr = reinterpret_cast<uptr>(__builtin_thread_pointer()) -
                                      ThreadDescriptorSize();
# elif defined(__s390__)
  descr_addr = reinterpret_cast<uptr>(__builtin_thread_pointer());
# elif defined(__powerpc64__)
  // PPC64LE uses TLS variant I. The thread pointer (in GPR 13)
  // points to the end of the TCB + 0x7000. The pthread_descr structure is
  // immediately in front of the TCB. TlsPreTcbSize() includes the size of the
  // TCB and the size of pthread_descr.
  const uptr kTlsTcbOffset = 0x7000;
  uptr thread_pointer;
  asm("addi %0,13,%1" : "=r"(thread_pointer) : "I"(-kTlsTcbOffset));
  descr_addr = thread_pointer - TlsPreTcbSize();
# else
#  error "unsupported CPU arch"
# endif
  return descr_addr;
}
#endif  // (x86_64 || i386 || MIPS) && SANITIZER_LINUX

#if SANITIZER_FREEBSD
static void **ThreadSelfSegbase() {
  void **segbase = 0;
# if defined(__i386__)
  // sysarch(I386_GET_GSBASE, segbase);
  __asm __volatile("mov %%gs:0, %0" : "=r" (segbase));
# elif defined(__x86_64__)
  // sysarch(AMD64_GET_FSBASE, segbase);
  __asm __volatile("movq %%fs:0, %0" : "=r" (segbase));
# else
#  error "unsupported CPU arch"
# endif
  return segbase;
}

uptr ThreadSelf() {
  return (uptr)ThreadSelfSegbase()[2];
}
#endif  // SANITIZER_FREEBSD

#if SANITIZER_NETBSD
static struct tls_tcb * ThreadSelfTlsTcb() {
  struct tls_tcb * tcb;
# ifdef __HAVE___LWP_GETTCB_FAST
  tcb = (struct tls_tcb *)__lwp_gettcb_fast();
# elif defined(__HAVE___LWP_GETPRIVATE_FAST)
  tcb = (struct tls_tcb *)__lwp_getprivate_fast();
# endif
  return tcb;
}

uptr ThreadSelf() {
  return (uptr)ThreadSelfTlsTcb()->tcb_pthread;
}

int GetSizeFromHdr(struct dl_phdr_info *info, size_t size, void *data) {
  const Elf_Phdr *hdr = info->dlpi_phdr;
  const Elf_Phdr *last_hdr = hdr + info->dlpi_phnum;

  for (; hdr != last_hdr; ++hdr) {
    if (hdr->p_type == PT_TLS && info->dlpi_tls_modid == 1) {
      *(uptr*)data = hdr->p_memsz;
      break;
    }
  }
  return 0;
}
#endif  // SANITIZER_NETBSD

#if !SANITIZER_GO
static void GetTls(uptr *addr, uptr *size) {
#if SANITIZER_LINUX && !SANITIZER_ANDROID
# if defined(__x86_64__) || defined(__i386__) || defined(__s390__)
  *addr = ThreadSelf();
  *size = GetTlsSize();
  *addr -= *size;
  *addr += ThreadDescriptorSize();
# elif defined(__mips__) || defined(__aarch64__) || defined(__powerpc64__) \
    || defined(__arm__)
  *addr = ThreadSelf();
  *size = GetTlsSize();
# else
  *addr = 0;
  *size = 0;
# endif
#elif SANITIZER_FREEBSD
  void** segbase = ThreadSelfSegbase();
  *addr = 0;
  *size = 0;
  if (segbase != 0) {
    // tcbalign = 16
    // tls_size = round(tls_static_space, tcbalign);
    // dtv = segbase[1];
    // dtv[2] = segbase - tls_static_space;
    void **dtv = (void**) segbase[1];
    *addr = (uptr) dtv[2];
    *size = (*addr == 0) ? 0 : ((uptr) segbase[0] - (uptr) dtv[2]);
  }
#elif SANITIZER_NETBSD
  struct tls_tcb * const tcb = ThreadSelfTlsTcb();
  *addr = 0;
  *size = 0;
  if (tcb != 0) {
    // Find size (p_memsz) of dlpi_tls_modid 1 (TLS block of the main program).
    // ld.elf_so hardcodes the index 1.
    dl_iterate_phdr(GetSizeFromHdr, size);

    if (*size != 0) {
      // The block has been found and tcb_dtv[1] contains the base address
      *addr = (uptr)tcb->tcb_dtv[1];
    }
  }
#elif SANITIZER_OPENBSD
  *addr = 0;
  *size = 0;
#elif SANITIZER_ANDROID
  *addr = 0;
  *size = 0;
#elif SANITIZER_SOLARIS
  // FIXME
  *addr = 0;
  *size = 0;
#else
# error "Unknown OS"
#endif
}
#endif

#if !SANITIZER_GO
uptr GetTlsSize() {
#if SANITIZER_FREEBSD || SANITIZER_ANDROID || SANITIZER_NETBSD ||              \
    SANITIZER_OPENBSD || SANITIZER_SOLARIS
  uptr addr, size;
  GetTls(&addr, &size);
  return size;
#elif defined(__mips__) || defined(__powerpc64__)
  return RoundUpTo(g_tls_size + TlsPreTcbSize(), 16);
#else
  return g_tls_size;
#endif
}
#endif

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#if SANITIZER_GO
  // Stub implementation for Go.
  *stk_addr = *stk_size = *tls_addr = *tls_size = 0;
#else
  GetTls(tls_addr, tls_size);

  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_addr = stack_bottom;
  *stk_size = stack_top - stack_bottom;

  if (!main) {
    // If stack and tls intersect, make them non-intersecting.
    if (*tls_addr > *stk_addr && *tls_addr < *stk_addr + *stk_size) {
      CHECK_GT(*tls_addr + *tls_size, *stk_addr);
      CHECK_LE(*tls_addr + *tls_size, *stk_addr + *stk_size);
      *stk_size -= *tls_size;
      *tls_addr = *stk_addr + *stk_size;
    }
  }
#endif
}

#if !SANITIZER_FREEBSD && !SANITIZER_OPENBSD
typedef ElfW(Phdr) Elf_Phdr;
#elif SANITIZER_WORDSIZE == 32 && __FreeBSD_version <= 902001 // v9.2
#define Elf_Phdr XElf32_Phdr
#define dl_phdr_info xdl_phdr_info
#define dl_iterate_phdr(c, b) xdl_iterate_phdr((c), (b))
#endif // !SANITIZER_FREEBSD && !SANITIZER_OPENBSD

struct DlIteratePhdrData {
  InternalMmapVectorNoCtor<LoadedModule> *modules;
  bool first;
};

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrData *data = (DlIteratePhdrData*)arg;
  InternalScopedString module_name(kMaxPathLength);
  if (data->first) {
    data->first = false;
    // First module is the binary itself.
    ReadBinaryNameCached(module_name.data(), module_name.size());
  } else if (info->dlpi_name) {
    module_name.append("%s", info->dlpi_name);
  }
  if (module_name[0] == '\0')
    return 0;
  LoadedModule cur_module;
  cur_module.set(module_name.data(), info->dlpi_addr);
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const Elf_Phdr *phdr = &info->dlpi_phdr[i];
    if (phdr->p_type == PT_LOAD) {
      uptr cur_beg = info->dlpi_addr + phdr->p_vaddr;
      uptr cur_end = cur_beg + phdr->p_memsz;
      bool executable = phdr->p_flags & PF_X;
      bool writable = phdr->p_flags & PF_W;
      cur_module.addAddressRange(cur_beg, cur_end, executable,
                                 writable);
    }
  }
  data->modules->push_back(cur_module);
  return 0;
}

#if SANITIZER_ANDROID && __ANDROID_API__ < 21
extern "C" __attribute__((weak)) int dl_iterate_phdr(
    int (*)(struct dl_phdr_info *, size_t, void *), void *);
#endif

static bool requiresProcmaps() {
#if SANITIZER_ANDROID && __ANDROID_API__ <= 22
  // Fall back to /proc/maps if dl_iterate_phdr is unavailable or broken.
  // The runtime check allows the same library to work with
  // both K and L (and future) Android releases.
  return AndroidGetApiLevel() <= ANDROID_LOLLIPOP_MR1;
#else
  return false;
#endif
}

static void procmapsInit(InternalMmapVectorNoCtor<LoadedModule> *modules) {
  MemoryMappingLayout memory_mapping(/*cache_enabled*/true);
  memory_mapping.DumpListOfModules(modules);
}

void ListOfModules::init() {
  clearOrInit();
  if (requiresProcmaps()) {
    procmapsInit(&modules_);
  } else {
    DlIteratePhdrData data = {&modules_, true};
    dl_iterate_phdr(dl_iterate_phdr_cb, &data);
  }
}

// When a custom loader is used, dl_iterate_phdr may not contain the full
// list of modules. Allow callers to fall back to using procmaps.
void ListOfModules::fallbackInit() {
  if (!requiresProcmaps()) {
    clearOrInit();
    procmapsInit(&modules_);
  } else {
    clear();
  }
}

// getrusage does not give us the current RSS, only the max RSS.
// Still, this is better than nothing if /proc/self/statm is not available
// for some reason, e.g. due to a sandbox.
static uptr GetRSSFromGetrusage() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage))  // Failed, probably due to a sandbox.
    return 0;
  return usage.ru_maxrss << 10;  // ru_maxrss is in Kb.
}

uptr GetRSS() {
  if (!common_flags()->can_use_proc_maps_statm)
    return GetRSSFromGetrusage();
  fd_t fd = OpenFile("/proc/self/statm", RdOnly);
  if (fd == kInvalidFd)
    return GetRSSFromGetrusage();
  char buf[64];
  uptr len = internal_read(fd, buf, sizeof(buf) - 1);
  internal_close(fd);
  if ((sptr)len <= 0)
    return 0;
  buf[len] = 0;
  // The format of the file is:
  // 1084 89 69 11 0 79 0
  // We need the second number which is RSS in pages.
  char *pos = buf;
  // Skip the first number.
  while (*pos >= '0' && *pos <= '9')
    pos++;
  // Skip whitespaces.
  while (!(*pos >= '0' && *pos <= '9') && *pos != 0)
    pos++;
  // Read the number.
  uptr rss = 0;
  while (*pos >= '0' && *pos <= '9')
    rss = rss * 10 + *pos++ - '0';
  return rss * GetPageSizeCached();
}

// sysconf(_SC_NPROCESSORS_{CONF,ONLN}) cannot be used on most platforms as
// they allocate memory.
u32 GetNumberOfCPUs() {
#if SANITIZER_FREEBSD || SANITIZER_NETBSD || SANITIZER_OPENBSD
  u32 ncpu;
  int req[2];
  size_t len = sizeof(ncpu);
  req[0] = CTL_HW;
  req[1] = HW_NCPU;
  CHECK_EQ(sysctl(req, 2, &ncpu, &len, NULL, 0), 0);
  return ncpu;
#elif SANITIZER_ANDROID && !defined(CPU_COUNT) && !defined(__aarch64__)
  // Fall back to /sys/devices/system/cpu on Android when cpu_set_t doesn't
  // exist in sched.h. That is the case for toolchains generated with older
  // NDKs.
  // This code doesn't work on AArch64 because internal_getdents makes use of
  // the 64bit getdents syscall, but cpu_set_t seems to always exist on AArch64.
  uptr fd = internal_open("/sys/devices/system/cpu", O_RDONLY | O_DIRECTORY);
  if (internal_iserror(fd))
    return 0;
  InternalScopedBuffer<u8> buffer(4096);
  uptr bytes_read = buffer.size();
  uptr n_cpus = 0;
  u8 *d_type;
  struct linux_dirent *entry = (struct linux_dirent *)&buffer[bytes_read];
  while (true) {
    if ((u8 *)entry >= &buffer[bytes_read]) {
      bytes_read = internal_getdents(fd, (struct linux_dirent *)buffer.data(),
                                     buffer.size());
      if (internal_iserror(bytes_read) || !bytes_read)
        break;
      entry = (struct linux_dirent *)buffer.data();
    }
    d_type = (u8 *)entry + entry->d_reclen - 1;
    if (d_type >= &buffer[bytes_read] ||
        (u8 *)&entry->d_name[3] >= &buffer[bytes_read])
      break;
    if (entry->d_ino != 0 && *d_type == DT_DIR) {
      if (entry->d_name[0] == 'c' && entry->d_name[1] == 'p' &&
          entry->d_name[2] == 'u' &&
          entry->d_name[3] >= '0' && entry->d_name[3] <= '9')
        n_cpus++;
    }
    entry = (struct linux_dirent *)(((u8 *)entry) + entry->d_reclen);
  }
  internal_close(fd);
  return n_cpus;
#elif SANITIZER_SOLARIS
  return sysconf(_SC_NPROCESSORS_ONLN);
#else
  cpu_set_t CPUs;
  CHECK_EQ(sched_getaffinity(0, sizeof(cpu_set_t), &CPUs), 0);
  return CPU_COUNT(&CPUs);
#endif
}

#if SANITIZER_LINUX

# if SANITIZER_ANDROID
static atomic_uint8_t android_log_initialized;

void AndroidLogInit() {
  openlog(GetProcessName(), 0, LOG_USER);
  atomic_store(&android_log_initialized, 1, memory_order_release);
}

static bool ShouldLogAfterPrintf() {
  return atomic_load(&android_log_initialized, memory_order_acquire);
}

extern "C" SANITIZER_WEAK_ATTRIBUTE
int async_safe_write_log(int pri, const char* tag, const char* msg);
extern "C" SANITIZER_WEAK_ATTRIBUTE
int __android_log_write(int prio, const char* tag, const char* msg);

// ANDROID_LOG_INFO is 4, but can't be resolved at runtime.
#define SANITIZER_ANDROID_LOG_INFO 4

// async_safe_write_log is a new public version of __libc_write_log that is
// used behind syslog. It is preferable to syslog as it will not do any dynamic
// memory allocation or formatting.
// If the function is not available, syslog is preferred for L+ (it was broken
// pre-L) as __android_log_write triggers a racey behavior with the strncpy
// interceptor. Fallback to __android_log_write pre-L.
void WriteOneLineToSyslog(const char *s) {
  if (&async_safe_write_log) {
    async_safe_write_log(SANITIZER_ANDROID_LOG_INFO, GetProcessName(), s);
  } else if (AndroidGetApiLevel() > ANDROID_KITKAT) {
    syslog(LOG_INFO, "%s", s);
  } else {
    CHECK(&__android_log_write);
    __android_log_write(SANITIZER_ANDROID_LOG_INFO, nullptr, s);
  }
}

extern "C" SANITIZER_WEAK_ATTRIBUTE
void android_set_abort_message(const char *);

void SetAbortMessage(const char *str) {
  if (&android_set_abort_message)
    android_set_abort_message(str);
}
# else
void AndroidLogInit() {}

static bool ShouldLogAfterPrintf() { return true; }

void WriteOneLineToSyslog(const char *s) { syslog(LOG_INFO, "%s", s); }

void SetAbortMessage(const char *str) {}
# endif  // SANITIZER_ANDROID

void LogMessageOnPrintf(const char *str) {
  if (common_flags()->log_to_syslog && ShouldLogAfterPrintf())
    WriteToSyslog(str);
}

#endif  // SANITIZER_LINUX

#if SANITIZER_LINUX && !SANITIZER_GO
// glibc crashes when using clock_gettime from a preinit_array function as the
// vDSO function pointers haven't been initialized yet. __progname is
// initialized after the vDSO function pointers, so if it exists, is not null
// and is not empty, we can use clock_gettime.
extern "C" SANITIZER_WEAK_ATTRIBUTE char *__progname;
INLINE bool CanUseVDSO() {
  // Bionic is safe, it checks for the vDSO function pointers to be initialized.
  if (SANITIZER_ANDROID)
    return true;
  if (&__progname && __progname && *__progname)
    return true;
  return false;
}

// MonotonicNanoTime is a timing function that can leverage the vDSO by calling
// clock_gettime. real_clock_gettime only exists if clock_gettime is
// intercepted, so define it weakly and use it if available.
extern "C" SANITIZER_WEAK_ATTRIBUTE
int real_clock_gettime(u32 clk_id, void *tp);
u64 MonotonicNanoTime() {
  timespec ts;
  if (CanUseVDSO()) {
    if (&real_clock_gettime)
      real_clock_gettime(CLOCK_MONOTONIC, &ts);
    else
      clock_gettime(CLOCK_MONOTONIC, &ts);
  } else {
    internal_clock_gettime(CLOCK_MONOTONIC, &ts);
  }
  return (u64)ts.tv_sec * (1000ULL * 1000 * 1000) + ts.tv_nsec;
}
#else
// Non-Linux & Go always use the syscall.
u64 MonotonicNanoTime() {
  timespec ts;
  internal_clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * (1000ULL * 1000 * 1000) + ts.tv_nsec;
}
#endif  // SANITIZER_LINUX && !SANITIZER_GO

} // namespace __sanitizer

#endif
//===-- sanitizer_stoptheworld_linux_libcdep.cc ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// See sanitizer_stoptheworld.h for details.
// This implementation was inspired by Markus Gutschke's linuxthreads.cc.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_LINUX && (defined(__x86_64__) || defined(__mips__) || \
                        defined(__aarch64__) || defined(__powerpc64__) || \
                        defined(__s390__) || defined(__i386__) || \
                        defined(__arm__))

#include "sanitizer_stoptheworld.h"

#include "sanitizer_platform_limits_posix.h"
#include "sanitizer_atomic.h"

#include <errno.h>
#include <sched.h> // for CLONE_* definitions
#include <stddef.h>
#include <sys/prctl.h> // for PR_* definitions
#include <sys/ptrace.h> // for PTRACE_* definitions
#include <sys/types.h> // for pid_t
#include <sys/uio.h> // for iovec
#include <elf.h> // for NT_PRSTATUS
#if defined(__aarch64__) && !SANITIZER_ANDROID
// GLIBC 2.20+ sys/user does not include asm/ptrace.h
# include <asm/ptrace.h>
#endif
#include <sys/user.h>  // for user_regs_struct
#if SANITIZER_ANDROID && SANITIZER_MIPS
# include <asm/reg.h>  // for mips SP register in sys/user.h
#endif
#include <sys/wait.h> // for signal-related stuff

#ifdef sa_handler
# undef sa_handler
#endif

#ifdef sa_sigaction
# undef sa_sigaction
#endif

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_libc.h"
#include "sanitizer_linux.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"

// Sufficiently old kernel headers don't provide this value, but we can still
// call prctl with it. If the runtime kernel is new enough, the prctl call will
// have the desired effect; if the kernel is too old, the call will error and we
// can ignore said error.
#ifndef PR_SET_PTRACER
#define PR_SET_PTRACER 0x59616d61
#endif

// This module works by spawning a Linux task which then attaches to every
// thread in the caller process with ptrace. This suspends the threads, and
// PTRACE_GETREGS can then be used to obtain their register state. The callback
// supplied to StopTheWorld() is run in the tracer task while the threads are
// suspended.
// The tracer task must be placed in a different thread group for ptrace to
// work, so it cannot be spawned as a pthread. Instead, we use the low-level
// clone() interface (we want to share the address space with the caller
// process, so we prefer clone() over fork()).
//
// We don't use any libc functions, relying instead on direct syscalls. There
// are two reasons for this:
// 1. calling a library function while threads are suspended could cause a
// deadlock, if one of the treads happens to be holding a libc lock;
// 2. it's generally not safe to call libc functions from the tracer task,
// because clone() does not set up a thread-local storage for it. Any
// thread-local variables used by libc will be shared between the tracer task
// and the thread which spawned it.

namespace __sanitizer {

class SuspendedThreadsListLinux : public SuspendedThreadsList {
 public:
  SuspendedThreadsListLinux() : thread_ids_(1024) {}

  tid_t GetThreadID(uptr index) const;
  uptr ThreadCount() const;
  bool ContainsTid(tid_t thread_id) const;
  void Append(tid_t tid);

  PtraceRegistersStatus GetRegistersAndSP(uptr index, uptr *buffer,
                                          uptr *sp) const;
  uptr RegisterCount() const;

 private:
  InternalMmapVector<tid_t> thread_ids_;
};

// Structure for passing arguments into the tracer thread.
struct TracerThreadArgument {
  StopTheWorldCallback callback;
  void *callback_argument;
  // The tracer thread waits on this mutex while the parent finishes its
  // preparations.
  BlockingMutex mutex;
  // Tracer thread signals its completion by setting done.
  atomic_uintptr_t done;
  uptr parent_pid;
};

// This class handles thread suspending/unsuspending in the tracer thread.
class ThreadSuspender {
 public:
  explicit ThreadSuspender(pid_t pid, TracerThreadArgument *arg)
    : arg(arg)
    , pid_(pid) {
      CHECK_GE(pid, 0);
    }
  bool SuspendAllThreads();
  void ResumeAllThreads();
  void KillAllThreads();
  SuspendedThreadsListLinux &suspended_threads_list() {
    return suspended_threads_list_;
  }
  TracerThreadArgument *arg;
 private:
  SuspendedThreadsListLinux suspended_threads_list_;
  pid_t pid_;
  bool SuspendThread(tid_t thread_id);
};

bool ThreadSuspender::SuspendThread(tid_t tid) {
  // Are we already attached to this thread?
  // Currently this check takes linear time, however the number of threads is
  // usually small.
  if (suspended_threads_list_.ContainsTid(tid)) return false;
  int pterrno;
  if (internal_iserror(internal_ptrace(PTRACE_ATTACH, tid, nullptr, nullptr),
                       &pterrno)) {
    // Either the thread is dead, or something prevented us from attaching.
    // Log this event and move on.
    VReport(1, "Could not attach to thread %zu (errno %d).\n", (uptr)tid,
            pterrno);
    return false;
  } else {
    VReport(2, "Attached to thread %zu.\n", (uptr)tid);
    // The thread is not guaranteed to stop before ptrace returns, so we must
    // wait on it. Note: if the thread receives a signal concurrently,
    // we can get notification about the signal before notification about stop.
    // In such case we need to forward the signal to the thread, otherwise
    // the signal will be missed (as we do PTRACE_DETACH with arg=0) and
    // any logic relying on signals will break. After forwarding we need to
    // continue to wait for stopping, because the thread is not stopped yet.
    // We do ignore delivery of SIGSTOP, because we want to make stop-the-world
    // as invisible as possible.
    for (;;) {
      int status;
      uptr waitpid_status;
      HANDLE_EINTR(waitpid_status, internal_waitpid(tid, &status, __WALL));
      int wperrno;
      if (internal_iserror(waitpid_status, &wperrno)) {
        // Got a ECHILD error. I don't think this situation is possible, but it
        // doesn't hurt to report it.
        VReport(1, "Waiting on thread %zu failed, detaching (errno %d).\n",
                (uptr)tid, wperrno);
        internal_ptrace(PTRACE_DETACH, tid, nullptr, nullptr);
        return false;
      }
      if (WIFSTOPPED(status) && WSTOPSIG(status) != SIGSTOP) {
        internal_ptrace(PTRACE_CONT, tid, nullptr,
                        (void*)(uptr)WSTOPSIG(status));
        continue;
      }
      break;
    }
    suspended_threads_list_.Append(tid);
    return true;
  }
}

void ThreadSuspender::ResumeAllThreads() {
  for (uptr i = 0; i < suspended_threads_list_.ThreadCount(); i++) {
    pid_t tid = suspended_threads_list_.GetThreadID(i);
    int pterrno;
    if (!internal_iserror(internal_ptrace(PTRACE_DETACH, tid, nullptr, nullptr),
                          &pterrno)) {
      VReport(2, "Detached from thread %d.\n", tid);
    } else {
      // Either the thread is dead, or we are already detached.
      // The latter case is possible, for instance, if this function was called
      // from a signal handler.
      VReport(1, "Could not detach from thread %d (errno %d).\n", tid, pterrno);
    }
  }
}

void ThreadSuspender::KillAllThreads() {
  for (uptr i = 0; i < suspended_threads_list_.ThreadCount(); i++)
    internal_ptrace(PTRACE_KILL, suspended_threads_list_.GetThreadID(i),
                    nullptr, nullptr);
}

bool ThreadSuspender::SuspendAllThreads() {
  ThreadLister thread_lister(pid_);
  bool added_threads;
  bool first_iteration = true;
  do {
    // Run through the directory entries once.
    added_threads = false;
    pid_t tid = thread_lister.GetNextTID();
    while (tid >= 0) {
      if (SuspendThread(tid))
        added_threads = true;
      tid = thread_lister.GetNextTID();
    }
    if (thread_lister.error() || (first_iteration && !added_threads)) {
      // Detach threads and fail.
      ResumeAllThreads();
      return false;
    }
    thread_lister.Reset();
    first_iteration = false;
  } while (added_threads);
  return true;
}

// Pointer to the ThreadSuspender instance for use in signal handler.
static ThreadSuspender *thread_suspender_instance = nullptr;

// Synchronous signals that should not be blocked.
static const int kSyncSignals[] = { SIGABRT, SIGILL, SIGFPE, SIGSEGV, SIGBUS,
                                    SIGXCPU, SIGXFSZ };

static void TracerThreadDieCallback() {
  // Generally a call to Die() in the tracer thread should be fatal to the
  // parent process as well, because they share the address space.
  // This really only works correctly if all the threads are suspended at this
  // point. So we correctly handle calls to Die() from within the callback, but
  // not those that happen before or after the callback. Hopefully there aren't
  // a lot of opportunities for that to happen...
  ThreadSuspender *inst = thread_suspender_instance;
  if (inst && stoptheworld_tracer_pid == internal_getpid()) {
    inst->KillAllThreads();
    thread_suspender_instance = nullptr;
  }
}

// Signal handler to wake up suspended threads when the tracer thread dies.
static void TracerThreadSignalHandler(int signum, __sanitizer_siginfo *siginfo,
                                      void *uctx) {
  SignalContext ctx(siginfo, uctx);
  Printf("Tracer caught signal %d: addr=0x%zx pc=0x%zx sp=0x%zx\n", signum,
         ctx.addr, ctx.pc, ctx.sp);
  ThreadSuspender *inst = thread_suspender_instance;
  if (inst) {
    if (signum == SIGABRT)
      inst->KillAllThreads();
    else
      inst->ResumeAllThreads();
    RAW_CHECK(RemoveDieCallback(TracerThreadDieCallback));
    thread_suspender_instance = nullptr;
    atomic_store(&inst->arg->done, 1, memory_order_relaxed);
  }
  internal__exit((signum == SIGABRT) ? 1 : 2);
}

// Size of alternative stack for signal handlers in the tracer thread.
static const int kHandlerStackSize = 8192;

// This function will be run as a cloned task.
static int TracerThread(void* argument) {
  TracerThreadArgument *tracer_thread_argument =
      (TracerThreadArgument *)argument;

  internal_prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0);
  // Check if parent is already dead.
  if (internal_getppid() != tracer_thread_argument->parent_pid)
    internal__exit(4);

  // Wait for the parent thread to finish preparations.
  tracer_thread_argument->mutex.Lock();
  tracer_thread_argument->mutex.Unlock();

  RAW_CHECK(AddDieCallback(TracerThreadDieCallback));

  ThreadSuspender thread_suspender(internal_getppid(), tracer_thread_argument);
  // Global pointer for the signal handler.
  thread_suspender_instance = &thread_suspender;

  // Alternate stack for signal handling.
  InternalScopedBuffer<char> handler_stack_memory(kHandlerStackSize);
  stack_t handler_stack;
  internal_memset(&handler_stack, 0, sizeof(handler_stack));
  handler_stack.ss_sp = handler_stack_memory.data();
  handler_stack.ss_size = kHandlerStackSize;
  internal_sigaltstack(&handler_stack, nullptr);

  // Install our handler for synchronous signals. Other signals should be
  // blocked by the mask we inherited from the parent thread.
  for (uptr i = 0; i < ARRAY_SIZE(kSyncSignals); i++) {
    __sanitizer_sigaction act;
    internal_memset(&act, 0, sizeof(act));
    act.sigaction = TracerThreadSignalHandler;
    act.sa_flags = SA_ONSTACK | SA_SIGINFO;
    internal_sigaction_norestorer(kSyncSignals[i], &act, 0);
  }

  int exit_code = 0;
  if (!thread_suspender.SuspendAllThreads()) {
    VReport(1, "Failed suspending threads.\n");
    exit_code = 3;
  } else {
    tracer_thread_argument->callback(thread_suspender.suspended_threads_list(),
                                     tracer_thread_argument->callback_argument);
    thread_suspender.ResumeAllThreads();
    exit_code = 0;
  }
  RAW_CHECK(RemoveDieCallback(TracerThreadDieCallback));
  thread_suspender_instance = nullptr;
  atomic_store(&tracer_thread_argument->done, 1, memory_order_relaxed);
  return exit_code;
}

class ScopedStackSpaceWithGuard {
 public:
  explicit ScopedStackSpaceWithGuard(uptr stack_size) {
    stack_size_ = stack_size;
    guard_size_ = GetPageSizeCached();
    // FIXME: Omitting MAP_STACK here works in current kernels but might break
    // in the future.
    guard_start_ = (uptr)MmapOrDie(stack_size_ + guard_size_,
                                   "ScopedStackWithGuard");
    CHECK(MprotectNoAccess((uptr)guard_start_, guard_size_));
  }
  ~ScopedStackSpaceWithGuard() {
    UnmapOrDie((void *)guard_start_, stack_size_ + guard_size_);
  }
  void *Bottom() const {
    return (void *)(guard_start_ + stack_size_ + guard_size_);
  }

 private:
  uptr stack_size_;
  uptr guard_size_;
  uptr guard_start_;
};

// We have a limitation on the stack frame size, so some stuff had to be moved
// into globals.
static __sanitizer_sigset_t blocked_sigset;
static __sanitizer_sigset_t old_sigset;

class StopTheWorldScope {
 public:
  StopTheWorldScope() {
    // Make this process dumpable. Processes that are not dumpable cannot be
    // attached to.
    process_was_dumpable_ = internal_prctl(PR_GET_DUMPABLE, 0, 0, 0, 0);
    if (!process_was_dumpable_)
      internal_prctl(PR_SET_DUMPABLE, 1, 0, 0, 0);
  }

  ~StopTheWorldScope() {
    // Restore the dumpable flag.
    if (!process_was_dumpable_)
      internal_prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
  }

 private:
  int process_was_dumpable_;
};

// When sanitizer output is being redirected to file (i.e. by using log_path),
// the tracer should write to the parent's log instead of trying to open a new
// file. Alert the logging code to the fact that we have a tracer.
struct ScopedSetTracerPID {
  explicit ScopedSetTracerPID(uptr tracer_pid) {
    stoptheworld_tracer_pid = tracer_pid;
    stoptheworld_tracer_ppid = internal_getpid();
  }
  ~ScopedSetTracerPID() {
    stoptheworld_tracer_pid = 0;
    stoptheworld_tracer_ppid = 0;
  }
};

void StopTheWorld(StopTheWorldCallback callback, void *argument) {
  StopTheWorldScope in_stoptheworld;
  // Prepare the arguments for TracerThread.
  struct TracerThreadArgument tracer_thread_argument;
  tracer_thread_argument.callback = callback;
  tracer_thread_argument.callback_argument = argument;
  tracer_thread_argument.parent_pid = internal_getpid();
  atomic_store(&tracer_thread_argument.done, 0, memory_order_relaxed);
  const uptr kTracerStackSize = 2 * 1024 * 1024;
  ScopedStackSpaceWithGuard tracer_stack(kTracerStackSize);
  // Block the execution of TracerThread until after we have set ptrace
  // permissions.
  tracer_thread_argument.mutex.Lock();
  // Signal handling story.
  // We don't want async signals to be delivered to the tracer thread,
  // so we block all async signals before creating the thread. An async signal
  // handler can temporary modify errno, which is shared with this thread.
  // We ought to use pthread_sigmask here, because sigprocmask has undefined
  // behavior in multithreaded programs. However, on linux sigprocmask is
  // equivalent to pthread_sigmask with the exception that pthread_sigmask
  // does not allow to block some signals used internally in pthread
  // implementation. We are fine with blocking them here, we are really not
  // going to pthread_cancel the thread.
  // The tracer thread should not raise any synchronous signals. But in case it
  // does, we setup a special handler for sync signals that properly kills the
  // parent as well. Note: we don't pass CLONE_SIGHAND to clone, so handlers
  // in the tracer thread won't interfere with user program. Double note: if a
  // user does something along the lines of 'kill -11 pid', that can kill the
  // process even if user setup own handler for SEGV.
  // Thing to watch out for: this code should not change behavior of user code
  // in any observable way. In particular it should not override user signal
  // handlers.
  internal_sigfillset(&blocked_sigset);
  for (uptr i = 0; i < ARRAY_SIZE(kSyncSignals); i++)
    internal_sigdelset(&blocked_sigset, kSyncSignals[i]);
  int rv = internal_sigprocmask(SIG_BLOCK, &blocked_sigset, &old_sigset);
  CHECK_EQ(rv, 0);
  uptr tracer_pid = internal_clone(
      TracerThread, tracer_stack.Bottom(),
      CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_UNTRACED,
      &tracer_thread_argument, nullptr /* parent_tidptr */,
      nullptr /* newtls */, nullptr /* child_tidptr */);
  internal_sigprocmask(SIG_SETMASK, &old_sigset, 0);
  int local_errno = 0;
  if (internal_iserror(tracer_pid, &local_errno)) {
    VReport(1, "Failed spawning a tracer thread (errno %d).\n", local_errno);
    tracer_thread_argument.mutex.Unlock();
  } else {
    ScopedSetTracerPID scoped_set_tracer_pid(tracer_pid);
    // On some systems we have to explicitly declare that we want to be traced
    // by the tracer thread.
    internal_prctl(PR_SET_PTRACER, tracer_pid, 0, 0, 0);
    // Allow the tracer thread to start.
    tracer_thread_argument.mutex.Unlock();
    // NOTE: errno is shared between this thread and the tracer thread.
    // internal_waitpid() may call syscall() which can access/spoil errno,
    // so we can't call it now. Instead we for the tracer thread to finish using
    // the spin loop below. Man page for sched_yield() says "In the Linux
    // implementation, sched_yield() always succeeds", so let's hope it does not
    // spoil errno. Note that this spin loop runs only for brief periods before
    // the tracer thread has suspended us and when it starts unblocking threads.
    while (atomic_load(&tracer_thread_argument.done, memory_order_relaxed) == 0)
      sched_yield();
    // Now the tracer thread is about to exit and does not touch errno,
    // wait for it.
    for (;;) {
      uptr waitpid_status = internal_waitpid(tracer_pid, nullptr, __WALL);
      if (!internal_iserror(waitpid_status, &local_errno))
        break;
      if (local_errno == EINTR)
        continue;
      VReport(1, "Waiting on the tracer thread failed (errno %d).\n",
              local_errno);
      break;
    }
  }
}

// Platform-specific methods from SuspendedThreadsList.
#if SANITIZER_ANDROID && defined(__arm__)
typedef pt_regs regs_struct;
#define REG_SP ARM_sp

#elif SANITIZER_LINUX && defined(__arm__)
typedef user_regs regs_struct;
#define REG_SP uregs[13]

#elif defined(__i386__) || defined(__x86_64__)
typedef user_regs_struct regs_struct;
#if defined(__i386__)
#define REG_SP esp
#else
#define REG_SP rsp
#endif

#elif defined(__powerpc__) || defined(__powerpc64__)
typedef pt_regs regs_struct;
#define REG_SP gpr[PT_R1]

#elif defined(__mips__)
typedef struct user regs_struct;
# if SANITIZER_ANDROID
#  define REG_SP regs[EF_R29]
# else
#  define REG_SP regs[EF_REG29]
# endif

#elif defined(__aarch64__)
typedef struct user_pt_regs regs_struct;
#define REG_SP sp
#define ARCH_IOVEC_FOR_GETREGSET

#elif defined(__s390__)
typedef _user_regs_struct regs_struct;
#define REG_SP gprs[15]
#define ARCH_IOVEC_FOR_GETREGSET

#else
#error "Unsupported architecture"
#endif // SANITIZER_ANDROID && defined(__arm__)

tid_t SuspendedThreadsListLinux::GetThreadID(uptr index) const {
  CHECK_LT(index, thread_ids_.size());
  return thread_ids_[index];
}

uptr SuspendedThreadsListLinux::ThreadCount() const {
  return thread_ids_.size();
}

bool SuspendedThreadsListLinux::ContainsTid(tid_t thread_id) const {
  for (uptr i = 0; i < thread_ids_.size(); i++) {
    if (thread_ids_[i] == thread_id) return true;
  }
  return false;
}

void SuspendedThreadsListLinux::Append(tid_t tid) {
  thread_ids_.push_back(tid);
}

PtraceRegistersStatus SuspendedThreadsListLinux::GetRegistersAndSP(
    uptr index, uptr *buffer, uptr *sp) const {
  pid_t tid = GetThreadID(index);
  regs_struct regs;
  int pterrno;
#ifdef ARCH_IOVEC_FOR_GETREGSET
  struct iovec regset_io;
  regset_io.iov_base = &regs;
  regset_io.iov_len = sizeof(regs_struct);
  bool isErr = internal_iserror(internal_ptrace(PTRACE_GETREGSET, tid,
                                (void*)NT_PRSTATUS, (void*)&regset_io),
                                &pterrno);
#else
  bool isErr = internal_iserror(internal_ptrace(PTRACE_GETREGS, tid, nullptr,
                                &regs), &pterrno);
#endif
  if (isErr) {
    VReport(1, "Could not get registers from thread %d (errno %d).\n", tid,
            pterrno);
    // ESRCH means that the given thread is not suspended or already dead.
    // Therefore it's unsafe to inspect its data (e.g. walk through stack) and
    // we should notify caller about this.
    return pterrno == ESRCH ? REGISTERS_UNAVAILABLE_FATAL
                            : REGISTERS_UNAVAILABLE;
  }

  *sp = regs.REG_SP;
  internal_memcpy(buffer, &regs, sizeof(regs));
  return REGISTERS_AVAILABLE;
}

uptr SuspendedThreadsListLinux::RegisterCount() const {
  return sizeof(regs_struct) / sizeof(uptr);
}
} // namespace __sanitizer

#endif  // SANITIZER_LINUX && (defined(__x86_64__) || defined(__mips__)
        // || defined(__aarch64__) || defined(__powerpc64__)
        // || defined(__s390__) || defined(__i386__) || defined(__arm__)
