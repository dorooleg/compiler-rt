//===-- asan_win.cc -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Windows-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_WINDOWS
#include <windows.h>

#include <dbghelp.h>
#include <stdlib.h>

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_report.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"

extern "C" {
  SANITIZER_INTERFACE_ATTRIBUTE
  int __asan_should_detect_stack_use_after_return() {
    __asan_init();
    return __asan_option_detect_stack_use_after_return;
  }
}

namespace __asan {

// ---------------------- TSD ---------------- {{{1
static bool tsd_key_inited = false;

static __declspec(thread) void *fake_tsd = 0;

void AsanTSDInit(void (*destructor)(void *tsd)) {
  // FIXME: we're ignoring the destructor for now.
  tsd_key_inited = true;
}

void *AsanTSDGet() {
  CHECK(tsd_key_inited);
  return fake_tsd;
}

void AsanTSDSet(void *tsd) {
  CHECK(tsd_key_inited);
  fake_tsd = tsd;
}

void PlatformTSDDtor(void *tsd) {
  AsanThread::TSDDtor(tsd);
}
// ---------------------- Various stuff ---------------- {{{1
void MaybeReexec() {
  // No need to re-exec on Windows.
}

void *AsanDoesNotSupportStaticLinkage() {
#if defined(_DEBUG)
#error Please build the runtime with a non-debug CRT: /MD or /MT
#endif
  return 0;
}

void AsanCheckDynamicRTPrereqs() {}

void AsanCheckIncompatibleRT() {}

void AsanPlatformThreadInit() {
  // Nothing here for now.
}

void ReadContextStack(void *context, uptr *stack, uptr *ssize) {
  UNIMPLEMENTED();
}

void AsanOnSIGSEGV(int, void *siginfo, void *context) {
  UNIMPLEMENTED();
}

static LPTOP_LEVEL_EXCEPTION_FILTER default_seh_handler;

long WINAPI SEHHandler(EXCEPTION_POINTERS *info) {
  EXCEPTION_RECORD *exception_record = info->ExceptionRecord;
  CONTEXT *context = info->ContextRecord;
  uptr pc = (uptr)exception_record->ExceptionAddress;
#ifdef _WIN64
  uptr bp = (uptr)context->Rbp, sp = (uptr)context->Rsp;
#else
  uptr bp = (uptr)context->Ebp, sp = (uptr)context->Esp;
#endif

  if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION ||
      exception_record->ExceptionCode == EXCEPTION_IN_PAGE_ERROR) {
    const char *description =
        (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION)
            ? "access-violation"
            : "in-page-error";
    uptr access_addr = exception_record->ExceptionInformation[1];
    ReportSIGSEGV(description, pc, sp, bp, context, access_addr);
  }

  // FIXME: Handle EXCEPTION_STACK_OVERFLOW here.

  return default_seh_handler(info);
}

int SetSEHFilter() {
  default_seh_handler = SetUnhandledExceptionFilter(SEHHandler);
  return 0;
}

// Put a pointer to SetSEHFilter at the end of the global list
// of C initializers, after the default handler is set by the CRT.
// See crt0dat.c in the CRT sources for the details.
#pragma section(".CRT$XIZ", long, read)  // NOLINT
__declspec(allocate(".CRT$XIZ")) int (*__intercept_seh)() = SetSEHFilter;

}  // namespace __asan

#endif  // _WIN32
