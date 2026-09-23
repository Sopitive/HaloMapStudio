// =============================================================================
// hms_windows_shim.h - minimal Win32 surface for the NON-Windows build of
// HaloMapStudioDLL, implemented over POSIX.
//
// The point of this file is that the ~53k lines of shared parser source stay
// BYTE-IDENTICAL across platforms: on Windows they include the real <windows.h>
// and this file is never compiled; on Linux the build adds `native/compat` to
// the include path so `#include <windows.h>` lands on our shim instead.
//
// Only the surface the OFFLINE PARSERS actually use is provided. The live-game
// hooking translation units (Forge*, FramePumpHook, *Snapshot) are excluded
// from the Linux build entirely, so their much larger Win32 surface is absent
// by design rather than stubbed.
// =============================================================================
#pragma once
#ifdef _WIN32
#error "hms_windows_shim.h is for non-Windows builds only"
#endif

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cstdarg>
#include <cstdlib>

// ---- calling-convention / export decorations -------------------------------
// x86-64 has a single calling convention, so __stdcall is already a no-op on
// Win64; dllexport becomes ELF default visibility.
#define __stdcall
#define WINAPI
#define APIENTRY
#define CALLBACK
#ifndef __declspec
#define __declspec(kind) HMS_DECLSPEC_##kind
#endif
#define HMS_DECLSPEC_dllexport __attribute__((visibility("default")))
#define HMS_DECLSPEC_dllimport
#define HMS_DECLSPEC_noinline  __attribute__((noinline))

// ---- fundamental types -----------------------------------------------------
typedef void*          HANDLE;
typedef void*          HMODULE;
typedef void*          HINSTANCE;
typedef void*          LPVOID;
typedef const void*    LPCVOID;
typedef uint32_t       DWORD;
typedef uint32_t       ULONG;
typedef uint16_t       WORD;
typedef uint8_t        BYTE;
typedef int            BOOL;
typedef int            INT;
typedef unsigned int   UINT;
typedef long long      LONGLONG;
typedef unsigned long long ULONGLONG;
typedef size_t         SIZE_T;
typedef wchar_t        WCHAR;
typedef wchar_t*       LPWSTR;
typedef const wchar_t* LPCWSTR;
typedef char*          LPSTR;
typedef const char*    LPCSTR;
typedef DWORD*         LPDWORD;

typedef union _LARGE_INTEGER {
    struct { DWORD LowPart; int32_t HighPart; };
    LONGLONG QuadPart;
} LARGE_INTEGER;

typedef struct _SYSTEMTIME {
    WORD wYear, wMonth, wDayOfWeek, wDay, wHour, wMinute, wSecond, wMilliseconds;
} SYSTEMTIME;

// ---- constants -------------------------------------------------------------
#define MAX_PATH 260
#define TRUE  1
#define FALSE 0
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define GENERIC_READ           0x80000000u
#define GENERIC_WRITE          0x40000000u
#define FILE_APPEND_DATA       0x00000004u
#define FILE_SHARE_READ        0x00000001u
#define FILE_SHARE_WRITE       0x00000002u
#define FILE_SHARE_DELETE      0x00000004u
#define CREATE_ALWAYS          2
#define OPEN_EXISTING          3
#define OPEN_ALWAYS            4
#define FILE_ATTRIBUTE_NORMAL  0x00000080u
#define PAGE_READONLY          0x02u
#define PAGE_READWRITE         0x04u
#define FILE_MAP_READ          0x0004u
#define FILE_MAP_ALL_ACCESS    0x000Fu
#define GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS 0x00000004u
#define ERROR_SUCCESS          0u
#define _TRUNCATE ((size_t)-1)

// ---- file / mapping API (implemented in hms_win_compat.cpp) ----------------
extern "C" {
HANDLE  CreateFileW(LPCWSTR path, DWORD access, DWORD share, void* sa,
                    DWORD disposition, DWORD flags, HANDLE tmpl);
HANDLE  CreateFileMappingW(HANDLE file, void* sa, DWORD protect,
                           DWORD maxHigh, DWORD maxLow, LPCWSTR name);
LPVOID  MapViewOfFile(HANDLE mapping, DWORD access, DWORD offHigh,
                      DWORD offLow, SIZE_T bytes);
BOOL    UnmapViewOfFile(LPCVOID base);
BOOL    CloseHandle(HANDLE h);
BOOL    GetFileSizeEx(HANDLE h, LARGE_INTEGER* size);
BOOL    WriteFile(HANDLE h, LPCVOID buf, DWORD len, LPDWORD written, void* ov);
DWORD   GetLastError(void);
void    SetLastError(DWORD e);
BOOL    GetModuleHandleExW(DWORD flags, LPCWSTR name, HMODULE* out);
DWORD   GetModuleFileNameW(HMODULE mod, LPWSTR buf, DWORD cch);
HMODULE LoadLibraryW(LPCWSTR path);
void*   GetProcAddress(HMODULE mod, LPCSTR name);
BOOL    FreeLibrary(HMODULE mod);
void    OutputDebugStringA(LPCSTR s);
void    OutputDebugStringW(LPCWSTR s);
void    GetLocalTime(SYSTEMTIME* st);
ULONGLONG GetTickCount64(void);
void    Sleep(DWORD ms);
}

// ---- MSVC secure-CRT shims -------------------------------------------------
// Variadic macros keep the call sites identical to the MSVC originals.
#define _snprintf_s(buf, cap, trunc, ...) snprintf((buf), (cap), __VA_ARGS__)
#define sprintf_s(buf, cap, ...)          snprintf((buf), (cap), __VA_ARGS__)
#define _vsnprintf_s(b, c, t, f, a)       vsnprintf((b), (c), (f), (a))
#define strncpy_s(d, cap, s, n)           ((void)snprintf((d), (cap), "%s", (s)))
#define strcpy_s(d, cap, s)               ((void)snprintf((d), (cap), "%s", (s)))
#define _stricmp   strcasecmp
#define _strnicmp  strncasecmp
#define _wcsicmp   wcscasecmp
#define __debugbreak() ((void)0)

template <size_t N> inline void wcsncat_s(wchar_t (&d)[N], const wchar_t* s, size_t) {
    const size_t n = wcslen(d); if (n + 1 < N) wcsncat(d, s, N - n - 1);
}
static inline void wcscpy_s(wchar_t* dst, size_t cap, const wchar_t* src) {
    wcsncpy(dst, src, cap); if (cap) dst[cap - 1] = 0;
}

// ---- extra calling conventions --------------------------------------------
#define __cdecl
#define __fastcall
#define __vectorcall

// ---- Structured Exception Handling ----------------------------------------
// MSVC SEH has no GCC/Clang equivalent. The parsers use __try/__except purely
// as a guard around decompressing untrusted map bytes, so we map it onto C++
// EH: the control flow and the "ok = 0" recovery path are preserved.
// CAVEAT: on Windows this also catches access violations; on Linux a genuine
// SIGSEGV is NOT catchable this way. A corrupt/hostile .map that would have
// been contained on Windows can therefore still fault the process here.
#define __try try
#define __except(filter) catch (...)
#define __finally
#define EXCEPTION_ACCESS_VIOLATION  0xC0000005u
#define EXCEPTION_EXECUTE_HANDLER   1
#define EXCEPTION_CONTINUE_SEARCH   0
static inline DWORD GetExceptionCode(void) { return EXCEPTION_ACCESS_VIOLATION; }

// ---- remaining CRT / Win32 odds and ends ----------------------------------
#include <climits>
#include <strings.h>

// MSVC ships both an explicit-capacity form and an array template that deduces
// it; the parsers use the array form (wcsncpy_s(buf, src, _TRUNCATE)).
static inline void wcsncpy_s(wchar_t* d, size_t cap, const wchar_t* s, size_t) {
    if (!cap) return; wcsncpy(d, s, cap); d[cap - 1] = 0;
}
template <size_t N> inline void wcsncpy_s(wchar_t (&d)[N], const wchar_t* s, size_t) {
    wcsncpy(d, s, N); d[N - 1] = 0;
}
static inline void wcscat_s(wchar_t* d, size_t cap, const wchar_t* s) {
    size_t n = wcslen(d); if (n + 1 < cap) wcsncat(d, s, cap - n - 1);
}
static inline size_t strnlen_s(const char* s, size_t n) { return s ? strnlen(s, n) : 0; }
// _heapmin was a no-op on Linux, so the app's periodic + post-load heap trims (ZH_TrimHeaps)
// never returned the parallel decode's freed pages to the OS. glibc's malloc_trim(0) walks every
// arena (the 30 decode threads each own one) and MADV_DONTNEEDs free chunks -- the exact analogue
// of the UCRT _heapmin the Windows build relies on. Other libcs (musl) have no such call: no-op.
#if defined(__GLIBC__)
#include <malloc.h>
static inline void   _heapmin(void) { malloc_trim(0); }
#else
static inline void   _heapmin(void) {}
#endif

extern "C" {
DWORD GetEnvironmentVariableA(LPCSTR name, LPSTR buf, DWORD cch);
BOOL  QueryPerformanceCounter(LARGE_INTEGER* v);
BOOL  QueryPerformanceFrequency(LARGE_INTEGER* v);
int   MultiByteToWideChar(UINT cp, DWORD flags, LPCSTR mb, int cb, LPWSTR wc, int cch);
int   WideCharToMultiByte(UINT cp, DWORD flags, LPCWSTR wc, int cch, LPSTR mb, int cb,
                          LPCSTR def, BOOL* used);
}
#define CP_UTF8 65001u
#define CP_ACP  0u

// ---- final CRT gaps --------------------------------------------------------
#include <cmath>     // std::isfinite &c. (MSVC pulls these in via <math.h>)

// MSVC array-template form: strncpy_s(buf, src, _TRUNCATE)
#undef strncpy_s
static inline void strncpy_s(char* d, size_t cap, const char* s, size_t) {
    if (!cap) return; snprintf(d, cap, "%s", s);
}
template <size_t N> inline void strncpy_s(char (&d)[N], const char* s, size_t) {
    snprintf(d, N, "%s", s);
}

// _dupenv_s: MSVC's allocating getenv. Returns 0 on success (value may be null).
static inline int _dupenv_s(char** out, size_t* len, const char* name) {
    if (!out) return 22;                       // EINVAL
    *out = nullptr; if (len) *len = 0;
    const char* v = name ? getenv(name) : nullptr;
    if (!v) return 0;                          // absent is success, value null
    const size_t n = strlen(v);
    char* buf = (char*)malloc(n + 1);
    if (!buf) return 12;                       // ENOMEM
    memcpy(buf, v, n + 1);
    *out = buf; if (len) *len = n + 1;
    return 0;
}

// glibc's <cmath> pulls the C `isfinite` macro out of the global namespace,
// but MSVC leaves it visible there; the parsers call it unqualified.
#ifndef isfinite
using std::isfinite;
using std::isnan;
using std::isinf;
#endif

// ---- critical sections -> pthread recursive mutex --------------------------
#include <pthread.h>
typedef struct _CRITICAL_SECTION { pthread_mutex_t m; int initialized; } CRITICAL_SECTION;
typedef CRITICAL_SECTION* LPCRITICAL_SECTION;
static inline void InitializeCriticalSection(LPCRITICAL_SECTION cs) {
    pthread_mutexattr_t a; pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);   // Win32 CS is recursive
    pthread_mutex_init(&cs->m, &a); pthread_mutexattr_destroy(&a);
    cs->initialized = 1;
}
static inline void EnterCriticalSection(LPCRITICAL_SECTION cs) {
    if (!cs->initialized) InitializeCriticalSection(cs);
    pthread_mutex_lock(&cs->m);
}
static inline void LeaveCriticalSection(LPCRITICAL_SECTION cs) { pthread_mutex_unlock(&cs->m); }
static inline void DeleteCriticalSection(LPCRITICAL_SECTION cs) {
    if (cs->initialized) { pthread_mutex_destroy(&cs->m); cs->initialized = 0; }
}

#define FILE_BEGIN   0
#define FILE_CURRENT 1
#define FILE_END     2
#define INVALID_SET_FILE_POINTER ((DWORD)-1)

extern "C" {
BOOL  CreateDirectoryW(LPCWSTR path, void* sa);
DWORD SetFilePointer(HANDLE h, int32_t distLow, int32_t* distHigh, DWORD method);
}

// Both MSVC shapes: explicit-capacity, and the array template the callers use
// as swprintf_s(buf, L"fmt", args...). A macro can't overload on arity, so these
// are real functions.
inline int swprintf_s(wchar_t* d, size_t cap, const wchar_t* f, ...) {
    va_list ap; va_start(ap, f); const int r = vswprintf(d, cap, f, ap); va_end(ap); return r;
}
template <size_t N, class... A>
inline int swprintf_s(wchar_t (&d)[N], const wchar_t* f, A... a) {
    return swprintf(d, N, f, a...);
}
static inline BOOL IsDebuggerPresent(void) { return FALSE; }
