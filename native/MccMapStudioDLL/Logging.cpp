// Logging.cpp
// =============================================================================
// Minimal ZH_Logf for HaloMapStudioDLL.
//
// The copied snapshot publishers (ObjectTableSnapshot, MapInfoSnapshot, ...) and
// ForgePaletteSnapshot extern-declare `void ZH_Logf(const char*, ...)` and
// expect a real implementation in the same DLL. Just enough is provided that
// (a) the linker resolves the symbol, and (b) viewer-side debugging can pick up
// the messages from a known location.
//
// Sink: %LOCALAPPDATA%\HaloMapStudio\dll.log (created on first call). Each
// line is appended to the file under a critical section, and also forwarded
// to OutputDebugStringA for live debugger viewing. There is no truncation /
// rotation - the file is small (forge palette walk + map-info ticks log only
// on real state changes) and the user typically clears it between runs.
// =============================================================================

#include "pch.h"
#include <windows.h>
#include <shlobj.h>     // SHGetFolderPathW
#include <cstdarg>
#include <cstdio>
#include <cstring>

#pragma comment(lib, "shell32.lib")

namespace {

CRITICAL_SECTION g_LogLock;
bool             g_LogLockInit = false;
HANDLE           g_LogFile     = INVALID_HANDLE_VALUE;
bool             g_LogTried    = false;

// Build the log file path. Returns false if %LOCALAPPDATA% can't be resolved
// or the directory can't be created.
static bool ResolveLogPath(wchar_t out[MAX_PATH])
{
    wchar_t appData[MAX_PATH] = {};
    if (FAILED(SHGetFolderPathW(nullptr, CSIDL_LOCAL_APPDATA, nullptr, 0, appData)))
        return false;

    wchar_t dir[MAX_PATH];
    if (swprintf_s(dir, L"%s\\HaloMapStudio", appData) < 0) return false;
    CreateDirectoryW(dir, nullptr); // ignore exists / failure here

    if (swprintf_s(out, MAX_PATH, L"%s\\dll.log", dir) < 0) return false;
    return true;
}

static void EnsureLogOpen()
{
    if (!g_LogLockInit) {
        InitializeCriticalSection(&g_LogLock);
        g_LogLockInit = true;
    }
    if (g_LogTried) return;
    g_LogTried = true;

    wchar_t path[MAX_PATH] = {};
    if (!ResolveLogPath(path)) return;

    g_LogFile = CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (g_LogFile == INVALID_HANDLE_VALUE) return;

    SetFilePointer(g_LogFile, 0, nullptr, FILE_END);
    const char banner[] = "\r\n--- HaloMapStudioDLL log opened ---\r\n";
    DWORD w = 0;
    WriteFile(g_LogFile, banner, (DWORD)(sizeof(banner) - 1), &w, nullptr);
}

} // namespace

extern "C" void ZH_Logf(const char* fmt, ...)
{
    EnsureLogOpen();

    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;

    // OutputDebugStringA does a synchronous kernel transition on EVERY call (a
    // full IPC round-trip when a debugger is attached). ZH_Logf is reachable
    // from the per-frame engine pump, so emitting the debug string unconditionally
    // was a per-log stall on the frame thread. Only emit it when actually
    // debugging; the file sink below is the normal path.
    if (IsDebuggerPresent()) OutputDebugStringA(buf);

    if (g_LogFile == INVALID_HANDLE_VALUE) return;
    EnterCriticalSection(&g_LogLock);
    DWORD w = 0;
    WriteFile(g_LogFile, buf, (DWORD)strlen(buf), &w, nullptr);
    LeaveCriticalSection(&g_LogLock);
}

// Optional: detach hook to close the file cleanly.
extern "C" void ZH_Logf_Shutdown()
{
    if (g_LogLockInit) {
        EnterCriticalSection(&g_LogLock);
        if (g_LogFile != INVALID_HANDLE_VALUE) {
            CloseHandle(g_LogFile);
            g_LogFile = INVALID_HANDLE_VALUE;
        }
        LeaveCriticalSection(&g_LogLock);
    }
}
