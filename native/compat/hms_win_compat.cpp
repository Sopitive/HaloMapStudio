// =============================================================================
// hms_win_compat.cpp - POSIX implementations of the Win32 surface the offline
// parsers use. Compiled ONLY into the non-Windows build.
//
// Handles are heap cells rather than fds cast to pointers, because the parsers
// mix file handles and mapping handles in the same HANDLE-typed variables and
// close them with a single CloseHandle().
// =============================================================================
#include "hms_windows_shim.h"
#include "shlobj.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <errno.h>
#include <time.h>
#include <string>
#include <mutex>
#include <map>

namespace {

enum class Kind { File, Mapping };

struct Cell {
    Kind  kind;
    int   fd     = -1;     // File: owned fd. Mapping: borrowed (not closed).
    off_t size   = 0;
    int   prot   = PROT_READ;
};

thread_local DWORD t_lastError = 0;
inline void t_lastErrorSet(DWORD e) { t_lastError = e; }

// Active mappings, so UnmapViewOfFile() can recover the length from the base
// pointer (munmap needs a size; Win32's UnmapViewOfFile does not take one).
std::mutex                 g_mapMutex;
std::map<void*, size_t>    g_mapSizes;

// UTF-32 wchar_t (Linux) -> UTF-8. The Rust side encodes paths to match.
std::string ToUtf8(const wchar_t* w) {
    if (!w) return {};
    std::string out;
    for (const wchar_t* p = w; *p; ++p) {
        uint32_t c = (uint32_t)*p;
        if (c < 0x80) {
            out += (char)c;
        } else if (c < 0x800) {
            out += (char)(0xC0 | (c >> 6));
            out += (char)(0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            out += (char)(0xE0 | (c >> 12));
            out += (char)(0x80 | ((c >> 6) & 0x3F));
            out += (char)(0x80 | (c & 0x3F));
        } else {
            out += (char)(0xF0 | (c >> 18));
            out += (char)(0x80 | ((c >> 12) & 0x3F));
            out += (char)(0x80 | ((c >> 6) & 0x3F));
            out += (char)(0x80 | (c & 0x3F));
        }
    }
    return out;
}

size_t FromUtf8(const std::string& s, wchar_t* out, size_t cch) {
    size_t n = 0;
    for (size_t i = 0; i < s.size() && n + 1 < cch; ) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp; int len;
        if      (c < 0x80)       { cp = c;          len = 1; }
        else if ((c & 0xE0)==0xC0){ cp = c & 0x1F;  len = 2; }
        else if ((c & 0xF0)==0xE0){ cp = c & 0x0F;  len = 3; }
        else                      { cp = c & 0x07;  len = 4; }
        for (int k = 1; k < len && i + k < s.size(); ++k)
            cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out[n++] = (wchar_t)cp;
        i += len;
    }
    if (cch) out[n] = 0;
    return n;
}

} // namespace

extern "C" {

HANDLE CreateFileW(LPCWSTR path, DWORD access, DWORD /*share*/, void* /*sa*/,
                   DWORD disposition, DWORD /*flags*/, HANDLE /*tmpl*/) {
    const std::string p = ToUtf8(path);
    int flags = 0;
    const bool wantWrite = (access & (GENERIC_WRITE | FILE_APPEND_DATA)) != 0;
    const bool wantRead  = (access & GENERIC_READ) != 0;

    if (wantWrite && wantRead)      flags = O_RDWR;
    else if (wantWrite)             flags = O_WRONLY;
    else                            flags = O_RDONLY;

    if (access & FILE_APPEND_DATA)  flags |= O_APPEND;
    if (disposition == CREATE_ALWAYS) flags |= O_CREAT | O_TRUNC;
    else if (disposition == OPEN_ALWAYS) flags |= O_CREAT;

    const int fd = ::open(p.c_str(), flags, 0644);
    if (fd < 0) { t_lastError = (DWORD)errno; return INVALID_HANDLE_VALUE; }

    struct stat st{};
    Cell* c = new Cell{Kind::File, fd, (::fstat(fd, &st) == 0) ? st.st_size : 0, PROT_READ};
    return (HANDLE)c;
}

HANDLE CreateFileMappingW(HANDLE file, void* /*sa*/, DWORD protect,
                          DWORD /*maxHigh*/, DWORD /*maxLow*/, LPCWSTR /*name*/) {
    // Only file-backed mappings are used by the parsers; the named page-file
    // mappings live in the excluded live-game translation units.
    if (file == INVALID_HANDLE_VALUE || !file) { t_lastError = EINVAL; return nullptr; }
    Cell* f = (Cell*)file;
    Cell* m = new Cell{Kind::Mapping, f->fd, f->size,
                       (protect == PAGE_READWRITE) ? (PROT_READ | PROT_WRITE) : PROT_READ};
    return (HANDLE)m;
}

LPVOID MapViewOfFile(HANDLE mapping, DWORD /*access*/, DWORD offHigh,
                     DWORD offLow, SIZE_T bytes) {
    if (!mapping || mapping == INVALID_HANDLE_VALUE) { t_lastError = EINVAL; return nullptr; }
    Cell* m = (Cell*)mapping;
    const off_t off = ((off_t)offHigh << 32) | (off_t)offLow;
    // Win32: 0 means "to end of file".
    const size_t len = bytes ? bytes : (size_t)(m->size - off);
    if (len == 0) { t_lastError = EINVAL; return nullptr; }

    void* base = ::mmap(nullptr, len, m->prot, MAP_SHARED, m->fd, off);
    if (base == MAP_FAILED) { t_lastError = (DWORD)errno; return nullptr; }

    std::lock_guard<std::mutex> lk(g_mapMutex);
    g_mapSizes[base] = len;
    return base;
}

BOOL UnmapViewOfFile(LPCVOID base) {
    if (!base) return FALSE;
    size_t len = 0;
    {
        std::lock_guard<std::mutex> lk(g_mapMutex);
        auto it = g_mapSizes.find((void*)base);
        if (it == g_mapSizes.end()) return FALSE;
        len = it->second;
        g_mapSizes.erase(it);
    }
    return ::munmap((void*)base, len) == 0 ? TRUE : FALSE;
}

BOOL CloseHandle(HANDLE h) {
    if (!h || h == INVALID_HANDLE_VALUE) return FALSE;
    Cell* c = (Cell*)h;
    if (c->kind == Kind::File && c->fd >= 0) ::close(c->fd);  // mappings borrow the fd
    delete c;
    return TRUE;
}

BOOL GetFileSizeEx(HANDLE h, LARGE_INTEGER* size) {
    if (!h || h == INVALID_HANDLE_VALUE || !size) return FALSE;
    Cell* c = (Cell*)h;
    struct stat st{};
    if (::fstat(c->fd, &st) != 0) { t_lastError = (DWORD)errno; return FALSE; }
    c->size = st.st_size;
    size->QuadPart = st.st_size;
    return TRUE;
}

BOOL WriteFile(HANDLE h, LPCVOID buf, DWORD len, LPDWORD written, void*) {
    if (!h || h == INVALID_HANDLE_VALUE) return FALSE;
    Cell* c = (Cell*)h;
    const ssize_t n = ::write(c->fd, buf, len);
    if (written) *written = (n > 0) ? (DWORD)n : 0;
    return n >= 0 ? TRUE : FALSE;
}

DWORD GetLastError(void)      { return t_lastError; }
void  SetLastError(DWORD e)   { t_lastError = e; }

// dladdr gives us the .so's own path - the ELF analogue of asking for the
// module handle containing a given function address.
BOOL GetModuleHandleExW(DWORD, LPCWSTR name, HMODULE* out) {
    if (!out) return FALSE;
    Dl_info info{};
    if (::dladdr((const void*)name, &info) && info.dli_fname) {
        *out = (HMODULE)info.dli_fbase;
        return TRUE;
    }
    *out = nullptr;
    return FALSE;
}

DWORD GetModuleFileNameW(HMODULE mod, LPWSTR buf, DWORD cch) {
    if (!buf || !cch) return 0;
    Dl_info info{};
    const void* probe = mod ? (const void*)mod : (const void*)&GetModuleFileNameW;
    if (::dladdr(probe, &info) && info.dli_fname)
        return (DWORD)FromUtf8(info.dli_fname, buf, cch);
    buf[0] = 0;
    return 0;
}

HMODULE LoadLibraryW(LPCWSTR path) { return (HMODULE)::dlopen(ToUtf8(path).c_str(), RTLD_NOW); }
void*   GetProcAddress(HMODULE m, LPCSTR n) { return ::dlsym(m, n); }
BOOL    FreeLibrary(HMODULE m) { return ::dlclose(m) == 0 ? TRUE : FALSE; }

void OutputDebugStringA(LPCSTR s) { if (s) ::fputs(s, stderr); }
void OutputDebugStringW(LPCWSTR s) { if (s) ::fputs(ToUtf8(s).c_str(), stderr); }

void GetLocalTime(SYSTEMTIME* st) {
    if (!st) return;
    struct timespec ts{}; ::clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv{}; ::localtime_r(&ts.tv_sec, &tmv);
    st->wYear = (WORD)(tmv.tm_year + 1900); st->wMonth  = (WORD)(tmv.tm_mon + 1);
    st->wDay  = (WORD)tmv.tm_mday;          st->wHour   = (WORD)tmv.tm_hour;
    st->wMinute = (WORD)tmv.tm_min;         st->wSecond = (WORD)tmv.tm_sec;
    st->wMilliseconds = (WORD)(ts.tv_nsec / 1000000);
    st->wDayOfWeek = (WORD)tmv.tm_wday;
}

ULONGLONG GetTickCount64(void) {
    struct timespec ts{}; ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ULONGLONG)ts.tv_sec * 1000ull + (ULONGLONG)(ts.tv_nsec / 1000000);
}

void Sleep(DWORD ms) { ::usleep((useconds_t)ms * 1000); }

} // extern "C"

// ---- extra Win32 odds and ends --------------------------------------------
extern "C" {

DWORD GetEnvironmentVariableA(LPCSTR name, LPSTR buf, DWORD cch) {
    const char* v = name ? ::getenv(name) : nullptr;
    if (!v) { t_lastErrorSet(0xCB); return 0; }          // ERROR_ENVVAR_NOT_FOUND
    const size_t n = ::strlen(v);
    if (!buf || cch == 0 || n + 1 > cch) return (DWORD)(n + 1);
    ::memcpy(buf, v, n + 1);
    return (DWORD)n;
}

BOOL QueryPerformanceFrequency(LARGE_INTEGER* v) {
    if (!v) return FALSE;
    v->QuadPart = 1000000000ll;                          // we report ns ticks
    return TRUE;
}

BOOL QueryPerformanceCounter(LARGE_INTEGER* v) {
    if (!v) return FALSE;
    struct timespec ts{}; ::clock_gettime(CLOCK_MONOTONIC, &ts);
    v->QuadPart = (long long)ts.tv_sec * 1000000000ll + ts.tv_nsec;
    return TRUE;
}

// The parsers only ever convert UTF-8 <-> wide for paths and tag names.
int MultiByteToWideChar(UINT, DWORD, LPCSTR mb, int cb, LPWSTR wc, int cch) {
    if (!mb) return 0;
    const std::string s = (cb < 0) ? std::string(mb) : std::string(mb, (size_t)cb);
    if (!wc || cch == 0) return (int)(s.size() + 1);
    return (int)FromUtf8(s, wc, (size_t)cch);
}

int WideCharToMultiByte(UINT, DWORD, LPCWSTR wc, int, LPSTR mb, int cb, LPCSTR, BOOL*) {
    if (!wc) return 0;
    const std::string s = ToUtf8(wc);
    if (!mb || cb == 0) return (int)(s.size() + 1);
    const size_t n = (s.size() + 1 <= (size_t)cb) ? s.size() : (size_t)cb - 1;
    ::memcpy(mb, s.data(), n); mb[n] = 0;
    return (int)n;
}

} // extern "C"

// SHGetFolderPathW -> XDG. Only CSIDL_LOCAL_APPDATA is asked for (log location).
extern "C" HRESULT SHGetFolderPathW(void*, int, HANDLE, DWORD, LPWSTR path) {
    if (!path) return (HRESULT)0x80004005;
    const char* xdg  = ::getenv("XDG_DATA_HOME");
    const char* home = ::getenv("HOME");
    std::string dir;
    if (xdg && *xdg)      dir = xdg;
    else if (home)        dir = std::string(home) + "/.local/share";
    else                  dir = "/tmp";
    FromUtf8(dir, path, MAX_PATH);
    return 0;
}

extern "C" {

BOOL CreateDirectoryW(LPCWSTR path, void*) {
    const std::string p = ToUtf8(path);
    if (::mkdir(p.c_str(), 0755) == 0) return TRUE;
    t_lastErrorSet((DWORD)errno);
    return (errno == EEXIST) ? FALSE : FALSE;      // Win32 also fails if it exists
}

DWORD SetFilePointer(HANDLE h, int32_t distLow, int32_t* distHigh, DWORD method) {
    if (!h || h == INVALID_HANDLE_VALUE) return INVALID_SET_FILE_POINTER;
    Cell* c = (Cell*)h;
    off_t off = distLow;
    if (distHigh) off |= ((off_t)*distHigh << 32);
    const int whence = (method == FILE_END) ? SEEK_END
                     : (method == FILE_CURRENT) ? SEEK_CUR : SEEK_SET;
    const off_t r = ::lseek(c->fd, off, whence);
    if (r < 0) { t_lastErrorSet((DWORD)errno); return INVALID_SET_FILE_POINTER; }
    if (distHigh) *distHigh = (int32_t)(r >> 32);
    return (DWORD)(r & 0xFFFFFFFFu);
}

}
