// MapCacheCommon.cpp
// =============================================================================
// Implementation of the shared cache-file infrastructure declared in
// MapCacheCommon.h. Extracted from MapBitmapParser so both the
// bitmap and the model parsers can share one open-cache handle.
//
// Responsibilities:
//   * AcquireCacheHandle / ReleaseCacheHandle - file mmap + handle table.
//   * ParseCacheHeader - detect MCC build, populate tag index + tag-name
//     table.
//   * ParseGestalt / ParseLayoutTable - global resource tables.
//   * EnsureResourceFixups - per-entry lazy fixup-block load.
//   * ReadResourceData - decompress / memcpy a segment to a malloc'd buffer.
//
// Codec note:
//   Resource pages in MCC HaloReach .map files are RAW DEFLATE (RFC 1951),
//   per Reclaimer source (Blam/Common/Annotations.cs:20-22 - Gen3+ default
//   of Deflate; Blam/Common/ContentFactory.cs:304-321 - DeflateStream raw).
//   We use the vendored miniz (richgel999/miniz, 3.1.x, public domain) via
//   tinfl_decompress_mem_to_mem with flags=0 (no zlib header). DecompressLZX
//   is kept as a fallback for genuinely older formats (Halo3-era / pre-MCC)
//   that still use xcompress64.dll.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"
#include <chrono>

#include <atomic>
#include <new>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>

// miniz: vendored single-header raw-deflate decoder. The implementation lives
// in miniz.c (the ONLY TU that defines MINIZ_IMPLEMENTATION); here we just pull
// in the declarations so DecompressDeflate below can call tinfl_decompress_mem_to_mem.
//
// The vendored miniz.h auto-emits the implementation unless either
// MINIZ_HEADER_FILE_ONLY or MINIZ_DECLARED_IMPL is set - define the former so
// this TU gets declarations only and we don't get LNK2005 multiply-defined
// symbols against miniz.c.
#define MINIZ_HEADER_FILE_ONLY
extern "C" {
#include "miniz.h"
}
#undef MINIZ_HEADER_FILE_ONLY

// libdeflate public API (vendored ebiggers/libdeflate v1.19). Has its own extern "C".
#include "libdeflate/libdeflate.h"

// Per-cache cache-purge hooks (defined in LightmapParser.cpp / MapBspParser.cpp).
// Called from ReleaseCacheHandle to free the ~155 MB-class decompressed LBSP pages and the
// per-shader material-walk caches when a map's cache handle is closed.
extern "C" void PurgeLightmapCachesForCache(void* cachePtr);
extern "C" void PurgeBspMaterialCachesForCache(void* cachePtr);

namespace zh_mcc {

// Native-side diagnostic log. Writes next to the host exe (C:\ root requires
// elevation for non-admin processes, which silently nukes the log). The path
// is resolved once on first call via GetModuleFileNameW(NULL, ...).
//
// Exposed to other translation units (MapModelParser.cpp, MapBitmapParser.cpp)
// via the declaration in MapCacheCommon.h.
void NativeDiag(const char* fmt, ...)
{
    // PERF: an unconditional log would CreateFile/WriteFile/CloseHandle
    // synchronously on EVERY resource read - 20k+ lines / ~56MB per map load,
    // pure overhead on the hot load path. Gated behind MMS_NATIVE_LOG=1
    // (default OFF). Checked once; benign init race (all threads compute the
    // same value).
    static std::atomic<int> s_diagEnabled{-1};
    int en = s_diagEnabled.load(std::memory_order_acquire);
    if (en < 0) {
        char v[8] = {0};
        DWORD r = GetEnvironmentVariableA("MMS_NATIVE_LOG", v, (DWORD)sizeof(v));
        en = (r > 0 && v[0] == '1') ? 1 : 0;
        s_diagEnabled.store(en, std::memory_order_release);
    }
    if (en == 0) return;

    char buf[1024];
    va_list ap; va_start(ap, fmt);
    int n = _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
    va_end(ap);
    if (n <= 0) return;

    static wchar_t s_logPath[MAX_PATH] = {0};
    static std::atomic<int> s_pathInit{0};
    int prev = s_pathInit.load(std::memory_order_acquire);
    if (prev == 0) {
        wchar_t exePath[MAX_PATH] = {0};
        DWORD got = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        if (got == 0 || got >= MAX_PATH) {
            s_pathInit.store(2, std::memory_order_release);
            return;
        }
        wchar_t* sep = wcsrchr(exePath, L'\\');
        if (sep) *(sep + 1) = 0;
        wcsncpy_s(s_logPath, exePath, _TRUNCATE);
        wcsncat_s(s_logPath, L"HaloMapStudio_native.log", _TRUNCATE);
        s_pathInit.store(1, std::memory_order_release);
    } else if (prev == 2) {
        return;
    }

    HANDLE h = CreateFileW(s_logPath,
        FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;

    char ts[32];
    SYSTEMTIME st; GetLocalTime(&st);
    int tn = _snprintf_s(ts, sizeof(ts), _TRUNCATE,
        "%02u:%02u:%02u.%03u ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);

    DWORD wrote = 0;
    if (tn > 0) WriteFile(h, ts, (DWORD)tn, &wrote, nullptr);
    WriteFile(h, buf, (DWORD)n, &wrote, nullptr);
    WriteFile(h, "\r\n", 2, &wrote, nullptr);
    CloseHandle(h);
}

// -----------------------------------------------------------------------------
// xcompress64.dll dynamic loading 
// -----------------------------------------------------------------------------

namespace {

enum XMemCodecType : int { XMEM_DEFAULT = 0, XMEM_LZX = 1 };

typedef long (WINAPI *PFN_XMemCreateDecompressionContext)(
    XMemCodecType codecType, const void* pCodecParams, int flags, void** pContext);
typedef void (WINAPI *PFN_XMemDestroyDecompressionContext)(void* context);
typedef long (WINAPI *PFN_XMemResetDecompressionContext)(void* context);
typedef long (WINAPI *PFN_XMemDecompressStream)(
    void* context, void* pDestination, size_t* pDestSize,
    const void* pSource, size_t* pSrcSize);

struct XCompressApi {
    HMODULE                                 module = nullptr;
    PFN_XMemCreateDecompressionContext      create  = nullptr;
    PFN_XMemDestroyDecompressionContext     destroy = nullptr;
    PFN_XMemResetDecompressionContext       reset   = nullptr;
    PFN_XMemDecompressStream                stream  = nullptr;
    bool                                    tried   = false;
    bool                                    ok      = false;
};

XCompressApi g_xcompress;
std::mutex   g_xcompressMutex;

HMODULE LoadXCompress64() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                       reinterpret_cast<LPCWSTR>(&LoadXCompress64), &self);

    wchar_t path[MAX_PATH];
    DWORD n = self ? GetModuleFileNameW(self, path, MAX_PATH) : 0;
    if (n > 0 && n < MAX_PATH) {
        for (DWORD i = n; i > 0; --i) {
            if (path[i - 1] == L'\\' || path[i - 1] == L'/') {
                path[i] = 0;
                break;
            }
        }
        wcscat_s(path, MAX_PATH, L"xcompress64.dll");
        HMODULE m = LoadLibraryW(path);
        if (m) return m;
    }
    return LoadLibraryW(L"xcompress64.dll");
}

bool EnsureXCompress() {
    std::lock_guard<std::mutex> lock(g_xcompressMutex);
    if (g_xcompress.tried) return g_xcompress.ok;
    g_xcompress.tried = true;

    g_xcompress.module = LoadXCompress64();
    if (!g_xcompress.module) return false;

    g_xcompress.create  = reinterpret_cast<PFN_XMemCreateDecompressionContext>(
        GetProcAddress(g_xcompress.module, "XMemCreateDecompressionContext"));
    g_xcompress.destroy = reinterpret_cast<PFN_XMemDestroyDecompressionContext>(
        GetProcAddress(g_xcompress.module, "XMemDestroyDecompressionContext"));
    g_xcompress.reset   = reinterpret_cast<PFN_XMemResetDecompressionContext>(
        GetProcAddress(g_xcompress.module, "XMemResetDecompressionContext"));
    g_xcompress.stream  = reinterpret_cast<PFN_XMemDecompressStream>(
        GetProcAddress(g_xcompress.module, "XMemDecompressStream"));

    g_xcompress.ok = g_xcompress.create && g_xcompress.destroy
                  && g_xcompress.reset && g_xcompress.stream;
    return g_xcompress.ok;
}

// Single XMemDecompressStream call, SEH-guarded so a malformed input cannot
// take down the host. Returns 0 on AV / negative rc; otherwise writes consumed
// + produced into the out params and returns 1.
//
// XMemDecompressStream raises a structured exception (access violation) when
// fed bytes that don't form a valid LZX block header - typically when we're
// scanning for the next chunk boundary and feed mid-stream garbage. The fault
// addresses are inside libxcompress64.dll's internal scratch state, NOT in
// our buffers, so we can resume safely.
// Only catch ACCESS_VIOLATION. Catching anything else (notably 0xC0000409
// __fastfail / stack-canary failures) via EXCEPTION_EXECUTE_HANDLER is UB -
// it leaves the thread's exception state partially corrupted and a future
// fault can transfer execution to a stale RIP that resolves to whatever
// happens to be at that offset in our DLL (we hit MatchBuild this way once).
static int SafeStreamCall(void* ctx,
                          uint8_t* dst, size_t* dstSize,
                          const uint8_t* src, size_t* srcSize,
                          long* outRc)
{
    int ok = 0;
    __try {
        *outRc = g_xcompress.stream(ctx, dst, dstSize,
                                    const_cast<uint8_t*>(src), srcSize);
        ok = 1;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION
                ? EXCEPTION_EXECUTE_HANDLER
                : EXCEPTION_CONTINUE_SEARCH) {
        ok = 0;
    }
    return ok;
}

// Chunked / iterative LZX decompress.
//
// Reach (and other Gen3 MCC) resource pages are chunked LZX: a single
// XMemDecompressStream call only consumes one inner LZX stream (returns
// dstSize=srcSize=0 at the boundary). XMemResetDecompressionContext alone
// does not reliably resume from the next byte - we have to destroy + recreate
// the context (verified empirically with verify_lzx_iter3.py against
// shared.map's rid=0xf7ed168a, which has 1030 chunks producing 158MB).
//
// Boundaries between chunks are NOT byte-aligned in any obvious way; the
// observed deltas between "consumed-up-to" and "next-valid-stream-start" range
// from 3 to 228 bytes and look like stale/padding tails XMemDecompressStream
// failed to consume. So if a fresh-context call from the reported srcOffset
// AVs or makes no progress, we scan forward up to ~kBoundaryScan bytes
// looking for a position where a new ctx can decode. Each probe is SEH-
// guarded; misalignments raise AVs inside xcompress64 scratch memory which
// the host process must NOT inherit.
//
// On any unrecoverable error we return what we have so far. Callers compare
// returned size to the expected decompressed size and log a partial-decode
// warning - better than dying, and partially-decoded model resources tend
// to surface a useful diagnostic upstream.
// Single-shot LZX decompress. The iterative chunk-scan version was crashing
// the host with /GS stack-canary failures (0xC0000409) - destroying and
// re-creating the xcompress64 context up to 512 times per resource while
// SEH-catching AVs from xcompress's internal scratch state evidently corrupts
// something the CRT then catches. Single-shot is crash-safe; the trade-off is
// that multi-chunk pages return only the first chunk's worth of data, which
// the caller treats as a partial decode and bails. Most simple model pages
// fit in one chunk anyway.
size_t DecompressLZX(const uint8_t* in, size_t inSize, uint8_t* outBuf, size_t outSize) {
    if (!EnsureXCompress()) return 0;
    if (!in || !outBuf || inSize == 0 || outSize == 0) return 0;

    void* ctx = nullptr;
    if (g_xcompress.create(XMEM_LZX, nullptr, 0, &ctx) < 0 || !ctx) {
        NativeDiag("DecompressLZX: create ctx failed");
        return 0;
    }
    g_xcompress.reset(ctx);

    size_t ds = outSize;
    size_t ss = inSize;
    long rc = 0;
    int ok = SafeStreamCall(ctx, outBuf, &ds, in, &ss, &rc);
    g_xcompress.destroy(ctx);

    if (!ok || rc < 0) {
        NativeDiag("DecompressLZX: stream call failed ok=%d rc=%ld inSize=%llu outSize=%llu",
            ok, rc, (unsigned long long)inSize, (unsigned long long)outSize);
        return 0;
    }
    NativeDiag("DecompressLZX: decoded %llu / %llu (consumed %llu / %llu)",
        (unsigned long long)ds, (unsigned long long)outSize,
        (unsigned long long)ss, (unsigned long long)inSize);
    return ds;
}

// -----------------------------------------------------------------------------
// Raw DEFLATE (RFC 1951) decompress via vendored miniz
// -----------------------------------------------------------------------------
//
// Reclaimer source confirms MccHaloReachU13 (and other Gen3+ MCC builds with
// the 3-arg [CacheMetadata] annotation) use raw deflate, not LZ4 and not LZX:
//
//   Reclaimer.Blam/Blam/Common/Annotations.cs:20-22
//     // Gen3+ defaults to Deflate when no codec is specified
//
//   Reclaimer.Blam/Blam/Common/ContentFactory.cs:304-321
//     using (var ds = new DeflateStream(reader.BaseStream, CompressionMode.Decompress))
//
// .NET's DeflateStream is RAW DEFLATE (no zlib/gzip wrapper). The miniz
// equivalent is tinfl_decompress_mem_to_mem with TINFL_FLAG_PARSE_ZLIB_HEADER
// UNSET (i.e. flags=0). We use that one-shot API rather than streaming
// inflate because the decompressed page size is bounded and known up front
// (page.decompressedSize), so the whole buffer fits in heap.
//
// Returns the number of bytes actually decompressed, or 0 on failure.
// segmentOffset handling is the caller's responsibility - ReadResourceData
// slices the result after this returns.
size_t DecompressDeflate(const uint8_t* in, size_t inSize,
                         uint8_t* outBuf, size_t outSize)
{
    if (!in || !outBuf || inSize == 0 || outSize == 0) return 0;

    // libdeflate: ~3x faster raw-DEFLATE than miniz for the map resource pages (the
    // dominant load cost). The decompressor object is NOT thread-safe, so keep one per
    // thread (ReadResourceData runs the inflate lock-free across rayon workers). Falls
    // back to miniz's tinfl on any libdeflate failure so a bad page still decodes.
    static thread_local struct libdeflate_decompressor* s_dd = nullptr;
    if (!s_dd) s_dd = libdeflate_alloc_decompressor();
    if (s_dd) {
        size_t got = 0;
        enum libdeflate_result r = libdeflate_deflate_decompress(
            s_dd, in, inSize, outBuf, outSize, &got);
        if (r == LIBDEFLATE_SUCCESS) {
            NativeDiag("Deflate: ok(libdeflate) decoded %llu / %llu",
                (unsigned long long)got, (unsigned long long)outSize);
            return got;
        }
        // r == LIBDEFLATE_INSUFFICIENT_SPACE means the stream decompresses to MORE than
        // outSize; that's a genuine size mismatch, not a codec issue - fall through to tinfl
        // which stops at outSize (matches the prior behavior for over-large pages).
    }

    // Fallback: miniz tinfl (flags=0 = raw deflate, no zlib header / adler check).
    size_t rc = tinfl_decompress_mem_to_mem(outBuf, outSize,
                                            in, inSize,
                                            /*flags*/ 0);
    if (rc == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED) {
        NativeDiag("Deflate: failed (both libdeflate + tinfl) inSize=%llu outSize=%llu",
            (unsigned long long)inSize, (unsigned long long)outSize);
        return 0;
    }
    NativeDiag("Deflate: ok(tinfl-fallback) decoded %llu / %llu",
        (unsigned long long)rc, (unsigned long long)outSize);
    return rc;
}

// -----------------------------------------------------------------------------
// Build detection
// -----------------------------------------------------------------------------

struct BuildBinding {
    const char* str;
    CacheType   type;
};

const BuildBinding kBuilds[] = {
    { "Jun 24 2019 00:36:03", CacheType::MccHaloReach },
    { "Jul 30 2019 14:17:16", CacheType::MccHaloReach },
    { "Oct 24 2019 15:56:32", CacheType::MccHaloReach },
    { "Jan 30 2020 16:55:25", CacheType::MccHaloReach },
    { "Mar 24 2020 12:10:36", CacheType::MccHaloReach },
    { "Jun  5 2020 10:40:14", CacheType::MccHaloReachU3 },
    { "Oct 15 2020 18:23:50", CacheType::MccHaloReachU3 },
    { "Nov 24 2020 18:32:37", CacheType::MccHaloReachU3 },
    { "Mar  4 2021 13:14:28", CacheType::MccHaloReachU3 },
    { "May 26 2021 10:02:45", CacheType::MccHaloReachU3 },
    { "Aug 11 2021 15:50:30", CacheType::MccHaloReachU8 },
    { "Sep 13 2021 09:49:52", CacheType::MccHaloReachU8 },
    { "Sep 17 2021 13:25:40", CacheType::MccHaloReachU8 },
    { "Jan 13 2022 00:54:50", CacheType::MccHaloReachU10 },
    { "Aug  5 2022 20:35:02", CacheType::MccHaloReachU10 },
    { "Aug 31 2022 11:53:49", CacheType::MccHaloReachU10 },
    { "Aug 31 2022 11:53:19", CacheType::MccHaloReachU10 },
    { "Oct 12 2022 01:55:01", CacheType::MccHaloReachU10 },
    { "Nov 16 2022 21:11:25", CacheType::MccHaloReachU10 },
    { "Nov 16 2022 21:13:04", CacheType::MccHaloReachU10 },
    { "Jun 21 2023 15:35:31", CacheType::MccHaloReachU13 },
    { "Jun 27 2023 08:55:51", CacheType::MccHaloReachU13 },
    { "Jun 27 2023 08:55:19", CacheType::MccHaloReachU13 },
    { "Jul 16 2023 16:12:13", CacheType::MccHaloReachU13 },
    { "Jul 16 2023 16:08:14", CacheType::MccHaloReachU13 },
};

CacheType MatchBuild(const char* s) {
    if (!s || !*s) return CacheType::Unknown;
    for (const auto& b : kBuilds)
        if (strcmp(s, b.str) == 0) return b.type;
    return CacheType::Unknown;
}

} // anonymous namespace

PointerExpander MakeExpander(const char* buildString) {
    PointerExpander e{ 0x50000000 };
    if (buildString && (
        strcmp(buildString, "Jun 24 2019 00:36:03") == 0 ||
        strcmp(buildString, "Jul 30 2019 14:17:16") == 0))
    {
        e.magic = 0x10000000;
    }
    return e;
}

// -----------------------------------------------------------------------------
// Address translators
// -----------------------------------------------------------------------------

int64_t TagAddrToFileOff(const CacheHandle* c, int64_t tagAddress) {
    int64_t fileOff = tagAddress - c->tagMagic;
    if (fileOff < 0 || (size_t)fileOff >= c->size) return -1;
    return fileOff;
}
int64_t HdrAddrToFileOff(const CacheHandle* c, int64_t hdrAddress) {
    int64_t fileOff = hdrAddress - c->headerMagic;
    if (fileOff < 0 || (size_t)fileOff >= c->size) return -1;
    return fileOff;
}
int64_t TagMetaFileOff(CacheHandle* cache, uint32_t metaPointerRaw) {
    int64_t expanded = cache->expander.Expand((int32_t)metaPointerRaw);
    return TagAddrToFileOff(cache, expanded);
}

// -----------------------------------------------------------------------------
// Cache header parse
// -----------------------------------------------------------------------------

bool ParseCacheHeader(CacheHandle* cache) {
    if (cache->size < 1300) { NativeDiag("Parse: size<1300 size=%llu", (unsigned long long)cache->size); return false; }
    const uint8_t* p = cache->base;
    // Accept either "head" (older builds) or "daeh" (U13+ MCC writes the
    // magic as a little-endian 32-bit integer, so the bytes on disk are
    // reversed). Reject anything else.
    bool magicHead = (p[0] == 'h' && p[1] == 'e' && p[2] == 'a' && p[3] == 'd');
    bool magicDaeh = (p[0] == 'd' && p[1] == 'a' && p[2] == 'e' && p[3] == 'h');
    if (!magicHead && !magicDaeh) {
        NativeDiag("Parse: magic mismatch %02x %02x %02x %02x", p[0], p[1], p[2], p[3]);
        return false;
    }
    NativeDiag("Parse: magic-ok %s", magicDaeh ? "daeh" : "head");

    char bs[33] = {0};
    memcpy(bs, p + 160, 32); bs[32] = 0;
    CacheType ct = MatchBuild(bs);
    NativeDiag("Parse: build@160='%s' ct-initial=%d", bs, (int)ct);
    bool isU8 = (ct == CacheType::MccHaloReachU8 ||
                 ct == CacheType::MccHaloReachU10 ||
                 ct == CacheType::MccHaloReachU13);

    if (!isU8) {
        memcpy(bs, p + 288, 32); bs[32] = 0;
        ct = MatchBuild(bs);
        if (ct == CacheType::Unknown) {
            memcpy(bs, p + 160, 32); bs[32] = 0;
            isU8 = true;
        }
    }

    memcpy(cache->buildString, bs, 33);
    cache->cacheType = ct;
    cache->expander = MakeExpander(bs);

    // Per-build offsets (defaults are MccHaloReach release / U2):
    int OFF_INDEX_POINTER       = 16;
    int OFF_FILE_COUNT          = 704;
    int OFF_FILE_TABLE_POINTER  = 708;
    int OFF_FILE_TABLE_SIZE     = 712;
    int OFF_FILE_TABLE_INDEX    = 716;
    int OFF_VIRTUAL_BASE        = 760;
    int OFF_SECTION_OFFSET_TBL  = 1204;
    int OFF_SECTION_TBL         = 1220;

    if (ct == CacheType::MccHaloReachU3) {
        OFF_FILE_COUNT          = 700;
        OFF_FILE_TABLE_POINTER  = 704;
        OFF_FILE_TABLE_SIZE     = 708;
        OFF_FILE_TABLE_INDEX    = 712;
        OFF_VIRTUAL_BASE        = 752;
        OFF_SECTION_OFFSET_TBL  = 1196;
        OFF_SECTION_TBL         = 1212;
    }

    if (isU8) {
        OFF_INDEX_POINTER       = 744;
        OFF_FILE_COUNT          = 32;
        OFF_FILE_TABLE_POINTER  = 36;
        OFF_FILE_TABLE_SIZE     = 40;
        OFF_FILE_TABLE_INDEX    = 44;
        OFF_VIRTUAL_BASE        = 736;
        OFF_SECTION_OFFSET_TBL  = 1196;
        OFF_SECTION_TBL         = 1212;
        if (ct == CacheType::MccHaloReachU10 || ct == CacheType::MccHaloReachU13) {
            OFF_SECTION_OFFSET_TBL = 1228;
            OFF_SECTION_TBL        = 1244;
        }
    }

    NativeDiag("Parse: ct=%d isU8=%d off-idx=%d off-vbase=%d off-sot=%d off-st=%d",
        (int)ct, (int)isU8, OFF_INDEX_POINTER, OFF_VIRTUAL_BASE, OFF_SECTION_OFFSET_TBL, OFF_SECTION_TBL);

    if (cache->size < (size_t)OFF_SECTION_TBL + 32) {
        NativeDiag("Parse: file too small for section table");
        return false;
    }

    const uint8_t* sot = p + OFF_SECTION_OFFSET_TBL;
    const uint8_t* st  = p + OFF_SECTION_TBL;
    uint32_t s0Addr = RU32(st + 0);
    uint32_t s2Addr = RU32(st + 16);
    uint32_t s0Off  = RU32(sot + 0);
    uint32_t s2Off  = RU32(sot + 8);
    int64_t  vbase  = R64(p + OFF_VIRTUAL_BASE);

    // Mirror Reclaimer's translator math (Reclaimer.Blam/Blam/Common/Gen3/{Section,Tag}AddressTranslator.cs):
    //   SectionAddressTranslator: Magic = (uint)(addr - (addr + off))   = -off (mod 2^32)
    //   TagAddressTranslator:     Magic = (long)(vbase - (uint)(addr + off))
    // The CRITICAL detail is that the inner sum (addr+off) MUST wrap as uint32
    // before being subtracted from vbase. With unwrapped 64-bit arithmetic the
    // tag magic comes out wrong on U13+ maps where addr+off overflows uint32
    // (e.g. forge_halo.map: 0xe3c1000 + 0xfee9a000 wraps to 0xd25b000).
    uint32_t s0SumU32 = (uint32_t)(s0Addr + s0Off);
    uint32_t s2SumU32 = (uint32_t)(s2Addr + s2Off);
    cache->headerMagic = (int64_t)(uint64_t)((uint32_t)(s0Addr - s0SumU32));
    cache->tagMagic    = vbase - (int64_t)(uint64_t)s2SumU32;

    int64_t indexPtrVal = R64(p + OFF_INDEX_POINTER);
    int64_t indexFileOff = TagAddrToFileOff(cache, indexPtrVal);
    NativeDiag("Parse: vbase=0x%llx s0a=0x%x s2a=0x%x s0o=0x%x s2o=0x%x hdrMagic=0x%llx tagMagic=0x%llx idxPtr=0x%llx idxFileOff=%lld",
        (unsigned long long)vbase, s0Addr, s2Addr, s0Off, s2Off,
        (unsigned long long)cache->headerMagic, (unsigned long long)cache->tagMagic,
        (unsigned long long)indexPtrVal, (long long)indexFileOff);
    if (indexFileOff < 0 || (size_t)indexFileOff + 32 > cache->size) {
        NativeDiag("Parse: index pointer out of range");
        return false;
    }

    const uint8_t* idx = p + indexFileOff;

    int32_t tagClassCount  = R32(idx + 0);
    int64_t tagClassDataPtr= R64(idx + 8);
    int32_t tagCount       = R32(idx + 16);
    int64_t tagDataPtr     = R64(idx + 24);

    NativeDiag("Parse: tagClassCount=%d tagCount=%d classPtr=0x%llx tagPtr=0x%llx",
        tagClassCount, tagCount,
        (unsigned long long)tagClassDataPtr, (unsigned long long)tagDataPtr);

    if (tagClassCount < 0 || tagClassCount > 0x10000) { NativeDiag("Parse: bad tagClassCount"); return false; }
    if (tagCount < 0 || tagCount > 0x80000) { NativeDiag("Parse: bad tagCount"); return false; }

    int64_t classDataAddr = TagAddrToFileOff(cache, tagClassDataPtr);
    int64_t tagDataAddr   = TagAddrToFileOff(cache, tagDataPtr);
    if (classDataAddr < 0 || (size_t)classDataAddr + (size_t)tagClassCount * 16 > cache->size) {
        NativeDiag("Parse: class data range bad classDataAddr=%lld", (long long)classDataAddr);
        return false;
    }
    if (tagDataAddr   < 0 || (size_t)tagDataAddr   + (size_t)tagCount      * 8   > cache->size) {
        NativeDiag("Parse: tag data range bad tagDataAddr=%lld", (long long)tagDataAddr);
        return false;
    }

    cache->classes.resize(tagClassCount);
    for (int i = 0; i < tagClassCount; ++i) {
        const uint8_t* c = p + classDataAddr + i * 16;
        cache->classes[i].classId = RU32(c);
        cache->classes[i].classCode[0] = c[3];
        cache->classes[i].classCode[1] = c[2];
        cache->classes[i].classCode[2] = c[1];
        cache->classes[i].classCode[3] = c[0];
        cache->classes[i].classCode[4] = 0;
    }

    cache->tags.resize(tagCount);
    for (int i = 0; i < tagCount; ++i) {
        const uint8_t* t = p + tagDataAddr + i * 8;
        int16_t classIndex = R16(t + 0);
        uint32_t metaRaw   = RU32(t + 4);
        cache->tags[i].classIndex = classIndex;
        cache->tags[i].metaPointerRaw = metaRaw;
        if (classIndex >= 0 && classIndex < (int32_t)cache->classes.size())
            memcpy(cache->tags[i].classCode, cache->classes[classIndex].classCode, 5);
        else
            cache->tags[i].classCode[0] = 0;
    }

    // String-table parse. Three layouts depending on cache version:
    //
    //   MccHaloReach (release/U2): Reclaimer's MccHaloReach.CacheFile.cs:119-133
    //     348 StringCount, 352 Size, 356 IndexPtr, 360 BlobPtr
    //
    //   MccHaloReachU3:  same struct as release but offsets shifted -12
    //     336 StringCount, 340 Size, 344 IndexPtr, 348 BlobPtr
    //
    //   MccHaloReachU8/U10/U13: completely different - `CacheFileU8.cs:75-85`
    //     48 StringCount, 52 BlobPtr, 56 Size, 60 IndexPtr
    //     (note the field order is also swapped: blob pointer comes BEFORE
    //      size, not after)
    //
    // All four pointers are header-space (subtract headerMagic via
    // HdrAddrToFileOff). Plaintext UTF-8.
    {
        bool isU8plus = (ct == CacheType::MccHaloReachU8  ||
                         ct == CacheType::MccHaloReachU10 ||
                         ct == CacheType::MccHaloReachU13);
        int OFF_STR_COUNT, OFF_STR_BLOB_SIZE, OFF_STR_IDX_PTR, OFF_STR_BLOB_PTR;
        if (isU8plus) {
            // CacheFileU8 layout - Reclaimer.Blam/MccHaloReach/CacheFileU8.cs
            OFF_STR_COUNT     = 48;
            OFF_STR_BLOB_PTR  = 52;
            OFF_STR_BLOB_SIZE = 56;
            OFF_STR_IDX_PTR   = 60;
        } else if (ct == CacheType::MccHaloReach) {
            // Release / U2.
            OFF_STR_COUNT     = 348;
            OFF_STR_BLOB_SIZE = 352;
            OFF_STR_IDX_PTR   = 356;
            OFF_STR_BLOB_PTR  = 360;
        } else {
            // MccHaloReachU3 (and U7).
            OFF_STR_COUNT     = 336;
            OFF_STR_BLOB_SIZE = 340;
            OFF_STR_IDX_PTR   = 344;
            OFF_STR_BLOB_PTR  = 348;
        }
        if ((size_t)OFF_STR_BLOB_PTR + 4 <= cache->size) {
            int32_t  strCount    = R32 (p + OFF_STR_COUNT);
            int32_t  strBlobSize = R32 (p + OFF_STR_BLOB_SIZE);
            uint32_t strIdxPtr   = RU32(p + OFF_STR_IDX_PTR);
            uint32_t strBlobPtr  = RU32(p + OFF_STR_BLOB_PTR);
            if (strCount > 0 && strCount <= 0x80000 &&
                strBlobSize > 0 && strBlobSize <= 0x4000000)
            {
                int64_t idxOff  = HdrAddrToFileOff(cache, (int64_t)(uint64_t)strIdxPtr);
                int64_t blobOff = HdrAddrToFileOff(cache, (int64_t)(uint64_t)strBlobPtr);
                if (idxOff  >= 0 && (size_t)idxOff  + (size_t)strCount * 4    <= cache->size &&
                    blobOff >= 0 && (size_t)blobOff + (size_t)strBlobSize     <= cache->size)
                {
                    cache->stringCount = strCount;
                    cache->stringIndices.resize(strCount);
                    memcpy(cache->stringIndices.data(), p + idxOff, (size_t)strCount * 4);
                    cache->stringBlob.assign(p + blobOff, p + blobOff + strBlobSize);
                    cache->stringTableParsed = true;
                    NativeDiag("Parse: stringTable count=%d blobSize=%d", strCount, strBlobSize);

                    // Build the StringId namespace translator. Bit widths from
                    // Reclaimer's MccHaloReachStrings.xml: U8/U10 use 17/8/7,
                    // U13 uses 19/8/5. The namespace table lives in the cache
                    // header at offset 64 (count) / 68 (pointer) for U8+.
                    if (isU8plus) {
                        cache->sidIndexBits     = (ct == CacheType::MccHaloReachU13) ? 19 : 17;
                        cache->sidNamespaceBits = 8;
                        int32_t  nsCount  = R32 (p + 64);
                        uint32_t nsPtrRaw = RU32(p + 68);
                        if (nsCount > 1 && nsCount <= 64) {
                            int64_t nsOff = HdrAddrToFileOff(cache, (int64_t)(uint64_t)nsPtrRaw);
                            if (nsOff >= 0 && (size_t)nsOff + (size_t)nsCount * 4 <= cache->size) {
                                std::vector<int32_t> nsArr(nsCount);
                                memcpy(nsArr.data(), p + nsOff, (size_t)nsCount * 4);
                                int32_t mask = (1 << cache->sidIndexBits) - 1;
                                int32_t start = nsArr[0] & mask;
                                cache->sidNamespaces.clear();
                                for (int i = 1; i < nsCount; ++i) {
                                    cache->sidNamespaces.push_back({ i, 0, start });
                                    start += nsArr[i] & mask;
                                }
                                cache->sidNamespaces.push_back({ 0, nsArr[0] & mask, start });
                                cache->stringIdTranslatorReady = true;
                                NativeDiag("Parse: sidTranslator nsCount=%d indexBits=%d ns0Min=%d ns0Start=%d",
                                    nsCount, cache->sidIndexBits,
                                    nsArr[0] & mask, start);
                            }
                        }
                    }
                } else {
                    NativeDiag("Parse: stringTable bounds bad idxOff=%lld blobOff=%lld",
                        (long long)idxOff, (long long)blobOff);
                }
            } else {
                NativeDiag("Parse: stringTable header bogus count=%d size=%d", strCount, strBlobSize);
            }
        }
    }

    int32_t  fileCount = R32(p + OFF_FILE_COUNT);
    uint32_t ftPtr     = RU32(p + OFF_FILE_TABLE_POINTER);
    int32_t  ftSize    = R32(p + OFF_FILE_TABLE_SIZE);
    uint32_t ftIdxPtr  = RU32(p + OFF_FILE_TABLE_INDEX);
    if (fileCount > 0 && ftSize > 0) {
        // Reclaimer reads the CacheHeader WITHOUT a registered PointerExpander
        // (see MccHaloReach.CacheFile.cs `using (var reader = CreateReader(HeaderTranslator))`),
        // so the FileTablePointer / StringTablePointer values are NOT expanded
        // even though they are typed as `Pointer`. They flow straight into
        // SectionAddressTranslator(0).GetAddress(value) = value - headerMagic.
        int64_t ftAddr  = HdrAddrToFileOff(cache, (int64_t)(uint64_t)ftPtr);
        int64_t ftIdxAd = HdrAddrToFileOff(cache, (int64_t)(uint64_t)ftIdxPtr);
        if (ftAddr  >= 0 && (size_t)ftAddr  + (size_t)ftSize           <= cache->size &&
            ftIdxAd >= 0 && (size_t)ftIdxAd + (size_t)fileCount * 4    <= cache->size)
        {
            const uint8_t* base = p + ftAddr;
            const uint8_t* idxArr = p + ftIdxAd;
            int loop = fileCount < (int32_t)cache->tags.size() ? fileCount : (int32_t)cache->tags.size();
            for (int i = 0; i < loop; ++i) {
                int32_t off = R32(idxArr + i * 4);
                if (off < 0 || off >= ftSize) continue;
                const char* str = reinterpret_cast<const char*>(base + off);
                size_t maxLen = (size_t)ftSize - (size_t)off;
                size_t len = strnlen_s(str, maxLen);
                cache->tags[i].tagName.assign(str, len);
            }
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// Shared-cache header parse (resource-only caches: shared.map, etc.)
// -----------------------------------------------------------------------------
//
// shared.map and its siblings are pure resource containers - no tag index, no
// virtual base, no usable section table. The header at offset 0 still starts
// with 'daeh' (or 'head') and the build string at +160 is identical to the
// primary cache, but everything from +704 (file table) through the section
// table at +1244 is either zero or scrambled. The ONLY field we need is the
// dataTableAddress, which Reclaimer reads from a build-specific fixed offset
// without parsing anything else (see HaloReach\ResourceIdentifier.cs:
// GetDataTableAddress).
//
// Verified empirically against haloreach\maps\shared.map (build "Jun 21 2023
// 15:35:31", 363,823,104 bytes):
//   * 'daeh' magic OK at +0
//   * build string OK at +160
//   * @1232 = 0xa000  (dataTableAddress)
//   * 0xa000 + section[1].size 0x15aee000 = 0x15af8000 = exact file size
// So the data section runs from 0xa000 to EOF, and page.dataOffset is added
// directly to dataTableAddress to find the LZX-compressed segment.
bool ParseSharedCacheHeader(CacheHandle* cache) {
    if (cache->size < 1300) {
        NativeDiag("ParseShared: size<1300 size=%llu",
            (unsigned long long)cache->size);
        return false;
    }
    const uint8_t* p = cache->base;
    bool magicHead = (p[0] == 'h' && p[1] == 'e' && p[2] == 'a' && p[3] == 'd');
    bool magicDaeh = (p[0] == 'd' && p[1] == 'a' && p[2] == 'e' && p[3] == 'h');
    if (!magicHead && !magicDaeh) {
        NativeDiag("ParseShared: magic mismatch %02x %02x %02x %02x",
            p[0], p[1], p[2], p[3]);
        return false;
    }

    // Build detection: prefer offset 160 (U8+ MCC), fall back to 288 (older).
    char bs[33] = {0};
    memcpy(bs, p + 160, 32); bs[32] = 0;
    CacheType ct = MatchBuild(bs);
    if (ct == CacheType::Unknown) {
        char bs2[33] = {0};
        memcpy(bs2, p + 288, 32); bs2[32] = 0;
        CacheType ct2 = MatchBuild(bs2);
        if (ct2 != CacheType::Unknown) {
            memcpy(bs, bs2, 33);
            ct = ct2;
        }
    }
    NativeDiag("ParseShared: magic-ok %s build='%s' ct=%d",
        magicDaeh ? "daeh" : "head", bs, (int)ct);

    memcpy(cache->buildString, bs, 33);
    cache->cacheType = ct;
    cache->expander = MakeExpander(bs);

    // dataTableAddress lives at a build-specific fixed offset. Mirror
    // Reclaimer's GetDataTableAddress switch in HaloReach\ResourceIdentifier.cs.
    size_t dtaOff = 1208;  // early MCC default
    switch (ct) {
        case CacheType::MccHaloReachU10:
        case CacheType::MccHaloReachU13:
            dtaOff = 1232;
            break;
        case CacheType::MccHaloReachU3:
        case CacheType::MccHaloReachU8:
            dtaOff = 1200;
            break;
        case CacheType::MccHaloReach:
        default:
            dtaOff = 1208;
            break;
    }
    if (dtaOff + 4 > cache->size) {
        NativeDiag("ParseShared: file too small for dta@%llu",
            (unsigned long long)dtaOff);
        return false;
    }
    cache->dataTableAddress = RU32(p + dtaOff);
    NativeDiag("ParseShared: dta@%llu = 0x%x size=%llu",
        (unsigned long long)dtaOff, cache->dataTableAddress,
        (unsigned long long)cache->size);

    // Sanity: dta should be > 0 and < file size. A fully-zero value here
    // means we picked the wrong offset (or this isn't actually a shared
    // cache for this build).
    if (cache->dataTableAddress == 0 ||
        (size_t)cache->dataTableAddress >= cache->size) {
        NativeDiag("ParseShared: implausible dta=0x%x size=%llu",
            cache->dataTableAddress, (unsigned long long)cache->size);
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------------
// TagBlock + globals
// -----------------------------------------------------------------------------

TagBlockRef ReadTagBlock(const uint8_t* p) {
    TagBlockRef r{ R32(p), RU32(p + 4) };
    return r;
}

int FindGlobalTag(CacheHandle* cache, const char* classCode4) {
    for (size_t i = 0; i < cache->tags.size(); ++i) {
        const auto& t = cache->tags[i];
        if (t.classIndex < 0) continue;
        if (memcmp(t.classCode, classCode4, 4) == 0)
            return (int)i;
    }
    return -1;
}

const char* ResolveStringId(CacheHandle* cache, int32_t stringId) {
    if (!cache || !cache->stringTableParsed) return nullptr;
    if (stringId == 0 || (uint32_t)stringId == 0xFFFFFFFFu) return nullptr;

    int32_t idx;
    if (cache->stringIdTranslatorReady) {
        // Reclaimer-spec translation (StringIdTranslator.GetStringIndex).
        int idxBits = cache->sidIndexBits;
        int nsBits  = cache->sidNamespaceBits;
        int indexMask = (1 << idxBits) - 1;
        int nsMask    = (1 << nsBits) - 1;
        int32_t lowIdx = stringId & indexMask;
        int     nsId   = (stringId >> idxBits) & nsMask;

        // Find the namespace. Fall back through nsId-1, nsId-2, ... if the
        // exact id isn't registered (matches Reclaimer's `while (!contains(id) && id > 0) id--`).
        const CacheHandle::SidNamespace* ns = nullptr;
        for (int probe = nsId; probe >= 0; --probe) {
            for (const auto& candidate : cache->sidNamespaces) {
                if (candidate.id == probe) { ns = &candidate; break; }
            }
            if (ns) break;
        }
        if (!ns) return nullptr;
        idx = (lowIdx < ns->min) ? lowIdx : (lowIdx - ns->min + ns->start);
    } else {
        // No translator available - fall back to low-16 mask (the old broken
        // path, kept so older builds with a single-namespace string table
        // still produce some output).
        idx = stringId & 0xFFFF;
    }

    if (idx < 0 || idx >= cache->stringCount) return nullptr;
    int32_t off = cache->stringIndices[idx];
    if (off < 0 || (size_t)off >= cache->stringBlob.size()) return nullptr;
    return reinterpret_cast<const char*>(cache->stringBlob.data() + off);
}

// -----------------------------------------------------------------------------
// Resource gestalt parse (zone tag)
// -----------------------------------------------------------------------------

bool ParseGestalt(CacheHandle* cache) {
    if (cache->gestaltParsed) return true;
    int zoneIdx = FindGlobalTag(cache, "zone");
    if (zoneIdx < 0) { NativeDiag("Gestalt: no zone tag"); return false; }
    int64_t metaOff = TagMetaFileOff(cache, cache->tags[zoneIdx].metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + 350 > cache->size) {
        NativeDiag("Gestalt: bad zone meta off=%lld raw=0x%x size=%llu",
            (long long)metaOff, cache->tags[zoneIdx].metaPointerRaw,
            (unsigned long long)cache->size);
        return false;
    }
    const uint8_t* meta = cache->base + metaOff;

    constexpr int OFF_RES_ENTRIES = 100;

    TagBlockRef resBlock = ReadTagBlock(meta + OFF_RES_ENTRIES);
    if (resBlock.count < 0 || resBlock.count > 0x100000) {
        NativeDiag("Gestalt: bad resBlock count=%d ptr=0x%x", resBlock.count, resBlock.pointer);
        return false;
    }

    int64_t entriesOff = TagMetaFileOff(cache, resBlock.pointer);
    constexpr int RES_ENTRY_SIZE = 64;
    if (entriesOff < 0 ||
        (size_t)entriesOff + (size_t)resBlock.count * RES_ENTRY_SIZE > cache->size) {
        NativeDiag("Gestalt: bad entries range off=%lld count=%d ptr=0x%x size=%llu",
            (long long)entriesOff, resBlock.count, resBlock.pointer,
            (unsigned long long)cache->size);
        return false;
    }
    NativeDiag("Gestalt: zoneIdx=%d metaOff=%lld resCount=%d entriesOff=%lld",
        zoneIdx, (long long)metaOff, resBlock.count, (long long)entriesOff);

    cache->resourceEntries.resize(resBlock.count);
    for (int i = 0; i < resBlock.count; ++i) {
        const uint8_t* r = cache->base + entriesOff + i * RES_ENTRY_SIZE;
        ResourceEntry& e = cache->resourceEntries[i];
        e.resourcePointer = R32(r + 16);
        e.fixupOffset     = R32(r + 20);
        e.fixupSize       = R32(r + 24);
        e.segmentIndex    = R16(r + 34);

        // ResourceFixups block lives at +40 (count + Pointer). We DON'T load
        // the fixups eagerly - only when the model parser requests them.
        TagBlockRef fb = ReadTagBlock(r + 40);
        e.fixupsBlockCount   = fb.count;
        e.fixupsBlockPointer = fb.pointer;
    }
    cache->gestaltParsed = true;
    return true;
}

bool EnsureResourceFixups(CacheHandle* cache, size_t entryIndex) {
    if (entryIndex >= cache->resourceEntries.size()) {
        NativeDiag("Fixups: entryIndex=%llu out of range count=%llu",
            (unsigned long long)entryIndex,
            (unsigned long long)cache->resourceEntries.size());
        return false;
    }
    ResourceEntry& e = cache->resourceEntries[entryIndex];
    if (e.fixupsLoaded) return true;
    if (e.fixupsBlockCount <= 0) {
        e.fixupsLoaded = true;
        return true;
    }
    if (e.fixupsBlockCount > 0x100000) {
        NativeDiag("Fixups: bad blockCount=%d entry=%llu",
            e.fixupsBlockCount, (unsigned long long)entryIndex);
        return false;
    }

    int64_t off = TagMetaFileOff(cache, e.fixupsBlockPointer);
    constexpr int FIXUP_SIZE = 8;
    if (off < 0 ||
        (size_t)off + (size_t)e.fixupsBlockCount * FIXUP_SIZE > cache->size) {
        NativeDiag("Fixups: bad range entry=%llu blockPtr=0x%x count=%d off=%lld",
            (unsigned long long)entryIndex, e.fixupsBlockPointer,
            e.fixupsBlockCount, (long long)off);
        return false;
    }

    e.fixups.resize(e.fixupsBlockCount);
    for (int i = 0; i < e.fixupsBlockCount; ++i) {
        const uint8_t* fp = cache->base + off + i * FIXUP_SIZE;
        e.fixups[i].unknown = R32(fp);
        e.fixups[i].offset  = R32(fp + 4);
    }
    e.fixupsLoaded = true;
    return true;
}

// -----------------------------------------------------------------------------
// Resource layout-table parse (play tag)
// -----------------------------------------------------------------------------

bool ParseLayoutTable(CacheHandle* cache) {
    // LAYOUT_RACE_FIX: serialize concurrent first-call work.
    // Without this, multiple mesh-tag parser threads racing through
    // ZH_MMP_OpenModel -> ReadResourceData -> ParseLayoutTable simultaneously
    // resize() the same vectors, freeing in-flight buffers and tripping
    // STATUS_HEAP_CORRUPTION (0xC0000374) in RtlFreeHeap. Crash dump trace:
    //   ParseLayoutTable+0x2C8 -> vector::_Resize_reallocate -> ucrtbase!free.
    // Threads after the first see layoutParsed=true under the lock and
    // return immediately - no measurable steady-state cost.
    std::lock_guard<std::mutex> lock(cache->layoutMutex);
    if (cache->layoutParsed) return true;

    constexpr int OFF_SHARED_CACHES = 12;
    constexpr int OFF_PAGES         = 24;
    constexpr int OFF_SEGMENTS      = 60;
    constexpr int SHARED_CACHE_SIZE = 264;
    constexpr int PAGE_SIZE         = 88;
    constexpr int SEGMENT_SIZE      = 16;

    bool gotPagesAndSegments = false;
    TagBlockRef sharedBlock{};
    sharedBlock.count = 0;
    sharedBlock.pointer = 0;

    // --- Primary path: read pages+segments from the play tag ---
    int playIdx = FindGlobalTag(cache, "play");
    if (playIdx >= 0 && cache->tags[playIdx].metaPointerRaw != 0) {
        int64_t metaOff = TagMetaFileOff(cache, cache->tags[playIdx].metaPointerRaw);
        if (metaOff >= 0 && (size_t)metaOff + 80 <= cache->size) {
            const uint8_t* meta = cache->base + metaOff;

            sharedBlock = ReadTagBlock(meta + OFF_SHARED_CACHES);
            TagBlockRef pageBlock    = ReadTagBlock(meta + OFF_PAGES);
            TagBlockRef segmentBlock = ReadTagBlock(meta + OFF_SEGMENTS);
            if (sharedBlock.count < 0 || sharedBlock.count > 0x100) {
                NativeDiag("Layout: bad sharedBlock count=%d", sharedBlock.count);
                sharedBlock.count = 0;
                sharedBlock.pointer = 0;
            }

            if (pageBlock.count > 0 && pageBlock.count <= 0x100000 &&
                segmentBlock.count > 0 && segmentBlock.count <= 0x100000) {
                int64_t pagesOff = TagMetaFileOff(cache, pageBlock.pointer);
                int64_t segOff   = TagMetaFileOff(cache, segmentBlock.pointer);
                if (pagesOff >= 0 && (size_t)pagesOff + (size_t)pageBlock.count * PAGE_SIZE <= cache->size &&
                    segOff   >= 0 && (size_t)segOff   + (size_t)segmentBlock.count * SEGMENT_SIZE <= cache->size) {

                    NativeDiag("Layout: play-tag pages=%d segments=%d pagesOff=%lld segOff=%lld",
                        pageBlock.count, segmentBlock.count, (long long)pagesOff, (long long)segOff);

                    cache->pages.resize(pageBlock.count);
                    for (int i = 0; i < pageBlock.count; ++i) {
                        const uint8_t* p = cache->base + pagesOff + i * PAGE_SIZE;
                        PageBlock& pb = cache->pages[i];
                        pb.cacheIndex       = R16(p + 4);
                        pb.dataOffset       = R32(p + 8);
                        pb.compressedSize   = R32(p + 12);
                        pb.decompressedSize = R32(p + 16);
                    }
                    cache->segments.resize(segmentBlock.count);
                    for (int i = 0; i < segmentBlock.count; ++i) {
                        const uint8_t* s = cache->base + segOff + i * SEGMENT_SIZE;
                        SegmentBlock& sb = cache->segments[i];
                        sb.primaryPageIndex     = R16(s + 0);
                        sb.secondaryPageIndex   = R16(s + 2);
                        sb.primaryPageOffset    = R32(s + 4);
                        sb.secondaryPageOffset  = R32(s + 8);
                    }
                    gotPagesAndSegments = true;
                }
            }
        }
        if (!gotPagesAndSegments) {
            NativeDiag("Layout: play tag meta unusable off=%lld raw=0x%x",
                (long long)TagMetaFileOff(cache, cache->tags[playIdx].metaPointerRaw),
                cache->tags[playIdx].metaPointerRaw);
        }
    }

    // --- Fallback: HREK/workshop maps embed pages+segments in the zone tag ---
    // The play tag in these maps has metaPointerRaw=0. Instead, the zone tag
    // (ResourceGestalt) carries the same page/segment arrays at:
    //   zone+52: TagBlock Pages   (88 bytes/entry)
    //   zone+88: TagBlock Segments (16 bytes/entry)
    // Page dataOffsets are absolute file offsets (dataTableAddress=0).
    if (!gotPagesAndSegments) {
        int zoneIdx = FindGlobalTag(cache, "zone");
        if (zoneIdx >= 0 && cache->tags[zoneIdx].metaPointerRaw != 0) {
            int64_t zoneMetaOff = TagMetaFileOff(cache, cache->tags[zoneIdx].metaPointerRaw);
            if (zoneMetaOff >= 0 && (size_t)zoneMetaOff + 112 <= cache->size) {
                const uint8_t* zoneMeta = cache->base + zoneMetaOff;
                constexpr int ZONE_OFF_PAGES    = 52;
                constexpr int ZONE_OFF_SEGMENTS = 88;

                TagBlockRef pageBlock    = ReadTagBlock(zoneMeta + ZONE_OFF_PAGES);
                TagBlockRef segmentBlock = ReadTagBlock(zoneMeta + ZONE_OFF_SEGMENTS);

                NativeDiag("Layout: zone-fallback attempt pageCount=%d segCount=%d",
                    pageBlock.count, segmentBlock.count);

                if (pageBlock.count > 0 && pageBlock.count <= 0x100000 &&
                    segmentBlock.count > 0 && segmentBlock.count <= 0x100000) {
                    int64_t pagesOff = TagMetaFileOff(cache, pageBlock.pointer);
                    int64_t segOff   = TagMetaFileOff(cache, segmentBlock.pointer);
                    if (pagesOff >= 0 && (size_t)pagesOff + (size_t)pageBlock.count * PAGE_SIZE <= cache->size &&
                        segOff   >= 0 && (size_t)segOff   + (size_t)segmentBlock.count * SEGMENT_SIZE <= cache->size) {

                        cache->pages.resize(pageBlock.count);
                        for (int i = 0; i < pageBlock.count; ++i) {
                            const uint8_t* p = cache->base + pagesOff + i * PAGE_SIZE;
                            PageBlock& pb = cache->pages[i];
                            pb.cacheIndex       = R16(p + 4);
                            pb.dataOffset       = R32(p + 8);
                            pb.compressedSize   = R32(p + 12);
                            pb.decompressedSize = R32(p + 16);
                        }
                        cache->segments.resize(segmentBlock.count);
                        for (int i = 0; i < segmentBlock.count; ++i) {
                            const uint8_t* s = cache->base + segOff + i * SEGMENT_SIZE;
                            SegmentBlock& sb = cache->segments[i];
                            sb.primaryPageIndex     = R16(s + 0);
                            sb.secondaryPageIndex   = R16(s + 2);
                            sb.primaryPageOffset    = R32(s + 4);
                            sb.secondaryPageOffset  = R32(s + 8);
                        }
                        gotPagesAndSegments = true;
                        NativeDiag("Layout: zone-fallback OK pages=%d segments=%d",
                            pageBlock.count, segmentBlock.count);
                    }
                }
            }
        }
    }

    if (!gotPagesAndSegments) {
        NativeDiag("Layout: no pages/segments from play or zone tag");
        return false;
    }

    // Parse SharedCaches[] (play-tag only; HREK maps are self-contained).
    if (sharedBlock.count > 0) {
        int64_t sharedOff = TagMetaFileOff(cache, sharedBlock.pointer);
        if (sharedOff < 0 ||
            (size_t)sharedOff + (size_t)sharedBlock.count * SHARED_CACHE_SIZE > cache->size) {
            NativeDiag("Layout: bad sharedCaches range off=%lld count=%d ptr=0x%x",
                (long long)sharedOff, sharedBlock.count, sharedBlock.pointer);
        } else {
            cache->sharedCacheNames.reserve(sharedBlock.count);
            for (int i = 0; i < sharedBlock.count; ++i) {
                const uint8_t* sc = cache->base + sharedOff + i * SHARED_CACHE_SIZE;
                char raw[33] = {0};
                memcpy(raw, sc, 32);
                raw[32] = 0;
                const char* base = raw;
                for (const char* q = raw; *q; ++q) {
                    if (*q == '\\' || *q == '/') base = q + 1;
                }
                cache->sharedCacheNames.emplace_back(base);
                NativeDiag("Layout: sharedCache[%d]='%s' raw='%s'",
                    i, cache->sharedCacheNames.back().c_str(), raw);
            }
            cache->sharedCaches.assign(sharedBlock.count, nullptr);
            cache->sharedTried.assign(sharedBlock.count, false);
        }
    }

    size_t dtaOff = 1208;
    switch (cache->cacheType) {
        case CacheType::MccHaloReachU10:
        case CacheType::MccHaloReachU13:
            dtaOff = 1232;
            break;
        case CacheType::MccHaloReachU3:
        case CacheType::MccHaloReachU8:
            dtaOff = 1200;
            break;
        default:
            dtaOff = 1208;
            break;
    }
    if (dtaOff + 4 <= cache->size)
        cache->dataTableAddress = RU32(cache->base + dtaOff);
    cache->layoutParsed = true;
    return true;
}

// -----------------------------------------------------------------------------
// Shared-cache lazy opener
// -----------------------------------------------------------------------------
//
// AcquireSharedHandleInternal mirrors the public AcquireCacheHandle but skips
// inserting into the global handle table - shared caches are owned by the
// parent CacheHandle and freed in ReleaseCacheHandle (recursively). It also
// sets isSharedChild=true so the (eventual) recursive-discovery hook stays
// off - even though shared caches do have their own cache_file_resource_*
// tags, we never need to walk them.
//
// Forward decls.
namespace { CacheHandle* OpenSharedHandle(const wchar_t* path); }

// Returns nullptr if the index is out-of-range, the open already failed, or
// the open fails on this attempt. Must be called with cache->pageRouteMutex held
// (so the per-shared open + cache populate is atomic).
static CacheHandle* GetOrOpenSharedCacheLocked(CacheHandle* cache, int sharedIdx)
{
    if (sharedIdx < 0 || sharedIdx >= (int)cache->sharedCaches.size()) {
        NativeDiag("Shared: cacheIndex=%d out of range count=%llu",
            sharedIdx, (unsigned long long)cache->sharedCaches.size());
        return nullptr;
    }
    if (cache->sharedCaches[sharedIdx]) return cache->sharedCaches[sharedIdx];
    if (cache->sharedTried[sharedIdx]) return nullptr;
    cache->sharedTried[sharedIdx] = true;

    // Build the sibling path: parent_dir + '\\' + sharedCacheNames[sharedIdx].
    std::wstring parentDir = cache->path;
    size_t slash = parentDir.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        NativeDiag("Shared: primary path has no directory separator '%ls'", cache->path.c_str());
        return nullptr;
    }
    parentDir.resize(slash + 1);

    // Convert UTF-8 sharedCacheNames[i] -> wide. Filenames here are ASCII
    // ("shared.map", "campaign.map") so a byte-cast suffices, but go through
    // MultiByteToWideChar for safety in case any path has high bytes.
    const std::string& nameUtf8 = cache->sharedCacheNames[sharedIdx];
    int wlen = MultiByteToWideChar(CP_UTF8, 0, nameUtf8.c_str(), (int)nameUtf8.size(),
                                   nullptr, 0);
    std::wstring nameW;
    if (wlen > 0) {
        nameW.resize(wlen);
        MultiByteToWideChar(CP_UTF8, 0, nameUtf8.c_str(), (int)nameUtf8.size(),
                            nameW.data(), wlen);
    } else {
        nameW.assign(nameUtf8.begin(), nameUtf8.end());
    }
    std::wstring fullPath = parentDir + nameW;

    NativeDiag("Shared: opening idx=%d name='%s' fullPath='%ls'",
        sharedIdx, nameUtf8.c_str(), fullPath.c_str());

    CacheHandle* h = OpenSharedHandle(fullPath.c_str());
    if (!h) {
        NativeDiag("Shared: open FAILED idx=%d name='%s'",
            sharedIdx, nameUtf8.c_str());
        return nullptr;
    }
    // dataTableAddress is already populated by ParseSharedCacheHeader inside
    // OpenSharedHandle, using the same per-build offset Reclaimer uses
    // (1232 for U10/U13, 1200 for U3-U8, 1208 for early MCC).
    NativeDiag("Shared: open OK idx=%d name='%s' size=%llu cacheType=%d dta=0x%x",
        sharedIdx, nameUtf8.c_str(),
        (unsigned long long)h->size, (int)h->cacheType, h->dataTableAddress);

    cache->sharedCaches[sharedIdx] = h;
    return h;
}

// -----------------------------------------------------------------------------
// Resource data reader
// -----------------------------------------------------------------------------

// Page-cache profiling counters (read+reset via ZH_MBP_GetPageStats). Ground-truth for
// the .mvar load-speed work - the append-mode text log is unreliable across runs.
static std::atomic<uint64_t> g_pcHits{0};
static std::atomic<uint64_t> g_pcInflates{0};
static std::atomic<uint64_t> g_pcInflateBytes{0};
static std::atomic<uint64_t> g_pcInflateNs{0};
static std::atomic<uint64_t> g_pcStores{0};
extern "C" __declspec(dllexport) void __stdcall ZH_MBP_GetPageStats(
    uint64_t* hits, uint64_t* inflates, uint64_t* inflateBytes, uint64_t* inflateNs, uint64_t* stores)
{
    if (hits) *hits = g_pcHits.exchange(0);
    if (inflates) *inflates = g_pcInflates.exchange(0);
    if (inflateBytes) *inflateBytes = g_pcInflateBytes.exchange(0);
    if (inflateNs) *inflateNs = g_pcInflateNs.exchange(0);
    if (stores) *stores = g_pcStores.exchange(0);
}

uint8_t* ReadResourceData(CacheHandle* cache, int resourceIdValue,
                          size_t maxLength, size_t* outSize)
{
    return ReadResourceDataPage(cache, resourceIdValue, maxLength, outSize, 0);
}

// which: 0 = auto (secondary page when present, else primary - the historical behaviour),
//        1 = primary page only, 2 = secondary page only (nullptr when the segment has none).
// Bitmaps keep their TOP mips in the secondary (high-res) page and the LOWER mips in the
// primary page, so a full mip chain needs both (see GetRawDDSInner).
static std::atomic<uint64_t> g_rrCalls{0}, g_rrParse{0}, g_rrRoute{0}, g_rrLockWait{0}, g_rrCopy{0}, g_rrMiss{0};
struct RrProfReporter { ~RrProfReporter() { if (!getenv("MMS_DECODE_PROF")) return;
    fprintf(stderr, "MMS_READRES_PROF calls=%llu parse=%.0f route=%.0f lockwait=%.0f copy=%.0f misspath=%.0f ms\n",
        (unsigned long long)g_rrCalls.load(), g_rrParse.load()/1e6, g_rrRoute.load()/1e6, g_rrLockWait.load()/1e6, g_rrCopy.load()/1e6, g_rrMiss.load()/1e6); } };
static RrProfReporter g_rrReporter;
static inline uint64_t RrNs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); }

uint8_t* ReadResourceDataPage(CacheHandle* cache, int resourceIdValue,
                              size_t maxLength, size_t* outSize, int which)
{
    *outSize = 0;
    g_rrCalls.fetch_add(1, std::memory_order_relaxed);
    auto _rr0 = std::chrono::steady_clock::now();
    if (!ParseGestalt(cache)) { NativeDiag("ReadRes: ParseGestalt fail"); return nullptr; }
    if (!ParseLayoutTable(cache)) { NativeDiag("ReadRes: ParseLayoutTable fail"); return nullptr; }
    auto _rr1 = std::chrono::steady_clock::now(); g_rrParse.fetch_add(RrNs(_rr0, _rr1), std::memory_order_relaxed);

    int resourceIndex = resourceIdValue & 0xFFFF;
    if (resourceIndex < 0 || resourceIndex >= (int)cache->resourceEntries.size()) {
        NativeDiag("ReadRes: bad resourceIndex=%d count=%llu rid=0x%x",
            resourceIndex, (unsigned long long)cache->resourceEntries.size(), resourceIdValue);
        return nullptr;
    }
    const ResourceEntry& entry = cache->resourceEntries[resourceIndex];
    if (entry.segmentIndex < 0 ||
        entry.segmentIndex >= (int)cache->segments.size()) {
        NativeDiag("ReadRes: bad segmentIndex=%d segCount=%llu rid=0x%x",
            entry.segmentIndex, (unsigned long long)cache->segments.size(), resourceIdValue);
        return nullptr;
    }

    const SegmentBlock& seg = cache->segments[entry.segmentIndex];
    if (which == 2 && seg.secondaryPageIndex < 0) return nullptr;
    bool useSecondary = (which == 2) || (which == 0 && seg.secondaryPageIndex >= 0);
    int pageIndex     = useSecondary ? seg.secondaryPageIndex : seg.primaryPageIndex;
    int segmentOffset = useSecondary ? seg.secondaryPageOffset : seg.primaryPageOffset;
    if (pageIndex < 0 || pageIndex >= (int)cache->pages.size() || segmentOffset < 0) {
        NativeDiag("ReadRes: bad page sel rid=0x%x pageIdx=%d pageCnt=%llu segOff=%d useSecondary=%d",
            resourceIdValue, pageIndex, (unsigned long long)cache->pages.size(),
            segmentOffset, (int)useSecondary);
        return nullptr;
    }

    PageBlock page = cache->pages[pageIndex];
    if (page.dataOffset < 0 || page.compressedSize <= 0) {
        if (which == 2) return nullptr;
        pageIndex     = seg.primaryPageIndex;
        segmentOffset = seg.primaryPageOffset;
        if (pageIndex < 0 || pageIndex >= (int)cache->pages.size() || segmentOffset < 0) {
            NativeDiag("ReadRes: fallback page bad rid=0x%x pageIdx=%d segOff=%d",
                resourceIdValue, pageIndex, segmentOffset);
            return nullptr;
        }
        page = cache->pages[pageIndex];
        if (page.dataOffset < 0 || page.compressedSize <= 0) {
            NativeDiag("ReadRes: fallback page empty rid=0x%x dataOff=%d csz=%d",
                resourceIdValue, page.dataOffset, page.compressedSize);
            return nullptr;
        }
    }

    // Page-data source: either this cache (cacheIndex == -1) or one of the
    // shared sibling caches (cacheIndex >= 0). When shared, the page's
    // dataOffset is into the SHARED cache's data section, scaled by the
    // shared cache's OWN dataTableAddress (re-read from its header). This
    // matches Reclaimer's ResourceIdentifier.ReadData logic.
    const uint8_t* sourceBase     = cache->base;
    size_t         sourceSize     = cache->size;
    uint32_t       sourceDataAddr = cache->dataTableAddress;

    if (page.cacheIndex >= 0) {
        // Routing decision - rate-limited so we don't write thousands of
        // lines for a fully-loaded scenario. Track per-shared-index hit
        // counts and only log the first ~4 hits per shared cache.
        static thread_local std::unordered_map<int, int> s_routeLogCounts;
        int& cnt = s_routeLogCounts[page.cacheIndex];
        bool logRoute = (cnt < 4);
        if (cnt < 1000000) ++cnt;

        // LOAD_PERF_FIX: use the dedicated pageRouteMutex
        // (was parseMutex). Splitting these unblocks parallel decode
        // beyond the prior 8-core ceiling - parseMutex stays for tag-
        // table mutation; pageRouteMutex only fires on shared-cache
        // open, which is idempotent so subsequent threads hit a
        // no-contention fast path.
        std::lock_guard<std::mutex> lk(cache->pageRouteMutex);
        CacheHandle* shared = GetOrOpenSharedCacheLocked(cache, page.cacheIndex);
        if (!shared) {
            NativeDiag("ReadRes: shared cache open failed cacheIdx=%d rid=0x%x",
                page.cacheIndex, resourceIdValue);
            return nullptr;
        }
        sourceBase     = shared->base;
        sourceSize     = shared->size;
        sourceDataAddr = shared->dataTableAddress;
        if (logRoute) {
            const char* nm = (page.cacheIndex < (int)cache->sharedCacheNames.size())
                ? cache->sharedCacheNames[page.cacheIndex].c_str() : "?";
            NativeDiag("ReadRes: route rid=0x%x cacheIdx=%d -> '%s' dta=0x%x size=%llu",
                resourceIdValue, page.cacheIndex, nm, sourceDataAddr,
                (unsigned long long)sourceSize);
        }
    }

    auto _rr2 = std::chrono::steady_clock::now(); g_rrRoute.fetch_add(RrNs(_rr1, _rr2), std::memory_order_relaxed);
    int64_t fileOff = (int64_t)sourceDataAddr + (int64_t)page.dataOffset;
    if (fileOff < 0 || (size_t)fileOff + (size_t)page.compressedSize > sourceSize) {
        NativeDiag("ReadRes: page data OOB rid=0x%x cacheIdx=%d dta=0x%x dataOff=%d csz=%d size=%llu",
            resourceIdValue, (int)page.cacheIndex, sourceDataAddr, page.dataOffset,
            page.compressedSize, (unsigned long long)sourceSize);
        return nullptr;
    }
    const uint8_t* compSrc = sourceBase + fileOff;

    size_t segLen = (size_t)page.decompressedSize - (size_t)segmentOffset;
    if (maxLength < segLen) segLen = maxLength;

    // Sanity cap on per-call decompression size. Reach shared.map pages can
    // legitimately be ~250MB decompressed (atlas pages under raw deflate in
    // U13+), so a 64MB ceiling is too tight. 512MB bounds the std::vector
    // allocation against truly corrupted size fields without rejecting valid
    // pages.
    constexpr size_t kMaxDecompressBytes = 512ull * 1024ull * 1024ull;
    if ((size_t)page.decompressedSize > kMaxDecompressBytes) {
        NativeDiag("ReadRes: SKIP rid=0x%x dsz=%d exceeds cap=%llu (%s)",
            resourceIdValue, page.decompressedSize,
            (unsigned long long)kMaxDecompressBytes,
            page.compressedSize == page.decompressedSize ? "passthrough" : "compressed");
        return nullptr;
    }

    if (page.compressedSize == page.decompressedSize) {
        if ((size_t)segmentOffset + segLen > (size_t)page.decompressedSize) {
            NativeDiag("ReadRes: passthrough OOB rid=0x%x segOff=%d segLen=%llu dsz=%d",
                resourceIdValue, segmentOffset, (unsigned long long)segLen, page.decompressedSize);
            return nullptr;
        }
        uint8_t* out = (uint8_t*)malloc(segLen);
        if (!out) { NativeDiag("ReadRes: malloc passthrough failed"); return nullptr; }
        memcpy(out, compSrc + segmentOffset, segLen);
        *outSize = segLen;
        NativeDiag("ReadRes: ok-passthrough rid=0x%x segIdx=%d page=%d csz=%d dsz=%d segOff=%d segLen=%llu",
            resourceIdValue, entry.segmentIndex, pageIndex, page.compressedSize,
            page.decompressedSize, segmentOffset, (unsigned long long)segLen);
        return out;
    }

    // #2 PAGE-CACHE: shared.map atlas pages hold many bitmaps;
    // re-inflating the whole page per bitmap was the p99 1.2-1.4s/bitmap cost.
    // Cache the decompressed page keyed by the resolved pageIndex, bounded by
    // MMS_PAGE_CACHE_MB (default 384; 0 disables). Read budget once.
    static int s_pageCacheMB = -1;
    if (s_pageCacheMB < 0) {
        char v[16] = {0};
        DWORD r = GetEnvironmentVariableA("MMS_PAGE_CACHE_MB", v, (DWORD)sizeof(v));
        // LOAD-SPEED: default RAISED 32 -> 1536 MB. The 32 MB default was far too
        // small to hold a big map's texture/lightmap pages (some pages are 155-250 MB), so nearly
        // EVERY bitmap/PVL/lightmap re-inflated its whole deflate page - the dominant cold-load cost
        // (panopticon bsp_geometry 17.6s). With the working set cached (inflate-once, engine-style),
        // bsp_geometry drops to ~3.8s and total wall 37.5s -> 9.1s, BYTE-IDENTICAL. The 1.5 GB peak
        // is transient: ZH_MBP_ClearPageCache (called by the app's POST-load trim, NOT the periodic
        // during-load trims) frees it after the load so steady memory returns to the #212 ~200 MB.
        // (The former 32 MB progression was tuned for steady memory; the post-load clear now gives
        // both - fast load AND low steady - so the big default is safe.)
        int mb = (r > 0) ? atoi(v) : 1536;
        s_pageCacheMB = (mb < 0) ? 0 : mb;
    }
    size_t pageCacheBudget = (size_t)s_pageCacheMB * 1024ull * 1024ull;
    // Post-load cap (set by ZH_MBP_ClearPageCache once the app's load has settled).
    {
        const size_t cap = cache->pageCacheBudgetCap.load(std::memory_order_relaxed);
        if (cap && cap < pageCacheBudget) pageCacheBudget = cap;
    }

    // Cache HIT: copy the segment straight out of the already-decompressed page
    // and skip the re-inflate entirely. (The decompress below stays lock-free.)
    std::shared_ptr<const std::vector<uint8_t>> hitPage;
    auto _rr4 = std::chrono::steady_clock::now();
    // Single-flight guard -- registered in cache->pageInflight while THIS thread inflates
    // `pageIndex`; released (with a notify_all) on every exit path below so waiters never hang.
    struct InflightGuard {
        CacheHandle* c = nullptr; int page = 0;
        ~InflightGuard() {
            if (!c) return;
            { std::lock_guard<std::mutex> pk(c->pageCacheMutex); c->pageInflight.erase(page); }
            c->pageInflightCv.notify_all();
        }
    } inflight;
    if (pageCacheBudget > 0) {
        auto _rr3 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> pk(cache->pageCacheMutex);
        _rr4 = std::chrono::steady_clock::now(); g_rrLockWait.fetch_add(RrNs(_rr3, _rr4), std::memory_order_relaxed);
        auto cit = cache->pageCache.find(pageIndex);
        if (cit == cache->pageCache.end() && cache->pageInflight.count(pageIndex)) {
            // Another thread is inflating this exact page -- wait for it instead of
            // allocating a second full-page buffer. Re-check the cache once it finishes.
            cache->pageInflightCv.wait(pk, [&] { return cache->pageInflight.count(pageIndex) == 0; });
            cit = cache->pageCache.find(pageIndex);
        }
        if (cit != cache->pageCache.end()) {
            hitPage = cit->second.data;   // shared ref: copy happens after the lock is released
            if (hitPage) cit->second.lastUsed = ++cache->pageCacheTick;
        } else {
            cache->pageInflight.insert(pageIndex);
            inflight.c = cache; inflight.page = pageIndex;
        }
    }
    if (hitPage) {
        // Lock-free copy: the malloc + memcpy must NOT run inside pageCacheMutex; with 30 decode
        // threads every bitmap read would queue behind every other thread's copy (measured: 252 s
        // of lock wait for 11 s of copying on a Forge World load). Only the shared_ptr clone is
        // under the lock.
        const std::vector<uint8_t>& cd = *hitPage;
        if ((size_t)segmentOffset < cd.size()) {
            size_t hitLen = cd.size() - (size_t)segmentOffset;
            if (hitLen > segLen) hitLen = segLen;
            uint8_t* hit = (uint8_t*)malloc(hitLen);
            if (hit) {
                memcpy(hit, cd.data() + (size_t)segmentOffset, hitLen);
                *outSize = hitLen;
                g_rrCopy.fetch_add(RrNs(_rr4, std::chrono::steady_clock::now()), std::memory_order_relaxed);
                g_pcHits.fetch_add(1, std::memory_order_relaxed);
                return hit;
            }
        }
    }

    auto _rr5 = std::chrono::steady_clock::now();
    std::vector<uint8_t> decompressed;
    try {
        decompressed.resize((size_t)page.decompressedSize);
    } catch (const std::bad_alloc&) {
        NativeDiag("ReadRes: bad_alloc dsz=%d rid=0x%x", page.decompressedSize, resourceIdValue);
        return nullptr;
    }

    // Codec dispatch.
    //
    // Reclaimer source is the source of truth: every MCC HaloReach build uses
    // RAW DEFLATE (RFC 1951), not LZ4 and not LZX. The Gen3+ default in
    // Reclaimer.Blam/Blam/Common/Annotations.cs:20-22 picks Deflate when no
    // explicit codec is given, and Reclaimer.Blam/Blam/Common/ContentFactory
    // .cs:304-321 implements that path with .NET's DeflateStream (which is raw
    // deflate - no zlib/gzip wrapper). The earlier LZ4 path was based on the
    // MCC-Win64-Shipping2.exe import table referencing liblz4, but those LZ4
    // imports are used elsewhere in MCC (network/save compression), NOT for
    // the .map resource pages. The xcompress64/LZX path is retained for
    // genuinely older builds (Halo3-era / pre-MCC formats) that still use it.
    //
    // Strategy:
    //   * For all MccHaloReach* cacheTypes (release through U13): try Deflate.
    //     Fall back to LZX only if deflate fails AND cacheType is the oldest
    //     pre-U3 release where the codec is least certain.
    //   * Unknown cacheType: try Deflate first, then LZX as a last resort.
    //   * LZ4 path is removed - it was a dead end and only ever produced
    //     spurious "decoded N / Y" lines for inputs the LZ4 frame parser
    //     happened not to reject.
    const bool isMccHaloReachBuild =
        (cache->cacheType == CacheType::MccHaloReach   ||
         cache->cacheType == CacheType::MccHaloReachU3 ||
         cache->cacheType == CacheType::MccHaloReachU8 ||
         cache->cacheType == CacheType::MccHaloReachU10 ||
         cache->cacheType == CacheType::MccHaloReachU13);

    size_t got = 0;
    const char* codecUsed = "deflate";
    LARGE_INTEGER _qf, _q0, _q1; QueryPerformanceFrequency(&_qf); QueryPerformanceCounter(&_q0);
    got = DecompressDeflate(compSrc, (size_t)page.compressedSize,
                            decompressed.data(), (size_t)page.decompressedSize);
    QueryPerformanceCounter(&_q1);
    g_pcInflates.fetch_add(1, std::memory_order_relaxed);
    g_pcInflateBytes.fetch_add((uint64_t)page.decompressedSize, std::memory_order_relaxed);
    g_pcInflateNs.fetch_add((uint64_t)((_q1.QuadPart - _q0.QuadPart) * 1000000000ull / (uint64_t)_qf.QuadPart), std::memory_order_relaxed);
    if (got == 0 && !isMccHaloReachBuild) {
        NativeDiag("ReadRes: Deflate failed for non-Reach build, trying LZX rid=0x%x csz=%d dsz=%d",
            resourceIdValue, page.compressedSize, page.decompressedSize);
        got = DecompressLZX(compSrc, (size_t)page.compressedSize,
                            decompressed.data(), (size_t)page.decompressedSize);
        codecUsed = "lzx";
    }
    if (got == 0) {
        NativeDiag("ReadRes: decompress failed rid=0x%x csz=%d dsz=%d cacheType=%d",
            resourceIdValue, page.compressedSize, page.decompressedSize,
            (int)cache->cacheType);
        return nullptr;
    }
    if ((size_t)segmentOffset + segLen > got) {
        if ((size_t)segmentOffset >= got) {
            NativeDiag("ReadRes: segOff>=got rid=0x%x segOff=%d got=%llu",
                resourceIdValue, segmentOffset, (unsigned long long)got);
            return nullptr;
        }
        segLen = got - (size_t)segmentOffset;
    }
    uint8_t* out = (uint8_t*)malloc(segLen);
    if (!out) { NativeDiag("ReadRes: malloc compressed failed segLen=%llu", (unsigned long long)segLen); return nullptr; }
    memcpy(out, decompressed.data() + segmentOffset, segLen);
    *outSize = segLen;
    g_rrMiss.fetch_add(RrNs(_rr5, std::chrono::steady_clock::now()), std::memory_order_relaxed);
    NativeDiag("ReadRes: ok-%s rid=0x%x segIdx=%d page=%d csz=%d dsz=%d segOff=%d segLen=%llu",
        codecUsed, resourceIdValue, entry.segmentIndex, pageIndex, page.compressedSize,
        page.decompressedSize, segmentOffset, (unsigned long long)segLen);
    // #2 PAGE-CACHE: stash the decompressed page (MOVE - `out` already holds a
    // copy of the segment) so sibling bitmaps in the same atlas page skip the
    // re-inflate. Evict the least-recently-used pages to stay under budget.
    if (pageCacheBudget > 0) {
        std::lock_guard<std::mutex> pk(cache->pageCacheMutex);
        if (cache->pageCache.find(pageIndex) == cache->pageCache.end()) {
            size_t sz = decompressed.size();
            if (sz > 0 && sz <= pageCacheBudget) {
                while (cache->pageCacheBytes + sz > pageCacheBudget &&
                       !cache->pageCache.empty()) {
                    auto victim = cache->pageCache.begin();
                    for (auto i2 = cache->pageCache.begin(); i2 != cache->pageCache.end(); ++i2)
                        if (i2->second.lastUsed < victim->second.lastUsed) victim = i2;
                    cache->pageCacheBytes -= victim->second.data ? victim->second.data->size() : 0;
                    cache->pageCache.erase(victim);
                }
                cache->pageCacheBytes += sz;
                auto& e = cache->pageCache[pageIndex];
                e.data = std::make_shared<const std::vector<uint8_t>>(std::move(decompressed));
                e.lastUsed = ++cache->pageCacheTick;
                g_pcStores.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Handle table
// -----------------------------------------------------------------------------

namespace {
std::mutex g_handlesMutex;
std::unordered_map<uint64_t, CacheHandle*> g_handles;
std::atomic<uint64_t> g_nextHandle{ 1 };

bool SehParseCacheHeader(CacheHandle* cache) {
    __try { return ParseCacheHeader(cache); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}
} // anonymous

CacheHandle* LookupHandle(uint64_t h) {
    std::lock_guard<std::mutex> lk(g_handlesMutex);
    auto it = g_handles.find(h);
    return it == g_handles.end() ? nullptr : it->second;
}

// Internal: open a shared sibling cache. Same mmap + ParseCacheHeader path as
// AcquireCacheHandle but DOES NOT register in g_handles - the parent owns it.
// Returns nullptr (and unmaps) on any failure. Caller takes ownership and is
// responsible for free-ing via the same UnmapViewOfFile/CloseHandle/delete
// sequence used in ReleaseCacheHandle.
namespace {
CacheHandle* OpenSharedHandle(const wchar_t* path) {
    if (!path || !*path) return nullptr;
    HANDLE hFile = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        NativeDiag("OpenShared: CreateFileW failed gle=%lu path='%ls'",
            GetLastError(), path);
        return nullptr;
    }
    LARGE_INTEGER fsz; fsz.QuadPart = 0;
    if (!GetFileSizeEx(hFile, &fsz) || fsz.QuadPart <= 0) {
        NativeDiag("OpenShared: GetFileSizeEx failed/zero size=%lld",
            (long long)fsz.QuadPart);
        CloseHandle(hFile);
        return nullptr;
    }
    HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) {
        NativeDiag("OpenShared: CreateFileMappingW failed gle=%lu", GetLastError());
        CloseHandle(hFile);
        return nullptr;
    }
    void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        NativeDiag("OpenShared: MapViewOfFile failed gle=%lu", GetLastError());
        CloseHandle(hMap); CloseHandle(hFile);
        return nullptr;
    }
    auto* c = new (std::nothrow) CacheHandle();
    if (!c) {
        UnmapViewOfFile(view); CloseHandle(hMap); CloseHandle(hFile);
        return nullptr;
    }
    c->hFile = hFile;
    c->hMap  = hMap;
    c->base  = static_cast<const uint8_t*>(view);
    c->size  = (size_t)fsz.QuadPart;
    c->path.assign(path);

    // Shared resource caches (shared.map, etc.) have a 'daeh' magic + a
    // valid build string but NO tag index / virtual base / section table -
    // ParseCacheHeader fails on them. Use the dedicated lightweight parser
    // that mirrors Reclaimer's behavior of reading dataTableAddress directly
    // from a build-specific fixed offset without touching the (absent) tag
    // tables.
    if (!ParseSharedCacheHeader(c)) {
        NativeDiag("OpenShared: ParseSharedCacheHeader failed for '%ls'", path);
        UnmapViewOfFile(view); CloseHandle(hMap); CloseHandle(hFile);
        delete c;
        return nullptr;
    }
    c->isSharedChild = true;
    return c;
}
} // anonymous

uint64_t AcquireCacheHandle(const wchar_t* path) {
    if (!path || !*path) { NativeDiag("Acquire: null path"); return 0; }

    NativeDiag("Acquire: begin path-len=%u", (unsigned)wcslen(path));

    HANDLE hFile = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        NativeDiag("Acquire: CreateFileW failed gle=%lu", GetLastError());
        return 0;
    }

    LARGE_INTEGER fsz; fsz.QuadPart = 0;
    if (!GetFileSizeEx(hFile, &fsz) || fsz.QuadPart <= 0) {
        NativeDiag("Acquire: GetFileSizeEx failed/zero size=%lld gle=%lu", (long long)fsz.QuadPart, GetLastError());
        CloseHandle(hFile);
        return 0;
    }

    HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) { NativeDiag("Acquire: CreateFileMappingW failed gle=%lu", GetLastError()); CloseHandle(hFile); return 0; }

    void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!view) { NativeDiag("Acquire: MapViewOfFile failed gle=%lu", GetLastError()); CloseHandle(hMap); CloseHandle(hFile); return 0; }

    auto* cache = new (std::nothrow) CacheHandle();
    if (!cache) { NativeDiag("Acquire: alloc CacheHandle failed"); UnmapViewOfFile(view); CloseHandle(hMap); CloseHandle(hFile); return 0; }
    cache->hFile = hFile;
    cache->hMap  = hMap;
    cache->base  = static_cast<const uint8_t*>(view);
    cache->size  = (size_t)fsz.QuadPart;
    cache->path.assign(path);

    NativeDiag("Acquire: mmap ok size=%llu first4=%02x %02x %02x %02x",
        (unsigned long long)cache->size,
        cache->base[0], cache->base[1], cache->base[2], cache->base[3]);

    bool ok = SehParseCacheHeader(cache);
    if (!ok) {
        NativeDiag("Acquire: SehParseCacheHeader returned false");
        UnmapViewOfFile(view);
        CloseHandle(hMap);
        CloseHandle(hFile);
        delete cache;
        return 0;
    }

    uint64_t handle = g_nextHandle.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(g_handlesMutex);
        g_handles[handle] = cache;
    }
    NativeDiag("Acquire: success handle=0x%llx", (unsigned long long)handle);
    return handle;
}

// Drain all open cache handles (called from ZH_MMP_PrepareUnload). Hot-reload
// path: the viewer is about to FreeLibrary us, so iterate every registered
// cache and unmap+free it.
void DrainAllCacheHandles() {
    std::vector<CacheHandle*> doomed;
    {
        std::lock_guard<std::mutex> lk(g_handlesMutex);
        doomed.reserve(g_handles.size());
        for (auto& kv : g_handles) doomed.push_back(kv.second);
        g_handles.clear();
    }
    for (auto* cache : doomed) {
        if (!cache) continue;
        if (cache->bitmapCache && cache->bitmapCacheDeleter) {
            cache->bitmapCacheDeleter(cache->bitmapCache);
            cache->bitmapCache = nullptr;
        }
        if (cache->modelCache && cache->modelCacheDeleter) {
            cache->modelCacheDeleter(cache->modelCache);
            cache->modelCache = nullptr;
        }
        for (auto* shared : cache->sharedCaches) {
            if (!shared) continue;
            if (shared->base) UnmapViewOfFile(shared->base);
            if (shared->hMap) CloseHandle(shared->hMap);
            if (shared->hFile != INVALID_HANDLE_VALUE) CloseHandle(shared->hFile);
            delete shared;
        }
        cache->sharedCaches.clear();
        if (cache->base) UnmapViewOfFile(cache->base);
        if (cache->hMap)  CloseHandle(cache->hMap);
        if (cache->hFile != INVALID_HANDLE_VALUE) CloseHandle(cache->hFile);
        delete cache;
    }
}

void ReleaseCacheHandle(uint64_t cacheHandle) {
    CacheHandle* cache = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_handlesMutex);
        auto it = g_handles.find(cacheHandle);
        if (it == g_handles.end()) return;
        cache = it->second;
        g_handles.erase(it);
    }
    if (!cache) return;

    // Free per-parser sub-caches (bitmap / model) before unmapping the file.
    if (cache->bitmapCache && cache->bitmapCacheDeleter) {
        cache->bitmapCacheDeleter(cache->bitmapCache);
        cache->bitmapCache = nullptr;
    }
    if (cache->modelCache && cache->modelCacheDeleter) {
        cache->modelCacheDeleter(cache->modelCache);
        cache->modelCache = nullptr;
    }

    // Free any lazily-opened shared cache siblings before unmapping the
    // primary. Shared children are not registered in g_handles, so this is
    // the only place they're released.
    for (auto* shared : cache->sharedCaches) {
        if (!shared) continue;
        if (shared->base) UnmapViewOfFile(shared->base);
        if (shared->hMap) CloseHandle(shared->hMap);
        if (shared->hFile != INVALID_HANDLE_VALUE) CloseHandle(shared->hFile);
        delete shared;
    }
    cache->sharedCaches.clear();

    // Free the per-cache material/lightmap caches (155 MB-class decompressed
    // LBSP pages) BEFORE unmapping - otherwise they leaked for the process lifetime.
    ::PurgeLightmapCachesForCache(cache);
    ::PurgeBspMaterialCachesForCache(cache);
    if (cache->base) UnmapViewOfFile(cache->base);
    if (cache->hMap)  CloseHandle(cache->hMap);
    if (cache->hFile != INVALID_HANDLE_VALUE) CloseHandle(cache->hFile);
    delete cache;
}

} // namespace zh_mcc

// #212: return the CRT heap's freed-but-retained pages to the OS. The parallel BSP decode
// allocates/frees gigabytes of transient geometry+bitmap buffers; the UCRT low-fragmentation
// heap keeps that commit reserved after free(). Called once after a load settles to shrink the
// process commit (paired with EmptyWorkingSet on the Rust side for the resident set).
extern "C" __declspec(dllexport) void __stdcall ZH_TrimHeaps() {
    _heapmin();
}
