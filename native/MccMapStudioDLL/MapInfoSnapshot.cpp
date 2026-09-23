// MapInfoSnapshot.cpp
// =============================================================================
// Publishes the currently-loaded scenario datum + the loaded .map file path
// (best-effort) so the viewer can find and open the matching .map via
// Reclaimer.Blam without having to guess.
//
// MMF: ZeroHour_MapInfo_Snapshot
// -----------------------------------------------------------------------------
//   Header (32 bytes)
//     u32 Magic           'ZMAP'  (0x504D415A LE)
//     u32 Version         1
//     u32 WriteCounter    bumped after publish
//     u32 ScenarioDatum   raw datum (haloreach.dll +0xAFBE38)
//     u32 LastTickMs      GetTickCount() at end of last publish
//     u32 LastError       0 = ok, 1 = no scnr datum, 2 = read fault
//     u32 ScenarioNameLen length of ScenarioName below
//     u32 MapPathLen      length of MapPath below
//
//   char ScenarioName[256]   asset path, NUL-terminated UTF-8 (best-effort).
//   char MapPath[260]        full filesystem path to the .map (best-effort).
//
// Total: 32 + 256 + 260 = 548 bytes (rounded up to 1024).
// =============================================================================

#include "pch.h"
#include "HaloReachTagHelpers.h"

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <cstdio>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

constexpr uint32_t kMagic   = 0x504D415Au; // 'ZMAP'
constexpr uint32_t kVersion = 1u;

// Same address ForgePaletteSnapshot uses; cross-checked there.
constexpr uintptr_t kRva_ScenarioDatum = 0xAFBE38;

// Loaded-map descriptor pointer. *(haloreach.dll + 0x24FB710) is a pointer
// to a struct laid out as:
//   +0x00  uint32  some hash/id (varies per map)
//   +0x04  uint32  padding (always zero observed)
//   +0x08  char[]  NUL-terminated UTF-8 asset path,
//                  e.g. "levels\multi\forge_halo\forge_halo"
constexpr uintptr_t kRva_MapDescriptorPtr = 0x24FB710;
constexpr size_t    kMapDescriptor_PathOffset = 0x08;

constexpr uint32_t kErr_Ok          = 0u;
constexpr uint32_t kErr_NoScenario  = 1u;
constexpr uint32_t kErr_ReadFault   = 2u;

#pragma pack(push, 1)
struct MapInfo_Shared {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteCounter;
    uint32_t ScenarioDatum;
    uint32_t LastTickMs;
    uint32_t LastError;
    uint32_t ScenarioNameLen;
    uint32_t MapPathLen;
    char     ScenarioName[256];
    char     MapPath[260];
    char     _Pad[1024 - (32 + 256 + 260)];
};
static_assert(sizeof(MapInfo_Shared) == 1024, "MapInfo size");
#pragma pack(pop)

const wchar_t kMapName[] = L"ZeroHour_MapInfo_Snapshot";

HANDLE          g_hMap   = nullptr;
MapInfo_Shared* g_Shared = nullptr;

MapInfo_Shared* EnsureShared()
{
    if (g_Shared) return g_Shared;
    g_hMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                (DWORD)sizeof(MapInfo_Shared), kMapName);
    if (!g_hMap) return nullptr;
    g_Shared = (MapInfo_Shared*)MapViewOfFile(g_hMap, FILE_MAP_ALL_ACCESS, 0, 0,
                                              sizeof(MapInfo_Shared));
    if (!g_Shared) { CloseHandle(g_hMap); g_hMap = nullptr; return nullptr; }

    __try {
        if (g_Shared->Magic != kMagic || g_Shared->Version != kVersion) {
            ZeroMemory(g_Shared, sizeof(MapInfo_Shared));
            g_Shared->Magic   = kMagic;
            g_Shared->Version = kVersion;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return g_Shared;
}

uint32_t SafeReadU32(uint64_t addr)
{
    uint32_t v = 0;
    __try { v = *(uint32_t*)addr; } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return v;
}

// Read the loaded map's asset path via the descriptor pointer at
// haloreach.dll + kRva_MapDescriptorPtr. Layout:
//   *(base + 0x24FB710) -> struct { u32 id; u32 _pad; char path[N]; }
// On success, fills outBuf with e.g. "levels\multi\forge_halo\forge_halo".
bool TryReadScenarioName(uint32_t /*scnrDatum*/, char outBuf[256])
{
    outBuf[0] = '\0';
    uint8_t* base = ZeroHour::HrTag::ReachBase();
    if (!base) return false;

    __try {
        uint64_t descPtr = *(uint64_t*)(base + kRva_MapDescriptorPtr);
        if (descPtr == 0) return false;
        const char* p = (const char*)(descPtr + kMapDescriptor_PathOffset);
        size_t n = 0;
        while (n < 255 && p[n] != '\0') {
            uint8_t b = (uint8_t)p[n];
            if (b < 0x20 || b > 0x7E) break;
            outBuf[n] = p[n];
            ++n;
        }
        outBuf[n] = '\0';
        return n >= 4;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        outBuf[0] = '\0';
        return false;
    }
}

// Try to read the loaded map's filesystem path. Without a known global to
// read this is also best-effort - leave empty if we can't find it.
bool TryReadMapFilePath(char outBuf[260])
{
    outBuf[0] = '\0';
    return false; // unknown global; viewer-side scan handles discovery.
}

} // namespace

extern "C" __declspec(dllexport) void MapInfoSnapshot_FramePumpTick()
{
    MapInfo_Shared* sh = EnsureShared();
    if (!sh) return;

    uint8_t* base = ZeroHour::HrTag::ReachBase();
    uint32_t scnrDatum = 0;
    if (base) scnrDatum = SafeReadU32((uint64_t)(base + kRva_ScenarioDatum));

    __try {
        sh->ScenarioDatum = scnrDatum;
        sh->LastTickMs    = GetTickCount();

        if (scnrDatum == 0u || scnrDatum == 0xFFFFFFFFu) {
            sh->ScenarioName[0] = '\0';
            sh->MapPath[0]      = '\0';
            sh->ScenarioNameLen = 0;
            sh->MapPathLen      = 0;
            sh->LastError       = kErr_NoScenario;
        }
        else {
            char name[256] = { 0 };
            char path[260] = { 0 };
            TryReadScenarioName(scnrDatum, name);
            TryReadMapFilePath(path);
            memcpy(sh->ScenarioName, name, sizeof(sh->ScenarioName));
            memcpy(sh->MapPath,      path, sizeof(sh->MapPath));
            sh->ScenarioNameLen = (uint32_t)strnlen_s(sh->ScenarioName, sizeof(sh->ScenarioName));
            sh->MapPathLen      = (uint32_t)strnlen_s(sh->MapPath,      sizeof(sh->MapPath));
            sh->LastError       = kErr_Ok;
        }

        // Bump after body is published so a torn reader sees consistent fields.
        sh->WriteCounter = sh->WriteCounter + 1u;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        // Mapped view fault - drop the handle so we re-create next tick.
        UnmapViewOfFile(sh);
        CloseHandle(g_hMap);
        g_hMap   = nullptr;
        g_Shared = nullptr;
    }
}
