// PlayerModeSnapshot.cpp (HaloMapStudioDLL)
// =============================================================================
// Publishes ZeroHour_PlayerMode_Snapshot v1 with the same 256-byte layout the
// viewer's existing client expects.
//
// Runs from the frame-pump hook (FramePumpHook.cpp), so we're on the engine
// thread and the descriptor resolver works. Same as ObjectTableSnapshot,
// but the per-tick walk is just 16 player slots -> biped datum -> primary tag
// -> hlmt -> mode chain.
//
// Player-list location: anchor + kRva_PlayerList + kPlayerHeaderOff (0x70).
// Per slot: stride 0x490, biped datum at slot+0x28.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"
#include "HaloReachTagHelpers.h"

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <unordered_map>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

constexpr uint32_t kMagic       = 0x534D505Au;  // 'ZPMS'
constexpr uint32_t kVersion     = 1u;
constexpr uint32_t kMaxPlayers  = 16u;

constexpr uint32_t kErr_Ok      = 0u;
constexpr uint32_t kErr_NoBase  = 1u;

// Player-list layout (matches PlayerListTLS.cpp).
constexpr size_t    kPlayerHeaderOff  = 0x70;
constexpr int       kPlayerStride     = 0x490;
constexpr int       kPlayerDatumOff   = 0x28;

#pragma pack(push, 1)
struct PlayerModeEntry {
    uint32_t BipedDatum;
    uint32_t ModeTagId;
};
struct PlayerMode_Shared {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteCounter;
    uint32_t PlayerCount;
    uint32_t LastTickMs;
    uint32_t LastError;
    uint32_t _Pad[2];
    PlayerModeEntry Players[kMaxPlayers];
    uint8_t  _Tail[256 - (32 + sizeof(PlayerModeEntry) * kMaxPlayers)];
};
static_assert(sizeof(PlayerMode_Shared) == 256, "ZPMS size");
#pragma pack(pop)

const wchar_t kMapName[] = L"ZeroHour_PlayerMode_Snapshot";

HANDLE             g_hMap   = nullptr;
PlayerMode_Shared* g_Shared = nullptr;

PlayerMode_Shared* EnsureShared()
{
    if (g_Shared) return g_Shared;
    g_hMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                (DWORD)sizeof(PlayerMode_Shared), kMapName);
    if (!g_hMap) return nullptr;
    g_Shared = (PlayerMode_Shared*)MapViewOfFile(g_hMap, FILE_MAP_ALL_ACCESS, 0, 0,
                                                 sizeof(PlayerMode_Shared));
    if (!g_Shared) { CloseHandle(g_hMap); g_hMap = nullptr; return nullptr; }
    __try {
        if (g_Shared->Magic != kMagic || g_Shared->Version != kVersion) {
            ZeroMemory(g_Shared, sizeof(PlayerMode_Shared));
            g_Shared->Magic       = kMagic;
            g_Shared->Version     = kVersion;
            g_Shared->PlayerCount = kMaxPlayers;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return g_Shared;
}

// Resolve the player-list slot 0 base by walking the same anchor the engine
// uses internally: anchor = *(haloreach + kRva_SessionAnchor); slot0 =
// anchor + kRva_PlayerList + kPlayerHeaderOff. NOT a TLS read - this lives
// in static engine state.
uint8_t* ResolvePlayerSlot0()
{
    using namespace HaloMapStudio::Engine;
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    uint8_t* base = (uint8_t*)mod;

    uint8_t* anchor = nullptr;
    __try { anchor = *(uint8_t**)(base + kRva_SessionAnchor); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    if (!anchor) return nullptr;

    return anchor + kRva_PlayerList + kPlayerHeaderOff;
}

uint32_t SafeReadDatum(uint8_t* slot0, int idx)
{
    if (!slot0) return 0;
    uint32_t v = 0;
    __try {
        v = *(uint32_t*)(slot0 + (size_t)idx * (size_t)kPlayerStride + kPlayerDatumOff);
    } __except (EXCEPTION_EXECUTE_HANDLER) { v = 0; }
    return v;
}

// Cache (datum -> mode) so the per-frame walk is a hash lookup once warm.
struct ModeCacheEntry { uint32_t modeTagId; uint64_t lastTick; };
std::unordered_map<uint32_t, ModeCacheEntry> g_ModeCache;
HMODULE g_ModeCache_LastHr = nullptr;
// Epoch reconcile - see ObjectTableSnapshot.cpp's identical pattern. We
// also need to drop g_ModeCache when the engine swaps scenarios (datum
// reuse means a cached `bipedDatum -> mode` may resolve a brand-new biped
// to the previous scenario's render model, then ObjectTableSnapshot
// publishes either zero positions or wrong ones).
uint64_t g_ModeCache_LastEpoch = 0;
constexpr size_t kModeCacheCap = 256;

// POD-leaf helper so the SEH read is in its own function (C2712 - can't
// use __try in a function that has C++ unwindable locals like the
// unordered_map iterator inside ResolvePlayerMode).
uint32_t SafeReadPrimaryTag(uint8_t* desc, uint32_t bipedDatum)
{
    using namespace HaloMapStudio::Engine;
    uint8_t* obj = ResolveDatumToObject(desc, bipedDatum);
    if (!obj) return 0;
    uint32_t v = 0;
    __try { v = *(uint32_t*)(obj + kObj_PrimaryTag); }
    __except (EXCEPTION_EXECUTE_HANDLER) { v = 0; }
    return v;
}

uint32_t ResolvePlayerMode(uint8_t* desc, uint8_t* slot0, int idx)
{
    using namespace ZeroHour::HrTag;

    uint32_t bipedDatum = SafeReadDatum(slot0, idx);
    if (bipedDatum == 0u || bipedDatum == 0xFFFFFFFFu) return 0;

    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (hr != g_ModeCache_LastHr) {
        g_ModeCache.clear();
        g_ModeCache_LastHr = hr;
        g_ModeCache_LastEpoch = HaloMapStudio::Engine::GetMapEpoch();
    }
    // Per-tick epoch reconcile - fires when ResolveObjectPoolDescriptor
    // bumped the global epoch on a scenario swap (covered by the snapshot
    // tick which always runs immediately before this).
    {
        uint64_t curEpoch = HaloMapStudio::Engine::GetMapEpoch();
        if (curEpoch != g_ModeCache_LastEpoch) {
            g_ModeCache.clear();
            g_ModeCache_LastEpoch = curEpoch;
        }
    }
    if (!hr) return 0;

    auto it = g_ModeCache.find(bipedDatum);
    if (it != g_ModeCache.end()) {
        it->second.lastTick = GetTickCount64();
        return it->second.modeTagId;
    }

    // Resolve biped -> obj* via the shared engine resolver, read +0x00
    // primary tag, walk hlmt -> mode via the header-only helper.
    uint32_t modeTagId = 0;
    uint32_t primaryTag = SafeReadPrimaryTag(desc, bipedDatum);
    if (primaryTag != 0u && primaryTag != 0xFFFFFFFFu) {
        uint32_t hlmt = 0;
        ResolveHlmtAndModeDatums_Cached(primaryTag, &hlmt, &modeTagId);
    }

    if (g_ModeCache.size() < kModeCacheCap)
        g_ModeCache.emplace(bipedDatum, ModeCacheEntry{modeTagId, GetTickCount64()});
    return modeTagId;
}

} // namespace

extern "C" __declspec(dllexport) void PlayerModeSnapshot_FramePumpTick()
{
    using namespace HaloMapStudio::Engine;

    PlayerMode_Shared* sh = EnsureShared();
    if (!sh) return;

    uint8_t* slot0 = ResolvePlayerSlot0();
    uint8_t* desc  = ResolveObjectPoolDescriptor();

    if (!slot0 || !desc) {
        __try {
            ZeroMemory(sh->Players, sizeof(sh->Players));
            sh->LastError    = kErr_NoBase;
            sh->LastTickMs   = GetTickCount();
            sh->WriteCounter = sh->WriteCounter + 1u;
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
        return;
    }

    __try {
        for (uint32_t i = 0; i < kMaxPlayers; ++i) {
            uint32_t bipedDatum = SafeReadDatum(slot0, (int)i);
            uint32_t modeTagId  = ResolvePlayerMode(desc, slot0, (int)i);
            sh->Players[i].BipedDatum = bipedDatum;
            sh->Players[i].ModeTagId  = modeTagId;
        }
        sh->LastError    = kErr_Ok;
        sh->LastTickMs   = GetTickCount();
        sh->WriteCounter = sh->WriteCounter + 1u;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        UnmapViewOfFile(sh);
        CloseHandle(g_hMap);
        g_hMap   = nullptr;
        g_Shared = nullptr;
    }
}
