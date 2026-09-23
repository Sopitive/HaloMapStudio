// ForgePaletteSnapshot.cpp
// =============================================================================
// Viewer-facing forge palette snapshot.
//
// Distinct from ForgePalette.cpp (which feeds the viewer's spawn list):
// this one publishes a flatter layout that ALSO carries the resolved HLMT
// and render-model ('mode') tag datums per palette entry, so the viewer can
// trail a Phase-3 mesh decoder off the same record without any further
// engine round trips.
//
// MMF: ZeroHour_ForgePalette_Snapshot
// -----------------------------------------------------------------------------
//   Header (32 bytes - v2; was 16 in v1)
//     u32 Magic          'FRGP' (0x50475246 LE)
//     u32 Version        2  (v1 had no RequestCounter; old viewers reading
//                            the v2 layout will see EntryCount where they
//                            expected it - same field offset - and the
//                            extra request slot below is invisible to them)
//     u32 EntryCount     0..kMaxEntries
//     u32 WriteCounter   bumped after publish
//     u32 RequestCounter consumer bumps to ask for a force re-walk; DLL
//                        edge-detects vs its private last-seen value and
//                        clears g_LastScenarioDatum/g_RetryCountdown.
//                        Read by the DLL only; never written by the DLL.
//     u32 _Pad[3]        zero-initialised; reserved for future fields
//                        (kept so the entry array stays 16-byte aligned).
//   ForgePaletteEntry[kMaxEntries] - 64 bytes each
//     u32 PaletteIndex
//     u32 VariantIndex   (entry-flat index - see note below)
//     u32 TagGroupMagic  big-endian fourcc as stored in memory
//     u32 TagGlobalId    raw datum (variant +0x10)
//     u32 HlmtTagId      0 if not resolvable
//     u32 ModeTagId      0 if not resolvable
//     char TagName[40]   ASCII, NUL-terminated; synthesised label
//
// Note on VariantIndex: the underlying scnr palette walk is three-deep
// (palette -> entry -> variant). For the viewer we flatten to (palette,
// flat-variant-index) so the viewer can feed the existing
// PublicDirectSpawn MMF without needing to re-walk the entry tree. The
// variant field stores `entry << 8 | variantWithinEntry` so the original
// 3-tuple can be recovered.
//
// Total MMF size: 32 + 512 * 64 = 32800 bytes (~32KB).
// =============================================================================

#include "pch.h"
#include "HaloReachTagHelpers.h"
#include "EngineThreadResolver.h"  // HaloMapStudio::Engine::GetMapEpoch()

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <cstdio>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

constexpr uint32_t kMagic       = 0x50475246u;  // 'FRGP' (LE: F,R,G,P)
// v3 adds a palette-names table to the header so the viewer's master palette
// can categorize using real names ("Vehicles", "Decorative", "Items/
// Weapons" etc.) instead of guessing from palette index. Names come from
// haloreach.dll!StringId_GetString @ +0x9D91C (RE'd in
// HaloReach_strings_RE.md). Per-palette name field is at pal+0x00 (the
// scnr palette struct's first field is StringId Name; entries tagblock
// starts at pal+0x08).
//
// v4 adds resolved entry+variant DISPLAY names per ForgePaletteEntry - 
// the strings the in-game forge UI shows ("Block 4x4", "Warthog", etc.).
// Each scnr palette entry has Name StringId at ent+0x00 (entries[] is
// 0x1C bytes), each variant has Name StringId at var+0x00 (variants[]
// is 0x18 bytes). The TagName field stays as fallback synthesised label.
constexpr uint32_t kVersion     = 4u;
constexpr uint32_t kMaxEntries  = 512u;
constexpr uint32_t kMaxPalettes = 32u;            // forge maps top out around ~12
constexpr uint32_t kPaletteNameBytes = 64u;       // engine ASCII strings well under 64

// haloreach.dll RVA for StringId_GetString - see HaloReach_strings_RE.md section A.2.
constexpr uintptr_t kRva_StringId_GetString = 0x9D91C;

// Same scenario-datum location ForgePalette.cpp uses; cross-checked there.
constexpr uintptr_t kRva_ScenarioDatum = 0xAFBE38;

#pragma pack(push, 1)
struct ForgePaletteEntry {
    uint32_t PaletteIndex;
    uint32_t VariantIndex;       // (entry << 8) | varInEntry
    uint32_t TagGroupMagic;      // big-endian as stored
    uint32_t TagGlobalId;
    uint32_t HlmtTagId;
    uint32_t ModeTagId;
    char     TagName[40];        // synthesised fallback ("scen p00 e00 v00")
    char     EntryName[64];      // v4: resolved entry-name StringId (ent+0x00)
    char     VariantName[64];    // v4: resolved variant-name StringId (var+0x00)
};
static_assert(sizeof(ForgePaletteEntry) == 64 + 128, "Entry layout v4 (192B)");

struct ForgePaletteSnapshot_Header {
    uint32_t Magic;
    uint32_t Version;
    uint32_t EntryCount;
    uint32_t WriteCounter;
    uint32_t RequestCounter;   // v2: viewer bumps to force a re-walk
    uint32_t PaletteCount;     // v3: number of valid PaletteNames entries
    uint32_t _Pad1;            // keep PaletteNames[] aligned
    uint32_t _Pad2;
    char     PaletteNames[kMaxPalettes][kPaletteNameBytes]; // v3
};
// Header is 32B fixed + 32 * 64B = 2080B total. Entries[] follows.
static_assert(sizeof(ForgePaletteSnapshot_Header) == 32 + kMaxPalettes * kPaletteNameBytes,
              "Header layout v3");

struct ForgePalette_Snapshot {
    ForgePaletteSnapshot_Header Header;
    ForgePaletteEntry           Entries[kMaxEntries];
};
#pragma pack(pop)

const wchar_t kMapName[] = L"ZeroHour_ForgePalette_Snapshot";

HANDLE                  g_hMap   = nullptr;
ForgePalette_Snapshot*  g_Shared = nullptr;
uint32_t                g_LastScenarioDatum  = 0;
int                     g_RetryCountdown     = 0;
// Edge-detect for the v2 force-rewalk request. Mirrors the
// DamageResponseControl.cpp pattern: capture initial value at first
// EnsureShared so an existing in-flight request from the previous DLL
// instance doesn't accidentally fire on attach.
uint32_t                g_LastRequestCounter = 0;
bool                    g_RequestSeeded      = false;
// Map-epoch the last PUBLISHED walk was made against. A map (re)load bumps
// the engine epoch (HaloMapStudio::Engine::GetMapEpoch); if our published walk
// is from a different epoch it's stale -> force a re-walk. Without this the
// tick would freeze a garbage walk for 30 s after a map reopen (the scnr
// datum value alone doesn't change, so the retry-pacing gate skipped re-walks).
uint64_t                g_LastPublishEpoch   = ~0ull;
// Stability verifier: the tag table can still be repopulating right after a
// load, so a single walk can read plausible-but-garbage entries. We require
// two consecutive walks to AGREE (same entry count + first tag id) before
// trusting the result and entering the 30 s lockout.
uint32_t                g_VerifyEntryCount   = 0xFFFFFFFFu;
uint32_t                g_VerifyFirstTag     = 0xFFFFFFFFu;
// STUTTER FIX: consecutive walks whose entry COUNT + map EPOCH held
// steady (firstTag-independent). The firstTag verifier can fail to converge
// forever (flaky cross-process read of the first entry / volatile first slot),
// which left this re-walking ALL entries every tick = a per-frame RPM stutter.
uint32_t                g_StableCountStreak  = 0;

ForgePalette_Snapshot* EnsureShared()
{
    if (g_Shared) return g_Shared;
    g_hMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                (DWORD)sizeof(ForgePalette_Snapshot), kMapName);
    if (!g_hMap) return nullptr;
    g_Shared = (ForgePalette_Snapshot*)MapViewOfFile(g_hMap, FILE_MAP_ALL_ACCESS, 0, 0,
                                                     sizeof(ForgePalette_Snapshot));
    if (!g_Shared) {
        CloseHandle(g_hMap);
        g_hMap = nullptr;
        return nullptr;
    }
    __try {
        // Re-init on either a fresh map or a v1->v2 layout upgrade. The v1
        // layout had no RequestCounter at all, so anything we'd read at
        // header+0x10 there is undefined; safer to zero and seed.
        if (g_Shared->Header.Magic != kMagic || g_Shared->Header.Version != kVersion) {
            ZeroMemory(g_Shared, sizeof(ForgePalette_Snapshot));
            g_Shared->Header.Magic   = kMagic;
            g_Shared->Header.Version = kVersion;
        }
        // Seed the request edge-detect with whatever the consumer already
        // wrote, so a stale RequestCounter doesn't trigger a phantom rewalk
        // the first tick after we attach.
        if (!g_RequestSeeded) {
            g_LastRequestCounter = g_Shared->Header.RequestCounter;
            g_RequestSeeded = true;
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

uint32_t ReadScenarioDatum()
{
    uint8_t* base = ZeroHour::HrTag::ReachBase();
    if (!base) return 0;
    return SafeReadU32((uint64_t)(base + kRva_ScenarioDatum));
}

// Render the 4 ASCII bytes of the BE fourcc into a small string ("scen", "vehi"
// etc.). Non-printable bytes are turned into '?'.
void GroupToString(uint32_t groupBE, char out[5])
{
    char chars[4] = {
        (char)((groupBE >> 24) & 0xFF),
        (char)((groupBE >> 16) & 0xFF),
        (char)((groupBE >>  8) & 0xFF),
        (char)((groupBE      ) & 0xFF),
    };
    for (int i = 0; i < 4; ++i) {
        unsigned char c = (unsigned char)chars[i];
        out[i] = (c >= 0x20 && c < 0x7F) ? (char)c : '?';
    }
    out[4] = '\0';
}

// Synthesised display label. Real string-id resolution would need the
// engine's stringid table, which is not resolved here
// - leave a sensible placeholder that's still useful in a list view.
void FormatTagName(uint32_t pal, uint32_t entry, uint32_t varInEnt,
                   uint32_t groupBE, char out[40])
{
    char grp[5];
    GroupToString(groupBE, grp);
    // Cap to 39 chars + NUL.
    int n = _snprintf_s(out, 40, _TRUNCATE,
                        "%s p%02u e%02u v%02u",
                        grp, (unsigned)pal, (unsigned)entry, (unsigned)varInEnt);
    (void)n;
}

using PFN_StringId_GetString = const char* (*)(int stringId);

// Lazy-resolve once per process; returns nullptr if the engine module
// isn't loaded yet (DLL injected before haloreach mapped). Stable for the
// lifetime of haloreach.dll.
PFN_StringId_GetString GetStringIdResolver()
{
    static PFN_StringId_GetString cached = nullptr;
    if (cached) return cached;
    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return nullptr;
    cached = reinterpret_cast<PFN_StringId_GetString>(
        reinterpret_cast<uint8_t*>(hr) + kRva_StringId_GetString);
    return cached;
}

// Look up a string id and copy the result into outBuf. Empty string on
// failure (resolver missing, id unknown, SEH fault during read). Used to
// publish palette names so the viewer's master palette can categorize using the
// authoritative scnr.palette[].Name strings.
void ResolveStringIdToBuf(uint32_t stringId, char* outBuf, size_t outBufSize)
{
    if (outBufSize == 0) return;
    outBuf[0] = '\0';
    auto fn = GetStringIdResolver();
    if (!fn) return;
    const char* s = nullptr;
    __try { s = fn(static_cast<int>(stringId)); }
    __except (EXCEPTION_EXECUTE_HANDLER) { s = nullptr; }
    if (!s) return;
    __try
    {
        size_t i = 0;
        for (; i + 1 < outBufSize; ++i)
        {
            char c = s[i];
            if (c == '\0') break;
            outBuf[i] = c;
        }
        outBuf[i] = '\0';
    }
    __except (EXCEPTION_EXECUTE_HANDLER) { outBuf[0] = '\0'; }
}

} // namespace

// -----------------------------------------------------------------------------
// Frame-pump tick - re-walks only on scenario change, with a 60-tick retry
// pacing for failed walks (engine partially loaded / tag table being
// re-populated).
// -----------------------------------------------------------------------------
extern "C" __declspec(dllexport) void ForgePaletteSnapshot_FramePumpTick()
{
    using namespace ZeroHour::HrTag;

    ForgePalette_Snapshot* sh = EnsureShared();
    if (!sh) return;

    // ---- v2 force-rewalk edge ----
    // Viewer bumps Header.RequestCounter to ask for a fresh walk (e.g. after
    // hitting "Force palette re-walk"). We clear our caches so the next
    // walk is unconditional regardless of scnr stability + retry pacing.
    uint32_t reqNow = 0;
    __try { reqNow = sh->Header.RequestCounter; }
    __except (EXCEPTION_EXECUTE_HANDLER) { reqNow = g_LastRequestCounter; }
    if (reqNow != g_LastRequestCounter) {
        g_LastRequestCounter  = reqNow;
        g_LastScenarioDatum   = 0;
        g_RetryCountdown      = 0;
        g_VerifyEntryCount    = 0xFFFFFFFFu;
        g_StableCountStreak   = 0;
        ZH_Logf("[ForgePaletteSnapshot] force-rewalk requested (req=%u)\n", reqNow);
    }

    // ---- map-epoch reconcile ----
    // A map (re)load bumps the engine epoch. Our last published walk was made
    // against g_LastPublishEpoch; if the epoch advanced, that result is for a
    // DIFFERENT map and must not be trusted - force an unconditional re-walk
    // and reset the stability verifier. This is the defense every other
    // snapshot tick has; the palette tick was missing it, which is why a map
    // reopen froze garbage for 30s (the scnr datum alone didn't change, so the
    // retry-pacing gate kept skipping re-walks). Survives a viewer restart too:
    // the DLL stays injected, so on the new epoch it re-walks instead of
    // serving the stale MMF.
    uint64_t curEpoch = HaloMapStudio::Engine::GetMapEpoch();
    if (curEpoch != g_LastPublishEpoch) {
        g_LastScenarioDatum = 0;
        g_RetryCountdown    = 0;
        g_VerifyEntryCount  = 0xFFFFFFFFu;
        g_StableCountStreak = 0;
    }

    uint32_t scnrDatum = ReadScenarioDatum();
    if (scnrDatum == 0u || scnrDatum == 0xFFFFFFFFu) {
        if (g_LastScenarioDatum != 0) {
            __try {
                sh->Header.EntryCount   = 0;
                sh->Header.WriteCounter = sh->Header.WriteCounter + 1u;
            } __except (EXCEPTION_EXECUTE_HANDLER) {}
            g_LastScenarioDatum = 0;
            ZH_Logf("[ForgePaletteSnapshot] scnr cleared (was loaded, now 0)\n");
        }
        // First-launch path: log once-per-N-ticks so we can see if the
        // scnr datum is the gate.
        static uint32_t s_NoScnrTick = 0;
        if ((++s_NoScnrTick % 600u) == 1u)
            ZH_Logf("[ForgePaletteSnapshot] no scnr datum yet (engine not loaded)\n");
        return;
    }
    if (scnrDatum == g_LastScenarioDatum && g_RetryCountdown > 0) {
        --g_RetryCountdown;
        return;
    }

    uint64_t scnrVA = ResolveTagAddress(scnrDatum);
    if (!scnrVA) {
        ZH_Logf("[ForgePaletteSnapshot] ResolveTagAddress(0x%08X) returned 0; retrying\n", scnrDatum);
        g_RetryCountdown = 60;
        return;
    }

    Tagblock palBlock{};
    if (!ReadTagblock(scnrVA + 0x228, palBlock)) {
        ZH_Logf("[ForgePaletteSnapshot] ReadTagblock(scnr+0x228) faulted; scnrVA=0x%llX\n",
            (unsigned long long)scnrVA);
        g_RetryCountdown = 60;
        return;
    }
    if (palBlock.count == 0) {
        ZH_Logf("[ForgePaletteSnapshot] palette tagblock count=0 (not a forge map?); scnrVA=0x%llX ptr=0x%X\n",
            (unsigned long long)scnrVA, palBlock.ptr);
        g_RetryCountdown = 60;
        return;
    }
    uint64_t palArr = ExpandContracted(palBlock.ptr);
    if (!palArr) {
        ZH_Logf("[ForgePaletteSnapshot] ExpandContracted(0x%X) returned 0; palette ptr unresolvable\n",
            palBlock.ptr);
        g_RetryCountdown = 60;
        return;
    }

    // Walk into a stack scratch buffer so a SEH abort mid-walk leaves the
    // mapped view's old contents intact for the viewer.
    ForgePaletteEntry tmp[kMaxEntries];
    uint32_t outCount = 0;
    // Per-palette resolved names. Indexed by palette index 0..N. Empty
    // string for palettes we couldn't resolve (StringId table not ready,
    // empty palette, etc.) so the viewer can fall back to "Palette N".
    char paletteNames[kMaxPalettes][kPaletteNameBytes] = {};
    uint32_t resolvedPaletteCount = 0;

    for (uint32_t p = 0; p < palBlock.count && outCount < kMaxEntries; ++p) {
        uint64_t pal = palArr + (uint64_t)p * 0x14;

        // scnr.palette[].Name @ pal+0x00 is a StringId. Resolve only for
        // the first kMaxPalettes; if the map has more than 32 palettes
        // the high indices fall back to "Palette N" in the viewer.
        if (p < kMaxPalettes)
        {
            uint32_t nameSid = SafeReadU32(pal + 0x00);
            ResolveStringIdToBuf(nameSid, paletteNames[p], kPaletteNameBytes);
            if (paletteNames[p][0] != '\0' && resolvedPaletteCount < p + 1)
                resolvedPaletteCount = p + 1;
        }

        Tagblock entBlock{};
        if (!ReadTagblock(pal + 0x08, entBlock) || entBlock.count == 0) continue;
        uint64_t entArr = ExpandContracted(entBlock.ptr);
        if (!entArr) continue;

        for (uint32_t e = 0; e < entBlock.count && outCount < kMaxEntries; ++e) {
            uint64_t ent = entArr + (uint64_t)e * 0x1C;
            // v4: scnr palette entry struct starts with StringId Name at
            // ent+0x00 - the in-game forge UI label for this entry group
            // (e.g. "block", "covenant_crate"). Variants tagblock follows
            // at ent+0x04.
            uint32_t entryNameSid = SafeReadU32(ent + 0x00);

            Tagblock varBlock{};
            if (!ReadTagblock(ent + 0x04, varBlock) || varBlock.count == 0) continue;
            uint64_t varArr = ExpandContracted(varBlock.ptr);
            if (!varArr) continue;

            for (uint32_t v = 0; v < varBlock.count && outCount < kMaxEntries; ++v) {
                uint64_t var = varArr + (uint64_t)v * 0x18;
                // v4: variant struct starts with StringId Name at var+0x00,
                // then a 16-byte tagref at var+0x04 (+0x00 fourcc, +0x0C
                // datum). Some variants leave Name as 0 (default variant
                // = parent entry's name).
                uint32_t variantNameSid = SafeReadU32(var + 0x00);
                uint32_t groupBE  = SafeReadU32(var + 0x04 + 0x00);
                uint32_t tagDatum = SafeReadU32(var + 0x04 + 0x0C);
                if (tagDatum == 0u || tagDatum == 0xFFFFFFFFu) continue;

                uint32_t hlmt = 0, mode = 0;
                ResolveHlmtAndModeDatums_Cached(tagDatum, &hlmt, &mode);

                ForgePaletteEntry& dst = tmp[outCount++];
                dst.PaletteIndex  = p;
                dst.VariantIndex  = (e << 8) | (v & 0xFFu);
                dst.TagGroupMagic = groupBE;
                dst.TagGlobalId   = tagDatum;
                dst.HlmtTagId     = hlmt;
                dst.ModeTagId     = mode;
                FormatTagName(p, e, v, groupBE, dst.TagName);
                // v4: resolve real string-table display names. Both calls
                // return empty strings on failure; viewer side falls back to
                // tag-asset-path (FullPath) -> synthesised TagName.
                ResolveStringIdToBuf(entryNameSid,   dst.EntryName,   sizeof(dst.EntryName));
                ResolveStringIdToBuf(variantNameSid, dst.VariantName, sizeof(dst.VariantName));
            }
        }
    }

    __try {
        for (uint32_t i = 0; i < outCount; ++i) sh->Entries[i] = tmp[i];
        // v3 publish - palette names resolved via engine StringId_GetString.
        memcpy(sh->Header.PaletteNames, paletteNames, sizeof(paletteNames));
        sh->Header.PaletteCount = resolvedPaletteCount;
        sh->Header.EntryCount   = outCount;
        sh->Header.WriteCounter = sh->Header.WriteCounter + 1u;
    } __except (EXCEPTION_EXECUTE_HANDLER) {}

    // ---- stability verification before the 30s lockout ----
    // The walk above publishes every time, but the tag table can still be
    // repopulating right after a (re)load, so a single walk can read
    // plausible-but-garbage entries. Only TRUST the result - and enter the
    // long lockout - when two consecutive walks AGREE (same entry count + first
    // tag id) AND the map epoch held steady across the walk. Until then,
    // re-walk next tick (g_RetryCountdown=0 so the scnr-stable gate doesn't
    // skip). This defeats the "walked too early against half-loaded tags" race
    // (which would otherwise freeze garbage for 30s).
    uint32_t firstTag   = (outCount > 0) ? tmp[0].TagGlobalId : 0u;
    uint64_t epochAfter = HaloMapStudio::Engine::GetMapEpoch();
    // Count + epoch steady across consecutive walks already proves the map is
    // loaded and the palette is fully populated. The firstTag agreement is a
    // belt-and-suspenders extra that can legitimately never converge (flaky
    // cross-process read of entry 0), so we must NOT block on it forever.
    bool countEpochStable = (outCount == g_VerifyEntryCount) && (epochAfter == curEpoch);
    bool tagStable        = countEpochStable && (firstTag == g_VerifyFirstTag);
    if (countEpochStable) ++g_StableCountStreak; else g_StableCountStreak = 0;
    g_VerifyEntryCount = outCount;
    g_VerifyFirstTag   = firstTag;

    // Accept when firstTag agrees (fast path, 2 walks) OR when count+epoch have
    // held steady for a few PACED walks (firstTag-independent fallback).
    if (tagStable || g_StableCountStreak >= 3) {
        g_LastScenarioDatum = scnrDatum;
        g_LastPublishEpoch  = epochAfter;
        g_RetryCountdown    = 1800; // ~30s - palette only changes on map/forge transition
        g_StableCountStreak = 0;
        ZH_Logf("[ForgePaletteSnapshot] scnr=0x%08X palettes=%u entries=%u namedPalettes=%u STABLE (epoch=%llu via=%s)\n",
                scnrDatum, palBlock.count, outCount, resolvedPaletteCount,
                (unsigned long long)epochAfter, tagStable ? "tag" : "count");
    } else {
        // Not yet trusted - re-walk, but PACED (g_RetryCountdown>0 so the
        // scnr-stable gate skips the intervening ticks) instead of re-walking
        // all entries EVERY tick (the old g_RetryCountdown=0 was a per-frame
        // RPM palette walk = a constant stutter).
        g_LastScenarioDatum = scnrDatum;
        g_RetryCountdown    = 30;
        ZH_Logf("[ForgePaletteSnapshot] scnr=0x%08X entries=%u UNVERIFIED (re-walking paced; streak=%u epoch=%llu/%llu)\n",
                scnrDatum, outCount, g_StableCountStreak,
                (unsigned long long)curEpoch, (unsigned long long)epochAfter);
        return;
    }
    // Surface resolved palette names so we can audit which scnr produced
    // which categories without attaching a debugger.
    for (uint32_t p = 0; p < resolvedPaletteCount && p < kMaxPalettes; ++p) {
        if (paletteNames[p][0] != '\0')
            ZH_Logf("[ForgePaletteSnapshot]   palette[%u] name='%s'\n", p, paletteNames[p]);
    }
}
