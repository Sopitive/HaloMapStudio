// HaloReachTagHelpers.h
// =============================================================================
// Shared helpers for walking the haloreach.dll tag table from inside the
// injected DLL. Centralises the tag-address resolver, contracted-pointer
// expansion, fourcc tag-ref scanning and the chain (primary -> 'hlmt' ->
// 'mode') that several modules need.
//
// This header hosts the canonical inline implementations; modules MUST
// include it instead of pasting their own copies.
//
// Constraints:
//   * SEH-fence every cross-process pointer dereference. Tag memory can
//     transiently fault while the engine reloads tags between maps.
//   * No global static initialisers that touch the engine - the helpers
//     are pure functions; callers decide when to invoke.
// =============================================================================

#pragma once

#include <windows.h>
#include <cstdint>

namespace ZeroHour { namespace HrTag {

inline constexpr uintptr_t kRva_TagTablePtr = 0xC1A600;
inline constexpr uintptr_t kRva_TagSegments = 0x4E39F20;

// Big-endian fourcc constant (Halo tag groups are stored that way in memory).
inline constexpr uint32_t Fourcc(char a, char b, char c, char d)
{
    return ((uint32_t)(uint8_t)a << 24) |
           ((uint32_t)(uint8_t)b << 16) |
           ((uint32_t)(uint8_t)c <<  8) |
           ((uint32_t)(uint8_t)d);
}

inline constexpr uint32_t kFourcc_hlmt = Fourcc('h','l','m','t');
inline constexpr uint32_t kFourcc_mode = Fourcc('m','o','d','e');
inline constexpr uint32_t kFourcc_scnr = Fourcc('s','c','n','r');

// In-memory tagblock layout (12 bytes): count, contracted-ptr, unk.
struct Tagblock { uint32_t count; uint32_t ptr; uint32_t unk; };

// Resolve haloreach.dll module base; returns nullptr if not yet loaded.
inline uint8_t* ReachBase()
{
    HMODULE m = GetModuleHandleW(L"haloreach.dll");
    return (uint8_t*)m;
}

// Expand a contracted (28-bit-segment-tagged) pointer into a full VA.
// Returns 0 on any read fault or when the segment base is not yet populated.
inline uint64_t ExpandContracted(uint32_t contracted)
{
    if (contracted == 0u || contracted == 0xFFFFFFFFu) return 0;
    uint8_t* base = ReachBase();
    if (!base) return 0;
    uint32_t segIdx = (contracted >> 28) & 0xFu;
    uint64_t segBase = 0;
    __try { segBase = *(uint64_t*)(base + kRva_TagSegments + (uint64_t)segIdx * 8); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
    if (!segBase) return 0;
    return segBase + ((uint64_t)contracted << 2);
}

// Resolve a tag datum -> in-process VA via the engine's tag table.
// Returns 0 for invalid/zero datums or if the tag table isn't ready yet.
inline uint64_t ResolveTagAddress(uint32_t datum)
{
    if (datum == 0u || datum == 0xFFFFFFFFu) return 0;
    uint8_t* base = ReachBase();
    if (!base) return 0;
    uint64_t tagTable = 0;
    __try { tagTable = *(uint64_t*)(base + kRva_TagTablePtr); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
    if (!tagTable) return 0;

    uint32_t index = datum & 0xFFFFu;
    uint32_t contracted = 0;
    __try { contracted = *(uint32_t*)(tagTable + 4 + (uint64_t)index * 8); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
    return ExpandContracted(contracted);
}

// Read a tagblock header at fieldVA. Returns false on a read fault.
inline bool ReadTagblock(uint64_t fieldVA, Tagblock& out)
{
    out = {};
    if (!fieldVA) return false;
    __try {
        out.count = *(uint32_t*)(fieldVA + 0);
        out.ptr   = *(uint32_t*)(fieldVA + 4);
        out.unk   = *(uint32_t*)(fieldVA + 8);
    } __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
    return true;
}

// Scan the first `maxBytes` of a tag's payload for a tag-ref whose primary
// fourcc matches `fourccBE`; return the datum stored at +0xC of the ref, or 0
// if nothing matched. Tag refs are 16 bytes:
//   +0x00 u32 primary fourcc (BE)
//   +0x04 u32 secondary
//   +0x08 u32 name offset
//   +0x0C u32 datum
inline uint32_t FindTagRefDatum(uint64_t tagVA, uint32_t fourccBE, uint32_t maxBytes = 0x400)
{
    if (!tagVA) return 0;
    uint32_t out = 0;
    __try {
        uint32_t* p = (uint32_t*)tagVA;
        uint32_t dwords = maxBytes / 4;
        if (dwords < 4) return 0;
        dwords -= 3;
        for (uint32_t i = 0; i < dwords; ++i) {
            if (p[i] == fourccBE) { out = p[i + 3]; break; }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) { out = 0; }
    return out;
}

// Convenience: resolve obj-tag -> hlmt datum -> mode datum.
// Either output can be 0 if the chain breaks; non-zero outputs are well-formed
// datums but NOT guaranteed to resolve via ResolveTagAddress (caller decides).
// NOTE: when calling this in a tight loop (e.g. forge palette walks),
// prefer ResolveHlmtAndModeDatums_Cached below to avoid frame stutters.
inline void ResolveHlmtAndModeDatums(uint32_t primaryTagDatum,
                                     uint32_t* outHlmtDatum,
                                     uint32_t* outModeDatum)
{
    if (outHlmtDatum) *outHlmtDatum = 0;
    if (outModeDatum) *outModeDatum = 0;
    if (primaryTagDatum == 0u || primaryTagDatum == 0xFFFFFFFFu) return;
    uint64_t tagVA = ResolveTagAddress(primaryTagDatum);
    if (!tagVA) return;
    uint32_t hlmt = FindTagRefDatum(tagVA, kFourcc_hlmt, 0x400);
    if (outHlmtDatum) *outHlmtDatum = hlmt;
    if (hlmt == 0u || hlmt == 0xFFFFFFFFu) return;
    uint64_t hlmtVA = ResolveTagAddress(hlmt);
    if (!hlmtVA) return;
    uint32_t mode = FindTagRefDatum(hlmtVA, kFourcc_mode, 0x100);
    if (outModeDatum) *outModeDatum = mode;
}

// Cached variant. Memoises (primaryTag -> hlmt, mode) for the lifetime of
// the loaded haloreach.dll module via an open-addressed hash table.
struct HlmtModeCacheEntry { uint32_t key; uint32_t hlmt; uint32_t mode; };
inline thread_local HlmtModeCacheEntry tls_HlmtModeCache[1024] = {};
inline thread_local HMODULE             tls_HlmtModeCache_LastHr = nullptr;
// Per-thread observed map epoch. The HaloMapStudioDLL ResolveObjectPoolDescriptor
// path bumps a global epoch when the engine swaps to a new scenario; we
// snapshot it here so tag-resolution caches can wholesale-clear at the same
// time the descriptor cache does. Defined as a `uint64_t` so HaloMapStudioDLL's
// `HaloMapStudio::Engine::GetMapEpoch()` (also `uint64_t`) can be assigned
// directly. Initialise to 0 so the first frame after DLL load detects a
// mismatch against the >=1 global epoch and flushes - ensures we never
// leak a tls-init zero state into a real lookup.
inline thread_local uint64_t            tls_HlmtModeCache_LastEpoch = 0;

// Flushes the TLS cache from outside without re-entering the resolver.
// Called when the snapshot publisher detects a map change.
inline void ResolveHlmtAndModeDatums_FlushCache()
{
    for (auto& e : tls_HlmtModeCache) { e.key = 0; e.hlmt = 0; e.mode = 0; }
    tls_HlmtModeCache_LastHr = nullptr;
    tls_HlmtModeCache_LastEpoch = 0;
}

inline void ResolveHlmtAndModeDatums_Cached(uint32_t primaryTagDatum,
                                            uint32_t* outHlmtDatum,
                                            uint32_t* outModeDatum)
{
    if (outHlmtDatum) *outHlmtDatum = 0;
    if (outModeDatum) *outModeDatum = 0;
    if (primaryTagDatum == 0u || primaryTagDatum == 0xFFFFFFFFu) return;

    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (hr != tls_HlmtModeCache_LastHr) {
        for (auto& e : tls_HlmtModeCache) { e.key = 0; e.hlmt = 0; e.mode = 0; }
        tls_HlmtModeCache_LastHr = hr;
        // Reset epoch tracker too - module change implies a fresh epoch.
        tls_HlmtModeCache_LastEpoch = 0;
    }
    if (!hr) return;

    uint32_t h = primaryTagDatum * 2654435761u;
    constexpr uint32_t kCap = 1024u;
    constexpr uint32_t kMask = kCap - 1u;
    for (uint32_t probe = 0; probe < kCap; ++probe) {
        uint32_t slot = (h + probe) & kMask;
        HlmtModeCacheEntry& e = tls_HlmtModeCache[slot];
        if (e.key == primaryTagDatum) {
            if (outHlmtDatum) *outHlmtDatum = e.hlmt;
            if (outModeDatum) *outModeDatum = e.mode;
            return;
        }
        if (e.key == 0) {
            uint32_t hlmt = 0, mode = 0;
            ResolveHlmtAndModeDatums(primaryTagDatum, &hlmt, &mode);
            e.key  = primaryTagDatum;
            e.hlmt = hlmt;
            e.mode = mode;
            if (outHlmtDatum) *outHlmtDatum = hlmt;
            if (outModeDatum) *outModeDatum = mode;
            return;
        }
    }
    ResolveHlmtAndModeDatums(primaryTagDatum, outHlmtDatum, outModeDatum);
}

} } // namespace ZeroHour::HrTag
