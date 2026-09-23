// ChangeColorWalker.cpp
// =============================================================================
// Native reader for the Halo Reach TEAM / player "change colors" - the 8-ish
// predefined RGB tints (red, blue, green, orange, purple, gold, brown, pink)
// the engine multiplies into team-tinted materials (spawn points, hill markers,
// light strips). Previously the Rust viewer hard-coded an APPROXIMATION of these
// (scene.rs CHANGE_COLORS); this walker pulls the AUTHORITATIVE values straight
// from the cache globals tag so the hue/saturation match the game exactly.
//
// Where the values live (discovered empirically via the DUMP path below, then
// pinned): the 8 predefined player colors are a tagblock inside game_globals
// ('matg') - a small array whose entries each carry RGB float3 triples. The
// discovery dump (MMS_NATIVE_LOG=1) walks every first-level tagblock in matg
// and mulg and flags any block that decodes as an array of color-like float3s;
// the constants CC_* below record the winning block offset / stride / rgb sub-
// offset once confirmed against the log.
//
// Defensive contract mirrors the other walkers: every cross-tag deref is bounds-
// checked against cache->size and wrapped in SEH so a schema mismatch returns
// "not found" (Rust falls back to the built-in table) rather than crashing.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>

using namespace zh_mcc;

#pragma pack(push, 1)
// Public ABI surface - must match the Rust ZhChangeColors mirror in
// crates/hms-native/src/lib.rs.
struct ZH_ChangeColors {
    // 11 RGB triples (LINEAR, 0..1) in the canonical engine Color-enum order:
    // 0 red,1 blue,2 green,3 orange,4 purple,5 gold,6 brown,7 pink,
    // 8 white/neutral,9 black,10 zombie. Entries 8..10 are engine-synthetic
    // (not authored in the tag) and are filled with sensible defaults so the
    // Rust side can copy all 11 verbatim.
    float    Rgb[11][3];
    uint32_t Count;   // number of AUTHORED colors read from the tag (normally 8)
    uint32_t Found;   // 1 = tag block located & read, 0 = fell back to defaults
};
#pragma pack(pop)

namespace {

constexpr const char* TC_MATG = "matg";
constexpr const char* TC_MULG = "mulg";

// -------------------------------------------------------------------------
// Confirmed layout (game_globals 'matg' - Reach MCC).
//
// The 8 predefined team/player "change colors" live in a first-level tagblock
// at matg meta + 0x4D4. Byte-verified via the DUMP path (MMS_NATIVE_LOG=1) on
// forge_halo (Hemorrhage): the block header is {count=8, ptr}; each entry is a
// 16-byte `real_argb_color` = [leading float (unused, 0), R, G, B]. Decoding at
// stride 16 with RGB at entry+4 yields exactly the engine Color-enum order:
//
//   [0] red    (0.5029, 0.0437, 0.0437)   [1] blue   (0.0902, 0.2157, 0.5255)
//   [2] green  (0.2039, 0.3176, 0.0627)   [3] orange (0.8431, 0.3490, 0.0431)
//   [4] purple (0.2078, 0.1216, 0.5490)   [5] gold   (0.6902, 0.8706, 0.1059)
//   [6] brown  (0.4706, 0.4000, 0.1059)   [7] pink   (0.8995, 0.6173, 0.6746)
//
// Values are exact byte/255 fractions (artist-authored sRGB); the viewer stores
// them directly (same space the old approximations used). CC_BLOCK_OFF is the
// pinned offset; a tightened heuristic scan (matching the red-first/blue-second/
// orange-fourth signature) is the fallback if a future build drifts the offset.
// -------------------------------------------------------------------------
constexpr int   CC_EXPECTED   = 8;      // authored team/change colors
constexpr int   CC_BLOCK_OFF  = 0x4D4;  // matg-relative tagblock header offset
constexpr int   CC_ENTRY_SIZE = 16;     // real_argb_color (A,R,G,B floats)
constexpr int   CC_RGB_OFF    = 4;      // RGB begins after the leading float

// The team-color SIGNATURE: entry[0] red-dominant, entry[1] blue-dominant,
// entry[3] orange (r>g>b, r bright). This uniquely identifies the change-color
// palette and rejects unrelated color arrays (e.g. the 30-entry HUD gradient
// block that also lives in matg).
bool MatchesTeamSignature(const float rgb[][3], int count) {
    if (count < CC_EXPECTED) return false;
    float r0=rgb[0][0], g0=rgb[0][1], b0=rgb[0][2];
    float r1=rgb[1][0], g1=rgb[1][1], b1=rgb[1][2];
    float r3=rgb[3][0], g3=rgb[3][1], b3=rgb[3][2];
    bool red0    = (r0 > g0 + 0.10f) && (r0 > b0 + 0.10f) && (r0 > 0.2f);
    bool blue1   = (b1 > r1 + 0.10f) && (b1 > g1 + 0.05f) && (b1 > 0.2f);
    bool orange3 = (r3 > g3 + 0.05f) && (g3 >= b3) && (r3 > 0.4f);
    return red0 && blue1 && orange3;
}

// Read `count` entries from `blk` at stride CC_ENTRY_SIZE, RGB at +CC_RGB_OFF,
// into rgbOut. Returns false on OOB. Does not validate the colors.
bool ReadArgbBlock(const uint8_t* blk, int count, size_t avail,
                   float rgbOut[][3], int maxOut) {
    if (count <= 0 || count > maxOut) return false;
    if ((size_t)count * CC_ENTRY_SIZE > avail) return false;
    for (int i = 0; i < count; ++i) {
        const uint8_t* e = blk + (size_t)i * CC_ENTRY_SIZE + CC_RGB_OFF;
        memcpy(&rgbOut[i][0], e + 0, 4);
        memcpy(&rgbOut[i][1], e + 4, 4);
        memcpy(&rgbOut[i][2], e + 8, 4);
    }
    return true;
}

// Commit up to 8 authored colors (indices 0..7) into `out`. Leaves 8..10 for
// the synthetic tail (filled by the caller).
void CommitColors(const float rgb[][3], int count, ZH_ChangeColors* out) {
    int n = count < CC_EXPECTED ? count : CC_EXPECTED;
    for (int i = 0; i < n; ++i) {
        out->Rgb[i][0] = rgb[i][0];
        out->Rgb[i][1] = rgb[i][1];
        out->Rgb[i][2] = rgb[i][2];
    }
    out->Count = (uint32_t)n;
    out->Found = 1;
}

// Locate + read the 8 change colors from a globals tag. Tries the pinned offset
// (matg+0x4D4) first, then a tightened team-signature scan as the drift-safe
// fallback. Emits a NativeDiag trace of the accepted block (MMS_NATIVE_LOG).
bool ScanGlobalsForColors(CacheHandle* cache, uint32_t tagIdx,
                          const char* tagLabel, ZH_ChangeColors* out)
{
    if (tagIdx >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[tagIdx];
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + CC_BLOCK_OFF + 8 > cache->size) return false;
    const uint8_t* meta = cache->base + metaOff;

    float rgb[32][3];

    // --- Primary: the pinned block header at meta + CC_BLOCK_OFF. ---
    {
        int32_t  count = R32(meta + CC_BLOCK_OFF);
        uint32_t ptr   = RU32(meta + CC_BLOCK_OFF + 4);
        int64_t  blkOff = TagMetaFileOff(cache, ptr);
        if (count >= CC_EXPECTED && count <= 16 && blkOff >= 0 &&
            (size_t)blkOff < cache->size) {
            size_t avail = cache->size - (size_t)blkOff;
            if (ReadArgbBlock(cache->base + blkOff, count, avail, rgb, 32) &&
                MatchesTeamSignature(rgb, count)) {
                CommitColors(rgb, count, out);
                NativeDiag("ChangeColor[ACCEPT] %s pinned block@meta+0x%X count=%d "
                           "red=(%.3f,%.3f,%.3f) blue=(%.3f,%.3f,%.3f) orange=(%.3f,%.3f,%.3f)",
                           tagLabel, CC_BLOCK_OFF, count,
                           rgb[0][0],rgb[0][1],rgb[0][2], rgb[1][0],rgb[1][1],rgb[1][2],
                           rgb[3][0],rgb[3][1],rgb[3][2]);
                return true;
            }
        }
    }

    // --- Fallback: scan first-level headers for the team-color signature. ---
    const int WINDOW = 0x6000;
    size_t maxScan = (size_t)WINDOW;
    if ((size_t)metaOff + maxScan > cache->size)
        maxScan = cache->size - (size_t)metaOff;
    NativeDiag("ChangeColor[SCAN] %s pinned miss; scanning metaOff=%lld window=0x%zX",
               tagLabel, (long long)metaOff, maxScan);

    for (size_t o = 0; o + 8 <= maxScan; o += 4) {
        int32_t  count = R32(meta + o);
        uint32_t ptr   = RU32(meta + o + 4);
        if (count < CC_EXPECTED || count > 16) continue;
        int64_t blkOff = TagMetaFileOff(cache, ptr);
        if (blkOff < 0 || (size_t)blkOff >= cache->size) continue;
        size_t avail = cache->size - (size_t)blkOff;
        if (!ReadArgbBlock(cache->base + blkOff, count, avail, rgb, 32)) continue;
        if (!MatchesTeamSignature(rgb, count)) continue;
        CommitColors(rgb, count, out);
        NativeDiag("ChangeColor[ACCEPT] %s scan block@meta+0x%zX count=%d (team-signature)",
                   tagLabel, o, count);
        return true;
    }
    return false;
}

// Engine-synthetic defaults for the neutral/black/zombie slots (8..10) that the
// tag does not author. Kept identical to the Rust built-in table so a partial
// read (8 authored + 3 synthetic) is seamless.
void FillSyntheticTail(ZH_ChangeColors* out) {
    const float tail[3][3] = {
        {1.0f, 1.0f, 1.0f},  // 8 white / neutral
        {0.05f, 0.05f, 0.05f}, // 9 black
        {0.40f, 0.50f, 0.20f}, // 10 zombie
    };
    for (int i = 8; i < 11; ++i) {
        out->Rgb[i][0] = tail[i-8][0];
        out->Rgb[i][1] = tail[i-8][1];
        out->Rgb[i][2] = tail[i-8][2];
    }
}

bool GetChangeColorsInner(CacheHandle* cache, ZH_ChangeColors* out) {
    memset(out, 0, sizeof(*out));
    FillSyntheticTail(out);

    int matg = FindGlobalTag(cache, TC_MATG);
    int mulg = FindGlobalTag(cache, TC_MULG);
    NativeDiag("ChangeColor: matg=%d mulg=%d", matg, mulg);

    bool found = false;
    if (matg >= 0) found = ScanGlobalsForColors(cache, (uint32_t)matg, "matg", out);
    if (!found && mulg >= 0) found = ScanGlobalsForColors(cache, (uint32_t)mulg, "mulg", out);

    // Re-fill the synthetic tail in case a short authored read wrote fewer than
    // 8 entries (leaves 8..10 as our defaults regardless).
    FillSyntheticTail(out);
    return found;
}

} // anonymous namespace

// =============================================================================
// Public export
// =============================================================================
extern "C" __declspec(dllexport) bool __stdcall ZH_GLOBALS_GetChangeColors(
    uint64_t cacheHandle, ZH_ChangeColors* out)
{
    if (!out) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) { memset(out, 0, sizeof(*out)); return false; }
    __try {
        return GetChangeColorsInner(cache, out);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        memset(out, 0, sizeof(*out));
        FillSyntheticTail(out);
        return false;
    }
}
