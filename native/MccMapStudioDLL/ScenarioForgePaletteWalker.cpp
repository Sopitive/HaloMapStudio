// ScenarioForgePaletteWalker.cpp
// =============================================================================
// Offline-cache analog of ForgePaletteSnapshot.cpp. The latter walks Reach's
// scnr forge palette from MCC's LIVE process memory; this one reads the same
// layout from a CACHE FILE (the opened .map on disk) so MMS can resolve the
// scenario's (PaletteIndex, EntryIndex, VariantIndex) -> tagId mapping for
// standalone .mvar loads where MCC isn't running.
//
// Scenario forge palette layout (per ForgePaletteSnapshot.cpp confirmed
// against U10/U13 runtime memory):
//
//   scnr + 0x228   tagblock(palette[0x14])
//     palette[p]:
//       +0x00      StringId  Name
//       +0x08      tagblock(entry[0x1C])
//     entry[e]:
//       +0x00      StringId  Name
//       +0x04      tagblock(variant[0x18])
//     variant[v]:
//       +0x00      StringId  Name
//       +0x04      tag_reference -> obje-derived (+0x00 fourcc, +0x0C tagId)
//
// Same scnr+0x228 offset confirmed on U10 + U13 (runtime version). Cache
// meta-file layout matches; tagblock count+pointer field has the same
// 12-byte shape (4-byte count, 4-byte unused, 4-byte pointer-low).
//
// Public export:
//   uint32_t ZH_SCNR_EnumerateForgePalette(
//       uint64_t cacheHandle,
//       uint32_t scnrTagId,
//       ZH_ScnrForgePaletteEntry* outEntries,  // may be null to size-query
//       uint32_t maxCount);                    // returns entries written (or
//                                              // total available when outEntries=null)
//
// Caps: 512 entries (matches the runtime ForgePaletteSnapshot kMaxEntries).
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>

using namespace zh_mcc;

namespace {

constexpr int    SCNR_FORGE_PALETTE_OFFSET = 0x228;
constexpr int    SCNR_PALETTE_STRIDE       = 0x14;
constexpr int    SCNR_PALETTE_NAME_OFFSET  = 0x00;
constexpr int    SCNR_PALETTE_ENTRIES_OFFSET = 0x08;
constexpr int    SCNR_ENTRY_STRIDE         = 0x1C;
constexpr int    SCNR_ENTRY_NAME_OFFSET    = 0x00;
constexpr int    SCNR_ENTRY_VARIANTS_OFFSET = 0x04;
constexpr int    SCNR_VARIANT_STRIDE       = 0x18;
constexpr int    SCNR_VARIANT_NAME_OFFSET  = 0x00;
constexpr int    SCNR_VARIANT_TAGREF_OFFSET = 0x04;

// Sanity caps - defend against garbage tagblock descriptors.
constexpr int    PALETTE_COUNT_SANITY      = 64;
constexpr int    ENTRY_COUNT_SANITY        = 1024;
constexpr int    VARIANT_COUNT_SANITY      = 32;
constexpr uint32_t TOTAL_ENTRY_CAP         = 1024;

constexpr const char* TC_SCNR = "scnr";

#pragma pack(push, 1)
struct ZH_ScnrForgePaletteEntry {
    uint32_t paletteIndex;
    uint32_t entryWithinPalette;
    uint32_t variantWithinEntry;
    uint32_t tagGroupMagic;        // big-endian fourcc as stored
    uint32_t tagGlobalId;          // raw datum from tag_ref+0x0C
    uint32_t tagShortId;           // tagGlobalId & 0xFFFF - the cache tag-table index
    char     paletteName[40];      // resolved from cache stringTable
    char     entryName[40];
    char     variantName[40];
    uint32_t variantNameSid;       // #269: raw variant Name stringId (variant+0x00) - the key the
                                   // engine uses to select the hlmt model variant; Rust matches it
                                   // against ZhAttachment.variant_name_sid + the variant section mask.
};
#pragma pack(pop)

static_assert(sizeof(ZH_ScnrForgePaletteEntry) == 4 + 4 + 4 + 4 + 4 + 4 + 40 + 40 + 40 + 4,
              "ZH_ScnrForgePaletteEntry layout drift");

void CopyResolvedString(CacheHandle* cache, uint32_t stringId, char* outBuf, size_t outBufLen)
{
    if (outBuf == nullptr || outBufLen == 0) return;
    outBuf[0] = '\0';
    if (stringId == 0u || stringId == 0xFFFFFFFFu) return;
    const char* resolved = ResolveStringId(cache, (int32_t)stringId);
    if (resolved == nullptr) return;
    size_t n = strlen(resolved);
    if (n >= outBufLen) n = outBufLen - 1;
    memcpy(outBuf, resolved, n);
    outBuf[n] = '\0';
}

} // namespace

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_SCNR_EnumerateForgePalette(
    uint64_t cacheHandle,
    uint32_t scnrTagId,
    ZH_ScnrForgePaletteEntry* outEntries,
    uint32_t maxCount)
{
    // The opaque ulong handle is a token into the DLL's handle table, NOT
    // a raw CacheHandle* - casting directly crashes. Same pattern every
    // other walker uses.
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (cache == nullptr || cache->base == nullptr) return 0;
    if (scnrTagId >= cache->tags.size()) return 0;

    const TagEntry& scnrEntry = cache->tags[scnrTagId];
    if (memcmp(scnrEntry.classCode, TC_SCNR, 4) != 0) return 0;

    int64_t scnrMetaOff = TagMetaFileOff(cache, scnrEntry.metaPointerRaw);
    if (scnrMetaOff < 0) return 0;
    if ((size_t)scnrMetaOff + SCNR_FORGE_PALETTE_OFFSET + 12 > cache->size) return 0;
    const uint8_t* scnrMeta = cache->base + scnrMetaOff;

    // Top-level palette tagblock @ scnr+0x228.
    TagBlockRef paletteBlock = ReadTagBlock(scnrMeta + SCNR_FORGE_PALETTE_OFFSET);
    if (paletteBlock.count <= 0 || paletteBlock.count > PALETTE_COUNT_SANITY)
        return 0;
    int64_t paletteArrOff = TagMetaFileOff(cache, paletteBlock.pointer);
    if (paletteArrOff < 0) return 0;
    if ((size_t)paletteArrOff + (size_t)paletteBlock.count * SCNR_PALETTE_STRIDE > cache->size)
        return 0;
    const uint8_t* paletteArr = cache->base + paletteArrOff;

    uint32_t written = 0;

    for (int p = 0; p < paletteBlock.count; ++p)
    {
        if (written >= TOTAL_ENTRY_CAP) break;

        const uint8_t* palette = paletteArr + (size_t)p * SCNR_PALETTE_STRIDE;
        uint32_t paletteNameSid = RU32(palette + SCNR_PALETTE_NAME_OFFSET);

        char paletteNameBuf[40] = {};
        CopyResolvedString(cache, paletteNameSid, paletteNameBuf, sizeof(paletteNameBuf));

        TagBlockRef entryBlock = ReadTagBlock(palette + SCNR_PALETTE_ENTRIES_OFFSET);
        if (entryBlock.count <= 0 || entryBlock.count > ENTRY_COUNT_SANITY) continue;
        int64_t entryArrOff = TagMetaFileOff(cache, entryBlock.pointer);
        if (entryArrOff < 0) continue;
        if ((size_t)entryArrOff + (size_t)entryBlock.count * SCNR_ENTRY_STRIDE > cache->size)
            continue;
        const uint8_t* entryArr = cache->base + entryArrOff;

        for (int e = 0; e < entryBlock.count; ++e)
        {
            if (written >= TOTAL_ENTRY_CAP) break;

            const uint8_t* entry = entryArr + (size_t)e * SCNR_ENTRY_STRIDE;
            uint32_t entryNameSid = RU32(entry + SCNR_ENTRY_NAME_OFFSET);

            char entryNameBuf[40] = {};
            CopyResolvedString(cache, entryNameSid, entryNameBuf, sizeof(entryNameBuf));

            // A .mvar object's variant_quota_index is a DIRECT positional index into the flat
            // list of ENTRY slots (category-major, up to 256). We MUST emit a row for every
            // entry slot - including empty ones (no variant block / all-invalid tags) - or the
            // flat index shifts and high-index objects resolve to the wrong type (or vanish).
            int emittedForEntry = 0;
            TagBlockRef variantBlock = ReadTagBlock(entry + SCNR_ENTRY_VARIANTS_OFFSET);
            int64_t variantArrOff = (variantBlock.count > 0 && variantBlock.count <= VARIANT_COUNT_SANITY)
                ? TagMetaFileOff(cache, variantBlock.pointer) : -1;
            bool variantsOk = variantArrOff >= 0 &&
                (size_t)variantArrOff + (size_t)variantBlock.count * SCNR_VARIANT_STRIDE <= cache->size;
            if (variantsOk)
            {
                const uint8_t* variantArr = cache->base + variantArrOff;
                for (int v = 0; v < variantBlock.count; ++v)
                {
                    if (written >= TOTAL_ENTRY_CAP) break;

                    const uint8_t* variant = variantArr + (size_t)v * SCNR_VARIANT_STRIDE;
                    uint32_t variantNameSid = RU32(variant + SCNR_VARIANT_NAME_OFFSET);

                    // tag_reference is 16 bytes: fourcc@+0, padding@+4..11, tagId@+12.
                    uint32_t groupBE  = RU32(variant + SCNR_VARIANT_TAGREF_OFFSET + 0x00);
                    uint32_t tagDatum = RU32(variant + SCNR_VARIANT_TAGREF_OFFSET + 0x0C);
                    if (tagDatum == 0u || tagDatum == 0xFFFFFFFFu) continue;
                    uint32_t tagShort = tagDatum & 0xFFFFu;
                    if (tagShort >= cache->tags.size()) continue;

                    if (outEntries != nullptr && written < maxCount)
                    {
                        ZH_ScnrForgePaletteEntry& dst = outEntries[written];
                        memset(&dst, 0, sizeof(dst));
                        dst.paletteIndex       = (uint32_t)p;
                        dst.entryWithinPalette = (uint32_t)e;
                        dst.variantWithinEntry = (uint32_t)v;
                        dst.tagGroupMagic      = groupBE;
                        dst.tagGlobalId        = tagDatum;
                        dst.tagShortId         = tagShort;
                        memcpy(dst.paletteName, paletteNameBuf, sizeof(dst.paletteName));
                        memcpy(dst.entryName,   entryNameBuf,   sizeof(dst.entryName));
                        CopyResolvedString(cache, variantNameSid, dst.variantName, sizeof(dst.variantName));
                        // #269 RE-CONFIRMED: the scnr forge-palette variant block has TWO sids - 
                        // @0x00 is the forge DISPLAY name ("warthog_rocket"), @0x14 is the actual
                        // hlmt MODEL-variant Name sid ("rocket"=1728) the engine uses to select the
                        // body permutations + attachments. Thread the @0x14 sid so it matches the
                        // hlmt variant Name (== ZhAttachment.variant_name_sid). 0 => default variant.
                        dst.variantNameSid = RU32(variant + 0x14);
                    }
                    ++written;
                    ++emittedForEntry;
                }
            }
            // Placeholder row so the entry SLOT is preserved in the flat index even when it
            // has no renderable variant (tagShortId 0 -> caller skips rendering it).
            if (emittedForEntry == 0 && written < TOTAL_ENTRY_CAP)
            {
                if (outEntries != nullptr && written < maxCount)
                {
                    ZH_ScnrForgePaletteEntry& dst = outEntries[written];
                    memset(&dst, 0, sizeof(dst));
                    dst.paletteIndex       = (uint32_t)p;
                    dst.entryWithinPalette = (uint32_t)e;
                    dst.variantWithinEntry = 0;
                    dst.tagShortId         = 0; // empty slot
                    memcpy(dst.paletteName, paletteNameBuf, sizeof(dst.paletteName));
                    memcpy(dst.entryName,   entryNameBuf,   sizeof(dst.entryName));
                }
                ++written;
            }
        }
    }
    return written;
}
