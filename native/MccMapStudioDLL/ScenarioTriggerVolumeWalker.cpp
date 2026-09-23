// ScenarioTriggerVolumeWalker.cpp
// =============================================================================
// Reads Halo: Reach scenario (scnr) TRIGGER VOLUMES from the opened map cache
// so HaloMapStudio can draw toggleable wireframe boxes for:
//   * KILL volumes        (scnr+0x4C8 index block -> Trigger Volumes)
//   * SAFE-ZONE volumes   (scnr+0x4D4 index block -> Trigger Volumes)
//   * plain trigger volumes (referenced by neither)
//
// Key structural fact:
// the kill/safe blocks hold NO geometry - they are tiny index blocks whose
// int16 selects a Trigger Volume. The geometry lives once in the Trigger
// Volumes block.
//
//   scnr + 0x280   tagblock(triggerVolume[0x7C])   -- the oriented boxes
//     triggerVolume[i]:
//       +0x00  StringId  Name
//       +0x0C  enum16    Type  { 0 = Bounding Box, 1 = Sector }
//       +0x10  float32x3 Forward
//       +0x1C  float32x3 Up
//       +0x28  float32x3 Position  (box center, world space)
//       +0x34  float32x3 Extents   (size along Forward/Up/Right)
//   scnr + 0x4C8   tagblock(killIndex[0x04])    int16@+0 = triggerVolume index
//   scnr + 0x4D4   tagblock(safeIndex[0x04])    int16@+0 = triggerVolume index
//
// Soft ceilings (scnr+0x25C metadata -> sddt triangles) are NOT handled here - 
// they're planar triangle soup in a different tag, not oriented boxes.
//
// Public export:
//   uint32_t ZH_SCNR_EnumerateTriggerVolumes(
//       uint64_t cacheHandle, uint32_t scnrTagId,
//       ZH_ScnrTriggerVolume* outVolumes, uint32_t maxCount);
//   (outVolumes=null -> size query)
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>

using namespace zh_mcc;

namespace {

constexpr int SCNR_TRIGGER_VOLUMES_OFFSET = 0x280;
constexpr int SCNR_TRIGGER_VOLUME_STRIDE  = 0x7C;
constexpr int TV_NAME_OFFSET     = 0x00;  // StringId
constexpr int TV_TYPE_OFFSET     = 0x0C;  // enum16 (0=box, 1=sector)
constexpr int TV_FORWARD_OFFSET  = 0x10;  // f32x3
constexpr int TV_UP_OFFSET       = 0x1C;  // f32x3
constexpr int TV_POSITION_OFFSET = 0x28;  // f32x3
constexpr int TV_EXTENTS_OFFSET  = 0x34;  // f32x3

constexpr int SCNR_KILL_INDEX_OFFSET = 0x4C8;
constexpr int SCNR_SAFE_INDEX_OFFSET = 0x4D4;
constexpr int INDEX_BLOCK_STRIDE     = 0x04;
constexpr int INDEX_VOLUME_OFFSET    = 0x00;  // int16

// Category bitmask.
constexpr uint32_t CAT_PLAIN = 0u;
constexpr uint32_t CAT_KILL  = 1u;
constexpr uint32_t CAT_SAFE  = 2u;

constexpr int      TV_COUNT_SANITY = 4096;
constexpr uint32_t TOTAL_CAP       = 2048;

constexpr const char* TC_SCNR = "scnr";

inline float RF32(const uint8_t* p) { float v; memcpy(&v, p, 4); return v; }

#pragma pack(push, 1)
struct ZH_ScnrTriggerVolume {
    uint32_t category;       // bitmask: 1=Kill, 2=SafeZone, 0=Plain
    uint32_t shape;          // 0=BoundingBox, 1=Sector
    float    posXYZ[3];
    float    fwdXYZ[3];
    float    upXYZ[3];
    float    extentsXYZ[3];
    uint32_t volumeIndex;
    char     name[40];
};
#pragma pack(pop)

static_assert(sizeof(ZH_ScnrTriggerVolume) == 4 + 4 + 12 + 12 + 12 + 12 + 4 + 40,
              "ZH_ScnrTriggerVolume layout drift");

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

// Read an index block (kill / safe) and OR `catBit` into the category of each
// referenced trigger volume. Defensive against garbage descriptors.
void ApplyIndexCategory(CacheHandle* cache, const uint8_t* scnrMeta,
                        int blockOffset, uint32_t catBit,
                        uint32_t* categories, int volumeCount)
{
    if ((size_t)(scnrMeta - cache->base) + blockOffset + 12 > cache->size) return;
    TagBlockRef block = ReadTagBlock(scnrMeta + blockOffset);
    if (block.count <= 0 || block.count > TV_COUNT_SANITY) return;
    int64_t arrOff = TagMetaFileOff(cache, block.pointer);
    if (arrOff < 0) return;
    if ((size_t)arrOff + (size_t)block.count * INDEX_BLOCK_STRIDE > cache->size) return;
    const uint8_t* arr = cache->base + arrOff;
    for (int i = 0; i < block.count; ++i)
    {
        int16_t volIdx = R16(arr + (size_t)i * INDEX_BLOCK_STRIDE + INDEX_VOLUME_OFFSET);
        if (volIdx >= 0 && volIdx < volumeCount)
            categories[volIdx] |= catBit;
    }
}

} // namespace

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_SCNR_EnumerateTriggerVolumes(
    uint64_t cacheHandle,
    uint32_t scnrTagId,
    ZH_ScnrTriggerVolume* outVolumes,
    uint32_t maxCount)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (cache == nullptr || cache->base == nullptr) return 0;
    if (scnrTagId >= cache->tags.size()) return 0;

    const TagEntry& scnrEntry = cache->tags[scnrTagId];
    if (memcmp(scnrEntry.classCode, TC_SCNR, 4) != 0) return 0;

    int64_t scnrMetaOff = TagMetaFileOff(cache, scnrEntry.metaPointerRaw);
    if (scnrMetaOff < 0) return 0;
    if ((size_t)scnrMetaOff + SCNR_TRIGGER_VOLUMES_OFFSET + 12 > cache->size) return 0;
    const uint8_t* scnrMeta = cache->base + scnrMetaOff;

    // Trigger Volumes (the geometry) @ scnr+0x280.
    TagBlockRef tvBlock = ReadTagBlock(scnrMeta + SCNR_TRIGGER_VOLUMES_OFFSET);
    if (tvBlock.count <= 0 || tvBlock.count > TV_COUNT_SANITY) return 0;
    int64_t tvArrOff = TagMetaFileOff(cache, tvBlock.pointer);
    if (tvArrOff < 0) return 0;
    if ((size_t)tvArrOff + (size_t)tvBlock.count * SCNR_TRIGGER_VOLUME_STRIDE > cache->size)
        return 0;
    const uint8_t* tvArr = cache->base + tvArrOff;

    int volumeCount = tvBlock.count;

    // Categorize via the kill / safe index blocks. Heap-alloc the small
    // category table (one u32 per volume), zero-initialised = CAT_PLAIN.
    uint32_t* categories = (uint32_t*)calloc((size_t)volumeCount, sizeof(uint32_t));
    if (categories == nullptr) return 0;

    ApplyIndexCategory(cache, scnrMeta, SCNR_KILL_INDEX_OFFSET, CAT_KILL, categories, volumeCount);
    ApplyIndexCategory(cache, scnrMeta, SCNR_SAFE_INDEX_OFFSET, CAT_SAFE, categories, volumeCount);

    uint32_t written = 0;
    for (int i = 0; i < volumeCount; ++i)
    {
        if (written >= TOTAL_CAP) break;
        const uint8_t* tv = tvArr + (size_t)i * SCNR_TRIGGER_VOLUME_STRIDE;

        if (outVolumes != nullptr && written < maxCount)
        {
            ZH_ScnrTriggerVolume& dst = outVolumes[written];
            memset(&dst, 0, sizeof(dst));
            dst.category    = categories[i];
            dst.shape       = (uint32_t)RU16(tv + TV_TYPE_OFFSET);
            dst.posXYZ[0]   = RF32(tv + TV_POSITION_OFFSET + 0);
            dst.posXYZ[1]   = RF32(tv + TV_POSITION_OFFSET + 4);
            dst.posXYZ[2]   = RF32(tv + TV_POSITION_OFFSET + 8);
            dst.fwdXYZ[0]   = RF32(tv + TV_FORWARD_OFFSET + 0);
            dst.fwdXYZ[1]   = RF32(tv + TV_FORWARD_OFFSET + 4);
            dst.fwdXYZ[2]   = RF32(tv + TV_FORWARD_OFFSET + 8);
            dst.upXYZ[0]    = RF32(tv + TV_UP_OFFSET + 0);
            dst.upXYZ[1]    = RF32(tv + TV_UP_OFFSET + 4);
            dst.upXYZ[2]    = RF32(tv + TV_UP_OFFSET + 8);
            dst.extentsXYZ[0] = RF32(tv + TV_EXTENTS_OFFSET + 0);
            dst.extentsXYZ[1] = RF32(tv + TV_EXTENTS_OFFSET + 4);
            dst.extentsXYZ[2] = RF32(tv + TV_EXTENTS_OFFSET + 8);
            dst.volumeIndex = (uint32_t)i;
            CopyResolvedString(cache, RU32(tv + TV_NAME_OFFSET), dst.name, sizeof(dst.name));
        }
        ++written;
    }

    free(categories);
    return written;
}
