// PoseSnapshot.cpp (HaloMapStudioDLL)
// =============================================================================
// Publishes ZeroHour_Pose_Snapshot MMF - per-frame skeletal pose data for
// up to 16 player bipeds. The viewer reads this to drive live skinning of
// loaded biped meshes (replacing the static A-pose).
//
// Pose-pool layout (see docs/rendering/biped_jmad_re.md):
//   g_180C4D958              pose-pool base pointer (RVA 0xC4D958)
//   ring_header_ptr  =  *(u64*)(TlsGetValue(TLS_idx_at_0xC17B18) + 0x70)
//     ring_header + 0x00  u8   curr_frame_slot   (0 or 1)
//     ring_header + 0x01  u8   prev_frame_slot
//     ring_header + 0x02  u8   ring_active       (0 = poses unavailable)
//     ring_header + 0x04  f32  lerp_t
//   snapshot_base = g_180C4D958 + frame_slot * 0x1A2B834
//   obj_record    = snapshot_base + S * 0x3410        (S resolved via 0xCE778)
//     obj_record + 0x04  u32  object_index (must equal obj_idx)
//     obj_record + 0x08  u32  bone_count
//     obj_record + 0x0C  4xf32 root_xform 0
//     obj_record + 0x18  4xf32 root_xform 1
//     obj_record + 0x24  4xf32 root_xform 2
//     obj_record + 0x48  bone_array[bone_count]  (stride 0x34 per node)
//
// Per-bone (0x34 stride):
//   +0x00  quat (compressed, 16 bytes)
//   +0x10  pos  (3x float, 12 bytes)
//   +0x1C  scale (1x float, 4 bytes)
//   +0x20  padding to 0x34
//
// Runs on engine thread (FramePumpHook). TlsGetValue here returns the engine
// thread's TLS slot - that's the slot the engine uses for the ring header.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"

#include <windows.h>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t kMagic       = 0x45534F50u;  // 'POSE'
constexpr uint32_t kVersion     = 1u;
constexpr uint32_t kMaxObjects  = 16u;
constexpr uint32_t kMaxBones    = 252u;
constexpr uint32_t kBoneStride  = 0x34u;
constexpr uint32_t kBoneArrayBytes = kMaxBones * kBoneStride;  // 0x33C8

constexpr uint32_t kErr_Ok              = 0u;
constexpr uint32_t kErr_NoModule        = 1u;
constexpr uint32_t kErr_NoTlsSlot       = 2u;
constexpr uint32_t kErr_RingInactive    = 3u;
constexpr uint32_t kErr_NoPlayers       = 4u;

// haloreach.dll RVAs from biped_jmad_re.md.
constexpr uintptr_t kRva_PosePoolBase = 0xC4D958u;     // void*
constexpr uintptr_t kRva_PoseTlsIdx   = 0xC17B18u;     // u32 TLS index
constexpr uintptr_t kRva_LookupSlot   = 0xCE778u;      // sub_1800CE778(obj_idx, &out_slot)

constexpr size_t   kFrameStride       = 0x1A2B834u;    // pose snapshot stride
constexpr size_t   kObjRecordStride   = 0x3410u;
constexpr size_t   kObjRec_Idx        = 0x04u;
constexpr size_t   kObjRec_BoneCount  = 0x08u;
constexpr size_t   kObjRec_RootRow0   = 0x0Cu;
constexpr size_t   kObjRec_BoneArray  = 0x48u;

#pragma pack(push, 1)
struct PoseEntry {
    uint32_t Datum;
    uint32_t BoneCount;
    float    RootMatrix[12];     // 3x4 row-major (rows 0..2)
    uint8_t  BoneData[kBoneArrayBytes];
};
static_assert(sizeof(PoseEntry) == 4 + 4 + 48 + (252 * 0x34), "PoseEntry layout");

struct Pose_Shared {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteCounter;
    uint32_t ObjectCount;
    uint32_t SchemaSize;
    uint32_t LastTickMs;
    uint32_t LastError;
    float    FrameLerpT;
    PoseEntry Entries[kMaxObjects];
};
#pragma pack(pop)

const wchar_t kMapName[] = L"ZeroHour_Pose_Snapshot";

// Player-list layout (matches PlayerModeSnapshot.cpp). Inlined here because
// the PlayerModeSnapshot helpers live in its anonymous namespace and aren't
// linkable across TUs.
constexpr size_t    kPlayerHeaderOff = 0x70;
constexpr int       kPlayerStride    = 0x490;
constexpr int       kPlayerDatumOff  = 0x28;

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

HANDLE        g_hMap   = nullptr;
Pose_Shared*  g_Shared = nullptr;

Pose_Shared* EnsureShared()
{
    if (g_Shared) return g_Shared;
    g_hMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                (DWORD)sizeof(Pose_Shared), kMapName);
    if (!g_hMap) return nullptr;
    g_Shared = (Pose_Shared*)MapViewOfFile(g_hMap, FILE_MAP_ALL_ACCESS, 0, 0,
                                           sizeof(Pose_Shared));
    if (!g_Shared) { CloseHandle(g_hMap); g_hMap = nullptr; return nullptr; }

    __try {
        if (g_Shared->Magic != kMagic || g_Shared->Version != kVersion ||
            g_Shared->SchemaSize != (uint32_t)sizeof(PoseEntry))
        {
            ZeroMemory(g_Shared, sizeof(Pose_Shared));
            g_Shared->Magic      = kMagic;
            g_Shared->Version    = kVersion;
            g_Shared->SchemaSize = (uint32_t)sizeof(PoseEntry);
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return g_Shared;
}

// Resolve obj_idx -> pool slot S via the engine's own validator. -1 means
// "no live pose for this object this frame" (e.g. not yet initialized,
// or paused mid-update - viewer falls back to A-pose for this datum).
using LookupSlotFn = uint32_t(__fastcall*)(uint32_t /*obj_idx*/, uint32_t* /*out_slot*/);

bool ReadRingHeader(uint8_t* hr, /*out*/ uint8_t** outRing)
{
    *outRing = nullptr;
    uint32_t tlsIdx = 0;
    __try { tlsIdx = *(uint32_t*)(hr + kRva_PoseTlsIdx); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
    if (tlsIdx == 0u || tlsIdx >= 1088u) return false;

    void* tlsVal = TlsGetValue(tlsIdx);
    if (!tlsVal) return false;

    uint8_t* ringPtr = nullptr;
    __try { ringPtr = *(uint8_t**)((uint8_t*)tlsVal + 0x70u); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
    if (!ringPtr) return false;
    *outRing = ringPtr;
    return true;
}

template <typename T>
static bool SafeReadAt(const uint8_t* p, size_t off, T& out)
{
    __try { out = *(const T*)(p + off); return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

} // namespace

extern "C" void ZH_Logf(const char* fmt, ...);

extern "C" __declspec(dllexport) void PoseSnapshot_FramePumpTick()
{
    using namespace HaloMapStudio::Engine;

    // SKINNED_BIPED diag: periodic heartbeat so the viewer can see if
    // the publisher is actually being driven. Logs once on first call +
    // every 600 frames (~10s at 60Hz) showing the most recent error code
    // and object count.
    static int s_tickCounter = 0;
    static int s_lastLoggedError = -1;
    static uint32_t s_lastLoggedCount = 0xFFFFFFFFu;
    bool firstCall = (s_tickCounter == 0);
    bool periodic  = ((s_tickCounter % 600) == 0);

    Pose_Shared* sh = EnsureShared();
    if (!sh) {
        if (firstCall || periodic) {
            ZH_Logf("[PoseSnapshot] tick=%d EnsureShared FAILED", s_tickCounter);
        }
        s_tickCounter++;
        return;
    }

    HMODULE hrMod = GetModuleHandleW(L"haloreach.dll");
    if (!hrMod) {
        __try {
            sh->ObjectCount  = 0;
            sh->LastError    = kErr_NoModule;
            sh->LastTickMs   = GetTickCount();
            sh->WriteCounter = sh->WriteCounter + 1u;
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
        return;
    }
    uint8_t* hr = (uint8_t*)hrMod;

    // Bypass the ring-header TLS read entirely. The TLS slot at +0xC17B18 is populated only
    // on the engine's animation/sim thread; the frame-pump hook fires on
    // the render thread which sees TLS[idx]=0. Instead of trying to find
    // the "correct" thread (fragile, varies by build), scan BOTH pool
    // frame slots ourselves and pick the one with valid data per object.
    //
    // Per-object validity gate: each obj_record's +0x04 holds the object
    // index - if it matches the obj idx we're scanning, that slot has a
    // valid pose for this frame. If neither slot matches, the object has
    // no live pose (non-animated). This is exactly what sub_1800CE778
    // checks internally; we just inline it without the TLS-based "ring
    // active" gate.
    uint8_t* poolBase = nullptr;
    __try { poolBase = *(uint8_t**)(hr + kRva_PosePoolBase); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
    if (!poolBase) {
        __try {
            sh->ObjectCount  = 0;
            sh->LastError    = kErr_RingInactive;  // reused - "no pool base"
            sh->LastTickMs   = GetTickCount();
            sh->WriteCounter = sh->WriteCounter + 1u;
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
        return;
    }

    // Frame slots - try both. Engine writes new poses to one slot per frame
    // and game logic reads from the other; we don't care which is "current"
    // as long as we find valid data for the objects we publish.
    uint8_t* snapshotBase0 = poolBase + 0 * kFrameStride;
    uint8_t* snapshotBase1 = poolBase + 1 * kFrameStride;
    float    lerpT = 0.0f;  // lerp factor unknown without ring; report 0

    // Step 2 - iterate the OBJECT TABLE (not just the player list). In
    // forge mode the user is usually a Monitor (not animated) while the
    // actual bipeds are forge-spawned objects - those live in the object
    // table, not the player list. Walking the object descriptor lets us
    // pick up poses for ANY animated object: players, AI, forge-spawned
    // bipeds, vehicles, etc.
    uint8_t* desc = ResolveObjectPoolDescriptor();
    if (!desc) {
        __try {
            sh->ObjectCount  = 0;
            sh->LastError    = kErr_NoPlayers;  // reused - "no source to walk"
            sh->LastTickMs   = GetTickCount();
            sh->FrameLerpT   = lerpT;
            sh->WriteCounter = sh->WriteCounter + 1u;
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
        return;
    }

    uint32_t entrySize = 0, maxCount = 0;
    uint8_t* entries = nullptr;
    if (!SafeReadAt(desc, kDesc_EntrySize, entrySize) ||
        !SafeReadAt(desc, kDesc_MaxCount,  maxCount)  ||
        !SafeReadAt(desc, kDesc_Entries,   entries)   ||
        entrySize == 0 || entrySize > 0x100 ||
        maxCount  == 0 || maxCount  > 0x4000 ||
        !entries)
    {
        __try {
            sh->ObjectCount  = 0;
            sh->LastError    = kErr_NoPlayers;
            sh->LastTickMs   = GetTickCount();
            sh->FrameLerpT   = lerpT;
            sh->WriteCounter = sh->WriteCounter + 1u;
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
        return;
    }

    // SKINNED_BIPED_V3 DIAGNOSTIC: full pool scan. The previous V2 assumed
    // pool_slot == obj_idx (1:1 mapping) which produces zero matches in
    // practice - the engine uses a sparse remapping (sub_1804739CC). To
    // discover the real layout we scan ALL pool slots in both frames and
    // log which ones have non-zero / non-(-1) data. Once the layout is
    // understood from these diag lines we can resolve obj_idx -> slot
    // correctly. For now, BUILD AN INDEX from this scan and use it.
    constexpr uint32_t kMaxPoolSlots = 2080;  // 0x1A2B834 / 0x3410 = 2073.0
    uint32_t poolPopulated = 0;
    uint32_t firstFewIdx[8] = {0};
    uint32_t firstFewBones[8] = {0};
    int firstFewCnt = 0;
    // idxToSlot maps obj_idx_low_16 -> (slotIdx | frame<<16). 0xFFFFFFFF =
    // not found. Indexed by obj_idx_low_16; max 65536 entries (256 KB).
    // Stack-allocated to avoid heap churn per tick. Only initialized for
    // entries we touch.
    static thread_local uint32_t idxToRecOffset[0x10000];
    static thread_local int idxToRecValid = 0;
    // Reset to 0xFFFFFFFF lazily - only the bins we wrote last frame need
    // clearing.
    static thread_local uint16_t prevWrittenIdx[kMaxPoolSlots * 2];
    static thread_local int prevWrittenCount = 0;
    if (!idxToRecValid) {
        for (int q = 0; q < 0x10000; q++) idxToRecOffset[q] = 0xFFFFFFFFu;
        idxToRecValid = 1;
    } else {
        for (int q = 0; q < prevWrittenCount; q++)
            idxToRecOffset[prevWrittenIdx[q]] = 0xFFFFFFFFu;
        prevWrittenCount = 0;
    }
    int writtenIdxCount = 0;
    __try {
        // Scan both frames, build the obj_idx -> record-offset map.
        for (int frame = 0; frame < 2; frame++)
        {
            uint8_t* base = (frame == 0) ? snapshotBase0 : snapshotBase1;
            for (uint32_t s = 0; s < kMaxPoolSlots; s++)
            {
                uint8_t* rec = base + (size_t)s * kObjRecordStride;
                uint32_t recIdx = 0xFFFFFFFFu;
                uint32_t boneCount = 0;
                if (!SafeReadAt(rec, kObjRec_Idx, recIdx)) continue;
                if (recIdx == 0xFFFFFFFFu) continue;
                if (!SafeReadAt(rec, kObjRec_BoneCount, boneCount)) continue;
                if (boneCount == 0 || boneCount > kMaxBones) continue;
                poolPopulated++;
                if (firstFewCnt < 8) {
                    firstFewIdx[firstFewCnt] = recIdx;
                    firstFewBones[firstFewCnt] = boneCount;
                    firstFewCnt++;
                }
                uint32_t idxLo = recIdx & 0xFFFFu;
                if (writtenIdxCount < (int)(kMaxPoolSlots * 2) &&
                    idxToRecOffset[idxLo] == 0xFFFFFFFFu)
                {
                    // Pack: low 24 bits = byte offset into poolBase; high 8 = frame.
                    size_t offBytes = (size_t)((uint8_t*)rec - poolBase);
                    if (offBytes < 0x80000000u)
                    {
                        idxToRecOffset[idxLo] = (uint32_t)offBytes;
                        prevWrittenIdx[writtenIdxCount++] = (uint16_t)idxLo;
                    }
                }
            }
        }
        prevWrittenCount = writtenIdxCount;
    } __except (EXCEPTION_EXECUTE_HANDLER) {}

    uint32_t outCount = 0;
    uint32_t matchedViaMap = 0;
    __try {
        // Walk obj-table; lookup each obj's pose via the built map.
        for (uint32_t i = 0; i < maxCount && outCount < kMaxObjects; ++i) {
            uint8_t* ent = entries + (size_t)i * entrySize;
            uint16_t salt = 0;
            uint8_t* obj  = nullptr;
            if (!SafeReadAt(ent, kEntry_Salt,   salt) || salt == 0) continue;
            if (!SafeReadAt(ent, kEntry_ObjPtr, obj)  || !obj)      continue;

            uint32_t datum    = ((uint32_t)salt << 16) | (i & 0xFFFFu);
            uint16_t objIdxLo = (uint16_t)(i & 0xFFFFu);

            uint32_t recOff = idxToRecOffset[objIdxLo];
            if (recOff == 0xFFFFFFFFu) continue;  // no pose for this object

            uint8_t* objRec = poolBase + recOff;
            uint32_t boneCount = 0;
            if (!SafeReadAt(objRec, kObjRec_BoneCount, boneCount)) continue;
            if (boneCount == 0 || boneCount > kMaxBones) continue;
            matchedViaMap++;

            PoseEntry& dst = sh->Entries[outCount++];
            dst.Datum     = datum;
            dst.BoneCount = boneCount;
            __try {
                memcpy(dst.RootMatrix, objRec + kObjRec_RootRow0, sizeof(dst.RootMatrix));
                memcpy(dst.BoneData,   objRec + kObjRec_BoneArray, (size_t)boneCount * kBoneStride);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                --outCount;
                matchedViaMap--;
            }
        }
        if (firstCall || periodic) {
            ZH_Logf("[PoseSnapshot] tick=%d poolBase=%p poolPopulated=%u matched=%u "
                    "first=[idx=%u bones=%u, idx=%u bones=%u, idx=%u bones=%u]",
                    s_tickCounter, (void*)poolBase, poolPopulated, matchedViaMap,
                    firstFewCnt > 0 ? firstFewIdx[0] : 0u,
                    firstFewCnt > 0 ? firstFewBones[0] : 0u,
                    firstFewCnt > 1 ? firstFewIdx[1] : 0u,
                    firstFewCnt > 1 ? firstFewBones[1] : 0u,
                    firstFewCnt > 2 ? firstFewIdx[2] : 0u,
                    firstFewCnt > 2 ? firstFewBones[2] : 0u);
        }
        sh->ObjectCount  = outCount;
        sh->FrameLerpT   = lerpT;
        sh->LastError    = kErr_Ok;
        sh->LastTickMs   = GetTickCount();
        sh->WriteCounter = sh->WriteCounter + 1u;

        // Heartbeat / state-change log.
        if (firstCall || periodic ||
            (int)sh->LastError != s_lastLoggedError ||
            sh->ObjectCount != s_lastLoggedCount)
        {
            ZH_Logf("[PoseSnapshot] tick=%d direct-scan lerp=%.3f objs=%u err=%u",
                    s_tickCounter, lerpT,
                    sh->ObjectCount, sh->LastError);
            s_lastLoggedError = (int)sh->LastError;
            s_lastLoggedCount = sh->ObjectCount;
        }
        s_tickCounter++;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        // SEH at MMF tail - drop the mapping and rebuild on the next tick.
        UnmapViewOfFile(sh);
        CloseHandle(g_hMap);
        g_hMap   = nullptr;
        g_Shared = nullptr;
    }
}
