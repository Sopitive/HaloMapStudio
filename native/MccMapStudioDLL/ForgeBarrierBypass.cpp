// ForgeBarrierBypass.cpp
// =============================================================================
// Kill volumes / soft ceilings / safe-zone enforcement bypass.
//
// Zeroing the scnr tagblock COUNT fields does NOT work in-game: kill
// volumes still kill the player even with the count read-back showing 0.
// Root cause:
//
//   The engine does NOT re-read the scnr kill_triggers / safe_zones tagblock
//   *count* on the per-tick "is the player inside a kill volume?" check. At
//   map load the engine COMPILES the trigger volumes into a runtime list /
//   Havok broadphase and evaluates point-in-OBB against that compiled copy.
//   Zeroing the scnr tag count after load is a no-op for enforcement because
//   nothing re-iterates that count - the runtime already cached its own copy
//   (and the iteration bound). This was confirmed structurally:
//     * The scnr field offsets ARE correct (verified vs Lord Zedd's ReachMCC
//       scnr.xml plugin AND the fact that ForgePaletteSnapshot reads the LIVE
//       resident meta at the SAME offsets - scnr+0x228 - successfully).
//     * `offset-xref haloreach 0x278..0x288` and `0x4C0..0x4E8` find ZERO
//       literal-displacement reads of the trigger-volume / kill-trigger
//       offsets, i.e. enforcement does NOT touch scnr+0x280 / scnr+0x4C8 by
//       displacement at all - it walks a runtime list whose base+stride was
//       resolved once at load.
//
// THE V4 FIX - neutralize the GEOMETRY, not the count:
//
//   The trigger-volume OBBs (Position + Extents) live in the shared Trigger
//   Volumes block at scnr+0x280 (stride 0x7C), referenced by BOTH the kill
//   (scnr+0x4C8) and safe-zone (scnr+0x4D4) index blocks. Critically, this
//   geometry lives in the SAME resident tag meta the renderer walks live
//   (ScenarioTriggerVolumeWalker confirms the layout; ForgePaletteSnapshot
//   confirms the live meta is readable via contracted-pointer expansion).
//
//   Even if the per-tick check runs against a compiled copy, that compiled
//   copy is *derived from* this geometry at load. By collapsing each KILL
//   trigger volume's Extents to ~0 AND shoving its Position far below the
//   playable space, the point-in-OBB test can never report the player inside
//   a kill volume: a zero-extent box at (0,0,-1e9) contains nothing. We do
//   this on the LIVE meta every tick (so a re-compile picks it up) and zero
//   the counts too (belt-and-suspenders for any path that DOES read the
//   count). SAFE-zone volumes are LEFT INTACT - collapsing safe zones would
//   be counter-productive (a player standing in a now-tiny safe zone would
//   fall OUT of it and get killed by the "outside all safe zones => die"
//   rule). We only neutralize KILL triggers + soft ceilings + MOPP.
//
//   Soft ceilings (scnr+0x25C) push the player down rather than kill, but the
//   user groups them under "barriers"; we zero that count as before (cheap).
//
// Called every frame from FramePumpHook's RunSnapshotTicks via
// HaloMapStudio_ForgeBarrier_Tick(). Cheap when disabled (one atomic load).
// =============================================================================

#include "pch.h"
#include "HaloReachTagHelpers.h"

#include <windows.h>
#include <cstdint>
#include <atomic>
#include <cstring>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

using ZeroHour::HrTag::ResolveTagAddress;
using ZeroHour::HrTag::ReadTagblock;
using ZeroHour::HrTag::ExpandContracted;
using ZeroHour::HrTag::Tagblock;

// User-requested state: 1 = barriers disabled, 0 = normal engine behavior.
std::atomic<int> g_Disabled{ 0 };

// scnr tag-meta offsets for the enforcement tagblocks. Verified against
// Assembly/Plugins/ReachMCC/scnr.xml (same plugin layout as the forge palette
// +0x228 anchor MMS already reads from LIVE meta). Each offset points to a
// 12-byte tagblock descriptor: { u32 count; u32 contracted-ptr; u32 unk }.
constexpr size_t kScnr_SoftCeilings   = 0x25C;
constexpr size_t kScnr_TriggerVolumes = 0x280;  // geometry (OBBs), stride 0x7C
constexpr size_t kScnr_KillTriggers   = 0x4C8;  // index block -> trigger volume
constexpr size_t kScnr_SafeZones      = 0x4D4;  // index block -> trigger volume
constexpr size_t kScnr_MoppTriggers   = 0x4E0;

// Trigger-volume element layout (stride 0x7C). Position = box center,
// Extents = half-size along the Forward/Up/Right basis. Both world-space.
constexpr size_t kTV_Stride   = 0x7C;
constexpr size_t kTV_Position = 0x28;  // f32x3
constexpr size_t kTV_Extents  = 0x34;  // f32x3
// Index-block element layout (kill/safe): int16 trigger-volume index @ +0.
constexpr size_t kIdx_Stride  = 0x04;

constexpr int kSanityMaxVolumes = 4096;

// Where we shove a neutralized kill volume so it can't contain the player.
constexpr float kFarBelow = -1.0e9f;

// scnr active-scenario datum global (haloreach.dll+0xAFBE38). Same location
// ForgePaletteSnapshot/ForgePalette read; cross-checked there.
constexpr uintptr_t kRva_ScenarioDatum = 0xAFBE38;

// ---------------------------------------------------------------------------
// Backup state so the user can toggle the bypass off and restore the authored
// counts AND geometry (without a map reload).
// ---------------------------------------------------------------------------
struct VolBackup { uint32_t index; float pos[3]; float ext[3]; };

struct ScnrBackup
{
    uint32_t Datum;
    uint64_t ScnrVA;
    int32_t  SoftCeilingsCount;
    int32_t  KillTriggersCount;
    int32_t  SafeZonesCount;
    int32_t  MoppTriggersCount;
    bool     Populated;

    // Geometry backup for the kill-referenced trigger volumes we collapse.
    VolBackup Vols[256];
    uint32_t  VolCount;
};
ScnrBackup g_Backup{};

// One-shot verify-log gate. Reset on scnr change.
bool g_LoggedVerify = false;

// ---------------------------------------------------------------------------
// SEH-guarded primitive reads/writes (geometry lives in resident tag meta,
// which can transiently fault mid-map-swap).
// ---------------------------------------------------------------------------
bool SafeWriteI32(uint64_t va, int32_t value)
{
    DWORD oldProt = 0;
    BOOL vpOk = VirtualProtect((LPVOID)va, sizeof(int32_t), PAGE_READWRITE, &oldProt);
    bool ok = false;
    __try { *reinterpret_cast<int32_t*>(va) = value; ok = true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { ok = false; }
    if (vpOk) { DWORD ignored = 0; VirtualProtect((LPVOID)va, sizeof(int32_t), oldProt, &ignored); }
    return ok;
}

bool SafeWriteF32(uint64_t va, float value)
{
    DWORD oldProt = 0;
    BOOL vpOk = VirtualProtect((LPVOID)va, sizeof(float), PAGE_READWRITE, &oldProt);
    bool ok = false;
    __try { *reinterpret_cast<float*>(va) = value; ok = true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { ok = false; }
    if (vpOk) { DWORD ignored = 0; VirtualProtect((LPVOID)va, sizeof(float), oldProt, &ignored); }
    return ok;
}

bool SafeReadI32(uint64_t va, int32_t& out)
{
    __try { out = *reinterpret_cast<const int32_t*>(va); return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SafeReadF32(uint64_t va, float& out)
{
    __try { out = *reinterpret_cast<const float*>(va); return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SafeReadI16(uint64_t va, int16_t& out)
{
    __try { out = *reinterpret_cast<const int16_t*>(va); return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

// ---------------------------------------------------------------------------
// Resolve the live scnr meta address. Re-runs every call (map change moves it).
// ---------------------------------------------------------------------------
uint32_t ReadScenarioDatumInline()
{
    uint8_t* base = (uint8_t*)GetModuleHandleW(L"haloreach.dll");
    if (!base) return 0;
    uint32_t v = 0;
    __try { v = *(uint32_t*)(base + kRva_ScenarioDatum); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
    return v;
}

uint64_t ResolveLiveScnrVA(uint32_t& outDatum)
{
    outDatum = ReadScenarioDatumInline();
    if (outDatum == 0u || outDatum == 0xFFFFFFFFu) return 0;
    return ResolveTagAddress(outDatum);
}

// Resolve the live Trigger Volumes geometry array (the OBBs). Returns the
// base VA of element 0 and the count, or {0,0} on any failure. Uses the same
// contracted-pointer expansion the renderer-side walker / ForgePaletteSnapshot
// rely on (the resident tag meta stores tagblock pointers contracted).
uint64_t ResolveTriggerVolumeArray(uint64_t scnrVA, uint32_t& outCount)
{
    outCount = 0;
    Tagblock tvBlock{};
    if (!ReadTagblock(scnrVA + kScnr_TriggerVolumes, tvBlock)) return 0;
    if (tvBlock.count == 0 || tvBlock.count > (uint32_t)kSanityMaxVolumes) return 0;
    uint64_t arr = ExpandContracted(tvBlock.ptr);
    if (!arr) return 0;
    outCount = tvBlock.count;
    return arr;
}

// Resolve a kill/safe INDEX block (int16 indices into the trigger-volume
// array). Returns the base VA + count, or {0,0}.
uint64_t ResolveIndexBlock(uint64_t scnrVA, size_t blockOffset, uint32_t& outCount)
{
    outCount = 0;
    Tagblock blk{};
    if (!ReadTagblock(scnrVA + blockOffset, blk)) return 0;
    if (blk.count == 0 || blk.count > (uint32_t)kSanityMaxVolumes) return 0;
    uint64_t arr = ExpandContracted(blk.ptr);
    if (!arr) return 0;
    outCount = blk.count;
    return arr;
}

// ---------------------------------------------------------------------------
// Capture backup once per scnr load: the four counts + the original geometry
// of every KILL-referenced trigger volume (so we can restore on toggle-off).
// ---------------------------------------------------------------------------
void CaptureBackupIfNeeded(uint64_t scnrVA, uint32_t scnrDatum)
{
    if (g_Backup.Populated && g_Backup.Datum == scnrDatum) return;
    g_LoggedVerify = false;

    int32_t sc = 0, kt = 0, sz = 0, mp = 0;
    bool ok = SafeReadI32(scnrVA + kScnr_SoftCeilings, sc)
           && SafeReadI32(scnrVA + kScnr_KillTriggers, kt)
           && SafeReadI32(scnrVA + kScnr_SafeZones,    sz)
           && SafeReadI32(scnrVA + kScnr_MoppTriggers, mp);
    if (!ok) return;

    g_Backup = ScnrBackup{};
    g_Backup.Datum             = scnrDatum;
    g_Backup.ScnrVA            = scnrVA;
    g_Backup.SoftCeilingsCount = sc;
    g_Backup.KillTriggersCount = kt;
    g_Backup.SafeZonesCount    = sz;
    g_Backup.MoppTriggersCount = mp;

    // Capture geometry for the KILL-referenced trigger volumes.
    uint32_t tvCount = 0;
    uint64_t tvArr = ResolveTriggerVolumeArray(scnrVA, tvCount);
    uint32_t killIdxCount = 0;
    uint64_t killIdx = ResolveIndexBlock(scnrVA, kScnr_KillTriggers, killIdxCount);

    if (tvArr && killIdx)
    {
        for (uint32_t i = 0; i < killIdxCount && g_Backup.VolCount < 256; ++i)
        {
            int16_t volIdx = -1;
            if (!SafeReadI16(killIdx + (uint64_t)i * kIdx_Stride, volIdx)) continue;
            if (volIdx < 0 || (uint32_t)volIdx >= tvCount) continue;

            uint64_t tv = tvArr + (uint64_t)volIdx * kTV_Stride;
            VolBackup vb{};
            vb.index = (uint32_t)volIdx;
            bool g = SafeReadF32(tv + kTV_Position + 0, vb.pos[0])
                  && SafeReadF32(tv + kTV_Position + 4, vb.pos[1])
                  && SafeReadF32(tv + kTV_Position + 8, vb.pos[2])
                  && SafeReadF32(tv + kTV_Extents  + 0, vb.ext[0])
                  && SafeReadF32(tv + kTV_Extents  + 4, vb.ext[1])
                  && SafeReadF32(tv + kTV_Extents  + 8, vb.ext[2]);
            if (!g) continue;
            g_Backup.Vols[g_Backup.VolCount++] = vb;
        }
    }

    g_Backup.Populated = true;
    ZH_Logf("[ForgeBarrierBypass] backup captured scnr=0x%08X VA=0x%llX "
            "soft_ceilings=%d kill_triggers=%d safe_zones=%d mopp_triggers=%d "
            "trigger_volumes=%u kill_obbs_backed=%u\n",
            scnrDatum, (unsigned long long)scnrVA, sc, kt, sz, mp,
            tvCount, g_Backup.VolCount);
}

// ---------------------------------------------------------------------------
// Collapse every KILL-referenced trigger volume: zero its Extents and shove
// its Position far below the world. A zero-extent OBB at (0,0,-1e9) cannot
// contain the player, so the point-in-OBB kill test is defeated regardless of
// whether enforcement reads the scnr geometry directly or a load-time copy
// derived from it. Returns the number of OBBs neutralized this pass.
uint32_t CollapseKillVolumes(uint64_t scnrVA)
{
    uint32_t tvCount = 0;
    uint64_t tvArr = ResolveTriggerVolumeArray(scnrVA, tvCount);
    if (!tvArr) return 0;

    uint32_t killIdxCount = 0;
    uint64_t killIdx = ResolveIndexBlock(scnrVA, kScnr_KillTriggers, killIdxCount);
    if (!killIdx) return 0;

    uint32_t neutralized = 0;
    for (uint32_t i = 0; i < killIdxCount; ++i)
    {
        int16_t volIdx = -1;
        if (!SafeReadI16(killIdx + (uint64_t)i * kIdx_Stride, volIdx)) continue;
        if (volIdx < 0 || (uint32_t)volIdx >= tvCount) continue;

        uint64_t tv = tvArr + (uint64_t)volIdx * kTV_Stride;
        bool ok = SafeWriteF32(tv + kTV_Extents  + 0, 0.0f)
               && SafeWriteF32(tv + kTV_Extents  + 4, 0.0f)
               && SafeWriteF32(tv + kTV_Extents  + 8, 0.0f)
               && SafeWriteF32(tv + kTV_Position + 0, 0.0f)
               && SafeWriteF32(tv + kTV_Position + 4, 0.0f)
               && SafeWriteF32(tv + kTV_Position + 8, kFarBelow);
        if (ok) ++neutralized;
    }
    return neutralized;
}

// Restore the geometry of every collapsed kill volume from backup.
void RestoreKillVolumeGeometry()
{
    if (!g_Backup.Populated || g_Backup.ScnrVA == 0) return;
    uint32_t tvCount = 0;
    uint64_t tvArr = ResolveTriggerVolumeArray(g_Backup.ScnrVA, tvCount);
    if (!tvArr) return;

    uint32_t restored = 0;
    for (uint32_t i = 0; i < g_Backup.VolCount; ++i)
    {
        const VolBackup& vb = g_Backup.Vols[i];
        if (vb.index >= tvCount) continue;
        uint64_t tv = tvArr + (uint64_t)vb.index * kTV_Stride;
        bool ok = SafeWriteF32(tv + kTV_Position + 0, vb.pos[0])
               && SafeWriteF32(tv + kTV_Position + 4, vb.pos[1])
               && SafeWriteF32(tv + kTV_Position + 8, vb.pos[2])
               && SafeWriteF32(tv + kTV_Extents  + 0, vb.ext[0])
               && SafeWriteF32(tv + kTV_Extents  + 4, vb.ext[1])
               && SafeWriteF32(tv + kTV_Extents  + 8, vb.ext[2]);
        if (ok) ++restored;
    }
    ZH_Logf("[ForgeBarrierBypass] restored %u/%u kill-volume OBBs from backup\n",
            restored, g_Backup.VolCount);
}

// ---------------------------------------------------------------------------
// Stamp the disabled state: collapse kill geometry (the real fix) + zero the
// counts (belt-and-suspenders). Logs a verify line once per scnr load so the
// user can confirm from the native log exactly what happened.
// ---------------------------------------------------------------------------
void StampDisabled(uint64_t scnrVA)
{
    uint32_t neutralized = CollapseKillVolumes(scnrVA);

    SafeWriteI32(scnrVA + kScnr_SoftCeilings, 0);
    SafeWriteI32(scnrVA + kScnr_KillTriggers, 0);
    SafeWriteI32(scnrVA + kScnr_SafeZones,    0);
    SafeWriteI32(scnrVA + kScnr_MoppTriggers, 0);

    if (!g_LoggedVerify)
    {
        int32_t sc = -1, kt = -1, sz = -1, mp = -1;
        SafeReadI32(scnrVA + kScnr_SoftCeilings, sc);
        SafeReadI32(scnrVA + kScnr_KillTriggers, kt);
        SafeReadI32(scnrVA + kScnr_SafeZones,    sz);
        SafeReadI32(scnrVA + kScnr_MoppTriggers, mp);

        // Read back the first collapsed OBB (if any) to prove the geometry
        // write landed in resident memory.
        float ex = -1.0f, pz = 0.0f;
        if (g_Backup.VolCount > 0)
        {
            uint32_t tvCount = 0;
            uint64_t tvArr = ResolveTriggerVolumeArray(scnrVA, tvCount);
            if (tvArr && g_Backup.Vols[0].index < tvCount)
            {
                uint64_t tv = tvArr + (uint64_t)g_Backup.Vols[0].index * kTV_Stride;
                SafeReadF32(tv + kTV_Extents + 0, ex);
                SafeReadF32(tv + kTV_Position + 8, pz);
            }
        }

        ZH_Logf("[ForgeBarrierBypass] stamp verify scnrVA=0x%llX "
                "counts(soft=%d kill=%d safe=%d mopp=%d, all should be 0) "
                "kill_obbs_collapsed=%u firstOBB.extX=%.2f firstOBB.posZ=%.1f "
                "(extX should be 0, posZ should be ~-1e9)\n",
                (unsigned long long)scnrVA, sc, kt, sz, mp,
                neutralized, ex, pz);
        g_LoggedVerify = true;
    }
}

void RestoreFromBackup()
{
    if (!g_Backup.Populated) return;
    // Geometry first (the primary lever), then the counts.
    RestoreKillVolumeGeometry();
    SafeWriteI32(g_Backup.ScnrVA + kScnr_SoftCeilings, g_Backup.SoftCeilingsCount);
    SafeWriteI32(g_Backup.ScnrVA + kScnr_KillTriggers, g_Backup.KillTriggersCount);
    SafeWriteI32(g_Backup.ScnrVA + kScnr_SafeZones,    g_Backup.SafeZonesCount);
    SafeWriteI32(g_Backup.ScnrVA + kScnr_MoppTriggers, g_Backup.MoppTriggersCount);
    ZH_Logf("[ForgeBarrierBypass] restored authored counts soft_ceilings=%d "
            "kill_triggers=%d safe_zones=%d mopp_triggers=%d\n",
            g_Backup.SoftCeilingsCount, g_Backup.KillTriggersCount,
            g_Backup.SafeZonesCount,    g_Backup.MoppTriggersCount);
}

} // namespace

// ---- Exports (reached via GetProcAddress by an in-process host) ----

extern "C" void HaloMapStudio_ForgeBarrier_SetEnabled(int disable)
{
    int prev = g_Disabled.exchange(disable ? 1 : 0, std::memory_order_acq_rel);
    if (disable && !prev) {
        ZH_Logf("[ForgeBarrierBypass] kill volumes / soft ceilings / safe zones "
                "DISABLED by user (collapse kill-volume geometry + zero counts)\n");
        uint32_t datum = 0;
        uint64_t va = ResolveLiveScnrVA(datum);
        if (va) {
            CaptureBackupIfNeeded(va, datum);
            StampDisabled(va);
        } else {
            ZH_Logf("[ForgeBarrierBypass] no live scnr yet at toggle-on; the "
                    "frame-pump tick will stamp once the scenario loads\n");
        }
    }
    else if (!disable && prev) {
        ZH_Logf("[ForgeBarrierBypass] kill volumes / soft ceilings / safe zones "
                "RE-ENABLED by user\n");
        RestoreFromBackup();
    }
}

extern "C" int HaloMapStudio_ForgeBarrier_IsDisabled()
{
    return g_Disabled.load(std::memory_order_acquire);
}

// Called from FramePumpHook's RunSnapshotTicks every engine frame.
extern "C" void HaloMapStudio_ForgeBarrier_Tick()
{
    if (!g_Disabled.load(std::memory_order_acquire)) return;

    uint32_t datum = 0;
    uint64_t va = ResolveLiveScnrVA(datum);
    if (!va) return;

    CaptureBackupIfNeeded(va, datum);
    StampDisabled(va);
}

// No-op installer (the per-tick stamp + the geometry collapse cover everything
// via the existing FramePumpHook; no MinHook detour required).
extern "C" bool ForgeBarrierBypass_Install()
{
    ZH_Logf("[ForgeBarrierBypass] V4: collapse-kill-volume-geometry method "
            "(no detour required)\n");
    return true;
}
