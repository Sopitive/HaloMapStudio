// ForgeGtLabelsMmf.cpp
// =============================================================================
// Publishes the haloreach.dll gametype-label (gtLabel) string table to a
// memory-mapped file (`HaloMapStudio_ForgeGtLabels`) for the viewer to read.
//
// LAYOUT IN THE GAME
// ------------------
// Mjolnir-Forge-Editor's ForgeBridge.cs (the authoritative reference) reads
// gtLabels as:
//
//     memory.ReadString(gtLabelsPointer, 4096, false).Split('\0')
//         // stop at the first empty string
//
// i.e. the table is a packed sequence of NUL-terminated ASCII C strings,
// terminated by an extra NUL (empty string). Strings are variable-length - 
// NOT a fixed-stride array. The total byte budget is ~4096 B; in Reach in
// practice there are well under 64 distinct labels.
//
// MMF FORMAT (this file's output)
// -------------------------------
//     u32 Magic    = 'LBLF' (0x46_4C_42_4C LE - i.e. "LBLF" ASCII)
//     u32 Version  = 1
//     u32 Count    (0..kMaxLabels)
//     u32 _Pad
//     [kMaxLabels x char[32] Name] - ASCII, NUL-padded to 32 B.
//
// 32 B per slot is a comfortable upper bound on label length (Reach labels
// are short: "SCALE", "INV_SCALE", "INFECTED", etc.). Any string that
// happens to exceed 31 chars is truncated; we log a warning when we see it.
//
// LIFETIME
// --------
// Caller (ForgeObjectTableSnapshot_FramePumpTick) invokes
// ForgeGtLabelsMmf_FramePumpTick() once it has g_GtLabelsBase resolved. We
// only re-publish (write to the MMF view) when the snapshot's bytes differ
// from the last published snapshot - avoids burning memory bandwidth every
// tick for a table that effectively never changes mid-map.
// =============================================================================

#include "pch.h"
#include <windows.h>
#include <cstdint>
#include <cstring>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

constexpr uint32_t kMagic      = 0x464C424Cu; // 'LBLF' (little-endian when read as ASCII bytes 'L','B','L','F')
constexpr uint32_t kVersion    = 1u;
constexpr uint32_t kMaxLabels  = 64u;
constexpr size_t   kNameStride = 32u;
constexpr size_t   kScanBudget = 4096u;       // Mjolnir's read budget.

#pragma pack(push, 1)
struct Header {
    uint32_t Magic;
    uint32_t Version;
    uint32_t Count;
    uint32_t _Pad;
};
struct Slot {
    char Name[kNameStride];
};
struct Snapshot {
    Header H;
    Slot   Slots[kMaxLabels];
};
#pragma pack(pop)

const wchar_t kMapName[] = L"HaloMapStudio_ForgeGtLabels";

HANDLE     g_FileMapping = nullptr;
Snapshot*  g_SharedView  = nullptr;

// Last-published image - used to skip the write when nothing changed.
Snapshot   g_LastPublished{};
bool       g_HavePublished = false;
bool       g_LoggedTruncation = false;

bool EnsureMmf()
{
    if (g_SharedView) return true;
    size_t totalSize = sizeof(Snapshot);
    g_FileMapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        (DWORD)((uint64_t)totalSize >> 32), (DWORD)(totalSize & 0xFFFFFFFFu),
        kMapName);
    if (!g_FileMapping) return false;
    g_SharedView = (Snapshot*)MapViewOfFile(
        g_FileMapping, FILE_MAP_ALL_ACCESS, 0, 0, totalSize);
    if (!g_SharedView) {
        CloseHandle(g_FileMapping);
        g_FileMapping = nullptr;
        return false;
    }
    g_SharedView->H.Magic   = kMagic;
    g_SharedView->H.Version = kVersion;
    g_SharedView->H.Count   = 0;
    return true;
}

// SEH-safe scan of variable-length NUL-terminated C strings starting at
// `base`. Stops at the first empty string OR when we've scanned kScanBudget
// bytes OR when we've collected kMaxLabels labels. Each label is copied
// into `out` (NUL-padded to kNameStride). Returns the number of labels.
uint32_t ScanLabels(const uint8_t* base, Snapshot& out)
{
    uint32_t n = 0;
    size_t   i = 0;
    bool     truncatedAny = false;
    __try {
        while (n < kMaxLabels && i < kScanBudget) {
            // Find end of current C string.
            size_t start = i;
            while (i < kScanBudget && base[i] != 0) ++i;
            size_t len = i - start;
            if (len == 0) break;  // empty string = terminator (Mjolnir's rule)

            size_t copyLen = len;
            if (copyLen >= kNameStride) {
                copyLen = kNameStride - 1;
                truncatedAny = true;
            }
            memset(out.Slots[n].Name, 0, kNameStride);
            memcpy(out.Slots[n].Name, base + start, copyLen);
            // Belt-and-braces: scrub any non-printable bytes that snuck in
            // (defensive - gtLabel strings are pure ASCII letters / digits /
            // underscores in every observed Reach build).
            for (size_t k = 0; k < copyLen; ++k) {
                unsigned char c = (unsigned char)out.Slots[n].Name[k];
                if (c < 0x20 || c >= 0x7F) out.Slots[n].Name[k] = '?';
            }
            ++n;
            ++i;  // skip the NUL we just stopped on
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        // Partial result is fine - caller treats it as advisory.
    }
    if (truncatedAny && !g_LoggedTruncation) {
        ZH_Logf("[HaloMapStudioDLL] ForgeGtLabelsMmf: at least one label exceeded %zu B (truncated)\n",
                kNameStride - 1);
        g_LoggedTruncation = true;
    }
    return n;
}

} // namespace

// =============================================================================
// Public entry point - called once per frame from
// ForgeObjectTableSnapshot_FramePumpTick.
// =============================================================================
extern "C" void ForgeGtLabelsMmf_FramePumpTick(const uint8_t* gtLabelsBase)
{
    if (!EnsureMmf()) return;

    Snapshot local{};
    local.H.Magic   = kMagic;
    local.H.Version = kVersion;
    local.H._Pad    = 0;

    if (gtLabelsBase) {
        local.H.Count = ScanLabels(gtLabelsBase, local);
    } else {
        local.H.Count = 0;
    }

    // Skip the publish if nothing changed. (memcmp on the WHOLE snapshot
    // - header + all slots - is fast and avoids ratcheting the dirty page
    // every frame for what is essentially a static table.)
    if (g_HavePublished && memcmp(&local, &g_LastPublished, sizeof(Snapshot)) == 0) {
        return;
    }

    // Atomic-enough: header first (so any reader that races the count
    // field sees 0 / old value briefly, but never a count > populated
    // slot range), then slots, then count.
    g_SharedView->H.Count = 0;
    memcpy(g_SharedView->Slots, local.Slots, sizeof(local.Slots));
    g_SharedView->H.Count = local.H.Count;

    memcpy(&g_LastPublished, &local, sizeof(Snapshot));
    g_HavePublished = true;

    ZH_Logf("[HaloMapStudioDLL] ForgeGtLabelsMmf: published %u label(s)\n",
            local.H.Count);
}
