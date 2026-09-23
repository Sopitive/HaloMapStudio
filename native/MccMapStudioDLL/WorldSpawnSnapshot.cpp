// WorldSpawnSnapshot.cpp
// =============================================================================
// V2 ring-buffer spawn dispatcher. Reads multiple spawn entries per tick
// from ZeroHour_WorldSpawn_Shared (V2 layout: 16-byte header + 32x24-byte
// ring entries = 784 bytes).
//
// The host writes entries at ring[writeIdx % 32] and increments writeIdx.
// DLL drains all pending entries (readIdx to writeIdx-1) each tick.
// =============================================================================

#include "pch.h"
#include <windows.h>
#include <cstdint>
#include <algorithm>

extern "C" uint32_t HaloMapStudio_DispatchForgeSpawn(
    uint32_t paletteIndex, uint32_t entryIndex, uint32_t variantIndex,
    float x, float y, float z, uint32_t monitorSlot);

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

constexpr uint32_t kMagic    = 0x5357485Au;  // 'ZHWS'
constexpr uint32_t kVersionV1 = 1u;
constexpr uint32_t kVersionV2 = 2u;
constexpr int      kRingSize = 32;

#pragma pack(push, 1)
// V2 header
struct WS_Header {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteIdx;
    uint32_t ReadIdx;
};

// V2 ring entry
struct WS_Entry {
    uint32_t PalletIndex;
    uint32_t ObjectIndex;
    uint32_t VariantIndex;
    float    X, Y, Z;
};

// V1 layout (kept for backward compat detection)
struct WS_SharedV1 {
    uint32_t Magic;
    uint32_t Version;
    uint32_t TriggerCounter;
    uint32_t ResponseCounter;
    uint32_t PalletIndex;
    uint32_t ObjectIndex;
    uint32_t VariantIndex;
    uint32_t _pad;
    float    X, Y, Z;
    uint32_t ResultStatus;
};
#pragma pack(pop)

static_assert(sizeof(WS_Header) == 16, "header");
static_assert(sizeof(WS_Entry)  == 24, "entry");

constexpr DWORD kTotalSize = sizeof(WS_Header) + kRingSize * sizeof(WS_Entry);

const wchar_t kMapName[] = L"ZeroHour_WorldSpawn_Shared";

HANDLE     g_hMap   = nullptr;
uint8_t*   g_Base   = nullptr;
bool       g_IsV2   = false;

// V1 fallback state
uint32_t   g_LastTrigger = 0;

bool EnsureOpen()
{
    if (g_Base) return true;
    g_hMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                kTotalSize, kMapName);
    if (!g_hMap) return false;
    g_Base = (uint8_t*)MapViewOfFile(g_hMap, FILE_MAP_ALL_ACCESS, 0, 0, kTotalSize);
    if (!g_Base) { CloseHandle(g_hMap); g_hMap = nullptr; return false; }

    __try {
        auto* hdr = (WS_Header*)g_Base;
        if (hdr->Magic == kMagic && hdr->Version == kVersionV2) {
            g_IsV2 = true;
        } else if (hdr->Magic == kMagic && hdr->Version == kVersionV1) {
            g_IsV2 = false;
            g_LastTrigger = ((WS_SharedV1*)g_Base)->TriggerCounter;
        } else {
            // Initialize as V2
            ZeroMemory(g_Base, kTotalSize);
            hdr->Magic   = kMagic;
            hdr->Version = kVersionV2;
            g_IsV2 = true;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
    return true;
}

void TickV2()
{
    auto* hdr = (WS_Header*)g_Base;
    uint32_t writeIdx, readIdx;
    __try {
        writeIdx = hdr->WriteIdx;
        readIdx  = hdr->ReadIdx;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return; }

    if (writeIdx == readIdx) return;

    uint32_t pending = writeIdx - readIdx;
    if (pending > (uint32_t)kRingSize) pending = kRingSize;

    for (uint32_t i = 0; i < pending; i++)
    {
        int slot = (int)((readIdx + i) % kRingSize);
        auto* e = (WS_Entry*)(g_Base + sizeof(WS_Header) + slot * sizeof(WS_Entry));

        uint32_t palette = 0, entry = 0, variant = 0;
        float x = 0, y = 0, z = 0;
        __try {
            palette = e->PalletIndex;
            entry   = e->ObjectIndex;
            variant = e->VariantIndex;
            x = e->X; y = e->Y; z = e->Z;
        } __except (EXCEPTION_EXECUTE_HANDLER) { continue; }

        ZH_Logf("[WorldSpawn V2] slot=%d P=%u E=%u V=%u pos=(%.2f,%.2f,%.2f)\n",
                slot, palette, entry, variant, x, y, z);

        HaloMapStudio_DispatchForgeSpawn(palette, entry, variant, x, y, z, 0u);
    }

    __try { hdr->ReadIdx = writeIdx; }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
}

void TickV1()
{
    auto* s = (WS_SharedV1*)g_Base;
    uint32_t trig = 0;
    __try { trig = s->TriggerCounter; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return; }

    if (trig == g_LastTrigger) return;
    g_LastTrigger = trig;

    uint32_t palette = 0, entry = 0, variant = 0;
    float x = 0, y = 0, z = 0;
    __try {
        palette = s->PalletIndex;
        entry   = s->ObjectIndex;
        variant = s->VariantIndex;
        x = s->X; y = s->Y; z = s->Z;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return; }

    ZH_Logf("[WorldSpawn V1] trigger=%u P=%u E=%u V=%u pos=(%.2f,%.2f,%.2f)\n",
            trig, palette, entry, variant, x, y, z);

    uint32_t dispatch = HaloMapStudio_DispatchForgeSpawn(palette, entry, variant, x, y, z, 0u);

    uint32_t resultStatus;
    switch (dispatch) {
        case 1:  resultStatus = 1u; break;
        case 2:  resultStatus = 4u; break;
        case 3:  resultStatus = 5u; break;
        default: resultStatus = 5u; break;
    }

    __try {
        s->ResultStatus    = resultStatus;
        s->ResponseCounter = trig;
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
}

} // namespace

extern "C" __declspec(dllexport) void WorldSpawnSnapshot_FramePumpTick()
{
    if (!EnsureOpen()) return;
    if (g_IsV2) TickV2(); else TickV1();
}
