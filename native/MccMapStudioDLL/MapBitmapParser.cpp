// MapBitmapParser.cpp
// =============================================================================
// Native bitmap-tag decoder for Halo MCC HaloReach .map files (tag layouts
// follow Reclaimer's Blam.HaloReach.bitmap definitions).
//
// The cache-file infrastructure (open file,
// header parse, tag index, gestalt, layout-table, resource-data reader) lives
// in MapCacheCommon.cpp. This file is now bitmap-tag-specific only:
//
//   * ParseBitmapTag - read the bitm metadata block (submaps + resource ids).
//   * BC1/BC3 + linear-format decoders.
//   * Public ZH_MBP_* exports.
//
// Mirrors Reclaimer.Blam.MccHaloReach.CacheFile + CacheFileU8 + the shared
// HaloReach.bitmap / cache_file_resource_gestalt / cache_file_resource_layout_table
// types.
//
// Limitations (defer):
//   * Xbox 360 Reach (CacheType.HaloReachRetail) not supported.
//   * BC6 (HDR float) not implemented - Reach forge maps don't ship any.
//   * BC7 is not decoded (Reach ships no BC7 bitmaps).
//   * Cubemaps / 3D / arrays handled as 2D submap[0] mip 0.
//
// What this parser handles:
//   * BC1 (DXT1, eng=14)
//   * BC2 (DXT3, eng=15)
//   * BC3 (DXT5, eng=16)
//   * BC4 (DXT5a / ATI1, eng=31, 40, 41, 42, 43)
//   * BC5 (DXN  / ATI2, eng=38, 44)
//   * Linear: A8R8G8B8, X8R8G8B8, R5G6B5, A1R5G5B5, A4R4G4B4, A8, Y8, AY8, A8Y8
// =============================================================================

#include "pch.h"
#include "MapBitmapParser.h"
#include "MapCacheCommon.h"
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstdio>

#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <new>
#ifndef _WIN32
#include <sys/mman.h> // #mem2: madvise for ZH_MBP_DropMapPages
#endif
#include <vector>
#include <unordered_map>
#include <mutex>
#include <stdio.h>

using namespace zh_mcc;

namespace {

// -----------------------------------------------------------------------------
// bitm tag-specific structures
// -----------------------------------------------------------------------------

struct BitmapSubmap {
    int16_t  width;
    int16_t  height;
    uint8_t  depth;
    uint8_t  flags;
    int8_t   bitmapType;
    int16_t  bitmapFormat;
    int16_t  moreFlags;
    uint8_t  mipCount;
    uint8_t  curve;
    uint8_t  interleavedIndex;
    uint8_t  index2;
    uint8_t  raw[56];   // verbatim bitmap_data block (MMS_BITMAP_DIAG dump)
};

struct BitmapTagParsed {
    std::vector<BitmapSubmap> submaps;
    std::vector<int32_t>      resourceIds;
    std::vector<int32_t>      interleavedIds;
};

// Per-cache bitmap parse cache (one entry per bitm tag id we've decoded).
struct BitmapSubCache {
    std::unordered_map<uint32_t, BitmapTagParsed> cache;
};

// Hooked into CacheHandle via cache->bitmapCache + cache->bitmapCacheDeleter
// at first access. ReleaseCacheHandle frees the sub-cache automatically.
BitmapSubCache* GetOrCreateBitmapCache(CacheHandle* cache) {
    if (cache->bitmapCache) return static_cast<BitmapSubCache*>(cache->bitmapCache);
    auto* sub = new (std::nothrow) BitmapSubCache();
    if (!sub) return nullptr;
    cache->bitmapCache = sub;
    cache->bitmapCacheDeleter = [](void* p) { delete static_cast<BitmapSubCache*>(p); };
    return sub;
}

// -----------------------------------------------------------------------------
// bitm tag metadata parse
// -----------------------------------------------------------------------------

bool ParseBitmapTag(CacheHandle* cache, uint32_t tagId, BitmapTagParsed& out) {
    if (tagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[tagId];
    if (te.classIndex < 0) return false;
    if (memcmp(te.classCode, "bitm", 4) != 0) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + 200 > cache->size) return false;
    const uint8_t* meta = cache->base + metaOff;

    constexpr int OFF_BITMAPS              = 124;
    constexpr int OFF_RESOURCES            = 168;
    constexpr int OFF_INTERLEAVED          = 180;
    constexpr int BITMAP_DATA_BLOCK_SIZE   = 56;
    constexpr int BITMAP_RESOURCE_SIZE     = 8;

    TagBlockRef bitmaps    = ReadTagBlock(meta + OFF_BITMAPS);
    TagBlockRef resources  = ReadTagBlock(meta + OFF_RESOURCES);
    TagBlockRef interleavd = ReadTagBlock(meta + OFF_INTERLEAVED);

    if (bitmaps.count < 0 || bitmaps.count > 1024) return false;
    if (resources.count < 0 || resources.count > 1024) return false;

    int64_t bitmapsOff = TagMetaFileOff(cache, bitmaps.pointer);
    int64_t resOff     = resources.count > 0   ? TagMetaFileOff(cache, resources.pointer)  : -1;
    int64_t intOff     = interleavd.count > 0  ? TagMetaFileOff(cache, interleavd.pointer) : -1;

    if (bitmapsOff < 0 ||
        (size_t)bitmapsOff + (size_t)bitmaps.count * BITMAP_DATA_BLOCK_SIZE > cache->size)
        return false;
    if (resources.count > 0 &&
        (resOff < 0 ||
         (size_t)resOff + (size_t)resources.count * BITMAP_RESOURCE_SIZE > cache->size))
        return false;

    out.submaps.resize(bitmaps.count);
    for (int i = 0; i < bitmaps.count; ++i) {
        const uint8_t* b = cache->base + bitmapsOff + i * BITMAP_DATA_BLOCK_SIZE;
        BitmapSubmap& s = out.submaps[i];
        s.width             = R16(b + 0);
        s.height            = R16(b + 2);
        s.depth             = b[4];
        s.flags             = b[5];
        s.bitmapType        = (int8_t)b[6];
        s.bitmapFormat      = R16(b + 8);
        s.moreFlags         = R16(b + 10);
        s.mipCount          = b[16];
        s.curve             = b[17];
        s.interleavedIndex  = b[18];
        s.index2            = b[19];
        memcpy(s.raw, b, sizeof(s.raw));
    }

    out.resourceIds.resize(resources.count);
    for (int i = 0; i < resources.count; ++i) {
        const uint8_t* r = cache->base + resOff + i * BITMAP_RESOURCE_SIZE;
        out.resourceIds[i] = R32(r);
    }

    if (interleavd.count > 0 && intOff >= 0 &&
        (size_t)intOff + (size_t)interleavd.count * BITMAP_RESOURCE_SIZE <= cache->size)
    {
        out.interleavedIds.resize(interleavd.count);
        for (int i = 0; i < interleavd.count; ++i) {
            const uint8_t* r = cache->base + intOff + i * BITMAP_RESOURCE_SIZE;
            out.interleavedIds[i] = R32(r);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Format helpers (HaloReach TextureFormat)
// -----------------------------------------------------------------------------

enum HRFormat {
    HR_A8        = 0,
    HR_Y8        = 1,
    HR_AY8       = 2,
    HR_A8Y8      = 3,
    HR_R5G6B5    = 6,
    HR_A1R5G5B5  = 8,
    HR_A4R4G4B4  = 9,
    HR_X8R8G8B8  = 10,
    HR_A8R8G8B8  = 11,
    HR_DXT1      = 14,
    HR_DXT3      = 15,
    HR_DXT5      = 16,
    // Reclaimer.Blam.HaloReach.TextureFormat extensions:
    // Reach format 30 = a16b16g16r16 (64 bpp, 4 x 16-bit UNORM, memory order R,G,B,A) per
    // reach_tag_test.exe's bitmap format enum + bitmap_format_bits_per_pixel_table[30] = 64. The shipped
    // lens-flare sprites (fx\bitmaps\lens_flares\star_flare, anamorphic_flare) use it; without this case
    // they failed to decode and the street lights fell back to a wrong-bitmap crossed-quad stand-in.
    HR_A16B16G16R16 = 30,
    HR_DXT5a     = 31,    // BC4 single-channel red
    // Reach format 36 = 4 bpp / 8 bytes per 4x4 block per reach_tag_test.exe's
    // bitmap_format_bits_per_pixel_table[36] = 4 and bitmap_format_bytes_per_block_table[36] = 8
    // (k_bitmap_format_count = 49). Single-channel BC4 layout; the engine's 3D LUT
    // `rasterizer\diffuse_power_specular\diffuse_power` (64x64x4) ships in it.
    HR_BC4_36    = 36,
    HR_DXN       = 38,    // BC5 dual-channel (X/Y normal)
    HR_DXT3a_alpha = 40,  // BC4 layout, source = DXT3 alpha plane
    HR_DXT3a_mono  = 41,  // BC4 layout, source = DXT3 alpha plane (mono)
    HR_DXT5a_alpha = 42,  // BC4 single-channel alpha
    HR_DXT5a_mono  = 43,  // BC4 single-channel mono
    HR_DXN_mono_alpha = 44, // BC5 layout, mono output
};

struct PixelLayout { int blockW; int blockH; int bytesPerBlock; bool supported; };

PixelLayout LayoutFor(int fmt) {
    switch (fmt) {
        case HR_DXT1:          return { 4, 4, 8,  true };
        case HR_DXT3:          return { 4, 4, 16, true };
        case HR_DXT5:          return { 4, 4, 16, true };
        // BC4 family: 8 bytes per 4x4 block.
        case HR_DXT5a:         return { 4, 4, 8,  true };
        case HR_BC4_36:        return { 4, 4, 8,  true };  // #glass-engine
        case HR_DXT3a_alpha:   return { 4, 4, 8,  true };
        case HR_DXT3a_mono:    return { 4, 4, 8,  true };
        case HR_DXT5a_alpha:   return { 4, 4, 8,  true };
        case HR_DXT5a_mono:    return { 4, 4, 8,  true };
        // BC5 family: 16 bytes per 4x4 block.
        case HR_DXN:           return { 4, 4, 16, true };
        case HR_DXN_mono_alpha:return { 4, 4, 16, true };
        case HR_A16B16G16R16:  return { 1, 1, 8,  true };  // #lamp-fx
        case HR_A8R8G8B8:      return { 1, 1, 4,  true };
        case HR_X8R8G8B8:      return { 1, 1, 4,  true };
        case HR_R5G6B5:        return { 1, 1, 2,  true };
        case HR_A1R5G5B5:      return { 1, 1, 2,  true };
        case HR_A4R4G4B4:      return { 1, 1, 2,  true };
        case HR_A8:            return { 1, 1, 1,  true };
        case HR_Y8:            return { 1, 1, 1,  true };
        case HR_AY8:           return { 1, 1, 1,  true };
        case HR_A8Y8:          return { 1, 1, 2,  true };
    }
    return { 0, 0, 0, false };
}

size_t MipByteSize(int fmt, int w, int h) {
    PixelLayout L = LayoutFor(fmt);
    if (!L.supported) return 0;
    int bw = (w + L.blockW - 1) / L.blockW;
    int bh = (h + L.blockH - 1) / L.blockH;
    if (bw < 1) bw = 1;
    if (bh < 1) bh = 1;
    return (size_t)bw * (size_t)bh * (size_t)L.bytesPerBlock;
}

// -----------------------------------------------------------------------------
// BC1 / BC3 decoders (output BGRA8)
// -----------------------------------------------------------------------------

static inline void Rgb565ToBgr8(uint16_t v, uint8_t out[3]) {
    int r = (v >> 11) & 0x1F;
    int g = (v >> 5)  & 0x3F;
    int b = v         & 0x1F;
    out[0] = (uint8_t)((b << 3) | (b >> 2));
    out[1] = (uint8_t)((g << 2) | (g >> 4));
    out[2] = (uint8_t)((r << 3) | (r >> 2));
}

void DecodeBC1(const uint8_t* src, int w, int h, uint8_t* dst) {
    int bw = (w + 3) >> 2;
    int bh = (h + 3) >> 2;
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            const uint8_t* b = src + (by * bw + bx) * 8;
            uint16_t c0 = RU16(b + 0);
            uint16_t c1 = RU16(b + 2);
            uint32_t bits = RU32(b + 4);
            uint8_t pal[4][4];
            uint8_t e0[3], e1[3];
            Rgb565ToBgr8(c0, e0);
            Rgb565ToBgr8(c1, e1);
            pal[0][0] = e0[0]; pal[0][1] = e0[1]; pal[0][2] = e0[2]; pal[0][3] = 255;
            pal[1][0] = e1[0]; pal[1][1] = e1[1]; pal[1][2] = e1[2]; pal[1][3] = 255;
            if (c0 > c1) {
                pal[2][0] = (uint8_t)((2 * e0[0] + e1[0]) / 3);
                pal[2][1] = (uint8_t)((2 * e0[1] + e1[1]) / 3);
                pal[2][2] = (uint8_t)((2 * e0[2] + e1[2]) / 3);
                pal[2][3] = 255;
                pal[3][0] = (uint8_t)((e0[0] + 2 * e1[0]) / 3);
                pal[3][1] = (uint8_t)((e0[1] + 2 * e1[1]) / 3);
                pal[3][2] = (uint8_t)((e0[2] + 2 * e1[2]) / 3);
                pal[3][3] = 255;
            } else {
                pal[2][0] = (uint8_t)((e0[0] + e1[0]) / 2);
                pal[2][1] = (uint8_t)((e0[1] + e1[1]) / 2);
                pal[2][2] = (uint8_t)((e0[2] + e1[2]) / 2);
                pal[2][3] = 255;
                pal[3][0] = 0; pal[3][1] = 0; pal[3][2] = 0; pal[3][3] = 0;
            }
            for (int py = 0; py < 4; ++py) {
                int dy = by * 4 + py;
                if (dy >= h) break;
                for (int px = 0; px < 4; ++px) {
                    int dx = bx * 4 + px;
                    if (dx >= w) break;
                    int idx = (bits >> ((py * 4 + px) * 2)) & 3;
                    uint8_t* o = dst + (dy * w + dx) * 4;
                    o[0] = pal[idx][0];
                    o[1] = pal[idx][1];
                    o[2] = pal[idx][2];
                    o[3] = pal[idx][3];
                }
            }
        }
    }
}

void DecodeBC3(const uint8_t* src, int w, int h, uint8_t* dst) {
    int bw = (w + 3) >> 2;
    int bh = (h + 3) >> 2;
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            const uint8_t* b = src + (by * bw + bx) * 16;
            uint8_t a0 = b[0], a1 = b[1];
            uint8_t aPal[8];
            aPal[0] = a0;
            aPal[1] = a1;
            if (a0 > a1) {
                aPal[2] = (uint8_t)((6 * a0 + 1 * a1) / 7);
                aPal[3] = (uint8_t)((5 * a0 + 2 * a1) / 7);
                aPal[4] = (uint8_t)((4 * a0 + 3 * a1) / 7);
                aPal[5] = (uint8_t)((3 * a0 + 4 * a1) / 7);
                aPal[6] = (uint8_t)((2 * a0 + 5 * a1) / 7);
                aPal[7] = (uint8_t)((1 * a0 + 6 * a1) / 7);
            } else {
                aPal[2] = (uint8_t)((4 * a0 + 1 * a1) / 5);
                aPal[3] = (uint8_t)((3 * a0 + 2 * a1) / 5);
                aPal[4] = (uint8_t)((2 * a0 + 3 * a1) / 5);
                aPal[5] = (uint8_t)((1 * a0 + 4 * a1) / 5);
                aPal[6] = 0;
                aPal[7] = 255;
            }
            uint64_t aBits = 0;
            for (int i = 0; i < 6; ++i) aBits |= ((uint64_t)b[2 + i]) << (8 * i);

            uint16_t c0 = RU16(b + 8);
            uint16_t c1 = RU16(b + 10);
            uint32_t cBits = RU32(b + 12);
            uint8_t pal[4][3];
            uint8_t e0[3], e1[3];
            Rgb565ToBgr8(c0, e0);
            Rgb565ToBgr8(c1, e1);
            pal[0][0] = e0[0]; pal[0][1] = e0[1]; pal[0][2] = e0[2];
            pal[1][0] = e1[0]; pal[1][1] = e1[1]; pal[1][2] = e1[2];
            pal[2][0] = (uint8_t)((2 * e0[0] + e1[0]) / 3);
            pal[2][1] = (uint8_t)((2 * e0[1] + e1[1]) / 3);
            pal[2][2] = (uint8_t)((2 * e0[2] + e1[2]) / 3);
            pal[3][0] = (uint8_t)((e0[0] + 2 * e1[0]) / 3);
            pal[3][1] = (uint8_t)((e0[1] + 2 * e1[1]) / 3);
            pal[3][2] = (uint8_t)((e0[2] + 2 * e1[2]) / 3);

            for (int py = 0; py < 4; ++py) {
                int dy = by * 4 + py;
                if (dy >= h) break;
                for (int px = 0; px < 4; ++px) {
                    int dx = bx * 4 + px;
                    if (dx >= w) break;
                    int ci = (cBits >> ((py * 4 + px) * 2)) & 3;
                    int ai = (int)((aBits >> ((py * 4 + px) * 3)) & 7);
                    uint8_t* o = dst + (dy * w + dx) * 4;
                    o[0] = pal[ci][0];
                    o[1] = pal[ci][1];
                    o[2] = pal[ci][2];
                    o[3] = aPal[ai];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// BC2 (DXT3) decoder - 4-bit-per-pixel uncompressed alpha plane + BC1-style
// RGB plane. 16 bytes per 4x4 block.
// -----------------------------------------------------------------------------

void DecodeBC2(const uint8_t* src, int w, int h, uint8_t* dst) {
    int bw = (w + 3) >> 2;
    int bh = (h + 3) >> 2;
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            const uint8_t* b = src + (by * bw + bx) * 16;

            // 8 bytes alpha (4-bit per pixel, row-major).
            uint64_t aBits = 0;
            for (int i = 0; i < 8; ++i) aBits |= ((uint64_t)b[i]) << (8 * i);

            // 8 bytes RGB (BC1-style; BC2 spec mandates the c0>c1 path even
            // when c0<=c1, but every encoder produces well-formed colour
            // tables, so the BC1 decoder logic is fine here too with a tiny
            // guard).
            uint16_t c0 = RU16(b + 8);
            uint16_t c1 = RU16(b + 10);
            uint32_t cBits = RU32(b + 12);
            uint8_t pal[4][3];
            uint8_t e0[3], e1[3];
            Rgb565ToBgr8(c0, e0);
            Rgb565ToBgr8(c1, e1);
            pal[0][0] = e0[0]; pal[0][1] = e0[1]; pal[0][2] = e0[2];
            pal[1][0] = e1[0]; pal[1][1] = e1[1]; pal[1][2] = e1[2];
            // BC2 always uses the 4-colour interpolation table (no 1-bit alpha).
            pal[2][0] = (uint8_t)((2 * e0[0] + e1[0]) / 3);
            pal[2][1] = (uint8_t)((2 * e0[1] + e1[1]) / 3);
            pal[2][2] = (uint8_t)((2 * e0[2] + e1[2]) / 3);
            pal[3][0] = (uint8_t)((e0[0] + 2 * e1[0]) / 3);
            pal[3][1] = (uint8_t)((e0[1] + 2 * e1[1]) / 3);
            pal[3][2] = (uint8_t)((e0[2] + 2 * e1[2]) / 3);

            for (int py = 0; py < 4; ++py) {
                int dy = by * 4 + py;
                if (dy >= h) break;
                for (int px = 0; px < 4; ++px) {
                    int dx = bx * 4 + px;
                    if (dx >= w) break;
                    int ci = (cBits >> ((py * 4 + px) * 2)) & 3;
                    int aNyb = (int)((aBits >> ((py * 4 + px) * 4)) & 0xF);
                    uint8_t a8 = (uint8_t)((aNyb << 4) | aNyb);
                    uint8_t* o = dst + (dy * w + dx) * 4;
                    o[0] = pal[ci][0];
                    o[1] = pal[ci][1];
                    o[2] = pal[ci][2];
                    o[3] = a8;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// BC4 single-channel decoder. 8 bytes per 4x4 block: 2 endpoints + 16 3-bit
// indices = 8 + 6 = 14 bytes... wait, 2 + 6 = 8. Correct.
//
// Layout: byte 0 = ref0, byte 1 = ref1, bytes 2..7 = 48-bit packed indices
// (3 bits per pixel, 16 pixels). Same scheme as BC3's alpha plane.
//
// Returns the decoded 8-bit values into a stride-1 buffer (caller decides
// where they go in BGRA).
// -----------------------------------------------------------------------------

void DecodeBC4Block(const uint8_t* b, uint8_t out[16]) {
    uint8_t r0 = b[0], r1 = b[1];
    uint8_t pal[8];
    pal[0] = r0;
    pal[1] = r1;
    if (r0 > r1) {
        pal[2] = (uint8_t)((6 * r0 + 1 * r1) / 7);
        pal[3] = (uint8_t)((5 * r0 + 2 * r1) / 7);
        pal[4] = (uint8_t)((4 * r0 + 3 * r1) / 7);
        pal[5] = (uint8_t)((3 * r0 + 4 * r1) / 7);
        pal[6] = (uint8_t)((2 * r0 + 5 * r1) / 7);
        pal[7] = (uint8_t)((1 * r0 + 6 * r1) / 7);
    } else {
        pal[2] = (uint8_t)((4 * r0 + 1 * r1) / 5);
        pal[3] = (uint8_t)((3 * r0 + 2 * r1) / 5);
        pal[4] = (uint8_t)((2 * r0 + 3 * r1) / 5);
        pal[5] = (uint8_t)((1 * r0 + 4 * r1) / 5);
        pal[6] = 0;
        pal[7] = 255;
    }
    uint64_t bits = 0;
    for (int i = 0; i < 6; ++i) bits |= ((uint64_t)b[2 + i]) << (8 * i);
    for (int i = 0; i < 16; ++i) {
        int idx = (int)((bits >> (i * 3)) & 7);
        out[i] = pal[idx];
    }
}

// Output mode for BC4 blocks: which BGRA channel(s) the decoded byte
// populates. Reach uses several variants.
enum BC4Out {
    BC4_OUT_RED_GREY,   // DXT5a / BC4: red channel; G=B=0 then we splat R into BGR
    BC4_OUT_ALPHA,      // DXT3a_alpha / DXT5a_alpha: alpha channel; RGB=0
    BC4_OUT_MONO,       // DXT3a_mono / DXT5a_mono: splat across BGR; A=255
};

void DecodeBC4(const uint8_t* src, int w, int h, uint8_t* dst, BC4Out mode) {
    int bw = (w + 3) >> 2;
    int bh = (h + 3) >> 2;
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            const uint8_t* b = src + (by * bw + bx) * 8;
            uint8_t v[16];
            DecodeBC4Block(b, v);
            for (int py = 0; py < 4; ++py) {
                int dy = by * 4 + py;
                if (dy >= h) break;
                for (int px = 0; px < 4; ++px) {
                    int dx = bx * 4 + px;
                    if (dx >= w) break;
                    uint8_t* o = dst + (dy * w + dx) * 4;
                    uint8_t s = v[py * 4 + px];
                    switch (mode) {
                        case BC4_OUT_RED_GREY:
                            o[0] = 0;     // B
                            o[1] = 0;     // G
                            o[2] = s;     // R
                            o[3] = 255;
                            break;
                        case BC4_OUT_ALPHA:
                            o[0] = 0;
                            o[1] = 0;
                            o[2] = 0;
                            o[3] = s;
                            break;
                        case BC4_OUT_MONO:
                            o[0] = s;
                            o[1] = s;
                            o[2] = s;
                            o[3] = 255;
                            break;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// BC5 (DXN / ATI2) decoder - two BC4 channels back-to-back. First block ->
// red (X), second block -> green (Y). For normal maps, Z is reconstructed
// from X/Y. 16 bytes per 4x4 block (8+8).
// -----------------------------------------------------------------------------

// BC4 block with SIGNED (int8) endpoints -> 0.5-centred UNORM bytes. Reach DXN normal maps
// are BC5_SNORM (the engine samples .xy directly, no *2-1). Decoding the signed endpoints as unsigned
// scrambles near-flat texels to +-1 (|xy|~1.3, z->0) -> flat/random normals = grainy DARK rocks.
static void DecodeBC4BlockSnorm(const uint8_t* b, uint8_t out[16]) {
    int e0 = (int8_t)b[0], e1 = (int8_t)b[1];
    if (e0 < -127) e0 = -127; if (e1 < -127) e1 = -127;
    int pal[8]; pal[0] = e0; pal[1] = e1;
    if (e0 > e1) { for (int k = 1; k <= 6; ++k) pal[k + 1] = ((7 - k) * e0 + k * e1) / 7; }
    else { for (int k = 1; k <= 4; ++k) pal[k + 1] = ((5 - k) * e0 + k * e1) / 5; pal[6] = -127; pal[7] = 127; }
    uint64_t bits = 0; for (int i = 0; i < 6; ++i) bits |= ((uint64_t)b[2 + i]) << (8 * i);
    for (int i = 0; i < 16; ++i) { int v = pal[(int)((bits >> (i * 3)) & 7)];
        float f = (float)v / 127.0f * 0.5f + 0.5f; int q = (int)(f * 255.0f + 0.5f); if (q < 0) q = 0; if (q > 255) q = 255; out[i] = (uint8_t)q; }
}
void DecodeBC5(const uint8_t* src, int w, int h, uint8_t* dst, bool monoAlpha) {
    int bw = (w + 3) >> 2;
    int bh = (h + 3) >> 2;
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            const uint8_t* b = src + (by * bw + bx) * 16;
            uint8_t r[16], g[16];
            if (monoAlpha) { DecodeBC4Block(b + 0, r); DecodeBC4Block(b + 8, g); }          // lightmap DM: UNORM data
            else           { DecodeBC4BlockSnorm(b + 0, r); DecodeBC4BlockSnorm(b + 8, g); } // DXN normal map: SNORM
            for (int py = 0; py < 4; ++py) {
                int dy = by * 4 + py;
                if (dy >= h) break;
                for (int px = 0; px < 4; ++px) {
                    int dx = bx * 4 + px;
                    if (dx >= w) break;
                    uint8_t* o = dst + (dy * w + dx) * 4;
                    uint8_t rx = r[py * 4 + px];
                    uint8_t gy = g[py * 4 + px];
                    if (monoAlpha) {
                        // DXN_mono_alpha: red plane -> grey, green plane -> alpha.
                        o[0] = rx;
                        o[1] = rx;
                        o[2] = rx;
                        o[3] = gy;
                    } else {
                        // BC5 normal map: reconstruct Z = sqrt(1 - X^2 - Y^2)
                        // in normalised [-1,1] space, then map back to [0,255].
                        float fx = (float)rx / 255.0f * 2.0f - 1.0f;
                        float fy = (float)gy / 255.0f * 2.0f - 1.0f;
                        float zz = 1.0f - fx * fx - fy * fy;
                        if (zz < 0.0f) zz = 0.0f;
                        float fz = (float)sqrt(zz);
                        uint8_t bz = (uint8_t)(int)((fz * 0.5f + 0.5f) * 255.0f + 0.5f);
                        o[0] = bz;     // B
                        o[1] = gy;     // G (Y normal)
                        o[2] = rx;     // R (X normal)
                        o[3] = 255;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Linear-format decoders
// -----------------------------------------------------------------------------

void DecodeA8R8G8B8(const uint8_t* src, int w, int h, uint8_t* dst, bool x8) {
    if (!x8) {
        memcpy(dst, src, (size_t)w * h * 4);
        return;
    }
    for (int i = 0; i < w * h; ++i) {
        dst[i * 4 + 0] = src[i * 4 + 0];
        dst[i * 4 + 1] = src[i * 4 + 1];
        dst[i * 4 + 2] = src[i * 4 + 2];
        dst[i * 4 + 3] = 255;
    }
}

// A16b16g16r16 -> BGRA8 (high byte of each 16-bit UNORM channel; source order R,G,B,A).
void DecodeA16B16G16R16(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int i = 0; i < w * h; ++i) {
        const uint8_t* p = src + (size_t)i * 8;
        dst[i * 4 + 0] = p[5];  // B
        dst[i * 4 + 1] = p[3];  // G
        dst[i * 4 + 2] = p[1];  // R
        dst[i * 4 + 3] = p[7];  // A
    }
}

void DecodeR5G6B5(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int i = 0; i < w * h; ++i) {
        uint16_t v = RU16(src + i * 2);
        uint8_t bgr[3];
        Rgb565ToBgr8(v, bgr);
        dst[i * 4 + 0] = bgr[0];
        dst[i * 4 + 1] = bgr[1];
        dst[i * 4 + 2] = bgr[2];
        dst[i * 4 + 3] = 255;
    }
}

void DecodeA1R5G5B5(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int i = 0; i < w * h; ++i) {
        uint16_t v = RU16(src + i * 2);
        int r = (v >> 10) & 0x1F;
        int g = (v >> 5)  & 0x1F;
        int b = v         & 0x1F;
        int a = (v >> 15) & 1;
        dst[i * 4 + 0] = (uint8_t)((b << 3) | (b >> 2));
        dst[i * 4 + 1] = (uint8_t)((g << 3) | (g >> 2));
        dst[i * 4 + 2] = (uint8_t)((r << 3) | (r >> 2));
        dst[i * 4 + 3] = (uint8_t)(a ? 255 : 0);
    }
}

void DecodeA4R4G4B4(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int i = 0; i < w * h; ++i) {
        uint16_t v = RU16(src + i * 2);
        int b = v & 0xF;
        int g = (v >> 4) & 0xF;
        int r = (v >> 8) & 0xF;
        int a = (v >> 12) & 0xF;
        dst[i * 4 + 0] = (uint8_t)((b << 4) | b);
        dst[i * 4 + 1] = (uint8_t)((g << 4) | g);
        dst[i * 4 + 2] = (uint8_t)((r << 4) | r);
        dst[i * 4 + 3] = (uint8_t)((a << 4) | a);
    }
}

void DecodeA8(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int i = 0; i < w * h; ++i) {
        dst[i * 4 + 0] = 0;
        dst[i * 4 + 1] = 0;
        dst[i * 4 + 2] = 0;
        dst[i * 4 + 3] = src[i];
    }
}

void DecodeY8(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int i = 0; i < w * h; ++i) {
        uint8_t y = src[i];
        dst[i * 4 + 0] = y;
        dst[i * 4 + 1] = y;
        dst[i * 4 + 2] = y;
        dst[i * 4 + 3] = 255;
    }
}

void DecodeAY8(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int i = 0; i < w * h; ++i) {
        uint8_t v = src[i];
        dst[i * 4 + 0] = v;
        dst[i * 4 + 1] = v;
        dst[i * 4 + 2] = v;
        dst[i * 4 + 3] = v;
    }
}

void DecodeA8Y8(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int i = 0; i < w * h; ++i) {
        uint8_t y = src[i * 2 + 0];
        uint8_t a = src[i * 2 + 1];
        dst[i * 4 + 0] = y;
        dst[i * 4 + 1] = y;
        dst[i * 4 + 2] = y;
        dst[i * 4 + 3] = a;
    }
}

bool DecodePixels(int fmt, const uint8_t* src, size_t srcSize, int w, int h, uint8_t* dst) {
    PixelLayout L = LayoutFor(fmt);
    if (!L.supported) return false;
    size_t need = MipByteSize(fmt, w, h);
    if (srcSize < need) return false;
    switch (fmt) {
        case HR_DXT1:     DecodeBC1(src, w, h, dst);             return true;
        case HR_DXT3:     DecodeBC2(src, w, h, dst);             return true;
        case HR_DXT5:     DecodeBC3(src, w, h, dst);             return true;
        case HR_DXT5a:        DecodeBC4(src, w, h, dst, BC4_OUT_RED_GREY); return true;
        case HR_BC4_36:       DecodeBC4(src, w, h, dst, BC4_OUT_MONO);     return true;  // #glass-engine
        case HR_DXT3a_alpha:  DecodeBC4(src, w, h, dst, BC4_OUT_ALPHA);    return true;
        case HR_DXT3a_mono:   DecodeBC4(src, w, h, dst, BC4_OUT_MONO);     return true;
        case HR_DXT5a_alpha:  DecodeBC4(src, w, h, dst, BC4_OUT_ALPHA);    return true;
        case HR_DXT5a_mono:   DecodeBC4(src, w, h, dst, BC4_OUT_MONO);     return true;
        case HR_DXN:          DecodeBC5(src, w, h, dst, /*monoAlpha=*/false); return true;
        case HR_DXN_mono_alpha:DecodeBC5(src, w, h, dst, /*monoAlpha=*/true); return true;
        case HR_A16B16G16R16: DecodeA16B16G16R16(src, w, h, dst); return true;  // #lamp-fx
        case HR_A8R8G8B8: DecodeA8R8G8B8(src, w, h, dst, false); return true;
        case HR_X8R8G8B8: DecodeA8R8G8B8(src, w, h, dst, true);  return true;
        case HR_R5G6B5:   DecodeR5G6B5(src, w, h, dst);          return true;
        case HR_A1R5G5B5: DecodeA1R5G5B5(src, w, h, dst);        return true;
        case HR_A4R4G4B4: DecodeA4R4G4B4(src, w, h, dst);        return true;
        case HR_A8:       DecodeA8(src, w, h, dst);              return true;
        case HR_Y8:       DecodeY8(src, w, h, dst);              return true;
        case HR_AY8:      DecodeAY8(src, w, h, dst);             return true;
        case HR_A8Y8:     DecodeA8Y8(src, w, h, dst);             return true;
    }
    return false;
}

// Bytes of a mip chain (levels 0..levels-1, stopping at 1x1). Used to cap
// ReadResourceData so a texture decode copies only its own chain out of the page cache
// instead of a 16-64 MB slab (the copy runs under pageCacheMutex - 30 decode threads were
// serialising behind each other's multi-MB memcpy).
static size_t ChainBytes(int fmt, int width, int height, int levels)
{
    size_t total = 0; int w = width, h = height;
    for (int m = 0; m < levels; ++m) {
        size_t ms = MipByteSize(fmt, w, h);
        if (ms == 0) break;
        total += ms;
        if (w <= 1 && h <= 1) break;
        w = w > 1 ? w / 2 : 1; h = h > 1 ? h / 2 : 1;
    }
    return total;
}

int64_t LocateMip(int fmt, int width, int height, int mipLevel,
                  size_t resSize, int* outW, int* outH)
{
    int w = width, h = height;
    int64_t off = 0;
    for (int m = 0; m < mipLevel; ++m) {
        size_t s = MipByteSize(fmt, w, h);
        if (s == 0) return -1;
        off += (int64_t)s;
        if ((size_t)off >= resSize) return -1;
        w = w > 1 ? w / 2 : 1;
        h = h > 1 ? h / 2 : 1;
    }
    *outW = w;
    *outH = h;
    return off;
}

const BitmapTagParsed* GetParsedBitmap(CacheHandle* cache, uint32_t tagId) {
    std::lock_guard<std::mutex> lk(cache->parseMutex);
    BitmapSubCache* sub = GetOrCreateBitmapCache(cache);
    if (!sub) return nullptr;
    auto it = sub->cache.find(tagId);
    if (it != sub->cache.end()) return &it->second;
    BitmapTagParsed parsed;
    if (!ParseBitmapTag(cache, tagId, parsed)) return nullptr;
    auto ins = sub->cache.emplace(tagId, std::move(parsed));
    return &ins.first->second;
}

// SEH wrappers - keep all C++ object construction/destruction outside __try.
const BitmapTagParsed* SehGetParsedBitmap(CacheHandle* cache, uint32_t tagId) {
    __try { return GetParsedBitmap(cache, tagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
}

// MMS_DECODE_PROF=1: per-stage timing of DecodeBitmapInner, printed to stderr at exit.
static std::atomic<uint64_t> g_dpCalls{0}, g_dpParse{0}, g_dpGap{0}, g_dpRead{0}, g_dpPre{0}, g_dpDecode{0};
static void DecodeProfAdd(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1,
                          std::chrono::steady_clock::time_point t2, std::chrono::steady_clock::time_point t3,
                          std::chrono::steady_clock::time_point t4, std::chrono::steady_clock::time_point t5)
{
    using namespace std::chrono;
    g_dpCalls.fetch_add(1);
    g_dpParse.fetch_add((uint64_t)duration_cast<nanoseconds>(t1 - t0).count());
    g_dpGap.fetch_add((uint64_t)duration_cast<nanoseconds>(t2 - t1).count());
    g_dpRead.fetch_add((uint64_t)duration_cast<nanoseconds>(t3 - t2).count());
    g_dpPre.fetch_add((uint64_t)duration_cast<nanoseconds>(t4 - t3).count());
    g_dpDecode.fetch_add((uint64_t)duration_cast<nanoseconds>(t5 - t4).count());
}
struct DecodeProfReporter {
    ~DecodeProfReporter() {
        if (!getenv("MMS_DECODE_PROF")) return;
        fprintf(stderr, "MMS_DECODE_PROF calls=%llu parse=%.0f gap=%.0f read=%.0f pre=%.0f decode=%.0f ms\n",
            (unsigned long long)g_dpCalls.load(), g_dpParse.load() / 1e6, g_dpGap.load() / 1e6, g_dpRead.load() / 1e6,
            g_dpPre.load() / 1e6, g_dpDecode.load() / 1e6);
    }
};
static DecodeProfReporter g_dpReporter;

bool DecodeBitmapInner(CacheHandle* cache, uint32_t tagId,
                       uint32_t submapIndex, uint32_t mipLevel,
                       uint8_t** outRgba, uint32_t* outW, uint32_t* outH,
                       bool keepAlpha = false)
{
    auto _dp0 = std::chrono::steady_clock::now();
    const BitmapTagParsed* parsed = GetParsedBitmap(cache, tagId);
    auto _dp1 = std::chrono::steady_clock::now();
    if (!parsed) return false;
    if (submapIndex >= parsed->submaps.size()) return false;
    const BitmapSubmap& s = parsed->submaps[submapIndex];

    PixelLayout L = LayoutFor(s.bitmapFormat);
    if (!L.supported) return false;

    if (submapIndex >= parsed->resourceIds.size()) return false;
    int32_t resourceId = parsed->resourceIds[submapIndex];
    if (!parsed->interleavedIds.empty() &&
        s.interleavedIndex < parsed->interleavedIds.size())
    {
        resourceId = parsed->interleavedIds[s.interleavedIndex];
    }

    size_t resSize = 0;
    constexpr size_t kMaxRead = 16 * 1024 * 1024;
    // ReadResourceData takes cache->parseMutex internally for the shared-cache
    // open path; don't hold it here (std::mutex is non-recursive - double-lock
    // throws system_error which the SEH wrapper would silently swallow).
    size_t needA = ChainBytes(s.bitmapFormat, s.width, s.height, (int)mipLevel + 1); // #tight-read
    size_t capA = (needA > 0 && needA < kMaxRead) ? needA : kMaxRead;
    auto _dp2 = std::chrono::steady_clock::now();
    uint8_t* res = ReadResourceData(cache, resourceId, capA, &resSize);
    auto _dp3 = std::chrono::steady_clock::now();
    if (!res) return false;

    int mw = 0, mh = 0;
    int64_t mipOff = LocateMip(s.bitmapFormat, s.width, s.height,
                               (int)mipLevel, resSize, &mw, &mh);
    if (mipOff < 0) { free(res); return false; }
    size_t mipBytes = MipByteSize(s.bitmapFormat, mw, mh);
    if (mipBytes == 0 || (size_t)mipOff + mipBytes > resSize) { free(res); return false; }

    // MMS_BITMAP_DIAG=<hex tag id>: debug aid (default off). Prints the raw
    // bitmap_data block + resource location for ONE bitmap to stderr and the
    // native log; MMS_BITMAP_DIAG_OUT=<path> additionally writes the raw
    // resource bytes (all mips, undecoded) to that file.
    {
        char dv[16] = {0};
        DWORD dn = GetEnvironmentVariableA("MMS_BITMAP_DIAG", dv, (DWORD)sizeof(dv));
        if (dn > 0 && dn < sizeof(dv) && (uint32_t)strtoul(dv, nullptr, 16) == tagId) {
            char rawHex[56 * 2 + 1];
            for (int i = 0; i < 56; ++i) _snprintf_s(rawHex + i * 2, 3, _TRUNCATE, "%02x", s.raw[i]);
            char first[16 * 2 + 1];
            for (int i = 0; i < 16; ++i) _snprintf_s(first + i * 2, 3, _TRUNCATE, "%02x", (i < (int)mipBytes) ? res[mipOff + i] : 0);
            char line[768];
            _snprintf_s(line, sizeof(line), _TRUNCATE,
                "BITMAP_DIAG tag=0x%x submap=%u mip=%u w=%d h=%d depth=%u flags=0x%02x type=%d fmt=%d moreFlags=0x%04x mips=%u curve=%u ilIdx=%u idx2=%u nsub=%u nres=%u nil=%u rid=0x%x resSize=%llu mipOff=%lld mipBytes=%llu mw=%d mh=%d first16=%s raw56=%s",
                tagId, submapIndex, mipLevel, (int)s.width, (int)s.height, (unsigned)s.depth, (unsigned)s.flags,
                (int)s.bitmapType, (int)s.bitmapFormat, (unsigned)(uint16_t)s.moreFlags, (unsigned)s.mipCount,
                (unsigned)s.curve, (unsigned)s.interleavedIndex, (unsigned)s.index2,
                (unsigned)parsed->submaps.size(), (unsigned)parsed->resourceIds.size(), (unsigned)parsed->interleavedIds.size(),
                (unsigned)resourceId, (unsigned long long)resSize, (long long)mipOff, (unsigned long long)mipBytes,
                mw, mh, first, rawHex);
            fprintf(stderr, "%s\n", line);
            NativeDiag("%s", line);
            char op[512] = {0};
            DWORD on = GetEnvironmentVariableA("MMS_BITMAP_DIAG_OUT", op, (DWORD)sizeof(op));
            if (on > 0 && on < sizeof(op)) {
                FILE* f = nullptr;
#ifdef _MSC_VER
                fopen_s(&f, op, "wb");
#else
                f = fopen(op, "wb");
#endif
                if (f) { fwrite(res, 1, resSize, f); fclose(f); }
            }
        }
    }

    size_t outBytes = (size_t)mw * mh * 4;
    uint8_t* dst = (uint8_t*)malloc(outBytes);
    if (!dst) { free(res); return false; }

    auto _dp4 = std::chrono::steady_clock::now();
    if (!DecodePixels(s.bitmapFormat, res + mipOff, mipBytes, mw, mh, dst)) {
        free(dst);
        free(res);
        return false;
    }
    auto _dp5 = std::chrono::steady_clock::now();
    DecodeProfAdd(_dp0, _dp1, _dp2, _dp3, _dp4, _dp5);

    free(res);
    // Alpha is preserved as authored. The viewer side decides per-material
    // whether to drop it: opaque BSP / object materials encode the texture
    // as Bgr24 via TextureAdapter.FromBitmapSource(dropAlpha: true), which
    // produces alpha=1 at the GPU sample regardless of what the bitmap
    // contains - so Reach's spec/glow-in-alpha bitmaps don't cause opaque
    // surfaces to read as see-through. Foliage / hologram / decal materials
    // pass dropAlpha:false so the alpha-shape mask survives - that's what
    // gives leaves their cutout silhouette and forge holos their painted
    // edges.
    //
    // Alpha is always preserved (a forced alpha=255 stamp would kill the
    // leaf-shape mask and render every foliage card as an opaque rectangle).
    // The keepAlpha parameter is retained for API stability and is a no-op.
    (void)keepAlpha;
    *outRgba = dst;
    *outW = (uint32_t)mw;
    *outH = (uint32_t)mh;
    return true;
}

bool SehDecodeBitmap(CacheHandle* cache, uint32_t tagId,
                     uint32_t submapIndex, uint32_t mipLevel,
                     uint8_t** outRgba, uint32_t* outW, uint32_t* outH,
                     bool keepAlpha = false)
{
    __try {
        return DecodeBitmapInner(cache, tagId, submapIndex, mipLevel,
                                 outRgba, outW, outH, keepAlpha);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Volume-aware decode: pull z-slice `sliceIndex` of `mipLevel` from a depth-N
// volume submap. The engine stores depth slices contiguously at each mip
// (slice-major: the N full 2D images for mip 0, then the N images for mip 1, ...),
// which matches the DX `Texture3D.Sample(float3(uv, z))` addressing used by
// lightmap_sampling.hlsl_include for lightprobe_hdr_color. We locate the mip's
// base offset exactly as DecodeBitmapInner does, then advance by
// `sliceIndex * mipBytes` to reach the requested slice. sliceIndex 0 is
// byte-identical to DecodeBitmapInner. Out-of-range slice -> fail (caller falls
// back to the dom stand-in).
bool DecodeBitmapSliceInner(CacheHandle* cache, uint32_t tagId,
                            uint32_t submapIndex, uint32_t mipLevel,
                            uint32_t sliceIndex,
                            uint8_t** outRgba, uint32_t* outW, uint32_t* outH)
{
    const BitmapTagParsed* parsed = GetParsedBitmap(cache, tagId);
    if (!parsed) return false;
    if (submapIndex >= parsed->submaps.size()) return false;
    const BitmapSubmap& s = parsed->submaps[submapIndex];

    // depth==0 is a malformed/unset field; treat as a single-slice 2D image.
    uint32_t depth = (s.depth == 0) ? 1u : (uint32_t)s.depth;
    if (sliceIndex >= depth) return false;
    // Fast path: slice 0 is identical to the plain 2D decode.
    if (sliceIndex == 0)
        return DecodeBitmapInner(cache, tagId, submapIndex, mipLevel,
                                 outRgba, outW, outH, /*keepAlpha=*/false);

    PixelLayout L = LayoutFor(s.bitmapFormat);
    if (!L.supported) return false;

    if (submapIndex >= parsed->resourceIds.size()) return false;
    int32_t resourceId = parsed->resourceIds[submapIndex];
    if (!parsed->interleavedIds.empty() &&
        s.interleavedIndex < parsed->interleavedIds.size())
    {
        resourceId = parsed->interleavedIds[s.interleavedIndex];
    }

    size_t resSize = 0;
    constexpr size_t kMaxRead = 48 * 1024 * 1024;   // volume payloads are larger
    uint8_t* res = ReadResourceData(cache, resourceId, kMaxRead, &resSize);
    if (!res) return false;

    int mw = 0, mh = 0;
    // For a volume the per-mip stride is depth*mipBytes; LocateMip's 2D walk
    // would under-skip. Recompute the mip base accounting for all slices.
    int w = s.width, h = s.height;
    int64_t mipBase = 0;
    bool mipOk = true;
    for (uint32_t m = 0; m < mipLevel; ++m) {
        size_t ms = MipByteSize(s.bitmapFormat, w, h);
        if (ms == 0) { mipOk = false; break; }
        mipBase += (int64_t)ms * (int64_t)depth;     // skip ALL slices of this mip
        if ((size_t)mipBase >= resSize) { mipOk = false; break; }
        w = w > 1 ? w / 2 : 1;
        h = h > 1 ? h / 2 : 1;
    }
    if (!mipOk) { free(res); return false; }
    mw = w; mh = h;

    size_t mipBytes = MipByteSize(s.bitmapFormat, mw, mh);
    if (mipBytes == 0) { free(res); return false; }
    int64_t sliceOff = mipBase + (int64_t)mipBytes * (int64_t)sliceIndex;
    if ((size_t)sliceOff + mipBytes > resSize) { free(res); return false; }

    size_t outBytes = (size_t)mw * mh * 4;
    uint8_t* dst = (uint8_t*)malloc(outBytes);
    if (!dst) { free(res); return false; }

    if (!DecodePixels(s.bitmapFormat, res + sliceOff, mipBytes, mw, mh, dst)) {
        free(dst);
        free(res);
        return false;
    }

    free(res);
    *outRgba = dst;
    *outW = (uint32_t)mw;
    *outH = (uint32_t)mh;
    return true;
}

bool SehDecodeBitmapSlice(CacheHandle* cache, uint32_t tagId,
                          uint32_t submapIndex, uint32_t mipLevel,
                          uint32_t sliceIndex,
                          uint8_t** outRgba, uint32_t* outW, uint32_t* outH)
{
    __try {
        return DecodeBitmapSliceInner(cache, tagId, submapIndex, mipLevel,
                                      sliceIndex, outRgba, outW, outH);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Decode ONE face of a CUBEMAP submap (bitmapType==2 -> 6 faces). A cube's `depth`
// field is 1 even though it stores 6 faces, so DecodeBitmapSliceInner refuses faces
// 1..5; this keys off a fixed 6-face count. strideMode: 0 = face-major (all mips of
// face 0, then face 1, ... : the DDS/D3D cube layout), 1 = mip-major (all 6 faces of
// mip 0, then mip 1, ... : the volume-slice layout with depth=6). Alpha is KEPT.
bool DecodeBitmapFaceInner(CacheHandle* cache, uint32_t tagId,
                           uint32_t submapIndex, uint32_t mipLevel,
                           uint32_t faceIndex, uint32_t strideMode,
                           uint8_t** outRgba, uint32_t* outW, uint32_t* outH)
{
    const BitmapTagParsed* parsed = GetParsedBitmap(cache, tagId);
    if (!parsed) return false;
    if (submapIndex >= parsed->submaps.size()) return false;
    const BitmapSubmap& s = parsed->submaps[submapIndex];
    if (s.bitmapType != 2) return false;          // not a cubemap
    constexpr uint32_t kFaces = 6u;
    if (faceIndex >= kFaces) return false;

    PixelLayout L = LayoutFor(s.bitmapFormat);
    if (!L.supported) return false;

    if (submapIndex >= parsed->resourceIds.size()) return false;
    int32_t resourceId = parsed->resourceIds[submapIndex];
    if (!parsed->interleavedIds.empty() &&
        s.interleavedIndex < parsed->interleavedIds.size())
    {
        resourceId = parsed->interleavedIds[s.interleavedIndex];
    }

    size_t resSize = 0;
    constexpr size_t kMaxRead = 48 * 1024 * 1024;
    size_t needB = (strideMode == 0) ? ChainBytes(s.bitmapFormat, s.width, s.height, (int)s.mipCount + 1) * (size_t)kFaces : 0; // #tight-read
    size_t capB = (needB > 0 && needB < kMaxRead) ? needB : kMaxRead;
    uint8_t* res = ReadResourceData(cache, resourceId, capB, &resSize);
    if (!res) return false;

    // Number of mip levels in one face chain (mipCount excludes the base level).
    uint32_t levels = (uint32_t)s.mipCount + 1u;

    int64_t faceOff = 0;
    int w = s.width, h = s.height;
    bool ok = true;
    if (strideMode == 0) {
        // face-major: one full mip chain per face, faces contiguous.
        int64_t chainBytes = 0;
        int cw = s.width, ch = s.height;
        for (uint32_t m = 0; m < levels; ++m) {
            size_t ms = MipByteSize(s.bitmapFormat, cw, ch);
            if (ms == 0) { ok = false; break; }
            chainBytes += (int64_t)ms;
            if (cw == 1 && ch == 1) break;
            cw = cw > 1 ? cw / 2 : 1;
            ch = ch > 1 ? ch / 2 : 1;
        }
        if (!ok) { free(res); return false; }
        faceOff = chainBytes * (int64_t)faceIndex;
        for (uint32_t m = 0; m < mipLevel; ++m) {
            size_t ms = MipByteSize(s.bitmapFormat, w, h);
            if (ms == 0) { ok = false; break; }
            faceOff += (int64_t)ms;
            w = w > 1 ? w / 2 : 1;
            h = h > 1 ? h / 2 : 1;
        }
    } else {
        // mip-major: all 6 faces of each mip contiguous (volume-slice math, depth = 6).
        for (uint32_t m = 0; m < mipLevel; ++m) {
            size_t ms = MipByteSize(s.bitmapFormat, w, h);
            if (ms == 0) { ok = false; break; }
            faceOff += (int64_t)ms * (int64_t)kFaces;
            w = w > 1 ? w / 2 : 1;
            h = h > 1 ? h / 2 : 1;
        }
        if (ok) {
            size_t ms = MipByteSize(s.bitmapFormat, w, h);
            if (ms == 0) ok = false; else faceOff += (int64_t)ms * (int64_t)faceIndex;
        }
    }
    if (!ok) { free(res); return false; }

    size_t mipBytes = MipByteSize(s.bitmapFormat, w, h);
    if (mipBytes == 0) { free(res); return false; }
    if (faceOff < 0 || (size_t)faceOff + mipBytes > resSize) { free(res); return false; }

    size_t outBytes = (size_t)w * h * 4;
    uint8_t* dst = (uint8_t*)malloc(outBytes);
    if (!dst) { free(res); return false; }
    if (!DecodePixels(s.bitmapFormat, res + faceOff, mipBytes, w, h, dst)) {
        free(dst); free(res); return false;
    }
    free(res);
    *outRgba = dst;
    *outW = (uint32_t)w;
    *outH = (uint32_t)h;
    return true;
}

bool SehDecodeBitmapFace(CacheHandle* cache, uint32_t tagId,
                         uint32_t submapIndex, uint32_t mipLevel,
                         uint32_t faceIndex, uint32_t strideMode,
                         uint8_t** outRgba, uint32_t* outW, uint32_t* outH)
{
    __try {
        return DecodeBitmapFaceInner(cache, tagId, submapIndex, mipLevel,
                                     faceIndex, strideMode, outRgba, outW, outH);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

extern "C" __declspec(dllexport) uint64_t __stdcall ZH_MBP_OpenCache(const wchar_t* path)
{
    return zh_mcc::AcquireCacheHandle(path);
}

extern "C" __declspec(dllexport) void __stdcall ZH_MBP_CloseCache(uint64_t cacheHandle)
{
    zh_mcc::ReleaseCacheHandle(cacheHandle);
}

// Generic tag-name lookup. Copies the cached tag name (UTF-8 path,
// e.g. "levels\multi\forge_halo\bitmaps\forge_halo_rocks_blend") into the
// caller's buffer. Returns the number of bytes written (excluding null), or 0
// on miss / invalid args. Buffer is always null-terminated when bufLen > 0.
extern "C" __declspec(dllexport) int __stdcall ZH_TAG_GetName(
    uint64_t cacheHandle, uint32_t tagId, char* buf, int bufLen)
{
    if (!buf || bufLen <= 0) return 0;
    buf[0] = 0;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;
    if (tagId >= cache->tags.size()) return 0;
    const std::string& name = cache->tags[tagId].tagName;
    if (name.empty()) return 0;
    int copy = (int)name.size();
    if (copy > bufLen - 1) copy = bufLen - 1;
    memcpy(buf, name.data(), (size_t)copy);
    buf[copy] = 0;
    return copy;
}

// Scan the first `maxBytes` of a tag's payload for a tag-ref whose
// 4-cc class code matches `groupAscii` (4 chars + NUL, e.g. "hlmt").
// Returns the resolved short tag id (0..tags.size()-1) of the first
// match whose tag-id field at +0xC is a valid in-cache index of the
// expected class. 0xFFFFFFFFu on no match.
//
// On-disk tag-ref layout (16 bytes):
//   +0x00  uint32   class id stored as the BIG-ENDIAN fourcc value
//                    e.g. 'hlmt' encodes to u32 0x686C6D74, which on
//                    x64 LE means bytes [0x74,0x6D,0x6C,0x68] - REVERSE
//                    of TagEntry.classCode's forward ASCII layout.
//   +0x04  4 bytes  secondary id / padding
//   +0x08  4 bytes  name offset
//   +0x0C  4 bytes  tag identity (low 16 bits = in-cache index)
//
// Mirrors HaloReachTagHelpers.h::FindTagRefDatum which reads the
// fourcc as a u32 and compares against the runtime Fourcc() value.
static uint32_t ScanTagRefByClass(CacheHandle* cache, const uint8_t* meta,
                                  size_t maxBytes, const char* groupAscii)
{
    if (maxBytes < 16) return 0xFFFFFFFFu;
    // Build the BE-fourcc u32 the way HaloReachTagHelpers does:
    //   'h','l','m','t' -> 0x686C6D74
    const uint32_t targetU32 =
        ((uint32_t)(uint8_t)groupAscii[0] << 24) |
        ((uint32_t)(uint8_t)groupAscii[1] << 16) |
        ((uint32_t)(uint8_t)groupAscii[2] <<  8) |
        ((uint32_t)(uint8_t)groupAscii[3]      );

    const size_t scanEnd = maxBytes - 16;
    for (size_t off = 0; off <= scanEnd; off += 4) {
        if (RU32(meta + off) != targetU32) continue;
        uint32_t raw = RU32(meta + off + 12);
        if (raw == 0xFFFFFFFFu) continue;
        uint32_t id = raw & 0xFFFFu;
        if (id >= cache->tags.size()) continue;
        // TagEntry.classCode is the forward ASCII form so we can still
        // memcmp against `groupAscii` here as a sanity check.
        if (memcmp(cache->tags[id].classCode, groupAscii, 4) != 0) continue;
        return id;
    }
    return 0xFFFFFFFFu;
}

// Resolve an object tag (any class derived from ObjectTagBase - bloc, scen,
// vehi, weap, eqip, ctrl, mach, ssce, crea, proj, ...) to its render_model
// (mode) tag id by walking the chain:
//   object -> hlmt tag-ref -> hlmt -> mode tag-ref -> mode
//
// The hlmt offset varies between object types (scen=+0x64, others differ),
// so we scan the first 0x400 bytes of the tag's meta for the hlmt fourcc.
// Inside hlmt, the mode tag-ref is the first reference (typically +0x00..
// +0x10), but we scan a small window for safety.
//
// Returns the mode tag id (0..tags.size()-1) on success, or 0xFFFFFFFFu on
// any failure (no hlmt found, no mode in hlmt, OOB meta offset).
// ZH_TAG_ReadMeta: copy up to `len` raw bytes of a tag's meta block starting at `offset` into `out`.
// Returns the byte count copied (0 on any failure). Generic accessor for small fixed-offset fields
// (e.g. the obje lightmap shadow mode) without a dedicated walker.
extern "C" __declspec(dllexport) int32_t __stdcall ZH_TAG_ReadMeta(
    uint64_t cacheHandle, uint32_t tagId, uint32_t offset, uint32_t len, uint8_t* out)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache || !out || len == 0 || len > 65536) return 0;
    if (tagId >= cache->tags.size()) return 0;
    const TagEntry& te = cache->tags[tagId];
    if (te.classIndex < 0) return 0;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return 0;
    if ((size_t)metaOff + (size_t)offset + (size_t)len > cache->size) return 0;
    memcpy(out, cache->base + metaOff + offset, len);
    return (int32_t)len;
}

// ZH_TAG_ReadPtr: copy `len` bytes addressed by a RAW tag-block pointer (as stored in a TagBlockRef)
// into `out`. Returns bytes copied (0 on failure). Companion of ZH_TAG_ReadMeta for walking blocks.
extern "C" __declspec(dllexport) int32_t __stdcall ZH_TAG_ReadPtr(
    uint64_t cacheHandle, uint32_t rawPtr, uint32_t len, uint8_t* out)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache || !out || len == 0 || len > 1048576) return 0;
    int64_t off = TagMetaFileOff(cache, rawPtr);
    if (off < 0 || (size_t)off + (size_t)len > cache->size) return 0;
    memcpy(out, cache->base + off, len);
    return (int32_t)len;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_ResolveModeTagId(
    uint64_t cacheHandle, uint32_t primaryTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    if (primaryTagId >= cache->tags.size()) return 0xFFFFFFFFu;

    const TagEntry& objTe = cache->tags[primaryTagId];
    if (objTe.classIndex < 0) return 0xFFFFFFFFu;

    int64_t objMetaOff = TagMetaFileOff(cache, objTe.metaPointerRaw);
    if (objMetaOff < 0) return 0xFFFFFFFFu;

    // ObjectTagBase headers are typically <0x200 bytes. Scan 0x400 to
    // cover any schema variation (campaign objects sometimes carry
    // larger inline tag-ref blocks).
    constexpr size_t kObjScanBytes  = 0x400;
    constexpr size_t kHlmtScanBytes = 0x100;

    size_t objAvail = (size_t)cache->size - (size_t)objMetaOff;
    if (objAvail < 16) return 0xFFFFFFFFu;
    size_t objScan = objAvail < kObjScanBytes ? objAvail : kObjScanBytes;

    uint32_t hlmtId = ScanTagRefByClass(
        cache, cache->base + objMetaOff, objScan, "hlmt");
    if (hlmtId == 0xFFFFFFFFu) return 0xFFFFFFFFu;

    int64_t hlmtMetaOff = TagMetaFileOff(cache, cache->tags[hlmtId].metaPointerRaw);
    if (hlmtMetaOff < 0) return 0xFFFFFFFFu;
    size_t hlmtAvail = (size_t)cache->size - (size_t)hlmtMetaOff;
    if (hlmtAvail < 16) return 0xFFFFFFFFu;
    size_t hlmtScan = hlmtAvail < kHlmtScanBytes ? hlmtAvail : kHlmtScanBytes;

    return ScanTagRefByClass(
        cache, cache->base + hlmtMetaOff, hlmtScan, "mode");
}

// Resolve an effect_scenery (efsc) tag to a representative BITMAP (bitm) tag id, for
// rendering a billboard stand-in of the effect. Walks efsc -> effe -> prt3 ->
// bitm heuristically (ScanTagRefByClass finds the first ref of each class anywhere in
// the tag's meta, so exact per-tag offsets aren't needed - same approach as
// ZH_TAG_ResolveModeTagId). Falls back to a direct efsc->prt3 / efsc->bitm ref if the
// full chain isn't present. Returns 0xFFFFFFFF if no bitmap can be found.
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_ResolveEfscBitmap(
    uint64_t cacheHandle, uint32_t efscTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    if (efscTagId >= cache->tags.size()) return 0xFFFFFFFFu;

    auto metaOf = [&](uint32_t id, size_t cap, size_t& scanOut) -> const uint8_t* {
        if (id >= cache->tags.size()) return nullptr;
        int64_t off = TagMetaFileOff(cache, cache->tags[id].metaPointerRaw);
        if (off < 0) return nullptr;
        size_t avail = (size_t)cache->size - (size_t)off;
        if (avail < 16) return nullptr;
        scanOut = avail < cap ? avail : cap;
        return cache->base + off;
    };

    // Wide scan windows: a Reach tag's tagblock data is laid out contiguously after its
    // header, so scanning a large region reaches refs nested in Events/ParticleSystems/
    // Textures blocks (the same reason ZH_TAG_ResolveModeTagId's scan finds mode inside
    // hlmt). The exact effe/prt3 offsets are external-plugin-only, so scan is the path.
    size_t scan = 0;
    const uint8_t* efsc = metaOf(efscTagId, 0x2000, scan);
    if (!efsc) return 0xFFFFFFFFu;

    // efsc -> effe -> prt3 -> bitm (scan each stage's meta for the next class).
    uint32_t effeId = ScanTagRefByClass(cache, efsc, scan, "effe");
    uint32_t prt3Id = 0xFFFFFFFFu;
    if (effeId != 0xFFFFFFFFu) {
        size_t s2 = 0;
        const uint8_t* effe = metaOf(effeId, 0x8000, s2);
        if (effe) prt3Id = ScanTagRefByClass(cache, effe, s2, "prt3");
    }
    if (prt3Id == 0xFFFFFFFFu) {
        prt3Id = ScanTagRefByClass(cache, efsc, scan, "prt3"); // direct efsc->prt3
    }
    if (prt3Id != 0xFFFFFFFFu) {
        size_t s3 = 0;
        const uint8_t* prt3 = metaOf(prt3Id, 0x2000, s3);
        if (prt3) {
            // Reliable tail: prt3+0x78 Postprocess tagblock -> entry+0x10 Textures
            // tagblock -> [0]+0x00 tag_reference -> tagId@+0x0C (mirrors DecalWalker's
            // decs/prt3 rmt2 sub-layout; these offsets ARE known and cache-stable).
            if (s3 >= 0x78 + 12) {
                TagBlockRef pp = ReadTagBlock(prt3 + 0x78);
                if (pp.count > 0 && pp.count < 256) {
                    int64_t ppOff = TagMetaFileOff(cache, pp.pointer);
                    if (ppOff >= 0 && (size_t)ppOff + 0x10 + 12 <= (size_t)cache->size) {
                        const uint8_t* ppE = cache->base + ppOff;
                        TagBlockRef tex = ReadTagBlock(ppE + 0x10);
                        if (tex.count > 0 && tex.count < 256) {
                            int64_t texOff = TagMetaFileOff(cache, tex.pointer);
                            if (texOff >= 0 && (size_t)texOff + 16 <= (size_t)cache->size) {
                                uint32_t raw = RU32(cache->base + texOff + 0x0C);
                                uint32_t id = raw & 0xFFFFu;
                                if (raw != 0xFFFFFFFFu && id < cache->tags.size()
                                    && memcmp(cache->tags[id].classCode, "bitm", 4) == 0) {
                                    return id;
                                }
                            }
                        }
                    }
                }
            }
            // Fallback: scan the prt3 meta for any bitm ref.
            uint32_t bitm = ScanTagRefByClass(cache, prt3, s3, "bitm");
            if (bitm != 0xFFFFFFFFu) return bitm;
        }
    }
    // Last resort: a bitmap referenced directly by the efsc.
    return ScanTagRefByClass(cache, efsc, scan, "bitm");
}

// -----------------------------------------------------------------------------
// DIAGNOSTIC: recursively dump a tag's outgoing tag-reference graph via
// NativeDiag (gated on MMS_NATIVE_LOG=1). Given a tag id, scans its meta for any
// 16-byte tag-ref (BE-fourcc@+0x00 that matches the referenced tag's classCode,
// valid tag-id@+0x0C) and logs offset -> ref tag id -> ref class + name. Recurses
// to `depth` levels so we can see eqip -> effe -> prt3 -> bitm without knowing the
// exact per-tag offsets. Used to reverse the armor-ability icon chain.
static void DumpTagRefGraph(CacheHandle* cache, uint32_t tagId, int depth,
                            int maxDepth, size_t scanCap)
{
    if (tagId >= cache->tags.size()) return;
    const TagEntry& te = cache->tags[tagId];
    if (te.classIndex < 0) return;
    int64_t off = TagMetaFileOff(cache, te.metaPointerRaw);
    if (off < 0) return;
    size_t avail = (size_t)cache->size - (size_t)off;
    if (avail < 16) return;
    size_t scan = avail < scanCap ? avail : scanCap;
    const uint8_t* meta = cache->base + off;

    char indent[16] = {0};
    for (int i = 0; i < depth && i < 15; ++i) indent[i] = ' ';
    NativeDiag("HOLOWALK%s[%d] tag=0x%X class=%.4s name=%s", indent, depth,
               tagId, te.classCode, te.tagName.c_str());
    if (depth >= maxDepth) return;

    const size_t scanEnd = scan - 16;
    uint32_t lastId = 0xFFFFFFFFu;
    for (size_t o = 0; o <= scanEnd; o += 4) {
        uint32_t fourccBE = RU32(meta + o);
        uint32_t raw = RU32(meta + o + 12);
        if (raw == 0xFFFFFFFFu) continue;
        uint32_t id = raw & 0xFFFFu;
        if (id >= cache->tags.size()) continue;
        const TagEntry& rt = cache->tags[id];
        if (rt.classIndex < 0) continue;
        uint32_t expectBE =
            ((uint32_t)(uint8_t)rt.classCode[0] << 24) |
            ((uint32_t)(uint8_t)rt.classCode[1] << 16) |
            ((uint32_t)(uint8_t)rt.classCode[2] <<  8) |
            ((uint32_t)(uint8_t)rt.classCode[3]      );
        if (fourccBE != expectBE) continue;
        if (id == tagId) continue;         // self-ref noise
        if (id == lastId) continue;        // adjacent dup
        lastId = id;
        NativeDiag("HOLOWALK%s  +0x%zX -> tag=0x%X class=%.4s name=%s", indent, o,
                   id, rt.classCode, rt.tagName.c_str());
        DumpTagRefGraph(cache, id, depth + 1, maxDepth, scanCap);
    }
}

// #267 armor-ability floating ICON. Each armor ability's holographic icon is the bitmap
// `objects\equipment\equipment_pack\bitmaps\unsc_equipment_drop_holo_icon_<token>`. At runtime
// an effe/prt3 draws it, but in the offline MCC corpus every armor-ability eqip references only
// the SHARED objects\equipment\equipment_pack hlmt (-> mode -> generic bitmaps + shaders\halogram)
// - the per-ability effe/prt3 is stripped, so the icon is UNREACHABLE from the eqip's tag graph.
// We therefore resolve by NAME: derive the ability token from the eqip's tag path and match the
// icon bitmap whose suffix token is a prefix of it (normalized: strip '_', lowercase).
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_ResolveArmorIconBitmap(
    uint64_t cacheHandle, uint32_t eqipTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    if (eqipTagId >= cache->tags.size()) return 0xFFFFFFFFu;
    const TagEntry& te = cache->tags[eqipTagId];
    if (te.classIndex < 0) return 0xFFFFFFFFu;
    {
        char v[8] = {0};
        DWORD r = GetEnvironmentVariableA("HMS_HOLOWALK", v, (DWORD)sizeof(v));
        if (r > 0 && v[0] == '1') {
            // One-time global name search for candidate ability-icon bitmaps.
            static bool once = false;
            if (!once) {
                once = true;
                const char* needles[] = { "icon", "holo_icon", "drop_holo", "hologram",
                    "sprint", "jet_pack", "jetpack", "camo", "armor_lock", "lockup",
                    "drop_shield", "evade", "equipment" };
                for (uint32_t i = 0; i < cache->tags.size(); ++i) {
                    const TagEntry& t = cache->tags[i];
                    if (t.classIndex < 0) continue;
                    const std::string& n = t.tagName;
                    if (n.empty()) continue;
                    for (const char* nd : needles) {
                        if (n.find(nd) != std::string::npos) {
                            NativeDiag("HOLONAME tag=0x%X class=%.4s name=%s",
                                       i, t.classCode, n.c_str());
                            break;
                        }
                    }
                }
            }
            // Shallow (depth<=2) graph dump for THIS eqip only.
            DumpTagRefGraph(cache, eqipTagId, 0, 2, 0x4000);
        }
    }

    // The armor-ability floating icon is the bitmap
    //   objects\equipment\equipment_pack\bitmaps\unsc_equipment_drop_holo_icon_<token>
    // (activecamo / armorlock / dropshield / evade / hologram / jetpack / sprint).
    //
    // In the offline MCC corpus these icons are NOT reachable from the eqip's
    // tag-reference graph: every armor-ability eqip references only the SHARED
    // objects\equipment\equipment_pack hlmt (-> mode -> generic equipment
    // bitmaps + shaders\halogram); the per-ability effe/prt3 that draws the icon
    // at runtime is stripped (see #267 investigation). So we resolve by NAME.
    //
    // Derive the ability token from the eqip's own tag path leaf and match the
    // icon bitmap whose suffix token is a prefix of it after normalization
    // (strip '_', lowercase):
    //   objects\...\jet_pack\jet_pack           -> "jetpack"          == icon "jetpack"
    //   objects\...\active_camouflage\...        -> "activecamouflage" starts-with "activecamo"
    //   objects\...\armor_lockup\armor_lockup    -> "armorlockup"      starts-with "armorlock"
    //   objects\...\drop_shield\drop_shield      -> "dropshield"       == icon "dropshield"
    //   hologram / sprint / evade                -> identity
    auto normalize = [](const std::string& s, size_t start) {
        std::string o;
        for (size_t i = start; i < s.size(); ++i) {
            char c = s[i];
            if (c == '_') continue;
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
            o.push_back(c);
        }
        return o;
    };

    const std::string& en = te.tagName;
    size_t slash = en.find_last_of("\\/");
    std::string ability = normalize(en, slash == std::string::npos ? 0 : slash + 1);
    if (ability.empty()) return 0xFFFFFFFFu;

    static const char kPrefix[] = "unsc_equipment_drop_holo_icon_";
    const size_t kPrefixLen = sizeof(kPrefix) - 1;

    // Pick the LONGEST matching icon token so partial tokens can't win over a
    // more specific one (defensive; the seven ability tokens don't overlap).
    uint32_t best = 0xFFFFFFFFu;
    size_t bestLen = 0;
    for (uint32_t i = 0; i < cache->tags.size(); ++i) {
        const TagEntry& t = cache->tags[i];
        if (t.classIndex < 0) continue;
        if (memcmp(t.classCode, "bitm", 4) != 0) continue;
        size_t p = t.tagName.find(kPrefix);
        if (p == std::string::npos) continue;
        std::string token = normalize(t.tagName, p + kPrefixLen);
        if (token.empty()) continue;
        // ability path starts with the icon token (activecamouflage ~ activecamo).
        if (ability.size() >= token.size() &&
            ability.compare(0, token.size(), token) == 0) {
            if (token.size() > bestLen) { bestLen = token.size(); best = i; }
        }
    }
    return best;
}

// Enumerate every tag in the cache whose 4-cc class code matches `groupBE`
// (big-endian fourcc, e.g. 'bloc' = 'b','l','o','c' on disk). Writes the
// matching short tag ids (0..tags.size()-1) into outIds[0..min(maxCount, hits)].
// Returns the TOTAL number of matches in the cache, even if it exceeds
// maxCount, so the caller can detect a too-small buffer and grow.
//
// Class codes in TagEntry are stored in DISK ORDER (4 ASCII chars, NUL-
// terminated to 5 bytes). The caller passes the same disk-order encoding - 
// e.g. for "bloc" the caller passes the bytes 'b','l','o','c' packed as
// the low-to-high bytes of a u32. We do a memcmp so we don't have to argue
// with endianness on either side.
// The 4-char class code of a tag as a packed u32 (bytes in disk order), 0 if unmapped.
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_GetClass(uint64_t cacheHandle, uint32_t tagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;
    if (tagId >= cache->tags.size()) return 0;
    const TagEntry& te = cache->tags[tagId];
    if (te.classIndex < 0) return 0;
    uint32_t v = 0;
    memcpy(&v, te.classCode, 4);
    return v;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_EnumerateByGroup(
    uint64_t cacheHandle, uint32_t groupBytes, uint32_t* outIds, uint32_t maxCount)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;
    // The on-disk class code is stored as 4 ASCII chars followed by a
    // terminator. Compare against the low 4 bytes of `groupBytes` packed
    // as the caller wrote them (little-endian on x64 means bytes 0..3 of
    // the u32 are the same order as the disk).
    const char* groupChars = reinterpret_cast<const char*>(&groupBytes);
    uint32_t hits = 0;
    uint32_t written = 0;
    const uint32_t total = (uint32_t)cache->tags.size();
    for (uint32_t i = 0; i < total; ++i) {
        const TagEntry& te = cache->tags[i];
        if (te.classIndex < 0) continue;          // unmapped slot
        if (memcmp(te.classCode, groupChars, 4) != 0) continue;
        ++hits;
        if (outIds && written < maxCount) {
            outIds[written++] = i;
        }
    }
    return hits;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_MBP_FindBitmapTag(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex,
    ZH_BitmapInfo* outInfo)
{
    if (!outInfo) return false;
    memset(outInfo, 0, sizeof(*outInfo));

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;

    const BitmapTagParsed* parsed = SehGetParsedBitmap(cache, tagId);
    if (!parsed) return false;
    if (submapIndex >= parsed->submaps.size()) return false;

    const BitmapSubmap& s = parsed->submaps[submapIndex];
    outInfo->Width       = (uint32_t)(uint16_t)s.width;
    outInfo->Height      = (uint32_t)(uint16_t)s.height;
    outInfo->Format      = (uint32_t)(uint16_t)s.bitmapFormat;
    outInfo->MipCount    = s.mipCount;
    outInfo->SubmapCount = (uint32_t)parsed->submaps.size();
    outInfo->Flags       = s.flags;
    outInfo->Depth       = (uint32_t)s.depth;   // volume z-slice count (1 for 2D)
    outInfo->BitmapType  = (uint32_t)(uint8_t)s.bitmapType; // 2 == cubemap (6 faces)
    outInfo->Curve       = (uint32_t)(uint8_t)s.curve;      // byte 17: gamma curve enum
    return true;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_MBP_DecodeBitmap(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex, uint32_t mipLevel,
    uint8_t** outRgba, uint32_t* outW, uint32_t* outH)
{
    if (!outRgba || !outW || !outH) return false;
    *outRgba = nullptr;
    *outW = 0;
    *outH = 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;

    bool ok = SehDecodeBitmap(cache, tagId, submapIndex, mipLevel,
                              outRgba, outW, outH, /*keepAlpha=*/false);
    if (!ok) {
        if (*outRgba) { free(*outRgba); *outRgba = nullptr; }
        *outW = 0;
        *outH = 0;
    }
    return ok;
}

// Decode WITHOUT the alpha=255 stamp. For blend masks (rmtr terrain layer
// weights) the alpha channel is the m3 layer's per-pixel weight; the regular
// path corrupts that. Same call shape as ZH_MBP_DecodeBitmap.
extern "C" __declspec(dllexport) bool __stdcall ZH_MBP_DecodeBitmapKeepAlpha(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex, uint32_t mipLevel,
    uint8_t** outRgba, uint32_t* outW, uint32_t* outH)
{
    if (!outRgba || !outW || !outH) return false;
    *outRgba = nullptr;
    *outW = 0;
    *outH = 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;

    bool ok = SehDecodeBitmap(cache, tagId, submapIndex, mipLevel,
                              outRgba, outW, outH, /*keepAlpha=*/true);
    if (!ok) {
        if (*outRgba) { free(*outRgba); *outRgba = nullptr; }
        *outW = 0;
        *outH = 0;
    }
    return ok;
}

// Decode a single z-slice of a volume submap. See header. sliceIndex 0 ==
// ZH_MBP_DecodeBitmap; sliceIndex >= depth (or a non-volume bitmap with
// sliceIndex > 0) returns false so the caller can fall back.
extern "C" __declspec(dllexport) bool __stdcall ZH_MBP_DecodeBitmapSlice(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex, uint32_t mipLevel,
    uint32_t sliceIndex,
    uint8_t** outRgba, uint32_t* outW, uint32_t* outH)
{
    if (!outRgba || !outW || !outH) return false;
    *outRgba = nullptr;
    *outW = 0;
    *outH = 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;

    bool ok = SehDecodeBitmapSlice(cache, tagId, submapIndex, mipLevel,
                                   sliceIndex, outRgba, outW, outH);
    if (!ok) {
        if (*outRgba) { free(*outRgba); *outRgba = nullptr; }
        *outW = 0;
        *outH = 0;
    }
    return ok;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_MBP_DecodeBitmapFace(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex, uint32_t mipLevel,
    uint32_t faceIndex, uint32_t strideMode,
    uint8_t** outRgba, uint32_t* outW, uint32_t* outH)
{
    if (!outRgba || !outW || !outH) return false;
    *outRgba = nullptr; *outW = 0; *outH = 0;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;
    bool ok = SehDecodeBitmapFace(cache, tagId, submapIndex, mipLevel,
                                  faceIndex, strideMode, outRgba, outW, outH);
    if (!ok) {
        if (*outRgba) { free(*outRgba); *outRgba = nullptr; }
        *outW = 0; *outH = 0;
    }
    return ok;
}

extern "C" __declspec(dllexport) void __stdcall ZH_MBP_FreeBuffer(uint8_t* buf)
{
    if (buf) free(buf);
}

// =============================================================================
// Raw DDS export - bypasses CPU BCn decode, returns compressed mip data as-is.
// =============================================================================

static uint32_t HRFormatToDxgiSrgb(int fmt) {
    switch (fmt) {
        case HR_DXT1:          return 72;  // BC1_UNORM_SRGB
        case HR_DXT3:          return 75;  // BC2_UNORM_SRGB
        case HR_DXT5:          return 78;  // BC3_UNORM_SRGB
        // Reach DXN normal maps are SIGNED. The compiled terrain prepass (ps_17856...) and
        // HREK sample_bumpmap use the sampled .xy DIRECTLY (no *2-1), i.e. a BC5_SNORM view. Exporting as
        // UNORM and doing *2-1 scrambles near-flat texels to +-1 (|xy|~1.3, z->0): the grainy DARK rocks.
        case HR_DXN:           return 84;  // BC5_SNORM (signed tangent-space normal maps)
        // HR_DXN_mono_alpha (44) is the SAME BC5 block layout (4x4x16). The
        // per-instance lightmap DM atlas (intensity in .x, sun-visibility in .y) ships in it.
        // Without this case the raw-DDS export returned 0 -> every GPU-lightmap fetch of that DM
        // failed silently -> those meshes were lit by a FLAT WHITE baked term (forge rock terrain
        // and any opaque instance on the same atlas). BC5_UNORM gives the shader the exact .x/.y
        // channels DecompressVMF reads.
        case HR_DXN_mono_alpha:return 83;  // BC5_UNORM (2-channel lightmap DM)
        case HR_DXT5a:         return 80;  // BC4_UNORM
        default:               return 0;
    }
}

constexpr size_t kDdsHeaderSize = 148; // magic(4) + DDS_HEADER(124) + DDS_HEADER_DXT10(20)

static void BuildDdsHeaderDXT10(uint8_t* dst, int width, int height,
                                int mipCount, uint32_t dxgiFormat,
                                size_t mip0LinearSize)
{
    memset(dst, 0, kDdsHeaderSize);
    *(uint32_t*)(dst + 0) = 0x20534444;
    *(uint32_t*)(dst + 4) = 124;
    uint32_t flags = 0x0000100Fu | 0x00080000u;
    if (mipCount > 1) flags |= 0x00020000u;
    *(uint32_t*)(dst + 8) = flags;
    *(uint32_t*)(dst + 12) = (uint32_t)height;
    *(uint32_t*)(dst + 16) = (uint32_t)width;
    *(uint32_t*)(dst + 20) = (uint32_t)mip0LinearSize;
    *(uint32_t*)(dst + 28) = (uint32_t)(mipCount > 0 ? mipCount : 1);
    *(uint32_t*)(dst + 76) = 32;
    *(uint32_t*)(dst + 80) = 0x4;
    *(uint32_t*)(dst + 84) = 0x30315844; // "DX10"
    uint32_t caps = 0x1000u;
    if (mipCount > 1) caps |= 0x8u | 0x400000u;
    *(uint32_t*)(dst + 108) = caps;
    *(uint32_t*)(dst + 128) = dxgiFormat;
    *(uint32_t*)(dst + 132) = 3; // TEXTURE2D
    *(uint32_t*)(dst + 140) = 1; // arraySize
}

static int GetRawDDSInner(CacheHandle* cache, uint32_t tagId,
                          uint32_t submapIndex,
                          uint8_t** outDds, uint32_t* outDdsLen)
{
    const BitmapTagParsed* parsed = GetParsedBitmap(cache, tagId);
    if (!parsed) return 0;
    if (submapIndex >= parsed->submaps.size()) return 0;
    const BitmapSubmap& s = parsed->submaps[submapIndex];

    uint32_t dxgiFormat = HRFormatToDxgiSrgb(s.bitmapFormat);
    if (dxgiFormat == 0) return 0;

    if (submapIndex >= parsed->resourceIds.size()) return 0;
    int32_t resourceId = parsed->resourceIds[submapIndex];
    if (!parsed->interleavedIds.empty() &&
        s.interleavedIndex < parsed->interleavedIds.size())
    {
        resourceId = parsed->interleavedIds[s.interleavedIndex];
    }

    size_t resSize = 0;
    constexpr size_t kMaxRead = 64 * 1024 * 1024;
    // The secondary (high-res) page holds the TOP mip(s) only; the lower mips live in
    // the primary page. The old single-page read gave 1024^2 textures a ONE-level chain (no
    // minification -> aliasing shimmer on tiled terrain layers). Read both and concatenate when
    // the byte counts tile into one contiguous chain (top mips exactly fill the secondary page).
    uint8_t* res = nullptr;
    {
        size_t hiSize = 0, loSize = 0;
        size_t needD = ChainBytes(s.bitmapFormat, s.width, s.height, s.mipCount > 0 ? (int)s.mipCount : 1); // #tight-read
        size_t capD = (needD > 0 && needD < kMaxRead) ? needD : kMaxRead;
        uint8_t* hi = ReadResourceDataPage(cache, resourceId, capD, &hiSize, 2);
        uint8_t* lo = ReadResourceDataPage(cache, resourceId, capD, &loSize, 1);
        if (hi && lo) {
            int totalMips = s.mipCount > 0 ? s.mipCount : 1;
            size_t acc = 0; int k = -1;
            int mw = s.width, mh = s.height;
            for (int m = 0; m < totalMips; ++m) {
                acc += MipByteSize(s.bitmapFormat, mw, mh);
                if (acc == hiSize) { k = m + 1; break; }
                if (acc > hiSize) break;
                mw = mw > 1 ? mw / 2 : 1; mh = mh > 1 ? mh / 2 : 1;
            }
            size_t rest = 0;
            if (k > 0) {
                int rw = s.width >> k, rh = s.height >> k; if (rw < 1) rw = 1; if (rh < 1) rh = 1;
                for (int m = k; m < totalMips; ++m) {
                    rest += MipByteSize(s.bitmapFormat, rw, rh);
                    rw = rw > 1 ? rw / 2 : 1; rh = rh > 1 ? rh / 2 : 1;
                }
            }
            if (k > 0 && rest > 0 && loSize >= rest) {
                res = (uint8_t*)malloc(hiSize + rest);
                if (res) {
                    memcpy(res, hi, hiSize);
                    memcpy(res + hiSize, lo, rest);
                    resSize = hiSize + rest;
                }
            }
            if (!res) { res = hi; resSize = hiSize; hi = nullptr; }
            if (hi) free(hi);
            free(lo);
        } else if (hi) { res = hi; resSize = hiSize; if (lo) free(lo); }
        else if (lo) { res = lo; resSize = loSize; }
    }
    if (!res) return 0;

    // Sum total mip data size: mip0 + mip1 + ... mip(n-1).
    int mipCount = s.mipCount > 0 ? s.mipCount : 1;
    size_t mipDataSize = 0;
    for (int m = 0; m < mipCount; ++m) {
        int mw = s.width  >> m; if (mw < 1) mw = 1;
        int mh = s.height >> m; if (mh < 1) mh = 1;
        mipDataSize += MipByteSize(s.bitmapFormat, mw, mh);
    }
    // Clamp mipCount so the DDS header never promises more data than exists.
    if (mipDataSize > resSize) {
        mipDataSize = 0;
        mipCount = 0;
        int mw = s.width, mh = s.height;
        for (int m = 0; ; ++m) {
            size_t ms = MipByteSize(s.bitmapFormat, mw, mh);
            if (mipDataSize + ms > resSize) break;
            mipDataSize += ms;
            mipCount = m + 1;
            if (mw <= 1 && mh <= 1) break;
            mw = mw > 1 ? mw / 2 : 1;
            mh = mh > 1 ? mh / 2 : 1;
        }
        if (mipCount == 0) { free(res); return 0; }
    }

    size_t mip0Size = MipByteSize(s.bitmapFormat, s.width, s.height);

    size_t totalSize = kDdsHeaderSize + mipDataSize;
    uint8_t* buf = (uint8_t*)malloc(totalSize);
    if (!buf) { free(res); return 0; }

    BuildDdsHeaderDXT10(buf, s.width, s.height, mipCount, dxgiFormat, mip0Size);
    memcpy(buf + kDdsHeaderSize, res, mipDataSize);

    free(res);
    *outDds = buf;
    *outDdsLen = (uint32_t)totalSize;
    return 1;
}

extern "C" __declspec(dllexport) int __stdcall ZH_MBP_GetRawDDS(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex,
    uint8_t** outDds, uint32_t* outDdsLen)
{
    if (!outDds || !outDdsLen) return 0;
    *outDds = nullptr;
    *outDdsLen = 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;

    __try {
        return GetRawDDSInner(cache, tagId, submapIndex, outDds, outDdsLen);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        if (*outDds) { free(*outDds); *outDds = nullptr; }
        *outDdsLen = 0;
        return 0;
    }
}

extern "C" __declspec(dllexport) void __stdcall ZH_MBP_FreeRawDDS(uint8_t* buf)
{
    if (buf) free(buf);
}

// LOAD-SPEED: free the inflate-once page cache after a load settles. The big default
// page-cache budget (1536 MB) makes the load fast by inflating each deflate page ONCE (bsp_geometry
// 17.6s -> 3.8s), but those decompressed pages would otherwise stay resident. The app calls this
// from its POST-load trim (NOT the periodic during-load trims) so steady memory returns to the #212
// ~200 MB target. Clears the parent cache AND every opened shared.map child (textures/lightmaps for
// big maps live in shared children). Idempotent; safe to call when idle.
extern "C" __declspec(dllexport) void __stdcall ZH_MBP_ClearPageCache(uint64_t cacheHandle)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return;
    auto clear_one = [](CacheHandle* c) {
        if (!c) return;
        std::lock_guard<std::mutex> pk(c->pageCacheMutex);
        c->pageCache.clear();
        c->pageCacheBytes = 0;
        // The load is over -- keep the cache small from here on (the steady-state readers
        // are a handful of on-demand object/preview decodes, not the bulk load).
        c->pageCacheBudgetCap.store(256ull * 1024ull * 1024ull, std::memory_order_relaxed);
    };
    clear_one(cache);
    // Shared children (lazily opened; entries may be nullptr).
    {
        std::lock_guard<std::mutex> lk(cache->pageRouteMutex);
        for (CacheHandle* child : cache->sharedCaches) clear_one(child);
    }
}

// Bytes currently held by the inflate-once page cache (parent + shared children), for the
// app's HMS_MEMDIAG memory report. Print-only diagnostic; 0 when the handle is unknown.
extern "C" __declspec(dllexport) uint64_t __stdcall ZH_MBP_GetPageCacheBytes(uint64_t cacheHandle)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;
    auto bytes_one = [](CacheHandle* c) -> uint64_t {
        if (!c) return 0;
        std::lock_guard<std::mutex> pk(c->pageCacheMutex);
        return (uint64_t)c->pageCacheBytes;
    };
    uint64_t total = bytes_one(cache);
    {
        std::lock_guard<std::mutex> lk(cache->pageRouteMutex);
        for (CacheHandle* child : cache->sharedCaches) total += bytes_one(child);
    }
    return total;
}

// Drop the RESIDENT pages of the read-only .map file mapping(s) (parent + shared children)
// from this process. The load touches most of the map (hundreds of MB of file-backed RSS on a big
// map); after it settles only a few tag/resource pages are read again, and those simply re-fault
// from the OS file cache on the next access. Linux: MADV_DONTNEED on the MAP_SHARED PROT_READ view
// (never on anonymous memory). Windows: the app's EmptyWorkingSet already trims mapped file pages,
// so this is a no-op there. Idempotent; safe to call while the map is in use.
extern "C" __declspec(dllexport) void __stdcall ZH_MBP_DropMapPages(uint64_t cacheHandle)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return;
    auto drop_one = [](CacheHandle* c) {
        if (!c || !c->base || !c->size) return;
#ifndef _WIN32
        madvise((void*)c->base, c->size, MADV_DONTNEED);
#endif
    };
    drop_one(cache);
    {
        std::lock_guard<std::mutex> lk(cache->pageRouteMutex);
        for (CacheHandle* child : cache->sharedCaches) drop_one(child);
    }
}
