// MapBitmapParser.h
// =============================================================================
// Self-contained C++ bitmap-tag decoder for Halo MCC HaloReach .map files
// (the tag layouts follow Reclaimer's Blam.HaloReach.bitmap definitions).
// The public C exports below are loaded by crates/hms-native; the DLL keeps
// all its other (in-process / injection) exports.
//
// All exports are __stdcall and use POD-only signatures so the FFI layer needs
// no marshaling beyond raw pointers and integers.
//
// Threading model:
//   * One open cache handle per map file. Cache parsing happens on the first
//     ZH_MBP_FindBitmapTag call after open. Subsequent calls re-use the parsed
//     index.
//   * A single SRWLOCK per cache guards the lazy parse and the resource cache.
//     Decode itself is lock-free once the cache is parsed (mmap pointer + tag
//     index are immutable thereafter).
//   * Multiple cache handles can coexist (one per map). They never share state.
// =============================================================================

#pragma once
#include <cstdint>

extern "C" {

// Submap descriptor returned via FindBitmapTag.
struct ZH_BitmapInfo {
    uint32_t Width;
    uint32_t Height;
    uint32_t Format;        // engine-format enum (DXT1=14, DXT5=16, A8R8G8B8=11, ...)
    uint32_t MipCount;
    uint32_t SubmapCount;   // bitmap tags can carry multiple submaps; usually 1
    uint32_t Flags;
    uint32_t Depth;         // volume-texture depth (z-slice count); 1 for 2D bitmaps.
                            // (was Reserved[0]) - used by the dual-VMF lightprobe path
                            // to detect the 3-slice lightprobe_hdr_color volume so the
                            // fill-lobe color (slice 2) can be decoded independently.
    uint32_t BitmapType;    // (was Reserved[1]) submap bitmapType byte: 0=2D texture,
                            // 1=3D/volume, 2=CUBEMAP (6 faces), 3=sprite/array. A cube's
                            // Depth is 1 while it physically stores 6 faces - the authored
                            // environment_map cube path keys off BitmapType==2, not Depth.
    uint32_t Curve;         // submap byte 17: bitmap gamma curve. Reach enum:
                            // 0=unknown/xRGB, 1=linear(raw), 2=sRGB, 3=gamma2(pow2.0),
                            // 4=offset_log. The engine samples through the texture format
                            // that this curve selects (sRGB -> auto sRGB -> linear decode;
                            // linear/offset_log -> raw). Terrain detail maps are usually
                            // authored linear (utility), base maps sRGB - so a blanket
                            // pow(2.2) in the shader over-darkens linear detail maps.
};

// Open a .map file. Returns a non-zero handle on success, 0 on failure.
__declspec(dllexport) uint64_t __stdcall ZH_MBP_OpenCache(const wchar_t* path);

// Close the cache. Frees the mmap and tag index.
__declspec(dllexport) void __stdcall ZH_MBP_CloseCache(uint64_t cacheHandle);

// Look up a bitmap tag by id. Returns true on success.
__declspec(dllexport) bool __stdcall ZH_MBP_FindBitmapTag(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex,
    ZH_BitmapInfo* outInfo);

// Decode one mip of one submap to RGBA8 (BGRA byte layout - matches the viewer
// PixelFormats.Bgra32). On success, *outRgba points to a malloc'd buffer
// (caller must call ZH_MBP_FreeBuffer to release).
__declspec(dllexport) bool __stdcall ZH_MBP_DecodeBitmap(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex, uint32_t mipLevel,
    uint8_t** outRgba, uint32_t* outW, uint32_t* outH);

// Decode one z-slice of a volume (Texture3D) submap to RGBA8 (BGRA layout).
// For a depth-N volume bitmap the engine stores N full 2D images contiguously at
// mip 0 (slice-major); this decodes slice `sliceIndex` (0..depth-1) of `mipLevel`.
// sliceIndex 0 is byte-identical to ZH_MBP_DecodeBitmap. Used by the dual-VMF
// lightprobe path to pull slice 2 (fill-lobe color) out of the 3-slice
// lightprobe_hdr_color volume. On success *outRgba is a malloc'd BGRA8 buffer
// (free via ZH_MBP_FreeBuffer). Returns false (and leaves outputs zeroed) when
// the slice is out of range or the format/resource can't be decoded.
__declspec(dllexport) bool __stdcall ZH_MBP_DecodeBitmapSlice(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex, uint32_t mipLevel,
    uint32_t sliceIndex,
    uint8_t** outRgba, uint32_t* outW, uint32_t* outH);

// Decode one FACE (0..5) of a cubemap (bitmapType==2) submap to RGBA8 (BGRA
// layout). A cube's Depth field is 1 even though it stores 6 faces, so the
// volume-slice decoder refuses faces 1..5; this keys off a 6-face count. Alpha
// is preserved (BC3 cube env maps carry an HDR scale in alpha). strideMode
// selects the byte layout: 0 = face-major (each face a full mip chain), 1 =
// mip-major (all 6 faces per mip - the CORRECT layout for Reach, empirically
// confirmed; matches the slice-major addressing used for volume textures).
// faceIndex 0 is byte-identical to ZH_MBP_DecodeBitmap. On success *outRgba is a
// malloc'd BGRA8 buffer (free via ZH_MBP_FreeBuffer). Returns false (outputs
// zeroed) for a non-cube, an out-of-range face, or an undecodable format.
__declspec(dllexport) bool __stdcall ZH_MBP_DecodeBitmapFace(
    uint64_t cacheHandle, uint32_t tagId, uint32_t submapIndex, uint32_t mipLevel,
    uint32_t faceIndex, uint32_t strideMode,
    uint8_t** outRgba, uint32_t* outW, uint32_t* outH);

// Free a buffer returned by DecodeBitmap.
__declspec(dllexport) void __stdcall ZH_MBP_FreeBuffer(uint8_t* buf);

}  // extern "C"
