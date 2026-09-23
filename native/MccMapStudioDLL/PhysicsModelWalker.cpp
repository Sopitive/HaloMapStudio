// PhysicsModelWalker.cpp
// =============================================================================
// Native walker for the physics_model ('phmo') tag in Halo MCC HaloReach
// .map files. Mirrors CollisionModelWalker.cpp, but resolves the Havok
// rigid-body physics model instead of the collision_model.
//
// Motivation: the Forge "hidden" / invisible building blocks ship a
// render_model ('mode') that is present (non-empty) but renders invisibly in
// game (null/invisible shader). The collision-overlay path only triggers for
// ZERO-section render models ("nut blockers"), so it never fires for these.
// The user wants to visualize how the PHYSICS is laid out for them, so this
// walker decodes the phmo's rigid-body shapes into a model-space triangle soup
// the viewer wraps in a mesh and renders as a distinct (orange/magenta)
// translucent overlay.
//
// Resolution chain (mirror of ZH_TAG_ResolveCollTagId for "phmo"):
//   object tag -> hlmt tag-ref -> hlmt -> phmo tag-ref -> phmo
//
// Geometry layout (authoritative: Assembly ReachMCC/phmo.xml, the same plugin
// source the coll walker cites). physics_model, baseSize 0x19C:
//
//   Rigid Bodies[]   @ 0x5C  elemSize 0xD0  (align 0x10)
//     Shape Type   enum16 @ 0xA8   (0=Sphere 1=Pill 2=Box 3=Triangle
//                                   4=Polyhedron 5=Multi Sphere 0xE=List ...)
//     Shape Index  int16  @ 0xAA   index into the parallel shape block
//
//   Shape blocks (parallel, indexed by Shape Index):
//     Spheres[]      @ 0x74  elemSize 0xB0   Radius f32 @ 0x40, Translation v3 @ 0xA0
//     Pills[]        @ 0x8C  elemSize 0x70   Radius f32 @ 0x40, Bottom v3 @ 0x50, Top v3 @ 0x60
//     Boxes[]        @ 0x98  elemSize 0xE0   Half Extents v3 @ 0x50,
//                                            Rotation i/j/k v3 @ 0xA0/0xB0/0xC0,
//                                            Translation v3 @ 0xD0
//     Polyhedra[]    @ 0xB0  elemSize 0xB0   Number Of Vertices i32 @ 0x80,
//                                            Four Vectors Size i32 @ 0x78
//       (vertices live in the parallel "Polyhedron Four Vectors" block,
//        consumed sequentially per polyhedron, ceil(N/4) elems each)
//     Polyhedron Four Vectors[] @ 0xBC  elemSize 0x30  (Havok SOA: 4 verts per
//                                            elem: x[4] @ 0x0/0x4/0x8 + 0xC pad,
//                                            actually x-triple @ 0x0, y-triple
//                                            @ 0x10, z-triple @ 0x20). See
//                                            UnpackFourVectors below.
//     Lists[]        @ 0xE0  / List Shapes[] @ 0xEC  (containers -> child shapes)
//
// Triangulation per shape:
//   Box        -> 8 corners (+-halfExtents) transformed by rot+translation,
//                 12 tris.
//   Sphere     -> icosphere-ish low-poly tessellation at Translation, Radius.
//   Pill       -> capsule = cylinder between Bottom/Top + 2 hemispheres,
//                 low-poly.
//   Polyhedron -> convex-hull triangulation of its N vertices (gift-wrap is
//                 overkill for an overlay; we fan from the centroid over a
//                 simple convex-hull-by-AABB-faces approximation is NOT used - 
//                 instead we use the plane equations? No: the most robust is a
//                 3D convex hull. For an overlay we use the incremental hull.)
//   List       -> recurse into child shapes (one level; capped).
//
// Defensive contract (matches CollisionModelWalker.cpp):
//   * Every cross-tag deref is SEH-fenced so a busted chain returns cleanly.
//   * Block counts are sanity-capped before any pointer math.
//   * All buffer offsets are validated against cache->size.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace zh_mcc;

namespace {

// ---- phmo schema offsets (ReachMCC/phmo.xml) --------------------------------
constexpr int OFF_PHMO_RIGID_BODIES   = 0x5C;   // tagblock, elem 0xD0
constexpr int RB_BLOCK_SIZE           = 0xD0;
constexpr int RB_SHAPE_TYPE           = 0xA8;   // enum16
constexpr int RB_SHAPE_INDEX          = 0xAA;   // int16

constexpr int OFF_PHMO_SPHERES        = 0x74;   // tagblock, elem 0xB0
constexpr int SPHERE_BLOCK_SIZE       = 0xB0;
constexpr int SPHERE_RADIUS           = 0x40;   // f32
constexpr int SPHERE_TRANSLATION      = 0xA0;   // v3

constexpr int OFF_PHMO_PILLS          = 0x8C;   // tagblock, elem 0x70
constexpr int PILL_BLOCK_SIZE         = 0x70;
constexpr int PILL_RADIUS             = 0x40;   // f32
constexpr int PILL_BOTTOM             = 0x50;   // v3
constexpr int PILL_TOP                = 0x60;   // v3

constexpr int OFF_PHMO_BOXES          = 0x98;   // tagblock, elem 0xE0
constexpr int BOX_BLOCK_SIZE          = 0xE0;
constexpr int BOX_HALF_EXTENTS        = 0x50;   // v3
constexpr int BOX_ROT_I               = 0xA0;   // v3
constexpr int BOX_ROT_J               = 0xB0;   // v3
constexpr int BOX_ROT_K               = 0xC0;   // v3
constexpr int BOX_TRANSLATION         = 0xD0;   // v3

constexpr int OFF_PHMO_POLYHEDRA      = 0xB0;   // tagblock, elem 0xB0
constexpr int POLY_BLOCK_SIZE         = 0xB0;
constexpr int POLY_FOUR_VECTORS_SIZE  = 0x78;   // i32 (# of four-vector elems)
constexpr int POLY_NUM_VERTICES       = 0x80;   // i32

constexpr int OFF_PHMO_FOUR_VECTORS   = 0xBC;   // tagblock, elem 0x30
constexpr int FOURVEC_BLOCK_SIZE      = 0x30;

constexpr int OFF_PHMO_LISTS          = 0xE0;   // tagblock, elem 0x90
constexpr int LIST_BLOCK_SIZE         = 0x90;
constexpr int LIST_CHILD_SHAPES_COUNT = 0x38;   // i32 Child Shapes Size (count)
constexpr int OFF_PHMO_LIST_SHAPES    = 0xEC;   // tagblock, elem 0x20
constexpr int LISTSHAPE_BLOCK_SIZE    = 0x20;
constexpr int LISTSHAPE_TYPE          = 0x00;   // enum16
constexpr int LISTSHAPE_INDEX         = 0x02;   // int16

// Havok shape type enum values (RB_SHAPE_TYPE).
enum HkShapeType : int {
    HK_SPHERE      = 0x0,
    HK_PILL        = 0x1,
    HK_BOX         = 0x2,
    HK_TRIANGLE    = 0x3,
    HK_POLYHEDRON  = 0x4,
    HK_MULTISPHERE = 0x5,
    HK_PHANTOM     = 0x6,
    HK_LIST        = 0xE,
    HK_MOPP        = 0xF,
};

// Sanity caps - physics models are small relative to render models.
constexpr int32_t MAX_RIGID_BODIES = 1024;
constexpr int32_t MAX_SHAPES       = 4096;   // per shape-type block
constexpr int32_t MAX_FOURVECS     = 200000; // global four-vector pool
constexpr int32_t MAX_LISTSHAPES   = 8192;
constexpr int      MAX_POLY_VERTS  = 4096;   // verts in a single polyhedron
constexpr int      MAX_LIST_DEPTH  = 2;      // recursion guard for List shapes

struct PhmoGeom {
    std::vector<float>    verts;   // x,y,z triples (model space, world units)
    std::vector<uint32_t> indices; // triangle list
    // shape-type histogram for diagnostics.
    int nBox = 0, nSphere = 0, nPill = 0, nPoly = 0, nList = 0, nOther = 0;
    int nUnhandled = 0;
};

// Cached, validated shape-block pointers for one phmo (resolved once).
struct ShapeBlocks {
    const uint8_t* sphereBase = nullptr; int32_t sphereCount = 0;
    const uint8_t* pillBase   = nullptr; int32_t pillCount   = 0;
    const uint8_t* boxBase    = nullptr; int32_t boxCount    = 0;
    const uint8_t* polyBase   = nullptr; int32_t polyCount   = 0;
    const uint8_t* fourVecBase = nullptr; int32_t fourVecCount = 0;
    const uint8_t* listBase   = nullptr; int32_t listCount   = 0;
    const uint8_t* listShapeBase = nullptr; int32_t listShapeCount = 0;
};

inline float RF(const uint8_t* p) { float v; memcpy(&v, p, 4); return v; }

// Translate a (count + raw pointer) tagblock to a validated base pointer.
const uint8_t* ResolveBlock(CacheHandle* cache, const TagBlockRef& blk,
                            int elemSize, int32_t maxCount)
{
    if (blk.count <= 0 || blk.count > maxCount) return nullptr;
    int64_t off = TagMetaFileOff(cache, blk.pointer);
    if (off < 0) return nullptr;
    if ((size_t)off + (size_t)blk.count * (size_t)elemSize > cache->size) return nullptr;
    return cache->base + off;
}

inline void PushVert(PhmoGeom& out, float x, float y, float z)
{
    out.verts.push_back(x); out.verts.push_back(y); out.verts.push_back(z);
}
inline void PushTri(PhmoGeom& out, uint32_t a, uint32_t b, uint32_t c)
{
    out.indices.push_back(a); out.indices.push_back(b); out.indices.push_back(c);
}

// ---- Box: 8 corners transformed by the 3x3 rotation + translation -----------
void EmitBox(const uint8_t* box, PhmoGeom& out)
{
    float hx = RF(box + BOX_HALF_EXTENTS + 0);
    float hy = RF(box + BOX_HALF_EXTENTS + 4);
    float hz = RF(box + BOX_HALF_EXTENTS + 8);
    if (!isfinite(hx) || !isfinite(hy) || !isfinite(hz)) return;
    // Clamp absurd extents (corrupt data guard).
    const float kMax = 1.0e5f;
    if (fabsf(hx) > kMax || fabsf(hy) > kMax || fabsf(hz) > kMax) return;

    float r[3][3] = {
        { RF(box + BOX_ROT_I + 0), RF(box + BOX_ROT_J + 0), RF(box + BOX_ROT_K + 0) },
        { RF(box + BOX_ROT_I + 4), RF(box + BOX_ROT_J + 4), RF(box + BOX_ROT_K + 4) },
        { RF(box + BOX_ROT_I + 8), RF(box + BOX_ROT_J + 8), RF(box + BOX_ROT_K + 8) },
    };
    float tx = RF(box + BOX_TRANSLATION + 0);
    float ty = RF(box + BOX_TRANSLATION + 4);
    float tz = RF(box + BOX_TRANSLATION + 8);

    // If the rotation columns are all ~zero (untransformed / identity-less
    // box), fall back to identity so the box is at least axis-aligned.
    float rnorm = 0;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) rnorm += fabsf(r[i][j]);
    if (rnorm < 1e-6f) { r[0][0] = r[1][1] = r[2][2] = 1.0f; }

    const float sx[8] = { -1,-1,-1,-1, 1, 1, 1, 1 };
    const float sy[8] = { -1,-1, 1, 1,-1,-1, 1, 1 };
    const float sz[8] = { -1, 1,-1, 1,-1, 1,-1, 1 };

    uint32_t base = (uint32_t)(out.verts.size() / 3);
    for (int i = 0; i < 8; i++)
    {
        float lx = sx[i] * hx, ly = sy[i] * hy, lz = sz[i] * hz;
        float wx = r[0][0]*lx + r[0][1]*ly + r[0][2]*lz + tx;
        float wy = r[1][0]*lx + r[1][1]*ly + r[1][2]*lz + ty;
        float wz = r[2][0]*lx + r[2][1]*ly + r[2][2]*lz + tz;
        PushVert(out, wx, wy, wz);
    }
    // 12 triangles (corner indexing: bit0=z, bit1=y, bit2=x).
    static const int faces[12][3] = {
        {0,2,3},{0,3,1},  // x=-
        {4,5,7},{4,7,6},  // x=+
        {0,1,5},{0,5,4},  // y=-
        {2,6,7},{2,7,3},  // y=+
        {0,4,6},{0,6,2},  // z=-
        {1,3,7},{1,7,5},  // z=+
    };
    for (auto& f : faces) PushTri(out, base + f[0], base + f[1], base + f[2]);
}

// ---- Sphere: low-poly UV sphere ---------------------------------------------
void EmitSphereAt(float cx, float cy, float cz, float radius, PhmoGeom& out)
{
    if (!isfinite(radius) || radius <= 0 || radius > 1.0e5f) return;
    const int kStacks = 6, kSlices = 8;
    uint32_t base = (uint32_t)(out.verts.size() / 3);
    for (int i = 0; i <= kStacks; i++)
    {
        float v = (float)i / kStacks;        // 0..1
        float phi = v * 3.14159265f;          // 0..pi
        float sp = sinf(phi), cp = cosf(phi);
        for (int j = 0; j <= kSlices; j++)
        {
            float u = (float)j / kSlices;     // 0..1
            float th = u * 6.2831853f;        // 0..2pi
            float x = cx + radius * sp * cosf(th);
            float y = cy + radius * sp * sinf(th);
            float z = cz + radius * cp;
            PushVert(out, x, y, z);
        }
    }
    int row = kSlices + 1;
    for (int i = 0; i < kStacks; i++)
    {
        for (int j = 0; j < kSlices; j++)
        {
            uint32_t a = base + i * row + j;
            uint32_t b = base + (i + 1) * row + j;
            uint32_t c = base + (i + 1) * row + (j + 1);
            uint32_t d = base + i * row + (j + 1);
            PushTri(out, a, b, c);
            PushTri(out, a, c, d);
        }
    }
}

void EmitSphere(const uint8_t* sph, PhmoGeom& out)
{
    float r  = RF(sph + SPHERE_RADIUS);
    float cx = RF(sph + SPHERE_TRANSLATION + 0);
    float cy = RF(sph + SPHERE_TRANSLATION + 4);
    float cz = RF(sph + SPHERE_TRANSLATION + 8);
    EmitSphereAt(cx, cy, cz, r, out);
}

// ---- Pill (capsule): cylinder between Bottom/Top + endpoint spheres ---------
void EmitPill(const uint8_t* pill, PhmoGeom& out)
{
    float r  = RF(pill + PILL_RADIUS);
    float bx = RF(pill + PILL_BOTTOM + 0), by = RF(pill + PILL_BOTTOM + 4), bz = RF(pill + PILL_BOTTOM + 8);
    float tx = RF(pill + PILL_TOP + 0),    ty = RF(pill + PILL_TOP + 4),    tz = RF(pill + PILL_TOP + 8);
    if (!isfinite(r) || r <= 0 || r > 1.0e5f) return;

    // Axis frame.
    float ax = tx - bx, ay = ty - by, az = tz - bz;
    float alen = sqrtf(ax*ax + ay*ay + az*az);
    if (!isfinite(alen)) return;
    if (alen < 1e-5f) { EmitSphereAt(bx, by, bz, r, out); return; }
    ax /= alen; ay /= alen; az /= alen;
    // Build two perpendicular axes.
    float ux, uy, uz;
    if (fabsf(ax) < 0.9f) { ux = 1; uy = 0; uz = 0; } else { ux = 0; uy = 1; uz = 0; }
    // u = normalize(u - (u.a)a)
    float dot = ux*ax + uy*ay + uz*az;
    ux -= dot*ax; uy -= dot*ay; uz -= dot*az;
    float ulen = sqrtf(ux*ux + uy*uy + uz*uz);
    if (ulen < 1e-5f) return;
    ux /= ulen; uy /= ulen; uz /= ulen;
    // w = a x u
    float wx = ay*uz - az*uy, wy = az*ux - ax*uz, wz = ax*uy - ay*ux;

    const int kSlices = 8;
    uint32_t base = (uint32_t)(out.verts.size() / 3);
    // bottom ring then top ring.
    for (int ring = 0; ring < 2; ring++)
    {
        float ox = ring ? tx : bx, oy = ring ? ty : by, oz = ring ? tz : bz;
        for (int j = 0; j <= kSlices; j++)
        {
            float th = (float)j / kSlices * 6.2831853f;
            float ct = cosf(th), st = sinf(th);
            float px = ox + r * (ct*ux + st*wx);
            float py = oy + r * (ct*uy + st*wy);
            float pz = oz + r * (ct*uz + st*wz);
            PushVert(out, px, py, pz);
        }
    }
    int row = kSlices + 1;
    for (int j = 0; j < kSlices; j++)
    {
        uint32_t a = base + j, b = base + row + j, c = base + row + j + 1, d = base + j + 1;
        PushTri(out, a, b, c);
        PushTri(out, a, c, d);
    }
    // Endpoint hemispheres approximated by full low-poly spheres (overlay).
    EmitSphereAt(bx, by, bz, r, out);
    EmitSphereAt(tx, ty, tz, r, out);
}

// ---- Polyhedron four-vectors unpack -----------------------------------------
// Havok hkpConvexVerticesShape stores vertices as "four vectors": each 0x30
// element packs 4 vertices, SOA-style. ReachMCC/phmo.xml labels the element as
// three vector3s ("Four Vectors x" @0x0, "...y" @0x10, "...z" @0x20). The
// interpretation: x-triple holds X of verts 0,1,2 (and 0xC pad holds vert3.X);
// y-triple holds Y of verts 0,1,2; z-triple holds Z of verts 0,1,2; and the
// 0xC/0x1C/0x2C "w" floats hold the 4th vertex's X/Y/Z. So one element = up to
// 4 verts: (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3) where
//   x0=@0x00 x1=@0x04 x2=@0x08 x3=@0x0C
//   y0=@0x10 y1=@0x14 y2=@0x18 y3=@0x1C
//   z0=@0x20 z1=@0x24 z2=@0x28 z3=@0x2C
// We emit only the first `numVerts` of the (numFourVecs*4) decoded slots.
void UnpackPolyhedronVerts(const uint8_t* fvBase, int32_t fvStartElem,
                           int32_t fvElemCount, int32_t numVerts,
                           std::vector<float>& vx, std::vector<float>& vy,
                           std::vector<float>& vz)
{
    int emitted = 0;
    for (int e = 0; e < fvElemCount && emitted < numVerts; e++)
    {
        const uint8_t* fv = fvBase + (size_t)(fvStartElem + e) * FOURVEC_BLOCK_SIZE;
        for (int k = 0; k < 4 && emitted < numVerts; k++)
        {
            float x = RF(fv + 0x00 + k * 4);
            float y = RF(fv + 0x10 + k * 4);
            float z = RF(fv + 0x20 + k * 4);
            if (!isfinite(x) || !isfinite(y) || !isfinite(z)) { emitted++; continue; }
            vx.push_back(x); vy.push_back(y); vz.push_back(z);
            emitted++;
        }
    }
}

// 3D incremental convex hull (small N; an overlay tolerates approximate hulls).
// For robustness and simplicity we use a brute-force hull over all triangles
// of candidate faces: for every triple of distinct verts, if all other verts
// lie on one side of the plane, the triple is a hull face. O(N^4) but N is
// tiny (forge convex blocks have <=64 verts). Capped hard.
void HullTriangulate(const std::vector<float>& vx, const std::vector<float>& vy,
                     const std::vector<float>& vz, PhmoGeom& out)
{
    int n = (int)vx.size();
    if (n < 4) {
        // Degenerate: emit a tiny point cloud as a fan if we have a triangle.
        if (n == 3) {
            uint32_t base = (uint32_t)(out.verts.size() / 3);
            for (int i = 0; i < 3; i++) PushVert(out, vx[i], vy[i], vz[i]);
            PushTri(out, base, base + 1, base + 2);
        }
        return;
    }
    // Brute-force hull cap: skip if N too large to keep O(N^4) bounded.
    if (n > 96) {
        // Fall back to AABB box of the points (still a useful overlay).
        float mnx=vx[0],mny=vy[0],mnz=vz[0],mxx=vx[0],mxy=vy[0],mxz=vz[0];
        for (int i=1;i<n;i++){ if(vx[i]<mnx)mnx=vx[i]; if(vy[i]<mny)mny=vy[i]; if(vz[i]<mnz)mnz=vz[i];
                               if(vx[i]>mxx)mxx=vx[i]; if(vy[i]>mxy)mxy=vy[i]; if(vz[i]>mxz)mxz=vz[i]; }
        uint32_t base = (uint32_t)(out.verts.size()/3);
        const float cx[8]={mnx,mnx,mnx,mnx,mxx,mxx,mxx,mxx};
        const float cy[8]={mny,mny,mxy,mxy,mny,mny,mxy,mxy};
        const float cz[8]={mnz,mxz,mnz,mxz,mnz,mxz,mnz,mxz};
        for(int i=0;i<8;i++) PushVert(out,cx[i],cy[i],cz[i]);
        static const int F[12][3]={{0,2,3},{0,3,1},{4,5,7},{4,7,6},{0,1,5},{0,5,4},
                                   {2,6,7},{2,7,3},{0,4,6},{0,6,2},{1,3,7},{1,7,5}};
        for(auto&f:F) PushTri(out,base+f[0],base+f[1],base+f[2]);
        return;
    }

    uint32_t base = (uint32_t)(out.verts.size() / 3);
    for (int i = 0; i < n; i++) PushVert(out, vx[i], vy[i], vz[i]);

    const float kEps = 1e-4f;
    for (int a = 0; a < n; a++)
    for (int b = a + 1; b < n; b++)
    for (int c = b + 1; c < n; c++)
    {
        // Plane normal.
        float ux = vx[b]-vx[a], uy = vy[b]-vy[a], uz = vz[b]-vz[a];
        float wx = vx[c]-vx[a], wy = vy[c]-vy[a], wz = vz[c]-vz[a];
        float nx = uy*wz - uz*wy, ny = uz*wx - ux*wz, nz = ux*wy - uy*wx;
        float nl = sqrtf(nx*nx+ny*ny+nz*nz);
        if (nl < kEps) continue;       // degenerate triple
        nx/=nl; ny/=nl; nz/=nl;
        float d = nx*vx[a]+ny*vy[a]+nz*vz[a];
        int pos=0, neg=0;
        for (int p = 0; p < n; p++)
        {
            if (p==a||p==b||p==c) continue;
            float s = nx*vx[p]+ny*vy[p]+nz*vz[p] - d;
            if (s >  kEps) pos++;
            if (s < -kEps) neg++;
            if (pos && neg) break;
        }
        if (pos && neg) continue;      // not a hull face
        // Orient so normal points outward (away from interior). If all points
        // are on the negative side, flip winding so the face faces outward.
        if (pos == 0) PushTri(out, base + a, base + b, base + c);
        else          PushTri(out, base + a, base + c, base + b);
    }
}

void EmitPolyhedron(const uint8_t* poly, const ShapeBlocks& sb,
                    int32_t& fourVecCursor, PhmoGeom& out)
{
    int32_t numVerts = R32(poly + POLY_NUM_VERTICES);
    int32_t fvSize   = R32(poly + POLY_FOUR_VECTORS_SIZE);
    if (numVerts <= 0 || numVerts > MAX_POLY_VERTS) {
        // Still advance the cursor by the declared four-vector count so
        // subsequent polyhedra stay aligned.
        if (fvSize > 0 && fvSize <= sb.fourVecCount) fourVecCursor += fvSize;
        return;
    }
    // Number of four-vector elements this polyhedron owns. Prefer the
    // explicit Four Vectors Size; fall back to ceil(numVerts/4).
    int32_t fvElems = (fvSize > 0) ? fvSize : ((numVerts + 3) / 4);
    if (fvElems <= 0) return;
    if (fourVecCursor < 0) fourVecCursor = 0;
    if (fourVecCursor + fvElems > sb.fourVecCount) {
        // Out of range - bail without advancing past the pool.
        return;
    }

    std::vector<float> vx, vy, vz;
    vx.reserve(numVerts); vy.reserve(numVerts); vz.reserve(numVerts);
    UnpackPolyhedronVerts(sb.fourVecBase, fourVecCursor, fvElems, numVerts, vx, vy, vz);
    fourVecCursor += fvElems;

    HullTriangulate(vx, vy, vz, out);
}

void DecodePhmoInner(CacheHandle* cache, uint32_t phmoTagId, PhmoGeom& out)
{
    if (phmoTagId >= cache->tags.size()) return;
    const TagEntry& te = cache->tags[phmoTagId];
    if (te.classIndex < 0) return;
    if (memcmp(te.classCode, "phmo", 4) != 0) return;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return;
    if ((size_t)metaOff + 0x19C > cache->size) return;
    const uint8_t* meta = cache->base + metaOff;

    // Resolve all shape blocks once.
    ShapeBlocks sb;
    {
        TagBlockRef b;
        b = ReadTagBlock(meta + OFF_PHMO_SPHERES);
        sb.sphereBase = ResolveBlock(cache, b, SPHERE_BLOCK_SIZE, MAX_SHAPES); sb.sphereCount = sb.sphereBase ? b.count : 0;
        b = ReadTagBlock(meta + OFF_PHMO_PILLS);
        sb.pillBase = ResolveBlock(cache, b, PILL_BLOCK_SIZE, MAX_SHAPES); sb.pillCount = sb.pillBase ? b.count : 0;
        b = ReadTagBlock(meta + OFF_PHMO_BOXES);
        sb.boxBase = ResolveBlock(cache, b, BOX_BLOCK_SIZE, MAX_SHAPES); sb.boxCount = sb.boxBase ? b.count : 0;
        b = ReadTagBlock(meta + OFF_PHMO_POLYHEDRA);
        sb.polyBase = ResolveBlock(cache, b, POLY_BLOCK_SIZE, MAX_SHAPES); sb.polyCount = sb.polyBase ? b.count : 0;
        b = ReadTagBlock(meta + OFF_PHMO_FOUR_VECTORS);
        sb.fourVecBase = ResolveBlock(cache, b, FOURVEC_BLOCK_SIZE, MAX_FOURVECS); sb.fourVecCount = sb.fourVecBase ? b.count : 0;
        b = ReadTagBlock(meta + OFF_PHMO_LISTS);
        sb.listBase = ResolveBlock(cache, b, LIST_BLOCK_SIZE, MAX_SHAPES); sb.listCount = sb.listBase ? b.count : 0;
        b = ReadTagBlock(meta + OFF_PHMO_LIST_SHAPES);
        sb.listShapeBase = ResolveBlock(cache, b, LISTSHAPE_BLOCK_SIZE, MAX_LISTSHAPES); sb.listShapeCount = sb.listShapeBase ? b.count : 0;
    }

    // ---- Polyhedra: emit ALL polyhedra in block order, consuming the global
    // four-vector pool sequentially. This is independent of which rigid body
    // references them; the pool layout is positional. Every convex hull block
    // therefore renders even if the rigid-body indirection is unusual. ----
    int32_t polyCursor = 0;
    for (int32_t i = 0; i < sb.polyCount; i++)
    {
        const uint8_t* poly = sb.polyBase + (size_t)i * POLY_BLOCK_SIZE;
        EmitPolyhedron(poly, sb, polyCursor, out);
        out.nPoly++;
    }

    // ---- Boxes / Spheres / Pills: emit ALL in block order. These shapes are
    // self-contained (carry their own transform), so emitting the whole block
    // gives the full physics silhouette regardless of rigid-body indirection.
    for (int32_t i = 0; i < sb.boxCount; i++)    { EmitBox   (sb.boxBase    + (size_t)i * BOX_BLOCK_SIZE,    out); out.nBox++; }
    for (int32_t i = 0; i < sb.sphereCount; i++) { EmitSphere(sb.sphereBase + (size_t)i * SPHERE_BLOCK_SIZE, out); out.nSphere++; }
    for (int32_t i = 0; i < sb.pillCount; i++)   { EmitPill  (sb.pillBase   + (size_t)i * PILL_BLOCK_SIZE,   out); out.nPill++; }

    // Lists / MOPPs reference the above primitive blocks by index; since we
    // already emit every primitive once, recursing into lists would
    // double-render. We count them for diagnostics only.
    out.nList += sb.listCount;
}

bool SehDecodePhmo(CacheHandle* cache, uint32_t phmoTagId, PhmoGeom& out)
{
    __try { DecodePhmoInner(cache, phmoTagId, out); return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

// ---- phmo resolution (mirror of ResolveCollInner for "phmo") ----------------
uint32_t ScanTagRefByClassPhmo(CacheHandle* cache, const uint8_t* meta,
                               size_t maxBytes, const char* groupAscii)
{
    if (maxBytes < 16) return 0xFFFFFFFFu;
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
        if (memcmp(cache->tags[id].classCode, groupAscii, 4) != 0) continue;
        return id;
    }
    return 0xFFFFFFFFu;
}

uint32_t ResolvePhmoInner(CacheHandle* cache, uint32_t primaryTagId)
{
    if (primaryTagId >= cache->tags.size()) return 0xFFFFFFFFu;
    const TagEntry& objTe = cache->tags[primaryTagId];
    if (objTe.classIndex < 0) return 0xFFFFFFFFu;

    int64_t objMetaOff = TagMetaFileOff(cache, objTe.metaPointerRaw);
    if (objMetaOff < 0) return 0xFFFFFFFFu;

    constexpr size_t kObjScanBytes  = 0x400;
    constexpr size_t kHlmtScanBytes = 0x100;

    size_t objAvail = (size_t)cache->size - (size_t)objMetaOff;
    if (objAvail < 16) return 0xFFFFFFFFu;
    size_t objScan = objAvail < kObjScanBytes ? objAvail : kObjScanBytes;

    uint32_t hlmtId = ScanTagRefByClassPhmo(cache, cache->base + objMetaOff, objScan, "hlmt");
    if (hlmtId == 0xFFFFFFFFu) return 0xFFFFFFFFu;

    int64_t hlmtMetaOff = TagMetaFileOff(cache, cache->tags[hlmtId].metaPointerRaw);
    if (hlmtMetaOff < 0) return 0xFFFFFFFFu;
    size_t hlmtAvail = (size_t)cache->size - (size_t)hlmtMetaOff;
    if (hlmtAvail < 16) return 0xFFFFFFFFu;
    size_t hlmtScan = hlmtAvail < kHlmtScanBytes ? hlmtAvail : kHlmtScanBytes;

    return ScanTagRefByClassPhmo(cache, cache->base + hlmtMetaOff, hlmtScan, "phmo");
}

uint32_t SehResolvePhmo(CacheHandle* cache, uint32_t primaryTagId)
{
    __try { return ResolvePhmoInner(cache, primaryTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

} // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

// Resolve an object tag (bloc/scen/vehi/...) to its physics_model ('phmo')
// tag id by walking object -> hlmt -> phmo. Returns the phmo tag id
// (0..tags.size()-1) on success, or 0xFFFFFFFFu if the object has no phmo,
// the chain breaks, or the handle is bad.
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_ResolvePhmoTagId(
    uint64_t cacheHandle, uint32_t primaryTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    return SehResolvePhmo(cache, primaryTagId);
}

// Decode a physics_model's rigid-body shapes into a triangle soup.
//
//   outVerts      -> malloc'd float[3*outVertCount] (x,y,z per vertex, model space)
//   outVertCount  -> number of vertices
//   outIndices    -> malloc'd uint32[outIndexCount] (triangle list)
//   outIndexCount -> number of indices (multiple of 3)
//
// Returns true on success (counts may be 0 for an empty phmo); false on bad
// handle / bad tag / decode fault. Caller frees both buffers via
// ZH_PHMO_FreeBuffer. On failure all out params are zeroed.
extern "C" __declspec(dllexport) bool __stdcall ZH_PHMO_DecodeGeometry(
    uint64_t cacheHandle, uint32_t phmoTagId,
    float**  outVerts,  uint32_t* outVertCount,
    uint32_t** outIndices, uint32_t* outIndexCount)
{
    if (outVerts)      *outVerts = nullptr;
    if (outVertCount)  *outVertCount = 0;
    if (outIndices)    *outIndices = nullptr;
    if (outIndexCount) *outIndexCount = 0;
    if (!outVerts || !outVertCount || !outIndices || !outIndexCount) return false;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;

    PhmoGeom geom;
    if (!SehDecodePhmo(cache, phmoTagId, geom)) return false;

    uint32_t vCount = (uint32_t)(geom.verts.size() / 3);
    uint32_t iCount = (uint32_t)geom.indices.size();

    if (vCount > 0) {
        float* vb = (float*)malloc((size_t)vCount * 3 * sizeof(float));
        if (!vb) return false;
        memcpy(vb, geom.verts.data(), (size_t)vCount * 3 * sizeof(float));
        *outVerts = vb;
        *outVertCount = vCount;
    }
    if (iCount > 0) {
        uint32_t* ib = (uint32_t*)malloc((size_t)iCount * sizeof(uint32_t));
        if (!ib) { free(*outVerts); *outVerts = nullptr; *outVertCount = 0; return false; }
        memcpy(ib, geom.indices.data(), (size_t)iCount * sizeof(uint32_t));
        *outIndices = ib;
        *outIndexCount = iCount;
    }

    NativeDiag("Phmo[%u]: verts=%u tris=%u shapes(box=%d sph=%d pill=%d poly=%d list=%d other=%d)",
               phmoTagId, vCount, iCount / 3,
               geom.nBox, geom.nSphere, geom.nPill, geom.nPoly, geom.nList, geom.nOther);
    return true;
}

// Free a buffer returned by ZH_PHMO_DecodeGeometry (verts OR indices). Safe on
// nullptr.
extern "C" __declspec(dllexport) void __stdcall ZH_PHMO_FreeBuffer(void* buf)
{
    if (buf) free(buf);
}
