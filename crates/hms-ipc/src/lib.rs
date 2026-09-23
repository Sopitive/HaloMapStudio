//! hms-ipc — memory-mapped-file client for the injected DLL's shared buffers.
//!
//! Layouts are byte-identical to the C++ side (`WorldSpawnSnapshot.cpp`,
//! `ForgeSpawnAndPose.cpp`, etc.) so the same named mappings interoperate.
//! The producer is HaloMapStudioDLL injected into MCC, so the mappings only
//! open on Windows (hms-app `injection` feature); elsewhere every open fails
//! with a clear error and the pure-Rust layout code still compiles.

use anyhow::{bail, Result};
#[cfg(windows)]
use anyhow::Context;
use bytemuck::{Pod, Zeroable};

#[cfg(windows)]
use windows::core::PCWSTR;
#[cfg(windows)]
use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
#[cfg(windows)]
use windows::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS,
    MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
};

#[cfg(windows)]
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A named page-file-backed shared memory region, opened-or-created.
#[cfg(windows)]
pub struct Mmf {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    base: *mut u8,
    size: usize,
}

/// Non-Windows stand-in. These mappings are created by `HaloMapStudioDLL.dll`
/// injected into MCC, which only exists on Windows, so the constructors always
/// fail here and every `SomeClient::open().ok()` call site degrades to `None`
/// (i.e. the offline map viewer/editor, with the live-game panels inert).
/// The buffer keeps `read`/`write` well-defined if one is ever constructed.
#[cfg(not(windows))]
pub struct Mmf {
    _buf: Vec<u8>,
    base: *mut u8,
    size: usize,
}

// The mapping is process-shared; the pointer is only used single-threaded from
// the UI/tick thread, but mark Send so it can live in app state.
unsafe impl Send for Mmf {}

#[cfg(windows)]
impl Mmf {
    pub fn create_or_open(name: &str, size: usize) -> Result<Mmf> {
        let wname = wide(name);
        unsafe {
            let handle = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                None,
                PAGE_READWRITE,
                0,
                size as u32,
                PCWSTR(wname.as_ptr()),
            )
            .context("CreateFileMappingW failed")?;
            let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
            if view.Value.is_null() {
                let _ = CloseHandle(handle);
                bail!("MapViewOfFile failed for {name}");
            }
            Ok(Mmf {
                handle,
                view,
                base: view.Value as *mut u8,
                size,
            })
        }
    }

    /// Open an EXISTING named mapping (read/write view) — fails if the producer
    /// (the injected DLL) hasn't created it yet. Use this for consumer clients so
    /// we never pre-create a mapping the DLL expects to own.
    pub fn open_existing(name: &str, size: usize) -> Result<Mmf> {
        let wname = wide(name);
        unsafe {
            let handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS.0, false, PCWSTR(wname.as_ptr()))
                .context("OpenFileMappingW failed (producer not present yet)")?;
            let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
            if view.Value.is_null() {
                let _ = CloseHandle(handle);
                bail!("MapViewOfFile failed for {name}");
            }
            Ok(Mmf {
                handle,
                view,
                base: view.Value as *mut u8,
                size,
            })
        }
    }

}

#[cfg(not(windows))]
impl Mmf {
    pub fn create_or_open(name: &str, _size: usize) -> Result<Mmf> {
        bail!("shared memory {name} is unavailable: the producer is HaloMapStudioDLL injected into MCC (Windows only)")
    }

    pub fn open_existing(name: &str, _size: usize) -> Result<Mmf> {
        bail!("shared memory {name} is unavailable: the producer is HaloMapStudioDLL injected into MCC (Windows only)")
    }
}

impl Mmf {
    #[inline]
    pub fn read<T: Pod>(&self, off: usize) -> T {
        assert!(off + std::mem::size_of::<T>() <= self.size);
        unsafe { std::ptr::read_unaligned(self.base.add(off) as *const T) }
    }

    #[inline]
    pub fn write<T: Pod>(&self, off: usize, val: T) {
        assert!(off + std::mem::size_of::<T>() <= self.size);
        unsafe { std::ptr::write_unaligned(self.base.add(off) as *mut T, val) }
    }

    /// Copy `bytes` into the mapping at `off` (clamped to the view size).
    #[inline]
    pub fn write_bytes(&self, off: usize, bytes: &[u8]) {
        if off >= self.size {
            return;
        }
        let n = bytes.len().min(self.size - off);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(off), n) }
    }

    /// Copy `len` bytes at `off` out of the mapping (clamped to the view size).
    #[inline]
    pub fn read_bytes(&self, off: usize, len: usize) -> Vec<u8> {
        if off >= self.size {
            return Vec::new();
        }
        let n = len.min(self.size - off);
        unsafe { std::slice::from_raw_parts(self.base.add(off), n).to_vec() }
    }
}

#[cfg(windows)]
impl Drop for Mmf {
    fn drop(&mut self) {
        unsafe {
            let _ = UnmapViewOfFile(self.view);
            if !self.handle.is_invalid() {
                let _ = CloseHandle(self.handle);
            }
        }
    }
}

// =============================================================================
// MapInfo — ZeroHour_MapInfo_Snapshot (MapInfoSnapshot.cpp). The injected DLL
// publishes the running scenario datum (+ best-effort name/path). Lets the
// viewer auto-select which detected .map is currently loaded in the game.
// =============================================================================
pub mod map_info {
    use super::*;

    pub const MAP_NAME: &str = "ZeroHour_MapInfo_Snapshot";
    pub const MAGIC: u32 = 0x504D_415A; // 'ZMAP'
    const TOTAL: usize = 1024;
    const HEADER: usize = 32;
    const NAME_OFF: usize = HEADER;
    const PATH_OFF: usize = HEADER + 256;

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct Header {
        magic: u32,
        version: u32,
        write_counter: u32,
        scenario_datum: u32,
        last_tick_ms: u32,
        last_error: u32,
        scenario_name_len: u32,
        map_path_len: u32,
    }

    /// One read of the map-info snapshot. `valid` is false when the DLL hasn't
    /// published yet (not injected / wrong magic).
    #[derive(Clone, Debug, Default)]
    pub struct MapInfo {
        pub valid: bool,
        pub scenario_datum: u32,
        pub scenario_name: String,
        pub map_path: String,
    }

    pub struct MapInfoClient {
        mmf: Mmf,
    }

    impl MapInfoClient {
        pub fn open() -> Result<Self> {
            Ok(Self { mmf: Mmf::open_existing(MAP_NAME, TOTAL)? })
        }

        pub fn read(&self) -> MapInfo {
            let h: Header = self.mmf.read(0);
            if h.magic != MAGIC || h.version < 1 {
                return MapInfo::default();
            }
            let nlen = (h.scenario_name_len.min(255)) as usize;
            let plen = (h.map_path_len.min(259)) as usize;
            let name = self.mmf.read_bytes(NAME_OFF, nlen);
            let path = self.mmf.read_bytes(PATH_OFF, plen);
            let trim0 = |b: &[u8]| -> String {
                let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
                String::from_utf8_lossy(&b[..end]).into_owned()
            };
            MapInfo {
                valid: true,
                scenario_datum: h.scenario_datum,
                scenario_name: trim0(&name),
                map_path: trim0(&path),
            }
        }
    }
}

// =============================================================================
// Pose snapshot — ZeroHour_Pose_Snapshot. The DLL publishes per-object root
// matrices; we read the first entry's world translation to frame the camera on
// the player. Header(32) + Entry(SchemaSize each), root xlate at +20/+36/+52.
// =============================================================================
pub mod pose_snapshot {
    use super::*;

    pub const MAP_NAME: &str = "ZeroHour_Pose_Snapshot";
    const TOTAL: usize = 1 << 20; // 1 MiB view is plenty
    const HEADER: usize = 32;

    pub struct PoseSnapshotClient {
        mmf: Mmf,
    }
    impl PoseSnapshotClient {
        pub fn open() -> Result<Self> {
            Ok(Self { mmf: Mmf::open_existing(MAP_NAME, TOTAL)? })
        }
        /// First object's (datum, world position). None if empty/unpublished.
        pub fn first(&self) -> Option<(u32, [f32; 3])> {
            let count: u32 = self.mmf.read(12);
            let schema: u32 = self.mmf.read(16);
            if count == 0 || schema < 56 || schema as usize > TOTAL {
                return None;
            }
            let datum: u32 = self.mmf.read(HEADER);
            if datum == 0 {
                return None;
            }
            // Root matrix translation column: R03@+20, R13@+36, R23@+52.
            let x: f32 = self.mmf.read(HEADER + 20);
            let y: f32 = self.mmf.read(HEADER + 36);
            let z: f32 = self.mmf.read(HEADER + 52);
            Some((datum, [x, y, z]))
        }
    }
}

// =============================================================================
// WorldSpawn V2 ring — ZeroHour_WorldSpawn_Shared (WorldSpawnSnapshot.cpp).
// Header(16) + Ring[32] * Entry(24) = 784 bytes. The host writes ring[writeIdx%32]
// and bumps writeIdx; the DLL drains readIdx..writeIdx-1.
// =============================================================================
pub mod world_spawn {
    use super::*;

    pub const MAP_NAME: &str = "ZeroHour_WorldSpawn_Shared";
    pub const MAGIC: u32 = 0x5357_485A; // 'ZHWS'
    pub const VERSION: u32 = 2;
    pub const RING: usize = 32;
    const HEADER: usize = 16;
    const ENTRY: usize = 24;
    pub const TOTAL: usize = HEADER + RING * ENTRY;

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct Header {
        magic: u32,
        version: u32,
        write_idx: u32,
        read_idx: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct Entry {
        pallet: u32,
        object: u32,
        variant: u32,
        x: f32,
        y: f32,
        z: f32,
    }

    pub struct WorldSpawnClient {
        mmf: Mmf,
    }

    impl WorldSpawnClient {
        pub fn open() -> Result<Self> {
            let mmf = Mmf::create_or_open(MAP_NAME, TOTAL)?;
            let h: Header = mmf.read(0);
            if h.magic != MAGIC || h.version != VERSION {
                mmf.write(
                    0,
                    Header {
                        magic: MAGIC,
                        version: VERSION,
                        write_idx: 0,
                        read_idx: 0,
                    },
                );
            }
            Ok(Self { mmf })
        }

        /// Queue a palette spawn at a world point. Returns the new write index
        /// (nonzero) on success, or 0 if the ring is full.
        pub fn queue_spawn_at(
            &self,
            pallet: u32,
            object: u32,
            variant: u32,
            x: f32,
            y: f32,
            z: f32,
        ) -> u32 {
            let mut h: Header = self.mmf.read(0);
            let pending = h.write_idx.wrapping_sub(h.read_idx);
            if pending >= RING as u32 {
                return 0;
            }
            let slot = (h.write_idx as usize) % RING;
            let off = HEADER + slot * ENTRY;
            self.mmf.write(
                off,
                Entry {
                    pallet,
                    object,
                    variant,
                    x,
                    y,
                    z,
                },
            );
            h.write_idx = h.write_idx.wrapping_add(1);
            self.mmf.write(0, h);
            h.write_idx
        }
    }
}

// =============================================================================
// ForgeSpawnAndPose — HaloMapStudio_ForgeSpawnAndPose_Shared (80 bytes).
// Single-shot trigger: viewer writes the request + SpawnMethod and bumps
// TriggerCounter; the DLL handles it, writes the response, bumps ResponseCounter.
// SpawnMethod: 0 = default (palette / local), 1 = live SpawnObjectFromDefinition
// (PaletteIndex carries the tag id).
// =============================================================================
pub mod forge_spawn_pose {
    use super::*;

    pub const MAP_NAME: &str = "HaloMapStudio_ForgeSpawnAndPose_Shared";
    pub const MAGIC: u32 = 0x4653_4D4D; // 'MMSF'
    pub const VERSION: u32 = 1;
    pub const TOTAL: usize = std::mem::size_of::<Shared>();

    pub const RAW_TAG_SENTINEL: u32 = 0xFFFF_FFFF;
    pub const METHOD_DEFAULT: u32 = 0;
    pub const METHOD_LIVE: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct Shared {
        magic: u32,
        version: u32,
        trigger: u32,
        response: u32,

        palette: u32,
        entry: u32,
        variant: u32,
        method: u32,

        x: f32,
        y: f32,
        z: f32,
        fx: f32,
        fy: f32,
        fz: f32,
        ux: f32,
        uy: f32,
        uz: f32,

        result_status: u32,
        spawned_datum: u32,
        _pad1: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Status {
        Idle,
        Success,
        SpawnFail,
        PoseFail,
        NotReady,
        NotImpl,
        Other(u32),
    }
    impl From<u32> for Status {
        fn from(v: u32) -> Self {
            match v {
                0 => Status::Idle,
                1 => Status::Success,
                2 => Status::SpawnFail,
                3 => Status::PoseFail,
                4 => Status::NotReady,
                99 => Status::NotImpl,
                o => Status::Other(o),
            }
        }
    }

    pub struct ForgeSpawnClient {
        mmf: Mmf,
    }

    impl ForgeSpawnClient {
        pub fn open() -> Result<Self> {
            let mmf = Mmf::create_or_open(MAP_NAME, TOTAL)?;
            let s: Shared = mmf.read(0);
            if s.magic != MAGIC || s.version != VERSION {
                let mut init = Shared::default();
                init.magic = MAGIC;
                init.version = VERSION;
                mmf.write(0, init);
            }
            Ok(Self { mmf })
        }

        /// Queue a spawn-with-pose. `method` = METHOD_DEFAULT or METHOD_LIVE
        /// (for live: pass the tag id in `palette` and `entry = RAW_TAG_SENTINEL`).
        /// Returns the trigger counter the DLL will mirror back.
        #[allow(clippy::too_many_arguments)]
        pub fn queue_spawn_and_pose(
            &self,
            palette: u32,
            entry: u32,
            variant: u32,
            pos: [f32; 3],
            fwd: [f32; 3],
            up: [f32; 3],
            method: u32,
        ) -> u32 {
            let mut s: Shared = self.mmf.read(0);
            s.palette = palette;
            s.entry = entry;
            s.variant = variant;
            s.method = method;
            s.x = pos[0];
            s.y = pos[1];
            s.z = pos[2];
            s.fx = fwd[0];
            s.fy = fwd[1];
            s.fz = fwd[2];
            s.ux = up[0];
            s.uy = up[1];
            s.uz = up[2];
            s.trigger = s.trigger.wrapping_add(1);
            self.mmf.write(0, s);
            s.trigger
        }

        /// Snapshot the response body: (response_counter, status, spawned_datum).
        pub fn read_response(&self) -> (u32, Status, u32) {
            let s: Shared = self.mmf.read(0);
            (s.response, Status::from(s.result_status), s.spawned_datum)
        }
    }
}

// =============================================================================
// Object table snapshot — ZeroHour_ObjectTable_Snapshot (ObjectTableSnapshot.cpp)
// v5: Header(32) + Object(92) * 1024. The DLL bumps WriteCounter AFTER the body
// is published; we torn-read fence on it.
// =============================================================================
pub mod object_table {
    use super::*;

    pub const MAP_NAME: &str = "ZeroHour_ObjectTable_Snapshot";
    pub const MAGIC: u32 = 0x544A_424F; // 'OBJT'
    pub const VERSION_MIN: u32 = 5;
    pub const MAX_OBJECTS: usize = 1024;
    const HEADER: usize = 32;
    const OBJ: usize = 92;
    pub const TOTAL: usize = HEADER + OBJ * MAX_OBJECTS;

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct Header {
        magic: u32,
        version: u32,
        write_counter: u32,
        object_count: u32,
        schema_size: u32,
        last_tick_ms: u32,
        last_error: u32,
        reserved: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct RawObject {
        datum: u32,
        type_sig: u16,
        sig0: u8,
        sig1: u8,
        pos: [f32; 3],
        health: f32,
        shield: f32,
        mode_tag: u32,
        fwd: [f32; 3],
        up: [f32; 3],
        attached: [u32; 8],
        primary_tag: u32,
    }

    /// Object-table entry (the fields the viewer needs to place + identify).
    #[derive(Clone, Copy, Debug, PartialEq)] // #outline-ui: PartialEq so an edit snapshot can detect "nothing changed"
    pub struct ObjectInfo {
        pub datum: u32,
        pub type_sig: u16,
        pub sig0: u8,
        pub sig1: u8,
        pub pos: [f32; 3],
        pub health: f32,
        pub shield: f32,
        pub mode_tag: u32,
        pub fwd: [f32; 3],
        pub up: [f32; 3],
        pub attached: [u32; 8],
        pub primary_tag: u32,
        /// hlmt model-variant Name stringId for this placement (e.g. "rocket" for a rocket
        /// warthog), threaded from the forge palette. 0 = default variant. NOT part of the shared
        /// MMF `RawObject` — this is Rust-side only; the live-object `read()` path sets it to 0.
        pub variant_name_sid: u32,
    }

    pub struct ObjectTableClient {
        mmf: Mmf,
    }

    impl ObjectTableClient {
        pub fn open() -> Result<Self> {
            Ok(Self {
                mmf: Mmf::open_existing(MAP_NAME, TOTAL)?,
            })
        }

        /// Read a consistent snapshot of live objects (empty if not published or
        /// version too old). Torn-read fenced on WriteCounter.
        pub fn read(&self) -> Vec<ObjectInfo> {
            for _ in 0..2 {
                let h: Header = self.mmf.read(0);
                if h.magic != MAGIC || h.version < VERSION_MIN {
                    return Vec::new();
                }
                let count = (h.object_count as usize).min(MAX_OBJECTS);
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let r: RawObject = self.mmf.read(HEADER + i * OBJ);
                    out.push(ObjectInfo {
                        datum: r.datum,
                        type_sig: r.type_sig,
                        sig0: r.sig0,
                        sig1: r.sig1,
                        pos: r.pos,
                        health: r.health,
                        shield: r.shield,
                        mode_tag: r.mode_tag,
                        fwd: r.fwd,
                        up: r.up,
                        attached: r.attached,
                        primary_tag: r.primary_tag,
                        variant_name_sid: 0, // live objects: variant not tracked in the MMF
                    });
                }
                // Re-check the counter; if unchanged the read was consistent.
                let h2: Header = self.mmf.read(0);
                if h2.write_counter == h.write_counter {
                    return out;
                }
            }
            Vec::new()
        }
    }

    // Layout guards (compile-time).
    const _: () = assert!(std::mem::size_of::<Header>() == HEADER);
    const _: () = assert!(std::mem::size_of::<RawObject>() == OBJ);
}

// =============================================================================
// Forge palette snapshot — ZeroHour_ForgePalette_Snapshot (ForgePaletteSnapshot
// .cpp). v4: Header(32 + 32*64 palette-names = 2080) + Entry(192) * 512.
// =============================================================================
pub mod forge_palette {
    use super::*;

    pub const MAP_NAME: &str = "ZeroHour_ForgePalette_Snapshot";
    pub const MAGIC: u32 = 0x5047_5246; // 'FRGP'
    pub const VERSION: u32 = 4;
    pub const MAX_ENTRIES: usize = 512;
    const MAX_PALETTES: usize = 32;
    const PALETTE_NAME_BYTES: usize = 64;
    const HEADER: usize = 32 + MAX_PALETTES * PALETTE_NAME_BYTES; // 2080
    const ENTRY: usize = 192;
    pub const TOTAL: usize = HEADER + ENTRY * MAX_ENTRIES;

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct Header {
        magic: u32,
        version: u32,
        entry_count: u32,
        write_counter: u32,
        request_counter: u32,
        palette_count: u32,
        _pad1: u32,
        _pad2: u32,
    }

    /// A forge-palette entry (what the master-palette browser lists + spawns).
    #[derive(Clone, Debug)]
    pub struct PaletteEntry {
        pub palette_index: u32,
        /// raw (entry<<8 | variant)
        pub variant_index: u32,
        pub tag_group: u32,
        pub tag_global_id: u32,
        pub hlmt_tag_id: u32,
        pub mode_tag_id: u32,
        pub entry_name: String,
        pub variant_name: String,
    }
    impl PaletteEntry {
        pub fn entry_within(&self) -> u32 {
            (self.variant_index >> 8) & 0x00FF_FFFF
        }
        pub fn variant_within(&self) -> u32 {
            self.variant_index & 0xFF
        }
        pub fn display(&self) -> String {
            if !self.variant_name.is_empty() && self.variant_name != self.entry_name {
                format!("{} / {}", self.entry_name, self.variant_name)
            } else if !self.entry_name.is_empty() {
                self.entry_name.clone()
            } else {
                format!("tag {:#010X}", self.tag_global_id)
            }
        }
    }

    pub struct ForgePaletteClient {
        mmf: Mmf,
    }

    impl ForgePaletteClient {
        pub fn open() -> Result<Self> {
            Ok(Self {
                mmf: Mmf::open_existing(MAP_NAME, TOTAL)?,
            })
        }

        pub fn read(&self) -> Vec<PaletteEntry> {
            for _ in 0..2 {
                let h: Header = self.mmf.read(0);
                if h.magic != MAGIC || h.version != VERSION {
                    return Vec::new();
                }
                let count = (h.entry_count as usize).min(MAX_ENTRIES);
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let b = HEADER + i * ENTRY;
                    out.push(PaletteEntry {
                        palette_index: self.mmf.read::<u32>(b),
                        variant_index: self.mmf.read::<u32>(b + 4),
                        tag_group: self.mmf.read::<u32>(b + 8),
                        tag_global_id: self.mmf.read::<u32>(b + 12),
                        hlmt_tag_id: self.mmf.read::<u32>(b + 16),
                        mode_tag_id: self.mmf.read::<u32>(b + 20),
                        // TagName[40] @ +24, EntryName[64] @ +64, VariantName[64] @ +128
                        entry_name: self.asciiz(b + 64, 64),
                        variant_name: self.asciiz(b + 128, 64),
                    });
                }
                let h2: Header = self.mmf.read(0);
                if h2.write_counter == h.write_counter {
                    return out;
                }
            }
            Vec::new()
        }

        fn asciiz(&self, off: usize, max: usize) -> String {
            let mut s = String::new();
            for i in 0..max {
                let c: u8 = self.mmf.read(off + i);
                if c == 0 {
                    break;
                }
                s.push(c as char);
            }
            s
        }
    }

    const _: () = assert!(std::mem::size_of::<Header>() == 32);
}

// =============================================================================
// Forge object table — ZeroHour_ForgeObjectTable_Snapshot (team/color per datum).
// Header(32) + Entry(88) * 650. We only read datum + Team + Color for tinting.
// =============================================================================
pub mod forge_object_table {
    use super::*;
    use std::collections::HashMap;

    pub const MAP_NAME: &str = "ZeroHour_ForgeObjectTable_Snapshot";
    pub const MAGIC: u32 = 0x4F47_5246; // 'FRGO'
    pub const VERSION_MIN: u32 = 1;
    pub const MAX_OBJECTS: usize = 650;
    const HEADER: usize = 32;
    const ENTRY: usize = 88;
    pub const TOTAL: usize = HEADER + ENTRY * MAX_OBJECTS;
    // Field offsets within an Entry (Pack=1, DLL ForgeObjectTableSnapshot::Entry):
    // EngineDatum@+4, Show@+8, ItemCategory@+10, Pos@+16, Fwd@+28, Up@+40,
    // ItemVariant@+54, Team@+79, Color@+82.
    const OFF_DATUM: usize = 4;
    const OFF_SHOW: usize = 8;
    const OFF_CATEGORY: usize = 10;
    const OFF_POS: usize = 16;
    const OFF_FWD: usize = 28;
    const OFF_UP: usize = 40;
    const OFF_VARIANT: usize = 54;
    const OFF_TEAM: usize = 79;
    const OFF_COLOR: usize = 82;

    /// A live Forge placement read directly from the AOB-scanned forge table.
    /// This is populated even when the engine object POOL reads empty client-side, so
    /// it's the reliable source for rendering placed forge objects from memory. The
    /// render model is resolved via the master palette: type_key = (category<<8)|variant
    /// matches ForgePalette `variant_index` → `mode_tag_id`.
    #[derive(Clone)]
    pub struct ForgePlacement {
        pub forge_idx: u32,
        pub datum: u32,
        pub show: bool,
        pub pos: [f32; 3],
        pub fwd: [f32; 3],
        pub up: [f32; 3],
        pub type_key: u32, // (ItemCategory<<8)|ItemVariant
        pub team: u8,
        pub color: u8,
    }

    pub struct ForgeObjectTableClient {
        mmf: Mmf,
    }
    impl ForgeObjectTableClient {
        pub fn open() -> Result<Self> {
            Ok(Self {
                mmf: Mmf::open_existing(MAP_NAME, TOTAL)?,
            })
        }
        /// Map datum → (team, color) for live forge objects.
        pub fn read(&self) -> HashMap<u32, (u8, u8)> {
            let mut map = HashMap::new();
            let magic: u32 = self.mmf.read(0);
            let version: u32 = self.mmf.read(4);
            if magic != MAGIC || version < VERSION_MIN {
                return map;
            }
            let count = (self.mmf.read::<u32>(12) as usize).min(MAX_OBJECTS);
            for i in 0..count {
                let b = HEADER + i * ENTRY;
                let datum: u32 = self.mmf.read(b + OFF_DATUM);
                if datum == 0 {
                    continue;
                }
                let team: u8 = self.mmf.read(b + OFF_TEAM);
                let color: u8 = self.mmf.read(b + OFF_COLOR);
                map.insert(datum, (team, color));
            }
            map
        }

        /// Full live placements (pos/orientation/type/show) from the forge table.
        /// Used to render placed forge objects from memory when the engine object pool is
        /// empty client-side (the documented CLIENT_OBJ_DIAG condition). Only entries with
        /// show=true and a finite position are returned.
        pub fn read_placements(&self) -> Vec<ForgePlacement> {
            let mut out = Vec::new();
            let magic: u32 = self.mmf.read(0);
            let version: u32 = self.mmf.read(4);
            if magic != MAGIC || version < VERSION_MIN {
                return out;
            }
            let count = (self.mmf.read::<u32>(12) as usize).min(MAX_OBJECTS);
            for i in 0..count {
                let b = HEADER + i * ENTRY;
                let show: u16 = self.mmf.read(b + OFF_SHOW);
                if show & 0xFF == 0 {
                    continue; // hidden / empty slot
                }
                let pos = [
                    self.mmf.read::<f32>(b + OFF_POS),
                    self.mmf.read::<f32>(b + OFF_POS + 4),
                    self.mmf.read::<f32>(b + OFF_POS + 8),
                ];
                if !pos.iter().all(|v| v.is_finite()) {
                    continue;
                }
                let fwd = [
                    self.mmf.read::<f32>(b + OFF_FWD),
                    self.mmf.read::<f32>(b + OFF_FWD + 4),
                    self.mmf.read::<f32>(b + OFF_FWD + 8),
                ];
                let up = [
                    self.mmf.read::<f32>(b + OFF_UP),
                    self.mmf.read::<f32>(b + OFF_UP + 4),
                    self.mmf.read::<f32>(b + OFF_UP + 8),
                ];
                let category: u16 = self.mmf.read(b + OFF_CATEGORY);
                let variant: u8 = self.mmf.read(b + OFF_VARIANT);
                out.push(ForgePlacement {
                    forge_idx: i as u32,
                    datum: self.mmf.read(b + OFF_DATUM),
                    show: true,
                    pos,
                    fwd,
                    up,
                    type_key: ((category as u32) << 8) | variant as u32,
                    team: self.mmf.read(b + OFF_TEAM),
                    color: self.mmf.read(b + OFF_COLOR),
                });
            }
            out
        }

        /// Full rows: (forge_index, datum, team, color). The entry index is the
        /// forge object index used by the ForgeObjectEdit request.
        pub fn read_full(&self) -> Vec<(u32, u32, u8, u8)> {
            let mut out = Vec::new();
            let magic: u32 = self.mmf.read(0);
            let version: u32 = self.mmf.read(4);
            if magic != MAGIC || version < VERSION_MIN {
                return out;
            }
            let count = (self.mmf.read::<u32>(12) as usize).min(MAX_OBJECTS);
            for i in 0..count {
                let b = HEADER + i * ENTRY;
                let datum: u32 = self.mmf.read(b + OFF_DATUM);
                if datum == 0 {
                    continue;
                }
                let team: u8 = self.mmf.read(b + OFF_TEAM);
                let color: u8 = self.mmf.read(b + OFF_COLOR);
                out.push((i as u32, datum, team, color));
            }
            out
        }
    }
}

// =============================================================================
// ForgeObjectEdit request ring — HaloMapStudio_ForgeObjectEdit_Request (v3).
// The viewer WRITES full property edits (team/color/pose/show/…); the DLL drains
// the ring each frame pump. Header(32) + Entry(104) * 512. Producer = us.
// =============================================================================
pub mod forge_object_edit {
    use super::*;

    pub const MAP_NAME: &str = "HaloMapStudio_ForgeObjectEdit_Request";
    pub const MAGIC: u32 = 0x4547_4F46; // 'FOGE'
    pub const VERSION: u32 = 3;
    pub const CAP: usize = 512;
    const HEADER: usize = 32;
    const ENTRY: usize = 104;
    pub const TOTAL: usize = HEADER + CAP * ENTRY;

    // ForgeObjectEditField bits.
    pub const F_POSITION: u32 = 1 << 0;
    pub const F_ROTATION: u32 = 1 << 1;
    pub const F_TEAM: u32 = 1 << 7;
    pub const F_COLOR: u32 = 1 << 8;
    pub const F_SPAWN_SEQ: u32 = 1 << 9;
    pub const F_SHOW: u32 = 1 << 16;
    pub const F_LIVE_SCALE: u32 = 1 << 17;

    fn put_u32(e: &mut [u8], off: usize, v: u32) {
        e[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn put_u16(e: &mut [u8], off: usize, v: u16) {
        e[off..off + 2].copy_from_slice(&v.to_le_bytes());
    }
    fn put_f32(e: &mut [u8], off: usize, v: f32) {
        e[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    pub struct ForgeObjectEditClient {
        mmf: Mmf,
    }

    impl ForgeObjectEditClient {
        pub fn open() -> Result<Self> {
            let mmf = Mmf::create_or_open(MAP_NAME, TOTAL)?;
            // Initialise the header if the producer hasn't (or a stale one).
            let magic: u32 = mmf.read(0);
            let ver: u32 = mmf.read(4);
            let cap: u32 = mmf.read(16);
            if magic != MAGIC || ver != VERSION || cap != CAP as u32 {
                mmf.write(0, MAGIC);
                mmf.write(4, VERSION);
                mmf.write(8, 0u32); // WriteSeq
                mmf.write(12, 0u32); // ReadSeq
                mmf.write(16, CAP as u32);
                mmf.write(20, 0u32); // LastStatus
                mmf.write(24, 0u32);
                mmf.write(28, 0u32);
            }
            Ok(Self { mmf })
        }

        fn enqueue(&self, entry: [u8; ENTRY]) {
            let seq: u32 = self.mmf.read(8);
            let off = HEADER + (seq as usize % CAP) * ENTRY;
            self.mmf.write_bytes(off, &entry);
            self.mmf.write(8, seq.wrapping_add(1)); // bump WriteSeq
        }

        /// Blank entry addressed to a forge object (index + datum hint).
        fn base_entry(forge_idx: u32, datum: u32, mask: u32) -> [u8; ENTRY] {
            let mut e = [0u8; ENTRY];
            put_u32(&mut e, 0, forge_idx);
            put_u32(&mut e, 4, mask);
            put_u32(&mut e, 8, datum);
            e
        }

        /// Set team + color (Team@+87, Color@+90).
        pub fn set_team_color(&self, forge_idx: u32, datum: u32, team: u8, color: u8) {
            let mut e = Self::base_entry(forge_idx, datum, F_TEAM | F_COLOR);
            e[87] = team;
            e[90] = color;
            self.enqueue(e);
        }

        /// Set full pose (Pos@+24, Fwd@+36, Up@+48).
        pub fn set_pose(&self, forge_idx: u32, datum: u32, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) {
            let mut e = Self::base_entry(forge_idx, datum, F_POSITION | F_ROTATION);
            put_f32(&mut e, 24, pos[0]);
            put_f32(&mut e, 28, pos[1]);
            put_f32(&mut e, 32, pos[2]);
            put_f32(&mut e, 36, fwd[0]);
            put_f32(&mut e, 40, fwd[1]);
            put_f32(&mut e, 44, fwd[2]);
            put_f32(&mut e, 48, up[0]);
            put_f32(&mut e, 52, up[1]);
            put_f32(&mut e, 56, up[2]);
            self.enqueue(e);
        }

        /// Set the persistent SpawnSequence (i8 @+81) — used by the scale
        /// convention to encode a saved scale into the forge variant.
        pub fn set_spawn_seq(&self, forge_idx: u32, datum: u32, seq: i8) {
            let mut e = Self::base_entry(forge_idx, datum, F_SPAWN_SEQ);
            e[81] = seq as u8;
            self.enqueue(e);
        }

        /// Live engine resize (Object_SetScale). LiveScale float @+96, mask F_LIVE_SCALE.
        pub fn set_scale(&self, forge_idx: u32, datum: u32, scale: f32) {
            let mut e = Self::base_entry(forge_idx, datum, F_LIVE_SCALE);
            put_f32(&mut e, 96, scale);
            self.enqueue(e);
        }

        /// Show=0 → engine despawns the object (Show@+16).
        pub fn delete(&self, forge_idx: u32, datum: u32) {
            let mut e = Self::base_entry(forge_idx, datum, F_SHOW);
            put_u16(&mut e, 16, 0);
            self.enqueue(e);
        }
    }
}

// =============================================================================
// Transform queue — ZeroHour_TransformQueue_Shared. Header(32) + Entry(64) * 16.
// The DLL drains it into Enqueue_UpdateObjectPosAndAiming. We enqueue moves.
// =============================================================================
pub mod transform_queue {
    use super::*;

    pub const MAP_NAME: &str = "ZeroHour_TransformQueue_Shared";
    pub const MAGIC: u32 = 0x5154_485A; // 'ZHTQ'
    pub const VERSION: u32 = 2;
    pub const CAP: usize = 16;
    const HEADER: usize = 32;
    const ENTRY: usize = 64;
    pub const TOTAL: usize = HEADER + ENTRY * CAP;

    pub const FLAG_HAS_POS: u32 = 0x1;
    pub const FLAG_HAS_FWD_UP: u32 = 0x2;
    pub const FLAG_DELETE: u32 = 0x4;

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct Header {
        magic: u32,
        version: u32,
        write_counter: u32,
        read_counter: u32,
        capacity: u32,
        _p0: u32,
        _p1: u32,
        _p2: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    struct Entry {
        datum: u32,
        flags: u32,
        p: [f32; 3],
        fwd: [f32; 3],
        up: [f32; 3],
        _r: [u32; 5],
    }

    pub struct TransformQueueClient {
        mmf: Mmf,
    }
    impl TransformQueueClient {
        pub fn open() -> Result<Self> {
            let mmf = Mmf::create_or_open(MAP_NAME, TOTAL)?;
            let h: Header = mmf.read(0);
            if h.magic != MAGIC || h.version != VERSION {
                let mut init = Header {
                    magic: MAGIC,
                    version: VERSION,
                    capacity: CAP as u32,
                    ..Default::default()
                };
                init.capacity = CAP as u32;
                mmf.write(0, init);
            }
            Ok(Self { mmf })
        }
        pub fn enqueue_pose(
            &self,
            datum: u32,
            flags: u32,
            pos: [f32; 3],
            fwd: [f32; 3],
            up: [f32; 3],
        ) -> bool {
            let mut h: Header = self.mmf.read(0);
            let slot = (h.write_counter as usize) % CAP;
            self.mmf.write(
                HEADER + slot * ENTRY,
                Entry { datum, flags, p: pos, fwd, up, _r: [0; 5] },
            );
            h.write_counter = h.write_counter.wrapping_add(1);
            self.mmf.write(0, h);
            true
        }
        pub fn enqueue_translate(&self, datum: u32, pos: [f32; 3]) -> bool {
            self.enqueue_pose(datum, FLAG_HAS_POS, pos, [0.0; 3], [0.0; 3])
        }
        /// Set position AND orientation (used by rotate).
        pub fn enqueue_pose_full(&self, datum: u32, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) -> bool {
            self.enqueue_pose(datum, FLAG_HAS_POS | FLAG_HAS_FWD_UP, pos, fwd, up)
        }
        pub fn enqueue_delete(&self, datum: u32) -> bool {
            self.enqueue_pose(datum, FLAG_DELETE, [0.0; 3], [0.0; 3], [0.0; 3])
        }
    }

    const _: () = assert!(std::mem::size_of::<Header>() == HEADER);
    const _: () = assert!(std::mem::size_of::<Entry>() == ENTRY);
}

pub use forge_object_edit::ForgeObjectEditClient;
pub use forge_object_table::{ForgeObjectTableClient, ForgePlacement};
pub use forge_palette::{ForgePaletteClient, PaletteEntry};
pub use map_info::{MapInfo, MapInfoClient};
pub use pose_snapshot::PoseSnapshotClient;
pub use forge_spawn_pose::{
    ForgeSpawnClient, Status as ForgeSpawnStatus, METHOD_DEFAULT, METHOD_LIVE, RAW_TAG_SENTINEL,
};
pub use object_table::{ObjectInfo, ObjectTableClient};
pub use transform_queue::TransformQueueClient;
pub use world_spawn::WorldSpawnClient;
