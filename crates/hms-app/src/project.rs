//! .mmsproj project files — Rust port of the C# MmsProject. Stores the map, the camera view, and
//! a snapshot of placed forge objects (datum + pose + team/colour).
//!
//! There are two ways in:
//!   * the QUICK SLOT — one well-known file in the settings dir, for "save where I was" with no
//!     dialog (`save`/`load`);
//!   * a NAMED project anywhere on disk (`save_to`/`load_from`), which is what you need to keep
//!     more than one and to open a specific one. The quick slot alone meant "Load project" could
//!     only ever reopen the single last thing, with no way to navigate to another.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone)]
pub struct ProjObject {
    pub datum: u32,
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    pub team: u8,
    pub color: u8,
}

/// One object's PSEUDO-flag overrides (SCALED / SHADOW). These never go into the
/// .mvar, so the project file is where a session keeps them. Keyed by the object's .mvar SLOT
/// (stable across saves; datums are slot arithmetic that goes stale) with the datum as the
/// fallback key for objects that have no slot yet (placed in HMS, not saved). Only objects with
/// an explicit override are recorded — `None` means "follow the derived default".
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ProjObjFlags {
    pub slot: u16,
    pub datum: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<bool>,
    /// Displayed damage state (1 minor .. 4 destroyed); absent = the undamaged default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damage: Option<u8>,
}

/// One externally imported model (#4) and its pose. The project records WHERE the file is and
/// HOW it is placed; the geometry itself is re-read from `path` on load (an .obj can be tens of
/// megabytes, and the file is the user's own asset — the project is not an archive of it).
///
/// `fwd`/`up` is the same orientation representation a placed forge object uses, so the editor's
/// gizmo drives an import with no conversion; `scale` is uniform.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ProjImport {
    pub path: String,
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    pub scale: f32,
}

impl Default for ProjImport {
    fn default() -> Self {
        Self { path: String::new(), pos: [0.0; 3], fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], scale: 1.0 }
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct Project {
    pub version: u32,
    pub map_path: String,
    pub cam_pos: [f32; 3],
    pub cam_yaw: f32,
    pub cam_pitch: f32,
    pub objects: Vec<ProjObject>,
    /// The open .mvar (so reopening the project restores the same variant and the
    /// flag overrides below have something to attach to). Empty = no variant was open.
    #[serde(default)]
    pub variant_path: String,
    /// Per-object pseudo-flag overrides (see [`ProjObjFlags`]).
    #[serde(default)]
    pub obj_flags: Vec<ProjObjFlags>,
    /// #4: externally imported models (OBJ/GLB) with their user-driven poses. Absent in projects
    /// written before imports were movable, which is why it defaults to empty.
    #[serde(default)]
    pub imports: Vec<ProjImport>,
}

/// The per-user settings/data directory, on every platform.
///
/// `%LOCALAPPDATA%\HaloMapStudio` on Windows (Wine/Proton runs define LOCALAPPDATA too); on
/// Linux/macOS that variable does not exist, so this falls back to `$XDG_DATA_HOME/HaloMapStudio`,
/// then `~/.local/share/HaloMapStudio`.
pub fn settings_dir() -> Option<PathBuf> {
    if let Some(la) = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(la).join("HaloMapStudio"));
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(base.join("HaloMapStudio"))
}

pub fn project_path() -> Option<PathBuf> {
    settings_dir().map(|d| d.join("project.mmsproj"))
}

/// Write the project to an explicit path (File ▸ Save Project As…).
pub fn save_to(p: &Project, path: &std::path::Path) -> anyhow::Result<PathBuf> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let json = serde_json::to_string_pretty(p)?;
    std::fs::write(path, json)?;
    Ok(path.to_path_buf())
}

/// Read a project from an explicit path (File ▸ Open Project…).
pub fn load_from(path: &std::path::Path) -> anyhow::Result<Project> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

/// Quick slot: save to the well-known file in the settings dir.
pub fn save(p: &Project) -> anyhow::Result<PathBuf> {
    let path = project_path().ok_or_else(|| anyhow::anyhow!("cannot locate a settings directory (no LOCALAPPDATA, XDG_DATA_HOME or HOME)"))?;
    save_to(p, &path)
}

pub fn load() -> Option<Project> {
    let path = project_path()?;
    let json = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&json).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The settings dir must resolve on Linux, where LOCALAPPDATA is not set.
    #[test]
    fn settings_dir_resolves_without_localappdata() {
        // This test process is the Linux case: LOCALAPPDATA is not set here.
        if std::env::var_os("LOCALAPPDATA").is_some() {
            // On a Windows/Wine run the variable exists and is used directly — still must resolve.
            assert!(settings_dir().is_some());
            return;
        }
        let d = settings_dir().expect("no settings dir resolved without LOCALAPPDATA");
        assert!(d.ends_with("HaloMapStudio"), "unexpected settings dir {d:?}");
        assert!(d.is_absolute(), "settings dir must be absolute, got {d:?}");
        let p = project_path().expect("no project path resolved");
        assert!(p.ends_with("project.mmsproj"), "unexpected project path {p:?}");
    }

    /// A NAMED project must round-trip to an arbitrary path, and two projects must be able to
    /// coexist (the quick slot alone can only hold the last session).
    #[test]
    fn named_projects_round_trip_and_coexist() {
        let tmp = std::env::temp_dir().join(format!("hms-named-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let a = Project {
            version: 1,
            map_path: "/maps/alpha.map".into(),
            cam_pos: [1.0, 2.0, 3.0],
            cam_yaw: 0.5,
            cam_pitch: -0.25,
            objects: vec![ProjObject { datum: 0xD0000001, pos: [4.0, 5.0, 6.0], fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], team: 3, color: 4 }],
            ..Default::default()
        };
        let b = Project {
            version: 1,
            map_path: "/maps/beta.map".into(),
            cam_pos: [9.0, 8.0, 7.0],
            cam_yaw: -1.5,
            cam_pitch: 0.75,
            objects: vec![],
            ..Default::default()
        };
        let pa = tmp.join("alpha.mmsproj");
        let pb = tmp.join("beta.mmsproj");
        save_to(&a, &pa).expect("save alpha");
        save_to(&b, &pb).expect("save beta");
        assert!(pa.exists() && pb.exists(), "both projects must exist side by side");

        let ra = load_from(&pa).expect("load alpha");
        let rb = load_from(&pb).expect("load beta");
        assert_eq!(ra.map_path, "/maps/alpha.map");
        assert_eq!(rb.map_path, "/maps/beta.map", "the second project was overwritten by the first");
        assert_eq!(ra.objects.len(), 1);
        assert_eq!(ra.objects[0].datum, 0xD0000001);
        assert_eq!(ra.objects[0].team, 3);
        assert!((ra.cam_yaw - 0.5).abs() < 1e-6 && (rb.cam_pitch - 0.75).abs() < 1e-6);

        // saving into a directory that does not exist yet must still work
        let deep = tmp.join("nested").join("sub").join("deep.mmsproj");
        save_to(&a, &deep).expect("save into a new directory");
        assert!(deep.exists());

        // a missing / unreadable project is an error, not a panic
        assert!(load_from(&tmp.join("nope.mmsproj")).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The per-object pseudo-flags are NOT in the .mvar, so the project must carry
    /// them (keyed by slot, datum fallback) plus the variant they belong to; an old project
    /// without the fields must still load (both default empty).
    #[test]
    fn obj_flags_round_trip_and_old_projects_load() {
        let tmp = std::env::temp_dir().join(format!("hms-objflags-proj-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = Project {
            version: 1,
            map_path: "/maps/forge_halo.map".into(),
            variant_path: "/variants/shadow.mvar".into(),
            obj_flags: vec![
                ProjObjFlags { slot: 12, datum: 0xD000000C, scaled: Some(false), shadow: None, damage: None },
                ProjObjFlags { slot: 0xFFFF, datum: 0xF0000003, scaled: None, shadow: Some(true), damage: None },
                ProjObjFlags { slot: 7, datum: 0xD0000007, scaled: None, shadow: None, damage: Some(4) },
            ],
            ..Default::default()
        };
        let path = tmp.join("flags.mmsproj");
        save_to(&p, &path).expect("save");
        let back = load_from(&path).expect("load");
        assert_eq!(back.variant_path, "/variants/shadow.mvar");
        assert_eq!(back.obj_flags, p.obj_flags);
        // a project without the flag fields still loads
        let old = r#"{"version":1,"map_path":"/maps/a.map","cam_pos":[0,0,0],"cam_yaw":0,"cam_pitch":0,"objects":[]}"#;
        let legacy: Project = serde_json::from_str(old).expect("legacy project must parse");
        assert!(legacy.variant_path.is_empty() && legacy.obj_flags.is_empty());
        // a record that carries a retired field (the Halo 4 pseudo-scale, now in the .mvar) still parses
        let pre_scale = r#"{"version":1,"map_path":"","cam_pos":[0,0,0],"cam_yaw":0,"cam_pitch":0,"objects":[],"obj_flags":[{"slot":3,"datum":3489660931,"scaled":true,"scale":2.5}]}"#;
        let ps: Project = serde_json::from_str(pre_scale).expect("project with a retired field must parse");
        assert_eq!(ps.obj_flags, vec![ProjObjFlags { slot: 3, datum: 0xD0000003, scaled: Some(true), shadow: None, damage: None }]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// #4: an imported model's POSE must survive a save/load round trip, and a project written
    /// before imports existed (no `imports` key) must still load.
    #[test]
    fn imports_round_trip_and_old_projects_load() {
        let tmp = std::env::temp_dir().join(format!("hms-imports-proj-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = Project {
            version: 1,
            map_path: "/maps/forge_halo.map".into(),
            imports: vec![
                ProjImport { path: "/home/u/my models/ship.obj".into(), pos: [12.5, -3.0, 40.25], fwd: [0.0, 1.0, 0.0], up: [0.0, 0.0, 1.0], scale: 2.5 },
                ProjImport { path: "/elsewhere/terrain.glb".into(), ..Default::default() },
            ],
            ..Default::default()
        };
        let path = tmp.join("imports.mmsproj");
        save_to(&p, &path).expect("save");
        let back = load_from(&path).expect("load");
        assert_eq!(back.imports, p.imports, "the imported models' poses did not round-trip");
        // the pose must be exact, not rounded through the JSON
        assert_eq!(back.imports[0].pos, [12.5, -3.0, 40.25]);
        assert!((back.imports[0].scale - 2.5).abs() < 1e-9);
        // a path with spaces must come back intact
        assert_eq!(back.imports[0].path, "/home/u/my models/ship.obj");
        // the default pose is the identity (where a fresh import lands)
        assert_eq!(back.imports[1].fwd, [1.0, 0.0, 0.0]);
        assert_eq!(back.imports[1].up, [0.0, 0.0, 1.0]);
        assert!((back.imports[1].scale - 1.0).abs() < 1e-9);
        // a project from before imports were movable has no `imports` key at all
        let old = r#"{"version":1,"map_path":"/maps/a.map","cam_pos":[0,0,0],"cam_yaw":0,"cam_pitch":0,"objects":[]}"#;
        let legacy: Project = serde_json::from_str(old).expect("legacy project must parse");
        assert!(legacy.imports.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// #model-import: a project records the model's PATH and pose, never its geometry — so the
    /// materials and textures must come back from the file itself on reload. This writes a real
    /// textured OBJ + MTL + PNG, round-trips a project that references it, and re-reads the
    /// restored path to prove the materials are identical.
    #[test]
    fn imports_round_trip_keeps_their_materials() {
        let tmp = std::env::temp_dir().join(format!("hms-import-mat-proj-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        // a 2x2 PNG, an .mtl that binds it, and an OBJ with two usemtl groups
        let mut png = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut png, 2, 2);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255]).unwrap();
        }
        std::fs::write(tmp.join("skin.png"), &png).unwrap();
        std::fs::write(tmp.join("m.mtl"), "newmtl painted\nKd 1 1 1\nmap_Kd skin.png\n\nnewmtl plain\nKd 0.2 0.6 0.4\n").unwrap();
        let obj_path = tmp.join("thing.obj");
        std::fs::write(
            &obj_path,
            "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
             usemtl painted\nf 1/1 2/2 3/3\nusemtl plain\nf 1/1 3/3 4/4\n",
        )
        .unwrap();

        // what the model yields BEFORE the round trip
        let before = crate::modelimport::load(&obj_path).expect("the authored OBJ must load");
        assert_eq!(before.materials.len(), 2);
        assert_eq!(before.images.len(), 1);
        assert_eq!(before.parts.len(), 2);

        let p = Project {
            version: 1,
            imports: vec![ProjImport {
                path: obj_path.to_string_lossy().into_owned(),
                pos: [3.5, -2.0, 11.0],
                fwd: [0.0, 1.0, 0.0],
                up: [0.0, 0.0, 1.0],
                scale: 1.75,
            }],
            ..Default::default()
        };
        let proj = tmp.join("p.mmsproj");
        save_to(&p, &proj).expect("save");
        let back = load_from(&proj).expect("load");
        assert_eq!(back.imports, p.imports, "path + pose must round-trip");

        // ... and re-reading the restored path must give the SAME materials and textures
        let after = crate::modelimport::load(std::path::Path::new(&back.imports[0].path)).expect("the restored path must still load");
        assert_eq!(after.materials.len(), before.materials.len());
        assert_eq!(after.images.len(), before.images.len());
        assert_eq!(after.parts.len(), before.parts.len());
        assert_eq!(after.textured_material_count(), 1);
        assert!(after.has_uvs(), "the restored model must still carry texture coordinates");
        assert_eq!(after.material_summary(), before.material_summary());
        for (a, b) in after.materials.iter().zip(before.materials.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.base_color, b.base_color);
            assert_eq!(a.image, b.image);
        }
        assert_eq!(after.images[0].rgba, before.images[0].rgba, "the decoded texture must be identical");
        // the restored pose drives the mesh exactly as a live import does
        let m = {
            let r = &back.imports[0];
            let mut m = crate::extimport::ImportedModel::at_identity(std::path::PathBuf::from(&r.path), vec![]);
            m.pos = r.pos;
            m.fwd = r.fwd;
            m.up = r.up;
            m.scale = r.scale;
            m
        };
        assert_eq!(m.pos, [3.5, -2.0, 11.0]);
        assert!((m.scale - 1.75).abs() < 1e-6);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Saving must actually write the file, not just resolve a path.
    #[test]
    fn save_then_load_roundtrips() {
        let tmp = std::env::temp_dir().join(format!("hms-proj-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        // point the resolver at a scratch dir for this process
        std::env::set_var("XDG_DATA_HOME", &tmp);
        std::env::remove_var("LOCALAPPDATA");
        let p = Project { cam_pitch: 0.5, ..Default::default() };
        let written = save(&p).expect("save failed");
        assert!(written.exists(), "save reported success but wrote nothing");
        let back = load().expect("load failed");
        assert!((back.cam_pitch - 0.5).abs() < 1e-6);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
