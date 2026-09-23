//! What an object IS — its obje tag path, object class, tag id and (for the map's own
//! placements) the scenario placement coordinates. Pure formatting: the App / script host
//! resolve the tag path + class through the cache and hand the result here, so the status
//! line, the Object panel, the objects-list tooltip and the `get` / `pick` script verbs all print
//! the same thing.

/// Where an object comes from (decides which extra coordinates are meaningful).
#[derive(Clone, Debug, PartialEq)]
pub enum ObjSource {
    /// A baked scnr placement. `index` = HMS's enumeration index (datum low bits),
    /// `palette_index` = index into the category's scnr palette block, `name_index` = the
    /// scnr object-names index (-1 = unnamed).
    Scenario { index: u32, palette_index: i16, name_index: i16 },
    /// A .mvar / Forge-variant object: `palette_name` = the sandbox palette item the user knows,
    /// `slot` = its .mvar slot (None = placed this session, not yet saved).
    Variant { palette_name: String, slot: Option<u16> },
    /// A locally-placed object (no variant record).
    Placed { palette_name: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjIdentity {
    pub datum: u32,
    /// obj (scen/vehi/weap/…) tag short id.
    pub obj_tag: u32,
    /// Resolved render_model tag id (0 / 0xFFFFFFFF = none).
    pub mode_tag: u32,
    /// Full obje tag path (lowercase, backslashes), "" when the cache cannot name it.
    pub tag_path: String,
    /// Human object class ("scenery", "machine", …), "" when unknown.
    pub class: String,
    pub source: ObjSource,
}

/// Scenario category id (native walker `kCategories` order) → object class name.
pub fn category_name(cat: u16) -> &'static str {
    match cat {
        0 => "scenery",
        1 => "biped",
        2 => "vehicle",
        3 => "equipment",
        4 => "weapon",
        5 => "machine",
        6 => "control",
        7 => "giant",
        8 => "effect scenery",
        9 => "crate",
        _ => "",
    }
}

/// Tag class code (4cc, e.g. "scen") → object class name. Unknown codes pass through unchanged
/// (still informative: the user sees the raw group), "" stays "".
pub fn class_from_code(code: &str) -> String {
    let c = code.trim().to_ascii_lowercase();
    let known = match c.as_str() {
        "scen" => "scenery",
        "bipd" => "biped",
        "vehi" => "vehicle",
        "eqip" => "equipment",
        "weap" => "weapon",
        "mach" => "machine",
        "ctrl" => "control",
        "gint" => "giant",
        "efsc" => "effect scenery",
        "bloc" => "crate",
        "term" => "terminal",
        "ssce" => "sound scenery",
        "crea" => "creature",
        "proj" => "projectile",
        "armr" => "armor",
        "argd" => "armor device",
        "" => "",
        _ => return c,
    };
    known.to_string()
}

impl ObjIdentity {
    /// The tag path's leaf ("lamp_post" of "objects\…\lamp_post"), or "" when unnamed.
    pub fn leaf(&self) -> &str {
        self.tag_path.rsplit(['\\', '/']).next().unwrap_or("")
    }

    /// `0xE0000012 objects\levels\multi\forge_halo\lamp_post (scenery, tag 0x1ac4, scnr #18)`.
    /// One line for the status bar and the `pick` verb.
    pub fn status_line(&self) -> String {
        let name = if self.tag_path.is_empty() { "(unnamed tag)".to_string() } else { self.tag_path.clone() };
        let mut parts: Vec<String> = Vec::new();
        if !self.class.is_empty() {
            parts.push(self.class.clone());
        }
        parts.push(format!("tag 0x{:04x}", self.obj_tag & 0xFFFF));
        match &self.source {
            ObjSource::Scenario { index, .. } => parts.push(format!("scnr #{index}")),
            ObjSource::Variant { palette_name, slot } => {
                if !palette_name.is_empty() {
                    parts.push(format!("palette '{palette_name}'"));
                }
                match slot {
                    Some(s) => parts.push(format!("slot #{s}")),
                    None => parts.push("unsaved".to_string()),
                }
            }
            ObjSource::Placed { palette_name } => {
                if !palette_name.is_empty() {
                    parts.push(format!("palette '{palette_name}'"));
                }
                parts.push("placed".to_string());
            }
        }
        format!("0x{:08X} {name} ({})", self.datum, parts.join(", "))
    }

    /// Multi-line block for the `get` verb / dumps (two-space indented, trailing newline).
    pub fn dump_lines(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("  tag_path={}\n", if self.tag_path.is_empty() { "(unnamed)" } else { &self.tag_path }));
        s.push_str(&format!(
            "  class={} tag_id=0x{:04x} mode_tag=0x{:04x}\n",
            if self.class.is_empty() { "?" } else { &self.class },
            self.obj_tag & 0xFFFF,
            self.mode_tag & 0xFFFF
        ));
        match &self.source {
            ObjSource::Scenario { index, palette_index, name_index } => s.push_str(&format!(
                "  source=scenario placement #{index} (palette_index={palette_index} name_index={name_index})\n"
            )),
            ObjSource::Variant { palette_name, slot } => s.push_str(&format!(
                "  source=variant palette='{palette_name}' slot={}\n",
                slot.map(|v| v.to_string()).unwrap_or_else(|| "new".to_string())
            )),
            ObjSource::Placed { palette_name } => s.push_str(&format!("  source=placed palette='{palette_name}'\n")),
        }
        s
    }

    /// Objects-list row tooltip (a few short lines).
    pub fn tooltip(&self) -> String {
        let mut s = String::new();
        match &self.source {
            ObjSource::Variant { palette_name, slot } => {
                if !palette_name.is_empty() {
                    s.push_str(&format!("palette: {palette_name}\n"));
                }
                match slot {
                    Some(v) => s.push_str(&format!("slot: #{v}\n")),
                    None => s.push_str("slot: not saved yet\n"),
                }
            }
            ObjSource::Placed { palette_name } => {
                if !palette_name.is_empty() {
                    s.push_str(&format!("palette: {palette_name}\n"));
                }
                s.push_str("placed this session\n");
            }
            ObjSource::Scenario { index, .. } => s.push_str(&format!("map object, scenario placement #{index}\n")),
        }
        s.push_str(&format!("tag: {}\n", if self.tag_path.is_empty() { "(unnamed)" } else { &self.tag_path }));
        s.push_str(&format!(
            "class: {}   tag id: 0x{:04x}   datum: 0x{:08X}",
            if self.class.is_empty() { "?" } else { &self.class },
            self.obj_tag & 0xFFFF,
            self.datum
        ));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scen() -> ObjIdentity {
        ObjIdentity {
            datum: 0xE000_0012,
            obj_tag: 0x1ac4,
            mode_tag: 0x1ac5,
            tag_path: "objects\\levels\\multi\\forge_halo\\lamp_post".into(),
            class: "scenery".into(),
            source: ObjSource::Scenario { index: 0x12, palette_index: 3, name_index: -1 },
        }
    }

    #[test]
    fn category_and_class_names() {
        assert_eq!(category_name(0), "scenery");
        assert_eq!(category_name(5), "machine");
        assert_eq!(category_name(9), "crate");
        assert_eq!(category_name(42), "");
        assert_eq!(class_from_code("scen"), "scenery");
        assert_eq!(class_from_code("BLOC"), "crate");
        assert_eq!(class_from_code("vehi"), "vehicle");
        assert_eq!(class_from_code(""), "");
        // unknown codes pass through (lowercased) rather than vanishing
        assert_eq!(class_from_code("xyzw"), "xyzw");
    }

    #[test]
    fn scenario_status_line() {
        assert_eq!(
            scen().status_line(),
            "0xE0000012 objects\\levels\\multi\\forge_halo\\lamp_post (scenery, tag 0x1ac4, scnr #18)"
        );
        assert_eq!(scen().leaf(), "lamp_post");
    }

    #[test]
    fn variant_status_line_and_unnamed() {
        let mut v = scen();
        v.datum = 0xD000_0003;
        v.source = ObjSource::Variant { palette_name: "wall_coliseum".into(), slot: Some(41) };
        assert_eq!(
            v.status_line(),
            "0xD0000003 objects\\levels\\multi\\forge_halo\\lamp_post (scenery, tag 0x1ac4, palette 'wall_coliseum', slot #41)"
        );
        v.source = ObjSource::Variant { palette_name: String::new(), slot: None };
        v.tag_path.clear();
        v.class.clear();
        assert_eq!(v.status_line(), "0xD0000003 (unnamed tag) (tag 0x1ac4, unsaved)");
        assert_eq!(v.leaf(), "");
    }

    #[test]
    fn dump_and_tooltip_carry_the_placement_coordinates() {
        let d = scen().dump_lines();
        assert!(d.contains("  tag_path=objects\\levels\\multi\\forge_halo\\lamp_post\n"), "{d}");
        assert!(d.contains("  class=scenery tag_id=0x1ac4 mode_tag=0x1ac5\n"), "{d}");
        assert!(d.contains("  source=scenario placement #18 (palette_index=3 name_index=-1)\n"), "{d}");
        let t = scen().tooltip();
        assert!(t.starts_with("map object, scenario placement #18\n"), "{t}");
        assert!(t.contains("tag id: 0x1ac4"), "{t}");
        // every line stays ASCII (egui's bundled fonts lack most symbols)
        assert!(t.is_ascii() && d.is_ascii() && scen().status_line().is_ascii());
    }
}
