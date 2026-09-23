//! Halo 4 CHANGE COLOURS (`#h4-veh`): where the four per-object colours the shipped
//! `*_colorchangemap` / `*_teamcolor` / `srf_char_spartan_armor` / `srf_char_visor_mp` pixel
//! shaders read come from, and what a scenario / Forge placement resolves to.
//!
//! SHADER SIDE (DXBC, `srf_blinn_colorchangemap` entry 01; see docs/halo4_lighting_model.md
//! section 14): the four colours are one cbuffer array
//!   `EngineMaterialPS.ps_material_object_parameters[4]`  (float4 x 4, +0 .. +0x3C)
//! = [primary, secondary, tertiary, quaternary]; the register NUMBER of that cbuffer differs per
//! entry point (cb1 in the G-buffer entries, cb2 in the boundary family, cb3 in the `teamcolor`
//! lighting entries), only the name and the layout are stable. The `*_colorchangemap` law is
//!   albedo = lerp(color.rgb, luma709(color.rgb) * cc.rgb, pcc_amount_map.r * cc.a) * up160.rgb
//! with `cc = ps_material_object_parameters[0]` (the PRIMARY colour) - so its `.a` is the
//! per-object MASK WEIGHT, not an opacity.
//!
//! GAME SIDE: a placement resolves its primary colour exactly as Reach does
//! (scene.rs `forge_tint` / `forge_tint_index`, RE'd from `reach_tag_test.exe`
//! `c_map_variant::update_object_multiplayer_properties` -> `object_update_forge_change_color`):
//!   * an explicit object COLOUR (mvar record 3-bit `color` field, 0..7) wins -> that palette entry
//!   * else a real TEAM (0..7) -> the team's colour
//!   * else (team 8 = neutral, or team none) -> the engine's constant mid grey (0.5, 0.5, 0.5)
//!   * a SCENARIO placement (no multiplayer properties at all) keeps the object's AUTHORED default
//!     change colours (`objects::default_change_colors`, obje +0x148).
//! The Halo 4 palette is the `mulg` (multiplayer globals) **Teams** block, +0x20, 0x24-byte
//! elements: name string id @0, `real_rgb_color` PRIMARY @4, SECONDARY @0x10 (Assembly
//! `Halo4MCC/mulg.xml`; the values are printed by `tests::dump_team_colors` and asserted in
//! `team_colors_are_authored`). Note the Teams block is NESTED: `mulg` +0x00 is the one-element
//! `Universal` block (0x74 B) and Teams is at that element +0x20. The eight shipped colours of
//! `multiplayer\\multiplayer_globals` on Ravine are, in block order (linear RGB):
//!   0 (0.7435, 0.1666, 0.1082) red      4 (0.5211, 0.2339, 0.8190) purple
//!   1 (0.1596, 0.2773, 0.4867) blue     5 (0.4638, 0.7435, 0.0717) lime
//!   2 (1.0000, 0.7951, 0.0000) gold     6 (1.0000, 0.4981, 0.1423) orange
//!   3 (0.0000, 0.4259, 0.0000) green    7 (0.1217, 0.6608, 0.7951) cyan
//! which is the order the mvar record's 3-bit `color` field and its 4-bit `team` field index.
//!
//! PROVISIONAL: the neutral constant (0.5, 0.5, 0.5) is Reach's, carried over unchanged - the Halo
//! 4 engine's own neutral constant was not located in `halo4.dll`. The secondary / tertiary /
//! quaternary slots of a Forge placement are left at the object's authored defaults (Reach
//! overwrites only the primary); no shipped Forge-palette material samples them.

use super::cache::{ByteRead, H4Cache};
use super::objects::default_change_colors;

/// `mulg` +0x00: the Universal block (one element, 0x74 B).
pub const OFF_MULG_UNIVERSAL: usize = 0x00;
pub const MULG_UNIVERSAL_ELEM: usize = 0x74;
/// Universal element +0x20: the Teams block (0x24 B elements).
pub const OFF_MULG_TEAMS: usize = 0x20;
pub const MULG_TEAM_ELEM: usize = 0x24;

/// The engine's neutral / no-team primary change colour (Reach's
/// `object_update_forge_change_color` constant, 0x14070C890).
pub const NEUTRAL_CHANGE_COLOR: [f32; 3] = [0.5, 0.5, 0.5];

/// The eight team primary colours of the loaded cache (`mulg` Teams block), linear RGB.
/// Falls back to the neutral grey for teams the tag does not list.
pub fn team_colors(c: &H4Cache) -> [[f32; 3]; 8] {
    let mut out = [NEUTRAL_CHANGE_COLOR; 8];
    let d = c.data();
    let Some(mulg) = c.find_tags(b"mulg").first().copied() else { return out };
    let Some(mm) = c.tag_meta(mulg) else { return out };
    let Some((_, uo)) = c.block(mm + OFF_MULG_UNIVERSAL) else { return out };
    let Some((n, o)) = c.block(uo + OFF_MULG_TEAMS) else { return out };
    for i in 0..n.min(8) {
        let e = o + i * MULG_TEAM_ELEM;
        let rgb = [d.f32_at(e + 4), d.f32_at(e + 8), d.f32_at(e + 12)];
        if rgb.iter().all(|v| v.is_finite() && (0.0..=8.0).contains(v)) { out[i] = rgb; }
    }
    out
}

/// The PRIMARY change colour a placement resolves to, plus the mask weight the shader
/// multiplies the `pcc_amount_map` by (`cc.a`).
///
/// `mp` = the variant record's `(team, color)` for a Forge / map-variant placement (team -1 none,
/// 0..7 a real team, 8 neutral; `color` = the 3-bit override, `None` = inherit). `None` = a
/// SCENARIO placement, which keeps the object's authored default.
pub fn primary_change_color(c: &H4Cache, obje_tag: usize, variant_sid: u32, mp: Option<(i8, Option<u8>)>) -> [f32; 4] {
    let authored = default_change_colors(c, obje_tag, variant_sid);
    let rgb = match mp {
        None => authored[0],
        Some((team, color)) => {
            let table = team_colors(c);
            match (color, team) {
                (Some(ci), _) if (ci as usize) < 8 => table[ci as usize],
                (_, t) if (0..8).contains(&t) => table[t as usize],
                _ => NEUTRAL_CHANGE_COLOR,
            }
        }
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
}

/// A quantised key for batching draws by change colour (the instance lane is per-mesh, so two
/// placements may only share a draw when their colour matches).
pub fn color_key(cc: [f32; 4]) -> [i32; 4] {
    let q = |v: f32| (v.clamp(0.0, 8.0) * 4096.0) as i32;
    [q(cc[0]), q(cc[1]), q(cc[2]), q(cc[3])]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;

    fn open(map: &str) -> Option<H4Cache> {
        let p = maps_dir()?.join(map);
        p.exists().then(|| H4Cache::open(&p).ok()).flatten()
    }

    /// RE tool: the `mulg` Teams block of `HMS_H4_TINT_MAP` (default `ca_forge_ravine.map`).
    /// Print-only, ignored.
    #[test]
    #[ignore]
    fn dump_team_colors() {
        let map = std::env::var("HMS_H4_TINT_MAP").unwrap_or_else(|_| "ca_forge_ravine.map".into());
        let Some(c) = open(&map) else { eprintln!("{map} not found"); return };
        let d = c.data();
        for mulg in c.find_tags(b"mulg") {
            let Some(mm) = c.tag_meta(mulg) else { continue };
            let Some((_, uo)) = c.block(mm + OFF_MULG_UNIVERSAL) else { continue };
            let Some((n, o)) = c.block(uo + OFF_MULG_TEAMS) else { continue };
            eprintln!("mulg '{}': {n} teams", c.tag_name(mulg));
            for i in 0..n.min(16) {
                let e = o + i * MULG_TEAM_ELEM;
                eprintln!("  [{i}] '{}' primary ({:.5},{:.5},{:.5}) secondary ({:.5},{:.5},{:.5})", c.sid(d.u32_at(e)),
                    d.f32_at(e + 4), d.f32_at(e + 8), d.f32_at(e + 12), d.f32_at(e + 0x10), d.f32_at(e + 0x14), d.f32_at(e + 0x18));
            }
        }
    }

    /// The `mulg` Teams block reads as eight distinct, in-range colours whose first two are
    /// red-dominant and blue-dominant. A wrong offset / stride gives neither.
    #[test]
    fn team_colors_are_authored() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let t = team_colors(&c);
        for (i, rgb) in t.iter().enumerate() {
            assert!(rgb.iter().all(|v| v.is_finite() && (0.0..=8.0).contains(v)), "team {i} colour {rgb:?} out of range");
        }
        assert!(t[0][0] > t[0][1] && t[0][0] > t[0][2], "team 0 is not red-dominant: {:?}", t[0]);
        assert!(t[1][2] > t[1][0] && t[1][2] > t[1][1], "team 1 is not blue-dominant: {:?}", t[1]);
        assert_ne!(t[0], t[1], "teams 0 and 1 share a colour (stride wrong?)");
    }

    /// The placement rule: an explicit colour wins over the team, a real team wins over
    /// neutral, and neutral / none is the grey constant.
    #[test]
    fn placement_color_rule() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let Some(v) = c.find_tags(b"vehi").first().copied() else { return };
        let t = team_colors(&c);
        assert_eq!(primary_change_color(&c, v, 0, Some((1, Some(0))))[..3], t[0][..], "explicit colour wins");
        assert_eq!(primary_change_color(&c, v, 0, Some((1, None)))[..3], t[1][..], "team colour when inheriting");
        assert_eq!(primary_change_color(&c, v, 0, Some((8, None)))[..3], NEUTRAL_CHANGE_COLOR[..], "neutral team -> grey");
        assert_eq!(primary_change_color(&c, v, 0, Some((-1, None)))[..3], NEUTRAL_CHANGE_COLOR[..], "no team -> grey");
        assert_eq!(primary_change_color(&c, v, 0, None)[..3], default_change_colors(&c, v, 0)[0][..], "scenario keeps the authored default");
    }
}
