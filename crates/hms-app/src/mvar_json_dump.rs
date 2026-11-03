//! Ground-truth dump for the JavaScript `.mvar` reader (`docs/js/hms-mvar.js`). #mvar-js
//!
//! Test-only and `#[ignore]`d: it walks the folders named by `HMS_MVAR_DUMP_DIRS` (";"-separated,
//! two levels deep), decodes every `.mvar` with the SAME parsers the application uses
//! (`mvar::parse_variant`, `h3::mvar::parse_h3_variant`, `h4::mvar::parse_h4_variant`; the
//! dispatch is the `mvar` chunk's major version: 12 Halo 3, 31 Reach, 50 Halo 4, 52 H2A) and writes
//! one JSON line per file to `HMS_MVAR_DUMP_OUT`. Every f32 is written as its raw IEEE bits so
//! the comparison in `docs/js/test/hms-mvar.test.mjs` is exact, and every 64-bit value as a
//! decimal string. Print-only: nothing here writes next to the inputs, and the output holds
//! decoded fields only (no game content is copied anywhere the repo tracks).
//!
//!   HMS_MVAR_DUMP_DIRS="/path/a;/path/b" HMS_MVAR_DUMP_OUT=/tmp/truth.jsonl \
//!     cargo test --profile quick -p hms-app mvar_json_dump -- --ignored --nocapture

use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn fb(x: f32) -> u32 { x.to_bits() }
fn fbs(x: &[f32]) -> Vec<u32> { x.iter().map(|v| v.to_bits()).collect() }
fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }
fn s64(v: u64) -> String { v.to_string() }

fn walk(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            if depth > 0 { walk(&p, depth - 1, out); }
        } else if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) {
            out.push(p);
        }
    }
}

/// The `mvar` chunk's major version (None when the file has no readable `mvar` chunk).
fn mvar_major(data: &[u8]) -> Option<u16> {
    let mut pos = 0usize;
    while pos + 12 <= data.len() {
        let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if size < 12 || pos + size > data.len() { return None; }
        if &data[pos..pos + 4] == b"mvar" { return Some(u16::from_be_bytes([data[pos + 8], data[pos + 9]])); }
        pos += size;
    }
    None
}

fn hash_kind(data: &[u8]) -> Value {
    match crate::h4::mvar::detect_hash_kind(data) {
        Some(crate::h4::mvar::H4HashKind::PlainLe) => json!("plain-le"),
        Some(crate::h4::mvar::H4HashKind::SaltedBe360) => json!("salted-be-360"),
        None => Value::Null,
    }
}

fn reach(path: &Path, data: &[u8]) -> Result<Value, String> {
    let v = crate::mvar::parse_variant(path).ok_or("parse_variant returned None")?;
    let g = &v.globals;
    let objs: Vec<Value> = v.objects.iter().map(|o| json!({
        "folder": o.folder, "item": o.item, "pos": fbs(&o.pos), "team": o.team, "color": o.color,
        "fwd": fbs(&o.fwd), "up": fbs(&o.up), "up_is_global": o.up_is_global, "up_quant": o.up_quant,
        "forward_angle_q": o.forward_angle_q, "spawn_seq": o.spawn_seq, "respawn": o.respawn,
        "cached_type": o.cached_type, "label_idx": o.label_idx, "placement": o.placement,
        "boundary_shape": o.boundary_shape, "flags": o.flags, "in_bounds": o.in_bounds,
        "bsp_index": o.bsp_index, "spawn_rel": o.spawn_rel, "slot": o.slot, "boundary": o.boundary,
        "weapon_clips": o.weapon_clips, "tele_channel": o.tele_channel,
        "tele_passability": o.tele_passability, "location_name": o.location_name,
    })).collect();
    Ok(json!({
        "title": v.title, "description": v.description, "author": v.author, "editor": v.editor,
        "labels": v.labels, "objects": objs,
        "g": {
            "content_type": g.content_type, "file_size": g.file_size, "uid": s64(g.uid),
            "parent_uid": s64(g.parent_uid), "root_uid": s64(g.root_uid), "game_id": s64(g.game_id),
            "activity": g.activity, "game_mode": g.game_mode, "engine": g.engine,
            "header_map_id": g.header_map_id, "category": g.category,
            "created": [s64(g.created.0), s64(g.created.1)], "modified": [s64(g.modified.0), s64(g.modified.1)],
            "created_online": g.created_online, "modified_online": g.modified_online,
            "hopper_id": g.hopper_id, "version": g.version, "checksum": g.checksum,
            "palette_crc": g.palette_crc, "num_quotas": g.num_quotas, "map_id": g.map_id,
            "built_in": g.built_in, "built_from_xml": g.built_from_xml, "bounds": fbs(&g.bounds),
            "budget_max": g.budget_max, "budget_spent": g.budget_spent,
            "mcc_map_id": g.mcc_map_id.map(|b| hex(&b)),
            "quotas": g.quotas.iter().map(|q| json!([q.0, q.1, q.2])).collect::<Vec<_>>(),
            "end_bit": g.end_bit, "chunk_length": g.chunk_length, "hash_ok": g.hash_ok, "has_fsm": g.has_fsm,
            "hash_kind": hash_kind(data),
        },
    }))
}

fn h3(path: &Path) -> Result<Value, String> {
    let v = crate::h3::mvar::parse_h3_variant(path).map_err(|e| e.to_string())?;
    let h = &v.header;
    let objs: Vec<Value> = v.objects.iter().map(|o| json!({
        "slot": o.slot, "flags": o.flags, "quota": o.quota, "parent": o.parent.map(s64), "placed": o.placed,
        "pos": fbs(&o.pos), "fwd": fbs(&o.fwd), "up": fbs(&o.up), "pos_q": o.pos_q,
        "up_is_global": o.up_is_global, "up_q": o.up_q, "angle_q": o.angle_q, "boundary_q": o.boundary_q,
        "object_type": o.object_type, "mp_flags": o.mp_flags, "game_engine_flags": o.game_engine_flags,
        "shared_storage": o.shared_storage, "spawn_time": o.spawn_time, "team": o.team,
        "unique": o.mp_flags & crate::h3::edit::MP_UNIQUE != 0, "placed_at_start": o.mp_flags & crate::h3::edit::MP_NOT_AT_START == 0,
        "engine_mask": o.game_engine_flags & 0x3FF,
        "boundary_shape": o.boundary_shape, "boundary": fbs(&o.boundary),
    })).collect();
    Ok(json!({
        "header": {
            "unique_id": s64(h.unique_id), "name": h.name, "description": h.description, "author": h.author,
            "file_type": h.file_type, "author_is_online": h.author_is_online, "author_xuid": s64(h.author_xuid),
            "size_bytes": s64(h.size_bytes), "date": s64(h.date), "length_seconds": h.length_seconds,
            "campaign_id": h.campaign_id, "map_id": h.map_id, "game_engine_type": h.game_engine_type,
            "campaign_difficulty": h.campaign_difficulty, "hopper_id": h.hopper_id, "game_id": s64(h.game_id),
        },
        "chunk_major": v.chunk_major, "chunk_minor": v.chunk_minor, "version": v.version, "checksum": v.checksum,
        "num_scenario_objects": v.num_scenario_objects, "num_variant_objects": v.num_variant_objects,
        "num_quotas": v.num_quotas, "map_id": v.map_id, "built_in": v.built_in, "bounds": fbs(&v.bounds),
        "game_engine_subtype": v.game_engine_subtype, "maximum_budget": fb(v.maximum_budget),
        "spent_budget": fb(v.spent_budget), "objects": objs, "type_start": v.type_start,
        "quotas": v.quotas.iter().map(|q| json!({
            "definition": q.definition, "minimum": q.minimum, "maximum": q.maximum,
            "placed_on_map": q.placed_on_map, "maximum_allowed": q.maximum_allowed, "price": fb(q.price),
        })).collect::<Vec<_>>(),
        "mcc_map_id": v.mcc_map_id.map(|b| hex(&b)), "end_bit": v.end_bit,
    }))
}

fn h4(path: &Path, data: &[u8]) -> Result<Value, String> {
    use crate::h4::mvar::H4TypeData as T;
    let v = crate::h4::mvar::parse_h4_variant(path).map_err(|e| e.to_string())?;
    let objs: Vec<Value> = v.objects.iter().map(|o| {
        let td = match &o.type_data {
            T::Byte(b) => json!({"k": "byte", "v": b}),
            T::Index(i) => json!({"k": "index", "v": i}),
            T::TraitZone(t) => json!({"k": "trait_zone", "v": t}),
            T::Pair(a, b) => json!({"k": "pair", "a": a, "b": b}),
            T::InitialOrdnance { team, weapon, timing } => json!({"k": "initial_ordnance", "team": team, "weapon": weapon, "timing": timing}),
            T::RandomOrdnance(w) => json!({"k": "random_ordnance", "weapons": w}),
            T::ObjectiveOrdnance { lead, weapons } => json!({"k": "objective_ordnance", "lead": lead, "weapons": weapons}),
            T::Empty => json!({"k": "empty"}),
        };
        json!({
            "slot": o.slot, "bit": o.bit, "flags": o.flags, "quota": o.quota, "variant": o.variant, "in_bounds": o.in_bounds,
            "pos": fbs(&o.pos), "pos_q": o.pos_q, "h2a_bsp_ref": o.h2a_bsp_ref, "up_is_global": o.up_is_global,
            "up_quant": o.up_quant, "forward_angle_q": o.forward_angle_q, "fwd": fbs(&o.fwd), "up": fbs(&o.up),
            "parent": o.parent, "scale_q": o.scale_q, "locked": o.locked, "shape": o.shape as u8,
            "shape_values": o.shape_values, "object_type": o.object_type, "placement": o.placement, "team": o.team,
            "spawn_time": o.spawn_time, "color": o.color, "spawn_sequence": o.spawn_sequence, "user_data": o.user_data,
            "labels": o.labels, "h2a_extra": o.h2a_extra, "type_data": td,
            "scale": fb(o.scale()), "shape_wu": fbs(&o.shape_wu()),
        })
    }).collect();
    let mut m = serde_json::Map::new();
    let parts = [
        json!({
            "chunk_version": v.chunk_version, "title_key": v.title_key, "description_key": v.description_key,
            "author": v.author, "editor": v.editor, "created": [s64(v.created.0), s64(v.created.1)],
            "modified": [s64(v.modified.0), s64(v.modified.1)], "created_online": v.created_online,
            "modified_online": v.modified_online, "content_type": v.content_type, "file_size": v.file_size,
            "uid": s64(v.uid), "parent_uid": s64(v.parent_uid), "root_uid": s64(v.root_uid), "game_id": s64(v.game_id),
        }),
        json!({
            "activity": v.activity, "game_mode": v.game_mode, "engine": v.engine, "header_map_id": v.header_map_id,
            "category": v.category, "hopper_id": v.hopper_id, "version": v.version, "map_guid": v.map_guid.map(|b| hex(&b)),
            "checksum": v.checksum, "palette_crc": v.palette_crc, "num_quotas": v.num_quotas, "map_id": v.map_id,
            "built_in": v.built_in, "built_from_xml": v.built_from_xml, "bounds": fbs(&v.bounds),
        }),
        json!({
            "budget_max": v.budget_max, "budget_spent": v.budget_spent, "labels": v.labels, "objects": objs,
            "quotas": v.quotas.iter().map(|q| json!([q.minimum, q.maximum, q.placed])).collect::<Vec<_>>(),
            "obj_start_bit": v.obj_start_bit, "obj_end_bit": v.obj_end_bit, "quota_end_bit": v.quota_end_bit,
            "end_bit": v.end_bit, "chunk_length": v.chunk_length, "has_fsm": v.has_fsm,
            "hash_kind": hash_kind(data),
        }),
    ];
    for p in parts { if let Value::Object(o) = p { m.extend(o); } }
    Ok(Value::Object(m))
}

#[test]
#[ignore]
fn dump_mvar_json_for_js() {
    let dirs = std::env::var("HMS_MVAR_DUMP_DIRS").expect("set HMS_MVAR_DUMP_DIRS (\";\"-separated folders)");
    let out_path = std::env::var("HMS_MVAR_DUMP_OUT").expect("set HMS_MVAR_DUMP_OUT (output .jsonl)");
    let mut files = Vec::new();
    for d in dirs.split(';').filter(|s| !s.is_empty()) { walk(Path::new(d), 2, &mut files); }
    files.sort();
    files.dedup();
    let mut lines = String::new();
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for p in &files {
        let data = match crate::mvar::read_expanded(p) { Ok(d) => d, Err(_) => continue };
        let major = mvar_major(&data);
        let (game, res) = match major {
            Some(12) => ("halo3", h3(p)),
            Some(31) => ("reach", reach(p, &data)),
            Some(50) => ("halo4", h4(p, &data)),
            Some(52) => ("h2a", h4(p, &data)),
            _ => ("unknown", Err(format!("mvar chunk major {major:?} not handled"))),
        };
        *counts.entry(format!("{game}:{}", if res.is_ok() { "ok" } else { "err" })).or_default() += 1;
        let line = match res {
            Ok(v) => json!({"path": p.to_string_lossy(), "game": game, "major": major, "ok": true, "v": v}),
            Err(e) => json!({"path": p.to_string_lossy(), "game": game, "major": major, "ok": false, "error": e}),
        };
        lines.push_str(&line.to_string());
        lines.push('\n');
    }
    std::fs::write(&out_path, lines).expect("write dump");
    eprintln!("mvar_json_dump: {} files -> {counts:?}", files.len());
}

/// `variant get palette-json` prints exactly the table `HmsMvar.withPalette` reads.
#[test]
fn palette_json_field_is_the_documented_table() {
    let mut g = crate::mvar::VariantGlobals { game: "Reach", map_id: 3006, ..Default::default() };
    g.quotas = vec![(0, 1, 2), (0, 0, 0)];
    let e = crate::mvar::GlobalEdits::from_globals(&g);
    let names = vec!["Walls / Wall \"double\"".to_string(), String::new()];
    let s = crate::mvar::globals_report(&g, &e, &names, Some("palette-json"), "", "", "", "", &[]).unwrap();
    let v: Value = serde_json::from_str(&s).expect("valid JSON");
    assert_eq!(v["game"], "reach");
    assert_eq!(v["mapId"], 3006);
    assert_eq!(v["names"], json!(["Walls / Wall \"double\"", ""]));
}
