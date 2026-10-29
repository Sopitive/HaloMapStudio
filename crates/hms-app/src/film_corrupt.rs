//! #film-crc  A user reported "invalid CRC" opening a Theater clip. This reader validates NO
//! checksum (the engine does: `networking:saved_film: read compressed game state checksum
//! mismatch`, `read failed to header for file storage [checksum mismatch]`), so these gates pin
//! what a damaged, cut, padded or still-being-written clip does here, from a real film:
//! it opens whatever is readable, refuses only what is not a Reach film at all, and no error text
//! it can emit mentions a CRC.
use crate::film::{self, Film, FilmError};

fn smallest() -> Option<(String, Vec<u8>)> {
    let mut best: Option<(u64, std::path::PathBuf)> = None;
    for p in film::spec_corpus() {
        if p.extension().map_or(false, |e| e == "temp") { continue }
        let Ok(m) = std::fs::metadata(&p) else { continue };
        if m.len() > 0x30000 && best.as_ref().map_or(true, |(s, _)| m.len() < *s) {
            best = Some((m.len(), p));
        }
    }
    let (_, p) = best?;
    let name = p.file_name()?.to_string_lossy().into_owned();
    Some((name, std::fs::read(&p).ok()?))
}

fn describe(r: &Result<Film, FilmError>) -> String {
    match r {
        Ok(f) => format!("OK ticks={} records={} framing_exact={}", f.tick_count(), f.data.records.len(), f.data.framing_exact),
        Err(e) => format!("ERR \"{e}\""),
    }
}

#[test]
fn damaged_cut_and_padded_clips() {
    let Some((name, good)) = smallest() else { eprintln!("no film: skipped"); return };
    let base = film::parse(&good).expect("the known-good clip opens");
    let flmd = base.data.chunk_offset;
    let n = base.data.records.len();
    assert!(n > 100, "fixture too small to be meaningful");
    println!("fixture {name}: {} bytes, {}", good.len(), describe(&Ok(base.clone())));
    let mut texts: Vec<String> = Vec::new();
    let mut rec = |what: &str, bytes: Vec<u8>| -> Result<Film, FilmError> {
        let r = film::parse(&bytes);
        let d = describe(&r);
        println!("  {what:<44} {d}");
        texts.push(d);
        r
    };

    // --- must ACCEPT (the engine's own checks live in the game-state blob, which this reader never reads)
    let mut flipped = good.clone();
    let mid = flmd + 16 + (good.len() - flmd) / 2;
    flipped[mid] ^= 0xFF;
    flipped[mid + 1] ^= 0x55;
    let r = rec("bit-flip mid-stream", flipped.clone()).expect("a flipped byte in the stream must still open");
    assert_eq!(r.header.map_variant.metadata.title, base.header.map_variant.metadata.title);
    // ...and the decoder must survive it (it reports a stop; it must not panic)
    let _ = crate::film_decode::build_index_film(&r);

    let mut padded = good.clone();
    padded.extend(std::iter::repeat(0u8).take(4096));
    let r = rec("4096 bytes of zero padding after the end", padded).expect("trailing padding must open");
    assert!(r.data.records.len() >= n, "padding must not lose records ({} < {n})", r.data.records.len());

    let mut garbage = good.clone();
    garbage.extend((0..777u32).map(|i| (i * 31 + 7) as u8));
    rec("777 bytes of garbage after the end", garbage).expect("trailing garbage must open");

    for (label, cut) in [("cut at 90% (MCC still finalising)", good.len() * 9 / 10), ("cut mid-record at 50%", good.len() / 2 + 3), ("cut right after the flmd header", flmd + 16)] {
        let r = rec(label, good[..cut].to_vec()).unwrap_or_else(|e| panic!("{label}: a cut clip must open what it has, got {e}"));
        assert!(r.data.records.len() <= n);
    }
    // a .temp: flmd size field still the template's 12
    let mut temp = good.clone();
    temp[flmd + 4..flmd + 8].copy_from_slice(&12u32.to_le_bytes());
    let r = rec("flmd size field = 12 (a .temp)", temp).expect("an in-progress recording opens");
    assert!(r.data.records.len() >= n * 9 / 10, "the whole stream is read when the size is the template's");
    let mut huge = good.clone();
    huge[flmd + 4..flmd + 8].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
    rec("flmd size field = 0xFFFFFFF0 (32-bit overflow)", huge).expect("an overflowing size field must not panic or refuse");

    // --- must REFUSE, with an error that says what is wrong
    let e = rec("header only, no flmh", good[..0x200].to_vec()).err().expect("no flmh is not openable");
    assert!(matches!(e, FilmError::MissingChunk(_) | FilmError::NotBlf), "{e}");
    let e = rec("not a film at all", b"PK\x03\x04 this is a zip".to_vec()).err().unwrap();
    assert!(matches!(e, FilmError::NotBlf));
    let mut other = good.clone();
    other[14..22].copy_from_slice(b"halo 3\0\0");
    let _ = rec("container name changed", other);

    // --- the point: nothing this reader can say mentions a checksum
    for t in &texts {
        let l = t.to_lowercase();
        assert!(!l.contains("crc") && !l.contains("checksum"), "{t}");
    }
}

/// Halo 4 saved films (`halo4 saved film` container) are valid films of another title. This reader
/// is Reach-only and must say so in plain words -- not panic, hang or claim damage. Runs on the
/// user-supplied sample when it is present.
#[test]
fn a_halo4_film_is_refused_with_the_clear_title_message() {
    let Some(home) = std::env::var_os("HOME") else { return };
    let dl = std::path::Path::new(&home).join("Downloads");
    let mut seen = 0;
    for name in ["Reach.mov", "asq_mglo-1_ca_forge__A5992DE3_6938D76B.mov"] {
        let Ok(bytes) = std::fs::read(dl.join(name)) else { continue };
        let t = std::time::Instant::now();
        let r = film::parse(&bytes);
        assert!(t.elapsed().as_secs() < 5, "{name}: refusing must be instant");
        let e = r.err().unwrap_or_else(|| panic!("{name}: expected a refusal"));
        println!("{name}: {} bytes -> \"{e}\"", bytes.len());
        assert!(matches!(&e, FilmError::WrongGame(n) if n == "halo4 saved film"), "{e}");
        assert!(e.to_string().contains("not a Reach film"));
        seen += 1;
    }
    eprintln!("halo 4 sample films checked: {seen}");
}
