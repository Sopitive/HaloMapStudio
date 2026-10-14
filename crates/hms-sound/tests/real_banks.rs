//! Tests against the installed game's own banks.
//!
//! These skip themselves when Reach MCC is not installed at the path below,
//! so the suite still passes on a machine without the game. When it IS
//! installed they are the real gate: they assert the measured facts the
//! parser was built from.

use std::path::PathBuf;

fn reach_dir() -> Option<PathBuf> {
    let p = PathBuf::from(
        "/mnt/990pro/SteamLibrary/steamapps/common/Halo The Master Chief Collection/haloreach",
    );
    if p.join("fmod/pc/sfx.fsb").exists() {
        Some(p)
    } else {
        None
    }
}

#[test]
fn the_sfx_bank_is_fmod_vorbis_at_48k_throughout() {
    let Some(reach) = reach_dir() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    let b = hms_sound::SoundBank::open(&reach.join("fmod/pc"), "sfx").expect("open sfx bank");
    assert_eq!(b.bank.codec, hms_sound::Codec::Vorbis, "codec");
    assert_eq!(b.bank.samples.len(), 16_162, "sample count");
    assert!(b.info.is_some(), "sidecar should be present and match");
    for s in &b.bank.samples {
        assert_eq!(s.sample_rate, 48_000, "sample {} rate", s.index);
    }
    // Only two distinct Vorbis setup headers are used by the whole bank.
    let region = b.bank.sample_header_region();
    let mut ids: Vec<u32> = b
        .bank
        .samples
        .iter()
        .filter_map(|s| s.vorbis_setup_id(region))
        .collect();
    assert_eq!(ids.len(), 16_162, "every sample carries a setup id");
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids, vec![0x632B_1FAC, 0xAD11_D38D], "setup ids in use");
}

#[test]
fn the_sidecar_gives_the_true_channel_count_and_the_asset_path() {
    let Some(reach) = reach_dir() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    let b = hms_sound::SoundBank::open(&reach.join("fmod/pc"), "sfx").unwrap();
    let info = b.info.as_ref().unwrap();
    assert_eq!(
        info.records[0].path,
        r"data\sound\characters\bodyfalls\generic_havok\generic2_1.aif"
    );
    // frames * channels * 2 == the recorded source size, for every sample.
    let mut checked = 0;
    for (i, s) in b.bank.samples.iter().enumerate() {
        let r = info.get(i).unwrap();
        if let Some(frames) = r.frames() {
            // The bank and the sidecar agree to within one frame; the sidecar
            // rounds the source length down.
            let d = s.frames as i64 - frames as i64;
            assert!(
                (0..=1).contains(&d),
                "sample {i} ({}) frames {} vs sidecar {}",
                r.path,
                s.frames,
                frames
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 16_162);
    // The sidecar records four-channel samples that the bank's 2-bit field
    // reports as mono.
    let quad = info.records.iter().filter(|r| r.channels == 4).count();
    assert_eq!(quad, 464, "four-channel samples");
}

#[test]
fn the_setup_table_comes_out_of_the_shipped_game_module() {
    let Some(reach) = reach_dir() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    let dll = reach.join("haloreach.dll");
    if !dll.exists() {
        eprintln!("skipped: haloreach.dll not present");
        return;
    }
    let t = hms_sound::SetupTable::from_module(&dll).expect("read setup table");
    let stereo = t.get(0xAD11_D38D).expect("stereo setup header");
    let mono = t.get(0x632B_1FAC).expect("mono setup header");
    // True lengths, recovered by parsing. The module's own size field is 79
    // and 72 bytes short respectively, which is why it is not used.
    assert_eq!(stereo.len(), 3484);
    assert_eq!(mono.len(), 3006);
    let hs = hms_sound::vorbis::parse_setup(stereo, 2).unwrap();
    assert_eq!(hs.codebooks, 34);
    assert_eq!(hs.coupling_steps, 1, "the stereo header couples channels");
    assert_eq!(hs.mode_blockflags, vec![false, true]);
    let hm = hms_sound::vorbis::parse_setup(mono, 1).unwrap();
    assert_eq!(hm.codebooks, 32);
    assert_eq!(
        hm.coupling_steps, 0,
        "the mono header declares no coupling, which is why mono, quad and \
         5.1 samples all share it"
    );
    // Because it declares no coupling it must parse identically at any
    // channel count.
    for ch in [1u8, 2, 4, 6] {
        assert_eq!(hms_sound::vorbis::parse_setup(mono, ch).unwrap(), hm);
    }
}

#[test]
fn an_assault_rifle_shot_reframes_into_a_decodable_ogg() {
    let Some(reach) = reach_dir() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    let dll = reach.join("haloreach.dll");
    if !dll.exists() {
        eprintln!("skipped: haloreach.dll not present");
        return;
    }
    let b = hms_sound::SoundBank::open(&reach.join("fmod/pc"), "sfx").unwrap();
    let t = hms_sound::SetupTable::from_module(&dll).unwrap();
    let info = b.info.as_ref().unwrap();
    let hits = info.search(r"assault_rifle\ar_fire\ar_reach1.aif");
    assert_eq!(hits.len(), 1, "ar_reach1 should be in the bank exactly once");
    let idx = hits[0].index;
    let declared = b.bank.samples[idx].frames;
    assert_eq!(declared, 57_671);

    let ogg = b.sample_to_ogg(idx, &t).expect("re-frame");
    assert_eq!(&ogg[..4], b"OggS");
    // Walk the pages: the first must be BOS, the last EOS with the granule
    // set to the declared frame count, which is what makes a decoder emit
    // exactly that many frames.
    let (pages, last_granule, saw_bos, saw_eos) = walk_pages(&ogg);
    assert!(pages >= 3, "pages = {pages}");
    assert!(saw_bos && saw_eos);
    assert_eq!(last_granule, declared as i64);
}

/// Count pages and return (pages, final granule, saw BOS, saw EOS).
fn walk_pages(d: &[u8]) -> (usize, i64, bool, bool) {
    let mut o = 0usize;
    let mut pages = 0;
    let mut granule = 0i64;
    let mut bos = false;
    let mut eos = false;
    while o + 27 <= d.len() && &d[o..o + 4] == b"OggS" {
        let flags = d[o + 5];
        granule = i64::from_le_bytes(d[o + 6..o + 14].try_into().unwrap());
        bos |= flags & hms_sound::ogg::BOS != 0;
        eos |= flags & hms_sound::ogg::EOS != 0;
        let nsegs = d[o + 26] as usize;
        let body: usize = d[o + 27..o + 27 + nsegs].iter().map(|s| *s as usize).sum();
        // Verify the page checksum: a mistake in the writer would show here.
        let end = o + 27 + nsegs + body;
        let mut copy = d[o..end].to_vec();
        let stored = u32::from_le_bytes(copy[22..26].try_into().unwrap());
        copy[22..26].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(stored, hms_sound::ogg::ogg_crc(&copy), "page {pages} checksum");
        o = end;
        pages += 1;
    }
    assert_eq!(o, d.len(), "trailing bytes after the last page");
    (pages, granule, bos, eos)
}

#[test]
fn the_map_gestalt_joins_to_the_bank_through_the_subsound_id() {
    let Some(reach) = reach_dir() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    let map = reach.join("maps/forge_halo.map");
    if !map.exists() {
        eprintln!("skipped: forge_halo.map not present");
        return;
    }
    let dir = reach.join("fmod/pc");
    let sfx = hms_sound::SoundBank::open(&dir, "sfx").unwrap();
    let eng = hms_sound::SoundBank::open(&dir, "english").unwrap();
    let si = sfx.info.as_ref().unwrap();
    let ei = eng.info.as_ref().unwrap();
    let cache = std::fs::read(&map).unwrap();

    let known = |v: u32| si.index_of_subsound(v).is_some() || ei.index_of_subsound(v).is_some();
    let hits = hms_sound::scan_cache_for_subsound_ids(&cache, &known, 50);
    assert!(
        hits.len() > 9_000,
        "expected the gestalt's permutation runs, found {}",
        hits.len()
    );

    // The decisive check: each permutation's `uncompressed sample count` must
    // equal the frame count of the bank sample its hash selects. If the join
    // were wrong these would be unrelated numbers.
    let mut agree = 0usize;
    for (_entry, hash, frames) in &hits {
        let rec = si
            .index_of_subsound(*hash)
            .and_then(|i| si.get(i))
            .or_else(|| ei.index_of_subsound(*hash).and_then(|i| ei.get(i)));
        let Some(rec) = rec else { continue };
        let Some(f) = rec.frames() else { continue };
        if (*frames as i64 - f as i64).abs() <= 1 {
            agree += 1;
        }
    }
    let ratio = agree as f64 / hits.len() as f64;
    assert!(
        ratio > 0.98,
        "only {agree} of {} permutations agree with the bank ({:.1}%)",
        hits.len(),
        ratio * 100.0
    );
}

#[test]
fn a_corrupted_bank_fails_cleanly_rather_than_panicking() {
    let Some(reach) = reach_dir() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    // Take the real bank's first 1 MB and corrupt the header fields one at a
    // time. Nothing may panic; everything must be an Err. Only the first
    // megabyte is read: the bank is 712 MB and pulling it into memory while
    // other builds are running is a good way to get the session killed.
    let mut head = vec![0u8; 1 << 20];
    {
        use std::io::Read;
        let mut f = std::fs::File::open(reach.join("fmod/pc/sfx.fsb")).unwrap();
        f.read_exact(&mut head).unwrap();
    }
    let head: &[u8] = &head;
    let dir = std::env::temp_dir().join(format!("hms-sound-corrupt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("c.fsb");
    let mut failures = 0;
    for at in [0x04usize, 0x08, 0x0c, 0x10, 0x14, 0x18] {
        for v in [0u32, 1, 0xFFFF_FFFF, 0x7FFF_FFFF] {
            let mut b = head.to_vec();
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
            std::fs::write(&p, &b).unwrap();
            // Truncated to 1 MB, the real bank already fails the
            // "regions account for the file" check; the point is that no
            // field value makes it panic.
            assert!(
                hms_sound::Bank::open(&p).is_err(),
                "field 0x{at:x} = 0x{v:x} should not parse"
            );
            failures += 1;
        }
    }
    assert_eq!(failures, 24);
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_dir(&dir);
}

// -------------------------------------------------------------------------------------------
// The in-process decoder, measured against ffmpeg.
// -------------------------------------------------------------------------------------------

/// The one sample every number in `re-notes/reach_fmod_sound.md` §4 was measured on.
const AR_REACH1: &str = r"assault_rifle\ar_fire\ar_reach1.aif";

fn open_sfx() -> Option<(hms_sound::SoundBank, hms_sound::SetupTable)> {
    let reach = reach_dir()?;
    let dll = reach.join("haloreach.dll");
    if !dll.exists() {
        eprintln!("skipped: haloreach.dll not present");
        return None;
    }
    let b = hms_sound::SoundBank::open(&reach.join("fmod/pc"), "sfx").ok()?;
    let t = hms_sound::SetupTable::from_module(&dll).ok()?;
    Some((b, t))
}

#[test]
fn the_decoder_reproduces_the_measured_assault_rifle_waveform() {
    let Some((b, t)) = open_sfx() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    let idx = b.info.as_ref().unwrap().search(AR_REACH1)[0].index;

    // THE PRODUCTION ENTRY POINT. Nothing here builds its own packet list or
    // its own header; a caller in the app calls exactly this.
    let pcm = b.decode_sample(idx, &t).expect("decode");

    // Measured against ffmpeg's independent decoder, 2026-10-13.
    //
    // NOTE: `re-notes/reach_fmod_sound.md` §4 recorded "peak -0.00 dBFS, RMS
    // -14.66 dB" for this sample. Those were measured through a CLIPPED
    // 16-bit conversion. The stream itself peaks at 17.78 (+25.00 dBFS) —
    // Reach's FMOD Vorbis streams run well above full scale — and ffmpeg
    // agrees to seven significant figures. §4's envelope figures match from
    // the second block onward; only the first block, the transient, differed,
    // which is exactly the clipping signature.
    assert_eq!(pcm.channels, 2);
    assert_eq!(pcm.sample_rate, 48_000);
    assert_eq!(pcm.frames(), 57_671, "frame count");
    assert!((pcm.duration_seconds() - 1.2015).abs() < 1e-3, "{}", pcm.duration_seconds());
    assert!(
        (pcm.peak() - 17.7798).abs() < 0.001,
        "peak {} ({:+.2} dBFS), measured 17.7798",
        pcm.peak(),
        20.0 * pcm.peak().log10()
    );
    assert!(
        (pcm.rms() - 0.701947).abs() < 1e-4,
        "RMS {} ({:+.2} dB), measured 0.701947",
        pcm.rms(),
        20.0 * pcm.rms().log10()
    );

    // The decaying envelope: a gunshot, not noise and not silence.
    let n = pcm.frames() / 20;
    let env: Vec<f64> = (0..20)
        .map(|k| {
            let a = k * n;
            let s: f64 = (a..a + n)
                .filter_map(|i| pcm.frame(i))
                .flat_map(|f| f.iter())
                .map(|v| (*v as f64) * (*v as f64))
                .sum();
            (s / (n * 2) as f64).sqrt()
        })
        .collect();
    assert!(
        env[0] / env[19] > 2000.0,
        "envelope ratio {:.0}:1, expected a decay near 7200:1 ({env:?})",
        env[0] / env[19]
    );
    for k in 1..19 {
        assert!(env[k] >= env[k + 1] * 0.5, "envelope rose at block {k}: {env:?}");
    }
}

#[test]
fn the_in_process_decode_agrees_with_ffmpeg_on_the_reframed_ogg() {
    let Some((b, t)) = open_sfx() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    if std::process::Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipped: ffmpeg not installed");
        return;
    }
    let info = b.info.as_ref().unwrap();
    // A spread of samples, not just the one the parser was built on: mono,
    // stereo, four-channel and 5.1 all share two setup headers, so a mistake
    // in the channel handling would only show on some of them.
    let mut cases: Vec<usize> = vec![info.search(AR_REACH1)[0].index, 0];
    for want in [1u32, 2, 4, 6] {
        if let Some(r) = info
            .records
            .iter()
            .find(|r| r.channels == want && (8_000..200_000).contains(&r.source_pcm_bytes))
        {
            cases.push(r.index);
        }
    }
    cases.sort_unstable();
    cases.dedup();

    let dir = std::env::temp_dir().join(format!("hms-sound-ffmpeg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut checked = 0usize;
    for idx in cases {
        let ch = b.channels(idx).unwrap();
        let pcm = b.decode_sample(idx, &t).expect("decode");
        let ogg = b.sample_to_ogg(idx, &t).expect("reframe");
        let p = dir.join(format!("s{idx}.ogg"));
        std::fs::write(&p, &ogg).unwrap();
        let out = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&p)
            .args(["-f", "f32le", "-acodec", "pcm_f32le", "-"])
            .output()
            .expect("run ffmpeg");
        let _ = std::fs::remove_file(&p);
        assert!(out.status.success(), "ffmpeg failed on sample {idx}: {}", String::from_utf8_lossy(&out.stderr));
        let theirs: Vec<f32> = out
            .stdout
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            theirs.len(),
            pcm.samples.len(),
            "sample {idx} ({ch} ch): ffmpeg gave {} values, we gave {}",
            theirs.len(),
            pcm.samples.len()
        );
        assert!(!theirs.is_empty(), "sample {idx} decoded to nothing");
        let mut sum = 0.0f64;
        let mut worst = 0.0f32;
        for (a, c) in pcm.samples.iter().zip(theirs.iter()) {
            let d = (a - c).abs();
            sum += d as f64;
            worst = worst.max(d);
        }
        let mean = sum / theirs.len() as f64;
        // The bound has to be RELATIVE to the stream's own level: these
        // streams peak anywhere from 0.1 to 115, so an absolute 1-LSB bound
        // would be vacuous on a quiet sample and impossible on a loud one.
        // The two implementations differ only in last-place rounding.
        let scale = pcm.peak().max(1.0) as f64;
        assert!(
            mean / scale < 1.0e-5,
            "sample {idx} ({ch} ch): mean |diff| {mean:.3e} on a peak of {scale:.3} \
             = {:.1} ppm of full output",
            1.0e6 * mean / scale
        );
        assert!(
            worst as f64 / scale < 1.0e-2,
            "sample {idx} ({ch} ch): worst |diff| {worst:.3e} on a peak of {scale:.3}"
        );
        eprintln!(
            "sample {idx:>6} {ch} ch {:>7} frames  peak {:8.4}  mean |diff| {:.3} ppm  worst {:.3} ppm",
            pcm.frames(),
            pcm.peak(),
            1.0e6 * mean / scale,
            1.0e6 * worst as f64 / scale
        );
        checked += 1;
    }
    let _ = std::fs::remove_dir(&dir);
    assert!(checked >= 3, "only {checked} samples cross-checked");
}

#[test]
fn a_spread_of_the_bank_decodes_to_exactly_the_declared_frame_count() {
    let Some((b, t)) = open_sfx() else {
        eprintln!("skipped: Reach MCC not installed");
        return;
    };
    // Every 97th sample under 2 s: a few hundred samples across the whole
    // bank, covering both setup ids and all four channel counts. The decode
    // must never be short of the bank's own declared length — a short decode
    // is what a wrong block size or a dropped packet looks like.
    let mut n = 0usize;
    let mut silent = 0usize;
    let mut chans = [0usize; 9];
    for i in (0..b.bank.samples.len()).step_by(97) {
        let s = &b.bank.samples[i];
        if s.frames == 0 || s.frames > 96_000 {
            continue;
        }
        let pcm = match b.decode_sample(i, &t) {
            Ok(p) => p,
            Err(e) => panic!("sample {i} ({:?}): {e:#}", b.asset_path(i)),
        };
        assert_eq!(
            pcm.frames(),
            s.frames as usize,
            "sample {i} ({:?})",
            b.asset_path(i)
        );
        if pcm.peak() <= 0.0 {
            silent += 1;
        }
        chans[pcm.channels as usize] += 1;
        n += 1;
    }
    assert!(n > 100, "only {n} samples decoded");
    // A decoder that silently produced zeroes would pass every structural
    // check above, so require that almost nothing is digital silence.
    assert!(
        silent * 100 < n,
        "{silent} of {n} samples decoded to pure silence"
    );
    eprintln!("decoded {n} samples, channel histogram {chans:?}, {silent} silent");
    assert!(chans[1] > 0 && chans[2] > 0, "channel coverage {chans:?}");
}
