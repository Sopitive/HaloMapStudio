// The sound AUDIT: signal analysis of the real decoded assets and an event audit of a scripted
// test-mode session on the real forge_halo cache. Everything renders into memory through the
// offline mixer (no device is ever opened). These gates measure the OUTPUT and the REAL SAMPLES,
// not the formulas the implementation uses.

/// In-place radix-2 FFT magnitude spectrum of the first `n` (power of two) samples of `x`
/// (Hann windowed). Returns `n/2` bin magnitudes.
pub(super) fn spectrum(x: &[f32]) -> Vec<f32> {
    let n = x.len().next_power_of_two() / if x.len().is_power_of_two() { 1 } else { 2 };
    let n = n.max(2);
    let mut re: Vec<f64> = (0..n).map(|i| f64::from(x[i]) * (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())).collect();
    let mut im = vec![0.0f64; n];
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        for s in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (a, b) = (s + k, s + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
        len <<= 1;
    }
    (0..n / 2).map(|i| (re[i] * re[i] + im[i] * im[i]).sqrt() as f32).collect()
}

/// `(spectral centroid Hz, strongest bin Hz, that bin's share of all energy)` of mono `x` at `rate`.
pub(super) fn describe(x: &[f32], rate: f32) -> (f32, f32, f32) {
    if x.len() < 64 {
        return (0.0, 0.0, 0.0);
    }
    let take = x.len().min(1 << 16);
    let s = spectrum(&x[..take]);
    let n = (s.len() * 2) as f32;
    let (mut num, mut den, mut best, mut bi) = (0.0f64, 0.0f64, 0.0f32, 0usize);
    for (i, m) in s.iter().enumerate().skip(1) {
        let e = f64::from(*m) * f64::from(*m);
        num += e * i as f64;
        den += e;
        if *m > best {
            best = *m;
            bi = i;
        }
    }
    if den <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let bw = rate / n;
    let share = ((f64::from(best) * f64::from(best)) / den) as f32;
    ((num / den) as f32 * bw, bi as f32 * bw, share)
}

/// Stereo interleaved -> mono by AVERAGING (the gates' `mono()` sums).
pub(super) fn avg_mono(x: &[f32]) -> Vec<f32> {
    x.chunks_exact(2).map(|c| 0.5 * (c[0] + c[1])).collect()
}

pub(super) fn mono_of(p: &hms_sound::Pcm) -> Vec<f32> {
    let c = p.channels.max(1) as usize;
    p.samples.chunks_exact(c).map(|f| f.iter().sum::<f32>() / c as f32).collect()
}

#[test]
fn the_spectrum_finds_a_tone() {
    let n = 8192;
    let x: Vec<f32> = (0..n).map(|i| (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / 48_000.0).sin()).collect();
    let (c, pk, share) = describe(&x, 48_000.0);
    assert!((pk - 3000.0).abs() < 12.0 && (c - 3000.0).abs() < 100.0 && share > 0.5, "{c} {pk} {share}");
}

/// PRINT-ONLY. For every sound the test mode can play, the REAL decoded sample against what the
/// tag declares: sample rate / frames / duration / centroid. A frames-vs-declared ratio that is
/// not 1.0 is a rate bug.
#[test]
#[ignore = "opens the native library; run alone with --ignored"]
fn probe_rates_of_real_assets() {
    let Some(mut rig) = rig() else { return };
    let (native, cache) = (rig.native.clone(), rig.cache);
    let g = rs::Gestalt::read(&native, cache).unwrap();
    // codec blocks
    let ugh = *native.tags_by_group(cache, "ugh!").first().unwrap();
    let h = native.tag_meta(cache, ugh, 4, 12).unwrap();
    let (cn, cp) = (u32::from_le_bytes(h[0..4].try_into().unwrap()), u32::from_le_bytes(h[4..8].try_into().unwrap()));
    let codecs = native.tag_ptr_bytes(cache, cp, cn * 3).unwrap_or_default();
    println!("codecs: {cn}");
    for i in 0..cn as usize {
        println!("  codec {i}: rate {} enc {} comp {}", codecs[i * 3], codecs[i * 3 + 1], codecs[i * 3 + 2]);
    }
    let keys = ["assault_rifle", "warthog", "spartan", "foley", "waterfall", "riverworld", "halo_ext", "lapping", "footstep", "plasma_rifle"];
    let mut tags: Vec<(String, u32)> = rs::sound_tags(&native, cache)
        .into_iter()
        .filter_map(|t| native.tag_name(cache, t).map(|n| (n, t)))
        .filter(|(n, _)| keys.iter().any(|k| n.contains(k)))
        .collect();
    tags.sort();
    println!("{} candidate sounds", tags.len());
    let mut asked: Vec<(String, u32, usize, u32, u8)> = Vec::new();
    for (n, t) in &tags {
        let Some(cs) = rs::CacheSound::read(&native, cache, *t) else { continue };
        let Some(clip) = rig.eng.resolve_one_shot(&native, cache, *t) else { continue };
        let perms = g.permutations_of(&cs);
        let Some(p0) = perms.first() else { continue };
        let _ = rig.eng.sample(clip.samples[0]);
        asked.push((n.clone(), *t, clip.samples[0], p0.sample_count, cs.codec_index as u8));
        let _ = p0;
    }
    let deadline = Instant::now() + Duration::from_secs(240);
    for (n, t, s, count, codec) in asked.iter().take(400) {
        let pcm = loop {
            if let Some(p) = rig.eng.sample(*s) {
                break p;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(3));
        };
        let cs = rs::CacheSound::read(&native, cache, *t).unwrap();
        let m = mono_of(&pcm);
        let (c, pk, share) = describe(&m[m.len().saturating_sub(1 << 16) / 2..], pcm.sample_rate as f32);
        let cdc = *codec as usize;
        // Band split of the whole sample: dBFS of everything, and of the part above 16 kHz.
        {
            let sp = spectrum(&m[..m.len().min(1 << 16)]);
            let tot: f64 = sp.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
            let k16 = (16_000.0 / (pcm.sample_rate as f32 / 2.0) * sp.len() as f32) as usize;
            let hf: f64 = sp[k16.min(sp.len())..].iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
            let rms = (m.iter().map(|v| v * v).sum::<f32>() / m.len().max(1) as f32).sqrt();
            println!("   [{n}] rms {:.1} dBFS, energy above 16 kHz {:.1}% (hf level {:.1} dBFS)", 20.0 * rms.max(1e-9).log10(), 100.0 * hf / tot.max(1e-30), 20.0 * (rms.max(1e-9) * (hf / tot.max(1e-30)).sqrt() as f32).max(1e-9).log10());
            if ["ar_pose_1", "dirt_wet", "concrete_wet", "light_lapping", "riverworld_interior", "run\\cobblestone_wet", "ar_fire", "ar_reload_full", "halo_ext"].iter().any(|k| n.contains(k)) {
                let dir = std::path::Path::new("/tmp/claude-1000/-home-sopitive/05268b5e-9d50-4b4a-9c35-93b66f9dc5fa/scratchpad/wav");
                let _ = std::fs::create_dir_all(dir);
                let mut b = Vec::new();
                let data: Vec<u8> = m.iter().flat_map(|v| ((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()).collect();
                b.extend_from_slice(b"RIFF");
                b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
                b.extend_from_slice(b"WAVEfmt ");
                b.extend_from_slice(&16u32.to_le_bytes());
                b.extend_from_slice(&1u16.to_le_bytes());
                b.extend_from_slice(&1u16.to_le_bytes());
                b.extend_from_slice(&pcm.sample_rate.to_le_bytes());
                b.extend_from_slice(&(pcm.sample_rate * 2).to_le_bytes());
                b.extend_from_slice(&2u16.to_le_bytes());
                b.extend_from_slice(&16u16.to_le_bytes());
                b.extend_from_slice(b"data");
                b.extend_from_slice(&(data.len() as u32).to_le_bytes());
                b.extend_from_slice(&data);
                let fname = n.replace('\\', "_");
                let _ = std::fs::write(dir.join(format!("{fname}.wav")), b);
            }
        }
        println!(
            "{:<70} class {:>2} codec[{}]={:?} perm.count {:>8} frames {:>8} ratio {:.3} rate {} ch {} dur {:.2}s maxplay {} ms  centroid {:.0} Hz peak {:.0} Hz ({:.2})",
            n.rsplit('\\').take(2).collect::<Vec<_>>().join("\\"), cs.class, cdc, codecs.get(cdc * 3..cdc * 3 + 3), count, pcm.frames(),
            pcm.frames() as f64 / (*count).max(1) as f64, pcm.sample_rate, pcm.channels, pcm.duration_seconds(), cs.maximum_play_time_ms, c, pk, share
        );
    }
}

/// PRINT-ONLY. Stand at several sound-scenery placements and report, per live scenery voice, the
/// playback ratio and gain it was started with, and the output's level and high-band share.
#[test]
#[ignore = "opens the native library; run alone with --ignored"]
fn probe_scenery_voices() {
    let Some(mut rig) = rig() else { return };
    rig.ps.mute_ambience = false;
    let mut r = Ride::with_vehicle(rig.native.clone(), rig.cache, "\\warthog\\warthog").expect("ride");
    let (native, cache) = (rig.native.clone(), rig.cache);
    let mut places: Vec<Vec3> = Vec::new();
    for k in 0..40 {
        let fr = Frame { native: &rig.native, cache: rig.cache, map_path: &rig.map, dt: DT, listener: listener_of(Vec3::new(0.0, 0.0, 2.0), 0.0), ride: None, weapon: 0, shots: 0, melee: false, reloading: false, dry_pulls: 0, offered: None, foot: None };
        rig.ps.update(&mut rig.eng, &fr);
        rig.eng.render(FRAMES_PER_TICK);
        if !rig.ps.scenery.placements.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20 + k));
    }
    for p in &rig.ps.scenery.placements {
        println!("placement {} at {:?} source {}", rig.ps.scenery.names[p.source].rsplit('\\').next().unwrap_or(""), p.pos, p.source);
        places.push(p.pos);
    }
    let _ = (&native, cache, &mut r);
    for (n, pos) in places.iter().take(8).enumerate() {
        let l = listener_of(*pos + Vec3::new(3.0, 0.0, 1.5), 0.0);
        let mut out: Vec<f32> = Vec::new();
        for i in 0..360 {
            let fr = Frame { native: &rig.native, cache: rig.cache, map_path: &rig.map, dt: DT, listener: l, ride: None, weapon: 0, shots: 0, melee: false, reloading: false, dry_pulls: 0, offered: None, foot: None };
            rig.ps.update(&mut rig.eng, &fr);
            let a = rig.eng.render(FRAMES_PER_TICK);
            if i >= 120 {
                out.extend(mono(&a));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let (c, pk, sh) = describe(&out[..out.len().min(1 << 16)], RATE as f32);
        println!("--- listener 3 wu from placement {n}: out rms {:.1} dBFS, centroid {c:.0} Hz, peak bin {pk:.0} Hz ({sh:.2}), scenery voices {}", 20.0 * rms(&out).max(1e-9).log10(), rig.ps.scenery.voices.len());
        for (k, v) in &rig.ps.scenery.voices {
            println!("    voice {:?}: src_rate {} pitch-ratio {:.4} gain {:.4}/{:.4} ({:.1} dB)", k, v.rate, v.mult, v.gain[0], v.gain[1], 20.0 * v.gain[0].max(v.gain[1]).max(1e-9).log10());
        }
    }
}

fn write_wav(path: &std::path::Path, x: &[f32], rate: u32) {
    let data: Vec<u8> = x.iter().flat_map(|v| ((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()).collect();
    let mut b = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(data.len() as u32).to_le_bytes());
    b.extend_from_slice(&data);
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let _ = std::fs::write(path, b);
}

const AUDIT_DIR: &str = "/tmp/claude-1000/-home-sopitive/05268b5e-9d50-4b4a-9c35-93b66f9dc5fa/scratchpad/audit";

/// PRINT-ONLY. One assault-rifle shot through the REAL mixer, against the same sample scaled by the
/// same authored gain with no limiter: how much distortion (and how much of it above 8 kHz, where
/// a squeak lives) does the mixer's output stage add?
#[test]
#[ignore = "opens the native library; run alone with --ignored"]
fn probe_shot_distortion() {
    let Some(mut rig) = rig() else { return };
    let (native, cache) = (rig.native.clone(), rig.cache);
    let weap = native
        .tags_by_group(cache, "weap")
        .into_iter()
        .find(|&t| native.tag_name(cache, t).map_or(false, |n| n.ends_with("assault_rifle\\assault_rifle")))
        .unwrap();
    rig.eng.warm_weapon(&native, cache, weap);
    let deadline = Instant::now() + Duration::from_secs(120);
    let l = listener_of(Vec3::ZERO, 0.0);
    loop {
        let fr = Frame { native: &rig.native, cache, map_path: &rig.map, dt: DT, listener: l, ride: None, weapon: weap, shots: 0, melee: false, reloading: false, dry_pulls: 0, offered: None, foot: None };
        rig.ps.update(&mut rig.eng, &fr);
        if rig.eng.sample_ready_for_weapon(&native, cache, weap) {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    // The clip as the engine plays it.
    let clip = rig.eng.weapon_fire_clip_for_test(&native, cache, weap).expect("a fire clip");
    for k in 0..clip.samples.len().min(3) {
        let seed = k as u32;
        let perm = clip.permutation_for(seed, weap);
        let pcm = rig.eng.sample(clip.samples[perm]).expect("decoded");
        let g = clip.gains_at([0.4, 0.0, 0.0], 0.4, perm);
        // Reference: the stereo sample times the authored gain, no limiter.
        let n = pcm.frames();
        let src_rate = pcm.sample_rate as f64;
        let step = src_rate / f64::from(RATE);
        let out_n = (n as f64 / step) as usize;
        let mut ok = rig.eng.play_weapon_sound(&native, cache, weap, WeaponSound::Fire, [0.4, 0.0, 0.0], l, seed);
        let mut y: Vec<f32> = Vec::new();
        for _ in 0..(out_n / FRAMES_PER_TICK + 8) {
            y.extend(rig.eng.render(FRAMES_PER_TICK));
        }
        if !ok {
            ok = false;
        }
        let mut peak_ref = 0.0f32;
        let xr: Vec<f32> = (0..out_n)
            .map(|i| {
                let f = pcm.frame((i as f64 * step) as usize).unwrap_or(&[0.0, 0.0]);
                let (a, b) = if pcm.channels == 1 { (f[0], f[0]) } else { (f[0], f[1]) };
                let v = (a * g[0] + b * g[1]) * 0.5;
                peak_ref = peak_ref.max(v.abs());
                v
            })
            .collect();
        let ym: Vec<f32> = y.chunks_exact(2).map(|c| (c[0] + c[1]) * 0.5).skip(hms_sound::mix::LOOKAHEAD).take(out_n).collect();
        let (mut xy, mut xx) = (0.0f64, 0.0f64);
        for (a, b) in xr.iter().zip(ym.iter()) {
            xy += f64::from(*a) * f64::from(*b);
            xx += f64::from(*a) * f64::from(*a);
        }
        let alpha = (xy / xx.max(1e-30)) as f32;
        let resid: Vec<f32> = xr.iter().zip(ym.iter()).map(|(a, b)| b - alpha * a).collect();
        let hp = |x: &[f32]| -> f64 {
            let sp = spectrum(&x[..x.len().min(1 << 16)]);
            let k = sp.len() / 3; // above 8 kHz
            sp[k..].iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>()
        };
        let (er, ex) = (resid.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>(), xr.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>());
        println!(
            "shot perm {perm}: started {ok}, ref peak {peak_ref:.3} (authored gain {g:?}), mixer peak {:.3}, best-fit scale {alpha:.3}, distortion {:.1} dB below signal, HF(>8k) distortion {:.1} dB below HF signal",
            ym.iter().fold(0.0f32, |m, v| m.max(v.abs())),
            10.0 * (er / ex.max(1e-30)).log10(),
            10.0 * (hp(&resid) / hp(&xr).max(1e-30)).log10()
        );
        write_wav(&std::path::Path::new(AUDIT_DIR).join(format!("shot{k}_mixer.wav")), &ym, RATE);
        write_wav(&std::path::Path::new(AUDIT_DIR).join(format!("shot{k}_ref.wav")), &xr, RATE);
    }
}

/// What one scripted session did, for the audit table.
struct Session {
    out: Vec<f32>,
    /// `(label, first output frame, last output frame)` of each phase.
    phases: Vec<(&'static str, usize, usize)>,
    /// The per-category level meter at the end of each phase (dBFS: other, ambience, engine, body, weapon).
    cats: Vec<(&'static str, [f32; 5])>,
}

fn run_session(rig: &mut Rig, r: &mut Ride) -> Session {
    let mut s = Session { out: Vec::new(), phases: Vec::new(), cats: Vec::new() };
    let still = MoveInput::default();
    let mut tick = |rig: &mut Rig, r: &mut Ride, s: &mut Session, inp: &MoveInput| {
        let a = frame(rig, r, inp, None);
        s.out.extend(avg_mono(&a));
    };
    let phase = |s: &mut Session, rig: &Rig, name: &'static str, from: usize| {
        s.phases.push((name, from, s.out.len()));
        s.cats.push((name, rig.ps.cat_rms_1s.map(|v| 20.0 * v.max(1e-6).log10())));
    };
    // Spawn: the banks decode on a worker thread; the real app has the same startup.
    let t0 = s.out.len();
    for _ in 0..180 {
        tick(rig, r, &mut s, &still);
        std::thread::sleep(Duration::from_millis(2));
    }
    phase(&mut s, rig, "spawn+stand", t0);
    for (name, inp, secs) in [
        ("walk", MoveInput { fwd: 1.0, walk: true, ..Default::default() }, 4.0f32),
        ("run", MoveInput { fwd: 1.0, ..Default::default() }, 4.0),
        ("sprint", MoveInput { fwd: 1.0, sprint: true, ..Default::default() }, 4.0),
    ] {
        let t0 = s.out.len();
        for _ in 0..(secs / DT) as usize {
            tick(rig, r, &mut s, &inp);
        }
        phase(&mut s, rig, name, t0);
    }
    let t0 = s.out.len();
    for _ in 0..3 {
        tick(rig, r, &mut s, &MoveInput { jump: true, ..Default::default() });
        for _ in 0..90 {
            tick(rig, r, &mut s, &still);
        }
    }
    phase(&mut s, rig, "jump x3", t0);
    // Fire: the AR held for 3 s at 15 shots a second (a pull every 4 ticks).
    let t0 = s.out.len();
    for i in 0..(3.0 / DT) as usize {
        if i % 4 == 0 {
            r.fire_once();
        }
        tick(rig, r, &mut s, &still);
    }
    for _ in 0..60 {
        tick(rig, r, &mut s, &still);
    }
    phase(&mut s, rig, "fire 3 s", t0);
    let t0 = s.out.len();
    rig.reloading = true;
    for _ in 0..120 {
        tick(rig, r, &mut s, &still);
    }
    rig.reloading = false;
    for _ in 0..30 {
        tick(rig, r, &mut s, &still);
    }
    phase(&mut s, rig, "reload", t0);
    let t0 = s.out.len();
    r.melee_once();
    for _ in 0..90 {
        tick(rig, r, &mut s, &still);
    }
    phase(&mut s, rig, "melee", t0);
    // Board the Warthog, drive, shift, get out.
    let t0 = s.out.len();
    let tag = r.objects[0].primary_tag;
    r.force_seat(tag, 0).expect("seat");
    for _ in 0..60 {
        tick(rig, r, &mut s, &still);
    }
    // The session runs far faster than a clock; a boarding takes seconds. Let the decoder catch
    // up the way real time would (the status line's "loading" count drains).
    let wait = Instant::now() + Duration::from_secs(120);
    while rig.eng.status().loading > 0 && Instant::now() < wait {
        tick(rig, r, &mut s, &still);
        std::thread::sleep(Duration::from_millis(5));
    }
    phase(&mut s, rig, "seat", t0);
    let t0 = s.out.len();
    for _ in 0..(10.0 / DT) as usize {
        tick(rig, r, &mut s, &MoveInput { fwd: 1.0, ..Default::default() });
    }
    if let Some(st) = rig.ps.ride.as_ref() {
        for (k, v) in &st.voices {
            println!("  engine voice {k:?}: gain {:.5}/{:.5} ({:.1} dB) rate {:.4} src {} Hz", v.gain[0], v.gain[1], 20.0 * v.gain[0].max(v.gain[1]).max(1e-9).log10(), v.mult, v.rate);
        }
        println!("  engine voices live: {}, last scale/pitch {:?}", st.voices.len(), rig.ps.stats.last);
        for (k, t) in &st.tracks {
            println!("  track {k:?}: on {} fade {:.2} scale {:.3} alt_mix {:.2} cur_range {:?}", t.on, t.fade, t.scale, t.alt_mix, t.cur_range);
        }
        if let Some(vs) = rig.ps.vehicles.get(&tag) {
            for (si, src) in vs.sources.iter().enumerate() {
                for (ti, tr) in src.tracks.iter().enumerate() {
                    for (nm, l) in [("lp", tr.lp.as_ref()), ("alt", tr.alt.as_ref())] {
                        if let Some(l) = l {
                            for (ri, rg) in l.ranges.iter().enumerate() {
                                for &smp in &rg.samples {
                                    println!("  src {si} track {ti} {nm} range {ri} sample {smp}: decoded {}", rig.eng.sample(smp).is_some());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    phase(&mut s, rig, "drive 10 s", t0);
    let t0 = s.out.len();
    let (native, cache) = (r.native.clone(), r.cache);
    let _ = r.pm.exit_vehicle(&native, cache);
    for _ in 0..(4.0 / DT) as usize {
        tick(rig, r, &mut s, &still);
    }
    phase(&mut s, rig, "exit", t0);
    let t0 = s.out.len();
    for _ in 0..(3.0 / DT) as usize {
        tick(rig, r, &mut s, &MoveInput { fwd: 1.0, ..Default::default() });
    }
    phase(&mut s, rig, "walk again", t0);
    s
}

fn print_audit(rig: &Rig, s: &Session) {
    println!("{:<14} {:>8} {:>8} {:>6}   drops: {}", "event", "expected", "started", "%", crate::sound::REFUSAL_NAMES[1..].join(" | "));
    for (i, name) in CAT_NAMES.iter().enumerate() {
        let r = &rig.ps.audit.rows[i];
        let pct = if r.expected > 0 { 100.0 * r.started as f64 / r.expected as f64 } else { f64::NAN };
        println!("{name:<14} {:>8} {:>8} {pct:>5.0}%   {:?}", r.expected, r.started, &r.dropped[1..]);
    }
    println!("status drops {:?}", rig.eng.status().drops);
    for (n, c) in &s.cats {
        println!("meter {n:<12} dBFS: ambience {:>6.1} engine {:>6.1} body {:>6.1} weapon {:>6.1} other {:>6.1}", c[1], c[2], c[3], c[4], c[0]);
    }
    for (n, a, b) in &s.phases {
        let x = &s.out[*a..(*b).min(s.out.len())];
        let (c, pk, sh) = describe(&x[..x.len().min(1 << 16)], RATE as f32);
        println!("phase {n:<12} {:>6.2}..{:<6.2} s  rms {:>6.1} dBFS  peak {:.3}  centroid {c:>6.0} Hz  strongest bin {pk:>6.0} Hz ({sh:.2})", *a as f64 / f64::from(RATE), *b as f64 / f64::from(RATE), 20.0 * rms(x).max(1e-9).log10(), x.iter().fold(0.0f32, |m, v| m.max(v.abs())));
    }
}

fn session_rig() -> Option<(Rig, Ride)> {
    let mut rig = rig()?;
    rig.ps.mute_ambience = false;
    let (native, cache) = (rig.native.clone(), rig.cache);
    let mut r = Ride::with_vehicle(native.clone(), cache, "\\warthog\\warthog").expect("a ride harness");
    let ar = native
        .tags_by_group(cache, "weap")
        .into_iter()
        .find(|&t| native.tag_name(cache, t).map_or(false, |n| n.ends_with("assault_rifle\\assault_rifle")))
        .expect("the assault rifle");
    let msg = r.pm.give_weapon(&native, cache, Some(ar), &mut r.actors, &[], &|_| false);
    println!("give_weapon: {msg}");
    Some((rig, r))
}

/// PRINT-ONLY: the audit table and the session's spectrogram input.
#[test]
#[ignore = "opens the native library; run alone with --ignored"]
fn probe_event_audit() {
    let Some((mut rig, mut r)) = session_rig() else { return };
    let s = run_session(&mut rig, &mut r);
    print_audit(&rig, &s);
    write_wav(&std::path::Path::new(AUDIT_DIR).join("session.wav"), &s.out, RATE);
    println!("{}", rig.ps.caption(&rig.eng));
}

/// First sample of the first sound whose tag name satisfies `want`.
fn first_sample_of(rig: &mut Rig, want: &dyn Fn(&str) -> bool) -> Option<(String, usize)> {
    let (native, cache) = (rig.native.clone(), rig.cache);
    for t in rs::sound_tags(&native, cache) {
        let Some(n) = native.tag_name(cache, t) else { continue };
        if want(&n) {
            if let Some(c) = rig.eng.resolve_one_shot(&native, cache, t) {
                return Some((n, c.samples[0]));
            }
        }
    }
    None
}

/// Decode `idx` (waiting for the worker), render it alone through the mixer at gain `g` and pitch
/// ratio 1.0, and return `(source pcm, output mono)`.
fn render_alone(rig: &mut Rig, idx: usize, g: f32) -> (Arc<hms_sound::Pcm>, Vec<f32>) {
    let deadline = Instant::now() + Duration::from_secs(120);
    let pcm = loop {
        if let Some(p) = rig.eng.sample(idx) {
            break p;
        }
        assert!(Instant::now() < deadline, "sample {idx} never decoded");
        std::thread::sleep(Duration::from_millis(3));
    };
    assert!(rig.eng.play_pcm_for_test(pcm.clone(), [g, g], 1.0));
    let n_out = (pcm.frames() as f64 * f64::from(RATE) / f64::from(pcm.sample_rate)) as usize;
    let mut y = Vec::new();
    // Render in ticks until the voice has certainly ended (the limiter's look-ahead adds 64 frames).
    for _ in 0..(n_out / FRAMES_PER_TICK + 6) {
        y.extend(avg_mono(&rig.eng.render(FRAMES_PER_TICK)));
    }
    (pcm, y)
}

/// **RATE EXACTNESS.** Every class of sound the test mode plays, decoded from the REAL bank and
/// rendered ALONE through the real mixer at pitch ratio 1.0: the output is as long as the sample
/// is at its own rate, and its spectrum is the sample's spectrum (centroid within 3%). A rate bug
/// (an octave-up squeal is a ratio of 2) cannot hide here because nothing in this check comes from
/// the playback formulas: it compares the OUTPUT to the SOURCE.
#[test]
#[ignore = "opens the native library; run alone with --ignored"]
fn every_class_of_sound_renders_at_its_own_rate() {
    let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
    let cases: Vec<(&str, Box<dyn Fn(&str) -> bool>)> = vec![
        ("weapon shot", Box::new(|n| n.ends_with("assault_rifle\\ar_fire"))),
        ("footstep", Box::new(|n| n.ends_with("footsteps\\chief\\dirt\\run"))),
        ("foley", Box::new(|n| n.ends_with("assault_rifle\\ar_ready"))),
        ("ambience loop", Box::new(|n| n.contains("halo_ext") && n.ends_with("\\loop"))),
        ("scenery loop", Box::new(|n| n.contains("light_lapping") && n.ends_with("\\loop"))),
        ("engine loop", Box::new(|n| n.contains("warthog") && n.ends_with("\\track1\\loop"))),
        ("engine start", Box::new(|n| n.contains("warthog") && n.ends_with("\\track1\\in"))),
    ];
    for (label, want) in cases {
        let Some((name, idx)) = first_sample_of(&mut rig, &*want) else {
            panic!("no sound for the class '{label}' in forge_halo");
        };
        // A small gain so the limiter is out of the measurement (these streams peak far above 1).
        let (pcm, y) = render_alone(&mut rig, idx, 0.01);
        let src = mono_of(&pcm);
        let want_len = pcm.frames() as f64 * f64::from(RATE) / f64::from(pcm.sample_rate);
        // Output length = frames until the last audible frame (the voice's own fades are ~4 ms).
        let thresh = 1e-4 * src.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-3);
        // (Not the last 256 frames: the mixer fades a one-shot's tail there, and the band limit's
        // wrap-around puts a trace of the sample's head into them.)
        let body = src.len().saturating_sub(256);
        let src_last = src[..body].iter().rposition(|v| v.abs() * 0.01 > thresh).unwrap_or(0) as f64 * f64::from(RATE) / f64::from(pcm.sample_rate);
        let last = y[..(body + hms_sound::mix::LOOKAHEAD).min(y.len())].iter().rposition(|v| v.abs() > thresh).unwrap_or(0) as f64;
        let take = src.len().min(1 << 16);
        let (cs, _, _) = describe(&src[..take], pcm.sample_rate as f32);
        // The output carries the limiter's 64-frame latency.
        let (co, _, _) = describe(&y[hms_sound::mix::LOOKAHEAD..(hms_sound::mix::LOOKAHEAD + take).min(y.len())], RATE as f32);
        println!("{label:<14} {name}: src {} Hz {} ch {:.3} s ({} frames); out last audible frame {last:.0} (source's {src_last:.0}, ratio {:.4}); centroid src {cs:.0} Hz out {co:.0} Hz (ratio {:.3})", pcm.sample_rate, pcm.channels, want_len / f64::from(RATE), pcm.frames(), last / src_last.max(1.0), co / cs.max(1.0));
        // The coded band ends at ~16.5 kHz; what is above is not content (see `hms_sound::BAND_LIMIT_HZ`).
        let sp = spectrum(&src[..take]);
        let k = (17_500.0 / (pcm.sample_rate as f32 / 2.0) * sp.len() as f32) as usize;
        let tot: f64 = sp.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let hf: f64 = sp[k..].iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let hf_db = 10.0 * (hf / tot.max(1e-30)).max(1e-12).log10();
        println!("    energy above 17.5 kHz: {hf_db:.1} dB below the whole");
        assert!(hf_db < -45.0, "'{label}' carries energy above the encoder's own band: {hf_db:.1} dB re total");
        assert_eq!(pcm.sample_rate, RATE, "'{label}' is not at the mixer's rate: the mixer would resample it");
        let ratio = (last - hms_sound::mix::LOOKAHEAD as f64) / src_last.max(1.0);
        assert!((0.985..=1.015).contains(&ratio), "'{label}' renders {ratio:.3}x its own length");
        assert!((0.97..=1.03).contains(&(co / cs.max(1.0))), "'{label}' spectrum moved: centroid {cs:.0} Hz -> {co:.0} Hz");
    }
}

/// **THE EVENT AUDIT.** A scripted test-mode session on the real forge_halo cache (spawn, walk,
/// run, sprint, jump x3, the assault rifle held for 3 s, reload, melee, board the Warthog, drive
/// 10 s, get out, walk) through the production path and the offline mixer. Every gameplay event
/// that should make a sound is counted when it happens and again when a voice starts. Delivered /
/// expected must be 100% (a sound the TAG authors no clip for is "silent by the tag", counted
/// apart), with nothing dropped as not-decoded, out of range or no voice; every phase must be
/// audible; the output never exceeds full scale.
#[test]
#[ignore = "opens the native library; run alone with --ignored"]
fn the_event_audit_delivers_every_event() {
    let Some((mut rig, mut r)) = session_rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
    let s = run_session(&mut rig, &mut r);
    print_audit(&rig, &s);
    write_wav(&std::path::Path::new(AUDIT_DIR).join("session.wav"), &s.out, RATE);
    let rows = &rig.ps.audit.rows;
    let noclip = crate::sound::Refusal::NoClip as usize;
    let floor = crate::sound::Refusal::BelowFloor as usize;
    // Independent expectations: the harness fired 45 shots, jumped 3 times, swung once, reloaded once.
    assert_eq!(rows[Cat::Fire as usize].expected, 45, "shots seen by the sound");
    assert_eq!(rows[Cat::Jump as usize].expected, 3);
    assert_eq!(rows[Cat::Melee as usize].expected, 1);
    assert_eq!(rows[Cat::Reload as usize].expected, 1);
    assert!(rows[Cat::Step as usize].expected >= 20, "a 12 s walk/run/sprint made {} footstep events", rows[Cat::Step as usize].expected);
    assert!(rows[Cat::Land as usize].expected >= 3, "three jumps, {} landings", rows[Cat::Land as usize].expected);
    assert!(rows[Cat::EngineStart as usize].expected >= 1 && rows[Cat::EngineStop as usize].expected >= 1, "engine start/stop events");
    for (i, name) in CAT_NAMES.iter().enumerate() {
        let r = &rows[i];
        if r.expected == 0 {
            continue;
        }
        let silent_by_tag = r.dropped[noclip] + if i == Cat::Shift as usize { r.dropped[floor] } else { 0 };
        assert_eq!(r.started + silent_by_tag, r.expected, "{name}: {} of {} events made a sound, {} dropped: {:?}", r.started, r.expected, r.expected - r.started - silent_by_tag, r.dropped);
        for (k, rn) in crate::sound::REFUSAL_NAMES.iter().enumerate() {
            if k != noclip && !(i == Cat::Shift as usize && k == floor) && k != 0 {
                assert_eq!(r.dropped[k], 0, "{name}: events dropped as '{rn}'");
            }
        }
    }
    // The level meter sees each category where it should (the meter is what the status line
    // shows, so it has to be right): the gun in the fire phase, the engine while driving, the feet
    // on the move, the ambience always.
    let meter = |name: &str| s.cats.iter().find(|(n, _)| *n == name).map(|(_, c)| *c).expect("phase");
    assert!(meter("fire 3 s")[4] > -45.0, "weapon meter in the fire phase: {:?}", meter("fire 3 s"));
    assert!(meter("drive 10 s")[2] > -55.0, "engine meter while driving: {:?}", meter("drive 10 s"));
    assert!(meter("run")[3] > -75.0, "body meter while running: {:?}", meter("run"));
    assert!(meter("walk again")[1] > -80.0, "ambience meter: {:?}", meter("walk again"));
    // Audibility of every phase, and the output never past full scale.
    for (n, a, b) in &s.phases {
        let x = &s.out[*a..(*b).min(s.out.len())];
        assert!(rms(x) > 1e-3, "phase '{n}' is silent: {:.1} dBFS", 20.0 * rms(x).max(1e-9).log10());
        assert!(x.iter().all(|v| v.abs() <= 1.0), "phase '{n}' clips");
    }
}

/// PRINT-ONLY. Peak level of 150 bank samples spread over the whole bank, after the playback
/// band limit. (Raw, the bank decoded to a median of +9.2 dBFS with a maximum of +41; a clean
/// sample is at or below 0 dBFS.)
#[test]
#[ignore = "opens the native library; run alone with --ignored"]
fn probe_peak_levels_after_the_band_limit() {
    let Some(mut rig) = rig() else { return };
    let total = rig.eng.status().bank_samples;
    let mut peaks: Vec<f32> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(600);
    for k in 0..150usize {
        let idx = k * (total / 150).max(1);
        let pcm = loop {
            if let Some(p) = rig.eng.sample(idx) {
                break Some(p);
            }
            if Instant::now() > deadline {
                break None;
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        if let Some(p) = pcm {
            if p.peak() > 0.0 {
                peaks.push(20.0 * p.peak().log10());
            }
        }
    }
    peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f32| peaks[((peaks.len() - 1) as f32 * f) as usize];
    println!("{} samples: peak dBFS min {:.1} p25 {:.1} median {:.1} p75 {:.1} p95 {:.1} max {:.1}; over full scale: {}", peaks.len(), q(0.0), q(0.25), q(0.5), q(0.75), q(0.95), q(1.0), peaks.iter().filter(|p| **p > 0.0).count());
}

/// PRINT-ONLY. The Warthog's engine loops: pitch span of the scale, and every pitch range's bend /
/// full-gain bounds and natural pitch, against the pitch the engine asks for at each throttle.
#[test]
#[ignore = "opens the native library; run alone with --ignored"]
fn probe_engine_pitch_ranges() {
    let Some(mut rig) = rig() else { return };
    let mut r = Ride::with_vehicle(rig.native.clone(), rig.cache, "\\warthog\\warthog").expect("ride");
    let tag = r.objects[0].primary_tag;
    r.force_seat(tag, 0).unwrap();
    prime(&mut rig, &mut r);
    let vs = &rig.ps.vehicles[&tag];
    for (si, src) in vs.sources.iter().enumerate() {
        for (ti, tr) in src.tracks.iter().enumerate() {
            for (name, l) in [("loop", tr.lp.as_ref()), ("alt", tr.alt.as_ref())] {
                let Some(l) = l else { continue };
                println!("src {si} ({}) track {ti} {name}: all_ranges_at_once {} pitch(0) {:.0} pitch(0.5) {:.0} pitch(1) {:.0}", src.lsnd_name.rsplit('\\').next().unwrap_or(""), l.all_ranges_at_once(), l.pitch_at(0.0), l.pitch_at(0.5), l.pitch_at(1.0));
                for (i, rg) in l.ranges.iter().enumerate() {
                    println!("     range {i}: natural {:.0}  bend {:?}  full {:?}  samples {}", rg.natural_pitch, rg.bend, rg.full, rg.samples.len());
                }
                for sc in [0.0f32, 0.25, 0.5, 0.75, 0.9, 1.0] {
                    let p = l.pitch_at(sc);
                    println!("     scale {sc:.2}: pitch {p:.0} -> ranges {:?}", l.ranges_at(p, None));
                }
            }
        }
    }
}
