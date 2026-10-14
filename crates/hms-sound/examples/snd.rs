//! `snd` — inspect and export Reach's FMOD sound banks from the command line.
//!
//! This exists so every claim about the audio can be reproduced and checked
//! with files and numbers instead of by listening.
//!
//! ```text
//!   cargo run --profile quick -p hms-sound --example snd -- \
//!       --reach <MCC>/haloreach <verb> [args]
//!
//!   search <text>        asset paths containing <text>, with sample index
//!   stats  <index|path>  decode and print frames / peak / RMS / envelope
//!   ogg    <index|path> <out.ogg>   re-frame as a standard Ogg Vorbis file
//!   wav    <index|path> <out.wav>   decode in process to 32-bit float WAV
//!   banks                every bank beside the module, with its sample count
//! ```
//!
//! `--bank <stem>` picks a bank other than `sfx` (e.g. `english`).

use anyhow::{bail, Context, Result};
use hms_sound::{SetupTable, SoundBank};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut reach: Option<PathBuf> = None;
    let mut stem = "sfx".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--reach" => {
                reach = args.get(i + 1).map(PathBuf::from);
                args.drain(i..=i + 1.min(args.len() - 1 - i));
            }
            "--bank" => {
                stem = args.get(i + 1).cloned().unwrap_or(stem);
                args.drain(i..=i + 1.min(args.len() - 1 - i));
            }
            _ => i += 1,
        }
    }
    let reach = reach.context("pass --reach <path to the haloreach folder>")?;
    let dir = SoundBank::bank_dir_for_reach(&reach)
        .with_context(|| format!("no fmod/pc under {}", reach.display()))?;

    let verb = args.first().cloned().unwrap_or_default();
    if verb == "banks" {
        for e in std::fs::read_dir(&dir)? {
            let p = e?.path();
            if p.extension().and_then(|s| s.to_str()) != Some("fsb") {
                continue;
            }
            let s = p.file_stem().unwrap().to_string_lossy().to_string();
            match SoundBank::open(&dir, &s) {
                Ok(b) => println!(
                    "{:>10} {:>7} samples  codec {:<7} sidecar {}",
                    s,
                    b.bank.samples.len(),
                    b.bank.codec.name(),
                    if b.info.is_some() { "yes" } else { "no" }
                ),
                Err(e) => println!("{s:>10} ERROR {e:#}"),
            }
        }
        return Ok(());
    }

    let bank = SoundBank::open(&dir, &stem)?;
    let resolve = |s: &str| -> Result<usize> {
        if let Ok(n) = s.parse::<usize>() {
            return Ok(n);
        }
        let info = bank.info.as_ref().context("this bank has no sidecar")?;
        let hits = info.search(s);
        match hits.len() {
            0 => bail!("no sample path contains {s:?}"),
            1 => Ok(hits[0].index),
            n => {
                for h in hits.iter().take(10) {
                    eprintln!("  {:>6} {}", h.index, h.path);
                }
                bail!("{n} samples match {s:?}; be more specific or pass an index")
            }
        }
    };
    let setups = || -> Result<SetupTable> {
        // The module is two directories above the banks.
        let m = dir
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("haloreach.dll"))
            .context("cannot find haloreach.dll above the banks")?;
        SetupTable::from_module(&m)
    };

    match verb.as_str() {
        "search" => {
            let needle = args.get(1).context("search <text>")?;
            let info = bank.info.as_ref().context("this bank has no sidecar")?;
            let hits = info.search(needle);
            for h in &hits {
                println!(
                    "{:>6} {:>8} frames {} ch  {}",
                    h.index,
                    bank.bank.samples[h.index].frames,
                    h.channels,
                    h.path
                );
            }
            println!("{} of {}", hits.len(), info.records.len());
        }
        "stats" => {
            let idx = resolve(args.get(1).context("stats <index|path>")?)?;
            let t = setups()?;
            let pcm = bank.decode_sample(idx, &t)?;
            let s = &bank.bank.samples[idx];
            println!("sample      {idx}");
            println!("path        {}", bank.asset_path(idx).unwrap_or("?"));
            println!("declared    {} frames", s.frames);
            println!("decoded     {} frames", pcm.frames());
            println!("rate        {} Hz   channels {}", pcm.sample_rate, pcm.channels);
            println!("duration    {:.4} s", pcm.duration_seconds());
            println!("peak        {:.6}  ({:+.2} dBFS)", pcm.peak(), 20.0 * pcm.peak().log10());
            println!("rms         {:.6}  ({:+.2} dB)", pcm.rms(), 20.0 * pcm.rms().log10());
            let n = (pcm.frames() / 20).max(1);
            let env: Vec<String> = (0..20)
                .map(|k| {
                    let a = k * n;
                    let mut sum = 0.0f64;
                    let mut c = 0usize;
                    for f in a..(a + n).min(pcm.frames()) {
                        for v in pcm.frame(f).unwrap_or(&[]) {
                            sum += (*v as f64) * (*v as f64);
                            c += 1;
                        }
                    }
                    format!("{:.0}", (sum / c.max(1) as f64).sqrt() * 32768.0)
                })
                .collect();
            println!("envelope    {}", env.join(" "));
        }
        "ogg" => {
            let idx = resolve(args.get(1).context("ogg <index|path> <out.ogg>")?)?;
            let out = args.get(2).context("ogg <index|path> <out.ogg>")?;
            let t = setups()?;
            let d = bank.sample_to_ogg(idx, &t)?;
            std::fs::write(out, &d)?;
            println!("{} bytes -> {out}", d.len());
        }
        "wav" => {
            let idx = resolve(args.get(1).context("wav <index|path> <out.wav>")?)?;
            let out = args.get(2).context("wav <index|path> <out.wav>")?;
            let t = setups()?;
            let pcm = bank.decode_sample(idx, &t)?;
            std::fs::write(out, wav_f32(&pcm))?;
            println!("{} frames -> {out}", pcm.frames());
        }
        // Peak/RMS over every `step`th sample. This is how the "FMOD Vorbis
        // streams decode well above full scale" fact was established, so it
        // has to stay reproducible.
        "survey" => {
            let step: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(101);
            let t = setups()?;
            let mut peaks: Vec<(f32, usize)> = Vec::new();
            let mut over = 0usize;
            let mut failed = 0usize;
            for idx in (0..bank.bank.samples.len()).step_by(step.max(1)) {
                if bank.bank.samples[idx].frames == 0 || bank.bank.samples[idx].frames > 480_000 {
                    continue;
                }
                match bank.decode_sample(idx, &t) {
                    Ok(p) => {
                        let pk = p.peak();
                        if pk > 1.0 {
                            over += 1;
                        }
                        peaks.push((pk, idx));
                    }
                    Err(e) => {
                        failed += 1;
                        eprintln!("sample {idx}: {e:#}");
                    }
                }
            }
            peaks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let n = peaks.len();
            println!("decoded {n} samples (step {step}), {failed} failed");
            println!("peaks over full scale: {over} ({:.1}%)", 100.0 * over as f64 / n as f64);
            for (label, q) in [("min", 0.0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p95", 0.95), ("max", 1.0)] {
                let (pk, idx) = peaks[((n - 1) as f64 * q) as usize];
                println!(
                    "  {label:>6} {pk:9.4} ({:+6.2} dBFS)  sample {idx} {}",
                    20.0 * pk.log10(),
                    bank.asset_path(idx).unwrap_or("?")
                );
            }
        }
        // Does the gestalt's per-permutation `encoded gain:dB` explain why
        // the Vorbis streams decode above full scale? Pairs each permutation
        // found in a raw cache image with the peak of the sample it names.
        "gaincheck" => {
            let map = args.get(1).context("gaincheck <map.map> [limit]")?;
            let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
            let cache = std::fs::read(map)?;
            let info = bank.info.as_ref().context("no sidecar")?;
            let known = |v: u32| info.index_of_subsound(v).is_some();
            let hits = hms_sound::scan_cache_for_subsound_ids(&cache, &known, 50);
            println!("{} permutation entries in {map}", hits.len());
            let t = setups()?;
            let step = (hits.len() / limit.max(1)).max(1);
            let mut rows: Vec<(i8, f32, usize)> = Vec::new();
            for (entry, hash, _frames) in hits.iter().step_by(step) {
                let Some(i) = info.index_of_subsound(*hash) else { continue };
                if bank.bank.samples[i].frames == 0 || bank.bank.samples[i].frames > 480_000 {
                    continue;
                }
                let gain = cache[entry + 0x0E] as i8;
                let Ok(p) = bank.decode_sample(i, &t) else { continue };
                rows.push((gain, p.peak(), i));
            }
            let mut by_gain: std::collections::BTreeMap<i8, Vec<f32>> = Default::default();
            for (g, pk, _) in &rows {
                by_gain.entry(*g).or_default().push(*pk);
            }
            println!("{} sampled.  encoded gain -> peak (dBFS)", rows.len());
            for (g, v) in &by_gain {
                let mut v = v.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = v[v.len() / 2];
                println!(
                    "  gain {g:>4}  n {:>4}  median peak {med:8.4} ({:+6.2} dBFS)  min {:+6.2}  max {:+6.2}",
                    v.len(),
                    20.0 * med.log10(),
                    20.0 * v[0].log10(),
                    20.0 * v[v.len() - 1].log10()
                );
            }
        }
        other => bail!(
            "unknown verb {other:?}; try banks / search / stats / ogg / wav / survey / gaincheck"
        ),
    }
    Ok(())
}

/// A 32-bit-float WAV, so nothing is clipped on the way out.
fn wav_f32(p: &hms_sound::Pcm) -> Vec<u8> {
    let ch = p.channels as u16;
    let rate = p.sample_rate;
    let bytes = (p.samples.len() * 4) as u32;
    let mut v = Vec::with_capacity(44 + bytes as usize);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + bytes).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    v.extend_from_slice(&ch.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&(rate * ch as u32 * 4).to_le_bytes());
    v.extend_from_slice(&(ch * 4).to_le_bytes());
    v.extend_from_slice(&32u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&bytes.to_le_bytes());
    for s in &p.samples {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}
