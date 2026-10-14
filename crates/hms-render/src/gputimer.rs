//! Per-pass GPU timing with wgpu timestamp queries. MEASUREMENT ONLY — the timer is `None`
//! unless something explicitly asks for it (the in-app graphics benchmark), so the normal frame
//! path records nothing and allocates nothing.
//!
//! Why this exists: before this module the only performance evidence in the codebase was CPU
//! stage timers (`HMS_FRAMEPROF` for `update()`, `HMS_RENDER_PROF` for the `render()` prologue).
//! Both measure the *encode* cost, not the GPU cost — and the renderer's heavy items (a 2×
//! supersampled scene pass, eleven unconditional bloom passes, a full-resolution depth blit)
//! cost almost nothing to encode. Attributing GPU milliseconds to named passes is the only way
//! to rank them.
//!
//! Usage from `render()`:
//!   * `begin_frame()` once at the top (resets the slot cursor),
//!   * `pass("name")` / `compute("name")` in each pass descriptor's `timestamp_writes`,
//!   * `resolve(&mut encoder)` just before `queue.submit`,
//!   * `collect(device)` after submit — this BLOCKS on the GPU, which is exactly what a
//!     benchmark wants (it also makes the per-pass numbers free of cross-frame overlap).
//!
//! `report()` returns the per-pass mean in milliseconds over every collected frame.

use eframe::wgpu;
use std::cell::{Cell, RefCell};

/// Query slots. Two per pass; the renderer's worst frame is ~20 passes (11 of them bloom).
const CAP: u32 = 96;

pub struct GpuTimer {
    set: wgpu::QuerySet,
    /// `resolve_query_set` destination (QUERY_RESOLVE | COPY_SRC).
    resolve_buf: wgpu::Buffer,
    /// Host-visible copy of `resolve_buf`.
    read_buf: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
    /// Next free query index for this frame.
    next: Cell<u32>,
    /// Pass label per slot pair, this frame.
    labels: RefCell<Vec<&'static str>>,
    /// Accumulated milliseconds per label, over every collected frame, in first-seen order.
    acc: RefCell<Vec<(&'static str, f64)>>,
    /// Per-frame totals (ms) for the mean/p99 of the WHOLE GPU frame.
    frame_ms: RefCell<Vec<f64>>,
}

impl GpuTimer {
    /// `None` when the adapter/device does not expose `TIMESTAMP_QUERY` (the feature is
    /// requested masked, so this is the normal outcome on a driver without it).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("hms-gpu-timer"),
            ty: wgpu::QueryType::Timestamp,
            count: CAP,
        });
        let bytes = (CAP as u64) * 8;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hms-gpu-timer-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hms-gpu-timer-read"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            set,
            resolve_buf,
            read_buf,
            period_ns: queue.get_timestamp_period() as f64,
            next: Cell::new(0),
            labels: RefCell::new(Vec::new()),
            acc: RefCell::new(Vec::new()),
            frame_ms: RefCell::new(Vec::new()),
        })
    }

    pub fn begin_frame(&self) {
        self.next.set(0);
        self.labels.borrow_mut().clear();
    }

    /// Claim a slot pair. Returns `None` once `CAP` is exhausted so an over-long frame degrades
    /// to "untimed tail" instead of panicking inside wgpu validation.
    fn claim(&self, label: &'static str) -> Option<(u32, u32)> {
        let i = self.next.get();
        if i + 2 > CAP {
            return None;
        }
        self.next.set(i + 2);
        self.labels.borrow_mut().push(label);
        Some((i, i + 1))
    }

    pub fn pass(&self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (a, b) = self.claim(label)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(a),
            end_of_pass_write_index: Some(b),
        })
    }

    pub fn compute(&self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (a, b) = self.claim(label)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(a),
            end_of_pass_write_index: Some(b),
        })
    }

    /// Resolve the claimed queries into the readback buffer. Call once per frame, after the last
    /// pass and before `queue.submit`.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let n = self.next.get();
        if n == 0 {
            return;
        }
        encoder.resolve_query_set(&self.set, 0..n, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.read_buf, 0, (n as u64) * 8);
    }

    /// Read the resolved timestamps back, blocking until the frame has executed, and accumulate
    /// the per-pass means. Blocking is intentional: a benchmark wants the GPU drained so the
    /// numbers belong to the frame just submitted.
    pub fn collect(&self, device: &wgpu::Device) {
        let n = self.next.get();
        if n == 0 {
            return;
        }
        let bytes = (n as u64) * 8;
        let slice = self.read_buf.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        if !matches!(rx.recv(), Ok(Ok(()))) {
            return;
        }
        let ticks: Vec<u64> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        self.read_buf.unmap();
        let labels = self.labels.borrow();
        let mut acc = self.acc.borrow_mut();
        let mut frame_total = 0.0f64;
        for (k, label) in labels.iter().enumerate() {
            let (a, b) = (ticks[k * 2], ticks[k * 2 + 1]);
            if b <= a {
                continue; // a driver that did not write the pair
            }
            let ms = (b - a) as f64 * self.period_ns / 1.0e6;
            frame_total += ms;
            match acc.iter_mut().find(|(l, _)| *l == *label) {
                Some(e) => e.1 += ms,
                None => acc.push((label, ms)),
            }
        }
        self.frame_ms.borrow_mut().push(frame_total);
    }

    pub fn frames(&self) -> usize {
        self.frame_ms.borrow().len()
    }

    /// Per-pass mean milliseconds over every collected frame, biggest first.
    pub fn report(&self) -> Vec<(String, f64)> {
        let n = self.frames().max(1) as f64;
        let mut v: Vec<(String, f64)> = self.acc.borrow().iter().map(|(l, ms)| ((*l).to_string(), ms / n)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// (mean, p99, max) of the summed per-pass GPU time per frame, in milliseconds.
    pub fn frame_stats(&self) -> (f64, f64, f64) {
        let mut v = self.frame_ms.borrow().clone();
        if v.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let p99 = v[((v.len() as f64 * 0.99) as usize).min(v.len() - 1)];
        (mean, p99, *v.last().unwrap())
    }

    pub fn reset_stats(&self) {
        self.acc.borrow_mut().clear();
        self.frame_ms.borrow_mut().clear();
    }
}
