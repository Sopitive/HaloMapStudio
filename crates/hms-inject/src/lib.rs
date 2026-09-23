//! hms-inject — inject / cleanly evict HaloMapStudioDLL in the MCC process.
//!
//! Optional, Windows-only (hms-app `injection` cargo feature; the default
//! build never touches the game). MCC is a Windows process, so on non-Windows
//! targets this crate compiles to an empty crate rather than a build error.
//!
//!   * Injection is not gated on host status — a client can inject too.
//!   * `reinject_fresh` does the clean swap: remote-call `ZH_MMP_PrepareUnload`
//!     (detaches hooks), FreeLibrary the old copy out of the game, poll until it
//!     is fully gone, then inject the new one — old out before new in.
//!
//! The remote export resolve walks the REMOTE module's PE export table via
//! ReadProcessMemory, because the game's DLL copy sits at a different base and
//! may be an older build with different export RVAs than any local copy.
#![cfg(windows)]

use std::ffi::c_void;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use windows::core::PCSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Module32FirstW, Module32NextW, Process32FirstW, Process32NextW,
    MODULEENTRY32W, PROCESSENTRY32W, TH32CS_SNAPMODULE, TH32CS_SNAPMODULE32, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
use windows::Win32::System::Memory::{
    VirtualAllocEx, VirtualFreeEx, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
};
use windows::Win32::System::Threading::{
    CreateRemoteThread, OpenProcess, WaitForSingleObject, LPTHREAD_START_ROUTINE,
    PROCESS_ALL_ACCESS,
};
use windows::Win32::System::Diagnostics::Debug::{ReadProcessMemory, WriteProcessMemory};

/// RAII wrapper so process handles always get closed.
struct OwnedHandle(HANDLE);
impl Drop for OwnedHandle {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            unsafe { let _ = CloseHandle(self.0); }
        }
    }
}

/// Find the first process id whose image name equals `name` (case-insensitive,
/// e.g. "MCC-Win64-Shipping.exe").
pub fn find_process_by_name(name: &str) -> Option<u32> {
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0).ok()?;
        let snap = OwnedHandle(snap);
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        if Process32FirstW(snap.0, &mut entry).is_err() {
            return None;
        }
        loop {
            let exe = wstr_to_string(&entry.szExeFile);
            if exe.eq_ignore_ascii_case(name) {
                return Some(entry.th32ProcessID);
            }
            if Process32NextW(snap.0, &mut entry).is_err() {
                break;
            }
        }
        None
    }
}

/// Base address of `module_name` within process `pid`, or None if not loaded.
/// Uses a fresh ToolHelp snapshot each call (so it reflects live load/unload).
pub fn remote_module_base(pid: u32, module_name: &str) -> Option<usize> {
    unsafe {
        // TH32CS_SNAPMODULE32 too, so a 64-bit tool sees 64-bit modules reliably.
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid).ok()?;
        let snap = OwnedHandle(snap);
        let mut entry = MODULEENTRY32W {
            dwSize: std::mem::size_of::<MODULEENTRY32W>() as u32,
            ..Default::default()
        };
        if Module32FirstW(snap.0, &mut entry).is_err() {
            return None;
        }
        loop {
            let m = wstr_to_string(&entry.szModule);
            if m.eq_ignore_ascii_case(module_name) {
                return Some(entry.modBaseAddr as usize);
            }
            if Module32NextW(snap.0, &mut entry).is_err() {
                break;
            }
        }
        None
    }
}

fn open_process(pid: u32) -> Result<OwnedHandle> {
    let h = unsafe { OpenProcess(PROCESS_ALL_ACCESS, false, pid) }
        .context("OpenProcess failed")?;
    Ok(OwnedHandle(h))
}

/// kernel32 is mapped at the same base in every process of a session, so a
/// local GetProcAddress on it yields an address valid in the remote process.
fn kernel32_proc(name: &str) -> Result<usize> {
    unsafe {
        let k32 = GetModuleHandleA(PCSTR(b"kernel32.dll\0".as_ptr()))
            .context("GetModuleHandleA(kernel32) failed")?;
        let cname = std::ffi::CString::new(name).unwrap();
        let p = GetProcAddress(k32, PCSTR(cname.as_ptr() as *const u8));
        match p {
            Some(f) => Ok(f as usize),
            None => bail!("GetProcAddress({name}) failed"),
        }
    }
}

/// Run `start_addr(param)` on a new thread in the target and wait for it.
/// Returns the thread's exit code (low 32 bits of the routine's return).
fn remote_call(hproc: HANDLE, start_addr: usize, param: usize, wait_ms: u32) -> Result<()> {
    unsafe {
        let routine: LPTHREAD_START_ROUTINE = std::mem::transmute(start_addr);
        let thread = CreateRemoteThread(
            hproc,
            None,
            0,
            routine,
            Some(param as *const c_void),
            0,
            None,
        )
        .context("CreateRemoteThread failed")?;
        let th = OwnedHandle(thread);
        WaitForSingleObject(th.0, wait_ms);
        Ok(())
    }
}

fn rpm(hproc: HANDLE, addr: usize, buf: &mut [u8]) -> bool {
    let mut read = 0usize;
    unsafe {
        ReadProcessMemory(
            hproc,
            addr as *const c_void,
            buf.as_mut_ptr() as *mut c_void,
            buf.len(),
            Some(&mut read),
        )
        .is_ok()
            && read == buf.len()
    }
}

fn rpm_u32(hproc: HANDLE, addr: usize) -> Option<u32> {
    let mut b = [0u8; 4];
    rpm(hproc, addr, &mut b).then(|| u32::from_le_bytes(b))
}
fn rpm_u16(hproc: HANDLE, addr: usize) -> Option<u16> {
    let mut b = [0u8; 2];
    rpm(hproc, addr, &mut b).then(|| u16::from_le_bytes(b))
}
fn rpm_asciiz(hproc: HANDLE, addr: usize, max: usize) -> String {
    let mut s = String::new();
    let mut b = [0u8; 1];
    for i in 0..max {
        if !rpm(hproc, addr + i, &mut b) || b[0] == 0 {
            break;
        }
        s.push(b[0] as char);
    }
    s
}

/// Resolve an exported function's address inside a remote module by walking its
/// PE export table via ReadProcessMemory (PE32+/x64 layout).
fn remote_export_addr(hproc: HANDLE, mod_base: usize, export: &str) -> Option<usize> {
    let e_lfanew = rpm_u32(hproc, mod_base + 0x3C)? as usize;
    let nt = mod_base + e_lfanew;
    // PE32+: OptionalHeader at nt+0x18; DataDirectory[0] (export) VA at +0x70 => nt+0x88.
    let exp_rva = rpm_u32(hproc, nt + 0x88)? as usize;
    if exp_rva == 0 {
        return None;
    }
    let exp = mod_base + exp_rva;
    let num_names = rpm_u32(hproc, exp + 0x18)?;
    let funcs_rva = rpm_u32(hproc, exp + 0x1C)? as usize;
    let names_rva = rpm_u32(hproc, exp + 0x20)? as usize;
    let ords_rva = rpm_u32(hproc, exp + 0x24)? as usize;
    if num_names == 0 || num_names > 8192 {
        return None;
    }
    for i in 0..num_names as usize {
        let name_rva = rpm_u32(hproc, mod_base + names_rva + i * 4)? as usize;
        if rpm_asciiz(hproc, mod_base + name_rva, 256) != export {
            continue;
        }
        let ord = rpm_u16(hproc, mod_base + ords_rva + i * 2)? as usize;
        let func_rva = rpm_u32(hproc, mod_base + funcs_rva + ord * 4)? as usize;
        return Some(mod_base + func_rva);
    }
    None
}

/// Inject `dll_path` into `pid` via CreateRemoteThread(LoadLibraryW). No-op-ish
/// if the DLL is already loaded (LoadLibraryW just bumps the refcount).
pub fn inject(pid: u32, dll_path: &Path) -> Result<()> {
    if !dll_path.exists() {
        bail!("DLL not found: {}", dll_path.display());
    }
    let hproc = open_process(pid)?;
    // Wide, NUL-terminated absolute path.
    let full = std::fs::canonicalize(dll_path)
        .unwrap_or_else(|_| dll_path.to_path_buf());
    let mut wpath: Vec<u16> = full.to_string_lossy().encode_utf16().collect();
    wpath.push(0);
    let byte_len = wpath.len() * 2;

    unsafe {
        let remote = VirtualAllocEx(
            hproc.0,
            None,
            byte_len,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );
        if remote.is_null() {
            bail!("VirtualAllocEx failed");
        }
        let mut written = 0usize;
        WriteProcessMemory(
            hproc.0,
            remote,
            wpath.as_ptr() as *const c_void,
            byte_len,
            Some(&mut written),
        )
        .context("WriteProcessMemory failed")?;

        let load_lib = kernel32_proc("LoadLibraryW")?;
        let res = remote_call(hproc.0, load_lib, remote as usize, 10_000);
        let _ = VirtualFreeEx(hproc.0, remote, 0, MEM_RELEASE);
        res
    }
}

/// Game-side HOT RELOAD: evict the old injected DLL cleanly, then inject the new
/// one. Old out before new in.
pub fn reinject_fresh(process_name: &str, dll_path: &Path) -> Result<()> {
    if !dll_path.exists() {
        bail!("DLL not found: {}", dll_path.display());
    }
    let pid = find_process_by_name(process_name)
        .ok_or_else(|| anyhow!("process not found: {process_name}"))?;
    let dll_name = dll_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("bad dll path"))?
        .to_string();

    {
        let hproc = open_process(pid)?;
        if let Some(mod_base) = remote_module_base(pid, &dll_name) {
            // 1. Clean detach FIRST (hooks off, handles drained) so no game
            //    thread is in our code when we unload.
            let prepare = remote_export_addr(hproc.0, mod_base, "ZH_MMP_PrepareUnload")
                .ok_or_else(|| {
                    anyhow!("ZH_MMP_PrepareUnload not found in game module; refusing unsafe unload")
                })?;
            remote_call(hproc.0, prepare, 0, 15_000)?;

            // 2. FreeLibrary out of the game (loop for refcount > 1).
            let free_lib = kernel32_proc("FreeLibrary")?;
            for _ in 0..8 {
                match remote_module_base(pid, &dll_name) {
                    None => break,
                    Some(cur) => {
                        remote_call(hproc.0, free_lib, cur, 10_000)?;
                        std::thread::sleep(std::time::Duration::from_millis(60));
                    }
                }
            }

            // 3. Verify fully gone before injecting — old out before new in.
            for _ in 0..40 {
                if remote_module_base(pid, &dll_name).is_none() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if remote_module_base(pid, &dll_name).is_some() {
                bail!("old DLL still loaded after FreeLibrary (pinned / extra refcount)");
            }
        }
    }

    // 4. Old copy is clear (or was never present) — inject the fresh DLL.
    inject(pid, dll_path)
}


fn wstr_to_string(w: &[u16]) -> String {
    let end = w.iter().position(|&c| c == 0).unwrap_or(w.len());
    String::from_utf16_lossy(&w[..end])
}

/// Convenience: MCC's process image name.
pub const MCC_PROCESS: &str = "MCC-Win64-Shipping.exe";
