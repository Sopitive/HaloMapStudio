//! #icon: embed the Windows application icon so Explorer, the taskbar and Alt-Tab show it on the
//! .exe itself (the egui `with_icon` call only dresses the live window). Pure no-op everywhere else.
//!
//! Written by hand rather than via the `winres` crate so the cross-compile from Linux
//! (x86_64-pc-windows-gnu) works with the mingw `windres` that is already required to link.
fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target != "windows" {
        return;
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let rc = out.join("icon.rc");
    let ico = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icon.ico");
    if !ico.exists() {
        println!("cargo:warning=assets/icon.ico missing — the exe will have no icon");
        return;
    }
    // IDI_ICON1 = 1 is the id Windows uses for an executable's display icon.
    std::fs::write(&rc, format!("1 ICON \"{}\"\n", ico.display().to_string().replace('\\', "\\\\"))).unwrap();

    // Prefer the GNU toolchain's windres; fall back to llvm-rc / rc.exe for MSVC hosts.
    let obj = out.join("icon.o");
    let tried = [
        ("x86_64-w64-mingw32-windres", vec!["-i".into(), rc.display().to_string(), "-o".into(), obj.display().to_string()]),
        ("windres", vec!["-i".into(), rc.display().to_string(), "-o".into(), obj.display().to_string()]),
    ];
    for (tool, args) in tried {
        match std::process::Command::new(tool).args(&args).status() {
            Ok(st) if st.success() => {
                println!("cargo:rustc-link-arg-bins={}", obj.display());
                return;
            }
            _ => continue,
        }
    }
    println!("cargo:warning=no windres found — the .exe will have no embedded icon");
}
