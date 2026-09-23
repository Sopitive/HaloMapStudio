#!/usr/bin/env bash
# Build Halo Map Studio for BOTH platforms from Linux.
#
#   ./build-all.sh              both targets, release
#   ./build-all.sh linux        Linux only
#   ./build-all.sh windows      Windows (MSVC ABI) only
#   ./build-all.sh --debug ...  debug profile
#
# Windows output is genuine MSVC ABI (via cargo-xwin). Release builds are the
# offline editor: the optional `injection` cargo feature is NOT enabled here.
set -euo pipefail
cd "$(dirname "$0")"
PROFILE=release; PROFILE_FLAG=--release
[ "${1:-}" = "--debug" ] && { PROFILE=debug; PROFILE_FLAG=; shift; }
WHAT="${1:-both}"

# License files ship inside every dist folder (PolyForm Noncommercial + commercial notice).
copy_license() { for d in dist/linux-x64 dist/windows-x64; do [ -d "$d" ] && cp LICENSE.md COMMERCIAL-LICENSE.md THIRD-PARTY-NOTICES.md "$d/" && cp dist/README.md "$d/README.md"; done; }

# Replace a file ATOMICALLY (write next to it, then rename). A plain `cp` rewrites the
# existing inode in place, and a running hms-app has libhalomapstudio.so mmapped -> its
# code pages change under it -> SIGSEGV/SIGBUS.
install_atomic() { local src="$1" dst="$2"; cp "$src" "$dst.tmp.$$" && mv -f "$dst.tmp.$$" "$dst"; }

build_linux() {
  echo "==> native: libhalomapstudio.so"
  ( cd native && ./build-linux.sh )
  mkdir -p "target/$PROFILE"
  install_atomic native/build-linux/libhalomapstudio.so "target/$PROFILE/libhalomapstudio.so"
  echo "==> rust: linux"
  cargo build $PROFILE_FLAG -p hms-app
  install_atomic native/build-linux/libhalomapstudio.so "target/$PROFILE/libhalomapstudio.so"
  echo "    -> target/$PROFILE/hms-app"
}

build_windows() {
  command -v cargo-xwin >/dev/null || { echo "!! cargo-xwin missing: paru -S cargo-xwin"; exit 1; }
  echo "==> rust: windows-msvc"
  cargo xwin build $PROFILE_FLAG --target x86_64-pc-windows-msvc -p hms-app
  D="target/x86_64-pc-windows-msvc/$PROFILE"
  # The C++ map parser DLL is cross-built by native/build-windows.sh (clang-cl +
  # lld-link against the cargo-xwin MSVC/SDK cache); ship it next to the exe.
  if command -v clang-cl >/dev/null && command -v lld-link >/dev/null; then
    echo "==> native: HaloMapStudioDLL.dll"
    ( cd native && ./build-windows.sh )
    install_atomic native/build-windows/HaloMapStudioDLL.dll "$D/HaloMapStudioDLL.dll"
  elif [ -f native/build-windows/HaloMapStudioDLL.dll ]; then
    echo "!! clang-cl/lld-link missing: shipping the previously built native/build-windows/HaloMapStudioDLL.dll"
    install_atomic native/build-windows/HaloMapStudioDLL.dll "$D/HaloMapStudioDLL.dll"
  else
    echo "!! clang-cl/lld-link missing and no prebuilt DLL: install llvm (clang-cl, lld) and rerun"; exit 1
  fi
  echo "    -> $D/hms-app.exe (+ HaloMapStudioDLL.dll)"
}

case "$WHAT" in
  linux)   build_linux ;;
  windows) build_windows ;;
  both)    build_linux; build_windows ;;
  *) echo "usage: $0 [--debug] [linux|windows|both]"; exit 1 ;;
esac
copy_license
echo "== done =="
