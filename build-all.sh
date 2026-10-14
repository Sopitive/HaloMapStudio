#!/usr/bin/env bash
# Build Halo Map Studio for BOTH platforms from Linux.
#
#   ./build-all.sh              both targets, release
#   ./build-all.sh linux        Linux only
#   ./build-all.sh windows      Windows (MSVC ABI) only
#   ./build-all.sh --debug ...  debug profile
#   ./build-all.sh package      assemble dist/linux-x64 + dist/windows-x64 from the built targets and
#                               write the RELEASE ARCHIVES the in-app updater downloads:
#                                 dist/release/HaloMapStudio-<version>-linux-x64.zip   (+ .sha256)
#                                 dist/release/HaloMapStudio-<version>-windows-x64.zip (+ .sha256)
#                               <version> = [workspace.package].version in Cargo.toml; the archive
#                               holds one top-level folder of the same name. See RELEASING.md.
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

# The release-archive contract (RELEASING.md): HaloMapStudio-<version>-<platform>.zip with one
# top-level folder of the same name, plus a sha256sum-format <zip>.sha256 next to it. The updater
# (crates/hms-app/src/updater.rs) picks the asset by exactly this name, checks the size the GitHub
# API reports and the .sha256 digest, then unpacks the folder over the install.
package_release() {
  VER=$(grep -m1 '^version = ' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
  [ -n "$VER" ] || { echo "!! cannot read the workspace version from Cargo.toml"; exit 1; }
  command -v zip >/dev/null || { echo "!! zip missing"; exit 1; }
  mkdir -p dist/linux-x64 dist/windows-x64 dist/release
  if [ -f "target/$PROFILE/hms-app" ]; then
    install_atomic "target/$PROFILE/hms-app" dist/linux-x64/hms-app
    install_atomic native/build-linux/libhalomapstudio.so dist/linux-x64/libhalomapstudio.so
    chmod +x dist/linux-x64/hms-app
  fi
  if [ -f "target/x86_64-pc-windows-msvc/$PROFILE/hms-app.exe" ]; then
    install_atomic "target/x86_64-pc-windows-msvc/$PROFILE/hms-app.exe" dist/windows-x64/hms-app.exe
    install_atomic native/build-windows/HaloMapStudioDLL.dll dist/windows-x64/HaloMapStudioDLL.dll
  fi
  copy_license
  for plat in linux-x64 windows-x64; do
    exe=hms-app; [ "$plat" = windows-x64 ] && exe=hms-app.exe
    [ -f "dist/$plat/$exe" ] || { echo "-- $plat: no $exe built, skipping"; continue; }
    name="HaloMapStudio-$VER-$plat"
    tmp=$(mktemp -d "dist/release/.pkg-XXXXXX")
    mkdir -p "$tmp/$name"
    # only the shipped file set (dist/<plat>/ may hold stray logs from test runs)
    for f in "$exe" libhalomapstudio.so HaloMapStudioDLL.dll README.md LICENSE.md COMMERCIAL-LICENSE.md THIRD-PARTY-NOTICES.md; do
      [ -f "dist/$plat/$f" ] && cp "dist/$plat/$f" "$tmp/$name/"
    done
    ( cd "$tmp" && zip -X -q -r "../$name.zip.tmp" "$name" )
    mv -f "dist/release/$name.zip.tmp" "dist/release/$name.zip"
    rm -rf "$tmp"
    ( cd dist/release && sha256sum "$name.zip" > "$name.zip.sha256" )
    echo "    -> dist/release/$name.zip ($(du -h "dist/release/$name.zip" | cut -f1)) + .sha256"
  done
}

case "$WHAT" in
  linux)   build_linux ;;
  windows) build_windows ;;
  both)    build_linux; build_windows ;;
  package) package_release; echo "== done =="; exit 0 ;;
  *) echo "usage: $0 [--debug] [linux|windows|both|package]"; exit 1 ;;
esac
copy_license
echo "== done =="
