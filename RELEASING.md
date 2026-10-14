# Releasing Halo Map Studio

The in-app updater (`crates/hms-app/src/updater.rs`) reads the GitHub Releases
API and installs what it finds there, so a release must follow the contract
below exactly. Everything the updater relies on is listed here.

## 0. Release notes

`CHANGELOG.md` at the repo root is the **running draft**: every change is logged there as it
lands, under the next version's heading. At release time:

1. Read `CHANGELOG.md`, fold its entries into the `docs/changelog.html` entry for this version
   (and re-sync `dist/github-root/docs/`), and set that entry's date to the release date.
2. Paste the same list into the GitHub release body.
3. Clear the folded entries out of `CHANGELOG.md`, leaving the heading for the next version.

Do not write release notes from the commit log: this tree is developed uncommitted, so the log is
not a record of what changed.

## 1. Version

One version, everywhere:

* `Cargo.toml` → `[workspace.package] version = "X.Y.Z"` (the app reads it as
  `CARGO_PKG_VERSION`; the title bar, Help menu and Update window show it).
* `docs/*.html` (header pill "vX.Y.Z docs", footers, `docs/changelog.html`
  entry with the `current` tag, `docs/getting-started.html` archive names),
  `README.md`, `dist/README.md` (shipped in the archives), and the
  `dist/github-root/` copies (`README.md`, `docs/`).

The Git tag is `vX.Y.Z`. The updater accepts `vX.Y.Z` and `X.Y.Z` and compares
semver-wise (`1.10.0 > 1.9.9`; `X.Y.Z-pre` sorts below `X.Y.Z`).

## 2. Build and package

```sh
./build-all.sh            # Linux + Windows (cargo-xwin) release builds
./build-all.sh package    # assembles dist/linux-x64, dist/windows-x64 and the archives
```

### Rebuild the native libraries first — BOTH of them

`./build-all.sh` runs `native/build-linux.sh` for every Linux build, but it only
runs `native/build-windows.sh` when **clang-cl and lld-link are on PATH**;
without them it silently ships whatever `native/build-windows/HaloMapStudioDLL.dll`
happens to be lying around (it prints a `!!` line and carries on). A stale DLL
is a shipped bug, not a build failure, so check it every release:

```sh
command -v clang-cl lld-link                 # both must resolve, or install llvm
( cd native && ./build-windows.sh )          # -> native/build-windows/HaloMapStudioDLL.dll
```

**This is mandatory for any release that adds a native export.** The C++ parser gained
`ZH_MBP_SetSharedCacheDirs` in 1.3.x (it is how the app points the parser at the stock
`shared.map` so Workshop / modded maps decode their shared-paged models — see
the modded-map fixes in the changelog). The Linux `.so` has it; the DLL in the
tree at the time of writing does **not**, and the Windows build will fail to
load modded maps until it is rebuilt. Verify before packaging:

```sh
nm -D --defined-only native/build-linux/libhalomapstudio.so | grep ZH_MBP_SetSharedCacheDirs
strings -a native/build-windows/HaloMapStudioDLL.dll        | grep ZH_MBP_SetSharedCacheDirs
```

Both must print a match. In general: whenever an export is added to the native
library, rebuild **both** targets in the same release, because the Rust side
resolves the symbol at load time on each platform.

`package` writes, into `dist/release/`:

| File | Contents |
|---|---|
| `HaloMapStudio-X.Y.Z-linux-x64.zip` | one top-level folder `HaloMapStudio-X.Y.Z-linux-x64/` with `hms-app`, `libhalomapstudio.so`, `README.md`, `LICENSE.md`, `COMMERCIAL-LICENSE.md`, `THIRD-PARTY-NOTICES.md` |
| `HaloMapStudio-X.Y.Z-windows-x64.zip` | one top-level folder `HaloMapStudio-X.Y.Z-windows-x64/` with `hms-app.exe`, `HaloMapStudioDLL.dll` and the same text files |
| `HaloMapStudio-X.Y.Z-<platform>.zip.sha256` | `sha256sum` format: `<64 hex>  HaloMapStudio-X.Y.Z-<platform>.zip` |

### The asset contract (what the updater looks for)

1. **Archive name:** `HaloMapStudio-<version>-<platform>.zip`, platform =
   `linux-x64` or `windows-x64`, `<version>` = the tag without the `v`. The
   updater looks for exactly this name first. As a fallback it accepts any
   `.zip` asset whose name contains the platform token and not `portable`
   (that is how the pre-1.3 `halo-map-studio-rust-0.1.0-<platform>-<date>.zip`
   releases are still installable), so do not upload a second `.zip` per
   platform unless it carries `portable` in its name.
2. **Archive layout:** a single top-level folder (any name; it is stripped) or
   flat files. Everything in it is installed; the executable must be present
   (`hms-app` / `hms-app.exe`). Deflate or stored entries only (what `zip`
   produces by default; no zip64, no bzip2/zstd).
3. **Checksum:** `<archive>.sha256` next to each archive, or one combined
   `SHA256SUMS` file. Optional but strongly recommended: the updater refuses a
   download whose digest does not match. It always checks the byte size the
   API reports for the asset.
4. **Pre-releases:** the updater calls `/releases/latest`, which GitHub
   resolves to the newest release that is neither a draft nor a pre-release.
   Mark test builds as pre-release and nobody is offered them.
5. **Release notes:** the release body is shown as plain text in the Update
   window (Markdown is not rendered; keep it short, plain lines and bullets).

## 3. Publish

```sh
V=$(grep -m1 '^version = ' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
gh release create "v$V" \
  dist/release/HaloMapStudio-$V-linux-x64.zip \
  dist/release/HaloMapStudio-$V-linux-x64.zip.sha256 \
  dist/release/HaloMapStudio-$V-windows-x64.zip \
  dist/release/HaloMapStudio-$V-windows-x64.zip.sha256 \
  -R Sopitive/HaloMapStudio --title "v$V" --notes-file /path/to/notes.md
```

Then push the docs (`dist/github-root/` = the public repository root; `docs/`
is served by GitHub Pages).

## 4. How the updater installs (per platform)

* **Check:** at startup (4 s after the window is up, on a worker thread), every
  24 h while running, and on *Check now*. Offline / rate limit = no badge, one
  log line. Last check time and last seen version are in
  `<settings>/updates.json` so restarts do not re-hit the API early.
* **Download:** to `<settings>/updates/<asset>`, size + SHA-256 verified, then
  unpacked ("staged") into `<install>/.hms-update-<version>/` — next to the
  program, on the same filesystem, so the last step is a rename, never a copy.
  A read-only install folder fails here, before anything is touched, with an
  "Open download folder" fallback.
* **Install (all platforms, the same code):** the running executable and the
  mapped native library cannot be overwritten in place, but both can be
  *renamed*. Every file the archive replaces is renamed into
  `<install>/.hms-backup/`, every staged file is renamed into place; the first
  failure rolls everything back. Then the NEW executable is started with
  `--resume-after-update <version> --wait-pid <old pid>` and the old process
  closes. The new process waits (up to 60 s) for the old pid to exit before it
  opens a window (single instance), restores the recovery snapshot and shows
  "Updated to X.Y.Z — your session was restored". `.hms-backup` is kept
  through that launch and removed on the next ordinary start.
  * Linux: `rename(2)` on the mmapped `.so` and the running ELF is allowed; the
    old process keeps its mapping until it exits.
  * Windows: renaming a running `.exe` / loaded `.dll` on the same volume is
    allowed (only overwrite and delete are locked) — the classic self-update
    move. `--wait-pid` uses `OpenProcess`/`WaitForSingleObject`.
  * Wine / Proton: detected through ntdll's `wine_get_version` export; the
    Windows archive is used (the UI says so) and the same rename path applies
    (Wine's loader opens images with `FILE_SHARE_DELETE`).
  * **No `\\?\` paths reach a process spawn.** `fs::canonicalize` always returns
    the *verbatim* form (`\\?\C:\HaloMapStudio`); `install_dir()` folds it back
    with `updater::plain_path()` before anything uses it, because
    `CreateProcess`'s `lpCurrentDirectory` is not specified to accept the
    extended-length form — if it is rejected the user is left with no app
    running at all. `spawn_resumed` also retries without a working directory if
    the spawn fails for any reason (the app finds its native library next to
    the exe, not through the cwd), so a relaunch is never lost.

  Verified under `wine-11.18` with a standalone probe (a Windows build of the
  same code paths, run from the prefix's `C:` drive):

  | Check | Result |
  |---|---|
  | `canonicalize().parent()` | `\\?\C:\HaloMapStudio` — the prefix *is* added |
  | spawn with a verbatim cwd | **OK on this Wine** (so the prefix is tolerated here; real Windows is not guaranteed, hence the fold + the fallback) |
  | spawn with a plain cwd | OK |
  | `wine_get_version` probe | fires → `HaloMapStudio-<ver>-windows-x64.zip` chosen |
  | rename a **loaded** `.dll` | OK |
  | rename the **running** `.exe` | OK |
  | overwrite the running `.exe` | refused, `Sharing violation (os error 32)` — this is why `apply()` renames and never writes over a file |

  The GUI itself has **not** been run under Wine or on real Windows for this
  feature; only these mechanics were exercised there.
* **Recovery snapshot** (written before the swap): `<settings>/recovery/<stamp>.mmsproj`
  (the ordinary project record: map, camera, per-object flags), `<stamp>.mvar`
  (an exact copy of the open variant INCLUDING unsaved edits — the user's real
  file is never written) and `<stamp>.recovery.json` (which real file the copy
  stands for, open panels, versions). After the restore, File ▸ Save points at
  the real file again. A snapshot found on an ordinary start (the relaunch did
  not happen) is offered with Restore / Discard.

## 5. QA switch (diagnostic only)

`HMS_UPDATE_FEED=<file or URL>` makes the check read a release JSON of your
own instead of the GitHub API (and checks immediately at startup). Inside it,
`browser_download_url` may be a local path or `file://` URL, so the whole
download → verify → stage → install → restart → restore flow can be run against
a local archive. It is never needed by users and is not read by anything else.

The second diagnostic is the script verb `dialog update <check|open|close|download|auto|install|dismiss|status>`
(editor command executor, so it works over the TCP command server on
`HMS_CMD_PORT`, default 47800). It presses the Update window's own buttons and
prints one state line — nothing it does is unreachable with a mouse. Together
with `dialog shot <path>` the whole flow runs unattended:

```sh
# in one terminal: an isolated install + settings dir, pointed at the QA feed
XDG_DATA_HOME=/tmp/upd/xdg HMS_UPDATE_FEED=/tmp/feed.json HMS_CMD_PORT=47811 /tmp/upd/install/hms-app
# in another: check -> badge -> window -> download -> install -> restart -> restore
#   dialog update check | dialog update open | dialog update download | dialog update install
```

Both switches are **diagnostics**: with neither set (how users run the app) the
updater talks to the real GitHub API and every control is the GUI's own.

```sh
# fake a 9.9.9 release out of the current build
./build-all.sh package
cat > /tmp/feed.json <<EOF
{"tag_name":"v9.9.9","name":"QA","body":"test release","html_url":"https://example","prerelease":false,
 "assets":[{"name":"HaloMapStudio-9.9.9-linux-x64.zip","size":$(stat -c %s dist/release/HaloMapStudio-*-linux-x64.zip),
            "browser_download_url":"$PWD/dist/release/$(basename dist/release/HaloMapStudio-*-linux-x64.zip)"}]}
EOF
HMS_UPDATE_FEED=/tmp/feed.json /path/to/install/hms-app
```
