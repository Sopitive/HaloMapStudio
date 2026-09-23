#!/usr/bin/env bash
# Install the freshly built libhalomapstudio.so next to the Rust binaries ATOMICALLY.
# Never `cp` over target/release/libhalomapstudio.so: a running hms-app has it mmapped and
# an in-place rewrite crashes it (SIGSEGV/SIGBUS). Rename swaps the directory entry instead.
set -euo pipefail
cd "$(dirname "$0")"
for d in ../target/release ../target/debug; do
  [ -d "$d" ] || continue
  cp build-linux/libhalomapstudio.so "$d/libhalomapstudio.so.tmp.$$" && mv -f "$d/libhalomapstudio.so.tmp.$$" "$d/libhalomapstudio.so"
  echo "installed -> $d/libhalomapstudio.so"
done
