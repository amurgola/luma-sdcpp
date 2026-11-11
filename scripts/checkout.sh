#!/usr/bin/env bash
# Bash twin of checkout.ps1, for the Linux builds: clone upstream
# stable-diffusion.cpp at the pinned UPSTREAM_REF into work/ and apply the
# patch stack with git am. Idempotent: re-running resets work/ to a clean
# patched state. On a ref bump, a patch that no longer applies stops here with
# git am's 3-way conflict markers to resolve.
#
#   scripts/checkout.sh [ref]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ref="${1:-$(head -n 1 "$root/UPSTREAM_REF" | tr -d '[:space:]')}"
upstream="${UPSTREAM_URL:-https://github.com/leejet/stable-diffusion.cpp}"
work="$root/work/stable-diffusion.cpp"

if [ ! -d "$work/.git" ]; then
  mkdir -p "$(dirname "$work")"
  git clone "$upstream" "$work"
fi
cd "$work"

if [ -d .git/rebase-apply ]; then git am --abort; fi
git fetch origin
git checkout --detach "$ref"
git clean -fd
# ggml and the webp/webm libraries are submodules; the release build needs the
# pinned copies, not whatever a plain checkout leaves behind.
git submodule update --init --recursive
echo "== upstream @ $(git rev-parse --short HEAD), applying patch stack =="

shopt -s nullglob
patches=("$root"/patches/*.patch)
if [ "${#patches[@]}" -gt 0 ]; then git am --3way "${patches[@]}"; fi
git log --oneline "-$(( ${#patches[@]} + 1 ))"
