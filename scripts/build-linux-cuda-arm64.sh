#!/usr/bin/env bash
# Build sd-server + sd-cli (CUDA 13, Linux arm64) from work/stable-diffusion.cpp
# natively on an aarch64 host, and package the two tarballs LumaBrowser's
# runtime installer expects (names follow the Windows pairs):
#   sd-<tag>-bin-linux-cuda13-arm64.tar.gz   sd-server, sd-cli, ggml + sd shared libs
#   cudart-sd-bin-linux-cu13-arm64.tar.gz    CUDA runtime libs (redistributed per
#                                            the CUDA EULA redist list)
# Requires: an aarch64 Linux host, cmake + ninja, a C/C++ toolchain, and a
# CUDA 13.x toolkit. The workflow runs it inside NVIDIA's
# nvidia/cuda:13.5.0-devel-ubuntu24.04 image, which is all of that but cmake,
# ninja and gcc-14.
#
#   scripts/build-linux-cuda-arm64.sh [-t tag] [-c cuda-root] [-a archs] [-j jobs]
set -euo pipefail

tag='local'
cuda="${CUDA_PATH:-/usr/local/cuda}"
# Empty: ggml's own release set for the toolkit (75/80/90 virtual, 86/89 real,
# 120a + 121a real on CUDA 13), which covers the DGX Spark, Jetson and
# Arm servers with a datacenter card. '121a-real' is a much faster build for a
# DGX Spark alone.
archs=''
jobs="$(nproc)"
while getopts 't:c:a:j:' opt; do
  case "$opt" in
    t) tag="$OPTARG" ;;
    c) cuda="$OPTARG" ;;
    a) archs="$OPTARG" ;;
    j) jobs="$OPTARG" ;;
    *) echo "usage: $0 [-t tag] [-c cuda-root] [-a archs] [-j jobs]" >&2; exit 2 ;;
  esac
done

die() { echo "build-linux-cuda-arm64: $*" >&2; exit 1; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$root/work/stable-diffusion.cpp"
dist="$root/dist"
[ "$(uname -m)" = 'aarch64' ] || die "this builds natively on aarch64, not $(uname -m)"
[ -d "$work" ] || die 'work/stable-diffusion.cpp missing - run scripts/checkout.sh first.'
[ -x "$cuda/bin/nvcc" ] || die "no nvcc under $cuda (pass -c <cuda-root>)"

# GGML_CPU_ALL_VARIANTS builds an armv9.2 CPU backend with the 'sme' march
# feature, which GCC 13 (Ubuntu 24.04's default) does not know. ggml-org's own
# arm64 CUDA release build switches to gcc-14 for the same reason.
if [ -z "${CC:-}" ] && command -v gcc-14 >/dev/null 2>&1 && command -v g++-14 >/dev/null 2>&1; then
  export CC=gcc-14 CXX=g++-14
fi

# Upstream's Linux release defines (shared libs, per-microarch CPU backends
# picked at load time, $ORIGIN rpath so the libs resolve beside the binary),
# plus CUDA and no server web frontend (needs pnpm; LumaBrowser talks to the
# HTTP API only). GGML_CUDA_NCCL=OFF: NVIDIA's devel image ships NCCL, ggml
# links it when found, and a libnccl.so.2 dependency would stop the CUDA
# backend loading on every machine without NCCL installed.
defines=(
  -DCMAKE_BUILD_TYPE=Release
  -DSD_CUDA=ON
  -DGGML_CUDA_NCCL=OFF
  -DSD_BUILD_SHARED_LIBS=ON
  -DSD_BUILD_SHARED_GGML_LIB=ON
  -DGGML_BACKEND_DL=ON
  -DGGML_NATIVE=OFF
  -DGGML_CPU_ALL_VARIANTS=ON
  -DSD_SERVER_BUILD_FRONTEND=OFF
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  '-DCMAKE_INSTALL_RPATH=$ORIGIN'
  "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc"
  "-DCUDAToolkit_ROOT=$cuda"
)
if [ -n "$archs" ]; then defines+=("-DCMAKE_CUDA_ARCHITECTURES=$archs"); fi

cmake -S "$work" -B "$work/build" -G Ninja "${defines[@]}"
cmake --build "$work/build" -- -j "$jobs"

bin="$work/build/bin"
[ -x "$bin/sd-server" ] || die 'build finished without build/bin/sd-server'
ls "$bin"/libggml-cuda.so >/dev/null 2>&1 || die 'build finished without the ggml CUDA backend (libggml-cuda.so)'

# The backend may only need what the cudart tarball, the driver (libcuda) and
# a stock distro provide.
if command -v readelf >/dev/null 2>&1; then
  needed="$(readelf -d "$bin/libggml-cuda.so" | grep NEEDED | grep -o '\[.*\]' | tr -d '[]')"
  allowed='^(libggml(-base)?|libcudart|libcublas(Lt)?|libcuda|libstdc\+\+|libm|libgcc_s|libc|libgomp|libdl|libpthread|librt|ld-linux-aarch64)\.so'
  extra="$(printf '%s\n' "$needed" | grep -Ev "$allowed" || true)"
  [ -z "$extra" ] || die "libggml-cuda.so needs libraries the release does not ship: $(echo $extra)"
fi

mkdir -p "$dist"
bin_tar="$dist/sd-$tag-bin-linux-cuda13-arm64.tar.gz"
cudart_tar="$dist/cudart-sd-bin-linux-cu13-arm64.tar.gz"
rm -f "$bin_tar" "$cudart_tar"

cp "$work/LICENSE" "$bin/stable-diffusion.cpp.txt"
cp "$work/ggml/LICENSE" "$bin/ggml.txt"
tar -czf "$bin_tar" -C "$bin" .

# cp -L turns the SONAME symlinks into plain files, so the tarball holds
# exactly the three libraries the CUDA backend loads, under the names it asks for.
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
major="$("$cuda/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -n 1)"
[ -n "$major" ] || die 'could not read the CUDA major version from nvcc'
libdir="$cuda/lib64"
for lib in libcudart libcublas libcublasLt; do
  [ -e "$libdir/$lib.so.$major" ] || die "no $lib.so.$major under $libdir"
  cp -L "$libdir/$lib.so.$major" "$stage/"
done
tar -czf "$cudart_tar" -C "$stage" .

ls -lh "$bin_tar" "$cudart_tar"
