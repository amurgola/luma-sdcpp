# luma-sdcpp

LumaByte's patched [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
builds for [LumaBrowser](https://lumabyte.com). Like
[luma-llamacpp](https://github.com/amurgola/luma-llamacpp), this is **not a
fork**: it is a small, auditable patch stack applied on top of a pinned
upstream commit, rebuilt as upstream moves. Every patch is an upstream
candidate or LumaBrowser-specific glue.

## Why

LumaBrowser's image editing sends every reference image (the picture being
edited plus a face, a garment, a scene) to sd-server on each request. The
patches here target the time sd-server spends on the CPU before the GPU does
any work.

## Patch stack

| # | Patch | Status |
|---|---|---|
| 0001 | `Tensor::index()` without a heap allocation per element, and a nearest-neighbour `interpolate` that walks per-axis offset tables instead of two coordinate vectors per element | upstream candidate |
| 0002 | The compute workspace remembers the last two graph measurements, so the conditional and unconditional passes of a guided run both hit the cache instead of re-measuring the whole graph on every pass | upstream candidate |
| 0003 | Wan VAE: the uncached first frame of a causal conv3d runs as a conv2d with the last temporal tap (every image decode and encode of Anima and Qwen-Image). Taken unchanged from [unslothai/stable-diffusion.cpp](https://github.com/unslothai/stable-diffusion.cpp) commit `430ee75`, MIT, author kept in the patch | third-party, upstream candidate there |
| 0004 | CUDA INT8: `linear -> GELU -> ConvRot quantize` fused through the existing INT8 linear epilogue matcher, next to its SwiGLU form | upstream candidate |
| 0005 | CUDA: a row-wise relayout kernel for strided F32 copies and F32 to F16 casts (the permuted head layouts around attention), one warp per contiguous source row | upstream candidate |
| 0006 | CUDA INT8: LayerNorm folded into the kernel that modulates, rotates and quantizes its result (`LayerNorm -> Modulate -> ConvRot`), next to the existing RMSNorm form. Not bit-identical: the reduction order differs | upstream candidate |
| 0007 | Anima: one SiLU per pass for the timestep embedding (85 adaptive norms recomputed it) and the shift, scale and gate slices of the modulation vector used as views | upstream candidate |
| 0008 | RD Flow 64 (Retro Diffusion, 64x64 pixel art): the pixel-space palette-pointer network, its own sampler with palette extraction and pointer commit, RGBA output without a VAE, and the `palette` / `opaque_background` request fields | new model, upstream candidate |

`Tensor::index(x, y, c, n)` built a `std::vector` for every element it
touched, and the image paths call it once per pixel per channel
(`sd_image_to_tensor`, `ImagePreprocessor::apply_transform`, the nearest
`ops::interpolate`). That cost about 9 s of CPU per megapixel of input image
on every reference of every edit request.

Measured on a three-image Qwen-Image 2.1 edit at 768x1344 (Q8_0, Viggle turbo
LoRA, 6 steps, RTX 5090, Windows 11), same seed:

| | upstream `b167b94` | patched |
|---|---|---|
| request wall time | 38.1 s | **12.3 s** |
| reference encode (incl. resize) | 5.72 s | 0.55 s |
| preprocess, per reference | ~9 s/MP | ~0.03 s |
| sampling (6 steps) | 8.26 s | 7.63 s |

The output is pixel-identical to upstream (zero differing channels).

### Patches 0002 to 0005 (`luma-b5`)

These target the diffusion pass itself. Measured on Anima (a 2B DiT, 896x1152,
25 steps, cfg 4, same seed), RTX 5090 and RTX 3090, Windows 11:

| | `luma-b4` | `luma-b5` |
|---|---|---|
| RTX 5090, INT8 ConvRot weights, `--sage-attn` | 5.6 it/s | **7.2 it/s** |
| RTX 3090, INT8 ConvRot weights, `--sage-attn` | 2.36 it/s | **2.61 it/s** |
| RTX 5090, BF16 weights | 3.41 it/s | 3.45 it/s |
| Wan VAE decode, 896x1152 | 0.33 s | 0.26 s |

Per kernel on the RTX 5090 at 4032 tokens: the fused GELU step takes 329 us
where the linear, the GELU pass and the quantize pass took 488 us; an F32 to
F16 head-layout cast takes 20 us where the generic copy took 133 us, and the
permuted F32 copy 24 us where it took 132 us.

Patches 0002, 0004 and 0005 are pixel-identical to `luma-b4` on Anima (BF16
and INT8), SDXL, Qwen-Image 2.1 and MiniMax-H3-Image. Patch 0003 changes the
summation order of the VAE convolutions: the largest channel difference
measured was 4 of 255 (PSNR 70 dB); `SD_VAE_FRAME_COLLAPSE=0` restores the
full conv3d. The two CUDA fusions follow upstream's `GGML_CUDA_DISABLE_FUSION`
switch.

### Patches 0006, 0007 and CUDA Graph replay (`luma-b6`)

The Windows x64 build is also compiled with `-DGGML_CUDA_GRAPHS=ON`, which
enables the extension executor's CUDA Graph replay (off upstream;
`GGML_CUDA_DISABLE_GRAPHS=1` turns it off at run time). Same Anima run as
above, INT8 ConvRot weights, `--sage-attn`:

| | `luma-b5` | with 0006 | with 0006, 0007 and graph replay |
|---|---|---|---|
| RTX 5090 | 7.2 it/s | 7.9 it/s | **8.3 it/s** |
| RTX 3090 | 2.61 it/s | 2.73 it/s | 2.80 it/s |

Patch 0007 and graph replay are pixel-identical (Anima BF16 and INT8, SDXL,
Qwen-Image 2.1, MiniMax-H3-Image, and ten mixed-size requests on one warm
sd-server). Graph replay keeps about 130 MiB more VRAM once every request
shape has been seen. Patch 0006 only applies to INT8 ConvRot models with an
adaptive LayerNorm (Anima); its output differs from the unfused path
(PSNR 34 dB on a short run) and looks equivalent.

### Patch 0008: RD Flow 64

[RD Flow 64](https://huggingface.co/Retro-Diffusion/rd-flow-64) draws 64x64
pixel art directly on the pixel grid: every pixel is an exact palette colour
or fully transparent. It has no VAE and its own sampler, so `generate_image`
branches to `src/pipeline/rd_flow.cpp`. Usage and the optional
Q8_0 conversion of the text encoder (the released file loads as is) are in `docs/rd_flow_64.md` of the patched tree. The weights
are non-commercial.

Checked against the reference PyTorch pipeline (fp32 drawing model, same
caption, seed and `--rng cpu`) on an RTX 3090:

| | result |
|---|---|
| locked palette (Sweetie 16), sd-cli and sd-server | pixel-identical |
| locked palette, Q8_0 text encoder on GPU / on CPU | 99.9% / 100% identical pixels |
| palette chosen by the model | 91% identical pixels, same mask, 30 of 32 colours |
| prompt read, Q8_0 on the CPU | 1.6 s (PyTorch on the CPU: 19 s) |
| 16 steps with guidance, one image | 1.5 s |
| VRAM, text encoder on the CPU | about 1 GB |

Former patch 0002 (auto-fit using an integrated GPU such as the RTX Spark N1X
when there is no discrete one) was dropped at `luma-b4`: upstream's auto-fit
now accepts `GGML_BACKEND_DEVICE_TYPE_IGPU` devices itself.

## Layout

```
UPSTREAM_REF                    pinned upstream commit the stack is tested against
patches/NNNN-*.patch            git format-patch files, applied in order with git am
scripts/checkout.ps1            clone upstream @ UPSTREAM_REF (with submodules) into work/ and apply the stack
scripts/build-windows-cuda.ps1        build sd-server + sd-cli (x64, CUDA 12) and package the release zips
scripts/build-windows-cuda-arm64.ps1  the same for Windows on Arm (CUDA 13, RTX Spark), built natively on ARM64
scripts/checkout.sh                   bash twin of checkout.ps1, for the Linux build
scripts/build-linux-cuda-arm64.sh     the same for Linux arm64 (CUDA 13), built natively on aarch64
.github/workflows/release.yml         tag-triggered CI release build
```

## Building locally (Windows)

Prereqs: VS 2022 Build Tools (C++ workload), CUDA toolkit 12.x, cmake + ninja
on PATH. nvcc 12.x cannot use the VS **2026** compiler (cudafe++ crashes); the
2022 toolset is required.

```powershell
.\scripts\checkout.ps1                                   # work/stable-diffusion.cpp @ UPSTREAM_REF + patches
.\scripts\build-windows-cuda.ps1 -Tag local -LowPriority # dist/sd-local-bin-win-cuda12-x64.zip (+ cudart zip)
```

The build compiles the CUDA kernels for every architecture leejet ships
(Pascal through Blackwell) and takes an hour or more. `-Jobs` defaults to
half the logical cores so the machine stays usable; `-Archs '86;89;120'` is
a much faster build for one's own cards.

### Windows on Arm (RTX Spark)

Prereqs: VS 2022 Build Tools with the ARM64 C++ tools
(`Microsoft.VisualStudio.Component.VC.Tools.ARM64`), cmake + ninja on PATH, and
a CUDA 13.x toolkit with Windows ARM64 libraries. NVIDIA's `windows-arm64`
redist archives (nvcc, crt, nvvm, cccl, cudart, cublas, profiler_api, nvrtc,
merged into one folder) are enough; the workflow installs exactly those.

```powershell
.\scripts\checkout.ps1
.\scripts\build-windows-cuda-arm64.ps1 -Tag local -CudaPath C:\path\to\cuda-13.4-arm64
# dist/sd-local-bin-win-cuda13-arm64.zip (+ cudart zip); ~5 min on an RTX Spark N1X
```

It builds for `sm_121a` only (the one NVIDIA GPU in a Windows on Arm machine
so far) and with `GGML_NATIVE=ON`, which ggml requires for MSVC on ARM.
Keep the repo in a short path such as `C:\lsb`: upstream's submodules nest
deep enough to pass Windows' 260-character limit from a long one.

## Building locally (Linux arm64)

Prereqs: an aarch64 host, git, cmake + ninja, gcc-14/g++-14 and a CUDA 13.x
toolkit. NVIDIA's `nvidia/cuda:13.5.0-devel-ubuntu24.04` image plus
`apt-get install git ca-certificates build-essential cmake ninja-build gcc-14 g++-14`
is exactly what the workflow uses.

```bash
bash scripts/checkout.sh
bash scripts/build-linux-cuda-arm64.sh -t local -a 121a-real   # DGX Spark only; drop -a for the release arch set
# dist/sd-local-bin-linux-cuda13-arm64.tar.gz (+ cudart tarball)
```

The build follows ggml-org's own arm64 CUDA release recipe: per-microarch CPU
backends chosen at load time (`GGML_CPU_ALL_VARIANTS`, which needs gcc-14 for
the armv9.2 `sme` variant), an `$ORIGIN` rpath so the shared libraries resolve
beside the binaries, and ggml's default CUDA architecture set unless `-a`
narrows it. Building in the Ubuntu 24.04 image makes glibc 2.39 the floor.

## Releases and LumaBrowser integration

Release assets deliberately match leejet's naming so LumaBrowser's runtime
installer regexes consume them unchanged:

```
sd-<tag>-bin-win-cuda12-x64.zip       sd-server, sd-cli, ggml + stable-diffusion DLLs
cudart-sd-bin-win-cu12-x64.zip        CUDA runtime DLLs (cudart, cublas, cublasLt)
sd-<tag>-bin-win-cuda13-arm64.zip     the same, Windows on Arm (CUDA 13)
cudart-sd-bin-win-cu13-arm64.zip      CUDA 13 ARM64 runtime DLLs
sd-<tag>-bin-linux-cuda13-arm64.tar.gz   sd-server, sd-cli and their shared libs, Linux arm64 (CUDA 13)
cudart-sd-bin-linux-cu13-arm64.tar.gz    CUDA 13 runtime libs (libcudart, libcublas, libcublasLt)
```

Tag `luma-b<n>` to publish a release from CI: all three builds run, and a
final job publishes one release with the six archives. A manual run (Actions,
"Run workflow") can build `x64`, `arm64`, `linux-arm64` or `all`, and uploads
the archives as workflow artifacts instead. LumaBrowser lists this feed as
the "stable-diffusion.cpp (LumaByte)" runtime under Setup.

## License

MIT, like upstream. Patches are (c) LumaByte, LLC, except 0003, which is
(c) its author and taken from unslothai/stable-diffusion.cpp under MIT.
