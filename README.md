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

## Layout

```
UPSTREAM_REF                    pinned upstream commit the stack is tested against
patches/NNNN-*.patch            git format-patch files, applied in order with git am
scripts/checkout.ps1            clone upstream @ UPSTREAM_REF (with submodules) into work/ and apply the stack
scripts/build-windows-cuda.ps1  build sd-server + sd-cli and package the release zips
.github/workflows/release.yml   tag-triggered CI release build
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

## Releases and LumaBrowser integration

Release assets deliberately match leejet's naming so LumaBrowser's runtime
installer regexes consume them unchanged:

```
sd-<tag>-bin-win-cuda12-x64.zip     sd-server, sd-cli, ggml + stable-diffusion DLLs
cudart-sd-bin-win-cu12-x64.zip      CUDA runtime DLLs (cudart, cublas, cublasLt)
```

Tag `luma-b<n>` to publish a release from CI. LumaBrowser lists this feed as
the "stable-diffusion.cpp (LumaByte)" runtime under Setup.

## License

MIT, like upstream. Patches are (c) LumaByte, LLC.
