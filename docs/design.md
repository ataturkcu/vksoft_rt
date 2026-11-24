# vksoft_rt — design notes

Status: **draft / discussion**. Decisions marked *(decided)* were agreed on; everything
else is a hypothesis to be validated by measurement.

## 1. Goal

A Vulkan layer that lets Vulkan 1.2+ GPUs **without dedicated RT cores** run games that
use hardware ray tracing, with as little performance loss as possible. Primary target:
NVIDIA on Linux (dev machine: GTX 1660 Ti, driver 615.78.08). AMD/RADV is secondary.

We do **not** write a driver. Everything happens in a layer that sits between the
application (or vkd3d-proton) and the real driver.

"No FPS loss" is not achievable without RT hardware; the realistic goal is the best
achievable performance of software RT, measured against the driver's own emulation.

## 2. Baseline findings (GTX 1660 Ti, Linux)

| Extension                         | Exposed by driver |
|-----------------------------------|-------------------|
| `VK_KHR_acceleration_structure`   | yes               |
| `VK_KHR_ray_tracing_pipeline`     | yes (`rayTracingPipeline = true`) |
| `VK_KHR_ray_query`                | **no**            |

The NVIDIA driver already emulates the pipeline path on this GPU. `ray_query` is missing,
so any game that requires it cannot start. The driver's BVH format is opaque, so a
`ray_query` implementation of ours cannot traverse driver-built acceleration structures.

### Performance baseline (driver emulation, no layer)

Quake II RTX, GTX 1660 Ti, Linux, measured by the project owner:

| Resolution | FPS  |
|------------|------|
| 1080p      | ~13  |
| 900p       | ~20  |
| 720p       | ~30  |

FPS scales almost linearly with pixel count, i.e. with **ray count**. This is the number
M2/M3 must beat, measured with identical settings and scene.

## 3. Core decision *(decided)*: own acceleration structures, single BVH

The layer intercepts all `VK_KHR_acceleration_structure` build/copy/query commands and
builds **its own BVH in its own format**, stored inside the buffer the application
allocated for the AS. Both `ray_query` and ray tracing pipelines are served by our
traversal code. The driver's emulation is not used for tracing.

Rejected: a hybrid (driver AS for pipelines, ours for ray_query) — two BVHs per scene
doubles build time and memory for no benefit.

Consequences:
- We must implement the whole AS API surface: build sizes, build/update, copy, compact,
  serialize/deserialize, device addresses, instance/TLAS semantics.
- We fully control the data layout, so we can optimise it for Turing (small, quantized).
- Dependence on the driver's RT emulation disappears.

## 4. Architecture

```
app / vkd3d-proton
        │  VK_KHR_acceleration_structure, ray_query, ray_tracing_pipeline
┌───────▼────────────────────────────────────────────┐
│ vksoft_rt layer                                    │
│  ├─ capability spoofing (extensions, features)     │
│  ├─ AS manager     (BVH build / refit / compact)   │
│  ├─ shader rewriter (SPIR-V → traversal inlined)   │
│  └─ RT pipeline → compute megakernel               │
└───────┬────────────────────────────────────────────┘
        │  plain Vulkan 1.2 compute
   real driver
```

### 4.1 Capability spoofing (M0, implemented)
Hook `vkEnumerateDeviceExtensionProperties`, `vkGetPhysicalDeviceFeatures2[KHR]`,
`vkCreateDevice`. Advertise `VK_KHR_ray_query` + `rayQuery`; strip them again before
calling the real `vkCreateDevice`. Advertising only happens if the driver exposes
`acceleration_structure` and does not already expose `ray_query`.

### 4.2 Acceleration structures
- Build on the GPU with compute shaders (LBVH or PLOC); GPU refit for updates
  (skinned meshes); compaction support.
- Node layout: wide BVH (4 or 8) with quantized child boxes. The 1660 Ti has
  288 GB/s and 6 GB, so memory traffic, not ALU, is likely the first bottleneck.
- Triangles stored compactly, pre-transformed where it pays off.
- Worst-case sizes are reported by `vkGetAccelerationStructureBuildSizesKHR`
  (BVH over N primitives has bounded node count), so the app's allocations suffice.
- Build work should be schedulable on an async compute queue when available.

#### Object model (implemented in M1.1)
- `VkAccelerationStructureKHR` handles are **fake**, owned by the layer (tagged 64-bit
  ids); the driver never sees them. Every consumer of those handles must therefore be
  intercepted: build/copy/serialize commands (currently logged no-ops), and — still to do —
  `vkUpdateDescriptorSets` / descriptor set layouts / pools for
  `VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR`, which will be rewritten into a
  storage-buffer descriptor over the AS' backing buffer (rewritten SPIR-V reads the BVH
  through it; BLAS references use buffer device addresses, which is why Vulkan 1.2
  `bufferDeviceAddress` is the baseline).
- `vkGetAccelerationStructureDeviceAddressKHR` = backing buffer address + offset.
- `vkGetAccelerationStructureBuildSizesKHR` returns conservative upper bounds from
  [layer/src/bvh_format.h](../layer/src/bvh_format.h) (unit-tested).
- Open: vkd3d-proton may build AS descriptors from raw GPU addresses instead of handles;
  to be checked before M4.
- The layer still enables the driver's own `acceleration_structure` extension on the real
  device (harmless; keeps the feature chain valid). Dropping that dependency is needed for
  other vendors later.

#### BLAS build pipeline (M1.2, implemented for triangle geometry)
Compute kernels in [layer/shaders/](../layer/shaders), recorded into the app's command
buffer by [layer/src/accel_build.cpp](../layer/src/accel_build.cpp). All memory is accessed
through buffer device addresses; parameters travel in a 128-byte push-constant block.

`init` -> `header` -> `fetch_tris` (per geometry: index/vertex fetch, geometry transform,
centroid bounds) -> `morton` (30-bit) -> 4 x [`radix_hist`, `scan1/2/3`, `radix_scatter`]
(stable 8-bit LSD radix sort, subgroup-ballot ranking) -> `karras` (binary radix tree) ->
`refit` (writes leaves in sorted order, bottom-up boxes).

Current format is a **binary BVH** (64 B nodes: two child boxes + two child refs, leaf =
`0x80000000 | primIndex`, one triangle per leaf, 128 B header with root/bounds). This stays
within the size bounds reserved for the planned wide/quantized format, so the collapse to
BVH4/8 (M3) is a drop-in replacement. Triangles are stored pre-transformed (48 B records).

Measured on the 1660 Ti (includes copying the 112 MB result back, so it is an upper
bound): 100k triangles 3.2 ms, 1M triangles 25 ms. Correctness is checked on the CPU by
`tests/blas_build.cpp` (tree shape, tight boxes, every leaf exactly once, exact triangle
set incl. ids/transforms, ties, block-edge sizes).

Known limitations / decisions:
- Only `R32G32B32_SFLOAT` triangles; AABB geometry, TLAS, host builds, copy/compact/
  serialize are not implemented yet (M1.3+). `mode = UPDATE` is treated as a full rebuild
  (always correct; refit-only update is an M3 optimisation).
- Kernels assume subgroup size 32 (+ballot/arithmetic); the layer disables the builder
  otherwise (AMD/Intel need a generic path later).
- The layer forces `shaderInt64` on devices that enable acceleration structures.
- The app's barriers name the driver's AS-build stage, which may not cover our compute
  work, so every build is fenced with ALL_COMMANDS barriers before and after. Correct but
  conservative; to be relaxed by tracking the app's barriers.
- A build clobbers the bound compute pipeline and compute push constants, so the layer
  tracks `vkCmdBindPipeline`/`vkCmdPushConstants` (only on devices using AS) and restores
  them after each build. `vkCmdPushConstants2` is not tracked yet.
- NaN triangles are not yet excluded (spec: such primitives are inactive).
- Refit finding: the second-arriving thread must read its sibling's box with atomics;
  plain/`coherent` loads returned stale data intermittently on NVIDIA. Atomics for all
  six words is correct but not free; revisit when optimising.

### 4.3 `ray_query`
Rewrite SPIR-V: replace `OpRayQuery*` instructions with calls to a generated traversal
function plus per-query state (SPIR-V rewriting via SPIRV-Tools passes). Traversal
stack in workgroup memory (short stack + restart trail), subgroup operations where useful.
Fast paths: opaque geometry, terminate-on-first-hit (shadow rays).

### 4.4 Ray tracing pipelines
Combine raygen + hit/miss/callable shaders into one compute megakernel:
- `traceRayEXT` → traversal call; hit shader chosen by SBT index via `switch`
  (SPIR-V has no function pointers, so hit shaders are inlined).
- Any-hit/intersection shaders run inside the traversal loop (alpha-tested foliage).
- Payload/hit attributes live in private/function storage; recursion needs an explicit
  stack or a depth limit.
- `vkCmdBindPipeline`/descriptor/push-constant binds at the ray-tracing bind point are
  remapped to the compute bind point; `vkCmdTraceRaysKHR` becomes `vkCmdDispatch` with
  tile/Morton-ordered thread mapping for ray coherence.

### 4.5 Performance levers beyond raw traversal speed
Traversal speed alone will give modest gains. Ray *count* is the larger lever:
checkerboard/sparse dispatch with reconstruction, distance culling in the TLAS, BLAS LOD.
These change semantics (the game's denoiser expects its own ray pattern) and need to know
which images are outputs, so they are **research items**, off by default, per-game profiles.

## 5. Milestones

| M  | Content | Validation |
|----|---------|------------|
| M0 | Layer skeleton, dispatch, ray_query advertisement | `vulkaninfo` shows ray_query *(done)* |
| M1 | AS build/update/copy, own BVH; `ray_query` via SPIR-V rewrite (done: object model, sizes, triangle BLAS GPU build; next: TLAS/AABB, then traversal + SPIR-V rewrite) | Khronos ray-query samples, Vulkan CTS RT tests, Sponza |
| M2 | RT pipeline megakernel | Quake II RTX, compare against driver emulation |
| M3 | Performance: wide/quantized BVH, refit, async build, coherence | Frame-time benchmarks vs. M2 and vs. driver |
| M4 | DXR games via vkd3d-proton / Proton | RT-required titles |

## 6. Risks / open questions

- RT pipeline megakernel: code bloat from inlined hit shaders, register pressure,
  recursion.
- Layer-level SPIR-V rewriting must cope with arbitrary game shaders (extensions,
  descriptor indexing, buffer device address).
- Beating NVIDIA's in-driver emulation from a layer is not guaranteed; fallback value is
  providing the missing `ray_query` and ray-count reduction.
- vkd3d-proton may require additional extensions/features; to be verified in M4.
- Advertising features that are not implemented is dangerous: until M1 the layer is
  explicit-only (`VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt`), never an implicit layer.

## 7. Developing

Tests: `ctest --test-dir build --output-on-failure` (`bvh_sizes` is pure host code;
`as_lifecycle` and `blas_build` run through the layer on the real GPU). Building needs
`glslc` and Python 3.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
VK_ADD_LAYER_PATH=$PWD/build VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt \
  VKSOFT_RT_LOG=1 vulkaninfo | grep -i ray_query
```
