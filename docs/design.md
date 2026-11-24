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
- Only `R32G32B32_SFLOAT` triangles; AABB geometry, host builds, copy/compact/
  serialize are not implemented yet. `mode = UPDATE` is treated as a full rebuild
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

#### TLAS build (M1.3, implemented)
Same sort/tree pipeline as the BLAS, with instance-specific stages:
`fetch_instances` reads the application's `VkAccelerationStructureInstanceKHR` array (or
array of pointers), looks up the referenced BLAS header (checked via magic/type), transforms
its bounds into a world-space box (Arvo) and writes a 24 B box record; `morton` then uses box
centres. The final `refit` pass writes the **128 B TLAS leaf record** (layout in
[bvh_format.h](../layer/src/bvh_format.h)): objectToWorld, a precomputed worldToObject
(3x3 cofactor inverse; zeros for singular matrices), custom index/mask, SBT offset/flags,
BLAS address, original instance index.

Inactive instances (null or invalid BLAS reference, or an empty BLAS) stay in the tree as
leaves with an *empty* box (min=+inf, max=-inf) and mask 0: empty boxes are the identity of
the box union, so no special casing in the refit, and they sort to the end of the Morton
order. The same trick will be used for NaN triangles.

The BLAS must be built before the TLAS in submission order (the spec requires this anyway);
the conservative barriers around each build make the BLAS header visible to the TLAS build.
Measured (includes readback copy): 5k instances 0.8 ms, 200k instances 37 ms.
`tests/tlas_build.cpp` verifies shape, tight world boxes, every instance exactly once, record
fields, the inverse transform, inactive handling, pointer-array mode and Morton ties.

### 4.3 Traversal (M1.4, implemented as a standalone GLSL library)
[layer/shaders/traverse.glsl](../layer/shaders/traverse.glsl) is a **resumable state machine**
(`rayInit` / `rayProceed` / `rayConfirmCandidate`), so one implementation serves both
`ray_query` (non-opaque candidates are handed to the caller) and the future pipeline
megakernel. It covers: two-level traversal with nearest-child-first order, instance masks,
all ray flags (opaque/no-opaque, terminate-on-first-hit, cull front/back/opaque/no-opaque,
skip triangles), instance flags (facing-cull-disable, flip facing, force (no-)opaque),
object-space facing, watertight triangle intersection (Woop 2013) and empty-box rejection.

`tests/trace_test.cpp` runs it from a compute shader and compares every ray with a
brute-force double-precision CPU reference in 9 flag/mask modes (hit identity, distance,
barycentrics, facing, custom index; ambiguity band for edge hits because float32 traversal
at coordinates ~1e2 carries ~1e-3 absolute error). Current result: all modes pass, 0.3% of
rays are edge-ambiguous.

**Baseline speed (naive, to be optimised in M3):** 10 Mrays/s closest-hit and 16 Mrays/s
terminate-on-first-hit on the GTX 1660 Ti for 1M fully random (incoherent) rays against a
1M-triangle random triangle soup (worst case for BVH quality, 38% hit). Known reasons:
binary uncompressed BVH, per-thread stack in local memory, `coherent` loads everywhere,
no ray sorting/coherence, no wide nodes. Real-game numbers need a real scene and an
apples-to-apples measurement of the driver's own emulation (pipeline path) — planned.

Open correctness items: facing convention (`dot(n, d) < 0` = front, to be confirmed against
Vulkan CTS), AABB/procedural geometry, any-hit/intersection shader callbacks for the
pipeline path, stack overflow currently drops entries silently (depth 64).

### 4.3b `ray_query` (M1.5, implemented)
**Shader side** ([layer/src/rq_rewrite.cpp](../layer/src/rq_rewrite.cpp)). Hooked into
`vkCreateShaderModule`: modules declaring `RayQueryKHR` are rewritten, others pass through
untouched. The rewriter works on SPIR-V *text* (SPIRV-Tools disassembler/assembler, raw
numeric ids), so no operand grammar tables are needed:
- The vk_rq_* library ([layer/shaders/rq_lib.comp](../layer/shaders/rq_lib.comp), compiled
  `-O0`, otherwise glslang drops what only the rewriter references) is merged into the module:
  only definitions reachable from the exported functions are copied, ids are renumbered, types
  and constants are deduplicated against the application's, `GLSL.std.450` imports are shared.
- Every `OpRayQuery*` becomes an `OpFunctionCall`. glslang passes parameters by pointer, so each
  argument goes through a hoisted temporary variable; integer results are bit-cast to the
  application's signedness.
- Each `rayQueryEXT` variable owns a **slot** of a private array in the library (its size is the
  library's spec constant, patched to the real slot count). Storage-class mismatches between
  `Private` query variables and `Function` pointer parameters made passing pointers
  impractical.
- Acceleration structure variables become `StorageBuffer` blocks `{uint lo; uint hi;}`; loads
  are rewritten into two loads + bitcast to `uint64`. The module gets `Int64`,
  `PhysicalStorageBufferAddresses` and the `PhysicalStorageBuffer64` addressing model.
- Unsupported patterns (a query passed to another function, arrays of queries or
  acceleration structures) are detected and rejected: the module is passed through and a
  message is logged. `VKSOFT_RT_VALIDATE=1` validates every rewritten module.

**Descriptor side** ([layer/src/descriptor_cells.cpp](../layer/src/descriptor_cells.cpp)).
`ACCELERATION_STRUCTURE_KHR` bindings become `STORAGE_BUFFER` in `vkCreateDescriptorSetLayout`
and `vkCreateDescriptorPool`; `vkUpdateDescriptorSets` turns AS writes into buffer writes over a
16-byte *address cell* (one per acceleration structure, 64k cells, released on destroy). The same
conversion covers push descriptors (`vkCmdPushDescriptorSet[KHR]`), `vkGetDescriptorSetLayoutSupport`
and **descriptor update templates**: the template handed to the application is the real driver
template, created from translated entries; its AS entries read `VkDescriptorBufferInfo` records
that the layer appends after a copy of the application's data blob at update time
(`vkUpdateDescriptorSetWithTemplate`, `vkCmdPushDescriptorSetWithTemplate`). Not covered:
`VK_EXT_descriptor_buffer` (`vkGetDescriptorEXT`) and `VK_EXT_mutable_descriptor_type` lists that
name the acceleration structure type; both are needed to check against vkd3d-proton (M4).

`tests/rq_test.cpp` runs an unmodified `GL_EXT_ray_query` GLSL shader (-O0 and -O builds)
through all of this and checks it against the same brute-force reference in 9 flag/mask
modes: all pass. Descriptors are delivered four ways (`vkUpdateDescriptorSets`, update
template, push descriptors, push template), each checked against the reference. `tests/test_rq_rewrite.cpp` validates rewritten modules with spirv-val.

**Speed of the ray_query path (first, unoptimised):** 4.5 Mrays/s closest-hit and 8.2
Mrays/s terminate-on-first-hit (same 1M-triangle worst-case scene as trace_test, where the
direct traversal reaches 10.4 / 16.1), i.e. the rewriting costs 2x. Suspects: the traversal
state is copied in and out of the slot array on every proceed/confirm, the library is not
optimised, and each argument goes through a temporary. To be fixed in M3 together with the
traversal itself.

### 4.3b2 Procedural geometry and AS arrays (implemented for Quake II RTX)
- **AABB geometry**: BLAS builds accept `VK_GEOMETRY_TYPE_AABBS_KHR` (a BLAS holds one geometry
  type). Boxes become leaf records with the same 48-byte stride as triangles (flags bit1 =
  procedural, bit0 = opaque); inactive boxes (min > max or NaN) get empty boxes and sort last.
- **Traversal**: a procedural leaf gives the ray query a *candidate* for any box the ray overlaps
  (opaque or not, as the spec requires); `rayQueryGenerateIntersection` commits it
  (committed type 2), `rayQueryConfirmIntersection` ignores procedural candidates. The
  candidate's effective opacity is available through `rayQueryGetIntersectionCandidateAABBOpaque`.
- **Arrays of acceleration structures** (`accelerationStructureEXT scenes[N]`): the variable
  becomes an array of address blocks, constant-indexed access chains are rewritten to point at
  one block, and descriptors/templates carry N entries.
- `tests/aabb_test.cpp` (6 flag modes vs. a double precision reference) and the array tests in
  `tests/rq_test.cpp` cover these. While writing them a bug in the ray generator of the tests
  turned up (zero direction vectors for "axis aligned" rays, which the reference silently
  treated as misses): fixed, real axis-aligned rays are now tested.

### 4.3b3 First real game result and the first optimisation
Quake II RTX through Steam with `ray_tracing_api query` (our layer): **7 fps at 720p**, against
~30 fps for the driver's own emulation (`ray_tracing_api pipeline`): 4x slower. The game's own
profiler (`profiler 1`) tells which of AS builds (`BVH_UPDATE`) or the tracing passes dominate;
numbers requested.

Optimisation 1 (measured with `rq_test`'s throughput run, same scene): the merged module is
now run through the SPIRV-Tools performance passes (`Optimizer::RegisterPerformancePasses`; the
library is compiled -O0, so the module was entirely unoptimised). ray_query closest-hit went
from ~3.9 to ~7.7 Mrays/s (direct traversal: 10.4), terminate-on-first-hit from 7.6 to 11.0
(direct: 16.1). `VKSOFT_RT_NO_OPT=1` disables it for A/B comparisons.

### 4.3c Open items of the ray query path
Not implemented: triangle vertex position fetch, descriptor buffers, dynamic (non-constant)
indexing of acceleration structure arrays with `NonUniform`, queries as function arguments.

### 4.3d Ray tracing pipelines and vkd3d-proton
Next milestones; see section 5.

### 4.3e Original plan for `ray_query` (superseded by 4.3b)
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

## 4.9 Target analysis: Quake II RTX (read from its open source, no code run)

Source: github.com/NVIDIA/Q2RTX, `src/refresh/vkpt`. Findings that drive the plan:

- **It has two ray tracing back ends**, chosen by the `ray_tracing_api` cvar
  (`auto`/`query`/`pipeline`). `query` compiles every ray generation shader as a *compute* shader
  (`-DKHR_RAY_QUERY`) using `GL_EXT_ray_query`. With `auto` on the proprietary NVIDIA driver it
  *prefers* ray query when the device exposes it. Our layer advertises `ray_query`, so the
  existing ray query path is enough to run the game, **without the pipeline megakernel**.
- The pipeline back end is the baseline we must beat: on the 1660 Ti the driver's own
  emulation is what the measured 13/20/30 fps (1080p/900p/720p) come from. Setting
  `ray_tracing_api pipeline` vs `query` gives an exact apples-to-apples comparison in the same
  game and scene.
- Pipeline back end details (for the later megakernel): `maxPipelineRayRecursionDepth = 1`,
  one raygen per pipeline, a closest-hit, a miss, five any-hit shaders (masked, particle,
  explosion, sprite, beam), one intersection shader (beams); hit groups are selected through
  the instance SBT offset, `sbtRecordStride = 0`.
- AS API surface used: triangle BLAS (`R32G32B32_SFLOAT`, index none or u16), **AABB BLAS**
  (beams), TLAS with instance flags FORCE_OPAQUE / FORCE_NO_OPAQUE / FACING_CULL_DISABLE /
  FRONT_COUNTERCLOCKWISE, build flags PREFER_FAST_BUILD / PREFER_FAST_TRACE. No update, copy
  or compaction.
- What the ray query path needs that we do not have yet: (1) **arrays of acceleration
  structures** in shaders and descriptors (`topLevelAS[2]`: geometry + effects), (2) **AABB
  geometry** in BLAS builds, procedural *candidates* in traversal and
  `rayQueryGenerateIntersectionEXT`.

## 5. Milestones

| M  | Content | Validation |
|----|---------|------------|
| M0 | Layer skeleton, dispatch, ray_query advertisement | `vulkaninfo` shows ray_query *(done)* |
| M1 | AS build/update/copy, own BVH; `ray_query` via SPIR-V rewrite (done: object model, sizes, triangle + AABB BLAS and TLAS builds, traversal library, ray_query SPIR-V rewrite, descriptors incl. templates / push / arrays; later: copy/compact) | Khronos ray-query samples, Vulkan CTS RT tests, Sponza |
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

## 6b. Running with Steam / Proton

The layer is self-contained: SPIRV-Tools is built in statically and the C++ runtime is linked
statically (`VKSOFT_BUNDLE_SPIRV_TOOLS`, default ON), so the .so only needs glibc (max symbol
version GLIBC_2.38). Steam's container runtimes may ship an older Vulkan loader, therefore the
legacy variables are the portable way to enable an explicit layer (see README).

## 7. Developing

Tests: `ctest --test-dir build --output-on-failure` (`bvh_sizes` is pure host code;
`as_lifecycle`, `blas_build`, `tlas_build`, `trace_test`, `rq_test` and `aabb_test` run through the layer on the real GPU). Building needs
`glslc`, Python 3 and SPIRV-Tools (libSPIRV-Tools).

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
VK_ADD_LAYER_PATH=$PWD/build VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt \
  VKSOFT_RT_LOG=1 vulkaninfo | grep -i ray_query
```
