# vksoft_rt

Software based ray tracing layer for Vulkan 1.2+ GPUs without dedicated RT cores.
Primary target: NVIDIA on Linux.

Status: M1 done (own acceleration structures: triangle/AABB BLAS and TLAS GPU builds, `ray_query`
shaders rewritten at SPIR-V level, descriptors incl. templates/push/arrays). The ray tracing
*pipeline* path (`vkCmdTraceRaysKHR`) is not implemented yet. See [docs/design.md](docs/design.md).

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
VK_ADD_LAYER_PATH=$PWD/build VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt vulkaninfo | grep -i ray_query
```

The layer is explicit (opt-in) on purpose: it advertises features it does not fully implement yet.

## Quake II RTX through Steam

Quake II RTX has a `ray_query` back end, which the layer makes available on GPUs where the driver
does not expose `VK_KHR_ray_query`. In Steam: game properties -> launch options:

```
VK_LAYER_PATH=/path/to/vksoft_rt/build VK_INSTANCE_LAYERS=VK_LAYER_VKSOFT_rt VKSOFT_RT_LOG=1 %command%
```

(`VK_LAYER_PATH` + `VK_INSTANCE_LAYERS` work with the old Vulkan loaders of Steam's container
runtimes; with a recent loader `VK_ADD_LAYER_PATH` + `VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt` also work.)
In the game console: `ray_tracing_api query`, then `vid_restart`. `ray_tracing_api pipeline` selects
the driver's own emulation, which is the baseline to compare against.
