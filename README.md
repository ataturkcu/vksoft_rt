# vksoft_rt

Software based ray tracing layer for Vulkan 1.2+ GPUs without dedicated RT cores.

Primary target: NVIDIA on Linux.

Status: M0 done, M1 in progress (own acceleration structures; triangle BLAS/TLAS GPU builds and `ray_query` shaders (SPIR-V rewritten) work). See
[docs/design.md](docs/design.md).

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
VK_ADD_LAYER_PATH=$PWD/build VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt vulkaninfo | grep -i ray_query
```

The layer is explicit (opt-in) on purpose: it advertises features it does not implement yet.
