// GPU acceleration structure builder (compute kernels recorded into the app's
// command buffer). See docs/design.md section 4.2.
#pragma once

#include <vulkan/vulkan.h>

namespace vksoft {

struct BuildKernels;

// Creates the internal compute pipelines. Returns nullptr (and logs) when the device
// cannot run them (e.g. subgroup size != 32, missing int64 / device address support).
BuildKernels* createBuildKernels(VkInstance, VkPhysicalDevice, VkDevice,
                                 PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr);
void destroyBuildKernels(BuildKernels*);

// Records the build of one acceleration structure. Returns false (and records nothing)
// when the request is outside what is implemented yet.
bool recordBuild(BuildKernels*, VkCommandBuffer, const VkAccelerationStructureBuildGeometryInfoKHR&,
                 const VkAccelerationStructureBuildRangeInfoKHR* ranges, VkDeviceAddress dstAddress);

}  // namespace vksoft
