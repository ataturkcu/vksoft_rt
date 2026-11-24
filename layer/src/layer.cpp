// vksoft_rt - M0: layer skeleton + VK_KHR_ray_query advertisement.
//
// This milestone only *advertises* VK_KHR_ray_query / rayQuery so we can verify
// the interception path with `vulkaninfo`. Nothing implements ray queries yet,
// so the layer is an explicit (opt-in) layer and must never be enabled
// globally. See docs/design.md.

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include "accel_build.h"
#include "bvh_format.h"
#include "descriptor_cells.h"
#include "rq_rewrite.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define VKSOFT_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr const char* kLayerName = "VK_LAYER_VKSOFT_rt";
constexpr const char* kRayQueryExt = VK_KHR_RAY_QUERY_EXTENSION_NAME;

bool logEnabled() {
  static const bool on = std::getenv("VKSOFT_RT_LOG") != nullptr;
  return on;
}
#define LOG(...)                                   \
  do {                                             \
    if (logEnabled()) {                            \
      std::fprintf(stderr, "[vksoft_rt] " __VA_ARGS__); \
      std::fputc('\n', stderr);                    \
    }                                              \
  } while (0)

// Loader-owned dispatchable handles start with a pointer to their dispatch
// table. That pointer is identical for an instance and all its physical
// devices, so it is a valid key for both.
template <typename T>
void* dispatchKey(T handle) {
  return *reinterpret_cast<void**>(handle);
}

struct InstanceData {
  VkInstance instance{};
  PFN_vkGetInstanceProcAddr gipa{};
  PFN_vkDestroyInstance DestroyInstance{};
  PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties{};
  PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2{};
  PFN_vkGetPhysicalDeviceFeatures2KHR GetPhysicalDeviceFeatures2KHR{};
  PFN_vkCreateDevice CreateDevice{};
};

struct DeviceData {
  VkDevice device{};
  PFN_vkGetDeviceProcAddr gdpa{};
  PFN_vkDestroyDevice DestroyDevice{};
  PFN_vkGetBufferDeviceAddress GetBufferDeviceAddress{};
  PFN_vkCmdBindPipeline CmdBindPipeline{};
  PFN_vkCmdPushConstants CmdPushConstants{};
  PFN_vkBeginCommandBuffer BeginCommandBuffer{};
  PFN_vkFreeCommandBuffers FreeCommandBuffers{};
  PFN_vkCreateShaderModule CreateShaderModule{};
  PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout{};
  PFN_vkCreateDescriptorPool CreateDescriptorPool{};
  PFN_vkUpdateDescriptorSets UpdateDescriptorSets{};
  vksoft::BuildKernels* kernels{};  // internal build pipelines (null if unavailable)
  vksoft::CellPool* cells{};        // TLAS address cells for rewritten descriptors
  bool asEnabled{false};            // app enabled VK_KHR_acceleration_structure
  bool rayQueryRequested{false};    // app asked for rayQuery (emulated by us)
};

// Compute state of a command buffer that our build kernels clobber and must restore.
struct CmdState {
  VkPipeline computePipeline{};
  VkPipelineLayout pushLayout{};
  VkShaderStageFlags pushStages{};
  uint32_t pushMin{UINT32_MAX}, pushMax{0};
  uint8_t push[256]{};
};

// An acceleration structure is a layer-owned object: the handle is fake (the driver
// never sees it) and the data lives in the buffer range the application provided.
struct AsData {
  VkBuffer buffer{};
  VkDeviceSize offset{};
  VkDeviceSize size{};
  VkAccelerationStructureTypeKHR type{};
  VkDeviceAddress address{};
};

std::mutex gMutex;
std::unordered_map<void*, InstanceData> gInstances;
std::unordered_map<void*, DeviceData> gDevices;
std::unordered_map<VkCommandBuffer, CmdState> gCmd;
std::unordered_map<uint64_t, AsData> gAccel;  // fake handle -> data
uint64_t gNextAsHandle = 1;
constexpr uint64_t kAsHandleTag = 0x5a53'0000'0000'0000ull;  // makes stray handles obvious

InstanceData* instanceFor(void* key) {
  std::lock_guard lock(gMutex);
  auto it = gInstances.find(key);
  return it == gInstances.end() ? nullptr : &it->second;
}

// Does the real driver expose everything ray_query builds upon?
bool hasExtension(InstanceData& inst, VkPhysicalDevice pd, const char* name) {
  uint32_t n = 0;
  inst.EnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> props(n);
  inst.EnumerateDeviceExtensionProperties(pd, nullptr, &n, props.data());
  for (auto& p : props)
    if (std::strcmp(p.extensionName, name) == 0) return true;
  return false;
}

bool canAdvertiseRayQuery(InstanceData& inst, VkPhysicalDevice pd) {
  return hasExtension(inst, pd, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
         !hasExtension(inst, pd, kRayQueryExt);  // only fill the gap
}

// ---- instance ------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance) {
  auto* link = reinterpret_cast<VkLayerInstanceCreateInfo*>(
      const_cast<void*>(pCreateInfo->pNext));
  while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                   link->function == VK_LAYER_LINK_INFO))
    link = reinterpret_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(link->pNext));
  if (!link) return VK_ERROR_INITIALIZATION_FAILED;

  PFN_vkGetInstanceProcAddr next = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  link->u.pLayerInfo = link->u.pLayerInfo->pNext;  // advance chain

  auto create = reinterpret_cast<PFN_vkCreateInstance>(next(VK_NULL_HANDLE, "vkCreateInstance"));
  if (!create) return VK_ERROR_INITIALIZATION_FAILED;
  VkResult res = create(pCreateInfo, pAllocator, pInstance);
  if (res != VK_SUCCESS) return res;

  InstanceData d;
  d.instance = *pInstance;
  d.gipa = next;
#define LOAD(fn) d.fn = reinterpret_cast<PFN_vk##fn>(next(*pInstance, "vk" #fn))
  LOAD(DestroyInstance);
  LOAD(EnumerateDeviceExtensionProperties);
  LOAD(GetPhysicalDeviceFeatures2);
  LOAD(GetPhysicalDeviceFeatures2KHR);
  LOAD(CreateDevice);
#undef LOAD
  {
    std::lock_guard lock(gMutex);
    gInstances[dispatchKey(*pInstance)] = d;
  }
  LOG("instance created");
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroyInstance(VkInstance instance,
                                                const VkAllocationCallbacks* pAllocator) {
  void* key = dispatchKey(instance);
  PFN_vkDestroyInstance next;
  {
    std::lock_guard lock(gMutex);
    next = gInstances[key].DestroyInstance;
    gInstances.erase(key);
  }
  next(instance, pAllocator);
}

// ---- physical device queries --------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL hook_EnumerateDeviceExtensionProperties(
    VkPhysicalDevice pd, const char* pLayerName, uint32_t* pCount,
    VkExtensionProperties* pProps) {
  InstanceData* inst = instanceFor(dispatchKey(pd));
  if (!inst) return VK_ERROR_INITIALIZATION_FAILED;

  // We expose no extensions of our own as a layer.
  if (pLayerName && std::strcmp(pLayerName, kLayerName) == 0) {
    *pCount = 0;
    return VK_SUCCESS;
  }
  if (pLayerName)  // some other layer's extensions: pass through
    return inst->EnumerateDeviceExtensionProperties(pd, pLayerName, pCount, pProps);

  uint32_t n = 0;
  VkResult res = inst->EnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
  if (res != VK_SUCCESS) return res;
  std::vector<VkExtensionProperties> all(n);
  res = inst->EnumerateDeviceExtensionProperties(pd, nullptr, &n, all.data());
  if (res != VK_SUCCESS && res != VK_INCOMPLETE) return res;
  all.resize(n);

  if (canAdvertiseRayQuery(*inst, pd)) {
    VkExtensionProperties rq{};
    std::strncpy(rq.extensionName, kRayQueryExt, VK_MAX_EXTENSION_NAME_SIZE - 1);
    rq.specVersion = VK_KHR_RAY_QUERY_SPEC_VERSION;
    all.push_back(rq);
  }

  if (!pProps) {
    *pCount = static_cast<uint32_t>(all.size());
    return VK_SUCCESS;
  }
  uint32_t copy = std::min<uint32_t>(*pCount, static_cast<uint32_t>(all.size()));
  std::memcpy(pProps, all.data(), copy * sizeof(VkExtensionProperties));
  *pCount = copy;
  return copy < all.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

void patchFeatures(InstanceData& inst, VkPhysicalDevice pd, VkPhysicalDeviceFeatures2* f) {
  if (!canAdvertiseRayQuery(inst, pd)) return;
  for (auto* s = reinterpret_cast<VkBaseOutStructure*>(f->pNext); s; s = s->pNext) {
    if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR)
      reinterpret_cast<VkPhysicalDeviceRayQueryFeaturesKHR*>(s)->rayQuery = VK_TRUE;
  }
}

VKAPI_ATTR void VKAPI_CALL hook_GetPhysicalDeviceFeatures2(VkPhysicalDevice pd,
                                                           VkPhysicalDeviceFeatures2* f) {
  InstanceData* inst = instanceFor(dispatchKey(pd));
  inst->GetPhysicalDeviceFeatures2(pd, f);
  patchFeatures(*inst, pd, f);
}

VKAPI_ATTR void VKAPI_CALL hook_GetPhysicalDeviceFeatures2KHR(VkPhysicalDevice pd,
                                                              VkPhysicalDeviceFeatures2* f) {
  InstanceData* inst = instanceFor(dispatchKey(pd));
  (inst->GetPhysicalDeviceFeatures2KHR ? inst->GetPhysicalDeviceFeatures2KHR
                                       : inst->GetPhysicalDeviceFeatures2)(pd, f);
  patchFeatures(*inst, pd, f);
}

// ---- device --------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateDevice(VkPhysicalDevice pd,
                                                 const VkDeviceCreateInfo* pCreateInfo,
                                                 const VkAllocationCallbacks* pAllocator,
                                                 VkDevice* pDevice) {
  InstanceData* inst = instanceFor(dispatchKey(pd));
  if (!inst) return VK_ERROR_INITIALIZATION_FAILED;

  auto* link = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(pCreateInfo->pNext));
  while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                   link->function == VK_LAYER_LINK_INFO))
    link = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(link->pNext));
  if (!link) return VK_ERROR_INITIALIZATION_FAILED;

  PFN_vkGetInstanceProcAddr nextGipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr nextGdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
  link->u.pLayerInfo = link->u.pLayerInfo->pNext;

  // Strip what the real driver does not know: the ray_query extension name and
  // the rayQuery feature struct. We remember that the app wanted it.
  VkDeviceCreateInfo ci = *pCreateInfo;
  std::vector<const char*> exts;
  bool wantsRayQuery = false;
  for (uint32_t i = 0; i < ci.enabledExtensionCount; ++i) {
    if (std::strcmp(ci.ppEnabledExtensionNames[i], kRayQueryExt) == 0) wantsRayQuery = true;
    else exts.push_back(ci.ppEnabledExtensionNames[i]);
  }
  bool wantsAs = false;
  for (uint32_t i = 0; i < ci.enabledExtensionCount; ++i)
    if (std::strcmp(ci.ppEnabledExtensionNames[i], VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
      wantsAs = true;
  bool strip = wantsRayQuery && canAdvertiseRayQuery(*inst, pd);

  // Our build kernels use 64-bit integers (buffer device addresses): force shaderInt64.
  VkPhysicalDeviceFeatures forcedFeatures{};
  VkPhysicalDeviceFeatures2* features2 = nullptr;
  VkBool32 savedInt64 = VK_FALSE;
  if (wantsAs) {
    for (auto* n = reinterpret_cast<const VkBaseInStructure*>(ci.pNext); n; n = n->pNext)
      if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
        features2 = reinterpret_cast<VkPhysicalDeviceFeatures2*>(const_cast<VkBaseInStructure*>(n));
    if (features2) {
      savedInt64 = features2->features.shaderInt64;
      features2->features.shaderInt64 = VK_TRUE;
    } else {
      if (ci.pEnabledFeatures) forcedFeatures = *ci.pEnabledFeatures;
      forcedFeatures.shaderInt64 = VK_TRUE;
      ci.pEnabledFeatures = &forcedFeatures;
    }
  }
  if (strip) {
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.data();
  }

  // Temporarily unlink VkPhysicalDeviceRayQueryFeaturesKHR from the pNext chain.
  VkBaseOutStructure* prev = reinterpret_cast<VkBaseOutStructure*>(&ci);
  VkBaseOutStructure* unlinked = nullptr;
  VkBaseOutStructure* unlinkedPrev = nullptr;
  if (strip) {
    for (auto* s = prev; s->pNext; s = s->pNext) {
      if (s->pNext->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR) {
        unlinkedPrev = s;
        unlinked = s->pNext;
        s->pNext = unlinked->pNext;
        break;
      }
    }
  }

  auto create = reinterpret_cast<PFN_vkCreateDevice>(nextGipa(inst->instance, "vkCreateDevice"));
  VkResult res = create(pd, &ci, pAllocator, pDevice);

  if (unlinked) {  // restore the app's chain
    unlinked->pNext = unlinkedPrev->pNext;
    unlinkedPrev->pNext = unlinked;
  }
  if (features2) features2->features.shaderInt64 = savedInt64;
  if (res != VK_SUCCESS) return res;

  DeviceData d;
  d.device = *pDevice;
  d.gdpa = nextGdpa;
  d.DestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(nextGdpa(*pDevice, "vkDestroyDevice"));
  d.GetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
      nextGdpa(*pDevice, "vkGetBufferDeviceAddress"));
  if (!d.GetBufferDeviceAddress)
    d.GetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
        nextGdpa(*pDevice, "vkGetBufferDeviceAddressKHR"));
  d.CmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(nextGdpa(*pDevice, "vkCmdBindPipeline"));
  d.CmdPushConstants = reinterpret_cast<PFN_vkCmdPushConstants>(nextGdpa(*pDevice, "vkCmdPushConstants"));
  d.BeginCommandBuffer = reinterpret_cast<PFN_vkBeginCommandBuffer>(nextGdpa(*pDevice, "vkBeginCommandBuffer"));
  d.FreeCommandBuffers = reinterpret_cast<PFN_vkFreeCommandBuffers>(nextGdpa(*pDevice, "vkFreeCommandBuffers"));
  d.CreateShaderModule = reinterpret_cast<PFN_vkCreateShaderModule>(nextGdpa(*pDevice, "vkCreateShaderModule"));
  d.CreateDescriptorSetLayout = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(nextGdpa(*pDevice, "vkCreateDescriptorSetLayout"));
  d.CreateDescriptorPool = reinterpret_cast<PFN_vkCreateDescriptorPool>(nextGdpa(*pDevice, "vkCreateDescriptorPool"));
  d.UpdateDescriptorSets = reinterpret_cast<PFN_vkUpdateDescriptorSets>(nextGdpa(*pDevice, "vkUpdateDescriptorSets"));
  d.asEnabled = wantsAs;
  if (wantsAs && d.GetBufferDeviceAddress) {
    d.kernels = vksoft::createBuildKernels(inst->instance, pd, *pDevice, nextGipa, nextGdpa);
    d.cells = vksoft::createCellPool(inst->instance, pd, *pDevice, nextGipa, nextGdpa);
  }
  d.rayQueryRequested = strip;
  {
    std::lock_guard lock(gMutex);
    gDevices[dispatchKey(*pDevice)] = d;
  }
  LOG("device created (rayQuery requested: %d)", strip ? 1 : 0);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroyDevice(VkDevice device,
                                              const VkAllocationCallbacks* pAllocator) {
  void* key = dispatchKey(device);
  PFN_vkDestroyDevice next;
  vksoft::BuildKernels* kernels;
  vksoft::CellPool* cells;
  {
    std::lock_guard lock(gMutex);
    next = gDevices[key].DestroyDevice;
    kernels = gDevices[key].kernels;
    cells = gDevices[key].cells;
    gDevices.erase(key);
  }
  vksoft::destroyBuildKernels(kernels);
  vksoft::destroyCellPool(cells);
  next(device, pAllocator);
}


// ---- acceleration structures (M1) ----------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateAccelerationStructureKHR(
    VkDevice device, const VkAccelerationStructureCreateInfoKHR* ci, const VkAllocationCallbacks*,
    VkAccelerationStructureKHR* pAs) {
  if (ci->offset % vksoft::bvh::kAsAlignment != 0) return VK_ERROR_VALIDATION_FAILED_EXT;
  PFN_vkGetBufferDeviceAddress getAddr;
  {
    std::lock_guard lock(gMutex);
    auto it = gDevices.find(dispatchKey(device));
    if (it == gDevices.end()) return VK_ERROR_INITIALIZATION_FAILED;
    getAddr = it->second.GetBufferDeviceAddress;
  }
  if (!getAddr) return VK_ERROR_FEATURE_NOT_PRESENT;  // bufferDeviceAddress not enabled

  VkBufferDeviceAddressInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  info.buffer = ci->buffer;
  AsData as;
  as.buffer = ci->buffer;
  as.offset = ci->offset;
  as.size = ci->size;
  as.type = ci->type;
  as.address = getAddr(device, &info) + ci->offset;

  std::lock_guard lock(gMutex);
  uint64_t handle = kAsHandleTag | gNextAsHandle++;
  gAccel[handle] = as;
  *pAs = reinterpret_cast<VkAccelerationStructureKHR>(handle);
  LOG("AS created: type=%d size=%llu addr=0x%llx", int(as.type), (unsigned long long)as.size,
      (unsigned long long)as.address);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroyAccelerationStructureKHR(VkDevice device, VkAccelerationStructureKHR as,
                                                                const VkAllocationCallbacks*) {
  vksoft::CellPool* cells;
  {
    std::lock_guard lock(gMutex);
    gAccel.erase(reinterpret_cast<uint64_t>(as));
    cells = gDevices[dispatchKey(device)].cells;
  }
  vksoft::releaseCell(cells, reinterpret_cast<uint64_t>(as));
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL hook_GetAccelerationStructureDeviceAddressKHR(
    VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR* info) {
  std::lock_guard lock(gMutex);
  auto it = gAccel.find(reinterpret_cast<uint64_t>(info->accelerationStructure));
  return it == gAccel.end() ? 0 : it->second.address;
}

VKAPI_ATTR void VKAPI_CALL hook_GetAccelerationStructureBuildSizesKHR(
    VkDevice, VkAccelerationStructureBuildTypeKHR, const VkAccelerationStructureBuildGeometryInfoKHR* b,
    const uint32_t* pMaxPrimitiveCounts, VkAccelerationStructureBuildSizesInfoKHR* out) {
  using namespace vksoft::bvh;
  uint64_t n = 0;
  uint64_t perPrim = kTrianglePrimBytes;
  for (uint32_t i = 0; i < b->geometryCount; ++i) {
    n += pMaxPrimitiveCounts[i];
    const VkAccelerationStructureGeometryKHR& g = b->pGeometries ? b->pGeometries[i] : *b->ppGeometries[i];
    uint64_t pb = g.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR       ? kAabbPrimBytes
                : g.geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR   ? kInstanceBytes
                                                                     : kTrianglePrimBytes;
    perPrim = (i == 0) ? pb : std::max(perPrim, pb);  // mixed kinds: bound by the largest
  }
  out->accelerationStructureSize = accelerationStructureSize(n, perPrim);
  out->buildScratchSize = buildScratchSize(n);
  out->updateScratchSize = updateScratchSize(n);
}

// ---- command buffer state tracking (needed to restore state after our kernels) --------

VKAPI_ATTR void VKAPI_CALL hook_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
  PFN_vkCmdBindPipeline next;
  {
    std::lock_guard lock(gMutex);
    next = gDevices[dispatchKey(cb)].CmdBindPipeline;
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) gCmd[cb].computePipeline = p;
  }
  next(cb, bp, p);
}

VKAPI_ATTR void VKAPI_CALL hook_CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout,
                                                 VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                                                 const void* values) {
  PFN_vkCmdPushConstants next;
  {
    std::lock_guard lock(gMutex);
    next = gDevices[dispatchKey(cb)].CmdPushConstants;
    if ((stages & VK_SHADER_STAGE_COMPUTE_BIT) && offset + size <= sizeof(CmdState::push)) {
      CmdState& st = gCmd[cb];
      std::memcpy(st.push + offset, values, size);
      st.pushLayout = layout;
      st.pushStages = stages;
      st.pushMin = std::min(st.pushMin, offset);
      st.pushMax = std::max(st.pushMax, offset + size);
    }
  }
  next(cb, layout, stages, offset, size, values);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi) {
  PFN_vkBeginCommandBuffer next;
  {
    std::lock_guard lock(gMutex);
    next = gDevices[dispatchKey(cb)].BeginCommandBuffer;
    gCmd.erase(cb);
  }
  return next(cb, bi);
}

VKAPI_ATTR void VKAPI_CALL hook_FreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t count,
                                                   const VkCommandBuffer* cbs) {
  PFN_vkFreeCommandBuffers next;
  {
    std::lock_guard lock(gMutex);
    next = gDevices[dispatchKey(device)].FreeCommandBuffers;
    for (uint32_t i = 0; i < count; ++i) gCmd.erase(cbs[i]);
  }
  next(device, pool, count, cbs);
}

VKAPI_ATTR void VKAPI_CALL hook_CmdBuildAccelerationStructuresKHR(
    VkCommandBuffer cb, uint32_t infoCount, const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* ppRanges) {
  vksoft::BuildKernels* kernels;
  PFN_vkCmdBindPipeline bindPipeline;
  PFN_vkCmdPushConstants pushConstants;
  {
    std::lock_guard lock(gMutex);
    DeviceData& dd = gDevices[dispatchKey(cb)];
    kernels = dd.kernels;
    bindPipeline = dd.CmdBindPipeline;
    pushConstants = dd.CmdPushConstants;
  }
  if (!kernels) {
    LOG("CmdBuildAccelerationStructuresKHR: build kernels unavailable on this device (ignored)");
    return;
  }

  for (uint32_t i = 0; i < infoCount; ++i) {
    VkDeviceAddress dst = 0;
    {
      std::lock_guard lock(gMutex);
      auto it = gAccel.find(reinterpret_cast<uint64_t>(pInfos[i].dstAccelerationStructure));
      if (it != gAccel.end()) dst = it->second.address;
    }
    if (!dst) {
      LOG("build %u: unknown destination acceleration structure (ignored)", i);
      continue;
    }
    if (!vksoft::recordBuild(kernels, cb, pInfos[i], ppRanges[i], dst))
      LOG("build %u: request not supported yet (ignored)", i);
  }

  // Our kernels rebound the compute pipeline and overwrote compute push constants;
  // the app expects its own state to be intact after the build command.
  CmdState saved;
  {
    std::lock_guard lock(gMutex);
    auto it = gCmd.find(cb);
    if (it != gCmd.end()) saved = it->second;
  }
  if (saved.computePipeline) bindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, saved.computePipeline);
  if (saved.pushLayout && saved.pushMax > saved.pushMin)
    pushConstants(cb, saved.pushLayout, saved.pushStages, saved.pushMin, saved.pushMax - saved.pushMin,
                  saved.push + saved.pushMin);
}

// Our handles must never reach the driver. Until the GPU build lands, commands that
// consume them are no-ops (loudly logged) rather than crashes inside the driver.
#define STUB_CMD(name)                                                        \
  VKAPI_ATTR void VKAPI_CALL hook_##name(VkCommandBuffer, ...) {              \
    LOG(#name ": not implemented yet (ignored)");                             \
  }
STUB_CMD(CmdBuildAccelerationStructuresIndirectKHR)
STUB_CMD(CmdCopyAccelerationStructureKHR)
STUB_CMD(CmdCopyAccelerationStructureToMemoryKHR)
STUB_CMD(CmdCopyMemoryToAccelerationStructureKHR)
STUB_CMD(CmdWriteAccelerationStructuresPropertiesKHR)
#undef STUB_CMD


// ---- ray query support: shader modules and descriptors --------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* ci,
                                                       const VkAllocationCallbacks* alloc, VkShaderModule* pModule) {
  PFN_vkCreateShaderModule next;
  {
    std::lock_guard lock(gMutex);
    next = gDevices[dispatchKey(device)].CreateShaderModule;
  }
  std::vector<uint32_t> rewritten;
  std::string error;
  static const bool validate = std::getenv("VKSOFT_RT_VALIDATE") != nullptr;
  switch (vksoft::rewriteRayQueries(ci->pCode, ci->codeSize / 4, rewritten, error, validate)) {
    case vksoft::RewriteResult::NotNeeded:
      break;
    case vksoft::RewriteResult::Rewritten: {
      LOG("shader module rewritten for ray queries (%zu -> %zu words)", size_t(ci->codeSize / 4), rewritten.size());
      VkShaderModuleCreateInfo patched = *ci;
      patched.codeSize = rewritten.size() * 4;
      patched.pCode = rewritten.data();
      return next(device, &patched, alloc, pModule);
    }
    case vksoft::RewriteResult::Failed:
      std::fprintf(stderr, "[vksoft_rt] ray query rewrite failed, passing the module through: %s\n", error.c_str());
      break;
  }
  return next(device, ci, alloc, pModule);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* ci,
                                                              const VkAllocationCallbacks* alloc,
                                                              VkDescriptorSetLayout* pLayout) {
  PFN_vkCreateDescriptorSetLayout next;
  {
    std::lock_guard lock(gMutex);
    next = gDevices[dispatchKey(device)].CreateDescriptorSetLayout;
  }
  bool any = false;
  for (uint32_t i = 0; i < ci->bindingCount; ++i)
    if (ci->pBindings[i].descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) any = true;
  if (!any) return next(device, ci, alloc, pLayout);
  std::vector<VkDescriptorSetLayoutBinding> b(ci->pBindings, ci->pBindings + ci->bindingCount);
  for (auto& x : b)
    if (x.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  VkDescriptorSetLayoutCreateInfo patched = *ci;
  patched.pBindings = b.data();
  return next(device, &patched, alloc, pLayout);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* ci,
                                                         const VkAllocationCallbacks* alloc, VkDescriptorPool* pPool) {
  PFN_vkCreateDescriptorPool next;
  {
    std::lock_guard lock(gMutex);
    next = gDevices[dispatchKey(device)].CreateDescriptorPool;
  }
  bool any = false;
  for (uint32_t i = 0; i < ci->poolSizeCount; ++i)
    if (ci->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) any = true;
  if (!any) return next(device, ci, alloc, pPool);
  std::vector<VkDescriptorPoolSize> sizes;
  uint32_t extra = 0;
  for (uint32_t i = 0; i < ci->poolSizeCount; ++i) {
    if (ci->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) extra += ci->pPoolSizes[i].descriptorCount;
    else sizes.push_back(ci->pPoolSizes[i]);
  }
  bool merged = false;
  for (auto& sz : sizes)
    if (sz.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) { sz.descriptorCount += extra; merged = true; }
  if (!merged) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, extra});
  VkDescriptorPoolCreateInfo patched = *ci;
  patched.poolSizeCount = uint32_t(sizes.size());
  patched.pPoolSizes = sizes.data();
  return next(device, &patched, alloc, pPool);
}

VKAPI_ATTR void VKAPI_CALL hook_UpdateDescriptorSets(VkDevice device, uint32_t writeCount, const VkWriteDescriptorSet* pWrites,
                                                     uint32_t copyCount, const VkCopyDescriptorSet* pCopies) {
  PFN_vkUpdateDescriptorSets next;
  vksoft::CellPool* cells;
  {
    std::lock_guard lock(gMutex);
    DeviceData& dd = gDevices[dispatchKey(device)];
    next = dd.UpdateDescriptorSets;
    cells = dd.cells;
  }
  bool any = false;
  for (uint32_t i = 0; i < writeCount; ++i)
    if (pWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) any = true;
  if (!any || !cells) return next(device, writeCount, pWrites, copyCount, pCopies);

  std::vector<VkWriteDescriptorSet> writes(pWrites, pWrites + writeCount);
  std::vector<std::vector<VkDescriptorBufferInfo>> infos(writeCount);
  for (uint32_t i = 0; i < writeCount; ++i) {
    if (writes[i].descriptorType != VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) continue;
    const VkWriteDescriptorSetAccelerationStructureKHR* as = nullptr;
    for (auto* n = reinterpret_cast<const VkBaseInStructure*>(writes[i].pNext); n; n = n->pNext)
      if (n->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
        as = reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureKHR*>(n);
    infos[i].resize(writes[i].descriptorCount);
    for (uint32_t j = 0; j < writes[i].descriptorCount; ++j) {
      uint64_t key = as ? reinterpret_cast<uint64_t>(as->pAccelerationStructures[j]) : 0;
      uint64_t addr = 0;
      if (key) {
        std::lock_guard lock(gMutex);
        auto it = gAccel.find(key);
        if (it != gAccel.end()) addr = it->second.address;
      }
      if (!vksoft::cellInfo(cells, key, addr, &infos[i][j])) {
        std::fprintf(stderr, "[vksoft_rt] acceleration structure cell pool exhausted\n");
        vksoft::cellInfo(cells, 0, 0, &infos[i][j]);
      }
    }
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pNext = nullptr;
    writes[i].pBufferInfo = infos[i].data();
  }
  next(device, writeCount, writes.data(), copyCount, pCopies);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL hook_GetDeviceProcAddr(VkDevice device, const char* name);

struct Hook {
  const char* name;
  PFN_vkVoidFunction fn;
};
#define H(n) {"vk" #n, reinterpret_cast<PFN_vkVoidFunction>(hook_##n)}
const Hook kInstanceHooks[] = {
    H(DestroyInstance),
    H(EnumerateDeviceExtensionProperties),
    H(GetPhysicalDeviceFeatures2),
    {"vkGetPhysicalDeviceFeatures2KHR", reinterpret_cast<PFN_vkVoidFunction>(hook_GetPhysicalDeviceFeatures2KHR)},
    H(CreateDevice),
    H(GetDeviceProcAddr),
};
// Hot-path commands we only intercept on devices where the app uses acceleration structures.
const Hook kTrackingHooks[] = {
    H(CreateShaderModule),
    H(CreateDescriptorSetLayout),
    H(CreateDescriptorPool),
    H(UpdateDescriptorSets),
    H(CmdBindPipeline),
    H(CmdPushConstants),
    H(BeginCommandBuffer),
    H(FreeCommandBuffers),
};
const Hook kDeviceHooks[] = {
    H(DestroyDevice),
    H(GetDeviceProcAddr),
    H(CreateAccelerationStructureKHR),
    H(DestroyAccelerationStructureKHR),
    H(GetAccelerationStructureDeviceAddressKHR),
    H(GetAccelerationStructureBuildSizesKHR),
    H(CmdBuildAccelerationStructuresKHR),
    H(CmdBuildAccelerationStructuresIndirectKHR),
    H(CmdCopyAccelerationStructureKHR),
    H(CmdCopyAccelerationStructureToMemoryKHR),
    H(CmdCopyMemoryToAccelerationStructureKHR),
    H(CmdWriteAccelerationStructuresPropertiesKHR),
};
#undef H

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL hook_GetDeviceProcAddr(VkDevice device, const char* name) {
  for (auto& h : kDeviceHooks)
    if (std::strcmp(h.name, name) == 0) return h.fn;
  PFN_vkGetDeviceProcAddr next;
  bool track;
  {
    std::lock_guard lock(gMutex);
    auto it = gDevices.find(dispatchKey(device));
    if (it == gDevices.end()) return nullptr;
    next = it->second.gdpa;
    track = it->second.asEnabled;
  }
  if (track)
    for (auto& h : kTrackingHooks)
      if (std::strcmp(h.name, name) == 0) return h.fn;
  return next(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL hook_GetInstanceProcAddr(VkInstance instance,
                                                                  const char* name) {
  if (std::strcmp(name, "vkCreateInstance") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(hook_CreateInstance);
  if (!instance) return nullptr;
  for (auto& h : kInstanceHooks)
    if (std::strcmp(h.name, name) == 0) return h.fn;
  for (auto& h : kDeviceHooks)  // apps may fetch device functions via the instance
    if (std::strcmp(h.name, name) == 0) return h.fn;
  PFN_vkGetInstanceProcAddr next;
  {
    std::lock_guard lock(gMutex);
    auto it = gInstances.find(dispatchKey(instance));
    if (it == gInstances.end()) return nullptr;
    next = it->second.gipa;
  }
  return next(instance, name);
}

}  // namespace

VKSOFT_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
  if (pVersionStruct->loaderLayerInterfaceVersion < 2) return VK_ERROR_INITIALIZATION_FAILED;
  pVersionStruct->loaderLayerInterfaceVersion = 2;
  pVersionStruct->pfnGetInstanceProcAddr = hook_GetInstanceProcAddr;
  pVersionStruct->pfnGetDeviceProcAddr = hook_GetDeviceProcAddr;
  pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
  return VK_SUCCESS;
}
