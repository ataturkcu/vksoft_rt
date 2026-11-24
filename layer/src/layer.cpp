// vksoft_rt - M0: layer skeleton + VK_KHR_ray_query advertisement.
//
// This milestone only *advertises* VK_KHR_ray_query / rayQuery so we can verify
// the interception path with `vulkaninfo`. Nothing implements ray queries yet,
// so the layer is an explicit (opt-in) layer and must never be enabled
// globally. See docs/design.md.

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include "bvh_format.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <mutex>
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
  bool rayQueryRequested{false};  // app asked for rayQuery (emulated by us)
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
  bool strip = wantsRayQuery && canAdvertiseRayQuery(*inst, pd);
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
  {
    std::lock_guard lock(gMutex);
    next = gDevices[key].DestroyDevice;
    gDevices.erase(key);
  }
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

VKAPI_ATTR void VKAPI_CALL hook_DestroyAccelerationStructureKHR(VkDevice, VkAccelerationStructureKHR as,
                                                                const VkAllocationCallbacks*) {
  std::lock_guard lock(gMutex);
  gAccel.erase(reinterpret_cast<uint64_t>(as));
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

// Our handles must never reach the driver. Until the GPU build lands, commands that
// consume them are no-ops (loudly logged) rather than crashes inside the driver.
#define STUB_CMD(name)                                                        \
  VKAPI_ATTR void VKAPI_CALL hook_##name(VkCommandBuffer, ...) {              \
    LOG(#name ": not implemented yet (ignored)");                             \
  }
STUB_CMD(CmdBuildAccelerationStructuresKHR)
STUB_CMD(CmdBuildAccelerationStructuresIndirectKHR)
STUB_CMD(CmdCopyAccelerationStructureKHR)
STUB_CMD(CmdCopyAccelerationStructureToMemoryKHR)
STUB_CMD(CmdCopyMemoryToAccelerationStructureKHR)
STUB_CMD(CmdWriteAccelerationStructuresPropertiesKHR)
#undef STUB_CMD

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
  {
    std::lock_guard lock(gMutex);
    auto it = gDevices.find(dispatchKey(device));
    if (it == gDevices.end()) return nullptr;
    next = it->second.gdpa;
  }
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
