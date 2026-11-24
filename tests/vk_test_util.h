// Shared helpers for the GPU integration tests (single translation unit each).
#pragma once
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(c)                                                          \
  do {                                                                    \
    if (!(c)) {                                                           \
      std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)
#define VK(c) CHECK((c) == VK_SUCCESS)

namespace {

VkPhysicalDevice gPd;
VkDevice gDev;
VkQueue gQueue;
uint32_t gQueueFamily;
VkCommandPool gPool;

PFN_vkCreateAccelerationStructureKHR pCreateAS;
PFN_vkDestroyAccelerationStructureKHR pDestroyAS;
PFN_vkGetAccelerationStructureBuildSizesKHR pSizes;
PFN_vkCmdBuildAccelerationStructuresKHR pBuild;
PFN_vkGetBufferDeviceAddress pBufAddr;

uint32_t findMemType(uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(gPd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  CHECK(false);
  return 0;
}

struct Buf {
  VkBuffer buf{};
  VkDeviceMemory mem{};
  VkDeviceAddress addr{};
  void* map{};
  VkDeviceSize size{};
};

Buf makeBuf(VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
  Buf b;
  b.size = size;
  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ci.size = size;
  ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VK(vkCreateBuffer(gDev, &ci, nullptr, &b.buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(gDev, b.buf, &mr);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &fi;
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemType(mr.memoryTypeBits,
      host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
           : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK(vkAllocateMemory(gDev, &ai, nullptr, &b.mem));
  VK(vkBindBufferMemory(gDev, b.buf, b.mem, 0));
  VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bi.buffer = b.buf;
  b.addr = pBufAddr(gDev, &bi);
  if (host) VK(vkMapMemory(gDev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
  return b;
}
void freeBuf(Buf& b) {
  vkDestroyBuffer(gDev, b.buf, nullptr);
  vkFreeMemory(gDev, b.mem, nullptr);
}

void initVulkan(bool rayQuery = false) {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  VK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n);
  vkEnumeratePhysicalDevices(inst, &n, pds.data());
  for (auto d : pds) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { gPd = d; break; }
  }
  CHECK(gPd);
  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(gPd, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(gPd, &qn, qf.data());
  gQueueFamily = UINT32_MAX;
  for (uint32_t i = 0; i < qn; ++i)
    if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { gQueueFamily = i; break; }
  CHECK(gQueueFamily != UINT32_MAX);

  const char* exts[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME};
  VkPhysicalDeviceRayQueryFeaturesKHR rqf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  rqf.rayQuery = VK_TRUE;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.bufferDeviceAddress = VK_TRUE;
  VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  asf.accelerationStructure = VK_TRUE;
  asf.pNext = &f12;
  float prio = 1.f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = gQueueFamily;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  if (rayQuery) rqf.pNext = &asf;
  dci.pNext = rayQuery ? static_cast<void*>(&rqf) : static_cast<void*>(&asf);  // shaderInt64 deliberately NOT enabled; the layer must force it
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = rayQuery ? 3 : 2;
  dci.ppEnabledExtensionNames = exts;
  VK(vkCreateDevice(gPd, &dci, nullptr, &gDev));
  vkGetDeviceQueue(gDev, gQueueFamily, 0, &gQueue);
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = gQueueFamily;
  VK(vkCreateCommandPool(gDev, &pci, nullptr, &gPool));

#define LOAD(n) p##n = (PFN_vk##n)vkGetDeviceProcAddr(gDev, "vk" #n); CHECK(p##n)
  pCreateAS = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(gDev, "vkCreateAccelerationStructureKHR");
  pDestroyAS = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(gDev, "vkDestroyAccelerationStructureKHR");
  pSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(gDev, "vkGetAccelerationStructureBuildSizesKHR");
  pBuild = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(gDev, "vkCmdBuildAccelerationStructuresKHR");
  pBufAddr = (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(gDev, "vkGetBufferDeviceAddress");
#undef LOAD
  CHECK(pCreateAS && pDestroyAS && pSizes && pBuild && pBufAddr);
}


void submitAndWait(VkCommandBuffer cb) {
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK(vkCreateFence(gDev, &fci, nullptr, &fence));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  VK(vkQueueSubmit(gQueue, 1, &si, fence));
  VK(vkWaitForFences(gDev, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000));
  vkDestroyFence(gDev, fence, nullptr);
}

VkCommandBuffer beginCb() {
  VkCommandBuffer cb;
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = gPool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VK(vkAllocateCommandBuffers(gDev, &cai, &cb));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK(vkBeginCommandBuffer(cb, &bi));
  return cb;
}

}  // namespace
