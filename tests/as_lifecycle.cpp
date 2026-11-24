// Integration test: creates a device with acceleration structures (through the
// layer when enabled), queries build sizes, creates/destroys AS objects and
// checks device addresses. Run with the layer enabled; see CMake `as_lifecycle`.
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(c)                                                              \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);     \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#define VK(c) CHECK((c) == VK_SUCCESS)

static uint32_t findMemType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  CHECK(false);
  return 0;
}

int main() {
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
  VkPhysicalDevice pd = VK_NULL_HANDLE;
  for (auto d : pds) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = d; break; }
  }
  CHECK(pd);

  const char* exts[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.bufferDeviceAddress = VK_TRUE;
  VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  asf.accelerationStructure = VK_TRUE;
  asf.pNext = &f12;
  float prio = 1.f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &asf;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 2;
  dci.ppEnabledExtensionNames = exts;
  VkDevice dev;
  VK(vkCreateDevice(pd, &dci, nullptr, &dev));

  auto pCreate = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR");
  auto pDestroy = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR");
  auto pAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureDeviceAddressKHR");
  auto pSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR");
  CHECK(pCreate && pDestroy && pAddr && pSizes);

  // Build sizes for a 1000-triangle BLAS.
  VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  geom.geometry.triangles.vertexStride = 12;
  geom.geometry.triangles.maxVertex = 2999;
  geom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &geom;
  uint32_t prims = 1000;
  VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  pSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &prims, &sz);
  std::printf("1000 tris: AS=%llu build scratch=%llu update scratch=%llu\n",
              (unsigned long long)sz.accelerationStructureSize,
              (unsigned long long)sz.buildScratchSize, (unsigned long long)sz.updateScratchSize);
  CHECK(sz.accelerationStructureSize >= 1000 * 36);
  CHECK(sz.buildScratchSize > 0);

  // Backing buffer for two ASes at distinct 256-aligned offsets.
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = sz.accelerationStructureSize * 2;
  bci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBuffer buf;
  VK(vkCreateBuffer(dev, &bci, nullptr, &buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buf, &mr);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &fi;
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex = findMemType(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkDeviceMemory mem;
  VK(vkAllocateMemory(dev, &mai, nullptr, &mem));
  VK(vkBindBufferMemory(dev, buf, mem, 0));

  VkAccelerationStructureKHR a0, a1;
  VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  ci.buffer = buf;
  ci.size = sz.accelerationStructureSize;
  ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  ci.offset = 0;
  VK(pCreate(dev, &ci, nullptr, &a0));
  ci.offset = sz.accelerationStructureSize;
  VK(pCreate(dev, &ci, nullptr, &a1));

  VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  ai.accelerationStructure = a0;
  VkDeviceAddress d0 = pAddr(dev, &ai);
  ai.accelerationStructure = a1;
  VkDeviceAddress d1 = pAddr(dev, &ai);
  std::printf("addresses: 0x%llx 0x%llx\n", (unsigned long long)d0, (unsigned long long)d1);
  CHECK(d0 != 0 && d1 == d0 + sz.accelerationStructureSize);

  pDestroy(dev, a0, nullptr);
  pDestroy(dev, a1, nullptr);
  vkDestroyBuffer(dev, buf, nullptr);
  vkFreeMemory(dev, mem, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);
  std::puts("as_lifecycle passed");
  return 0;
}
