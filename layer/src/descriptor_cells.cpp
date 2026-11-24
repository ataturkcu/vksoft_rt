#include "descriptor_cells.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vksoft {

constexpr uint32_t kCellBytes = 16;
constexpr uint32_t kCapacity = 65536;  // 1 MiB

struct CellPool {
  VkDevice device{};
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  uint8_t* mapped{};
  PFN_vkDestroyBuffer DestroyBuffer{};
  PFN_vkFreeMemory FreeMemory{};
  PFN_vkUnmapMemory UnmapMemory{};
  std::mutex mutex;
  uint32_t next = 1;  // cell 0 is the permanent null cell
  std::vector<uint32_t> freeList;
  std::unordered_map<uint64_t, uint32_t> cellOf;
};

CellPool* createCellPool(VkInstance instance, VkPhysicalDevice pd, VkDevice device, PFN_vkGetInstanceProcAddr gipa,
                         PFN_vkGetDeviceProcAddr gdpa) {
#define FN(name) reinterpret_cast<PFN_vk##name>(gdpa(device, "vk" #name))
  auto getMemProps = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(gipa(instance, "vkGetPhysicalDeviceMemoryProperties"));
  auto createBuffer = FN(CreateBuffer);
  auto getReqs = FN(GetBufferMemoryRequirements);
  auto allocate = FN(AllocateMemory);
  auto bind = FN(BindBufferMemory);
  auto map = FN(MapMemory);
  auto* p = new CellPool;
  p->device = device;
  p->DestroyBuffer = FN(DestroyBuffer);
  p->FreeMemory = FN(FreeMemory);
  p->UnmapMemory = FN(UnmapMemory);
#undef FN
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = uint64_t(kCapacity) * kCellBytes;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bool ok = createBuffer(device, &bci, nullptr, &p->buffer) == VK_SUCCESS;
  if (ok) {
    VkMemoryRequirements mr;
    getReqs(device, p->buffer, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    getMemProps(pd, &mp);
    uint32_t type = UINT32_MAX;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
      if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { type = i; break; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    void* ptr = nullptr;
    ok = type != UINT32_MAX && allocate(device, &ai, nullptr, &p->memory) == VK_SUCCESS &&
         bind(device, p->buffer, p->memory, 0) == VK_SUCCESS &&
         map(device, p->memory, 0, VK_WHOLE_SIZE, 0, &ptr) == VK_SUCCESS;
    p->mapped = static_cast<uint8_t*>(ptr);
  }
  if (!ok) {
    std::fprintf(stderr, "[vksoft_rt] could not create the acceleration structure cell pool\n");
    destroyCellPool(p);
    return nullptr;
  }
  std::memset(p->mapped, 0, kCellBytes);  // cell 0 = null
  return p;
}

void destroyCellPool(CellPool* p) {
  if (!p) return;
  if (p->mapped) p->UnmapMemory(p->device, p->memory);
  if (p->buffer) p->DestroyBuffer(p->device, p->buffer, nullptr);
  if (p->memory) p->FreeMemory(p->device, p->memory, nullptr);
  delete p;
}

bool cellInfo(CellPool* p, uint64_t key, uint64_t address, VkDescriptorBufferInfo* out) {
  uint32_t cell = 0;
  if (key != 0) {
    std::lock_guard lock(p->mutex);
    auto it = p->cellOf.find(key);
    if (it != p->cellOf.end()) {
      cell = it->second;
    } else {
      if (!p->freeList.empty()) { cell = p->freeList.back(); p->freeList.pop_back(); }
      else if (p->next < kCapacity) cell = p->next++;
      else return false;
      p->cellOf[key] = cell;
    }
    std::memcpy(p->mapped + uint64_t(cell) * kCellBytes, &address, sizeof(address));
  }
  out->buffer = p->buffer;
  out->offset = uint64_t(cell) * kCellBytes;
  out->range = kCellBytes;
  return true;
}

void releaseCell(CellPool* p, uint64_t key) {
  if (!p || key == 0) return;
  std::lock_guard lock(p->mutex);
  auto it = p->cellOf.find(key);
  if (it == p->cellOf.end()) return;
  std::memset(p->mapped + uint64_t(it->second) * kCellBytes, 0, kCellBytes);
  p->freeList.push_back(it->second);
  p->cellOf.erase(it);
}

}  // namespace vksoft
