// Address cells: rewritten ray query shaders read the TLAS address from a small storage
// buffer binding instead of an acceleration structure descriptor. Every acceleration
// structure that is written into a descriptor gets a 16-byte cell holding its device
// address; the descriptor becomes a storage buffer descriptor over that cell.
#pragma once

#include <vulkan/vulkan.h>

namespace vksoft {

struct CellPool;

CellPool* createCellPool(VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr);
void destroyCellPool(CellPool*);

// Fills `out` with the buffer range of the cell for acceleration structure `key` (its fake
// handle value; 0 = null descriptor -> a cell that always reads 0). Returns false if the
// pool is exhausted.
bool cellInfo(CellPool*, uint64_t key, uint64_t address, VkDescriptorBufferInfo* out);
void releaseCell(CellPool*, uint64_t key);

}  // namespace vksoft
