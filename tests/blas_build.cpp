// GPU integration test for the BLAS builder: builds triangle BLASes through the layer on
// the real device, reads the structure back and verifies it on the CPU.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../layer/src/bvh_format.h"

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

void initVulkan() {
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
                        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
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
  dci.pNext = &asf;  // note: shaderInt64 deliberately NOT enabled; the layer must force it
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 2;
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

struct Tri { float v[9]; uint32_t geom, prim; };

struct Spec {
  const char* name;
  uint32_t nPlain, nIndexed32, nIndexed16;
  bool identical = false;  // all triangles at the same place (Morton key ties)
};

// Row-major 3x4 transform applied to the u32-indexed geometry.
const float kXform[12] = {2, 0, 0, 5, 0, 3, 0, -1, 0, 0, 0.5f, 2};

void applyXform(const float* m, float* p) {
  float x = p[0], y = p[1], z = p[2];
  p[0] = m[0] * x + m[1] * y + m[2] * z + m[3];
  p[1] = m[4] * x + m[5] * y + m[6] * z + m[7];
  p[2] = m[8] * x + m[9] * y + m[10] * z + m[11];
}

struct Aabb { float mn[3], mx[3]; };
Aabb emptyBox() { return {{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}}; }
void grow(Aabb& a, const Aabb& b) {
  for (int i = 0; i < 3; ++i) { a.mn[i] = std::min(a.mn[i], b.mn[i]); a.mx[i] = std::max(a.mx[i], b.mx[i]); }
}
bool near(float a, float b) { return std::fabs(a - b) <= 1e-4f * (1.f + std::fabs(a) + std::fabs(b)); }

struct Readback {
  const uint32_t* w;  // whole AS as words
  float f(uint32_t i) const { float x; std::memcpy(&x, &w[i], 4); return x; }
};

struct Verifier {
  Readback rb;
  uint32_t nodeBase, primBase;       // word offsets
  uint32_t leafCount = 0, internalCount = 0;
  std::vector<int> seen;             // per prim slot
  std::vector<Tri> primsOut;         // by leaf
  Aabb visit(uint32_t ref) {
    if (ref & 0x80000000u) {
      uint32_t k = ref & 0x7FFFFFFFu;
      CHECK(k < seen.size());
      CHECK(seen[k]++ == 0);  // every leaf referenced exactly once
      ++leafCount;
      const uint32_t o = primBase + k * 12;
      Tri t{};
      for (int i = 0; i < 9; ++i) t.v[i] = rb.f(o + i);
      t.geom = rb.w[o + 9];
      t.prim = rb.w[o + 10];
      primsOut[k] = t;
      Aabb b = emptyBox();
      for (int v = 0; v < 3; ++v)
        for (int a = 0; a < 3; ++a) {
          b.mn[a] = std::min(b.mn[a], t.v[v * 3 + a]);
          b.mx[a] = std::max(b.mx[a], t.v[v * 3 + a]);
        }
      return b;
    }
    ++internalCount;
    const uint32_t o = nodeBase + ref * 16;
    Aabb result = emptyBox();
    for (int c = 0; c < 2; ++c) {
      Aabb stored;
      for (int a = 0; a < 3; ++a) { stored.mn[a] = rb.f(o + c * 6 + a); stored.mx[a] = rb.f(o + c * 6 + 3 + a); }
      Aabb actual = visit(rb.w[o + 12 + c]);
      for (int a = 0; a < 3; ++a) {
        if (!near(stored.mn[a], actual.mn[a]) || !near(stored.mx[a], actual.mx[a])) {
          std::fprintf(stderr, "node %u child %d axis %d: stored [%g, %g] actual [%g, %g] childRef=0x%x\n", ref, c,
                       a, stored.mn[a], stored.mx[a], actual.mn[a], actual.mx[a], rb.w[o + 12 + c]);
        }
        CHECK(near(stored.mn[a], actual.mn[a]));
        CHECK(near(stored.mx[a], actual.mx[a]));
      }
      grow(result, actual);
    }
    return result;
  }
};

void runCase(const Spec& sp) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> U(-10.f, 10.f), S(-0.3f, 0.3f);
  auto makeTri = [&](float* v) {
    float c[3] = {U(rng), U(rng), U(rng)};
    if (sp.identical) c[0] = c[1] = c[2] = 1.f;
    for (int i = 0; i < 3; ++i)
      for (int a = 0; a < 3; ++a) v[i * 3 + a] = c[a] + (sp.identical ? 0.f : S(rng));
  };

  // --- geometry data -------------------------------------------------------
  std::vector<Tri> expected;
  std::vector<float> vPlain(size_t(sp.nPlain) * 9);
  for (uint32_t i = 0; i < sp.nPlain; ++i) {
    makeTri(&vPlain[i * 9]);
    Tri t{}; std::memcpy(t.v, &vPlain[i * 9], 36); t.geom = 0; t.prim = i; expected.push_back(t);
  }
  const uint32_t nv32 = std::max(3u, sp.nIndexed32 * 2 / 3 + 3), nv16 = std::max(3u, std::min(60000u, sp.nIndexed16 * 2 / 3 + 3));
  std::vector<float> v32(size_t(nv32) * 3), v16(size_t(nv16) * 3);
  for (auto& x : v32) x = U(rng);
  for (auto& x : v16) x = U(rng);
  std::vector<uint32_t> i32(size_t(sp.nIndexed32) * 3);
  std::vector<uint16_t> i16(size_t(sp.nIndexed16) * 3);
  for (auto& x : i32) x = rng() % nv32;
  for (auto& x : i16) x = rng() % nv16;
  for (uint32_t i = 0; i < sp.nIndexed32; ++i) {
    Tri t{};
    for (int v = 0; v < 3; ++v) { std::memcpy(&t.v[v * 3], &v32[size_t(i32[i * 3 + v]) * 3], 12); applyXform(kXform, &t.v[v * 3]); }
    t.geom = 1; t.prim = i; expected.push_back(t);
  }
  for (uint32_t i = 0; i < sp.nIndexed16; ++i) {
    Tri t{};
    for (int v = 0; v < 3; ++v) std::memcpy(&t.v[v * 3], &v16[size_t(i16[i * 3 + v]) * 3], 12);
    t.geom = 2; t.prim = i; expected.push_back(t);
  }
  const uint32_t N = uint32_t(expected.size());

  auto upload = [&](const void* p, size_t bytes) {
    Buf b = makeBuf(std::max<size_t>(bytes, 16), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    if (bytes) std::memcpy(b.map, p, bytes);
    return b;
  };
  const VkBufferUsageFlags geomUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  (void)geomUsage;
  Buf bPlain = upload(vPlain.data(), vPlain.size() * 4), bV32 = upload(v32.data(), v32.size() * 4),
      bV16 = upload(v16.data(), v16.size() * 4), bI32 = upload(i32.data(), i32.size() * 4),
      bI16 = upload(i16.data(), i16.size() * 2), bX = upload(kXform, sizeof(kXform));

  std::vector<VkAccelerationStructureGeometryKHR> geoms;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
  std::vector<uint32_t> counts;
  auto addGeom = [&](uint32_t count, const Buf& vb, uint32_t maxVertex, VkIndexType it, const Buf* ib,
                     const Buf* xf) {
    VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& t = g.geometry.triangles;
    t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    t.vertexData.deviceAddress = vb.addr;
    t.vertexStride = 12;
    t.maxVertex = maxVertex;
    t.indexType = it;
    if (ib) t.indexData.deviceAddress = ib->addr;
    if (xf) t.transformData.deviceAddress = xf->addr;
    geoms.push_back(g);
    counts.push_back(count);
    ranges.push_back({count, 0, 0, 0});
  };
  addGeom(sp.nPlain, bPlain, std::max(1u, sp.nPlain * 3) - 1, VK_INDEX_TYPE_NONE_KHR, nullptr, nullptr);
  addGeom(sp.nIndexed32, bV32, nv32 - 1, VK_INDEX_TYPE_UINT32, &bI32, &bX);
  addGeom(sp.nIndexed16, bV16, nv16 - 1, VK_INDEX_TYPE_UINT16, &bI16, nullptr);

  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  bgi.geometryCount = uint32_t(geoms.size());
  bgi.pGeometries = geoms.data();
  VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  pSizes(gDev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, counts.data(), &sz);

  Buf asBuf = makeBuf(sz.accelerationStructureSize,
                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
  Buf scratch = makeBuf(sz.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  Buf readback = makeBuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  std::memset(readback.map, 0xCD, readback.size);

  VkAccelerationStructureKHR as;
  VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  ci.buffer = asBuf.buf;
  ci.size = sz.accelerationStructureSize;
  ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  VK(pCreateAS(gDev, &ci, nullptr, &as));
  bgi.dstAccelerationStructure = as;
  bgi.scratchData.deviceAddress = scratch.addr;

  VkCommandBuffer cb;
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = gPool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VK(vkAllocateCommandBuffers(gDev, &cai, &cb));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK(vkBeginCommandBuffer(cb, &bi));
  const VkAccelerationStructureBuildRangeInfoKHR* pr = ranges.data();
  pBuild(cb, 1, &bgi, &pr);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 1, &mb, 0, nullptr, 0, nullptr);
  VkBufferCopy cp{0, 0, sz.accelerationStructureSize};
  vkCmdCopyBuffer(cb, asBuf.buf, readback.buf, 1, &cp);
  VK(vkEndCommandBuffer(cb));

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK(vkCreateFence(gDev, &fci, nullptr, &fence));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  auto t0 = std::chrono::steady_clock::now();
  VK(vkQueueSubmit(gQueue, 1, &si, fence));
  VK(vkWaitForFences(gDev, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000));
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  // --- verification --------------------------------------------------------
  Readback rb{static_cast<const uint32_t*>(readback.map)};
  using namespace vksoft::bvh;
  CHECK(rb.w[0] == 0x42534B56u);
  CHECK(rb.w[2] == N);
  CHECK(rb.w[3] == (N ? N - 1 : 0));
  const uint32_t root = rb.w[4];
  if (N == 0) {
    CHECK(root == 0xFFFFFFFFu);
  } else {
    Verifier v{rb, uint32_t(rb.w[5] / 4), uint32_t(rb.w[6] / 4)};
    CHECK(rb.w[5] == kHeaderBytes);
    CHECK(rb.w[6] == kHeaderBytes + maxInternalNodes(N) * kNodeBytes);
    CHECK(rb.w[6] + uint64_t(N) * kTrianglePrimBytes <= sz.accelerationStructureSize);
    v.seen.assign(N, 0);
    v.primsOut.assign(N, Tri{});
    CHECK(((root & 0x80000000u) != 0) == (N == 1));
    Aabb all = v.visit(root);
    CHECK(v.leafCount == N);
    CHECK(v.internalCount == (N ? N - 1 : 0));
    for (int a = 0; a < 3; ++a) {  // header bounds == union of everything
      CHECK(near(rb.f(8 + a), all.mn[a]));
      CHECK(near(rb.f(11 + a), all.mx[a]));
    }
    // The leaves must be exactly the expected triangles (ids + transformed vertices).
    std::vector<const Tri*> byId(N, nullptr);
    std::vector<uint32_t> base = {0, sp.nPlain, sp.nPlain + sp.nIndexed32};
    for (const Tri& t : v.primsOut) {
      CHECK(t.geom < 3);
      uint32_t idx = base[t.geom] + t.prim;
      CHECK(idx < N && !byId[idx]);
      byId[idx] = &t;
      for (int i = 0; i < 9; ++i) CHECK(near(t.v[i], expected[idx].v[i]));
      CHECK(t.geom == expected[idx].geom && t.prim == expected[idx].prim);
    }
  }
  std::printf("  %-26s N=%-7u AS=%-9llu build+copy %.2f ms  OK\n", sp.name, N,
              (unsigned long long)sz.accelerationStructureSize, ms);

  vkDestroyFence(gDev, fence, nullptr);
  vkFreeCommandBuffers(gDev, gPool, 1, &cb);
  pDestroyAS(gDev, as, nullptr);
  for (Buf* b : {&bPlain, &bV32, &bV16, &bI32, &bI16, &bX, &asBuf, &scratch, &readback}) freeBuf(*b);
}

}  // namespace

int main() {
  initVulkan();
  std::puts("blas_build:");
  runCase({"empty", 0, 0, 0});
  runCase({"single triangle", 1, 0, 0});
  runCase({"two triangles", 2, 0, 0});
  runCase({"255+1 (block edge)", 255, 1, 0});
  runCase({"idx16 only", 0, 0, 700});
  runCase({"identical triangles (ties)", 1000, 0, 0, true});
  runCase({"mixed 3 geometries", 3000, 2000, 500});
  runCase({"100k plain", 100000, 0, 0});
  runCase({"1M mixed", 600000, 300000, 100000});
  std::puts("blas_build passed");
  return 0;
}
