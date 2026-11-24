// Scene-building helpers for GPU tests: random BLASes, instance transforms, TLAS builds.
#pragma once
#include "vk_test_util.h"

namespace {

PFN_vkGetAccelerationStructureDeviceAddressKHR pAddr;

struct Aabb { float mn[3], mx[3]; };
Aabb emptyBox() { return {{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}}; }
void grow(Aabb& a, const Aabb& b) {
  for (int i = 0; i < 3; ++i) { a.mn[i] = std::min(a.mn[i], b.mn[i]); a.mx[i] = std::max(a.mx[i], b.mx[i]); }
}
bool near(float a, float b) {
  if (std::isinf(a) || std::isinf(b)) return a == b;
  return std::fabs(a - b) <= 1e-4f * (1.f + std::fabs(a) + std::fabs(b));
}

struct Blas {
  VkAccelerationStructureKHR as{};
  Buf asBuf, scratch, verts;
  VkDeviceAddress addr{};
  Aabb bounds{};
  uint32_t nTris{};
};

Blas makeBlas(std::mt19937& rng, uint32_t nTris, float spread) {
  Blas b;
  b.nTris = nTris;
  std::uniform_real_distribution<float> U(-spread, spread), S(-0.2f, 0.2f);
  std::vector<float> v(size_t(nTris) * 9);
  b.bounds = emptyBox();
  for (uint32_t t = 0; t < nTris; ++t) {
    float c[3] = {U(rng), U(rng), U(rng)};
    for (int i = 0; i < 3; ++i)
      for (int a = 0; a < 3; ++a) {
        float x = c[a] + S(rng);
        v[t * 9 + i * 3 + a] = x;
        b.bounds.mn[a] = std::min(b.bounds.mn[a], x);
        b.bounds.mx[a] = std::max(b.bounds.mx[a], x);
      }
  }
  b.verts = makeBuf(v.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  std::memcpy(b.verts.map, v.data(), v.size() * 4);

  VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  auto& t = g.geometry.triangles;
  t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  t.vertexData.deviceAddress = b.verts.addr;
  t.vertexStride = 12;
  t.maxVertex = nTris * 3 - 1;
  t.indexType = VK_INDEX_TYPE_NONE_KHR;
  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &g;
  VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  pSizes(gDev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &nTris, &sz);
  b.asBuf = makeBuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false);
  b.scratch = makeBuf(sz.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  ci.buffer = b.asBuf.buf;
  ci.size = sz.accelerationStructureSize;
  ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  VK(pCreateAS(gDev, &ci, nullptr, &b.as));
  VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  ai.accelerationStructure = b.as;
  b.addr = pAddr(gDev, &ai);

  bgi.dstAccelerationStructure = b.as;
  bgi.scratchData.deviceAddress = b.scratch.addr;
  VkAccelerationStructureBuildRangeInfoKHR range{nTris, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;
  VkCommandBuffer cb = beginCb();
  pBuild(cb, 1, &bgi, &pr);
  VK(vkEndCommandBuffer(cb));
  submitAndWait(cb);
  return b;
}


struct Mat { float m[12]; };

Mat randomTransform(std::mt19937& rng, bool identity) {
  if (identity) return {{1, 0, 0, 3, 0, 1, 0, 4, 0, 0, 1, 5}};
  std::uniform_real_distribution<float> U(-1.f, 1.f), T(-50.f, 50.f), Sc(0.5f, 2.f);
  float ax[3] = {U(rng), U(rng), U(rng)};
  float len = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]) + 1e-6f;
  for (float& a : ax) a /= len;
  float ang = U(rng) * 3.14159f, c = std::cos(ang), s = std::sin(ang), k = 1 - c;
  float sc[3] = {Sc(rng), Sc(rng), Sc(rng)};
  float R[9] = {c + ax[0] * ax[0] * k, ax[0] * ax[1] * k - ax[2] * s, ax[0] * ax[2] * k + ax[1] * s,
                ax[1] * ax[0] * k + ax[2] * s, c + ax[1] * ax[1] * k, ax[1] * ax[2] * k - ax[0] * s,
                ax[2] * ax[0] * k - ax[1] * s, ax[2] * ax[1] * k + ax[0] * s, c + ax[2] * ax[2] * k};
  Mat r{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) r.m[i * 4 + j] = R[i * 3 + j] * sc[j];
    r.m[i * 4 + 3] = T(rng);
  }
  return r;
}

Aabb worldBox(const Aabb& b, const Mat& m) {
  Aabb out = emptyBox();
  for (int corner = 0; corner < 8; ++corner) {
    float p[3] = {(corner & 1) ? b.mx[0] : b.mn[0], (corner & 2) ? b.mx[1] : b.mn[1],
                  (corner & 4) ? b.mx[2] : b.mn[2]};
    for (int r = 0; r < 3; ++r) {
      float w = m.m[r * 4 + 0] * p[0] + m.m[r * 4 + 1] * p[1] + m.m[r * 4 + 2] * p[2] + m.m[r * 4 + 3];
      out.mn[r] = std::min(out.mn[r], w);
      out.mx[r] = std::max(out.mx[r], w);
    }
  }
  return out;
}

struct Tlas {
  VkAccelerationStructureKHR as{};
  Buf asBuf, scratch, instances;
  VkDeviceAddress addr{};
};

void initSceneUtil() {
  pAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(gDev, "vkGetAccelerationStructureDeviceAddressKHR");
  CHECK(pAddr);
}

// Builds a TLAS over `insts` and waits for completion.
Tlas makeTlas(const std::vector<VkAccelerationStructureInstanceKHR>& insts) {
  Tlas t;
  uint32_t count = uint32_t(insts.size());
  t.instances = makeBuf(std::max<size_t>(64, insts.size() * 64), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  if (count) std::memcpy(t.instances.map, insts.data(), insts.size() * 64);
  VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  g.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  g.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  g.geometry.instances.data.deviceAddress = t.instances.addr;
  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &g;
  VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  pSizes(gDev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &count, &sz);
  t.asBuf = makeBuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false);
  t.scratch = makeBuf(sz.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  ci.buffer = t.asBuf.buf;
  ci.size = sz.accelerationStructureSize;
  ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  VK(pCreateAS(gDev, &ci, nullptr, &t.as));
  VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  ai.accelerationStructure = t.as;
  t.addr = pAddr(gDev, &ai);
  bgi.dstAccelerationStructure = t.as;
  bgi.scratchData.deviceAddress = t.scratch.addr;
  VkAccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;
  VkCommandBuffer cb = beginCb();
  pBuild(cb, 1, &bgi, &pr);
  VK(vkEndCommandBuffer(cb));
  submitAndWait(cb);
  return t;
}

}  // namespace
