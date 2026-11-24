// GPU integration test for the TLAS builder: builds two BLASes and several TLASes
// (through the layer, on the real device), reads each TLAS back and verifies it on the CPU.
#include "scene_util.h"

#include "../layer/src/bvh_format.h"

namespace {

struct InstanceSpec {
  const char* name;
  uint32_t count;
  bool pointers = false;
  bool identical = false;
  float inactiveFraction = 0.05f;
};

struct Expected {
  VkAccelerationStructureInstanceKHR inst{};
  Aabb box = emptyBox();
  bool active = false;
};

struct Verifier {
  const uint32_t* w;
  uint32_t nodeBase, primBase;
  std::vector<Expected> exp;
  std::vector<int> seen;
  uint32_t leaves = 0, internals = 0;
  float f(uint32_t i) const { float x; std::memcpy(&x, &w[i], 4); return x; }

  Aabb visit(uint32_t ref) {
    if (ref & 0x80000000u) {
      uint32_t k = ref & 0x7FFFFFFFu;
      CHECK(k < seen.size() && seen[k]++ == 0);
      ++leaves;
      const uint32_t o = primBase + k * 32;
      uint32_t idx = w[o + 28];
      CHECK(idx < exp.size());
      const Expected& e = exp[idx];
      for (int i = 0; i < 12; ++i) CHECK(f(o + i) == e.inst.transform.matrix[i / 4][i % 4]);
      uint32_t custom = e.inst.instanceCustomIndex, mask = e.inst.mask;
      CHECK((w[o + 24] & 0xFFFFFFu) == custom);
      CHECK((w[o + 25] & 0xFFFFFFu) == e.inst.instanceShaderBindingTableRecordOffset);
      CHECK((w[o + 25] >> 24) == uint32_t(e.inst.flags));
      if (e.active) {
        CHECK((w[o + 24] >> 24) == mask);
        uint64_t ref64 = uint64_t(w[o + 26]) | (uint64_t(w[o + 27]) << 32);
        CHECK(ref64 == e.inst.accelerationStructureReference);
        // worldToObject must invert objectToWorld (3x3 part and translation).
        float R[3][3], t[3];
        for (int r = 0; r < 3; ++r) {
          t[r] = f(o + 12 + r * 4 + 3);
          for (int c = 0; c < 3; ++c) {
            R[r][c] = 0;
            for (int k2 = 0; k2 < 3; ++k2) R[r][c] += f(o + 12 + r * 4 + k2) * e.inst.transform.matrix[k2][c];
            CHECK(std::fabs(R[r][c] - (r == c ? 1.f : 0.f)) < 2e-3f);
          }
        }
        for (int r = 0; r < 3; ++r) {
          float v = t[r];
          for (int c = 0; c < 3; ++c) v += f(o + 12 + r * 4 + c) * e.inst.transform.matrix[c][3];
          CHECK(std::fabs(v) < 5e-2f);
        }
      } else {
        CHECK((w[o + 24] >> 24) == 0);  // inactive: mask 0, no BLAS reference
        CHECK(w[o + 26] == 0 && w[o + 27] == 0);
      }
      return e.box;
    }
    ++internals;
    const uint32_t o = nodeBase + ref * 16;
    Aabb result = emptyBox();
    for (int c = 0; c < 2; ++c) {
      Aabb stored;
      for (int a = 0; a < 3; ++a) { stored.mn[a] = f(o + c * 6 + a); stored.mx[a] = f(o + c * 6 + 3 + a); }
      Aabb actual = visit(w[o + 12 + c]);
      for (int a = 0; a < 3; ++a) {
        if (!near(stored.mn[a], actual.mn[a]) || !near(stored.mx[a], actual.mx[a]))
          std::fprintf(stderr, "node %u child %d axis %d: stored [%g,%g] actual [%g,%g]\n", ref, c, a,
                       stored.mn[a], stored.mx[a], actual.mn[a], actual.mx[a]);
        CHECK(near(stored.mn[a], actual.mn[a]));
        CHECK(near(stored.mx[a], actual.mx[a]));
      }
      grow(result, actual);
    }
    return result;
  }
};

void runCase(const InstanceSpec& sp, const Blas& A, const Blas& B) {
  std::mt19937 rng(777);
  std::vector<Expected> exp(sp.count);
  std::vector<VkAccelerationStructureInstanceKHR> insts(sp.count);
  for (uint32_t i = 0; i < sp.count; ++i) {
    const Blas& blas = (rng() & 1) ? A : B;
    Mat m = randomTransform(rng, sp.identical);
    bool active = (rng() % 1000) >= uint32_t(sp.inactiveFraction * 1000);
    VkAccelerationStructureInstanceKHR in{};
    std::memcpy(in.transform.matrix, m.m, sizeof(m.m));
    in.instanceCustomIndex = i & 0xFFFFFF;
    in.mask = 1 + (i % 255);
    in.instanceShaderBindingTableRecordOffset = (i * 3) & 0xFFFFFF;
    in.flags = i % 16;
    in.accelerationStructureReference = active ? blas.addr : 0;
    insts[i] = in;
    exp[i].inst = in;
    exp[i].active = active;
    if (active) exp[i].box = worldBox(blas.bounds, m);
  }
  Buf instBuf = makeBuf(std::max<size_t>(64, insts.size() * 64), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  if (!insts.empty()) std::memcpy(instBuf.map, insts.data(), insts.size() * 64);
  Buf ptrBuf = makeBuf(std::max<size_t>(16, insts.size() * 8), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  for (uint32_t i = 0; i < sp.count; ++i) static_cast<uint64_t*>(ptrBuf.map)[i] = instBuf.addr + uint64_t(i) * 64;

  VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  g.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  g.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  g.geometry.instances.arrayOfPointers = sp.pointers ? VK_TRUE : VK_FALSE;
  g.geometry.instances.data.deviceAddress = sp.pointers ? ptrBuf.addr : instBuf.addr;
  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &g;
  uint32_t count = sp.count;
  VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  pSizes(gDev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &count, &sz);

  Buf asBuf = makeBuf(sz.accelerationStructureSize,
                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
  Buf scratch = makeBuf(sz.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  Buf readback = makeBuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  std::memset(readback.map, 0xCD, readback.size);
  VkAccelerationStructureKHR tlas;
  VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  ci.buffer = asBuf.buf;
  ci.size = sz.accelerationStructureSize;
  ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  VK(pCreateAS(gDev, &ci, nullptr, &tlas));
  bgi.dstAccelerationStructure = tlas;
  bgi.scratchData.deviceAddress = scratch.addr;

  VkAccelerationStructureBuildRangeInfoKHR range{sp.count, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;
  VkCommandBuffer cb = beginCb();
  pBuild(cb, 1, &bgi, &pr);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                       &mb, 0, nullptr, 0, nullptr);
  VkBufferCopy cp{0, 0, sz.accelerationStructureSize};
  vkCmdCopyBuffer(cb, asBuf.buf, readback.buf, 1, &cp);
  VK(vkEndCommandBuffer(cb));
  auto t0 = std::chrono::steady_clock::now();
  submitAndWait(cb);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  using namespace vksoft::bvh;
  const uint32_t* w = static_cast<const uint32_t*>(readback.map);
  CHECK(w[0] == 0x42534B56u);
  CHECK(w[1] == 1u);  // TLAS
  CHECK(w[2] == sp.count);
  CHECK(w[3] == (sp.count ? sp.count - 1 : 0));
  if (sp.count == 0) {
    CHECK(w[4] == 0xFFFFFFFFu);
  } else {
    CHECK(w[5] == kHeaderBytes);
    CHECK(w[6] == kHeaderBytes + maxInternalNodes(sp.count) * kNodeBytes);
    CHECK(w[6] + uint64_t(sp.count) * kInstanceBytes <= sz.accelerationStructureSize);
    Verifier v{w, w[5] / 4, w[6] / 4, exp, std::vector<int>(sp.count, 0)};
    Aabb all = v.visit(w[4]);
    CHECK(v.leaves == sp.count && v.internals == sp.count - 1);
    for (int a = 0; a < 3; ++a) {
      float mn, mx;
      std::memcpy(&mn, &w[8 + a], 4);
      std::memcpy(&mx, &w[11 + a], 4);
      CHECK(near(mn, all.mn[a]) && near(mx, all.mx[a]));
    }
    uint32_t seenIdx = 0;
    std::vector<int> idxSeen(sp.count, 0);
    for (uint32_t k = 0; k < sp.count; ++k) { uint32_t i = w[w[6] / 4 + k * 32 + 28]; CHECK(i < sp.count && !idxSeen[i]++); ++seenIdx; }
    CHECK(seenIdx == sp.count);
  }
  uint32_t inactive = 0;
  for (auto& e : exp) inactive += e.active ? 0 : 1;
  std::printf("  %-28s N=%-6u inactive=%-4u TLAS=%-8llu build+copy %.2f ms  OK\n", sp.name, sp.count, inactive,
              (unsigned long long)sz.accelerationStructureSize, ms);

  pDestroyAS(gDev, tlas, nullptr);
  for (Buf* b : {&instBuf, &ptrBuf, &asBuf, &scratch, &readback}) freeBuf(*b);
}

}  // namespace

int main() {
  initVulkan();
  initSceneUtil();

  std::mt19937 rng(42);
  Blas A = makeBlas(rng, 1000, 5.f);
  Blas B = makeBlas(rng, 300, 2.f);
  std::puts("tlas_build:");
  runCase({"empty", 0}, A, B);
  runCase({"single instance", 1, false, false, 0.f}, A, B);
  runCase({"two instances", 2, false, false, 0.f}, A, B);
  runCase({"257 with inactive", 257, false, false, 0.2f}, A, B);
  runCase({"all inactive", 64, false, false, 1.f}, A, B);
  runCase({"identical (ties)", 500, false, true, 0.f}, A, B);
  runCase({"pointer array", 1000, true}, A, B);
  runCase({"5000 random", 5000}, A, B);
  runCase({"200k random", 200000}, A, B);
  std::puts("tlas_build passed");
  return 0;
}
