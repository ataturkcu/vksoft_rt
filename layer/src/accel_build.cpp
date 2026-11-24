#include "accel_build.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "bvh_format.h"
#include "kernels_spv.h"

#define LOGF(...)                                           \
  do {                                                      \
    if (std::getenv("VKSOFT_RT_LOG")) {                     \
      std::fprintf(stderr, "[vksoft_rt] " __VA_ARGS__);     \
      std::fputc('\n', stderr);                             \
    }                                                       \
  } while (0)

namespace vksoft {
namespace {

using namespace bvh;

enum Kernel {
  K_INIT, K_HEADER, K_FETCH, K_FETCH_INST, K_MORTON, K_HIST, K_SCATTER, K_SCAN1, K_SCAN2, K_SCAN3,
  K_KARRAS, K_REFIT, K_COUNT
};

struct SpvRef { const uint32_t* data; size_t bytes; };
#define SPV(n) SpvRef{k_##n##_spv, sizeof(k_##n##_spv)}
const SpvRef kSpv[K_COUNT] = {SPV(init), SPV(header), SPV(fetch_tris), SPV(fetch_instances), SPV(morton), SPV(radix_hist),
                              SPV(radix_scatter), SPV(scan1), SPV(scan2), SPV(scan3), SPV(karras),
                              SPV(refit)};
#undef SPV

// Mirrors `layout(push_constant) uniform PC` in layer/shaders/common.glsl.
struct Push {
  uint64_t a[8];
  uint32_t u[16];
};
static_assert(sizeof(Push) == 128);

}  // namespace

struct BuildKernels {
  VkDevice device{};
  VkPipelineLayout layout{};
  VkPipeline pipe[K_COUNT]{};
  PFN_vkDestroyPipeline DestroyPipeline{};
  PFN_vkDestroyPipelineLayout DestroyPipelineLayout{};
  PFN_vkCmdBindPipeline CmdBindPipeline{};
  PFN_vkCmdPushConstants CmdPushConstants{};
  PFN_vkCmdDispatch CmdDispatch{};
  PFN_vkCmdPipelineBarrier CmdPipelineBarrier{};
};

BuildKernels* createBuildKernels(VkInstance instance, VkPhysicalDevice pd, VkDevice device,
                                 PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa) {
  // The kernels use subgroup ballot/arithmetic and assume 32 lanes.
  auto getProps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      gipa(instance, "vkGetPhysicalDeviceProperties2"));
  if (getProps2) {
    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sg;
    getProps2(pd, &p2);
    const VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    if (sg.subgroupSize != 32 || (sg.supportedOperations & need) != need ||
        !(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
      LOGF("build kernels unavailable: subgroupSize=%u (need 32 + ballot/arithmetic)", sg.subgroupSize);
      return nullptr;
    }
  }

#define FN(name) reinterpret_cast<PFN_vk##name>(gdpa(device, "vk" #name))
  auto createModule = FN(CreateShaderModule);
  auto destroyModule = FN(DestroyShaderModule);
  auto createPipelineLayout = FN(CreatePipelineLayout);
  auto createPipelines = FN(CreateComputePipelines);
  auto* k = new BuildKernels;
  k->device = device;
  k->DestroyPipeline = FN(DestroyPipeline);
  k->DestroyPipelineLayout = FN(DestroyPipelineLayout);
  k->CmdBindPipeline = FN(CmdBindPipeline);
  k->CmdPushConstants = FN(CmdPushConstants);
  k->CmdDispatch = FN(CmdDispatch);
  k->CmdPipelineBarrier = FN(CmdPipelineBarrier);
#undef FN

  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &range;
  bool ok = createPipelineLayout(device, &plci, nullptr, &k->layout) == VK_SUCCESS;

  for (int i = 0; ok && i < K_COUNT; ++i) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = kSpv[i].bytes;
    smci.pCode = kSpv[i].data;
    VkShaderModule mod;
    if (createModule(device, &smci, nullptr, &mod) != VK_SUCCESS) { ok = false; break; }
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
    cpci.layout = k->layout;
    ok = createPipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &k->pipe[i]) == VK_SUCCESS;
    destroyModule(device, mod, nullptr);
  }
  if (!ok) {
    LOGF("build kernels: pipeline creation failed");
    destroyBuildKernels(k);
    return nullptr;
  }
  LOGF("build kernels ready (%d pipelines)", int(K_COUNT));
  return k;
}

void destroyBuildKernels(BuildKernels* k) {
  if (!k) return;
  for (auto p : k->pipe)
    if (p) k->DestroyPipeline(k->device, p, nullptr);
  if (k->layout) k->DestroyPipelineLayout(k->device, k->layout, nullptr);
  delete k;
}

namespace {

void dispatch(BuildKernels* k, VkCommandBuffer cb, Kernel which, const Push& pc, uint32_t groups) {
  if (groups == 0) return;
  k->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipe[which]);
  k->CmdPushConstants(cb, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc);
  k->CmdDispatch(cb, groups, 1, 1);
}

void barrier(BuildKernels* k, VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags srcAccess,
             VkPipelineStageFlags dst, VkAccessFlags dstAccess) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = srcAccess;
  mb.dstAccessMask = dstAccess;
  k->CmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void computeBarrier(BuildKernels* k, VkCommandBuffer cb) {
  barrier(k, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

constexpr uint32_t groups256(uint64_t n) { return static_cast<uint32_t>((n + 255) / 256); }

}  // namespace

bool recordBuild(BuildKernels* k, VkCommandBuffer cb, const VkAccelerationStructureBuildGeometryInfoKHR& info,
                 const VkAccelerationStructureBuildRangeInfoKHR* ranges, VkDeviceAddress dst) {
  if (!k) return false;
  const bool isTlas = info.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  if (info.type != VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR && !isTlas) {
    LOGF("build: unsupported acceleration structure type %d", int(info.type));
    return false;
  }
  // Validate before recording anything.
  uint64_t n = 0;
  if (isTlas) {
    if (info.geometryCount != 1) {
      LOGF("build: a TLAS needs exactly one instances geometry");
      return false;
    }
    const auto& geo = info.pGeometries ? info.pGeometries[0] : *info.ppGeometries[0];
    if (geo.geometryType != VK_GEOMETRY_TYPE_INSTANCES_KHR) {
      LOGF("build: TLAS geometry must be instances");
      return false;
    }
    n = ranges[0].primitiveCount;
  } else {
    for (uint32_t g = 0; g < info.geometryCount; ++g) {
      const auto& geo = info.pGeometries ? info.pGeometries[g] : *info.ppGeometries[g];
      if (geo.geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
        LOGF("build: only triangle geometry is implemented (geometry %u)", g);
        return false;
      }
      if (geo.geometry.triangles.vertexFormat != VK_FORMAT_R32G32B32_SFLOAT) {
        LOGF("build: only R32G32B32_SFLOAT vertices are implemented (format %d)",
             int(geo.geometry.triangles.vertexFormat));
        return false;
      }
      n += ranges[g].primitiveCount;
    }
  }

  const ScratchLayout L = scratchLayout(n);
  const VkDeviceAddress scratch = info.scratchData.deviceAddress;
  const uint64_t nodeRegion = maxInternalNodes(n) * kNodeBytes;
  const VkDeviceAddress nodesAddr = dst + kHeaderBytes;
  const VkDeviceAddress primsAddr = dst + kHeaderBytes + nodeRegion;

  // The app's barriers target the driver's own AS-build stage, which need not cover
  // our compute work: fence conservatively against everything before and after.
  barrier(k, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  Push pc{};
  pc.a[0] = scratch + L.bounds;
  pc.a[1] = scratch + L.flags;
  pc.a[2] = scratch + L.parent;
  pc.u[0] = static_cast<uint32_t>(n);
  dispatch(k, cb, K_INIT, pc, groups256(n > 6 ? n : 6));

  pc = {};
  pc.a[0] = dst;
  pc.u[0] = static_cast<uint32_t>(n);
  pc.u[1] = static_cast<uint32_t>(nodeRegion);
  pc.u[2] = isTlas ? 1u : 0u;
  dispatch(k, cb, K_HEADER, pc, 1);
  computeBarrier(k, cb);

  VkDeviceAddress instAddr = 0;
  uint32_t instPtrMode = 0;
  if (isTlas) {
    const auto& geo = info.pGeometries ? info.pGeometries[0] : *info.ppGeometries[0];
    instAddr = geo.geometry.instances.data.deviceAddress + ranges[0].primitiveOffset;
    instPtrMode = geo.geometry.instances.arrayOfPointers ? 1u : 0u;
    pc = {};
    pc.a[0] = instAddr;
    pc.a[1] = scratch + L.tris;
    pc.a[2] = scratch + L.bounds;
    pc.u[0] = static_cast<uint32_t>(n);
    pc.u[1] = instPtrMode;
    dispatch(k, cb, K_FETCH_INST, pc, groups256(n));
  } else {
    uint64_t primBase = 0;
    for (uint32_t g = 0; g < info.geometryCount; ++g) {
      const auto& geo = info.pGeometries ? info.pGeometries[g] : *info.ppGeometries[g];
      const auto& tri = geo.geometry.triangles;
      const auto& r = ranges[g];
      pc = {};
      const bool indexed = tri.indexType != VK_INDEX_TYPE_NONE_KHR;
      if (indexed) {
        pc.a[0] = tri.vertexData.deviceAddress;
        pc.a[1] = tri.indexData.deviceAddress + r.primitiveOffset;
        pc.u[5] = r.firstVertex;
      } else {
        pc.a[0] = tri.vertexData.deviceAddress + r.primitiveOffset + uint64_t(r.firstVertex) * tri.vertexStride;
      }
      pc.a[2] = tri.transformData.deviceAddress ? tri.transformData.deviceAddress + r.transformOffset : 0;
      pc.a[3] = scratch + L.tris + primBase * kTriangleScratchBytes;
      pc.a[4] = scratch + L.bounds;
      pc.u[0] = r.primitiveCount;
      pc.u[1] = static_cast<uint32_t>(tri.vertexStride);
      pc.u[2] = !indexed ? 0 : (tri.indexType == VK_INDEX_TYPE_UINT16 ? 1 : 2);
      pc.u[3] = g;
      pc.u[4] = (geo.flags & VK_GEOMETRY_OPAQUE_BIT_KHR) ? 1u : 0u;
      dispatch(k, cb, K_FETCH, pc, groups256(r.primitiveCount));
      primBase += r.primitiveCount;
    }
  }
  computeBarrier(k, cb);

  pc = {};
  pc.a[0] = scratch + L.tris;
  pc.a[1] = scratch + L.bounds;
  pc.a[2] = scratch + L.keysA;
  pc.a[3] = scratch + L.valsA;
  pc.u[0] = static_cast<uint32_t>(n);
  pc.u[1] = isTlas ? 1u : 0u;
  dispatch(k, cb, K_MORTON, pc, groups256(n));
  computeBarrier(k, cb);

  // 4 x 8-bit LSD passes: A -> B -> A -> B -> A, result ends in A.
  uint64_t kin = L.keysA, vin = L.valsA, kout = L.keysB, vout = L.valsB;
  for (uint32_t pass = 0; pass < 4 && n > 1; ++pass) {
    pc = {};
    pc.a[0] = scratch + kin;
    pc.a[1] = scratch + L.hist;
    pc.u[0] = static_cast<uint32_t>(n);
    pc.u[1] = pass * 8;
    pc.u[2] = L.blocks;
    dispatch(k, cb, K_HIST, pc, L.blocks);
    computeBarrier(k, cb);

    pc = {};
    pc.a[0] = scratch + L.hist;
    pc.a[1] = scratch + L.chunk;
    pc.u[0] = L.histWords;
    dispatch(k, cb, K_SCAN1, pc, L.chunkCount);
    computeBarrier(k, cb);

    pc = {};
    pc.a[0] = scratch + L.chunk;
    pc.u[0] = L.chunkCount;
    dispatch(k, cb, K_SCAN2, pc, 1);
    computeBarrier(k, cb);

    pc = {};
    pc.a[0] = scratch + L.hist;
    pc.a[1] = scratch + L.chunk;
    pc.u[0] = L.histWords;
    dispatch(k, cb, K_SCAN3, pc, groups256(L.histWords));
    computeBarrier(k, cb);

    pc = {};
    pc.a[0] = scratch + kin;
    pc.a[1] = scratch + vin;
    pc.a[2] = scratch + kout;
    pc.a[3] = scratch + vout;
    pc.a[4] = scratch + L.hist;
    pc.u[0] = static_cast<uint32_t>(n);
    pc.u[1] = pass * 8;
    pc.u[2] = L.blocks;
    dispatch(k, cb, K_SCATTER, pc, L.blocks);
    computeBarrier(k, cb);
    std::swap(kin, kout);
    std::swap(vin, vout);
  }

  pc = {};
  pc.a[0] = nodesAddr;
  pc.a[1] = scratch + L.keysA;
  pc.a[2] = scratch + L.parent;
  pc.u[0] = static_cast<uint32_t>(n);
  if (n > 1) dispatch(k, cb, K_KARRAS, pc, groups256(n - 1));
  computeBarrier(k, cb);

  pc = {};
  pc.a[0] = nodesAddr;
  pc.a[1] = primsAddr;
  pc.a[2] = scratch + L.tris;
  pc.a[3] = scratch + L.valsA;
  pc.a[4] = scratch + L.parent;
  pc.a[5] = scratch + L.flags;
  pc.a[6] = dst;
  pc.a[7] = instAddr;
  pc.u[0] = static_cast<uint32_t>(n);
  pc.u[1] = isTlas ? 1u : 0u;
  pc.u[2] = instPtrMode;
  dispatch(k, cb, K_REFIT, pc, groups256(n));

  barrier(k, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  return true;
}

}  // namespace vksoft
