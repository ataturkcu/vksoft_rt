// Procedural (AABB) geometry end to end: an AABB BLAS and a TLAS built by the layer, a GLSL
// ray query shader that intersects the boxes itself and calls rayQueryGenerateIntersection,
// checked against a brute-force double precision reference (object space slab test).
#include <fstream>

#include "trace_ref.h"

namespace {

std::vector<char> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  CHECK(f.good());
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct PC { uint32_t count, flags, mask; };

struct BoxD { double mn[3], mx[3]; bool valid; };

// 3x4 affine inverse in double precision.
void invert(const float* m, double* inv) {
  double a[3][3], t[3];
  for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) a[r][c] = m[r * 4 + c]; t[r] = m[r * 4 + 3]; }
  double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
               a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  double r[3][3];
  r[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det; r[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det; r[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  r[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det; r[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det; r[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  r[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det; r[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det; r[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) inv[i * 4 + j] = r[i][j];
    inv[i * 4 + 3] = -(r[i][0] * t[0] + r[i][1] * t[1] + r[i][2] * t[2]);
  }
}

struct BoxInst { double inv[12]; uint32_t mask; bool active; };

struct Best { bool found = false; double t = 0; uint32_t inst = 0, prim = 0; };
struct BoxClosest { Best in, out; };

BoxClosest reference(const RayCpu& r, const std::vector<BoxInst>& insts, const std::vector<BoxD>& boxes, uint32_t flags, uint32_t mask) {
  BoxClosest c;
  if (flags & 512u) return c;  // skip procedural
  const bool opaque = !(flags & 2u) || (flags & 1u);
  if (opaque && (flags & 64u)) return c;
  if (!opaque && (flags & 128u)) return c;
  for (uint32_t i = 0; i < insts.size(); ++i) {
    const BoxInst& in = insts[i];
    if (!in.active || (mask & in.mask) == 0) continue;
    double o[3], d[3];
    for (int k = 0; k < 3; ++k) {
      o[k] = in.inv[k * 4] * r.o[0] + in.inv[k * 4 + 1] * r.o[1] + in.inv[k * 4 + 2] * r.o[2] + in.inv[k * 4 + 3];
      d[k] = in.inv[k * 4] * r.d[0] + in.inv[k * 4 + 1] * r.d[1] + in.inv[k * 4 + 2] * r.d[2];
    }
    for (uint32_t b = 0; b < boxes.size(); ++b) {
      if (!boxes[b].valid) continue;
      double tn = -1e300, tf = 1e300;
      for (int k = 0; k < 3; ++k) {
        double t0 = (boxes[b].mn[k] - o[k]) / d[k], t1 = (boxes[b].mx[k] - o[k]) / d[k];
        if (d[k] == 0) {
          if (o[k] < boxes[b].mn[k] || o[k] > boxes[b].mx[k]) { tn = 1e300; tf = -1e300; }
          continue;
        }
        tn = std::max(tn, std::min(t0, t1));
        tf = std::min(tf, std::max(t0, t1));
      }
      double t = std::max(tn, 0.0);
      double margin = (tf - t) / (1.0 + std::fabs(tf));
      if (!(t <= r.tmax) || !(margin >= -1e-4)) continue;
      if (!c.out.found || t < c.out.t) c.out = {true, t, i, b};
      if (margin >= 1e-4 && (!c.in.found || t < c.in.t)) c.in = {true, t, i, b};
    }
  }
  return c;
}

}  // namespace

int main() {
  initVulkan(true);
  initSceneUtil();
  std::mt19937 rng(99);

  // ---- AABB BLAS ------------------------------------------------------------------------
  const uint32_t nBoxes = 400;
  std::uniform_real_distribution<float> C(-5.f, 5.f), H(0.1f, 0.6f);
  std::vector<float> boxData(nBoxes * 6);
  std::vector<BoxD> boxes(nBoxes);
  for (uint32_t i = 0; i < nBoxes; ++i) {
    bool invalid = (rng() % 100) < 5;  // inactive: min > max
    for (int a = 0; a < 3; ++a) {
      float c = C(rng), h = H(rng);
      float mn = c - h, mx = c + h;
      if (invalid) std::swap(mn, mx);
      boxData[i * 6 + a] = mn;
      boxData[i * 6 + 3 + a] = mx;
    }
    boxes[i].valid = !invalid;
    for (int a = 0; a < 3; ++a) { boxes[i].mn[a] = boxData[i * 6 + a]; boxes[i].mx[a] = boxData[i * 6 + 3 + a]; }
  }
  Buf boxBuf = makeBuf(boxData.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  std::memcpy(boxBuf.map, boxData.data(), boxData.size() * 4);
  VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  g.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  g.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
  g.geometry.aabbs.data.deviceAddress = boxBuf.addr;
  g.geometry.aabbs.stride = 24;
  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &g;
  uint32_t nb = nBoxes;
  VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  pSizes(gDev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &nb, &sz);
  Buf blasBuf = makeBuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false);
  Buf scratch = makeBuf(sz.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  ci.buffer = blasBuf.buf;
  ci.size = sz.accelerationStructureSize;
  ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  VkAccelerationStructureKHR blas;
  VK(pCreateAS(gDev, &ci, nullptr, &blas));
  VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  ai.accelerationStructure = blas;
  VkDeviceAddress blasAddr = pAddr(gDev, &ai);
  bgi.dstAccelerationStructure = blas;
  bgi.scratchData.deviceAddress = scratch.addr;
  VkAccelerationStructureBuildRangeInfoKHR range{nBoxes, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;
  VkCommandBuffer bcb = beginCb();
  pBuild(bcb, 1, &bgi, &pr);
  VK(vkEndCommandBuffer(bcb));
  submitAndWait(bcb);

  // ---- TLAS ------------------------------------------------------------------------------
  const uint32_t nInst = 48;
  std::vector<VkAccelerationStructureInstanceKHR> insts(nInst);
  std::vector<BoxInst> ref(nInst);
  for (uint32_t i = 0; i < nInst; ++i) {
    Mat m = randomTransform(rng, false);
    if (i % 5 == 2) { m.m[0] = -m.m[0]; m.m[4] = -m.m[4]; m.m[8] = -m.m[8]; }
    bool active = (rng() % 100) >= 5;
    VkAccelerationStructureInstanceKHR in{};
    std::memcpy(in.transform.matrix, m.m, sizeof(m.m));
    in.instanceCustomIndex = 1000 + i;
    in.mask = 1 + (i % 255);
    in.accelerationStructureReference = active ? blasAddr : 0;
    insts[i] = in;
    invert(m.m, ref[i].inv);
    ref[i].mask = in.mask;
    ref[i].active = active;
  }
  Tlas tlas = makeTlas(insts);

  // ---- rays: half aimed at box centres --------------------------------------------------
  const uint32_t R = 1200;
  std::uniform_real_distribution<double> U(-60.0, 60.0), D(-1.0, 1.0);
  std::vector<RayCpu> rays(R);
  Buf rayBuf = makeBuf(R * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  Buf resBuf = makeBuf(R * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  for (uint32_t i = 0; i < R; ++i) {
    RayCpu r{};
    for (int a = 0; a < 3; ++a) r.o[a] = U(rng);
    uint32_t kind = rng() % 100;
    uint32_t ii = rng() % nInst;
    if (kind < 55 && ref[ii].active) {  // aim at the world-space centre of a valid box of an active instance
      uint32_t bi = rng() % nBoxes;
      while (!boxes[bi].valid) bi = (bi + 1) % nBoxes;
      double c[3] = {(boxes[bi].mn[0] + boxes[bi].mx[0]) / 2, (boxes[bi].mn[1] + boxes[bi].mx[1]) / 2, (boxes[bi].mn[2] + boxes[bi].mx[2]) / 2};
      const float* mm = insts[ii].transform.matrix[0];
      for (int a = 0; a < 3; ++a) r.d[a] = mm[a * 4] * c[0] + mm[a * 4 + 1] * c[1] + mm[a * 4 + 2] * c[2] + mm[a * 4 + 3] - r.o[a];
    } else if (kind < 62) {
      static const double axes[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      const uint32_t axis = rng() % 6;  // one axis for all components (a per-component pick can give a zero vector)
      for (int a = 0; a < 3; ++a) r.d[a] = axes[axis][a];
    } else {
      for (int a = 0; a < 3; ++a) r.d[a] = D(rng);
    }
    double len = std::sqrt(dot3(r.d, r.d));
    for (double& x : r.d) x /= len;
    r.tmax = (rng() % 4 == 0) ? 40.0 : 400.0;
    float* f = static_cast<float*>(rayBuf.map) + i * 8;
    for (int a = 0; a < 3; ++a) { f[a] = float(r.o[a]); f[4 + a] = float(r.d[a]); }
    f[3] = 0.f;
    f[7] = float(r.tmax);
    for (int a = 0; a < 3; ++a) { r.o[a] = f[a]; r.d[a] = f[4 + a]; }
    rays[i] = r;
  }

  // ---- pipeline ---------------------------------------------------------------------------
  VkDescriptorSetLayoutBinding b[4]{};
  b[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  for (uint32_t i = 1; i < 4; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = 4;
  dlci.pBindings = b;
  VkDescriptorSetLayout dsl;
  VK(vkCreateDescriptorSetLayout(gDev, &dlci, nullptr, &dsl));
  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = ps;
  VkDescriptorPool pool;
  VK(vkCreateDescriptorPool(gDev, &dpci, nullptr, &pool));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet set;
  VK(vkAllocateDescriptorSets(gDev, &dsai, &set));
  VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  asw.accelerationStructureCount = 1;
  asw.pAccelerationStructures = &tlas.as;
  VkDescriptorBufferInfo bi[3] = {{rayBuf.buf, 0, VK_WHOLE_SIZE}, {resBuf.buf, 0, VK_WHOLE_SIZE}, {boxBuf.buf, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet w[4]{};
  w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asw, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
  for (uint32_t i = 0; i < 3; ++i)
    w[i + 1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i + 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[i], nullptr};
  vkUpdateDescriptorSets(gDev, 4, w, 0, nullptr);
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout layout;
  VK(vkCreatePipelineLayout(gDev, &plci, nullptr, &layout));
  auto code = readFile(std::string(TEST_SPV_DIR) + "/rq_aabb.spv");
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code.size();
  smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
  VkShaderModule mod;
  VK(vkCreateShaderModule(gDev, &smci, nullptr, &mod));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
  cpci.layout = layout;
  VkPipeline pipe;
  VK(vkCreateComputePipelines(gDev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

  struct AMode { const char* name; uint32_t flags, mask; bool anyHit; };
  const AMode modes[] = {
      {"closest hit", 0, 0xFFFFFFFFu, false},
      {"terminate on first hit", 4, 0xFFFFFFFFu, true},
      {"mask 0x0F", 0, 0x0F, false},
      {"force no-opaque", 2, 0xFFFFFFFFu, false},
      {"cull opaque", 64, 0xFFFFFFFFu, false},
      {"skip procedural", 512, 0xFFFFFFFFu, false},
  };
  std::puts("aabb_test:");
  for (const AMode& md : modes) {
    std::memset(resBuf.map, 0, resBuf.size);
    PC pc{R, md.flags, md.mask};
    VkCommandBuffer cb = beginCb();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, (R + 63) / 64, 1, 1);
    VK(vkEndCommandBuffer(cb));
    submitAndWait(cb);
    vkFreeCommandBuffers(gDev, gPool, 1, &cb);

    const uint32_t* res = static_cast<const uint32_t*>(resBuf.map);
    uint32_t hits = 0, misses = 0, borderline = 0;
    for (uint32_t i = 0; i < R; ++i) {
      const uint32_t* o = res + i * 8;
      bool gHit = o[1] != 0;
      float gt; std::memcpy(&gt, &o[0], 4);
      BoxClosest c = reference(rays[i], ref, boxes, md.flags, md.mask);
      CHECK(o[6] == (gHit ? 2u : 0u));  // committed type: generated or none
      if (!gHit) {
        if (c.in.found) {
          std::fprintf(stderr, "[%s] ray %u: GPU miss, CPU hit t=%g inst=%u box=%u\n", md.name, i, c.in.t, c.in.inst, c.in.prim);
          const RayCpu& r = rays[i];
          std::fprintf(stderr, "  ray o=(%g %g %g) d=(%g %g %g) tmax=%g\n", r.o[0], r.o[1], r.o[2], r.d[0], r.d[1], r.d[2], r.tmax);
          const BoxInst& in = ref[c.in.inst];
          double oo[3], dd[3];
          for (int k = 0; k < 3; ++k) {
            oo[k] = in.inv[k * 4] * r.o[0] + in.inv[k * 4 + 1] * r.o[1] + in.inv[k * 4 + 2] * r.o[2] + in.inv[k * 4 + 3];
            dd[k] = in.inv[k * 4] * r.d[0] + in.inv[k * 4 + 1] * r.d[1] + in.inv[k * 4 + 2] * r.d[2];
          }
          std::fprintf(stderr, "  object o=(%g %g %g) d=(%g %g %g) box=(%g %g %g / %g %g %g) mask=%u active=%d\n", oo[0], oo[1], oo[2], dd[0], dd[1], dd[2],
                       boxes[c.in.prim].mn[0], boxes[c.in.prim].mn[1], boxes[c.in.prim].mn[2], boxes[c.in.prim].mx[0], boxes[c.in.prim].mx[1], boxes[c.in.prim].mx[2], in.mask, int(in.active));
          std::exit(1);
        }
        if (c.out.found) ++borderline; else ++misses;
        continue;
      }
      ++hits;
      if (!c.out.found) { std::fprintf(stderr, "[%s] ray %u: GPU hit t=%g, CPU miss\n", md.name, i, gt); std::exit(1); }
      CHECK(o[2] < nInst && ref[o[2]].active);
      CHECK(o[7] == 1000 + o[2]);
      CHECK(o[3] == 0);                       // geometry index
      CHECK(o[4] < nBoxes && boxes[o[4]].valid);
      CHECK(o[5] == ((md.flags & 2u) && !(md.flags & 1u) ? 0u : 1u));  // candidate opacity
      if (md.anyHit) continue;
      bool idMatch = (c.in.found && c.in.inst == o[2] && c.in.prim == o[4]) || (c.out.found && c.out.inst == o[2] && c.out.prim == o[4]);
      bool tMatch = (c.in.found && tnear(c.in.t, gt)) || (c.out.found && tnear(c.out.t, gt));
      if (!idMatch && !tMatch) {
        std::fprintf(stderr, "[%s] ray %u: GPU t=%g inst=%u box=%u, CPU t=%g inst=%u box=%u\n", md.name, i, gt, o[2], o[4], c.out.t, c.out.inst, c.out.prim);
        std::exit(1);
      }
      if (c.in.found && !tnear(c.in.t, gt)) ++borderline;
    }
    std::printf("  %-26s rays=%u hits=%-5u misses=%-5u borderline=%u  OK\n", md.name, R, hits, misses, borderline);
    CHECK(borderline <= R / 50);
    if (md.flags == 0) CHECK(hits > R / 5);
    if (md.flags & (64u | 512u)) CHECK(hits == 0);
  }
  std::puts("aabb_test passed");
  return 0;
}
