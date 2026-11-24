// GPU traversal test: shoots rays against a TLAS/BLAS scene built by the layer using the
// traversal code from layer/shaders/traverse.glsl and compares every result with a
// brute-force double-precision CPU reference. Also reports a rough ray throughput.
#include <fstream>
#include <sstream>

#include "trace_ref.h"

namespace {

struct Push {
  uint64_t a[8];
  uint32_t u[16];
};

struct Pipe {
  VkPipelineLayout layout{};
  VkPipeline pipe{};
};

Pipe makePipe() {
  std::ifstream f(TRACE_TEST_SPV, std::ios::binary);
  CHECK(f.good());
  std::stringstream ss;
  ss << f.rdbuf();
  std::string code = ss.str();
  Pipe p;
  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &range;
  VK(vkCreatePipelineLayout(gDev, &plci, nullptr, &p.layout));
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code.size();
  smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
  VkShaderModule mod;
  VK(vkCreateShaderModule(gDev, &smci, nullptr, &mod));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod,
                "main", nullptr};
  cpci.layout = p.layout;
  VK(vkCreateComputePipelines(gDev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipe));
  vkDestroyShaderModule(gDev, mod, nullptr);
  return p;
}

// Returns wall time in ms of the dispatch.
double traceOnGpu(const Pipe& p, VkDeviceAddress tlas, const Buf& rays, const Buf& results, uint32_t count,
                  uint32_t flags, uint32_t mask) {
  Push pc{};
  pc.a[0] = tlas;
  pc.a[1] = rays.addr;
  pc.a[2] = results.addr;
  pc.u[0] = count;
  pc.u[1] = flags;
  pc.u[2] = mask;
  VkCommandBuffer cb = beginCb();
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
  vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdDispatch(cb, (count + 63) / 64, 1, 1);
  VK(vkEndCommandBuffer(cb));
  auto t0 = std::chrono::steady_clock::now();
  submitAndWait(cb);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  vkFreeCommandBuffers(gDev, gPool, 1, &cb);
  return ms;
}

}  // namespace

int main() {
  initVulkan();
  initSceneUtil();
  Pipe pipe = makePipe();
  std::mt19937 rng(2024);

  // ---------------- correctness ----------------
  Scene scene = makeScene(rng);
  const uint32_t R = 1500;
  RayBatch batch = makeRays(scene, rng, R);
  std::puts("trace_test:");
  for (const Mode& md : kModes) {
    traceOnGpu(pipe, scene.tlas.addr, batch.rayBuf, batch.resBuf, R, md.flags, md.mask);
    verifyMode(md, scene, batch);
  }

  // ---------------- throughput (rough) ----------------
  {
    Blas big = makeBlas(rng, 250000, 20.f);
    std::vector<VkAccelerationStructureInstanceKHR> bi(4);
    for (uint32_t i = 0; i < 4; ++i) {
      Mat m = randomTransform(rng, false);
      for (int k = 0; k < 12; ++k) if (k % 4 == 3) m.m[k] = (k / 4 - 1) * 12.f + i * 3.f; else m.m[k] = (k % 4 == k / 4) ? 1.f : 0.f;
      std::memcpy(bi[i].transform.matrix, m.m, sizeof(m.m));
      bi[i].mask = 0xFF;
      bi[i].accelerationStructureReference = big.addr;
    }
    Tlas bt = makeTlas(bi);
    const uint32_t N = 1u << 20;
    Buf rb = makeBuf(size_t(N) * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    Buf ob = makeBuf(size_t(N) * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    std::uniform_real_distribution<float> P(-30.f, 30.f), Dd(-1.f, 1.f);
    for (uint32_t i = 0; i < N; ++i) {
      float* f = static_cast<float*>(rb.map) + size_t(i) * 8;
      float d[3] = {Dd(rng), Dd(rng), Dd(rng)};
      float l = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) + 1e-6f;
      for (int a = 0; a < 3; ++a) { f[a] = P(rng); f[4 + a] = d[a] / l; }
      f[3] = 0.f;
      f[7] = 200.f;
    }
    traceOnGpu(pipe, bt.addr, rb, ob, N, 0, 0xFF);  // warm-up
    for (uint32_t fl : {0u, 4u}) {
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) best = std::min(best, traceOnGpu(pipe, bt.addr, rb, ob, N, fl, 0xFF));
      uint32_t hits = 0;
      for (uint32_t i = 0; i < N; ++i) hits += static_cast<const uint32_t*>(ob.map)[size_t(i) * 8 + 1] & 1;
      std::printf("  throughput %-22s %u rays, 1M tris, 4 instances: %.1f ms = %.1f Mrays/s (%.0f%% hit)\n",
                  fl ? "terminate-on-first-hit" : "closest hit", N, best, N / best / 1000.0, 100.0 * hits / N);
    }
  }
  std::puts("trace_test passed");
  return 0;
}
