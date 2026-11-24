// End-to-end test of the ray query path: a GLSL compute shader that uses GL_EXT_ray_query is
// created through the layer (SPIR-V rewritten), bound to a TLAS descriptor (rewritten into an
// address cell) and checked against the same brute-force reference as the traversal tests.
#include <fstream>
#include <sstream>

#include "trace_ref.h"

namespace {

std::vector<char> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  CHECK(f.good());
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct PC { uint32_t count, flags, mask; };

struct RqRun {
  VkDescriptorSetLayout dsl;
  VkDescriptorPool pool;
  VkDescriptorSet set;
  VkPipelineLayout layout;
  VkShaderModule mod;
  VkPipeline pipe;

  // returns wall time in ms
  double dispatch(uint32_t count, uint32_t flags, uint32_t mask) const {
    PC pc{count, flags, mask};
    VkCommandBuffer cb = beginCb();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, (count + 63) / 64, 1, 1);
    VK(vkEndCommandBuffer(cb));
    auto t0 = std::chrono::steady_clock::now();
    submitAndWait(cb);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    vkFreeCommandBuffers(gDev, gPool, 1, &cb);
    return ms;
  }
  void destroy() {
    vkDestroyPipeline(gDev, pipe, nullptr);
    vkDestroyShaderModule(gDev, mod, nullptr);
    vkDestroyPipelineLayout(gDev, layout, nullptr);
    vkDestroyDescriptorPool(gDev, pool, nullptr);
    vkDestroyDescriptorSetLayout(gDev, dsl, nullptr);
  }
};

RqRun setupRun(const char* spvName, VkAccelerationStructureKHR tlas, const Buf& rayBuf, const Buf& resBuf) {
  RqRun r{};
  auto code = readFile(std::string(TEST_SPV_DIR) + "/" + spvName);

  // descriptor set layout: 0 = acceleration structure, 1/2 = storage buffers
  VkDescriptorSetLayoutBinding b[3]{};
  b[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = 3;
  dlci.pBindings = b;
  VkDescriptorSetLayout& dsl = r.dsl;
  VK(vkCreateDescriptorSetLayout(gDev, &dlci, nullptr, &dsl));

  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = ps;
  VkDescriptorPool& pool = r.pool;
  VK(vkCreateDescriptorPool(gDev, &dpci, nullptr, &pool));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet& set = r.set;
  VK(vkAllocateDescriptorSets(gDev, &dsai, &set));

  VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  asw.accelerationStructureCount = 1;
  asw.pAccelerationStructures = &tlas;
  VkDescriptorBufferInfo rays{rayBuf.buf, 0, VK_WHOLE_SIZE}, res{resBuf.buf, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet w[3]{};
  w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asw, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
  w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &rays, nullptr};
  w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res, nullptr};
  vkUpdateDescriptorSets(gDev, 3, w, 0, nullptr);

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout& layout = r.layout;
  VK(vkCreatePipelineLayout(gDev, &plci, nullptr, &layout));
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code.size();
  smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
  VkShaderModule& mod = r.mod;
  VK(vkCreateShaderModule(gDev, &smci, nullptr, &mod));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
  cpci.layout = layout;
  VkPipeline& pipe = r.pipe;
  VK(vkCreateComputePipelines(gDev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));
  return r;
}

void runShader(const char* spvName, const Scene& scene, const RayBatch& batch) {
  std::printf("%s:\n", spvName);
  RqRun r = setupRun(spvName, scene.tlas.as, batch.rayBuf, batch.resBuf);
  for (const Mode& md : kModes) {
    std::memset(batch.resBuf.map, 0, batch.resBuf.size);
    r.dispatch(batch.count, md.flags, md.mask);
    verifyMode(md, scene, batch);
  }
  r.destroy();
}

// Same scene and rays as trace_test's throughput run, through the ray query path.
void throughput(std::mt19937& rng) {
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
  RqRun r = setupRun("rq_test_opt.spv", bt.as, rb, ob);
  r.dispatch(N, 0, 0xFF);  // warm-up
  for (uint32_t fl : {0u, 4u}) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) best = std::min(best, r.dispatch(N, fl, 0xFF));
    uint32_t hits = 0;
    for (uint32_t i = 0; i < N; ++i) hits += static_cast<const uint32_t*>(ob.map)[size_t(i) * 8 + 1] & 1;
    std::printf("  throughput via ray_query %-22s %u rays, 1M tris: %.1f ms = %.1f Mrays/s (%.0f%% hit)\n",
                fl ? "terminate-on-first-hit" : "closest hit", N, best, N / best / 1000.0, 100.0 * hits / N);
  }
  r.destroy();
}

}  // namespace

int main() {
  initVulkan(/*rayQuery=*/true);
  initSceneUtil();
  std::mt19937 rng(2024);
  Scene scene = makeScene(rng);
  RayBatch batch = makeRays(scene, rng, 1500);
  runShader("rq_test.spv", scene, batch);
  runShader("rq_test_opt.spv", scene, batch);
  throughput(rng);
  std::puts("rq_test passed");
  return 0;
}
