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

// How the descriptors reach the shader.
enum class Variant { Writes, Template, Push, PushTemplate };
const char* variantName(Variant v) {
  switch (v) {
    case Variant::Writes: return "vkUpdateDescriptorSets";
    case Variant::Template: return "descriptor update template";
    case Variant::Push: return "push descriptors";
    default: return "push descriptors with template";
  }
}

// Application data blob for the template variants (one entry per binding).
struct TemplateData {
  VkAccelerationStructureKHR as[2];
  VkDescriptorBufferInfo rays;
  VkDescriptorBufferInfo res;
};

struct RqRun {
  Variant variant{};
  uint32_t asCount = 1;
  TemplateData data{};
  VkDescriptorSetLayout dsl{};
  VkDescriptorPool pool{};
  VkDescriptorSet set{};
  VkDescriptorUpdateTemplate tmpl{};
  VkPipelineLayout layout{};
  VkShaderModule mod{};
  VkPipeline pipe{};
  PFN_vkCmdPushDescriptorSetKHR pushDescriptors{};
  PFN_vkCmdPushDescriptorSetWithTemplateKHR pushTemplate{};

  // returns wall time in ms
  double dispatch(uint32_t count, uint32_t flags, uint32_t mask) const {
    PC pc{count, flags, mask};
    VkCommandBuffer cb = beginCb();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    if (variant == Variant::Push) {
      VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
      asw.accelerationStructureCount = asCount;
      asw.pAccelerationStructures = data.as;
      VkWriteDescriptorSet w[3]{};
      w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asw, VK_NULL_HANDLE, 0, 0, asCount, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
      w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &data.rays, nullptr};
      w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &data.res, nullptr};
      pushDescriptors(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 3, w);
    } else if (variant == Variant::PushTemplate) {
      pushTemplate(cb, tmpl, layout, 0, &data);
    } else {
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    }
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
    if (tmpl) vkDestroyDescriptorUpdateTemplate(gDev, tmpl, nullptr);
    if (pool) vkDestroyDescriptorPool(gDev, pool, nullptr);
    vkDestroyDescriptorSetLayout(gDev, dsl, nullptr);
  }
};

RqRun setupRun(const char* spvName, std::vector<VkAccelerationStructureKHR> tlases, const Buf& rayBuf, const Buf& resBuf,
               Variant variant = Variant::Writes) {
  RqRun r{};
  r.variant = variant;
  r.asCount = uint32_t(tlases.size());
  CHECK(r.asCount >= 1 && r.asCount <= 2);
  r.data = {{tlases[0], tlases.size() > 1 ? tlases[1] : VK_NULL_HANDLE}, {rayBuf.buf, 0, VK_WHOLE_SIZE}, {resBuf.buf, 0, VK_WHOLE_SIZE}};
  r.pushDescriptors = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(gDev, "vkCmdPushDescriptorSetKHR");
  r.pushTemplate = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(gDev, "vkCmdPushDescriptorSetWithTemplateKHR");
  const bool push = variant == Variant::Push || variant == Variant::PushTemplate;
  auto code = readFile(std::string(TEST_SPV_DIR) + "/" + spvName);

  VkDescriptorSetLayoutBinding b[3]{};
  b[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, r.asCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.flags = push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0;
  dlci.bindingCount = 3;
  dlci.pBindings = b;
  VK(vkCreateDescriptorSetLayout(gDev, &dlci, nullptr, &r.dsl));

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &r.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VK(vkCreatePipelineLayout(gDev, &plci, nullptr, &r.layout));

  if (variant == Variant::Template || variant == Variant::PushTemplate) {
    VkDescriptorUpdateTemplateEntry e[3] = {
        {0, 0, r.asCount, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, offsetof(TemplateData, as), sizeof(VkAccelerationStructureKHR)},
        {1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, offsetof(TemplateData, rays), sizeof(VkDescriptorBufferInfo)},
        {2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, offsetof(TemplateData, res), sizeof(VkDescriptorBufferInfo)}};
    VkDescriptorUpdateTemplateCreateInfo tci{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};
    tci.descriptorUpdateEntryCount = 3;
    tci.pDescriptorUpdateEntries = e;
    tci.templateType = variant == Variant::Template ? VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET
                                                    : VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
    tci.descriptorSetLayout = variant == Variant::Template ? r.dsl : VK_NULL_HANDLE;
    tci.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    tci.pipelineLayout = r.layout;
    tci.set = 0;
    VK(vkCreateDescriptorUpdateTemplate(gDev, &tci, nullptr, &r.tmpl));
  }

  if (!push) {
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, r.asCount}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = ps;
    VK(vkCreateDescriptorPool(gDev, &dpci, nullptr, &r.pool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = r.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &r.dsl;
    VK(vkAllocateDescriptorSets(gDev, &dsai, &r.set));
    if (variant == Variant::Template) {
      vkUpdateDescriptorSetWithTemplate(gDev, r.set, r.tmpl, &r.data);
    } else {
      VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
      asw.accelerationStructureCount = r.asCount;
      asw.pAccelerationStructures = r.data.as;
      VkWriteDescriptorSet w[3]{};
      w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asw, r.set, 0, 0, r.asCount, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
      w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &r.data.rays, nullptr};
      w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &r.data.res, nullptr};
      vkUpdateDescriptorSets(gDev, 3, w, 0, nullptr);
    }
  }

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code.size();
  smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
  VK(vkCreateShaderModule(gDev, &smci, nullptr, &r.mod));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, r.mod, "main", nullptr};
  cpci.layout = r.layout;
  VK(vkCreateComputePipelines(gDev, VK_NULL_HANDLE, 1, &cpci, nullptr, &r.pipe));
  return r;
}

void runShader(const char* spvName, const Scene& scene, const RayBatch& batch, Variant variant, bool allModes) {
  std::printf("%s via %s:\n", spvName, variantName(variant));
  RqRun r = setupRun(spvName, {scene.tlas.as}, batch.rayBuf, batch.resBuf, variant);
  static const int subset[] = {0, 1, 4};  // closest, terminate-on-first-hit, mask
  for (int i = 0; i < int(sizeof(kModes) / sizeof(kModes[0])); ++i) {
    if (!allModes && std::find(std::begin(subset), std::end(subset), i) == std::end(subset)) continue;
    const Mode& md = kModes[i];
    std::memset(batch.resBuf.map, 0, batch.resBuf.size);
    r.dispatch(batch.count, md.flags, md.mask);
    verifyMode(md, scene, batch);
  }
  r.destroy();
}

// Arrays of acceleration structures (Quake II RTX binds geometry + effects TLAS as one array and
// picks one with a constant index): the real scene sits at the index the shader uses, an empty
// TLAS at the other one, so a wrong index would miss everything.
void runArray(const Scene& scene, const RayBatch& batch, Variant variant) {
  Tlas empty = makeTlas({});
  for (int index = 0; index < 2; ++index) {
    std::vector<VkAccelerationStructureKHR> tl = index == 0 ? std::vector<VkAccelerationStructureKHR>{scene.tlas.as, empty.as}
                                                           : std::vector<VkAccelerationStructureKHR>{empty.as, scene.tlas.as};
    const std::string name = "rq_array" + std::to_string(index) + ".spv";
    std::printf("%s via %s (index %d of 2):\n", name.c_str(), variantName(variant), index);
    RqRun r = setupRun(name.c_str(), tl, batch.rayBuf, batch.resBuf, variant);
    for (int m : {0, 4}) {
      std::memset(batch.resBuf.map, 0, batch.resBuf.size);
      r.dispatch(batch.count, kModes[m].flags, kModes[m].mask);
      verifyMode(kModes[m], scene, batch);
    }
    r.destroy();
  }
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
  RqRun r = setupRun("rq_test_opt.spv", {bt.as}, rb, ob);
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
  runShader("rq_test.spv", scene, batch, Variant::Writes, true);
  runShader("rq_test_opt.spv", scene, batch, Variant::Writes, true);
  runShader("rq_test_opt.spv", scene, batch, Variant::Template, false);
  runShader("rq_test_opt.spv", scene, batch, Variant::Push, false);
  runShader("rq_test_opt.spv", scene, batch, Variant::PushTemplate, false);
  runArray(scene, batch, Variant::Writes);
  runArray(scene, batch, Variant::Template);
  throughput(rng);
  std::puts("rq_test passed");
  return 0;
}
