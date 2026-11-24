// Host-only test of the ray query SPIR-V rewriter: every result is validated with the
// SPIRV-Tools validator (Vulkan 1.2 rules) and inspected for leftover ray query instructions.
#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../layer/src/rq_rewrite.h"

#define CHECK(c)                                                              \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);     \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static std::vector<uint32_t> load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  CHECK(f.good());
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  CHECK(b.size() % 4 == 0);
  std::vector<uint32_t> w(b.size() / 4);
  std::memcpy(w.data(), b.data(), b.size());
  return w;
}

int main() {
  using namespace vksoft;
  const std::string dir = TEST_SPV_DIR;
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);

  // ray query shaders, unoptimized and optimized
  for (const char* name : {"rq_test.spv", "rq_test_opt.spv", "rq_two.spv", "rq_two_opt.spv"}) {
    auto in = load(dir + "/" + name);
    std::vector<uint32_t> out;
    std::string err;
    RewriteResult r = rewriteRayQueries(in.data(), in.size(), out, err, /*validate=*/true);
    if (r != RewriteResult::Rewritten) std::fprintf(stderr, "%s: %s\n", name, err.c_str());
    CHECK(r == RewriteResult::Rewritten);
    std::string text;
    CHECK(tools.Disassemble(out, &text, SPV_BINARY_TO_TEXT_OPTION_NONE));
    CHECK(text.find("RayQuery") == std::string::npos);
    CHECK(text.find("AccelerationStructure") == std::string::npos);
    CHECK(text.find("PhysicalStorageBuffer64") != std::string::npos);
    std::printf("  %-18s rewritten: %zu -> %zu words, valid\n", name, in.size(), out.size());
  }
  // a module without ray queries is left alone
  {
    auto in = load(dir + "/plain.spv");
    std::vector<uint32_t> out;
    std::string err;
    CHECK(rewriteRayQueries(in.data(), in.size(), out, err, true) == RewriteResult::NotNeeded);
    std::puts("  plain.spv          untouched");
  }
  // unsupported pattern: must fail cleanly with a message
  {
    auto in = load(dir + "/rq_param.spv");
    std::vector<uint32_t> out;
    std::string err;
    RewriteResult r = rewriteRayQueries(in.data(), in.size(), out, err, true);
    CHECK(r == RewriteResult::Failed);
    CHECK(!err.empty());
    std::printf("  rq_param.spv       rejected: %s\n", err.c_str());
  }
  std::puts("rq_rewrite passed");
  return 0;
}
