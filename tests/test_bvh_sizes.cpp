#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#include "../layer/src/bvh_format.h"

using namespace vksoft::bvh;

#define CHECK(c)                                                     \
  do {                                                               \
    if (!(c)) {                                                      \
      std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

int main() {
  for (uint64_t n : {0ull, 1ull, 2ull, 3ull, 1000ull, 1ull << 20}) {
    uint64_t as = accelerationStructureSize(n, kTrianglePrimBytes);
    CHECK(as % kAsAlignment == 0);
    CHECK(as >= kHeaderBytes + n * kTrianglePrimBytes);
    CHECK(buildScratchSize(n) % kScratchAlignment == 0);
    CHECK(updateScratchSize(n) <= buildScratchSize(n));
    CHECK(buildScratchSize(n) >= kScratchFixed + n * 32 * 2);  // binary tree must fit
  }
  // monotonic in n
  CHECK(accelerationStructureSize(10, kTrianglePrimBytes) <=
        accelerationStructureSize(11, kTrianglePrimBytes));
  std::puts("bvh size tests passed");
  return 0;
}
