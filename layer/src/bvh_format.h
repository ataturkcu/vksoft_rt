// vksoft_rt BVH memory layout constants and size bounds (host + shader shared).
//
// These are *upper bounds* used to answer vkGetAccelerationStructureBuildSizesKHR
// before any geometry data is visible, so they must hold for every possible build.
// The concrete node encoding (quantized BVH4/8) is refined in later M1/M3 steps;
// changing the numbers here is cheap as long as the bounds stay conservative.
#pragma once

#include <cstdint>

namespace vksoft::bvh {

constexpr uint64_t kAsAlignment = 256;      // required AS offset alignment (Vulkan spec)
constexpr uint64_t kScratchAlignment = 256; // >= minAccelerationStructureScratchOffsetAlignment

constexpr uint64_t kHeaderBytes = 128;      // AS header: magic, type, counts, root, bounds
constexpr uint64_t kNodeBytes = 64;         // wide (BVH4) node with quantized child boxes
constexpr uint64_t kTrianglePrimBytes = 48; // 3 x vec3 + geometry/primitive ids
constexpr uint64_t kAabbPrimBytes = 32;     // aabb + geometry/primitive ids
constexpr uint64_t kInstanceBytes = 96;     // 3x4 transform + ids/mask/flags + BLAS address

// Scratch per primitive for a full build: morton keys + sort ping-pong + binary tree
// (2N-1 nodes x 32 B) before collapse into wide nodes.
constexpr uint64_t kBuildScratchPerPrim = 128;
// Refit needs only per-node bookkeeping.
constexpr uint64_t kUpdateScratchPerPrim = 16;
constexpr uint64_t kScratchFixed = 4096;    // histograms, counters, indirect args

constexpr uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// A wide BVH over N leaves has at most N-1 internal nodes (each has >= 2 children).
constexpr uint64_t maxInternalNodes(uint64_t n) { return n > 1 ? n - 1 : 1; }

enum class Kind { Triangles, Aabbs, Instances };

constexpr uint64_t primBytes(Kind k) {
  return k == Kind::Triangles ? kTrianglePrimBytes
       : k == Kind::Aabbs     ? kAabbPrimBytes
                              : kInstanceBytes;
}

// Worst-case acceleration-structure size for `n` primitives of one kind.
// Geometries of mixed kinds in one BLAS are bounded by passing the max prim size.
constexpr uint64_t accelerationStructureSize(uint64_t n, uint64_t perPrimBytes) {
  return alignUp(kHeaderBytes + maxInternalNodes(n) * kNodeBytes + n * perPrimBytes,
                 kAsAlignment);
}
constexpr uint64_t buildScratchSize(uint64_t n) {
  return alignUp(kScratchFixed + n * kBuildScratchPerPrim, kScratchAlignment);
}
constexpr uint64_t updateScratchSize(uint64_t n) {
  return alignUp(kScratchFixed + n * kUpdateScratchPerPrim, kScratchAlignment);
}

}  // namespace vksoft::bvh
