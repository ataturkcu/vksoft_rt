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
constexpr uint64_t kAabbPrimBytes = 48;     // box leaf record, same stride as a triangle (flags bit1 = procedural)
constexpr uint64_t kInstanceBytes = 128;    // TLAS leaf record, see below
// TLAS instance record (32 words): [0..11] objectToWorld 3x4 row-major, [12..23]
// worldToObject 3x4, [24] customIndex(24)|mask(8)<<24, [25] sbtOffset(24)|flags(8)<<24,
// [26,27] BLAS address lo/hi (0 = inactive), [28] instance index, [29..31] reserved.
// Inactive instances keep an empty box (min=+inf, max=-inf) and mask 0.

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

// ---- build scratch layout ---------------------------------------------------
// Byte offsets inside the application-provided scratch buffer for a triangle BLAS build
// over n primitives. Must always fit into buildScratchSize(n) (unit-tested).
constexpr uint32_t kRadixBlock = 256;  // elements per radix workgroup
constexpr uint32_t kScanChunk = 1024;  // elements per scan workgroup
constexpr uint64_t kTriangleScratchBytes = 48;

struct ScratchLayout {
  uint64_t bounds, flags, keysA, valsA, keysB, valsB, parent, hist, chunk, tris, total;
  uint32_t blocks;      // radix workgroups
  uint32_t histWords;   // 256 * blocks
  uint32_t chunkCount;  // scan chunks over the histogram
};

constexpr ScratchLayout scratchLayout(uint64_t n) {
  ScratchLayout l{};
  uint64_t cnt = n ? n : 1;
  l.blocks = static_cast<uint32_t>((cnt + kRadixBlock - 1) / kRadixBlock);
  l.histWords = 256u * l.blocks;
  l.chunkCount = (l.histWords + kScanChunk - 1) / kScanChunk;
  uint64_t o = 0;
  auto take = [&](uint64_t bytes) { uint64_t at = o; o += alignUp(bytes, 16); return at; };
  l.bounds = take(32);
  l.flags = take(cnt * 4);
  l.keysA = take(cnt * 4);
  l.valsA = take(cnt * 4);
  l.keysB = take(cnt * 4);
  l.valsB = take(cnt * 4);
  l.parent = take(2 * cnt * 4);
  l.hist = take(uint64_t(l.histWords) * 4);
  l.chunk = take(uint64_t(l.chunkCount) * 4);
  l.tris = take(cnt * kTriangleScratchBytes);
  l.total = o;
  return l;
}

}  // namespace vksoft::bvh
