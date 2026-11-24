// Shared by all vksoft_rt build kernels. Everything is addressed through buffer device
// addresses (the layer never owns descriptors for app memory); parameters arrive in a
// 128-byte push constant block: 8 addresses + 16 words, meaning defined per kernel.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer CWords { uint v[]; };

layout(push_constant) uniform PC {
  uint64_t a[8];
  uint u[16];
} pc;

const uint NONE = 0xFFFFFFFFu;
const uint LEAF_BIT = 0x80000000u;
const uint HEADER_BYTES = 128u;
const uint NODE_WORDS = 16u;   // 64 B: aabb0(6f) aabb1(6f) ref0 ref1 pad pad
const uint TRI_WORDS = 12u;    // v0 v1 v2 (9f), geometryIndex, primitiveIndex, flags
const uint TRI_BYTES = 48u;

uint ld(uint64_t base, uint i) { return Words(base).v[i]; }
void st(uint64_t base, uint i, uint x) { Words(base).v[i] = x; }
float ldf(uint64_t base, uint i) { return uintBitsToFloat(Words(base).v[i]); }
void stf(uint64_t base, uint i, float x) { Words(base).v[i] = floatBitsToUint(x); }

// Order-preserving float <-> uint mapping so float min/max can use integer atomics.
uint f2o(float f) {
  uint u = floatBitsToUint(f);
  return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}
float o2f(uint o) {
  return uintBitsToFloat((o & 0x80000000u) != 0u ? (o & 0x7FFFFFFFu) : ~o);
}

const uint AS_MAGIC = 0x42534B56u;  // 'VKSB'
const uint INSTANCE_WORDS = 32u;

uint64_t ld64(uint64_t base, uint i) { return uint64_t(ld(base, i)) | (uint64_t(ld(base, i + 1u)) << 32); }

// Address of application instance i (VkAccelerationStructureInstanceKHR, 64 B), either a
// packed array or an array of pointers.
uint64_t instanceAddr(uint64_t base, uint ptrMode, uint i) {
  return ptrMode != 0u ? ld64(base, i * 2u) : base + uint64_t(i) * 64ul;
}
