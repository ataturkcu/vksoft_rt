// Two-level BVH traversal, written as a resumable state machine so that the same code
// serves ray queries (candidate hits are handed back to the caller) and, later, the
// ray-tracing-pipeline megakernel. Include after common.glsl.
//
// Semantics follow the Vulkan ray traversal rules for: ray flags, instance flags, masks,
// opaque/non-opaque geometry, facing + culling, and watertight triangle intersection
// (Woop et al. 2013). Triangle facing is determined in object space.

const uint RF_OPAQUE = 1u, RF_NO_OPAQUE = 2u, RF_TERMINATE_FIRST = 4u, RF_CULL_BACK = 16u,
           RF_CULL_FRONT = 32u, RF_CULL_OPAQUE = 64u, RF_CULL_NO_OPAQUE = 128u,
           RF_SKIP_TRIANGLES = 256u, RF_SKIP_AABBS = 512u;
const uint IF_FACING_CULL_DISABLE = 1u, IF_FLIP_FACING = 2u, IF_FORCE_OPAQUE = 4u,
           IF_FORCE_NO_OPAQUE = 8u;

const uint STACK_DEPTH = 64u;
const uint MARKER = 0xFFFFFFFEu;  // pushed when descending into a BLAS

struct HitInfo {
  float t;
  float u, v;        // barycentrics of vertices 1 and 2
  uint instanceIndex;
  uint customIndex;
  uint sbtOffset;
  uint geometryIndex;
  uint primitiveIndex;
  bool frontFace;
  uint64_t instRecord;  // address of the TLAS instance record (transforms, ids)
  uint kind;            // 0 none, 1 triangle
};

struct Trav {
  // ray in world space
  vec3 wo, wd;
  float tmin, limit;
  bool strict;  // true once something is committed: later hits must be strictly closer
  uint flags, mask;
  // ray in the space of the AS being traversed
  vec3 o, d, inv;
  // AS being traversed
  uint64_t nodeBase, primBase;
  bool inBlas;
  uint64_t instRec;
  uint instFlags;
  // traversal state
  uint stack[STACK_DEPTH];
  uint sp;
  uint cur;  // next reference to process, NONE = pop
  bool done;
  HitInfo committed, cand;
};

HitInfo noHit() {
  HitInfo h;
  h.t = 0.0; h.u = 0.0; h.v = 0.0;
  h.instanceIndex = 0u; h.customIndex = 0u; h.sbtOffset = 0u;
  h.geometryIndex = 0u; h.primitiveIndex = 0u; h.frontFace = false;
  h.instRecord = 0ul; h.kind = 0u;
  return h;
}

vec3 safeInv(vec3 d) {
  // Avoid 0 * inf = NaN in the slab test; a tiny signed direction keeps it finite.
  vec3 s = vec3(d.x >= 0.0 ? 1.0 : -1.0, d.y >= 0.0 ? 1.0 : -1.0, d.z >= 0.0 ? 1.0 : -1.0);
  return 1.0 / (s * max(abs(d), vec3(1e-20)));
}

void setSpace(inout Trav s, vec3 o, vec3 d) {
  s.o = o; s.d = d; s.inv = safeInv(d);
}

void useAS(inout Trav s, uint64_t base) {
  s.nodeBase = base + uint64_t(ld(base, 5u));
  s.primBase = base + uint64_t(ld(base, 6u));
}

void rayInit(inout Trav s, uint64_t tlas, vec3 o, vec3 d, float tmin, float tmax, uint flags, uint cullMask) {
  s.wo = o; s.wd = d; s.tmin = tmin; s.limit = tmax; s.strict = false;
  s.flags = flags; s.mask = cullMask;
  setSpace(s, o, d);
  s.inBlas = false; s.instRec = 0ul; s.instFlags = 0u;
  s.sp = 0u; s.done = false;
  s.committed = noHit(); s.cand = noHit();
  s.cur = NONE;
  if (tlas == 0ul || ld(tlas, 0u) != AS_MAGIC || ld(tlas, 1u) != 1u) { s.done = true; return; }
  useAS(s, tlas);
  s.cur = ld(tlas, 4u);
  if (s.cur == NONE) s.done = true;
}

// Slab test against a box stored at node word offset `w`. Direction-signed slab selection
// makes empty boxes (min=+inf,max=-inf) always miss. Returns entry distance or -1.
float boxHit(uint64_t node, uint w, vec3 o, vec3 inv, float tmin, float limit) {
  vec3 mn = vec3(ldf(node, w), ldf(node, w + 1u), ldf(node, w + 2u));
  vec3 mx = vec3(ldf(node, w + 3u), ldf(node, w + 4u), ldf(node, w + 5u));
  vec3 lo = vec3(inv.x >= 0.0 ? mn.x : mx.x, inv.y >= 0.0 ? mn.y : mx.y, inv.z >= 0.0 ? mn.z : mx.z);
  vec3 hi = vec3(inv.x >= 0.0 ? mx.x : mn.x, inv.y >= 0.0 ? mx.y : mn.y, inv.z >= 0.0 ? mx.z : mn.z);
  vec3 t0 = (lo - o) * inv;
  vec3 t1 = (hi - o) * inv;
  float tn = max(max(t0.x, t0.y), max(t0.z, tmin));
  float tf = min(min(t1.x, t1.y), min(t1.z, limit));
  // Slightly widen the far side so rays grazing a box edge are not lost to rounding.
  return tn <= tf * 1.0000004 ? tn : -1.0;
}

bool intersectTriangle(vec3 o, vec3 d, vec3 v0, vec3 v1, vec3 v2, out float t, out float bu, out float bv) {
  t = 0.0; bu = 0.0; bv = 0.0;
  vec3 ad = abs(d);
  uint kz = ad.x > ad.y ? (ad.x > ad.z ? 0u : 2u) : (ad.y > ad.z ? 1u : 2u);
  uint kx = kz == 2u ? 0u : kz + 1u;
  uint ky = kx == 2u ? 0u : kx + 1u;
  if (d[kz] < 0.0) { uint tmp = kx; kx = ky; ky = tmp; }
  float Sz = 1.0 / d[kz];
  float Sx = d[kx] * Sz, Sy = d[ky] * Sz;
  vec3 A = v0 - o, B = v1 - o, C = v2 - o;
  float Ax = A[kx] - Sx * A[kz], Ay = A[ky] - Sy * A[kz];
  float Bx = B[kx] - Sx * B[kz], By = B[ky] - Sy * B[kz];
  float Cx = C[kx] - Sx * C[kz], Cy = C[ky] - Sy * C[kz];
  float U = Cx * By - Cy * Bx;
  float V = Ax * Cy - Ay * Cx;
  float W = Bx * Ay - By * Ax;
  if ((U < 0.0 || V < 0.0 || W < 0.0) && (U > 0.0 || V > 0.0 || W > 0.0)) return false;
  float det = U + V + W;
  if (det == 0.0) return false;
  float T = U * (Sz * A[kz]) + V * (Sz * B[kz]) + W * (Sz * C[kz]);
  float r = 1.0 / det;
  t = T * r; bu = V * r; bv = W * r;
  return true;
}

void leaveBlas(inout Trav s) {
  s.inBlas = false;
  s.instRec = 0ul; s.instFlags = 0u;
  setSpace(s, s.wo, s.wd);
}

void rayCommit(inout Trav s, HitInfo h) {
  s.committed = h;
  s.limit = h.t;
  s.strict = true;
  if ((s.flags & RF_TERMINATE_FIRST) != 0u) s.done = true;
}

// Triangle leaf. Returns true when a non-opaque candidate was recorded (caller decides).
bool processTriangle(inout Trav s, uint64_t tri) {
  if ((s.flags & RF_SKIP_TRIANGLES) != 0u) return false;
  bool opaque = (ld(tri, 11u) & 1u) != 0u;
  if (s.inBlas) {
    if ((s.instFlags & IF_FORCE_OPAQUE) != 0u) opaque = true;
    else if ((s.instFlags & IF_FORCE_NO_OPAQUE) != 0u) opaque = false;
  }
  if ((s.flags & RF_OPAQUE) != 0u) opaque = true;
  else if ((s.flags & RF_NO_OPAQUE) != 0u) opaque = false;
  if (opaque && (s.flags & RF_CULL_OPAQUE) != 0u) return false;
  if (!opaque && (s.flags & RF_CULL_NO_OPAQUE) != 0u) return false;

  vec3 v0 = vec3(ldf(tri, 0u), ldf(tri, 1u), ldf(tri, 2u));
  vec3 v1 = vec3(ldf(tri, 3u), ldf(tri, 4u), ldf(tri, 5u));
  vec3 v2 = vec3(ldf(tri, 6u), ldf(tri, 7u), ldf(tri, 8u));
  float t, bu, bv;
  if (!intersectTriangle(s.o, s.d, v0, v1, v2, t, bu, bv)) return false;
  if (t < s.tmin || (s.strict ? t >= s.limit : t > s.limit)) return false;

  bool front = dot(cross(v1 - v0, v2 - v0), s.d) < 0.0;
  if (s.inBlas && (s.instFlags & IF_FLIP_FACING) != 0u) front = !front;
  if (!(s.inBlas && (s.instFlags & IF_FACING_CULL_DISABLE) != 0u)) {
    if (!front && (s.flags & RF_CULL_BACK) != 0u) return false;
    if (front && (s.flags & RF_CULL_FRONT) != 0u) return false;
  }

  HitInfo h;
  h.t = t; h.u = bu; h.v = bv;
  h.instanceIndex = ld(s.instRec, 28u);
  h.customIndex = ld(s.instRec, 24u) & 0xFFFFFFu;
  h.sbtOffset = ld(s.instRec, 25u) & 0xFFFFFFu;
  h.geometryIndex = ld(tri, 9u);
  h.primitiveIndex = ld(tri, 10u);
  h.frontFace = front;
  h.instRecord = s.instRec;
  h.kind = 1u;
  if (opaque) { rayCommit(s, h); return false; }
  s.cand = h;
  return true;
}

// Advance until a non-opaque candidate needs a decision (returns true) or traversal ends
// (returns false). After a candidate, call rayConfirmCandidate() to accept it, then
// proceed again.
bool rayProceed(inout Trav s, uint64_t tlasBase) {
  while (!s.done) {
    if (s.cur == NONE) {  // pop
      bool got = false;
      while (s.sp > 0u) {
        uint r = s.stack[--s.sp];
        if (r == MARKER) { leaveBlas(s); useAS(s, tlasBase); continue; }
        s.cur = r; got = true; break;
      }
      if (!got) { s.done = true; break; }
      continue;
    }
    uint ref = s.cur;
    if ((ref & LEAF_BIT) == 0u) {  // internal node: test both children, near first
      uint64_t node = s.nodeBase + uint64_t(ref) * 64ul;
      float t0 = boxHit(node, 0u, s.o, s.inv, s.tmin, s.limit);
      float t1 = boxHit(node, 6u, s.o, s.inv, s.tmin, s.limit);
      uint c0 = ld(node, 12u), c1 = ld(node, 13u);
      bool h0 = t0 >= 0.0, h1 = t1 >= 0.0;  // boxHit: entry distance >= tmin >= 0, or -1 for a miss
      if (h0 && h1) {
        bool firstIs0 = t0 <= t1;
        uint nearC = firstIs0 ? c0 : c1, farC = firstIs0 ? c1 : c0;
        if (s.sp < STACK_DEPTH) s.stack[s.sp++] = farC;
        s.cur = nearC;
      } else if (h0) s.cur = c0;
      else if (h1) s.cur = c1;
      else s.cur = NONE;
      continue;
    }
    uint k = ref & ~LEAF_BIT;
    s.cur = NONE;
    if (!s.inBlas) {  // TLAS leaf = instance
      uint64_t rec = s.primBase + uint64_t(k) * 128ul;
      uint w24 = ld(rec, 24u);
      if ((s.mask & (w24 >> 24)) == 0u) continue;
      uint64_t blas = ld64(rec, 26u);
      if (blas == 0ul || ld(blas, 0u) != AS_MAGIC || ld(blas, 1u) != 0u) continue;
      uint root = ld(blas, 4u);
      if (root == NONE) continue;
      float m[12];
      for (uint i = 0u; i < 12u; ++i) m[i] = ldf(rec, 12u + i);
      vec3 o = vec3(m[0] * s.wo.x + m[1] * s.wo.y + m[2] * s.wo.z + m[3],
                    m[4] * s.wo.x + m[5] * s.wo.y + m[6] * s.wo.z + m[7],
                    m[8] * s.wo.x + m[9] * s.wo.y + m[10] * s.wo.z + m[11]);
      vec3 d = vec3(m[0] * s.wd.x + m[1] * s.wd.y + m[2] * s.wd.z,
                    m[4] * s.wd.x + m[5] * s.wd.y + m[6] * s.wd.z,
                    m[8] * s.wd.x + m[9] * s.wd.y + m[10] * s.wd.z);
      if (s.sp < STACK_DEPTH) s.stack[s.sp++] = MARKER;
      s.inBlas = true;
      s.instRec = rec;
      s.instFlags = ld(rec, 25u) >> 24;
      setSpace(s, o, d);
      useAS(s, blas);
      s.cur = root;
      continue;
    }
    if (processTriangle(s, s.primBase + uint64_t(k) * 48ul)) return true;
  }
  return false;
}

void rayConfirmCandidate(inout Trav s) {
  if (s.cand.kind != 0u) { rayCommit(s, s.cand); s.cand.kind = 0u; }
}
