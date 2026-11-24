// CPU reference and scene/ray generation shared by the traversal tests (trace_test runs the
// traversal library from a compute shader, rq_test runs the same checks through the ray
// query rewriter).
#pragma once
#include "scene_util.h"

namespace {

struct WTri {
  double v[3][3];
  uint32_t inst, prim, geom;
  double detSign;
  uint32_t instFlags, instMask;
};

struct RayCpu { double o[3], d[3], tmax; };

struct CpuHit {
  bool found = false;
  double t = 0;
  uint32_t inst = 0, prim = 0;
};

double det3(const float* m) {
  return double(m[0]) * (double(m[5]) * m[10] - double(m[6]) * m[9]) - double(m[1]) * (double(m[4]) * m[10] - double(m[6]) * m[8]) +
         double(m[2]) * (double(m[4]) * m[9] - double(m[5]) * m[8]);
}

void sub3(const double* a, const double* b, double* o) { for (int i = 0; i < 3; ++i) o[i] = a[i] - b[i]; }
void cross3(const double* a, const double* b, double* o) {
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Moller-Trumbore, two-sided. margin = min barycentric coordinate.
bool intersect(const RayCpu& r, const WTri& t, double& tt, double& u, double& v, double& margin) {
  double e1[3], e2[3], p[3], tv[3], q[3];
  sub3(t.v[1], t.v[0], e1);
  sub3(t.v[2], t.v[0], e2);
  cross3(r.d, e2, p);
  double det = dot3(e1, p);
  if (!(std::fabs(det) > 1e-12)) return false;  // parallel (also rejects NaN)
  double inv = 1.0 / det;
  sub3(r.o, t.v[0], tv);
  u = dot3(tv, p) * inv;
  cross3(tv, e1, q);
  v = dot3(r.d, q) * inv;
  tt = dot3(e2, q) * inv;
  margin = std::min({u, v, 1.0 - u - v});
  return true;
}

// Facing in object space, taking instance flip flag into account.
bool frontFacing(const RayCpu& r, const WTri& t) {
  double e1[3], e2[3], n[3];
  sub3(t.v[1], t.v[0], e1);
  sub3(t.v[2], t.v[0], e2);
  cross3(e1, e2, n);
  bool front = dot3(n, r.d) * t.detSign < 0;
  if (t.instFlags & 2u) front = !front;
  return front;
}

// Applies ray flags / mask / culling exactly like the traversal rules (geometry is opaque).
bool passesFilters(const RayCpu& r, const WTri& t, uint32_t flags, uint32_t mask) {
  if ((mask & t.instMask) == 0) return false;
  if (flags & 256u) return false;  // skip triangles
  bool opaque = true;
  if (flags & 1u) opaque = true; else if (flags & 2u) opaque = false;
  if (opaque && (flags & 64u)) return false;
  if (!opaque && (flags & 128u)) return false;
  if (!(t.instFlags & 1u)) {
    bool front = frontFacing(r, t);
    if (!front && (flags & 16u)) return false;
    if (front && (flags & 32u)) return false;
  }
  return true;
}

struct Closest { CpuHit in, out; bool any = false; };

Closest cpuClosest(const RayCpu& r, const std::vector<WTri>& tris, uint32_t flags, uint32_t mask) {
  Closest c;
  // float32 traversal of coordinates ~1e2 carries ~1e-3 absolute error, i.e. a few 1e-3 in
  // barycentric margin for these small triangles; real bugs produce gross mismatches.
  const double eps = 3e-3;
  for (const WTri& t : tris) {
    double tt, u, v, m;
    if (!intersect(r, t, tt, u, v, m)) continue;
    if (!(tt >= 0 && tt <= r.tmax && m >= -eps)) continue;  // written to reject NaN too
    if (!passesFilters(r, t, flags, mask)) continue;
    c.any = true;
    if (!c.out.found || tt < c.out.t) c.out = {true, tt, t.inst, t.prim};
    if (m >= eps && (!c.in.found || tt < c.in.t)) c.in = {true, tt, t.inst, t.prim};
  }
  return c;
}

bool tnear(double a, double b) { return std::fabs(a - b) <= 2e-4 * (1.0 + std::fabs(a) + std::fabs(b)); }


struct Scene {
  Blas A, B;
  std::vector<VkAccelerationStructureInstanceKHR> insts;
  std::vector<WTri> world;
  std::vector<uint32_t> instActive;
  Tlas tlas;
  uint32_t nInst = 64;
};

Scene makeScene(std::mt19937& rng) {
  Scene sc;
  sc.A = makeBlas(rng, 1000, 5.f);
  sc.B = makeBlas(rng, 300, 2.f);
  sc.insts.resize(sc.nInst);
  sc.instActive.assign(sc.nInst, 0);
  for (uint32_t i = 0; i < sc.nInst; ++i) {
    const Blas& blas = (rng() & 1) ? sc.A : sc.B;
    Mat m = randomTransform(rng, false);
    if (i % 7 == 3) { m.m[0] = -m.m[0]; m.m[4] = -m.m[4]; m.m[8] = -m.m[8]; }  // mirrored: det < 0
    bool active = (rng() % 100) >= 5;
    VkAccelerationStructureInstanceKHR in{};
    std::memcpy(in.transform.matrix, m.m, sizeof(m.m));
    in.instanceCustomIndex = 1000 + i;
    in.mask = 1 + (i % 255);
    in.flags = (i % 11 == 0 ? 2u : 0u) | (i % 13 == 0 ? 1u : 0u);
    in.accelerationStructureReference = active ? blas.addr : 0;
    sc.insts[i] = in;
    sc.instActive[i] = active;
    if (!active) continue;
    const float* vv = static_cast<const float*>(blas.verts.map);
    double ds = det3(m.m) < 0 ? -1.0 : 1.0;
    for (uint32_t t = 0; t < blas.nTris; ++t) {
      WTri w{};
      for (int k = 0; k < 3; ++k)
        for (int r = 0; r < 3; ++r) {
          const float* p = &vv[t * 9 + k * 3];
          w.v[k][r] = double(m.m[r * 4 + 0]) * p[0] + double(m.m[r * 4 + 1]) * p[1] + double(m.m[r * 4 + 2]) * p[2] + m.m[r * 4 + 3];
        }
      w.inst = i; w.prim = t; w.geom = 0; w.detSign = ds; w.instFlags = in.flags; w.instMask = in.mask;
      sc.world.push_back(w);
    }
  }
  sc.tlas = makeTlas(sc.insts);
  return sc;
}

struct RayBatch {
  std::vector<RayCpu> rays;
  Buf rayBuf, resBuf;
  uint32_t count = 0;
};

// Half the rays are aimed at random triangles (guaranteed hits), some are axis aligned.
RayBatch makeRays(const Scene& sc, std::mt19937& rng, uint32_t R) {
  RayBatch rb;
  rb.count = R;
  std::uniform_real_distribution<double> U(-60.0, 60.0), D(-1.0, 1.0), J(-0.05, 0.05);
  rb.rays.resize(R);
  rb.rayBuf = makeBuf(R * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  rb.resBuf = makeBuf(R * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  for (uint32_t i = 0; i < R; ++i) {
    RayCpu r{};
    for (int a = 0; a < 3; ++a) r.o[a] = U(rng);
    uint32_t kind = rng() % 100;
    if (kind < 50) {
      const WTri& t = sc.world[rng() % sc.world.size()];
      for (int a = 0; a < 3; ++a) r.d[a] = (t.v[0][a] + t.v[1][a] + t.v[2][a]) / 3.0 + J(rng) - r.o[a];
    } else if (kind < 56) {
      static const double axes[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      const uint32_t axis = rng() % 6;  // one axis for all components (a per-component pick can give a zero vector)
      for (int a = 0; a < 3; ++a) r.d[a] = axes[axis][a];
    } else {
      for (int a = 0; a < 3; ++a) r.d[a] = D(rng);
    }
    double len = std::sqrt(dot3(r.d, r.d));
    for (double& x : r.d) x /= len;
    r.tmax = (rng() % 4 == 0) ? 40.0 : 400.0;
    rb.rays[i] = r;
    float* f = static_cast<float*>(rb.rayBuf.map) + i * 8;
    for (int a = 0; a < 3; ++a) { f[a] = float(r.o[a]); f[4 + a] = float(r.d[a]); }
    f[3] = 0.f;
    f[7] = float(r.tmax);
  }
  // the CPU reference uses the float values the GPU sees
  for (uint32_t i = 0; i < R; ++i) {
    const float* f = static_cast<const float*>(rb.rayBuf.map) + i * 8;
    for (int a = 0; a < 3; ++a) { rb.rays[i].o[a] = f[a]; rb.rays[i].d[a] = f[4 + a]; }
  }
  return rb;
}

struct Mode { const char* name; uint32_t flags, mask; bool anyHit; };
const Mode kModes[] = {
    {"closest hit", 0, 0xFFFFFFFFu, false},
    {"terminate on first hit", 4, 0xFFFFFFFFu, true},
    {"cull back faces", 16, 0xFFFFFFFFu, false},
    {"cull front faces", 32, 0xFFFFFFFFu, false},
    {"mask 0x0F", 0, 0x0F, false},
    {"force no-opaque (auto-confirm)", 2, 0xFFFFFFFFu, false},
    {"no-opaque + cull no-opaque", 2 | 128, 0xFFFFFFFFu, false},
    {"cull opaque", 64, 0xFFFFFFFFu, false},
    {"skip triangles", 256, 0xFFFFFFFFu, false},
};

// Compares batch.resBuf (8 words per ray: t, kind|front<<1, instance, geometry, primitive,
// u, v, custom index) with the brute-force reference for one mode.
void verifyMode(const Mode& md, const Scene& sc, const RayBatch& batch) {
  const uint32_t R = batch.count;
  const uint32_t* res = static_cast<const uint32_t*>(batch.resBuf.map);
  uint32_t hits = 0, misses = 0, borderline = 0;
  for (uint32_t i = 0; i < R; ++i) {
    const uint32_t* o = res + i * 8;
    bool gHit = (o[1] & 1) != 0;
    float gt; std::memcpy(&gt, &o[0], 4);
    Closest c = cpuClosest(batch.rays[i], sc.world, md.flags, md.mask);
    if (!gHit) {
      if (c.in.found) { std::fprintf(stderr, "[%s] ray %u: GPU miss, CPU hit t=%g inst=%u prim=%u\n", md.name, i, c.in.t, c.in.inst, c.in.prim); std::exit(1); }
      if (c.out.found) ++borderline; else ++misses;
      continue;
    }
    ++hits;
    if (!c.out.found) { std::fprintf(stderr, "[%s] ray %u: GPU hit t=%g, CPU miss\n", md.name, i, gt); std::exit(1); }
    uint32_t inst = o[2], prim = o[4];
    CHECK(inst < sc.nInst && sc.instActive[inst]);
    CHECK(o[7] == 1000 + inst);  // custom index
    const WTri* wt = nullptr;
    for (const WTri& t : sc.world) if (t.inst == inst && t.prim == prim) { wt = &t; break; }
    CHECK(wt);
    double tt, u, v, m;
    CHECK(intersect(batch.rays[i], *wt, tt, u, v, m));
    CHECK(tnear(tt, gt));
    CHECK(m > -1e-4);
    float gu, gv; std::memcpy(&gu, &o[5], 4); std::memcpy(&gv, &o[6], 4);
    CHECK(std::fabs(gu - u) < 1e-3 && std::fabs(gv - v) < 1e-3);
    CHECK(passesFilters(batch.rays[i], *wt, md.flags, md.mask));
    CHECK(((o[1] & 2u) != 0) == frontFacing(batch.rays[i], *wt));
    if (md.anyHit) continue;
    bool idMatch = (c.in.found && c.in.inst == inst && c.in.prim == prim) || (c.out.found && c.out.inst == inst && c.out.prim == prim);
    bool tMatch = (c.in.found && tnear(c.in.t, gt)) || (c.out.found && tnear(c.out.t, gt));
    if (!idMatch && !tMatch) {
      std::fprintf(stderr, "[%s] ray %u: GPU t=%g inst=%u prim=%u, CPU t=%g inst=%u prim=%u\n", md.name, i, gt, inst, prim, c.out.t, c.out.inst, c.out.prim);
      std::exit(1);
    }
    if (c.in.found && !tnear(c.in.t, gt)) ++borderline;
  }
  std::printf("  %-32s rays=%u hits=%-5u misses=%-5u borderline=%u  OK\n", md.name, R, hits, misses, borderline);
  CHECK(borderline <= R / 100);
  if (md.flags == 0 && md.mask == 0xFFFFFFFFu) CHECK(hits > R / 4);  // sanity: the scene is hit
}

}  // namespace
