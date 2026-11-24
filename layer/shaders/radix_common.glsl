// Stable 8-bit-digit LSD radix sort building blocks. Workgroup = 256 threads, one
// element per thread; assumes subgroup size 32 (8 subgroups per workgroup).
shared uint sgCount[8][256];
shared uint blockTotal[256];

// Every thread of the workgroup must call this (it contains barriers). Returns the
// rank of the calling element among equal digits inside its own subgroup. Afterwards
// sgCount[sg][d] is the exclusive count of digit d in earlier subgroups and
// blockTotal[d] the number of elements with digit d in the whole workgroup.
uint radixPrepare(uint d, bool valid) {
  uint t = gl_LocalInvocationID.x;
  for (uint s = 0; s < 8; ++s) sgCount[s][t] = 0u;
  barrier();

  uvec4 act = subgroupBallot(valid);
  uint match = act.x;
  for (uint b = 0; b < 8; ++b) {
    bool bit = ((d >> b) & 1u) != 0u;
    uint bm = subgroupBallot(bit).x;
    match &= bit ? bm : ~bm;
  }
  uint lane = gl_SubgroupInvocationID;
  uint rank = bitCount(match & ((1u << lane) - 1u));
  if (valid && rank == 0u) sgCount[gl_SubgroupID][d] = bitCount(match);
  barrier();

  uint running = 0u;
  for (uint s = 0; s < 8; ++s) {
    uint c = sgCount[s][t];
    sgCount[s][t] = running;
    running += c;
  }
  blockTotal[t] = running;
  barrier();
  return rank;
}
