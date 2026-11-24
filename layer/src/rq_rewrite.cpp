#include "rq_rewrite.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "kernels_spv.h"

namespace vksoft {
namespace {

// ---------------------------------------------------------------------------------------
// Minimal SPIR-V text IR. Modules are disassembled with raw numeric ids ("%123"), so every
// id reference is a token starting with '%' and no operand grammar tables are needed.
// ---------------------------------------------------------------------------------------

struct Inst {
  uint32_t result = 0;
  std::string op;
  std::vector<std::string> args;

  std::string str() const {
    std::string s;
    if (result) s = "%" + std::to_string(result) + " = ";
    s += op;
    for (auto& a : args) { s += ' '; s += a; }
    return s;
  }
};

bool isId(const std::string& t) { return t.size() > 1 && t[0] == '%'; }
uint32_t idNum(const std::string& t) { return static_cast<uint32_t>(std::stoul(t.substr(1))); }
std::string idStr(uint32_t id) { return "%" + std::to_string(id); }

std::vector<std::string> splitTokens(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i >= s.size()) break;
    size_t j = i;
    if (s[i] == '"') {
      j = i + 1;
      while (j < s.size() && !(s[j] == '"' && s[j - 1] != '\\')) ++j;
      ++j;
    } else {
      while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
    }
    out.push_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

Inst parseInst(const std::string& line) {
  auto t = splitTokens(line);
  Inst in;
  size_t k = 0;
  if (t.size() >= 3 && t[1] == "=" && isId(t[0])) { in.result = idNum(t[0]); k = 2; }
  in.op = t[k];
  in.args.assign(t.begin() + k + 1, t.end());
  return in;
}

struct Module {
  std::vector<Inst> caps, exts, imports, memModel, entries, execModes, annots, globals, funcs;
  uint32_t maxId = 0;
};

bool oneOf(const std::string& op, std::initializer_list<const char*> l) {
  for (auto* x : l) if (op == x) return true;
  return false;
}

Module parseModule(const std::string& text) {
  Module m;
  std::istringstream is(text);
  std::string line;
  bool inFuncs = false;
  while (std::getline(is, line)) {
    size_t b = line.find_first_not_of(" \t");
    if (b == std::string::npos || line[b] == ';') continue;
    Inst in = parseInst(line.substr(b));
    if (in.result) m.maxId = std::max(m.maxId, in.result);
    for (auto& a : in.args) if (isId(a)) m.maxId = std::max(m.maxId, idNum(a));
    if (in.op == "OpFunction") inFuncs = true;
    if (inFuncs) { m.funcs.push_back(in); continue; }
    if (in.op == "OpCapability") m.caps.push_back(in);
    else if (in.op == "OpExtension") m.exts.push_back(in);
    else if (in.op == "OpExtInstImport") m.imports.push_back(in);
    else if (in.op == "OpMemoryModel") m.memModel.push_back(in);
    else if (in.op == "OpEntryPoint") m.entries.push_back(in);
    else if (oneOf(in.op, {"OpExecutionMode", "OpExecutionModeId"})) m.execModes.push_back(in);
    else if (oneOf(in.op, {"OpString", "OpSource", "OpSourceExtension", "OpSourceContinued", "OpName",
                           "OpMemberName", "OpModuleProcessed", "OpLine", "OpNoLine"})) continue;  // debug info dropped
    else if (oneOf(in.op, {"OpDecorate", "OpMemberDecorate", "OpDecorationGroup", "OpGroupDecorate",
                           "OpGroupMemberDecorate", "OpDecorateId", "OpDecorateString", "OpMemberDecorateString"}))
      m.annots.push_back(in);
    else m.globals.push_back(in);
  }
  return m;
}

// Names of the library functions come from OpName, which parseModule drops: read them raw.
std::map<uint32_t, std::string> readNames(const std::string& text) {
  std::map<uint32_t, std::string> names;
  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) {
    size_t b = line.find_first_not_of(" \t");
    if (b == std::string::npos || line.compare(b, 6, "OpName") != 0) continue;
    auto t = splitTokens(line.substr(b));
    if (t.size() >= 3 && isId(t[1])) {
      std::string n = t[2];
      if (n.size() >= 2) n = n.substr(1, n.size() - 2);
      names[idNum(t[1])] = n;
    }
  }
  return names;
}

void substitute(Inst& in, const std::unordered_map<uint32_t, uint32_t>& sub) {
  for (auto& a : in.args) {
    if (!isId(a)) continue;
    auto it = sub.find(idNum(a));
    if (it != sub.end()) a = idStr(it->second);
  }
}

void offsetIds(Inst& in, uint32_t off) {
  if (in.result) in.result += off;
  for (auto& a : in.args) if (isId(a)) a = idStr(idNum(a) + off);
}

bool dedupable(const std::string& op) {
  return oneOf(op, {"OpTypeVoid", "OpTypeBool", "OpTypeInt", "OpTypeFloat", "OpTypeVector", "OpTypeMatrix",
                    "OpTypePointer", "OpTypeFunction", "OpConstant", "OpConstantTrue", "OpConstantFalse",
                    "OpConstantComposite", "OpConstantNull"});
}

std::string keyOf(const Inst& in) {
  std::string k = in.op;
  for (auto& a : in.args) { k += ' '; k += a; }
  return k;
}

bool hasRayQueryCapability(const uint32_t* code, size_t words) {
  if (words < 5 || code[0] != 0x07230203u) return false;
  size_t i = 5;
  while (i < words) {
    uint32_t w = code[i];
    uint32_t count = w >> 16, op = w & 0xFFFFu;
    if (count == 0 || i + count > words) return false;
    if (op == 17 /*OpCapability*/ && count >= 2 && code[i + 1] == 4472 /*RayQueryKHR*/) return true;
    if (op == 54 /*OpFunction*/) return false;  // capabilities precede functions
    i += count;
  }
  return false;
}

struct Rewriter {
  std::string error;
  Module app, lib;
  std::map<uint32_t, std::string> libNames;
  uint32_t next = 0;           // next free id
  uint32_t libOffset = 0;

  std::vector<Inst> libDecl;   // library-side declarations, in dependency order
  std::unordered_map<std::string, uint32_t> declKeys;
  std::unordered_map<uint32_t, uint32_t> sub;  // app id -> replacement id (deduplicated globals)

  uint32_t fresh() { return next++; }

  uint32_t intern(const std::string& op, std::vector<std::string> args) {
    Inst in;
    in.op = op;
    in.args = std::move(args);
    std::string k = keyOf(in);
    auto it = declKeys.find(k);
    if (it != declKeys.end()) return it->second;
    in.result = fresh();
    declKeys[k] = in.result;
    libDecl.push_back(in);
    return in.result;
  }

  bool fail(const std::string& msg) { error = msg; return false; }

  bool run(const uint32_t* code, size_t words, std::vector<uint32_t>& out, bool validate);
};

struct OpMap { const char* op; bool hasResult; };

bool Rewriter::run(const uint32_t* code, size_t words, std::vector<uint32_t>& out, bool validate) {
  const uint32_t appVersion = code[1];
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
  std::string appText, libText;
  const uint32_t opts = SPV_BINARY_TO_TEXT_OPTION_NO_HEADER;
  if (!tools.Disassemble(code, words, &appText, opts)) return fail("disassembling the application module failed");
  if (!tools.Disassemble(k_rq_lib_spv, sizeof(k_rq_lib_spv) / 4, &libText, opts))
    return fail("disassembling the ray query library failed");

  app = parseModule(appText);
  lib = parseModule(libText);
  libNames = readNames(libText);

  // ---- library: renumber, find the exported functions, compute the reachable closure --------
  libOffset = app.maxId + 1;
  auto offsetModule = [&](std::vector<Inst>& v) { for (auto& in : v) offsetIds(in, libOffset); };
  offsetModule(lib.caps); offsetModule(lib.exts); offsetModule(lib.imports); offsetModule(lib.memModel);
  offsetModule(lib.annots); offsetModule(lib.globals); offsetModule(lib.funcs);
  std::map<uint32_t, std::string> names;
  for (auto& [id, n] : libNames) names[id + libOffset] = n;
  next = std::max(app.maxId, lib.maxId + libOffset) + 1;

  std::unordered_map<uint32_t, size_t> libDef;  // global-level id -> index into lib.globals
  for (size_t i = 0; i < lib.globals.size(); ++i)
    if (lib.globals[i].result) libDef[lib.globals[i].result] = i;
  struct FuncRange { uint32_t id; size_t begin, end; };
  std::vector<FuncRange> libFuncs;
  std::unordered_map<uint32_t, size_t> libFuncIndex;
  for (size_t i = 0; i < lib.funcs.size(); ++i) {
    if (lib.funcs[i].op != "OpFunction") continue;
    size_t j = i;
    while (lib.funcs[j].op != "OpFunctionEnd") ++j;
    libFuncIndex[lib.funcs[i].result] = libFuncs.size();
    libFuncs.push_back({lib.funcs[i].result, i, j + 1});
  }
  std::map<std::string, uint32_t> exported;  // "vk_rq_proceed" -> function id
  for (auto& f : libFuncs) {
    auto it = names.find(f.id);
    if (it == names.end() || it->second.rfind("vk_rq_", 0) != 0) continue;
    std::string n = it->second.substr(0, it->second.find('('));
    exported[n] = f.id;
  }
  if (exported.empty()) return fail("ray query library has no exported functions");

  std::unordered_set<uint32_t> keepGlobal, keepFunc;
  std::vector<uint32_t> work;
  for (auto& [n, id] : exported) work.push_back(id);
  uint32_t slotsConstId = 0;
  for (auto& [id, n] : names) if (n == "RQ_SLOTS") slotsConstId = id;
  while (!work.empty()) {
    uint32_t id = work.back();
    work.pop_back();
    auto visitArgs = [&](const Inst& in) {
      for (auto& a : in.args) if (isId(a)) work.push_back(idNum(a));
    };
    auto fi = libFuncIndex.find(id);
    if (fi != libFuncIndex.end()) {
      if (!keepFunc.insert(id).second) continue;
      for (size_t i = libFuncs[fi->second].begin; i < libFuncs[fi->second].end; ++i) visitArgs(lib.funcs[i]);
      continue;
    }
    auto gi = libDef.find(id);
    if (gi == libDef.end()) continue;  // local id, label or import
    if (!keepGlobal.insert(id).second) continue;
    visitArgs(lib.globals[gi->second]);
  }

  // ExtInstImport: reuse the application's import of the same set where there is one.
  std::unordered_map<uint32_t, uint32_t> libSub;
  std::vector<Inst> extraImports;
  for (auto& li : lib.imports) {
    bool found = false;
    for (auto& ai : app.imports)
      if (ai.args == li.args) { libSub[li.result] = ai.result; found = true; }
    if (!found) extraImports.push_back(li);
  }

  // ---- application: find ray query / acceleration structure types, variables, slots --------
  uint32_t rqType = 0, asType = 0;
  for (auto& g : app.globals) {
    if (g.op == "OpTypeRayQueryKHR") rqType = g.result;
    if (g.op == "OpTypeAccelerationStructureKHR") asType = g.result;
  }
  std::unordered_set<uint32_t> removedTypes, rqVars, asVars;
  if (rqType) removedTypes.insert(rqType);
  if (asType) removedTypes.insert(asType);
  std::map<uint32_t, std::string> asArrayLen;  // array-of-AS type id -> length operand
  for (auto& g : app.globals) {
    if (g.op == "OpTypeArray" && g.args.size() >= 2 && asType && g.args[0] == idStr(asType)) {
      removedTypes.insert(g.result);
      asArrayLen[g.result] = g.args[1];
    }
    if (g.op == "OpTypePointer" && g.args.size() >= 2 && isId(g.args[1]) && removedTypes.count(idNum(g.args[1])))
      removedTypes.insert(g.result);
  }
  for (auto& g : app.globals) {
    if (g.op == "OpTypeRayQueryKHR" || g.op == "OpTypeAccelerationStructureKHR") continue;
    if (asArrayLen.count(g.result)) continue;
    if (g.op == "OpTypePointer" && removedTypes.count(g.result)) continue;
    if (g.op == "OpVariable" && !g.args.empty() && removedTypes.count(idNum(g.args[0]))) continue;
    for (auto& a : g.args)
      if (isId(a) && removedTypes.count(idNum(a)))
        return fail("unsupported use of ray query / acceleration structure type in " + g.op);
  }
  std::map<uint32_t, uint32_t> slotOf;  // rq variable id -> slot
  std::map<uint32_t, uint32_t> asArrayVars;  // AS array variable -> its array type
  auto scanVars = [&](const std::vector<Inst>& v) {
    for (auto& in : v) {
      if (in.op != "OpVariable" || in.args.empty()) continue;
      uint32_t ty = idNum(in.args[0]);
      if (!removedTypes.count(ty)) continue;
      // pointee decides: find the pointer type definition
      for (auto& g : app.globals)
        if (g.result == ty) {
          uint32_t pointee = idNum(g.args[1]);
          if (pointee == rqType) { rqVars.insert(in.result); if (!slotOf.count(in.result)) { uint32_t s = uint32_t(slotOf.size()); slotOf[in.result] = s; } }
          if (pointee == asType) asVars.insert(in.result);
          if (asArrayLen.count(pointee)) { asArrayVars[in.result] = pointee; }
        }
    }
  };
  scanVars(app.globals);
  scanVars(app.funcs);
  const uint32_t slotCount = std::max<uint32_t>(1, uint32_t(slotOf.size()));

  // ---- library declarations (reachable globals, in order), RQ_SLOTS patched ----------------
  std::vector<Inst> libFuncInsts;
  for (auto& g0 : lib.globals) {
    if (!g0.result || !keepGlobal.count(g0.result)) continue;
    Inst g = g0;
    substitute(g, libSub);
    if (g.result == slotsConstId) {  // spec constant -> plain constant with the real slot count
      g.op = "OpConstant";
      g.args = {g.args[0], std::to_string(slotCount)};
    }
    if (dedupable(g.op)) declKeys[keyOf(g)] = g.result;
    libDecl.push_back(g);
  }
  std::vector<uint32_t> libGlobalVars;  // module-scope variables needing entry point interface entries
  for (auto& g : libDecl)
    if (g.op == "OpVariable" && g.args.size() >= 2 && g.args[1] != "Function") libGlobalVars.push_back(g.result);

  // ---- application globals: remove ray query types, deduplicate against library ------------
  std::vector<Inst> appGlobals;
  for (auto& g0 : app.globals) {
    if (g0.op == "OpTypeRayQueryKHR" || g0.op == "OpTypeAccelerationStructureKHR") continue;
    if (asArrayLen.count(g0.result)) continue;
    if (g0.op == "OpTypePointer" && removedTypes.count(g0.result)) continue;
    if (g0.op == "OpVariable" && !g0.args.empty() && removedTypes.count(idNum(g0.args[0]))) continue;
    Inst g = g0;
    substitute(g, sub);
    if (dedupable(g.op)) {
      std::string k = keyOf(g);
      auto it = declKeys.find(k);
      if (it != declKeys.end()) { sub[g.result] = it->second; continue; }
      declKeys[k] = g.result;
    }
    appGlobals.push_back(g);
  }

  // ---- acceleration structure variables -> storage buffer block {uint lo; uint hi;} ---------
  uint32_t uintT = intern("OpTypeInt", {"32", "0"});
  uint32_t intT = intern("OpTypeInt", {"32", "1"});
  uint32_t ulongT = intern("OpTypeInt", {"64", "0"});
  uint32_t v2uintT = intern("OpTypeVector", {idStr(uintT), "2"});
  uint32_t c0 = intern("OpConstant", {idStr(uintT), "0"});
  uint32_t c1 = intern("OpConstant", {idStr(uintT), "1"});
  uint32_t asBlock = 0, ptrSBBlock = 0, ptrSBUint = 0;
  std::vector<Inst> newAnnots;
  if (!asVars.empty() || !asArrayVars.empty()) {
    asBlock = fresh();
    libDecl.push_back({asBlock, "OpTypeStruct", {idStr(uintT), idStr(uintT)}});
    newAnnots.push_back({0, "OpDecorate", {idStr(asBlock), "Block"}});
    newAnnots.push_back({0, "OpMemberDecorate", {idStr(asBlock), "0", "Offset", "0"}});
    newAnnots.push_back({0, "OpMemberDecorate", {idStr(asBlock), "1", "Offset", "4"}});
    newAnnots.push_back({0, "OpMemberDecorate", {idStr(asBlock), "0", "NonWritable"}});
    newAnnots.push_back({0, "OpMemberDecorate", {idStr(asBlock), "1", "NonWritable"}});
    ptrSBBlock = intern("OpTypePointer", {"StorageBuffer", idStr(asBlock)});
    ptrSBUint = intern("OpTypePointer", {"StorageBuffer", idStr(uintT)});
    for (uint32_t v : asVars) appGlobals.push_back({v, "OpVariable", {idStr(ptrSBBlock), "StorageBuffer"}});
    // arrays: declared after the application's constants (the length is one of them)
    std::map<std::string, uint32_t> arrayPtrByLen;
    for (auto& [var, arrType] : asArrayVars) {
      std::string len = asArrayLen[arrType];
      if (isId(len)) { auto it = sub.find(idNum(len)); if (it != sub.end()) len = idStr(it->second); }
      auto pit = arrayPtrByLen.find(len);
      if (pit == arrayPtrByLen.end()) {
        uint32_t arr = fresh(), ptr = fresh();
        appGlobals.push_back({arr, "OpTypeArray", {idStr(asBlock), len}});
        appGlobals.push_back({ptr, "OpTypePointer", {"StorageBuffer", idStr(arr)}});
        pit = arrayPtrByLen.emplace(len, ptr).first;
      }
      appGlobals.push_back({var, "OpVariable", {idStr(pit->second), "StorageBuffer"}});
    }
  }

  // ---- library functions: copied verbatim (renumbered) ------------------------------------
  for (auto& f : libFuncs) {
    if (!keepFunc.count(f.id)) continue;
    for (size_t i = f.begin; i < f.end; ++i) {
      Inst in = lib.funcs[i];
      substitute(in, libSub);
      if (in.op == "OpLine" || in.op == "OpNoLine") continue;
      libFuncInsts.push_back(in);
    }
  }
  // return type of each exported function (for result adaptation)
  std::map<std::string, uint32_t> libRet;
  for (auto& [n, id] : exported) {
    const Inst& f = lib.funcs[libFuncs[libFuncIndex[id]].begin];
    libRet[n] = idNum(f.args[0]);
  }

  // glslang passes every parameter by pointer: remember the parameter types per function
  std::map<std::string, std::vector<std::string>> libParams;
  for (auto& [n, id] : exported) {
    size_t i = libFuncs[libFuncIndex[id]].begin + 1;
    while (lib.funcs[i].op == "OpFunctionParameter") { libParams[n].push_back(lib.funcs[i].args[0]); ++i; }
  }

  // ---- application functions: rewrite ray query instructions -------------------------------
  auto signedInt32 = [&](uint32_t typeId) -> int {  // 1 signed, 0 unsigned, -1 not a 32-bit int
    for (auto& g : appGlobals)
      if (g.result == typeId && g.op == "OpTypeInt" && g.args.size() >= 2 && g.args[0] == "32")
        return g.args[1] == "1" ? 1 : 0;
    return -1;
  };
  std::map<uint32_t, uint32_t> constType;  // constant id -> type id (app + generated)
  for (auto& g : appGlobals) if (g.op == "OpConstant" && g.result) constType[g.result] = idNum(g.args[0]);
  for (auto& g : libDecl) if (g.op == "OpConstant" && g.result) constType[g.result] = idNum(g.args[0]);

  std::vector<Inst> appFuncs;
  std::map<std::string, std::string> opToFn;
  auto fnFor = [&](const std::string& op) -> std::string {
    std::string n = op.substr(std::strlen("OpRayQuery"));
    if (n.size() > 3 && n.compare(n.size() - 3, 3, "KHR") == 0) n.resize(n.size() - 3);
    n[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(n[0])));
    return "vk_rq_" + n;
  };
  std::unordered_set<uint32_t> asPtrs;  // access chain results pointing at acceleration structure address blocks
  std::vector<Inst> hoisted;  // temporaries for by-pointer arguments, placed at the top of the function
  size_t labelPos = 0;
  bool seenLabel = false;
  for (auto& f0 : app.funcs) {
    Inst in = f0;
    substitute(in, sub);
    if (in.op == "OpLine" || in.op == "OpNoLine") continue;
    if (in.op == "OpFunction") { hoisted.clear(); seenLabel = false; }
    if (in.op == "OpFunctionEnd") {
      appFuncs.insert(appFuncs.begin() + long(labelPos), hoisted.begin(), hoisted.end());
      appFuncs.push_back(in);
      hoisted.clear();
      continue;
    }
    if (in.op == "OpLabel" && !seenLabel) {
      appFuncs.push_back(in);
      labelPos = appFuncs.size();
      seenLabel = true;
      continue;
    }
    if (in.op == "OpVariable" && !in.args.empty() && removedTypes.count(idNum(in.args[0]))) continue;
    if ((in.op == "OpAccessChain" || in.op == "OpInBoundsAccessChain") && in.args.size() >= 3 && isId(in.args[1]) &&
        asArrayVars.count(idNum(in.args[1])) && removedTypes.count(idNum(in.args[0]))) {
      in.args[0] = idStr(ptrSBBlock);  // pointer to one acceleration structure -> pointer to its address block
      asPtrs.insert(in.result);
      appFuncs.push_back(in);
      continue;
    }
    if (in.op == "OpLoad" && in.args.size() >= 2 && asType && isId(in.args[1]) &&
        (asVars.count(idNum(in.args[1])) || asPtrs.count(idNum(in.args[1])))) {
      uint32_t p0 = fresh(), p1 = fresh(), lo = fresh(), hi = fresh(), pair = fresh();
      appFuncs.push_back({p0, "OpAccessChain", {idStr(ptrSBUint), in.args[1], idStr(c0)}});
      appFuncs.push_back({p1, "OpAccessChain", {idStr(ptrSBUint), in.args[1], idStr(c1)}});
      appFuncs.push_back({lo, "OpLoad", {idStr(uintT), idStr(p0)}});
      appFuncs.push_back({hi, "OpLoad", {idStr(uintT), idStr(p1)}});
      appFuncs.push_back({pair, "OpCompositeConstruct", {idStr(v2uintT), idStr(lo), idStr(hi)}});
      appFuncs.push_back({in.result, "OpBitcast", {idStr(ulongT), idStr(pair)}});
      continue;
    }
    if (in.op.rfind("OpRayQuery", 0) == 0) {
      std::string fn = fnFor(in.op);
      auto fit = exported.find(fn);
      if (fit == exported.end()) return fail("unsupported ray query instruction " + in.op);
      const bool hasResult = in.result != 0;
      size_t first = hasResult ? 1 : 0;  // index of the ray query operand
      if (in.args.size() <= first) return fail("malformed " + in.op);
      uint32_t rqVar = idNum(in.args[first]);
      auto sit = slotOf.find(rqVar);
      if (sit == slotOf.end()) return fail("ray query used through an unsupported pointer in " + in.op);
      uint32_t slotConst = intern("OpConstant", {idStr(uintT), std::to_string(sit->second)});
      std::vector<std::string> callArgs{idStr(fit->second), idStr(slotConst)};
      for (size_t i = first + 1; i < in.args.size(); ++i) {
        std::string a = in.args[i];
        const bool isIntersection = fn.rfind("vk_rq_getIntersection", 0) == 0 && fn != "vk_rq_getIntersectionCandidateAABBOpaque" &&
                                    i == first + 1;
        if (isIntersection) {
          int s = -1;
          auto ct = constType.find(idNum(a));
          if (ct != constType.end()) s = signedInt32(ct->second);
          if (s == 0) {  // library takes a signed int
            uint32_t cast = fresh();
            appFuncs.push_back({cast, "OpBitcast", {idStr(intT), a}});
            a = idStr(cast);
          }
        }
        callArgs.push_back(a);
      }
      // by-pointer arguments: temporary variable + store for every value
      const auto& params = libParams[fn];
      if (params.size() != callArgs.size() - 1) return fail("argument count mismatch for " + in.op);
      std::vector<std::string> ptrArgs{callArgs[0]};
      for (size_t i = 0; i < params.size(); ++i) {
        uint32_t tmpVar = fresh();
        hoisted.push_back({tmpVar, "OpVariable", {params[i], "Function"}});
        appFuncs.push_back({0, "OpStore", {idStr(tmpVar), callArgs[i + 1]}});
        ptrArgs.push_back(idStr(tmpVar));
      }
      uint32_t ret = libRet[fn];
      if (!hasResult) {
        std::vector<std::string> a2{idStr(ret)};
        for (auto& x : ptrArgs) a2.push_back(x);
        appFuncs.push_back({fresh(), "OpFunctionCall", a2});
        continue;
      }
      uint32_t appRet = idNum(in.args[0]);
      std::vector<std::string> a2{idStr(ret)};
      for (auto& x : ptrArgs) a2.push_back(x);
      if (appRet == ret) {
        appFuncs.push_back({in.result, "OpFunctionCall", a2});
      } else {
        uint32_t tmp = fresh();
        appFuncs.push_back({tmp, "OpFunctionCall", a2});
        appFuncs.push_back({in.result, "OpBitcast", {idStr(appRet), idStr(tmp)}});
      }
      continue;
    }
    appFuncs.push_back(in);
  }
  // OpTypeVoid interned above may be used before the declaration list is emitted: fine,
  // libDecl is emitted whole before any function.

  // ---- assemble the output module -----------------------------------------------------------
  auto addUnique = [](std::vector<Inst>& dst, const Inst& in) {
    for (auto& d : dst) if (d.op == in.op && d.args == in.args) return;
    dst.push_back(in);
  };
  std::vector<Inst> caps, exts, imports, entries, execModes, annots, globals;
  for (auto& c : app.caps) if (!(c.args.size() == 1 && c.args[0] == "RayQueryKHR")) addUnique(caps, c);
  for (auto& c : lib.caps) addUnique(caps, c);
  addUnique(caps, {0, "OpCapability", {"Int64"}});
  addUnique(caps, {0, "OpCapability", {"PhysicalStorageBufferAddresses"}});
  for (auto& e : app.exts) if (!(e.args.size() == 1 && e.args[0] == "\"SPV_KHR_ray_query\"")) addUnique(exts, e);
  for (auto& e : lib.exts) addUnique(exts, e);
  addUnique(exts, {0, "OpExtension", {"\"SPV_KHR_physical_storage_buffer\""}});
  imports = app.imports;
  for (auto& i : extraImports) imports.push_back(i);

  Inst memModel = app.memModel.empty() ? Inst{0, "OpMemoryModel", {"Logical", "GLSL450"}} : app.memModel[0];
  memModel.args[0] = "PhysicalStorageBuffer64";

  std::unordered_set<uint32_t> goneIds;  // ids that no longer exist
  for (uint32_t v : rqVars) goneIds.insert(v);
  for (auto& [k, v] : sub) goneIds.insert(k);
  for (auto e : app.entries) {
    substitute(e, sub);
    std::vector<std::string> args;
    for (size_t i = 0; i < e.args.size(); ++i) {
      if (i >= 3 && isId(e.args[i]) && rqVars.count(idNum(e.args[i]))) continue;
      args.push_back(e.args[i]);
    }
    for (uint32_t v : libGlobalVars)
      if (std::find(args.begin(), args.end(), idStr(v)) == args.end()) args.push_back(idStr(v));
    e.args = args;
    entries.push_back(e);
  }
  for (auto e : app.execModes) { substitute(e, sub); execModes.push_back(e); }
  for (auto a : app.annots) {
    if (!a.args.empty() && isId(a.args[0]) && goneIds.count(idNum(a.args[0]))) continue;  // decoration of a removed id
    substitute(a, sub);
    annots.push_back(a);
  }
  for (auto& a : lib.annots) {
    if (a.args.empty() || !isId(a.args[0]) || !keepGlobal.count(idNum(a.args[0]))) continue;
    if (a.args.size() >= 2 && a.args[1] == "SpecId") continue;
    Inst x = a;
    substitute(x, libSub);
    annots.push_back(x);
  }
  for (auto& a : newAnnots) annots.push_back(a);
  for (auto& d : libDecl) globals.push_back(d);
  for (auto& g : appGlobals) globals.push_back(g);

  std::ostringstream os;
  auto emit = [&](const std::vector<Inst>& v) { for (auto& in : v) os << in.str() << '\n'; };
  emit(caps); emit(exts); emit(imports);
  os << memModel.str() << '\n';
  emit(entries); emit(execModes); emit(annots); emit(globals); emit(libFuncInsts); emit(appFuncs);
  std::string text = os.str();

  // anything left that still points at removed ids (other than re-declared AS variables)?
  for (auto& in : appFuncs)
    for (auto& a : in.args)
      if (isId(a) && (rqVars.count(idNum(a)) || (removedTypes.count(idNum(a)))))
        return fail("leftover reference to a ray query object in " + in.op);

  std::vector<uint32_t> bin;
  if (!tools.Assemble(text, &bin, SPV_TEXT_TO_BINARY_OPTION_NONE)) return fail("assembling the rewritten module failed");
  const uint32_t version = std::max<uint32_t>(appVersion, 0x00010500u);
  bin[1] = version;
  // The library is compiled -O0 (see CMake), so the merged module is unoptimised: inline the
  // library functions, promote the by-pointer temporaries and private state to registers.
  if (!std::getenv("VKSOFT_RT_NO_OPT")) {
    spvtools::Optimizer opt(version >= 0x00010600u ? SPV_ENV_VULKAN_1_3 : SPV_ENV_VULKAN_1_2);
    std::string optMsg;
    opt.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
      if (optMsg.empty()) optMsg = m;
    });
    opt.RegisterPerformancePasses();
    std::vector<uint32_t> optimized;
    if (opt.Run(bin.data(), bin.size(), &optimized)) bin = std::move(optimized);
    else if (std::getenv("VKSOFT_RT_LOG")) std::fprintf(stderr, "[vksoft_rt] optimizer failed (%s), keeping the unoptimised module\n", optMsg.c_str());
  }
  if (validate) {
    spvtools::SpirvTools vt(version >= 0x00010600u ? SPV_ENV_VULKAN_1_3 : SPV_ENV_VULKAN_1_2);
    std::string msg;
    vt.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& p, const char* m) {
      if (msg.empty()) msg = std::string(m) + " (instruction " + std::to_string(p.index) + ")";
    });
    if (!vt.Validate(bin)) {
      error = "rewritten module failed validation: " + msg;
      return false;
    }
  }
  out = std::move(bin);
  return true;
}

}  // namespace

RewriteResult rewriteRayQueries(const uint32_t* code, size_t words, std::vector<uint32_t>& out, std::string& error,
                                bool validate) {
  if (!hasRayQueryCapability(code, words)) return RewriteResult::NotNeeded;
  Rewriter rw;
  if (!rw.run(code, words, out, validate)) {
    error = rw.error;
    return RewriteResult::Failed;
  }
  return RewriteResult::Rewritten;
}

}  // namespace vksoft
