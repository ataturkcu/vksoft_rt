// SPIR-V rewriter that makes shaders using SPV_KHR_ray_query runnable on drivers without
// it: OpRayQuery* instructions become calls into the vk_rq_* library (layer/shaders/
// rq_lib.comp) which is merged into the module, acceleration structure variables become
// storage buffer blocks holding the TLAS address.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vksoft {

enum class RewriteResult { NotNeeded, Rewritten, Failed };

// `validate` runs spirv-val (Vulkan 1.2 rules) on the result and fails if it is invalid.
RewriteResult rewriteRayQueries(const uint32_t* code, size_t words, std::vector<uint32_t>& out,
                                std::string& error, bool validate);

}  // namespace vksoft
