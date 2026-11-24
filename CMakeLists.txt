cmake_minimum_required(VERSION 3.24)
project(vksoft_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  vulkan_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG        v1.4.369
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(vulkan_headers)

# The layer must also load inside the Steam Linux Runtime container (Proton), which has neither
# the host's SPIRV-Tools nor a compatible libstdc++: bundle both statically by default.
option(VKSOFT_BUNDLE_SPIRV_TOOLS "Build SPIRV-Tools into the layer and link the C++ runtime statically" ON)
if(VKSOFT_BUNDLE_SPIRV_TOOLS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
  set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
  set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
  set(SPIRV_BUILD_FUZZER OFF CACHE BOOL "" FORCE)
  set(SPIRV_TOOLS_BUILD_STATIC ON CACHE BOOL "" FORCE)
  set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
  FetchContent_Declare(spirv_headers GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
                       GIT_TAG vulkan-sdk-1.4.363.0 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(spirv_headers)
  FetchContent_Declare(spirv_tools GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
                       GIT_TAG v2026.4 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(spirv_tools)
  set(SPIRV_TOOLS SPIRV-Tools-static)
else()
  find_library(SPIRV_TOOLS SPIRV-Tools REQUIRED)
endif()

# --- compute kernels: GLSL -> SPIR-V -> embedded C++ arrays ----------------------
find_program(GLSLC glslc REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(KERNELS init header fetch_tris fetch_aabbs fetch_instances morton radix_hist radix_scatter scan1 scan2 scan3 karras refit)
set(LIBRARY_KERNELS rq_lib)  # compiled -O0: glslang/spirv-opt would drop what the rewriter copies
set(KERNEL_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${KERNEL_GEN_DIR})
file(GLOB KERNEL_INCLUDES CONFIGURE_DEPENDS layer/shaders/*.glsl)
set(KERNEL_SPVS)
set(KERNEL_EMBED_ARGS)
foreach(k ${KERNELS})
  add_custom_command(
    OUTPUT ${KERNEL_GEN_DIR}/${k}.spv
    COMMAND ${GLSLC} --target-env=vulkan1.2 -O -I${CMAKE_SOURCE_DIR}/layer/shaders
            -o ${KERNEL_GEN_DIR}/${k}.spv ${CMAKE_SOURCE_DIR}/layer/shaders/${k}.comp
    DEPENDS layer/shaders/${k}.comp ${KERNEL_INCLUDES}
    COMMENT "glslc ${k}.comp")
  list(APPEND KERNEL_SPVS ${KERNEL_GEN_DIR}/${k}.spv)
  list(APPEND KERNEL_EMBED_ARGS ${k}=${KERNEL_GEN_DIR}/${k}.spv)
endforeach()
foreach(k ${LIBRARY_KERNELS})
  add_custom_command(
    OUTPUT ${KERNEL_GEN_DIR}/${k}.spv
    COMMAND ${GLSLC} --target-env=vulkan1.2 -O0 -I${CMAKE_SOURCE_DIR}/layer/shaders
            -o ${KERNEL_GEN_DIR}/${k}.spv ${CMAKE_SOURCE_DIR}/layer/shaders/${k}.comp
    DEPENDS layer/shaders/${k}.comp ${KERNEL_INCLUDES}
    COMMENT "glslc ${k}.comp (library, -O0)")
  list(APPEND KERNEL_SPVS ${KERNEL_GEN_DIR}/${k}.spv)
  list(APPEND KERNEL_EMBED_ARGS ${k}=${KERNEL_GEN_DIR}/${k}.spv)
endforeach()
add_custom_command(
  OUTPUT ${KERNEL_GEN_DIR}/kernels_spv.h
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/embed_spv.py
          ${KERNEL_GEN_DIR}/kernels_spv.h ${KERNEL_EMBED_ARGS}
  DEPENDS ${KERNEL_SPVS} tools/embed_spv.py
  COMMENT "embedding SPIR-V kernels")

add_library(VkLayer_vksoft_rt SHARED layer/src/layer.cpp layer/src/accel_build.cpp layer/src/rq_rewrite.cpp
            layer/src/descriptor_cells.cpp ${KERNEL_GEN_DIR}/kernels_spv.h)
target_include_directories(VkLayer_vksoft_rt PRIVATE ${KERNEL_GEN_DIR})
target_link_libraries(VkLayer_vksoft_rt PRIVATE Vulkan::Headers ${SPIRV_TOOLS})
target_compile_options(VkLayer_vksoft_rt PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
target_link_options(VkLayer_vksoft_rt PRIVATE -Wl,--no-undefined)
if(VKSOFT_BUNDLE_SPIRV_TOOLS)
  target_link_options(VkLayer_vksoft_rt PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Layer manifest next to the library so VK_ADD_LAYER_PATH=<build dir> works.
configure_file(layer/VkLayer_vksoft_rt.json.in
               ${CMAKE_BINARY_DIR}/VkLayer_vksoft_rt.json @ONLY)

# --- tests -------------------------------------------------------------------
enable_testing()

add_executable(test_bvh_sizes tests/test_bvh_sizes.cpp)
add_test(NAME bvh_sizes COMMAND test_bvh_sizes)

# --- ray query rewriter test (host only) ----------------------------------------
set(RQ_TEST_SPVS)
foreach(spec rq_test:-O0 rq_test_opt:-O rq_two:-O0 rq_two_opt:-O rq_param:-O0 plain:-O0 rq_aabb:-O rq_array0:-O rq_array1:-O)
  string(REPLACE ":" ";" parts ${spec})
  list(GET parts 0 sname)
  list(GET parts 1 sopt)
  string(REGEX REPLACE "_opt$" "" ssrc ${sname})
  set(sdefs)
  if(sname MATCHES "^rq_array([01])$")
    set(ssrc rq_array)
    set(sdefs -DTLAS_INDEX=${CMAKE_MATCH_1})
  endif()
  add_custom_command(
    OUTPUT ${KERNEL_GEN_DIR}/${sname}.spv
    COMMAND ${GLSLC} --target-env=vulkan1.2 ${sopt} ${sdefs} -o ${KERNEL_GEN_DIR}/${sname}.spv
            ${CMAKE_SOURCE_DIR}/tests/shaders/${ssrc}.comp
    DEPENDS tests/shaders/${ssrc}.comp
    COMMENT "glslc ${ssrc}.comp -> ${sname}.spv")
  list(APPEND RQ_TEST_SPVS ${KERNEL_GEN_DIR}/${sname}.spv)
endforeach()
add_executable(test_rq_rewrite tests/test_rq_rewrite.cpp layer/src/rq_rewrite.cpp ${KERNEL_GEN_DIR}/kernels_spv.h ${RQ_TEST_SPVS})
target_include_directories(test_rq_rewrite PRIVATE ${KERNEL_GEN_DIR})
target_compile_definitions(test_rq_rewrite PRIVATE TEST_SPV_DIR="${KERNEL_GEN_DIR}")
target_link_libraries(test_rq_rewrite PRIVATE ${SPIRV_TOOLS})
add_test(NAME rq_rewrite COMMAND test_rq_rewrite)

# GPU integration test: needs a Vulkan driver with acceleration structures.
find_library(VULKAN_LOADER vulkan)
if(VULKAN_LOADER)
  add_executable(as_lifecycle tests/as_lifecycle.cpp)
  target_link_libraries(as_lifecycle PRIVATE Vulkan::Headers ${VULKAN_LOADER})
  add_executable(blas_build tests/blas_build.cpp)
  target_link_libraries(blas_build PRIVATE Vulkan::Headers ${VULKAN_LOADER})
  add_test(NAME blas_build COMMAND blas_build)
  # traversal test shader (uses the layer's traversal code through the same include dir)
  add_custom_command(
    OUTPUT ${KERNEL_GEN_DIR}/trace_test.spv
    COMMAND ${GLSLC} --target-env=vulkan1.2 -O -I${CMAKE_SOURCE_DIR}/layer/shaders
            -o ${KERNEL_GEN_DIR}/trace_test.spv ${CMAKE_SOURCE_DIR}/tests/shaders/trace_test.comp
    DEPENDS tests/shaders/trace_test.comp ${KERNEL_INCLUDES}
    COMMENT "glslc trace_test.comp")
  add_executable(trace_test tests/trace_test.cpp ${KERNEL_GEN_DIR}/trace_test.spv)
  target_compile_definitions(trace_test PRIVATE TRACE_TEST_SPV="${KERNEL_GEN_DIR}/trace_test.spv")
  target_link_libraries(trace_test PRIVATE Vulkan::Headers ${VULKAN_LOADER})
  add_test(NAME trace_test COMMAND trace_test)
  add_executable(rq_test tests/rq_test.cpp ${RQ_TEST_SPVS})
  target_compile_definitions(rq_test PRIVATE TEST_SPV_DIR="${KERNEL_GEN_DIR}")
  target_link_libraries(rq_test PRIVATE Vulkan::Headers ${VULKAN_LOADER})
  add_test(NAME rq_test COMMAND rq_test)
  add_executable(aabb_test tests/aabb_test.cpp ${RQ_TEST_SPVS})
  target_compile_definitions(aabb_test PRIVATE TEST_SPV_DIR="${KERNEL_GEN_DIR}")
  target_link_libraries(aabb_test PRIVATE Vulkan::Headers ${VULKAN_LOADER})
  add_test(NAME aabb_test COMMAND aabb_test)
  add_executable(tlas_build tests/tlas_build.cpp)
  target_link_libraries(tlas_build PRIVATE Vulkan::Headers ${VULKAN_LOADER})
  add_test(NAME tlas_build COMMAND tlas_build)
  add_test(NAME as_lifecycle COMMAND as_lifecycle)
  set_tests_properties(as_lifecycle blas_build tlas_build trace_test rq_test aabb_test PROPERTIES
    ENVIRONMENT "VK_ADD_LAYER_PATH=${CMAKE_BINARY_DIR};VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt;VKSOFT_RT_LOG=1")
endif()
