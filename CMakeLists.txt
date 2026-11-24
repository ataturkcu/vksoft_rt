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

# --- compute kernels: GLSL -> SPIR-V -> embedded C++ arrays ----------------------
find_program(GLSLC glslc REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(KERNELS init header fetch_tris morton radix_hist radix_scatter scan1 scan2 scan3 karras refit)
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
add_custom_command(
  OUTPUT ${KERNEL_GEN_DIR}/kernels_spv.h
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/embed_spv.py
          ${KERNEL_GEN_DIR}/kernels_spv.h ${KERNEL_EMBED_ARGS}
  DEPENDS ${KERNEL_SPVS} tools/embed_spv.py
  COMMENT "embedding SPIR-V kernels")

add_library(VkLayer_vksoft_rt SHARED layer/src/layer.cpp layer/src/accel_build.cpp
            ${KERNEL_GEN_DIR}/kernels_spv.h)
target_include_directories(VkLayer_vksoft_rt PRIVATE ${KERNEL_GEN_DIR})
target_link_libraries(VkLayer_vksoft_rt PRIVATE Vulkan::Headers)
target_compile_options(VkLayer_vksoft_rt PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
target_link_options(VkLayer_vksoft_rt PRIVATE -Wl,--no-undefined)

# Layer manifest next to the library so VK_ADD_LAYER_PATH=<build dir> works.
configure_file(layer/VkLayer_vksoft_rt.json.in
               ${CMAKE_BINARY_DIR}/VkLayer_vksoft_rt.json @ONLY)

# --- tests -------------------------------------------------------------------
enable_testing()

add_executable(test_bvh_sizes tests/test_bvh_sizes.cpp)
add_test(NAME bvh_sizes COMMAND test_bvh_sizes)

# GPU integration test: needs a Vulkan driver with acceleration structures.
find_library(VULKAN_LOADER vulkan)
if(VULKAN_LOADER)
  add_executable(as_lifecycle tests/as_lifecycle.cpp)
  target_link_libraries(as_lifecycle PRIVATE Vulkan::Headers ${VULKAN_LOADER})
  add_executable(blas_build tests/blas_build.cpp)
  target_link_libraries(blas_build PRIVATE Vulkan::Headers ${VULKAN_LOADER})
  add_test(NAME blas_build COMMAND blas_build)
  add_test(NAME as_lifecycle COMMAND as_lifecycle)
  set_tests_properties(as_lifecycle blas_build PROPERTIES
    ENVIRONMENT "VK_ADD_LAYER_PATH=${CMAKE_BINARY_DIR};VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt;VKSOFT_RT_LOG=1")
endif()
