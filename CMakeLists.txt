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

add_library(VkLayer_vksoft_rt SHARED layer/src/layer.cpp)
target_link_libraries(VkLayer_vksoft_rt PRIVATE Vulkan::Headers)
target_compile_options(VkLayer_vksoft_rt PRIVATE -Wall -Wextra -Wpedantic)
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
  add_test(NAME as_lifecycle COMMAND as_lifecycle)
  set_tests_properties(as_lifecycle PROPERTIES
    ENVIRONMENT "VK_ADD_LAYER_PATH=${CMAKE_BINARY_DIR};VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKSOFT_rt;VKSOFT_RT_LOG=1")
endif()
