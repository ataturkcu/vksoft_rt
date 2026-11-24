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
