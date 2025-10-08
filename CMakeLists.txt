cmake_minimum_required(VERSION 3.25)
project(render C)

set(CMAKE_C_STANDARD 99)

# FetchContent downloads and configures dependencies
include(FetchContent)
FetchContent_Declare(
  SDL3
  GIT_REPOSITORY "https://github.com/libsdl-org/SDL.git"
  GIT_TAG "main"
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(SDL3)

include_directories(${SDL3_INCLUDE_DIRS})

# Create your app executable target as usual
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.c)
add_executable(render ${SOURCES})

target_link_libraries(render PRIVATE m SDL3::SDL3)
