cmake_minimum_required(VERSION 3.25)
project(render C)

set(CMAKE_C_STANDARD 99)

option(USE_SDL3_FROM_GIT "Fetch SDL3 from git instead of using system pkg-config" OFF)

if(NOT USE_SDL3_FROM_GIT)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)
    message(STATUS "Using SDL3 from pkg-config: ${SDL3_VERSION}")
    # Use PkgConfig::SDL3 for IMPORTED_TARGET
    set(SDL3_LIB PkgConfig::SDL3)
else()
    include(FetchContent)
    FetchContent_Declare(
      SDL3
      GIT_REPOSITORY "https://github.com/libsdl-org/SDL.git"
      GIT_TAG "main"
      EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(SDL3)
    set(SDL3_LIB SDL3::SDL3)
    message(STATUS "Using SDL3 from git repository.")
endif()

# Create your app executable target as usual
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.c)
add_executable(render ${SOURCES})

target_link_libraries(render PRIVATE m ${SDL3_LIB})
