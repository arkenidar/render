cmake_minimum_required(VERSION 3.10)
project(render C)

set(CMAKE_C_STANDARD 99)

# Find SDL3
find_package(SDL3 REQUIRED)

include_directories(${SDL3_INCLUDE_DIRS})

add_executable(render src/main.c src/rasterizer.c)

target_link_libraries(render ${SDL3_LIBRARIES})
