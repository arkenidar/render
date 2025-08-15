# C99 SDL3 Software Rasterizer

This project is a C99 application using SDL3 (no OpenGL) that implements a CPU-based software rasterizer with z-buffer, 3D transformations (rotation, translation), triangulated indexed mesh rendering, greyscale output, and normal-based Gouraud shading.

![first screen share](docs/screens/Screenshot_20250815_152649.png)

## Features

- CPU software rasterizer (no OpenGL)
- Z-buffer for depth
- 3D transformations: rotation, translation
- Triangulated indexed mesh rendering
- Greyscale output
- Gouraud shading based on vertex normals

## Build Instructions

- Requires: SDL3 development libraries, C99 compiler (e.g., gcc)
- Build: See below for compilation instructions (to be updated)

## Usage

- Run the executable to see a sample mesh rendered with transformations and shading.

## TODO

- [ ] Implement mesh loader
- [ ] Implement transformation pipeline
- [ ] Implement rasterizer with z-buffer
- [ ] Implement Gouraud shading
- [ ] Add sample mesh

## License

This project and all files in the repository are dedicated to the public domain under CC0 1.0 Universal. See the `LICENSE` file for details.
