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

- Requires: SDL3 development libraries (system package via pkg-config **or** fetch from git), C99 compiler (e.g., gcc)

### Default: Use System SDL3 (Recommended)

By default, the build uses SDL3 from your system via pkg-config. Install SDL3 development files using your package manager:

```
# Example for Debian/Ubuntu:
sudo apt install libsdl3-dev
```

Then build with CMake:

```
cmake -S . -B build
cmake --build build
```

### Optional: Use SDL3 from Git

If you want to fetch and build SDL3 from the official git repository (e.g., for latest development version), configure CMake with:

```
cmake -S . -B build -DUSE_SDL3_FROM_GIT=ON
cmake --build build
```

This will download and build SDL3 as part of the build process.

If neither pkg-config nor git is available, the build will fail with an error.

### WebAssembly (browser) build

The project also builds to WebAssembly via [Emscripten](https://github.com/emscripten-core/emsdk). SDL3 is provided by Emscripten's built-in port (`-sUSE_SDL=3`), and the `assets/` directory is preloaded into the virtual filesystem so the existing relative OBJ paths resolve unchanged.

After installing and activating emsdk (`./emsdk install latest && ./emsdk activate latest`), in a shell where `emcc` is on PATH:

```
emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release
cmake --build build-web
```

This produces `build-web/render.{html,js,wasm,data}`. WebAssembly cannot be loaded over `file://`, so serve the directory locally:

```
python -m http.server 8000 -d build-web
# then open http://localhost:8000/render.html
```

Notes:
- Emscripten's SDL3 port is marked experimental; you'll see a `-Wexperimental` warning during the link step.
- The renderer uses `SDL_CreateRenderer` + a streaming `SDL_Texture` (rather than `SDL_GetWindowSurface` / `SDL_UpdateWindowSurface`) so that the same code path works on both desktop SDL3 and the experimental Emscripten port.
- On Windows, `wasm-ld.exe` and other emsdk binaries are unsigned, so Smart App Control / WDAC can block linking with `WinError 4551`. Disabling Smart App Control (Settings → Privacy & security → Windows Security → App & browser control) is the typical fix.

## Usage

- Run the executable to see a sample mesh rendered with transformations and shading.

## License

This project and all files in the repository are dedicated to the public domain under CC0 1.0 Universal. See the `LICENSE` file for details.
