// rasterizer.c
// Core software rasterizer implementation (skeleton)
#include "rasterizer.h"
#include <math.h>
#include <stdlib.h>
#include <SDL3/SDL_pixels.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Placeholder mesh data (single triangle)
typedef struct
{
    float x, y, z;
    float nx, ny, nz; // normal
} Vertex;

typedef struct
{
    int v0, v1, v2;
} Triangle;

static Vertex vertices[] = {
    {100, 100, 0.5, 0, 0, 1},
    {400, 100, 0.5, 0, 0, 1},
    {250, 400, 0.5, 0, 0, 1},
};
// static Triangle triangles[] = {
//     {0, 1, 2},
// };

// Helper: set a pixel on the surface
static void putpixel(SDL_Surface *surf, int x, int y, uint32_t col)
{
    if (x < 0 || y < 0 || x >= surf->w || y >= surf->h)
        return;
    int bpp;
    Uint32 rmask, gmask, bmask, amask;
    SDL_GetMasksForPixelFormat(surf->format, &bpp, &rmask, &gmask, &bmask, &amask);
    int bytes = bpp / 8;
    uint8_t *p = (uint8_t *)surf->pixels + y * surf->pitch + x * bytes;
    memcpy(p, &col, bytes);
}

// Helper: draw a line using Bresenham's algorithm
static void draw_line(SDL_Surface *surf, int x0, int y0, int x1, int y1, uint32_t col)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (1)
    {
        putpixel(surf, x0, y0, col);
        if (x0 == x1 && y0 == y1)
            break;
        e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

void rasterizer_render(SDL_Surface *surface)
{
    // Clear to black
    SDL_FillSurfaceRect(surface, NULL, SDL_MapRGB(SDL_GetPixelFormatDetails(surface->format), NULL, 0, 0, 0));
    // TODO: Implement z-buffer, transformations, and Gouraud shading
    // Draw a white triangle as placeholder (draw edges by setting pixels)
    uint32_t color = SDL_MapRGB(SDL_GetPixelFormatDetails(surface->format), NULL, 200, 200, 200);
    draw_line(surface, (int)vertices[0].x, (int)vertices[0].y, (int)vertices[1].x, (int)vertices[1].y, color);
    draw_line(surface, (int)vertices[1].x, (int)vertices[1].y, (int)vertices[2].x, (int)vertices[2].y, color);
    draw_line(surface, (int)vertices[2].x, (int)vertices[2].y, (int)vertices[0].x, (int)vertices[0].y, color);
}
