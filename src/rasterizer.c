// rasterizer.c
// Core software rasterizer implementation (skeleton)
#include "rasterizer.h"
#include <math.h>
#include <stdlib.h>
#include <SDL3/SDL.h>
#include <float.h>
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
    int bpp = SDL_BYTESPERPIXEL(surf->format);
    uint8_t *p = (uint8_t *)surf->pixels + y * surf->pitch + x * bpp;

    switch (bpp)
    {
    case 1:
        *p = (uint8_t)col;
        break;
    case 2:
        *(uint16_t *)p = (uint16_t)col;
        break;
    case 3:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
        p[0] = (col >> 16) & 0xFF;
        p[1] = (col >> 8) & 0xFF;
        p[2] = col & 0xFF;
#else
        p[0] = col & 0xFF;
        p[1] = (col >> 8) & 0xFF;
        p[2] = (col >> 16) & 0xFF;
#endif
        break;
    case 4:
        *(uint32_t *)p = col;
        break;
    default:
        /* Unsupported format */
        break;
    }
}

// Map 8-bit RGB into the surface's pixel format using masks
static uint32_t map_rgb_for_surface(SDL_Surface *surf, uint8_t r, uint8_t g, uint8_t b)
{
    int bpp;
    Uint32 rmask = 0, gmask = 0, bmask = 0, amask = 0;
    SDL_GetMasksForPixelFormat(surf->format, &bpp, &rmask, &gmask, &bmask, &amask);

    int rshift = 0, gshift = 0, bshift = 0;
    Uint32 tmp;

    // shift_for
    tmp = rmask;
    if (tmp == 0)
        rshift = 0;
    else
    {
        while ((tmp & 1u) == 0)
        {
            tmp >>= 1;
            rshift++;
        }
    }
    tmp = gmask;
    if (tmp == 0)
        gshift = 0;
    else
    {
        while ((tmp & 1u) == 0)
        {
            tmp >>= 1;
            gshift++;
        }
    }
    tmp = bmask;
    if (tmp == 0)
        bshift = 0;
    else
    {
        while ((tmp & 1u) == 0)
        {
            tmp >>= 1;
            bshift++;
        }
    }

    // count_bits
    int rbits = 0, gbits = 0, bbits = 0;
    tmp = (rmask >> rshift);
    while (tmp)
    {
        rbits += (tmp & 1u);
        tmp >>= 1;
    }
    tmp = (gmask >> gshift);
    while (tmp)
    {
        gbits += (tmp & 1u);
        tmp >>= 1;
    }
    tmp = (bmask >> bshift);
    while (tmp)
    {
        bbits += (tmp & 1u);
        tmp >>= 1;
    }

    uint32_t rval = (rbits > 0) ? ((r * ((1u << rbits) - 1u) + 127) / 255) : 0;
    uint32_t gval = (gbits > 0) ? ((g * ((1u << gbits) - 1u) + 127) / 255) : 0;
    uint32_t bval = (bbits > 0) ? ((b * ((1u << bbits) - 1u) + 127) / 255) : 0;

    uint32_t pixel = ((rval << rshift) & rmask) | ((gval << gshift) & gmask) | ((bval << bshift) & bmask);
    return pixel;
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

// Simple depth buffer
static float *depth_buffer = NULL;
static int depth_w = 0, depth_h = 0;

static void ensure_depth_buffer(SDL_Surface *surf)
{
    int w = surf->w;
    int h = surf->h;
    if (depth_buffer && depth_w == w && depth_h == h)
        return;
    free(depth_buffer);
    depth_buffer = (float *)malloc(sizeof(float) * (size_t)w * (size_t)h);
    if (!depth_buffer)
    {
        depth_w = depth_h = 0;
        return;
    }
    depth_w = w;
    depth_h = h;
}

static inline float edge_function(float x0, float y0, float x1, float y1, float x2, float y2)
{
    return (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0);
}

// Forward declaration for triangle rasterizer
static void draw_filled_triangle(SDL_Surface *surf, const Vertex *v0, const Vertex *v1, const Vertex *v2);

void rasterizer_render(SDL_Surface *surface)
{
    // Lock surface before direct pixel access
    if (SDL_LockSurface(surface) != 0)
    {
        // Couldn't lock the surface; proceed anyway and attempt best-effort rendering
        // (previously the code fell back to drawing 3 lines here — that path is disabled)
    }

    // Clear to black manually (avoid SDL_FillRect to prevent potential SDL3 symbol mismatch)
    {
        size_t total = (size_t)surface->h * (size_t)surface->pitch;
        memset(surface->pixels, 0, total);
    }

    // Initialize depth buffer to far (large) values
    ensure_depth_buffer(surface);
    if (depth_buffer)
    {
        for (int i = 0; i < depth_w * depth_h; ++i)
            depth_buffer[i] = FLT_MAX;
    }

    // Draw filled triangle with Gouraud shading
    draw_filled_triangle(surface, &vertices[0], &vertices[1], &vertices[2]);

    SDL_UnlockSurface(surface);
}

// Compute simple per-vertex diffuse intensity (directional light)
static void compute_vertex_light(const Vertex *v, float lx, float ly, float lz, float *out)
{
    // normalize light
    float llen = sqrtf(lx * lx + ly * ly + lz * lz);
    if (llen == 0)
        llen = 1.0f;
    lx /= llen;
    ly /= llen;
    lz /= llen;
    float dot = v->nx * lx + v->ny * ly + v->nz * lz;
    if (dot < 0)
        dot = 0;
    *out = dot;
}

// Draw filled triangle with Gouraud shading (per-vertex intensity)
static void draw_filled_triangle(SDL_Surface *surf, const Vertex *v0, const Vertex *v1, const Vertex *v2)
{
    // Ensure depth buffer
    ensure_depth_buffer(surf);
    if (!depth_buffer)
        return;

    int w = surf->w;
    int h = surf->h;

    // Precompute per-vertex intensities using a single directional light
    float i0, i1, i2;
    // Light coming from camera towards -Z
    compute_vertex_light(v0, 0.0f, 0.0f, 1.0f, &i0);
    compute_vertex_light(v1, 0.0f, 0.0f, 1.0f, &i1);
    compute_vertex_light(v2, 0.0f, 0.0f, 1.0f, &i2);

    // Screen coordinates (assume vertices are already in screen space x,y and z in [0..1])
    float x0 = v0->x, y0 = v0->y, z0 = v0->z;
    float x1 = v1->x, y1 = v1->y, z1 = v1->z;
    float x2 = v2->x, y2 = v2->y, z2 = v2->z;

    // Bounding box
    int xmin = (int)fmaxf(0.0f, floorf(fminf(fminf(x0, x1), x2)));
    int ymin = (int)fmaxf(0.0f, floorf(fminf(fminf(y0, y1), y2)));
    int xmax = (int)fminf((float)w - 1, ceilf(fmaxf(fmaxf(x0, x1), x2)));
    int ymax = (int)fminf((float)h - 1, ceilf(fmaxf(fmaxf(y0, y1), y2)));

    float area = edge_function(x0, y0, x1, y1, x2, y2);
    if (fabsf(area) < 1e-6f)
        return; // degenerate

    // Rasterize
    for (int y = ymin; y <= ymax; ++y)
    {
        for (int x = xmin; x <= xmax; ++x)
        {
            float px = (float)x + 0.5f;
            float py = (float)y + 0.5f;
            float w0 = edge_function(x1, y1, x2, y2, px, py);
            float w1 = edge_function(x2, y2, x0, y0, px, py);
            float w2 = edge_function(x0, y0, x1, y1, px, py);
            // If all barycentric coords have same sign as area, the point is inside
            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0))
            {
                float alpha = w0 / area;
                float beta = w1 / area;
                float gamma = w2 / area;

                // Interpolate depth (assuming smaller z = near, but earlier code uses z in 0..1)
                float z = alpha * z0 + beta * z1 + gamma * z2;

                int idx = y * w + x;
                if (z < depth_buffer[idx])
                {
                    depth_buffer[idx] = z;
                    // Interpolate intensity
                    float intensity = alpha * i0 + beta * i1 + gamma * i2;
                    uint8_t c = (uint8_t)(fminf(1.0f, intensity) * 255.0f);
                    uint32_t col = map_rgb_for_surface(surf, c, c, c);
                    putpixel(surf, x, y, col);
                }
            }
        }
    }
}
