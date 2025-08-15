// rasterizer.c
// Core software rasterizer implementation (skeleton)
#include "rasterizer.h"
#include <math.h>
#include <stdlib.h>
#include <SDL3/SDL.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

// model types and OBJ loader
#include "../mesh.h"
#include "../parse.h"

// Minimal 4x4 matrix helpers for a simple MVP pipeline
typedef struct { float m[4][4]; } mat4;

static mat4 mat4_identity()
{
    mat4 r; memset(&r, 0, sizeof(r));
    for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0f;
    return r;
}

static mat4 mat4_mul(const mat4 *a, const mat4 *b)
{
    mat4 r; memset(&r, 0, sizeof(r));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                r.m[i][j] += a->m[i][k] * b->m[k][j];
    return r;
}

static mat4 mat4_translate(float x, float y, float z)
{
    mat4 r = mat4_identity();
    r.m[0][3] = x; r.m[1][3] = y; r.m[2][3] = z;
    return r;
}

static mat4 mat4_perspective(float fovy_rad, float aspect, float znear, float zfar)
{
    float f = 1.0f / tanf(fovy_rad * 0.5f);
    mat4 r; memset(&r, 0, sizeof(r));
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = (zfar + znear) / (znear - zfar);
    r.m[2][3] = (2.0f * zfar * znear) / (znear - zfar);
    r.m[3][2] = -1.0f;
    return r;
}

// Transform a 3D point (x,y,z) by a 4x4 matrix, producing clip-space (x,y,z,w)
static void transform_point(const mat4 *m, float x, float y, float z, float *outx, float *outy, float *outz, float *outw)
{
    float vx = x, vy = y, vz = z, vw = 1.0f;
    float rx = m->m[0][0] * vx + m->m[0][1] * vy + m->m[0][2] * vz + m->m[0][3] * vw;
    float ry = m->m[1][0] * vx + m->m[1][1] * vy + m->m[1][2] * vz + m->m[1][3] * vw;
    float rz = m->m[2][0] * vx + m->m[2][1] * vy + m->m[2][2] * vz + m->m[2][3] * vw;
    float rw = m->m[3][0] * vx + m->m[3][1] * vy + m->m[3][2] * vz + m->m[3][3] * vw;
    *outx = rx; *outy = ry; *outz = rz; *outw = rw;
}

// Forward declarations for functions defined later
static mat4 mat4_lookat(const float eye[3], const float center[3], const float up[3]);
static void camera_autofit_model(void);

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

// Model switching
static const char *g_model_paths[] = {
    "assets/axes.obj",
    "assets/cube.obj",
    "assets/head.obj",
};
static const int g_model_count = sizeof(g_model_paths) / sizeof(g_model_paths[0]);
static model g_current_model = {0};
static int g_model_loaded = 0;
static int g_model_index = -1;

// Global pointer used by compatibility shims to draw into the active surface
SDL_Surface *g_current_surface = NULL;

// Camera state (spherical coordinates around center)
static float g_cam_yaw = 0.0f;   // around Y axis
static float g_cam_pitch = 0.0f; // up/down
static float g_cam_distance = 3.0f; // distance from center
static float g_cam_center[3] = {0.0f, 0.0f, 0.0f};

// Whether the auto-fit has been computed for the current model
static int g_cam_autofit_done = 0;

static void free_model(model *m)
{
    if (!m) return;
    free(m->vertex_positions.array);
    free(m->vertex_normals.array);
    free(m->mesh.array);
    m->vertex_positions.array = NULL; m->vertex_positions.count = 0;
    m->vertex_normals.array = NULL; m->vertex_normals.count = 0;
    m->mesh.array = NULL; m->mesh.count = 0;
}

void rasterizer_cycle_model(void)
{
    // free previous
    if (g_model_loaded)
    {
        free_model(&g_current_model);
        g_model_loaded = 0;
    }
    g_model_index = (g_model_index + 1) % g_model_count;
    const char *path = g_model_paths[g_model_index];
    g_current_model = load_model_obj(path);
    g_model_loaded = 1;
    // Automatically fit camera to the newly loaded model
    // (implemented below)
    {
        // compute bounds and autofit
        // defer to helper - call rasterizer_adjust_camera with zero deltas to trigger if needed
        extern void rasterizer_adjust_camera(float delta_yaw, float delta_pitch, float delta_zoom);
        rasterizer_adjust_camera(0.0f, 0.0f, 0.0f);
    }
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

    // Draw current model if any
    if (g_model_loaded && g_current_model.mesh.count > 0)
    {
        // Set current surface global for compatibility shims
        extern SDL_Surface *g_current_surface;
        g_current_surface = surface;

        int w = surface->w;
        int h = surface->h;
    mat4 model = mat4_identity();
        // Ensure camera autofit has been computed for the model
        if (!g_cam_autofit_done)
            camera_autofit_model();

    // Build view matrix from camera state
    // Camera transform will be computed from globals set by rasterizer_adjust_camera/auto-fit
    // compute camera position from spherical coordinates
    float camx = g_cam_center[0] + g_cam_distance * cosf(g_cam_pitch) * sinf(g_cam_yaw);
    float camy = g_cam_center[1] + g_cam_distance * sinf(g_cam_pitch);
    float camz = g_cam_center[2] + g_cam_distance * cosf(g_cam_pitch) * cosf(g_cam_yaw);
    float eye[3] = { camx, camy, camz };
    float center[3] = { g_cam_center[0], g_cam_center[1], g_cam_center[2] };
    float upv[3] = { 0.0f, 1.0f, 0.0f };
    mat4 view = mat4_lookat(eye, center, upv);
        float aspect = (float)w / (float)h;
        mat4 proj = mat4_perspective(45.0f * (3.14159265f / 180.0f), aspect, 0.1f, 100.0f);
        mat4 mv = mat4_mul(&view, &model);
        mat4 mvp = mat4_mul(&proj, &mv);

        for (int ti = 0; ti < g_current_model.mesh.count; ++ti)
        {
            triangle t = g_current_model.mesh.array[ti];
            Vertex v[3];
            for (int k = 0; k < 3; ++k)
            {
                int pi = t.array[k * 2 + 0] - 1;
                int ni = t.array[k * 2 + 1] - 1;
                if (pi < 0 || pi >= g_current_model.vertex_positions.count)
                {
                    v[k].x = 0; v[k].y = 0; v[k].z = 1;
                    v[k].nx = 0; v[k].ny = 0; v[k].nz = 1;
                    continue;
                }
                float *pos = g_current_model.vertex_positions.array[pi].array;
                float *nrm = (ni >= 0 && ni < g_current_model.vertex_normals.count) ? g_current_model.vertex_normals.array[ni].array : pos;
                float cx, cy, cz, cw;
                transform_point(&mvp, pos[0], pos[1], pos[2], &cx, &cy, &cz, &cw);
                if (cw == 0.0f) cw = 1e-6f;
                float ndc_x = cx / cw;
                float ndc_y = cy / cw;
                float ndc_z = cz / cw;
                v[k].x = (ndc_x * 0.5f + 0.5f) * (w - 1);
                v[k].y = (1.0f - (ndc_y * 0.5f + 0.5f)) * (h - 1);
                v[k].z = (ndc_z * 0.5f + 0.5f);
                v[k].nx = nrm[0]; v[k].ny = nrm[1]; v[k].nz = nrm[2];
            }
            draw_filled_triangle(surface, &v[0], &v[1], &v[2]);
        }

        g_current_surface = NULL;
    }

    SDL_UnlockSurface(surface);
}

// Build a look-at view matrix (column-major style matching our transform_point)
static mat4 mat4_lookat(const float eye[3], const float center[3], const float up[3])
{
    // compute forward, right, up vectors
    float fx = center[0] - eye[0];
    float fy = center[1] - eye[1];
    float fz = center[2] - eye[2];
    // normalize f
    float flen = sqrtf(fx*fx + fy*fy + fz*fz);
    if (flen == 0.0f) flen = 1.0f;
    fx /= flen; fy /= flen; fz /= flen;

    // up normalized
    float ux = up[0], uy = up[1], uz = up[2];
    float ulen = sqrtf(ux*ux + uy*uy + uz*uz);
    if (ulen == 0.0f) ulen = 1.0f;
    ux /= ulen; uy /= ulen; uz /= ulen;

    // s = f x up
    float sx = fy * uz - fz * uy;
    float sy = fz * ux - fx * uz;
    float sz = fx * uy - fy * ux;
    float slen = sqrtf(sx*sx + sy*sy + sz*sz);
    if (slen == 0.0f) slen = 1.0f;
    sx /= slen; sy /= slen; sz /= slen;

    // u' = s x f
    float ux2 = sy * fz - sz * fy;
    float uy2 = sz * fx - sx * fz;
    float uz2 = sx * fy - sy * fx;

    mat4 m; memset(&m, 0, sizeof(m));
    // first row
    m.m[0][0] = sx; m.m[0][1] = ux2; m.m[0][2] = -fx; m.m[0][3] = 0.0f;
    m.m[1][0] = sy; m.m[1][1] = uy2; m.m[1][2] = -fy; m.m[1][3] = 0.0f;
    m.m[2][0] = sz; m.m[2][1] = uz2; m.m[2][2] = -fz; m.m[2][3] = 0.0f;
    m.m[3][0] = 0.0f; m.m[3][1] = 0.0f; m.m[3][2] = 0.0f; m.m[3][3] = 1.0f;

    // translate
    mat4 t = mat4_translate(-eye[0], -eye[1], -eye[2]);
    return mat4_mul(&m, &t);
}

// Helper: compute model bounding box in model-space and set camera to fit
static void camera_autofit_model(void)
{
    if (!g_model_loaded || g_current_model.vertex_positions.count == 0)
        return;

    float minx = FLT_MAX, miny = FLT_MAX, minz = FLT_MAX;
    float maxx = -FLT_MAX, maxy = -FLT_MAX, maxz = -FLT_MAX;
    for (int i = 0; i < g_current_model.vertex_positions.count; ++i)
    {
        float *p = g_current_model.vertex_positions.array[i].array;
        if (p[0] < minx) minx = p[0]; if (p[0] > maxx) maxx = p[0];
        if (p[1] < miny) miny = p[1]; if (p[1] > maxy) maxy = p[1];
        if (p[2] < minz) minz = p[2]; if (p[2] > maxz) maxz = p[2];
    }
    // center
    g_cam_center[0] = (minx + maxx) * 0.5f;
    g_cam_center[1] = (miny + maxy) * 0.5f;
    g_cam_center[2] = (minz + maxz) * 0.5f;

    // find radius of bounding sphere
    float rx = (maxx - minx) * 0.5f;
    float ry = (maxy - miny) * 0.5f;
    float rz = (maxz - minz) * 0.5f;
    float radius = sqrtf(rx*rx + ry*ry + rz*rz);
    if (radius <= 0.0f) radius = 1.0f;

    // Fit camera distance using vertical FOV 45deg and horizontal aspect will be handled in render
    float fov = 45.0f * (3.14159265f / 180.0f);
    // distance such that radius fits: distance = radius / sin(fov/2)
    float dist = radius / sinf(fov * 0.5f);
    // add some padding
    dist *= 1.4f;
    g_cam_distance = dist;
    // reset orientation
    g_cam_yaw = 0.0f;
    g_cam_pitch = 0.0f;
    g_cam_autofit_done = 1;
}

// Public camera adjust API implementation
void rasterizer_adjust_camera(float delta_yaw, float delta_pitch, float delta_zoom)
{
    // apply deltas
    g_cam_yaw += delta_yaw;
    g_cam_pitch += delta_pitch;
    // clamp pitch to avoid flipping
    const float max_pitch = 1.4f; // ~80 degrees
    if (g_cam_pitch > max_pitch) g_cam_pitch = max_pitch;
    if (g_cam_pitch < -max_pitch) g_cam_pitch = -max_pitch;
    // zoom
    g_cam_distance += delta_zoom;
    if (g_cam_distance < 0.01f) g_cam_distance = 0.01f;
    // mark autofit as done/overridden
    g_cam_autofit_done = 1;
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
