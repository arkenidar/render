// main.c
// Entry point for the C99 SDL3 software rasterizer
#include <SDL3/SDL.h>
#include <stdio.h>
#include "rasterizer.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

typedef struct {
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;
    SDL_Surface *surface;
    int running;
} app_state;

static void frame(void *arg)
{
    app_state *s = (app_state *)arg;
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_EVENT_QUIT)
            s->running = 0;
        else if (event.type == SDL_EVENT_KEY_DOWN)
        {
            if (event.key.key == SDLK_ESCAPE)
                s->running = 0;
        }
        else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
        {
            if (event.button.button == SDL_BUTTON_LEFT) rasterizer_cycle_model();
        }
    }
    rasterizer_render(s->surface);
    SDL_UpdateTexture(s->texture, NULL, s->surface->pixels, s->surface->pitch);
    SDL_RenderClear(s->renderer);
    SDL_RenderTexture(s->renderer, s->texture, NULL, NULL);
    SDL_RenderPresent(s->renderer);

#ifdef __EMSCRIPTEN__
    if (!s->running) emscripten_cancel_main_loop();
#endif
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Software Rasterizer", 800, 600, 0);
    if (!window)
    {
        fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer)
    {
        fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    const SDL_PixelFormat fmt = SDL_PIXELFORMAT_XRGB8888;
    SDL_Surface *surface = SDL_CreateSurface(800, 600, fmt);
    SDL_Texture *texture = SDL_CreateTexture(renderer, fmt, SDL_TEXTUREACCESS_STREAMING, 800, 600);
    if (!surface || !texture)
    {
        fprintf(stderr, "Surface/Texture create failed: %s\n", SDL_GetError());
        if (texture) SDL_DestroyTexture(texture);
        if (surface) SDL_DestroySurface(surface);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    rasterizer_cycle_model();

    static app_state state;
    state.window = window;
    state.renderer = renderer;
    state.texture = texture;
    state.surface = surface;
    state.running = 1;

#ifdef __EMSCRIPTEN__
    // Browsers cannot tolerate a blocking infinite loop; hand control back
    // each frame via the Emscripten main-loop callback.
    emscripten_set_main_loop_arg(frame, &state, 0, 1);
#else
    while (state.running) frame(&state);
    SDL_DestroyTexture(texture);
    SDL_DestroySurface(surface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
#endif
    return 0;
}
