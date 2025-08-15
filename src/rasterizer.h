// rasterizer.h
#ifndef RASTERIZER_H
#define RASTERIZER_H
#include <SDL3/SDL.h>

void rasterizer_render(SDL_Surface *surface);

// Cycle to the next model in the assets list (click to switch)
void rasterizer_cycle_model(void);

#endif // RASTERIZER_H
