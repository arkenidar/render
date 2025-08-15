// rasterizer.h
#ifndef RASTERIZER_H
#define RASTERIZER_H
#include <SDL3/SDL.h>

void rasterizer_render(SDL_Surface *surface);

// Cycle to the next model in the assets list (click to switch)
void rasterizer_cycle_model(void);
// Adjust camera: yaw and pitch in radians, zoom is additive distance change
void rasterizer_adjust_camera(float delta_yaw, float delta_pitch, float delta_zoom);

#endif // RASTERIZER_H
