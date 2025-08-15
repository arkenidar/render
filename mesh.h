
#ifndef MESH_H
#define MESH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float array[3];
} vector;
typedef struct
{
    vector *array;
    int count;
} vector_array;

typedef struct
{
    int array[6];
} triangle;
typedef struct
{
    triangle *array;
    int count;
} triangle_array;

typedef struct
{
    vector_array vertex_positions;
    vector_array vertex_normals;
    triangle_array mesh;
} model;

#ifdef __cplusplus
}
#endif

#endif // MESH_H
