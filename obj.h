#include <GL/gl.h>
//-----------------------------------------

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

#include "parse.h" // parse model to load

void draw_triangle(triangle t, model m)
{

    vector vertex_normal, vertex_position;

    vertex_normal = m.vertex_normals.array[t.array[1] - 1];
    glNormal3f(vertex_normal.array[0], vertex_normal.array[1], vertex_normal.array[2]);
    vertex_position = m.vertex_positions.array[t.array[0] - 1];
    glVertex3f(vertex_position.array[0], vertex_position.array[1], vertex_position.array[2]);

    vertex_normal = m.vertex_normals.array[t.array[3] - 1];
    glNormal3f(vertex_normal.array[0], vertex_normal.array[1], vertex_normal.array[2]);
    vertex_position = m.vertex_positions.array[t.array[2] - 1];
    glVertex3f(vertex_position.array[0], vertex_position.array[1], vertex_position.array[2]);

    vertex_normal = m.vertex_normals.array[t.array[5] - 1];
    glNormal3f(vertex_normal.array[0], vertex_normal.array[1], vertex_normal.array[2]);
    vertex_position = m.vertex_positions.array[t.array[4] - 1];
    glVertex3f(vertex_position.array[0], vertex_position.array[1], vertex_position.array[2]);
}

void draw_model(model m)
{
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < m.mesh.count; i++)
        draw_triangle(m.mesh.array[i], m);
    glEnd();
}

model cube;

void draw_box(float scale)
{
    glPushMatrix();
    glScalef(scale, scale, scale);
    draw_model(cube);
    glPopMatrix();
}

model model1;
void init_models()
{
    model1 = load_model_obj("../assets/head.obj");
    cube = load_model_obj("../assets/cube.obj");
}