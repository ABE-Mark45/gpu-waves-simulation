#pragma once
#include <GL/glew.h>
#include <memory>
#include <vector>
#include <cuda_gl_interop.h>

#define INDEX(rowSize, i, j) ((rowSize) * (i) + (j))

class HeightFieldGPU
{
private:
    GLuint vao;
    GLuint vertex_buffer_loc;
    GLuint index_buffer_loc;
    GLuint height_buffer_loc;

    int resolution;
    int side_length;
    std::unique_ptr<float[]> vertex_positions;
    float *height_map;
    std::vector<unsigned int> indices;

    void setup_vertices()
    {
        vertex_positions = std::unique_ptr<float[]>(new float[resolution * resolution * 2]);

        for (int i = 0; i < resolution; i++)
        {
            for (int j = 0; j < resolution; j++)
            {
                float y = side_length * i / resolution - side_length / 2.0f;
                float x = side_length * j / resolution - side_length / 2.0f;

                int index = INDEX(resolution, i, j) * 2;
                vertex_positions[index] = x;
                vertex_positions[index + 1] = y;
            }
        }
    }

    void setup_indices()
    {
        for (int i = 0; i < resolution - 1; i++)
        {
            for (int j = 0; j < resolution - 1; j++)
            {
                int ul = INDEX(resolution, i, j);
                int ur = INDEX(resolution, i, j + 1);
                int dl = INDEX(resolution, i + 1, j);
                int dr = INDEX(resolution, i + 1, j + 1);

                indices.push_back(ul);
                indices.push_back(ur);
                indices.push_back(dl);

                indices.push_back(dl);
                indices.push_back(dr);
                indices.push_back(ur);
            }
        }
    }

    void setup_opengl_attributes()
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vertex_buffer_loc);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_loc);
        glBufferData(GL_ARRAY_BUFFER, resolution * resolution * 2 * sizeof(float), vertex_positions.get(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
        glEnableVertexAttribArray(0);

        glGenBuffers(1, &height_buffer_loc);
        glBindBuffer(GL_ARRAY_BUFFER, height_buffer_loc);
        glBufferData(GL_ARRAY_BUFFER, resolution * resolution * sizeof(float), nullptr, GL_DYNAMIC_COPY);
        cudaGLRegisterBufferObject(height_buffer_loc);

        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);

        glGenBuffers(1, &index_buffer_loc);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_loc);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
    }

public:
    // HeightFieldGPU() = delete;

    HeightFieldGPU(int _resolution, double _side_length)
        : resolution(_resolution), side_length(_side_length)
    {
        setup_vertices();
        setup_indices();
        setup_opengl_attributes();
    }

    GLuint get_height_buffer_loc() const
    {
        return height_buffer_loc;
    }

    void draw() const
    {
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_loc);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_loc);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);
    }
};