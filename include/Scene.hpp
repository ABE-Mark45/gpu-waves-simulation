#pragma once
#include "glm/glm.hpp"
#include <GLFW/glfw3.h>

class Scene
{
private:
    int width;
    int height;
    float aspect_ratio;
    float cameraX, cameraY, cameraZ;
    glm::mat4 projection_mat, view_mat;
    GLuint view_mat_loc;
    GLuint projection_mat_loc;
    GLFWwindow *window;
    GLuint rendering_program;

public:
    Scene(GLFWwindow *_window, GLuint _rendering_program, float _cameraX, float _cameraY, float _cameraZ);
};

Scene::Scene(GLFWwindow *_window, GLuint _rendering_program, float _cameraX, float _cameraY, float _cameraZ)
    : window(_window), rendering_program(_rendering_program), cameraX(_cameraX), cameraY(_cameraY), cameraZ(_cameraZ)
{
    view_mat_loc = glGetUniformLocation(rendering_program, "v_matrix");
    projection_mat_loc = glGetUniformLocation(rendering_program, "proj_matrix");

    glfwGetFramebufferSize(window, &width, &height);

    aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    projection_mat = glm::perspective(1.05f, aspect_ratio, 0.1f, 10000.0f); // 1.0472 radians = 60 degrees

    view_mat = glm::lookAt(glm::vec3(cameraX, cameraY, cameraZ),
                           glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0, 0, 1.0f));

    glUniformMatrix4fv(view_mat_loc, 1, GL_FALSE, glm::value_ptr(view_mat));
    glUniformMatrix4fv(projection_mat_loc, 1, GL_FALSE, glm::value_ptr(projection_mat));
}