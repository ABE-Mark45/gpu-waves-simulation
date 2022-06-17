#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <cuda_gl_interop.h>
#include <vector>
#include "Utils.h"
#include "constants.hpp"
#include "SimulationManager.cuh"
#include "shapes/HeightFieldGPU.cuh"
#include "Scene.hpp"

GLFWwindow *CreateWindow()
{
    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow *window = glfwCreateWindow(1920, 1080, "Water Waves", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK)
    {
        exit(EXIT_FAILURE);
    }
    glfwSwapInterval(1);
    return window;
}

void display(GLFWwindow *window, double currentTime, HeightFieldGPU &height_field, GLuint renderingProgram)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    height_field.draw();
}

int main(void)
{
    GLFWwindow *window = CreateWindow();

    GLuint rendering_program = Utils::createShaderProgram("./shaders/vertex_shader.glsl", "./shaders/fragment_shader.glsl");
    glUseProgram(rendering_program);

    Scene scene(window, rendering_program, CONSTANTS::CAMERA_X, CONSTANTS::CAMERA_Y, CONSTANTS::CAMERA_Z);

    HeightFieldGPU height_field(CONSTANTS::RESOLUTION, CONSTANTS::SIDE_LENGTH);
    SimulationManager manager(CONSTANTS::RESOLUTION, height_field.get_height_buffer_loc());

    while (!glfwWindowShouldClose(window))
    {
        manager.step();
        display(window, glfwGetTime(), height_field, rendering_program);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    // cudaDeviceReset();
    return 0;
}