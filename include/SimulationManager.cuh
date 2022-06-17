#pragma once
#include <constants.hpp>
#include "glm/glm.hpp"
#include <curand_kernel.h>
#include <curand.h>
#include <ctime>
#include <random>
#include <complex.cuh>
#include <fft.cuh>
#include <iostream>

__device__ __host__ double calc_phillips_coef(const glm::vec3 k)
{
    double L = (CONSTANTS::WIND_SPEED * CONSTANTS::WIND_SPEED) / CONSTANTS::GRAVITY;
    double l = L / 100;

    double local_wind_dir = k.x * CONSTANTS::WIND_DIRECTION_X + k.y * CONSTANTS::WIND_DIRECTION_Y + k.z * CONSTANTS::WIND_DIRECTION_Z;
    double k_squared = glm::dot(k, k);
    
    // To handle floating point errors
    if (k_squared < 1e-6)
        return 0;
    double phillips =
        (glm::exp(-1.0 / (k_squared * L * L)) / (k_squared * k_squared * k_squared)) * (local_wind_dir * local_wind_dir);

    // 
    if (local_wind_dir < 0)
        phillips *= 0.05;

    return phillips * glm::exp(-k_squared * l * l);
}


__global__ void gen_initial_spectrum(int resolution, complex *initial_spectrum_dev, double *angular_speeds_dev, curandState *states)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= resolution || j >= resolution)
        return;

    int index = INDEX(resolution, i, j);

    glm::vec3 k((float)resolution - 2 * i, (float)resolution - 2 * j, 0.0f);
    k *= M_PI / CONSTANTS::PATCH_SIZE;

    double phillips_scaled = std::sqrt(calc_phillips_coef(k) / 2);

    double rand_x = curand_normal_double(&states[index]);
    double rand_y = curand_normal_double(&states[index]);

    initial_spectrum_dev[index] = complex(rand_x * phillips_scaled, rand_y * phillips_scaled);
    angular_speeds_dev[index] = glm::sqrt(CONSTANTS::GRAVITY * k.length());
}

__global__ void calc_heights(double t, int resolution, double *angular_speeds_dev, complex *initial_spectrum_dev, complex *spectrum_dev)
{
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= resolution || x >= resolution)
        return;

    int i = INDEX(resolution, x, y);

    double wt = angular_speeds_dev[i] * t;
    complex h = initial_spectrum_dev[i];
    complex h1;
    if (y == 0 && x == 0)
        h1 = initial_spectrum_dev[resolution * resolution - 1];
    else if (y == 0)
        h1 = initial_spectrum_dev[resolution - 1 + (resolution - x) * resolution];
    else if (x == 0)
        h1 = initial_spectrum_dev[resolution - y + (resolution - x - 1) * resolution];
    else
        h1 = initial_spectrum_dev[resolution - y + (resolution - x) * resolution];

    glm::vec2 k(resolution * 0.5 - x, resolution * 0.5 - y);
    k /= k.length();

    complex pos_phase(cos(wt), sin(wt));
    complex neg_phase(cos(-wt), sin(-wt));

    spectrum_dev[i] = (h * pos_phase) + (h1.conj() * neg_phase);
}

__global__ void extract_heights(int resolution, float *heights_dev, complex *spectrum_dev)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= resolution || x >= resolution)
        return;

    int sign = (y + x) % 2 ? -1 : 1;
    int index = INDEX(resolution, y, x);
    heights_dev[index] = sign * spectrum_dev[index].real * CONSTANTS::RESOLUTION * CONSTANTS::RESOLUTION * 10;


}

__global__ void init_random_states(int resolution, curandState *states)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= resolution || j >= resolution)
        return;

    int index = INDEX(resolution, i, j);
    curand_init(30 * index, index, 0, &states[index]);
}

class SimulationManager
{
private:
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
    double *angular_speeds_dev;
    float *heights_dev;
    complex *initial_spectrum_dev;
    complex *spectrum_dev;
    curandState *states_dev;
    int resolution;
    dim3 grid_size;
    dim3 block_size;
    GLuint height_buffer_loc;

public:
    SimulationManager(int _resolution, GLuint _height_buffer_loc)
        : resolution(_resolution)
        , block_size(dim3(32, 32))
        , height_buffer_loc(_height_buffer_loc)
    {
        grid_size = dim3(resolution / 32, resolution / 32);
        distribution = std::normal_distribution<double>(0.0, 1.0);

        cudaMalloc(&angular_speeds_dev, resolution * resolution * sizeof(double));
        cudaMalloc(&initial_spectrum_dev, resolution * resolution * sizeof(complex));
        cudaMalloc(&spectrum_dev, resolution * resolution * sizeof(complex));
        cudaMalloc(&states_dev, resolution * resolution * sizeof(curandState));
        init_random_states<<<grid_size, block_size>>>(resolution, states_dev);
        cudaDeviceSynchronize();

        gen_initial_spectrum<<<grid_size, block_size>>>(resolution, initial_spectrum_dev, angular_speeds_dev, states_dev);
        cudaDeviceSynchronize();
    }

    ~SimulationManager()
    {
        cudaFree(angular_speeds_dev);
        cudaFree(initial_spectrum_dev);
        cudaFree(spectrum_dev);
        cudaFree(states_dev);
    }


    void step()
    {

        calc_heights<<<grid_size, block_size>>>(glfwGetTime() / 2.0, resolution, angular_speeds_dev, initial_spectrum_dev, spectrum_dev);
        cudaDeviceSynchronize();
        fft_2d_gpu(spectrum_dev, resolution, true);

        glBindBuffer(GL_ARRAY_BUFFER, height_buffer_loc);
        cudaGLMapBufferObject((void **)&heights_dev, height_buffer_loc);
        extract_heights<<<grid_size, block_size>>>(resolution, heights_dev, spectrum_dev);
        cudaDeviceSynchronize();
        cudaGLUnmapBufferObject(height_buffer_loc);
    }
};
