cmake_minimum_required(VERSION 3.17)

project(waves-simulation VERSION 1.0 LANGUAGES CXX CUDA)

# ;arch=compute_75;code=sm_75
find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++17;-g;-G;-gencode)
include_directories(/usr/local/cuda-11.0/include)
link_directories(/usr/local/cuda-11.0/lib64)

add_subdirectory(glm/glm)
include_directories("${PROJECT_SOURCE_DIR}/glm")

add_compile_options(-w)

include_directories(
    ${CUDA_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} main.cu)
# target_include_directories(${PROJECT_NAME} PUBLIC glm)
target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} GL dl glfw GLEW cudart ${CUDA_LIBRARIES})