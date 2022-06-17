#pragma once
#include "complex.cuh"

#define INDEX(rowSize, i, j) ((rowSize) * (i) + (j))

__device__ inline int bit_reverse(int num, int lg)
{
    int rev = 0;
    for (int i = 0; i < lg; i++)
        if (num & (1 << i))
            rev |= 1 << (lg - i - 1);
    return rev;
}

__global__ void fft_1d_swap_stage(complex *arr, int n, int lg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    int rev = bit_reverse(i, lg);
    if (i < rev)
    {
        complex tmp = arr[i];
        arr[i] = arr[rev];
        arr[rev] = tmp;
    }
}

__global__ void fft_1d_stage(complex *arr, int n, bool invert, int stage)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2)
        return;

    int h = stage / 2;
    int i = idx / h * stage;
    int j = idx % h;

    double angle = 2 * M_PI * j / stage * (invert ? 1 : -1);
    complex w(cos(angle), sin(angle));

    complex u = arr[i + j], v = arr[i + j + stage / 2] * w;
    arr[i + j] = u + v;
    arr[i + j + stage / 2] = u - v;
}

__global__ void fft_1d_invert_stage(complex *arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    arr[i] /= n;
}

void fft_1d_gpu(complex *arr_gpu, int n, bool invert)
{
    int h = n >> 1;
    int numBlocks = n / 1024 + (n % 1024 != 0);
    int numBlocksCompute = h / 1024 + (h % 1024 != 0);

    int lg = 0;
    while ((1 << lg) < n)
        lg++;

    fft_1d_swap_stage<<<numBlocks, 1024>>>(arr_gpu, n, lg);
    cudaDeviceSynchronize();
    for (int stage = 2; stage <= n; stage <<= 1)
    {
        fft_1d_stage<<<numBlocksCompute, 1024>>>(arr_gpu, n, invert, stage);
        cudaDeviceSynchronize();
    }

    if (invert)
    {
        fft_1d_invert_stage<<<numBlocks, 1024>>>(arr_gpu, n);
        cudaDeviceSynchronize();
    }
}

__global__ void fft_2d_swap_row_stage(complex *arr, int n, int lg)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || j >= n)
        return;
    int rev = bit_reverse(j, lg);

    if (j < rev)
    {
        complex tmp = arr[INDEX(n, row, j)];
        arr[INDEX(n, row, j)] = arr[INDEX(n, row, rev)];
        arr[INDEX(n, row, rev)] = tmp;
    }
}

__global__ void fft_2d_compute_row_stage(complex *arr, int n, bool invert, int stage)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || idx >= (n >> 1))
        return;

    int h = stage >> 1;
    int i = idx / h * stage;
    int j = idx % h;

    double angle = 2 * M_PI * j / stage * (invert ? 1 : -1);
    complex w(cos(angle), sin(angle));

    int even = i + j;
    int odd = i + j + (stage >> 1);

    complex u = arr[INDEX(n, row, even)], v = arr[INDEX(n, row, odd)] * w;
    arr[INDEX(n, row, even)] = u + v;
    arr[INDEX(n, row, odd)] = u - v;
}


__global__ void fft_2d_compute_col_stage(complex *arr, int n, bool invert, int stage)
{
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n || idx >= (n >> 1))
        return;

    int h = stage >> 1;
    int i = idx / h * stage;
    int j = idx % h;

    double angle = 2 * M_PI * j / stage * (invert ? 1 : -1);
    complex w(cos(angle), sin(angle));

    int even = i + j;
    int odd = i + j + (stage >> 1);

    complex u = arr[INDEX(n, even, col)], v = arr[INDEX(n, odd, col)] * w;
    arr[INDEX(n, even, col)] = u + v;
    arr[INDEX(n, odd, col)] = u - v;
}



__global__ void fft_2d_swap_col_stage(complex *arr, int n, int lg)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || col >= n)
        return;
    int rev = bit_reverse(i, lg);

    if (i < rev)
    {
        complex tmp = arr[INDEX(n, i, col)];
        arr[INDEX(n, i, col)] = arr[INDEX(n, rev, col)];
        arr[INDEX(n, rev, col)] = tmp;
    }
}


__global__ void fft_2d_invert_stage(complex* arr, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n)
        return;

    arr[INDEX(n, i, j)] /= n * n;
}


void fft_2d_gpu(complex *arr_gpu, int n, bool invert)
{
    int h = n >> 1;
    int numBlocks = n / 32 + (n % 32 != 0);
    int numBlocksCompute = h / 32 + (h % 32 != 0);

    dim3 gridSize(numBlocks, numBlocks);
    dim3 gridSizeColCompute(numBlocks, numBlocksCompute);
    dim3 gridSizeRowCompute(numBlocksCompute, numBlocks);

    dim3 blockSize(32, 32);
    int lg = 0;
    while ((1 << lg) < n)
        lg++;

    fft_2d_swap_row_stage<<<gridSize, blockSize>>>(arr_gpu, n, lg);
    cudaDeviceSynchronize();

    for (int stage = 2; stage <= n; stage <<= 1)
    {
        fft_2d_compute_row_stage<<<gridSizeRowCompute, blockSize>>>(arr_gpu, n, invert, stage);
        cudaDeviceSynchronize();
    }

    fft_2d_swap_col_stage<<<gridSize, blockSize>>>(arr_gpu, n, lg);
    cudaDeviceSynchronize();

    for (int stage = 2; stage <= n; stage <<= 1)
    {
        fft_2d_compute_col_stage<<<gridSizeColCompute, blockSize>>>(arr_gpu, n, invert, stage);
        cudaDeviceSynchronize();
    }

    if (invert)
    {
        fft_2d_invert_stage<<<gridSize, blockSize>>>(arr_gpu, n);
        cudaDeviceSynchronize();
    }
}
