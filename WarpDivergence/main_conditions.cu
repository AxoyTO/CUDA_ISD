#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <iostream>

__global__ void DivergencyKernel(float* a, int N) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (!(threadIdx.x % 2))
    a[x] = a[x] * (threadIdx.x + 1);
  else
    a[x] = a[x] * (threadIdx.x % 5);
}

__global__ void NoDivergencyKernel(float* a, int N) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  a[x] = threadIdx.x;
}

int main(int argc, char** argv) {
  if (argc == 2) {
    int N = atoi(argv[1]);
    size_t size = N * sizeof(float);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* h_A = (float*)malloc(size);
    if (h_A == NULL) {
      std::cerr << "Failed malloc for h_A!\n";
      return 1;
    }

    for (int i = 0; i < N; i++) {
      h_A[i] = i + 1;
    }

    float* d_A = NULL;
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = (N - 1) / BLOCK_SIZE + 1;
    cudaEventRecord(start);
    DivergencyKernel<<<BLOCK_SIZE, GRID_SIZE>>>(d_A, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    std::cout << "(Divergency) Kernel Time: " << msecs << " ms.\n";

    cudaEventRecord(start);
    NoDivergencyKernel<<<BLOCK_SIZE, GRID_SIZE>>>(d_A, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecs, start, stop);
    std::cout << "(Non-Divergency) Kernel Time: " << msecs << " ms.\n";

    cudaFree(d_A);
    free(h_A);
  }

  return 0;
}