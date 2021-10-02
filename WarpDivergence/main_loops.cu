#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

__global__ void addVectorsKernel(const double* a,
                                 const double* b,
                                 double* c,
                                 int n) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  while (x < n) {
    c[x] = a[x] + b[x];
    printf("x = %d | threadIdx.x = %d | blockIdx.x = %d\n", x, threadIdx.x,
           blockIdx.x);
  }
}

int main(int argc, char** argv) {
  if (argc == 2) {
    int n = atoi(argv[1]);
    size_t size = n * sizeof(double);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double* h_A = (double*)malloc(size);
    double* h_B = (double*)malloc(size);
    double* h_C = (double*)malloc(size);

    for (int i = 0; i < n; i++) {
      h_A[i] = rand() % 10;
      h_B[i] = rand() % 10;
    }

    double* d_A = NULL;
    double* d_B = NULL;
    double* d_C = NULL;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    double Blocks = 1024;                 // threads per block
    double Grids = (n - 1) / Blocks + 1;  // blocks per grid

    cudaEventRecord(start);
    addVectorsKernel<<<Grids, Blocks>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    std::cout << "GPU Elapsed Time: " << msecs << " ms.\n";

    for (int i = 0; i < n; i++) {
      if (h_C[i] != h_A[i] + h_B[i]) {
        std::cerr << "TEST FAILED...\n";
        return 1;
      }
    }

    std::cout << "TEST PASSED!\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
  }

  return 0;
}