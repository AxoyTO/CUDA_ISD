#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

void displayMatrix(int* A, size_t M, size_t N);
__global__ void transposeKernel(int* A, int* B, int M, int N) {
  int i_A = N * (blockDim.y * blockIdx.y + threadIdx.y) +
            blockDim.x * blockIdx.x + threadIdx.x;

            
}

cudaError_t transposeHost(int* h_A, int* h_B, int M, int N) {
  cudaError_t status = cudaSuccess;
  cudaEvent_t start, finish;
  cudaEventCreate(&start);
  cudaEventCreate(&finish);
  size_t size = M * N * sizeof(int);
  size_t pitch;

  float msecs = 0;

  int* d_A;
  int* d_B;
  const int BLOCK_SIZE = 1024;
  const int GRID_SIZE = (M - 1) / BLOCK_SIZE + 1;

  dim3 Dim3Blocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 Dim3Grids(N / BLOCK_SIZE, M / BLOCK_SIZE);

  int i = 0, k = 0;
  auto begin = std::chrono::high_resolution_clock::now();

  while (i < M * N) {
    for (int j = k; j < M * N; j += N) {
      h_B[i++] = h_A[j];
    }
    k++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cputime = end - begin;
  std::cout << "CPU Elapsed Time: " << cputime.count() << " ms" << std::endl;

  std::cout << "\n******* CPU *********\n";

  displayMatrix(h_A, M, N);
  displayMatrix(h_B, N, M);

  std::cout << "\n******* CPU *********\n\n";

  status = cudaMalloc((void**)&d_A, size);
  if (status != cudaSuccess) {
    std::cerr << "cudaMalloc failed for d_A!\n";
    goto Error;
  }

  status = cudaMalloc((void**)&d_B, size);
  if (status != cudaSuccess) {
    std::cerr << "cudaMalloc failed for d_B!\n";
    goto Error;
  }

  status = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    std::cerr << "cudaMemcpy failed for h_A to d_A.\n";
    goto Error;
  }
  /*
    for (int i = 0; i < M * N; i++) {
      h_B[i] = -1;
    }
  */

  cudaEventRecord(start);
  transposeKernel<<<Dim3Grids, Dim3Blocks>>>(d_A, d_B, M, N);
  cudaDeviceSynchronize();
  cudaEventRecord(finish);
  cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&msecs, start, finish);
  std::cout << "GPU(CUDA) Elapsed Time: " << msecs << "ms\n";

  displayMatrix(h_B, N, M);

Error:
  cudaFree(d_A);
  cudaFree(d_B);

  return status;
}

void displayMatrix(int* A, size_t M, size_t N) {
  for (size_t i = 0; i < M * N; i++) {
    if (i % N == 0)
      std::cout << "\n";
    std::cout << A[i] << " ";
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  if (argc == 3) {
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    std::cout << "M = " << M << ", N = " << N << "\n";
    size_t size = M * N * sizeof(int);

    int* h_A = (int*)malloc(size);
    if (h_A == NULL) {
      std::cerr << "Failed allocating memory for h_A!";
      return 1;
    }

    int* h_B = (int*)malloc(size);
    if (h_B == NULL) {
      std::cerr << "Failed allocating memory for h_B!";
      return 3;
    }

    for (int i = 0; i < M * N; i++) {
      // h_A[i] = rand() % 100;
      h_A[i] = i + 1;
    }

    cudaError_t status = transposeHost(h_A, h_B, M, N);
    if (status != cudaSuccess) {
      std::cerr << "transposeHost failed!\n";
      return 1;
    }

    free(h_A);
    free(h_B);

    return 0;
  }
}