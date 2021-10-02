#include <cuda.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

void displayMatrix(int** A, size_t M, size_t N);
__global__ void transposeKernel(int** A, int** B, int M, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  // printf("i = %d, j = %d, blockDim.x = %d, blockIdx.x = %d, threadIdx.x =
  // %d\n",
  //       i, j, blockDim.x, blockIdx.x, threadIdx.x);
  printf("j = %d, blockDim.y = %d, blockIdx.y = %d, threadIdx.y = %d\n", j,
         blockDim.y, blockIdx.y, threadIdx.y);
  if ((i < M) && (j < N)) {
    printf("in\n");
    B[j][i] = A[i][j];
  }
}

cudaError_t transposeHost(int** h_A, int** h_B, int M, int N) {
  cudaError_t status = cudaSuccess;
  cudaEvent_t start, finish;
  cudaEventCreate(&start);
  cudaEventCreate(&finish);
  const int BLOCK_SIZE = 1024;
  const int GRID_SIZE = (M - 1) / BLOCK_SIZE + 1;

  dim3 Dim3Blocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 Dim3Grids(N / BLOCK_SIZE, M / BLOCK_SIZE);

  int** d_A;
  int** d_B;
  float msecs = 0;

  auto begin = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      h_B[j][i] = h_A[i][j];
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cputime = end - begin;
  std::cout << "CPU Elapsed Time: " << cputime.count() << " ms" << std::endl;

  displayMatrix(h_A, M, N);
  displayMatrix(h_B, N, M);

  status = cudaMalloc((void**)&d_A, M * sizeof(int*));
  if (status != cudaSuccess) {
    std::cerr << "cudaMalloc failed for d_A!\n";
    goto Error;
  }
  for (int i = 0; i < M; i++) {
    status = cudaMalloc((void**)&d_A, N * sizeof(int));
    if (status != cudaSuccess) {
      std::cerr << "cudaMalloc failed for d_A!\n";
      goto Error;
    }
  }

  status = cudaMalloc((void**)&d_B, N * sizeof(int*));
  if (status != cudaSuccess) {
    std::cerr << "cudaMalloc failed for d_B!\n";
    goto Error;
  }
  status = cudaMemcpy(d_A, h_A, M * sizeof(int*), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    std::cerr << "cudaMemcpy failed for h_A to d_A.\n";
    goto Error;
  }

  /*
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        h_B[i][j] = -1;
      }
    }
  */

  cudaEventRecord(start);
  transposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B, M, N);
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

void displayMatrix(int** A, size_t M, size_t N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << A[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  if (argc == 3) {
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    std::cout << "M = " << M << ", N = " << N << "\n";
    size_t size_M = M * sizeof(int*);
    size_t size_N = N * sizeof(int*);

    int** h_A = (int**)malloc(size_M);
    if (h_A == NULL) {
      std::cerr << "Failed allocating memory for h_A!";
      return 1;
    }

    for (int i = 0; i < M; i++) {
      *(h_A + i) = (int*)malloc(N * sizeof(int));
      if (*(h_A + i) == NULL) {
        std::cerr << "Error allocating memory for h_A[" << i << "].\n";
        return 2;
      }
    }

    int** h_B = (int**)malloc(size_N);
    if (h_B == NULL) {
      std::cerr << "Failed allocating memory for h_B!";
      return 3;
    }

    for (int i = 0; i < N; i++) {
      *(h_B + i) = (int*)malloc(M * sizeof(int));
      if (*(h_B + i) == NULL) {
        std::cerr << "Failed allocating memory for h_B[" << i << "].\n";
        return 4;
      }
    }

    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        h_A[i][j] = rand() % 10;

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