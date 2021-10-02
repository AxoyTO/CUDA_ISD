//
//  main.cpp
//
//
//  Created by Elijah Afanasiev on 25.09.2018.
//
//

// System includes
#include <assert.h>
#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

__global__ void vectorAddGPU(float* a, float* b, float* c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

void unified_sample(int size = 1048576) {
  int n = size;
  int nBytes = n * sizeof(float);

  float *a, *b, *c;
  // float *d_a, *d_b, *d_c;

  cudaEvent_t unifiedStart, unifiedStop;
  cudaEventCreate(&unifiedStart);
  cudaEventCreate(&unifiedStop);

  dim3 block(256);
  dim3 grid((unsigned int)ceil(n / (float)block.x));
  printf("Allocating managed(unified) memory on both host and device..\n");

  cudaMallocManaged(&a, nBytes);
  cudaMallocManaged(&b, nBytes);
  cudaMallocManaged(&c, nBytes);

  for (int i = 0; i < n; i++) {
    a[i] = rand() / (float)RAND_MAX;
    b[i] = rand() / (float)RAND_MAX;
  }

  cudaEventRecord(unifiedStart);
  printf("Doing GPU Vector add\n");
  vectorAddGPU<<<grid, block>>>(a, b, c, n);
  cudaEventRecord(unifiedStop);
  cudaDeviceSynchronize();

  float elapsedUnified;
  cudaEventElapsedTime(&elapsedUnified, unifiedStart, unifiedStop);
  std::cout << "Unified-Memory copying Elapsed Time: " << elapsedUnified
            << " ms.\n";

  cudaThreadSynchronize();
}

void pinned_sample(int size = 1048576) {
  int n = size;
  int nBytes = n * sizeof(float);

  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;

  cudaEvent_t pinnedStart, pinnedStop;
  cudaEventCreate(&pinnedStart);
  cudaEventCreate(&pinnedStop);

  dim3 block(256);
  dim3 grid((unsigned int)ceil(n / (float)block.x));

  printf("Allocating device pinned memory on host\n");
  cudaMallocHost(&h_a, nBytes);
  cudaMallocHost(&h_b, nBytes);
  cudaMallocHost(&h_c, nBytes);
  cudaMalloc(&d_a, nBytes);
  cudaMalloc(&d_b, nBytes);
  cudaMalloc(&d_c, nBytes);

  for (int i = 0; i < n; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
    h_c[i] = 0;
  }

  printf("Copying to device..\n");
  printf("Doing GPU Vector Add\n");
  cudaEventRecord(pinnedStart);
  cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
  vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, n);

  cudaEventRecord(pinnedStop);
  cudaDeviceSynchronize();

  float elapsedPinned;
  cudaEventElapsedTime(&elapsedPinned, pinnedStart, pinnedStop);
  std::cout << "Pinned-Memory copying Elapsed Time: " << elapsedPinned
            << " ms.\n";

  cudaThreadSynchronize();
}

void usual_sample(int size = 1048576) {
  int n = size;

  int nBytes = n * sizeof(float);

  float *a, *b;  // host data
  float* c;      // results

  a = (float*)malloc(nBytes);
  b = (float*)malloc(nBytes);
  c = (float*)malloc(nBytes);

  float *a_d, *b_d, *c_d;

  dim3 block(256);
  dim3 grid((unsigned int)ceil(n / (float)block.x));

  for (int i = 0; i < n; i++) {
    a[i] = rand() / (float)RAND_MAX;
    b[i] = rand() / (float)RAND_MAX;
    c[i] = 0;
  }

  printf("Allocating device memory on host..\n");

  cudaMalloc((void**)&a_d, n * sizeof(float));
  cudaMalloc((void**)&b_d, n * sizeof(float));
  cudaMalloc((void**)&c_d, n * sizeof(float));

  printf("Copying to device..\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaMemcpy(a_d, a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice);

  printf("Doing GPU Vector add\n");

  vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("time: %f ms\n", milliseconds);

  cudaThreadSynchronize();

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  free(a);
  free(b);
  free(c);
}

int main(int argc, char** argv) {
  std::cout << "-------> USUAL SAMPLE <-------\n";
  usual_sample(atoi(argv[1]));
  std::cout << "-------> USUAL SAMPLE <-------\n\n";
  std::cout << "-------> PINNED SAMPLE <-------\n";
  pinned_sample(atoi(argv[1]));
  std::cout << "-------> PINNED SAMPLE <-------\n\n";
  std::cout << "-------> UNIFIED SAMPLE <-------\n";
  unified_sample(atoi(argv[1]));
  std::cout << "-------> UNIFIED SAMPLE <-------\n";
  return 0;
}