#include "cuda_error_hadling.h"
#include "bfs_gpu.cuh"

#include <limits>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "const.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// init levels
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_kernel(int *_levels, int _vertices_count, int _source_vertex)
{
    register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;

    // все вершины кроме источника еще не посещены
    if (idx < _vertices_count)
        _levels[idx] = UNVISITED_VERTEX;

    _levels[_source_vertex] = 1; // вершина-источник помещается на первый "уровень"
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// main computational algorithm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bfs_kernel(int *_levels,
                           long long *_outgoing_ptrs,
                           int *_outgoing_ids,
                           int _vertices_count,
                           long long _edges_count,
                           int *_changes,
                           int _current_level)
{
//    register const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    register const int src_id = (blockIdx.x * blockDim.x + threadIdx.x) / 64;

    if (src_id < _vertices_count) // для всех графовых вершин выполнить следующее
    {
        if(_levels[src_id] == _current_level) // если графовая вершина принадлежит текущему (ранее посещенному уровню)
        {
            const long long edge_start = _outgoing_ptrs[src_id]; // получаем положение первого ребра вершины
            const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id]; // получаем число смежных ребер вершины

            for(int edge_pos = threadIdx.x % 64; edge_pos < connections_count; edge_pos += 64) // для каждого смежного ребра делаем:
            {
                int dst_id = _outgoing_ids[edge_start + edge_pos]; // загружаем ID напарвляющей вершины ребра

                if (_levels[dst_id] == UNVISITED_VERTEX) // если направляющая вершина - не посещенная
                {
                    _levels[dst_id] = _current_level + 1; // то помечаем её следующим уровнем
                    _changes[0] = 1;
                }
            }
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// single GPU implememntation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_bfs_wrapper(long long *_outgoing_ptrs, int *_outgoing_ids, int _vertices_count, long long _edges_count, int _source_vertex, int *_levels)
{
    dim3 init_threads(1024);
    dim3 init_blocks((_vertices_count - 1) / init_threads.x + 1);

    // call init kernel
    SAFE_KERNEL_CALL((init_kernel <<< init_blocks, init_threads >>> (_levels, _vertices_count, _source_vertex)));

    // device variable to stop iterations, for each source vertex
    int *changes;
    SAFE_CALL(cudaMallocManaged((void**)&changes, sizeof(int)));

    // set grid size
    dim3 compute_threads(1024);
    dim3 compute_blocks(64 * (_vertices_count - 1) / compute_threads.x + 1);

    int current_level = 1;

    // compute shortest paths
    do
    {
        changes[0] = 0;

        SAFE_KERNEL_CALL((bfs_kernel <<< compute_blocks, compute_threads >>>
        (_levels, _outgoing_ptrs, _outgoing_ids, _vertices_count, _edges_count,
         changes, current_level)));

        current_level++;
    }
    while(changes[0] > 0);

    SAFE_CALL(cudaFree(changes));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////