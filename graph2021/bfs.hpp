/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bellman-Ford algorithm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <float.h>
#include "const.h"
#include <list>
#include <queue>
#include <vector>

void cpu_bfs(GraphCSR _graph, int *_levels, int _source_vertex)
{
    int vertices_count = _graph.vertices_count;
    long long edges_count = _graph.edges_count;

    long long *outgoing_ptrs = _graph.outgoing_ptrs; // |V|
    int *outgoing_ids = _graph.outgoing_ids; // |E|

    // Mark all the vertices as not visited
    for(int i = 0; i < vertices_count; i++)
        _levels[i] = UNVISITED_VERTEX;

    // Create a queue for BFS
    list<int> queue;

    // Mark the current node as visited and enqueue it
    _levels[_source_vertex] = 1;
    queue.push_back(_source_vertex);

    while(!queue.empty())
    {
        // Dequeue a vertex from queue and print it
        int s = queue.front();
        queue.pop_front();

        const long long edge_start = outgoing_ptrs[s];
        const int connections_count = outgoing_ptrs[s + 1] - outgoing_ptrs[s];

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            long long int global_edge_pos = edge_start + edge_pos;
            int v = outgoing_ids[global_edge_pos];
            if (_levels[v] == UNVISITED_VERTEX)
            {
                _levels[v] = _levels[s] + 1;
                queue.push_back(v);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
