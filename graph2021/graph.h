//
//  graph.h
//  
//
//  Created by Elijah Afanasiev on 23.09.2018.
//
//

#ifndef graph_h
#define graph_h

#include <string>

using namespace std;

struct Graph
{
public:
    int *src_ids; // |E| elements
    int *dst_ids; // |E| elements
    float *weights; // |E| elements - веса в данном алгоритме не используются
    int vertices_count; // = |V|
    long long edges_count; // = |E|

    Graph(int _vertices_count=1, long long _edges_count=1)
    {
        vertices_count = _vertices_count;
        edges_count = _edges_count;
        src_ids = new int[edges_count];
        dst_ids = new int[edges_count];
        weights = new float[edges_count];
    }

    ~Graph()
    {
        delete[]src_ids;
        delete[]dst_ids;
        delete[]weights;
    }

    void resize(int new_vertices_count, long long new_edges_count)
    {
        delete[]src_ids;
        delete[]dst_ids;
        delete[]weights;

        src_ids = new int[new_edges_count];
        dst_ids = new int[new_edges_count];
        weights = new float[new_edges_count];

        vertices_count = new_vertices_count;
        edges_count = new_edges_count;
    }

    bool save_to_binary_file(string _file_name)
    {
        FILE * graph_file = fopen(_file_name.c_str(), "wb");
        if(graph_file == NULL)
            return false;

        int vertices_count = this->vertices_count;
        long long edges_count = this->edges_count;
        fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
        fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);

        fwrite(reinterpret_cast<const void*>(src_ids), sizeof(int), this->edges_count, graph_file);
        fwrite(reinterpret_cast<const void*>(dst_ids), sizeof(int), this->edges_count, graph_file);
        //fwrite(reinterpret_cast<const void*>(weights), sizeof(float), this->edges_count, graph_file);

        fclose(graph_file);
        return true;
    }

    bool load_from_binary_file(string _file_name)
    {
        FILE * graph_file = fopen(_file_name.c_str(), "rb");
        if(graph_file == NULL)
            return false;

        fread(reinterpret_cast<void*>(&this->vertices_count), sizeof(int), 1, graph_file);
        fread(reinterpret_cast<void*>(&this->edges_count), sizeof(long long), 1, graph_file);

        this->resize(this->vertices_count, this->edges_count);

        fread(reinterpret_cast<void*>(src_ids), sizeof(int), this->edges_count, graph_file);
        fread(reinterpret_cast<void*>(dst_ids), sizeof(int), this->edges_count, graph_file);
        //fread(reinterpret_cast<void*>(weights), sizeof(float), this->edges_count, graph_file);

        fclose(graph_file);
        return true;
    }

    void convert_to_undirected()
    {
        int *new_src_ids = new int[2*edges_count];
        int *new_dst_ids = new int[2*edges_count];
        float *new_weights = new float[2*edges_count];

        #pragma omp parallel for
        for(long long i = 0; i < edges_count; i++)
        {
            new_src_ids[i] = src_ids[i];
            new_dst_ids[i] = dst_ids[i];
            new_weights[i] = weights[i];
            new_src_ids[i + edges_count] = dst_ids[i];
            new_dst_ids[i + edges_count] = src_ids[i];
            new_weights[i + edges_count] = weights[i];
        }

        edges_count = edges_count*2;
        delete[]src_ids;
        delete[]dst_ids;
        delete[]weights;

        src_ids = new_src_ids;
        dst_ids = new_dst_ids;
        weights = new_weights;
    }
};

struct GraphCSR
{
    long long *outgoing_ptrs; // |V|
    int *outgoing_ids; // |E|
    float *weights; // |E| elements - веса в данном алгоритме не используются
    int vertices_count; // = |V|
    long long edges_count; // = |E|
};

#endif /* graph_h */
