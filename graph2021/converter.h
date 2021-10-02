#pragma once
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>


using namespace std;



void convert_real_graph(Graph &_graph, string _txt_file_name, bool _append_with_reverse_edges)
{
    ifstream infile;
    infile.open(_txt_file_name.c_str());
    if (!infile.is_open())
        throw "can't open file during convert";

    int vertices_count = 0;
    long long edges_count = 0;
    bool directed = true;
    string line;
    getline(infile, line); // read first line
    if(line == string("asym positive")) // try to understand which header it is
    {
        getline(infile, line); // get edges and vertices count line
        istringstream vert_iss(line);
        vert_iss >> edges_count >> vertices_count;
    }
    else if(line == string("% asym unweighted"))
    {
        cout << "asym positive detected" << endl;
        getline(infile, line);
    }
    else if(line == string("% bip unweighted"))
    {
        cout << "bip unweighted" << endl;
        directed = false;
    }
    else if(line == string("% sym unweighted"))
    {
        cout << "sym unweighted" << endl;
        directed = false;
        getline(infile, line); // get edges and vertices count line
        istringstream vert_iss(line);
        vert_iss >> edges_count >> vertices_count;
        cout << "edges: " << edges_count << " and vertices: " << vertices_count << endl;
    }
    else
    {
        getline(infile, line); // skip second line
        getline(infile, line); // get vertices and edges count line
        istringstream vert_iss(line);
        vert_iss >> vertices_count >> edges_count;
        getline(infile, line); // skip forth line
    }
    cout << "vc: " << vertices_count << " ec: " << edges_count << endl;

    vector<int>tmp_src_ids;
    vector<int>tmp_dst_ids;

    long long i = 0;
    while (getline(infile, line))
    {
        istringstream iss(line);
        int src_id = 0, dst_id = 0;
        if (!(iss >> src_id >> dst_id))
        {
            continue;
        }

        if(src_id >= vertices_count)
            vertices_count = src_id + 1;

        if(dst_id >= vertices_count)
            vertices_count = dst_id + 1;

        tmp_src_ids.push_back(src_id);
        tmp_dst_ids.push_back(dst_id);

        if(!directed)
        {
            tmp_src_ids.push_back(dst_id);
            tmp_dst_ids.push_back(src_id);
        }
        i++;

        /*if((edges_count != 0) && (i > edges_count))
        {
            throw "ERROR: graph file is larger than expected";
        }*/
    }

    cout << "loaded " << vertices_count << " vertices_count" << endl;
    cout << "loaded " << i << " edges, expected amount " << edges_count << endl;

    edges_count = i;

    _graph.resize(vertices_count, edges_count);
    int seed = int(time(NULL));
    for(i = 0; i < edges_count; i++)
    {
        _graph.src_ids[i] = tmp_src_ids[i];
        _graph.dst_ids[i] = tmp_dst_ids[i];
    }

    // validate
    for(i = 0; i < edges_count; i++)
    {
        int src_id = _graph.src_ids[i];
        int dst_id = _graph.dst_ids[i];
        if((src_id >= vertices_count) || (src_id < 0))
        {
            cout << "error src: " << src_id << endl;
            throw "Error: incorrect src id on conversion";
        }
        if((dst_id >= vertices_count) || (dst_id < 0))
        {
            cout << "error dst: " << dst_id << endl;
            throw "Error: incorrect dst id on conversion";
        }
    }

    infile.close();
}