#include "converter.h"
#include "cuda_error_hadling.h"
#include "verification.h"

#define TEST_ITERATION_COUNT 200

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<string> split(const string& str, int delimiter(int) = ::isspace) {
  vector<string> result;
  auto e = str.end();
  auto i = str.begin();
  while (i != e) {
    i = find_if_not(i, e, delimiter);
    if (i == e)
      break;
    auto j = find_if(i, e, delimiter);
    result.push_back(string(i, j));
    i = j;
  }
  return result;
}

void parse_cmd_params(int _argc,
                      char** _argv,
                      int& _scale,
                      int& _avg_degree,
                      bool& _check,
                      string& _graph_type,
                      bool& _load_from_file,
                      string& _file_name,
                      bool& _convert,
                      string& _convert_name,
                      int& _iterations) {
  // get params from cmd line
  string all_params;
  for (int i = 1; i < _argc; i++) {
    string option(_argv[i]);
    all_params += option + " ";
  }

  std::vector<std::string> vstrings = split(all_params);

  for (int i = 0; i < vstrings.size(); i++) {
    string option = vstrings[i];

    // cout << "option: " << option << endl;

    if (option.compare("-s") == 0) {
      _scale = atoi(vstrings[++i].c_str());
    }

    if (option.compare("-e") == 0) {
      _avg_degree = atoi(vstrings[++i].c_str());
    }

    if (option.compare("-nocheck") == 0) {
      _check = false;
    }

    if (option.compare("-rmat") == 0) {
      _graph_type = "rmat";
    }

    if (option.compare("-random_uniform") == 0) {
      _graph_type = "random_uniform";
    }

    if (option.compare("-file") == 0 || option.compare("-load") == 0) {
      _load_from_file = true;
      _file_name = vstrings[++i].c_str();
    }

    if (option.compare("-convert") == 0) {
      _convert = true;
      _convert_name = vstrings[++i].c_str();
    }

    if (option.compare("-it") == 0) {
      _iterations = atoi(vstrings[++i].c_str());
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void user_copy_graph_to_device(GraphCSR& _cpu_graph, GraphCSR& _gpu_graph) {
  // FIXME: implement allocation of device pointers and
  _gpu_graph.vertices_count = _cpu_graph.vertices_count;
  _gpu_graph.edges_count = _cpu_graph.edges_count;

  SAFE_CALL(cudaMalloc((void**)&_gpu_graph.outgoing_ptrs,
                       sizeof(long long) * (_gpu_graph.vertices_count + 1)));
  SAFE_CALL(cudaMalloc((void**)&_gpu_graph.outgoing_ids,
                       sizeof(int) * _gpu_graph.edges_count));

  SAFE_CALL(cudaMemcpy(_gpu_graph.outgoing_ptrs, _cpu_graph.outgoing_ptrs,
                       sizeof(long long) * (_gpu_graph.vertices_count + 1),
                       cudaMemcpyHostToDevice));
  SAFE_CALL(cudaMemcpy(_gpu_graph.outgoing_ids, _cpu_graph.outgoing_ids,
                       _gpu_graph.edges_count * sizeof(int),
                       cudaMemcpyHostToDevice));
}

void user_algorithm(GraphCSR _graph, int* _levels, int _source_vertex) {
  // FIXME: implement GPU computations here
  int vertices_count = _graph.vertices_count;
  // cpu_bellman_ford_edges_list(_graph, _result, _source_vertex);

  int* device_levels;
  SAFE_CALL(cudaMalloc((void**)&device_levels, vertices_count * sizeof(int)));

  // compute shortest paths on GPU
  gpu_bfs_wrapper(_graph.outgoing_ptrs, _graph.outgoing_ids,
                  _graph.vertices_count, _graph.edges_count, _source_vertex,
                  device_levels);

  // copy results back to host
  SAFE_CALL(cudaMemcpy(_levels, device_levels, vertices_count * sizeof(int),
                       cudaMemcpyDeviceToHost));

  SAFE_CALL(cudaFree(device_levels));
}

void free_memory(GraphCSR _gpu_graph) {
  cudaFree(_gpu_graph.outgoing_ptrs);
  cudaFree(_gpu_graph.outgoing_ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct TempEdgeData {
  int dst_id;
  float weight;
  TempEdgeData(int _dst_id, float _weight) {
    dst_id = _dst_id;
    weight = _weight;
  };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void convert_edges_list_to_CSR(Graph& _edges_list_graph, GraphCSR& _csr_graph) {
  _csr_graph.vertices_count = _edges_list_graph.vertices_count;
  _csr_graph.edges_count = _edges_list_graph.edges_count;
  int vertices_count = _edges_list_graph.vertices_count;
  long long edges_count = _edges_list_graph.edges_count;

  vector<vector<TempEdgeData> > tmp_graph(_csr_graph.vertices_count);

  int* old_src_ids = _edges_list_graph.src_ids;
  int* old_dst_ids = _edges_list_graph.dst_ids;
  float* old_weights = _edges_list_graph.weights;

  for (long long int i = 0; i < edges_count; i++) {
    int src_id = old_src_ids[i];
    int dst_id = old_dst_ids[i];
    float weight = old_weights[i];

    tmp_graph[src_id].push_back(TempEdgeData(dst_id, weight));
  }

  _csr_graph.outgoing_ptrs = new long long[vertices_count];
  _csr_graph.outgoing_ids = new int[edges_count];
  _csr_graph.weights = new float[edges_count];

  // save optimised graph
  long long current_edge = 0;
  for (int cur_vertex = 0; cur_vertex < vertices_count; cur_vertex++) {
    int src_id = cur_vertex;
    _csr_graph.outgoing_ptrs[src_id] = current_edge;
    for (int i = 0; i < tmp_graph[src_id].size(); i++) {
      _csr_graph.outgoing_ids[current_edge] = tmp_graph[src_id][i].dst_id;
      _csr_graph.weights[current_edge] = tmp_graph[src_id][i].weight;
      current_edge++;
    }
  }
  _csr_graph.outgoing_ptrs[vertices_count] = edges_count;

#ifdef __PRINT_CSR_FORMAT__
  for (int i = 0; i < vertices_count + 1; i++)
    cout << _csr_graph.outgoing_ptrs[i] << " ";
  cout << endl << endl;
  for (int i = 0; i < edges_count; i++)
    cout << _csr_graph.outgoing_ids[i] << " ";
  cout << endl << endl;
  for (int i = 0; i < vertices_count; i++) {
    const long long edge_start =
        _csr_graph
            .outgoing_ptrs[i];  // получаем положение первого ребра вершины
    const int connections_count =
        _csr_graph.outgoing_ptrs[i + 1] -
        _csr_graph.outgoing_ptrs[i];  // получаем число смежных ребер вершины

    cout << "[";
    for (int edge_pos = 0; edge_pos < connections_count;
         edge_pos++)  // для каждого смежного ребра делаем:
    {
      int dst_id = _csr_graph.outgoing_ids[edge_start + edge_pos];
      cout << dst_id << " ";
    }
    cout << "] ";
  }
  cout << endl << endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  try {
    double t1 = omp_get_wtime();
    // считываем параметры командной сторки
    int scale = 12;
    int avg_degree = 15;
    string graph_type = "rmat";
    bool check = true;
    bool load_from_file = false;
    string file_name = "none";
    bool convert = false;
    string convert_name = "none";
    int iterations = 10;

    cout << "printing argv" << endl;
    for (int i = 0; i < argc; ++i)
      cout << argv[i] << endl;
    cout << "done" << endl;

    parse_cmd_params(argc, argv, scale, avg_degree, check, graph_type,
                     load_from_file, file_name, convert, convert_name,
                     iterations);
    cout << "cmd parameters parsed" << endl;

    Graph graph;

    if (convert) {
      cout << "convert mode: " << convert << endl;
      convert_real_graph(graph, convert_name, true);
      graph.save_to_binary_file(convert_name + ".el_graph");
      return 0;
    }

    if (load_from_file) {
      cout << "loading graph " << file_name << endl;
      graph.load_from_binary_file(file_name);
      cout << "loaded graph has " << graph.vertices_count << " vertices and "
           << graph.edges_count << " edges" << endl;
    } else {
      cout << "generating new graph" << endl;
      cout << "scale: " << scale << endl;
      cout << "avg_degree: " << avg_degree << endl;

      // генерируем граф
      if (graph_type == "rmat") {
        file_name = "rmat_" + std::to_string(scale) + "_" +
                    std::to_string(avg_degree) + ".el_graph";
        generate_R_MAT(graph, pow(2.0, scale), avg_degree);
      } else if (graph_type == "random_uniform") {
        file_name = "ru_" + std::to_string(scale) + "_" +
                    std::to_string(avg_degree) + ".el_graph";
        generate_random_uniform(graph, pow(2.0, scale), avg_degree);
      } else {
        cout << "Unknown graph type" << endl;
        return 1;
      }

      cout << "graph generated" << endl;
      graph.convert_to_undirected();

      cout << "conversion done" << endl;
      cout << "file name: " << file_name << endl;

      if (file_name != "none") {
        graph.save_to_binary_file(file_name);
        cout << "saved to " << file_name << " file" << endl;
      }
    }
    double t2 = omp_get_wtime();
    cout << "generation/load time: " << t2 - t1 << " sec" << endl;

    // выделяем память под ответ
    int* user_result = new int[graph.vertices_count];

    // преобразовываем граф
    GraphCSR csr_graph;
    cout << "conversion started" << endl;
    convert_edges_list_to_CSR(graph, csr_graph);
    cout << "converted" << endl;

    int* tmp;
    cudaMalloc((void**)&tmp, sizeof(int));
    cout << "test malloc done" << endl;

    double t_start = omp_get_wtime();
    // запускаем копирования данных
    GraphCSR gpu_graph;
    t1 = omp_get_wtime();
    user_copy_graph_to_device(csr_graph, gpu_graph);
    t2 = omp_get_wtime();
    cout << "Device->host copy time: " << t2 - t1 << " sec" << endl;

    // запускаем алгоритм
    cudaDeviceSynchronize();
    t1 = omp_get_wtime();
    int last_source = 0;
    cout << "will do " << iterations << " iterations" << endl;
    for (int i = 0; i < iterations; i++) {
      last_source = rand() % graph.vertices_count;
      user_algorithm(gpu_graph, user_result, last_source);
    }
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    double t_end = omp_get_wtime();
    cout << "BFS wall time: " << t2 - t1 << " sec" << endl;

    free_memory(gpu_graph);

    cout << endl;
    cout << "#algorithm executed!" << endl;
    cout << "#perf: "
         << ((double)(iterations)*graph.edges_count) / ((t_end - t_start) * 1e6)
         << endl;
    cout << "#time: " << t_end - t_start << endl;
    cout << "#check: " << check << endl;

    // делаем проверку корректности каждый раз
    if (check) {
      verify_result(csr_graph, user_result, last_source);
    }

    // освобождаем память
    delete[] user_result;

  } catch (const char* error) {
    cout << error << endl;
  } catch (...) {
    cout << "unknown error" << endl;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
