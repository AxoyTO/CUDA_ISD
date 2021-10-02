/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T rand_uniform_val(int _upper_border)
{
    return (_T)(rand() % _upper_border);
}

template <>
int rand_uniform_val(int _upper_border)
{
    return (int)(rand() % _upper_border);
}

template <>
float rand_uniform_val(int _upper_border)
{
    return (float)(rand() % _upper_border) / _upper_border;
}

template <>
double rand_uniform_val(int _upper_border)
{
    return (double)(rand() % _upper_border) / _upper_border;
}

float rand_flt()
{
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

void generate_random_uniform(Graph &graph, int _vertices_count, int _average_degree)
{
	// check input parameters correctness
	if (_average_degree > _vertices_count)
		throw "average connections in graph is greater than number of vertices";

	long long edges_count = (long long)_vertices_count * _average_degree;

	graph.resize(_vertices_count, edges_count);
    
    int _omp_threads = omp_get_max_threads();
    cout << "threads: " << _omp_threads << endl;
    
    #pragma omp parallel for
	for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
	{
		int from = rand() % _vertices_count;
		int to = rand() % _vertices_count;
        
        graph.src_ids[cur_edge] = from;
        graph.src_ids[cur_edge] = to;
        graph.weights[cur_edge] = rand_flt();
	}
    
    return graph;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void generate_R_MAT(Graph &graph, int _vertices_count, int _average_connections)
{
    int n = (int)log2(_vertices_count);
    int vertices_count = _vertices_count;
    
    int _a_prob = 57;
    int _b_prob = 19;
    int _c_prob = 19;
    int _d_prob = 5;
    
    long long edges_count = ((long long)_vertices_count) * _average_connections;
    
    graph.resize(_vertices_count, edges_count);
    
    //int _omp_threads = omp_get_max_threads();
    
    // generate and add edges to graph
    unsigned int seed = 0;
    #pragma omp parallel private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
        {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (long long i = 1; i < n; i++)
            {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;
            
                int step = (int)pow(2, n - (i + 1));
            
                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end)
                {
                    x_middle -= step, y_middle -= step;
                }
                else if (b_beg <= probability && probability < b_end)
                {
                    x_middle -= step, y_middle += step;
                }
                else if (c_beg <= probability && probability < c_end)
                {
                    x_middle += step, y_middle -= step;
                }
                else if (d_beg <= probability && probability < d_end)
                {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;
        
            int from = x_middle;
            int to = y_middle;
        
            graph.src_ids[cur_edge] = from;
            graph.dst_ids[cur_edge] = to;
            graph.weights[cur_edge] = rand_flt();
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
