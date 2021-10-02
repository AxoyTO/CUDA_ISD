#pragma once

#define find_min(a,b)            (((a) < (b)) ? (a) : (b))
#define find_max(a,b)            (((a) > (b)) ? (a) : (b))

#include <iostream>
#include <vector>
#include <omp.h>
#include <map>
#include <math.h>
#include "graph.h"


using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Graph generate_random_uniform(int _vertices_count, int _average_degree);

Graph generate_R_MAT(int _vertices_count, int _average_connections);

Graph generate_SSCA2(int _vertices_count, int _max_clique_size);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
