//
//  verification.h
//  
//
//  Created by Elijah Afanasiev on 23.09.2018.
//
//

#ifndef verification_h
#define verification_h

#include "graph_generation_API.h"
#include "bfs.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool equal_results(int *_user_result, int *_reference_result, int _vertices_count)
{
    int error_count = 0;
    for(int i = 0; i < _vertices_count; i++)
    {
        if(_user_result[i] != _reference_result[i])
        {
            if(error_count < 20)
                cout << "Error: " << _user_result[i] << " vs " << _reference_result[i] << " in pos " << i << endl;
            error_count++;
        }
    }
    //cout << "error count: " << error_count << endl;
    if(error_count > 0)
        return 0;
    else
        return 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void verify_result(GraphCSR _graph, int *_user_result, int _source_vertex)
{
    int *reference_result = new int[_graph.vertices_count];

    cpu_bfs(_graph, reference_result, _source_vertex);
    
    cout << "#verification: " << equal_results(_user_result, reference_result, _graph.vertices_count) << endl;
    
    delete []reference_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* verification_h */
