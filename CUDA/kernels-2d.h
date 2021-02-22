#ifndef CUDA_PROJ_HELPER
#define CUDA_PROJ_HELPER

#include <cuda_runtime.h>

#define BOUND_2D(i,j,max_ix,max_jx) (min((max_ix*max_jx-1),max(0,(i*max_jx + j))))

__constant__ int ixs[1024];

template<int D>
__device__
inline int stencil_fun2d(const int* arr){
    int sum_acc = 0;
    for (int i = 0; i < D; ++i){
        sum_acc += arr[i];
    }
    return sum_acc/(D);
}



template<int D>
__device__
void* stencil_global_mem(const int* A,
                         int* out
                         const int n_rows,
                         const int n_columns)
{
    for (int i = 0; i < D; ++i)
    {
        /* code */
    }
}



#endif

