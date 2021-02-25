#ifndef KERNELS3D
#define KERNELS3D

#include <cuda_runtime.h>
#include "constants.h"

template<int D, class T, int y_l, int x_l, int z_l>
__device__
inline T stencil_fun_inline_ix_3d(const T arr[y_l][x_l][z_l], const int y_off, const int x_off, const int z_off){
    T sum_acc = 0;
    for (int i = 0; i < D*3; i += 3 ){
        const int y = y_off + threadIdx.y + ixs[i  ];
        const int x = x_off + threadIdx.x + ixs[i+1];
        const int z = z_off + threadIdx.z + ixs[i+2];
        sum_acc += arr[y][x][z];
    }
    return sum_acc / D;
}


template<int D, class T>
__global__
void global_reads_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned x_len,
    const unsigned y_len,
    const unsigned z_len)
{
    const int gidx = blockIdx.z*blockDim.z + threadIdx.z;
    const int gidy = blockIdx.y*blockDim.y + threadIdx.y;
    const int gidz = blockIdx.x*blockDim.x + threadIdx.x;

    const int gindex = gidx*y_len*z_len + gidy*z_len + gidz;
    const int y_len_maxIdx = y_len - 1;
    const int x_len_maxIdx = x_len - 1;
    const int z_len_maxIdx = z_len - 1;
    
    if (gidx < x_len && gidy < y_len && gidz < z_len)
    {
        T sum_acc = 0;
        for (int i = 0; i < D; i++)
        {
            const int x = BOUND(gidx + ixs[i*3  ],x_len_maxIdx);
            const int y = BOUND(gidy + ixs[i*3+1],y_len_maxIdx);
            const int z = BOUND(gidz + ixs[i*3+2],z_len_maxIdx);
            const int index = x*y_len*z_len + y*z_len + z; 
            sum_acc += A[index];
        }
        out[gindex] = sum_acc / (T)D;
    }
}
/*
template<int ixs_len, int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max, class T>
__global__
void big_tile_2d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
*/
#endif