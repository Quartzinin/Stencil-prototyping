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
__device__
void global_reads_3d(
    const T* A,
    T* out,
    const unsigned y_len,
    const unsigned x_len,
    const unsigned z_len)
{
    const int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int gidy = blockIdx.y*blockDim.y + threadIdx.y;
    const int gidz = blockIdx.z*blockDim.z + threadIdx.z;

    const int gindex = gidy*x_len + gidx*z_len + gidz;
    const int y_len_maxIdx = y_len - 1;
    const int x_len_maxIdx = x_len - 1;
    const int z_len_maxIdx = z_len - 1;

    if (gidy < y_len && gidx < x_len && gidz < z_len)
    {
        T sum_acc = 0;
        for (int i = 0; i < D*3; i+=3)
        {
            const int y = BOUND(gidy + ixs[i  ],y_len_maxIdx);
            const int x = BOUND(gidx + ixs[i+1],x_len_maxIdx);
            const int z = BOUND(gidz + ixs[i+2],z_len_maxIdx);
            const int index = y*x_len + x*z_len + z; 
            sum_acc += A[index];
        }
        out[gindex] = sum_acc/D;
    }
}

#endif