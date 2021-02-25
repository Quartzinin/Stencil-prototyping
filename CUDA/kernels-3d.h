#ifndef KERNELS3D
#define KERNELS3D

#include <cuda_runtime.h>
#include "constants.h"

template<int D, class T, int z_l, int y_l, int x_l>
__device__
inline T stencil_fun_inline_ix_3d(const T arr[z_l][y_l][x_l], const int z_off, const int y_off, const int x_off){
    T sum_acc = 0;
    for (int i = 0; i < D; i++ ){
        const int z = z_off + ixs[i*3  ];
        const int y = y_off + ixs[i*3+1];
        const int x = x_off + ixs[i*3+2];
        sum_acc += arr[z][y][x];
    }
    return sum_acc / (T)D;
}


template<int D, class T>
__global__
void global_reads_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len, //outermost
    const unsigned y_len, //middle
    const unsigned x_len) //innermost
{
    const int gidz = blockIdx.z*blockDim.z + threadIdx.z;
    const int gidy = blockIdx.y*blockDim.y + threadIdx.y;
    const int gidx = blockIdx.x*blockDim.x + threadIdx.x;

    const int gindex = gidz*y_len*x_len + gidy*x_len + gidx;
    const int y_len_maxIdx = y_len - 1;
    const int x_len_maxIdx = x_len - 1;
    const int z_len_maxIdx = z_len - 1;
    
    if (gidx < x_len && gidy < y_len && gidz < z_len)
    {
        T sum_acc = 0;
        for (int i = 0; i < D; i++)
        {
            const int z = BOUND(gidz + ixs[i*3  ],z_len_maxIdx);
            const int y = BOUND(gidy + ixs[i*3+1],y_len_maxIdx);
            const int x = BOUND(gidx + ixs[i*3+2],x_len_maxIdx);
            const int index = z*y_len*x_len + y*x_len + x; 
            sum_acc += A[index];
        }
        out[gindex] = sum_acc / (T)D;
    }
}


template<int ixs_len, 
    int z_axis_min, int z_axis_max, 
    int y_axis_min, int y_axis_max, 
    int x_axis_min, int x_axis_max,
class T>
__global__
void small_tile_3d(
    const T* A,
    T* out,
    const unsigned z_len,
    const unsigned y_len,
    const unsigned x_len
    )
{
    const int x_block = SQ_BLOCKSIZE; \
    const int y_block = x_block/4; \
    const int z_block = x_block/y_block; \
    __shared__ T tile[z_block][y_block][x_block];
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int waste_z = z_axis_min + z_axis_max;
    const int gidx = blockIdx.x*(x_block - waste_x) + threadIdx.x - x_axis_min;
    const int gidy = blockIdx.y*(y_block - waste_y) + threadIdx.y - y_axis_min;
    const int gidz = blockIdx.z*(z_block - waste_z) + threadIdx.z - z_axis_min;
    const int gindex = gidz * y_len * x_len + gidy * x_len + gidx;
    const int max_x_idx = x_len - 1;
    const int max_y_idx = y_len - 1;
    const int max_z_idx = z_len - 1;

    const int x = BOUND(gidx, max_x_idx);
    const int y = BOUND(gidy, max_y_idx);
    const int z = BOUND(gidz, max_z_idx);

    const int index = z * y_len * x_len + y * x_len + x;

    tile[threadIdx.z][threadIdx.y][threadIdx.x] = A[index];

    __syncthreads();

    if (    (0 <= gidx && gidx < x_len)
        &&  (x_axis_min <= threadIdx.x && threadIdx.x < x_block - x_axis_max)
        &&  (0 <= gidy && gidy < y_len)
        &&  (y_axis_min <= threadIdx.y && threadIdx.y < y_block - y_axis_max)
        &&  (0 <= gidz && gidz < z_len)
        &&  (z_axis_min <= threadIdx.z && threadIdx.z < z_block - z_axis_max)
        )  
    {
        out[gindex] = stencil_fun_inline_ix_3d<ixs_len, T, z_block, y_block, x_block>
                                              (tile, threadIdx.z, threadIdx.y, threadIdx.x);
    }
}

template<int ixs_len, 
    int z_axis_min, int z_axis_max, 
    int y_axis_min, int y_axis_max, 
    int x_axis_min, int x_axis_max, 
    class T>
__global__
void big_tile_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len,
    const unsigned y_len,
    const unsigned x_len
    )
{
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int waste_z = z_axis_min + z_axis_max;
    const int block_x = SQ_BLOCKSIZE;
    const int block_y = block_x/4;
    const int block_z = block_x/block_y;

    const int block_offset_x = blockIdx.x*block_x;
    const int block_offset_y = blockIdx.y*block_y;
    const int block_offset_z = blockIdx.z*block_z;
    const int gidz = block_offset_z + threadIdx.z;
    const int gidy = block_offset_y + threadIdx.y;
    const int gidx = block_offset_x + threadIdx.x;
    const int gindex = gidz * y_len * x_len + gidy * x_len + gidx;
    const int max_z_idx = z_len - 1;
    const int max_y_idx = y_len - 1;
    const int max_x_idx = x_len - 1;

    const int shared_size_z = block_z + waste_z;
    const int shared_size_y = block_y + waste_y;
    const int shared_size_x = block_x + waste_x;

    __shared__ T tile[shared_size_z][shared_size_y][shared_size_x];

    const int z_iters = (shared_size_z + (block_z-1)) / block_z;
    const int y_iters = (shared_size_y + (block_y-1)) / block_y;
    const int x_iters = (shared_size_x + (block_x-1)) / block_x;

    for(int i = 0; i < z_iters; i++){
        for(int j = 0; j < y_iters; j++){
            for (int k = 0; k < x_iters; k++){
                const int local_z = threadIdx.z + i*block_z;
                const int local_y = threadIdx.y + j*block_y;
                const int local_x = threadIdx.x + k*block_x;
                
                if(local_x < shared_size_x && local_y < shared_size_y && local_z < shared_size_z){
                    const int gx = BOUND( local_x + block_offset_x - x_axis_min, max_x_idx);
                    const int gy = BOUND( local_y + block_offset_y - y_axis_min, max_y_idx);
                    const int gz = BOUND( local_z + block_offset_z - z_axis_min, max_z_idx);
                    const int index = gz * y_len * x_len + gy * x_len + gx;
                    tile[local_z][local_y][local_x] = A[index];
                }
            }
        }
    }
    __syncthreads();

    if((gidx < x_len) && (gidy < y_len) && (gidz < z_len))
    {
        out[gindex] = stencil_fun_inline_ix_3d<ixs_len, T, shared_size_z, shared_size_y, shared_size_x>
                                              (tile, threadIdx.z + z_axis_min, threadIdx.y + y_axis_min, threadIdx.x + x_axis_min);
    }
}



#endif