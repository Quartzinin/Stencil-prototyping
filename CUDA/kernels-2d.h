#ifndef KERNELS2D
#define KERNELS2D

#include <cuda_runtime.h>
#include "constants.h"

#define BOUND_2D(i,j,max_ix,max_jx) (min((max_ix*max_jx-1),max(0,(i*max_jx + j))))

template<int D, class T, int y_l, int x_l>
__device__
inline T stencil_fun_inline_ix_2d(const T arr[y_l][x_l], const int y_off, const int x_off){
    T sum_acc = 0;
    for (int i = 0; i < D*2; i += 2 ){
        const int y = y_off + ixs[i  ];
        const int x = x_off + ixs[i+1];
        sum_acc += arr[y][x];
    }
    return sum_acc / (T)D;
}

template<int ixs_len, class T>
__global__
void global_reads_2d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    const int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int gidy = blockIdx.y*blockDim.y + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    if (gidx < row_len && gidy < col_len)
    {
        T sum_acc = 0;
        for (int i = 0; i < ixs_len*2; i += 2 ){
            const int y = BOUND(gidy + ixs[i  ], max_y_ix);
            const int x = BOUND(gidx + ixs[i+1], max_x_ix);
            const int index = y * row_len + x;
            sum_acc += A[index];
        }
        out[gindex] = sum_acc / ixs_len;
    }
}

template<int ixs_len, int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max, class T>
__global__
void small_tile_2d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    __shared__ T tile[SQ_BLOCKSIZE][SQ_BLOCKSIZE];
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int gidx = blockIdx.x*(blockDim.x - waste_x) + threadIdx.x - x_axis_min;
    const int gidy = blockIdx.y*(blockDim.y - waste_y) + threadIdx.y - y_axis_min;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;
    const int x = BOUND(gidx, max_x_ix);
    const int y = BOUND(gidy, max_y_ix);
    const int index = y * row_len + x;
    tile[threadIdx.y][threadIdx.x] = A[index];
    __syncthreads();

    if (    (0 <= gidx && gidx < row_len)
        &&  (x_axis_min <= threadIdx.x && threadIdx.x < SQ_BLOCKSIZE - x_axis_max)
        &&  (0 <= gidy && gidy < col_len)
        &&  (y_axis_min <= threadIdx.y && threadIdx.y < SQ_BLOCKSIZE - y_axis_max)
        )
    {
        out[gindex] = stencil_fun_inline_ix_2d<ixs_len, T, SQ_BLOCKSIZE,SQ_BLOCKSIZE>
                                              (tile, threadIdx.y ,threadIdx.x);
    }
}

template<int ixs_len, int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max, class T>
__global__
void big_tile_2d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int block_offset_x = blockIdx.x*blockDim.x;
    const int block_offset_y = blockIdx.y*blockDim.y;
    const int gidx = block_offset_x + threadIdx.x;
    const int gidy = block_offset_y + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    const int shared_size_x = SQ_BLOCKSIZE + waste_x;
    const int shared_size_y = SQ_BLOCKSIZE + waste_y;
    __shared__ T tile[shared_size_y][shared_size_x];

    for(int local_y = threadIdx.y; local_y < shared_size_y; local_y += blockDim.y){
        for(int local_x = threadIdx.x; local_x < shared_size_x; local_x += blockDim.x){
            if(local_x < shared_size_x && local_y < shared_size_y){
                const int gx = BOUND( local_x + block_offset_x - x_axis_min, max_x_ix);
                const int gy = BOUND( local_y + block_offset_y - y_axis_min, max_y_ix);
                const int index = gy * row_len + gx;
                tile[local_y][local_x] = A[index];
            }
        }
    }
    __syncthreads();

    if((gidx < row_len) && (gidy < col_len))
    {
        out[gindex] = stencil_fun_inline_ix_2d<ixs_len, T, shared_size_y, shared_size_x>
                                              (tile, threadIdx.y + y_axis_min, threadIdx.x + x_axis_min);
    }
}


#endif

