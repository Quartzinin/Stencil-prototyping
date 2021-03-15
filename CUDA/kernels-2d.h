#ifndef KERNELS2D
#define KERNELS2D

#include <cuda_runtime.h>
#include "constants.h"


/*
 * CONST INDICE VERSIONS:
 */

template<int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max>
__global__
void global_reads_2d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    const int gidx = blockIdx.x*SQ_BLOCKSIZE + threadIdx.x;
    const int gidy = blockIdx.y*SQ_BLOCKSIZE + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    if (gidx < row_len && gidy < col_len)
    {
        const int x_range = x_axis_max + x_axis_min + 1;
        const int y_range = y_axis_max + y_axis_min + 1;
        const int total_range = x_range * y_range;
        T sum_acc = 0;
        #pragma unroll
        for(int i=0; i < y_range; i++){
            const int y = BOUND(gidy + i - y_axis_min, max_y_ix);
            #pragma unroll
            for(int j=0; j < x_range; j++){
                const int x = BOUND(gidx + j - x_axis_min, max_x_ix);
                const int index = y * row_len + x;
                sum_acc += A[index];
            }
        }
        out[gindex] = sum_acc / (T)total_range;
    }
}

template<int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max>
__global__
void small_tile_2d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    __shared__ T tile[SQ_BLOCKSIZE][SQ_BLOCKSIZE];
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int gidx = blockIdx.x*(SQ_BLOCKSIZE - waste_x) + threadIdx.x - x_axis_min;
    const int gidy = blockIdx.y*(SQ_BLOCKSIZE - waste_y) + threadIdx.y - y_axis_min;
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
        const int x_range = x_axis_max + x_axis_min + 1;
        const int y_range = y_axis_max + y_axis_min + 1;
        const int total_range = x_range * y_range;
        T sum_acc = 0;
        #pragma unroll
        for(int i=0; i < y_range; i++){
            #pragma unroll
            for(int j=0; j < x_range; j++){
                const int y = threadIdx.y + i - y_axis_min;
                const int x = threadIdx.x + j - x_axis_min;
                sum_acc += tile[y][x];
            }
        }
        out[gindex] = sum_acc / (T)total_range;
    }
}

template<int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max>
__global__
void big_tile_2d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int block_offset_x = blockIdx.x*SQ_BLOCKSIZE;
    const int block_offset_y = blockIdx.y*SQ_BLOCKSIZE;
    const int gidx = block_offset_x + threadIdx.x;
    const int gidy = block_offset_y + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    const int shared_size_x = SQ_BLOCKSIZE + waste_x;
    const int shared_size_y = SQ_BLOCKSIZE + waste_y;
    __shared__ T tile[shared_size_y][shared_size_x];

    const int x_iters = (shared_size_x + (SQ_BLOCKSIZE-1)) / SQ_BLOCKSIZE;
    const int y_iters = (shared_size_y + (SQ_BLOCKSIZE-1)) / SQ_BLOCKSIZE;
    #pragma unroll
    for(int i = 0; i < y_iters; i++){
        const int local_y = threadIdx.y + i*SQ_BLOCKSIZE;
        const int gy = BOUND( local_y + block_offset_y - y_axis_min, max_y_ix)
                     * row_len;
        #pragma unroll
        for(int j = 0; j < x_iters; j++){
            const int local_x = threadIdx.x + j*SQ_BLOCKSIZE;
            const int gx = BOUND( local_x + block_offset_x - x_axis_min, max_x_ix);
            if(local_x < shared_size_x && local_y < shared_size_y){
                tile[local_y][local_x] = A[gx + gy];
            }
        }
    }
    __syncthreads();

    if((gidx < row_len) && (gidy < col_len))
    {
        const int x_range = x_axis_max + x_axis_min + 1;
        const int y_range = y_axis_max + y_axis_min + 1;
        const int total_range = x_range * y_range;
        T sum_acc = 0;
        #pragma unroll
        for(int i=0; i < y_range; i++){
            #pragma unroll
            for(int j=0; j < x_range; j++){
                const int y = threadIdx.y + i;
                const int x = threadIdx.x + j;
                sum_acc += tile[y][x];
            }
        }
        out[gindex] = sum_acc / (T)total_range;
    }
}

template<int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max>
__global__
void big_tile_2d_const_flat(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int block_offset_x = blockIdx.x*SQ_BLOCKSIZE;
    const int block_offset_y = blockIdx.y*SQ_BLOCKSIZE;
    const int gidx = block_offset_x + threadIdx.x;
    const int gidy = block_offset_y + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    const int shared_size_x = SQ_BLOCKSIZE + waste_x;
    const int shared_size_y = SQ_BLOCKSIZE + waste_y;
    
    const int flatIndex = threadIdx.y*SQ_BLOCKSIZE + threadIdx.x;

    const int shared_size = shared_size_x*shared_size_y;
    __shared__ T tile[shared_size];
    
    const int flatBlock = SQ_BLOCKSIZE*SQ_BLOCKSIZE;
    const int iters = CEIL_DIV(shared_size, flatBlock);

    #pragma unroll
    for(int i = 0; i < iters; i++){
        const int local_ix = flatIndex + i*flatBlock;
        const int local_x = local_ix % shared_size_x;
        const int local_y = (local_ix / shared_size_x);

        const int gx = BOUND( local_x + block_offset_x - x_axis_min, max_x_ix);
        const int gy = BOUND( local_y + block_offset_y - y_axis_min, max_y_ix)
                     * row_len;

        if(local_ix < shared_size){
            tile[local_ix] = A[gx + gy];
        }
    }
    __syncthreads();

    if((gidx < row_len) && (gidy < col_len))
    {
        const int x_range = x_axis_max + x_axis_min + 1;
        const int y_range = y_axis_max + y_axis_min + 1;
        const int total_range = x_range * y_range;
        T sum_acc = 0;
        #pragma unroll
        for(int i=0; i < y_range; i++){
            #pragma unroll
            for(int j=0; j < x_range; j++){
                const int y = threadIdx.y + i;
                const int x = threadIdx.x + j;
                sum_acc += tile[shared_size_x*y + x];
            }
        }
        out[gindex] = sum_acc / (T)total_range;
    }
}


template<int ixs_len>
__global__
void global_reads_2d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    const int gidx = blockIdx.x*SQ_BLOCKSIZE + threadIdx.x;
    const int gidy = blockIdx.y*SQ_BLOCKSIZE + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    if (gidx < row_len && gidy < col_len)
    {
        T sum_acc = 0;
        #pragma unroll
        for (int i = 0; i < ixs_len; i++ ){
            const int y = BOUND(gidy + ixs_2d[i].y, max_y_ix);
            const int x = BOUND(gidx + ixs_2d[i].x, max_x_ix);
            const int index = y * row_len + x;
            sum_acc += A[index];
        }
        out[gindex] = sum_acc / ixs_len;
    }
}


/*
template<int ixs_len, int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max>
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
    const int gidx = blockIdx.x*(SQ_BLOCKSIZE - waste_x) + threadIdx.x - x_axis_min;
    const int gidy = blockIdx.y*(SQ_BLOCKSIZE - waste_y) + threadIdx.y - y_axis_min;
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
        T sum_acc = 0;
        #pragma unroll
        for (int i = 0; i < ixs_len; i++ ){
            const int y = threadIdx.y + ixs_2d[i].y;
            const int x = threadIdx.x + ixs_2d[i].x;
            sum_acc += tile[y][x];
        }
        out[gindex] = sum_acc / (T)ixs_len;

    }
}

template<int ixs_len, int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max>
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
    const int block_offset_x = blockIdx.x*SQ_BLOCKSIZE;
    const int block_offset_y = blockIdx.y*SQ_BLOCKSIZE;
    const int gidx = block_offset_x + threadIdx.x;
    const int gidy = block_offset_y + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    const int shared_size_x = SQ_BLOCKSIZE + waste_x;
    const int shared_size_y = SQ_BLOCKSIZE + waste_y;
    __shared__ T tile[shared_size_y][shared_size_x];

    const int x_iters = (shared_size_x + (SQ_BLOCKSIZE-1)) / SQ_BLOCKSIZE;
    const int y_iters = (shared_size_y + (SQ_BLOCKSIZE-1)) / SQ_BLOCKSIZE;
    #pragma unroll
    for(int i = 0; i < y_iters; i++){
        #pragma unroll
        for(int j = 0; j < x_iters; j++){
            const int local_y = threadIdx.y + i*SQ_BLOCKSIZE;
            const int local_x = threadIdx.x + j*SQ_BLOCKSIZE;
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
        T sum_acc = 0;
        #pragma unroll
        for (int i = 0; i < ixs_len; i++ ){
            const int y = threadIdx.y + y_axis_min + ixs_2d[i].y;
            const int x = threadIdx.x + x_axis_min + ixs_2d[i].x;
            sum_acc += tile[y][x];
        }
        out[gindex] = sum_acc / (T)ixs_len;
    }
}
*/

#endif



