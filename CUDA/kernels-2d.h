#ifndef KERNELS2D
#define KERNELS2D

#include <cuda_runtime.h>
#include "constants.h"


/*
 * Inlined indices using a provided associative and commutative operator with a neutral element.
 */

template<long x_axis_min, long x_axis_max, long y_axis_min, long y_axis_max>
__global__
void global_reads_2d_inline_reduce(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long row_len,
    const long col_len
    )
{
    const long gidx = blockIdx.x*SQ_BLOCKSIZE + threadIdx.x;
    const long gidy = blockIdx.y*SQ_BLOCKSIZE + threadIdx.y;
    const long gindex = gidy * row_len + gidx;
    const long max_x_ix = row_len - 1;
    const long max_y_ix = col_len - 1;

    if (gidx < row_len && gidy < col_len)
    {
        const long x_range = x_axis_max + x_axis_min + 1;
        const long y_range = y_axis_max + y_axis_min + 1;
        const long total_range = x_range * y_range;
        T sum_acc = 0;
        #pragma unroll
        for(long i=0; i < y_range; i++){
            const long y = BOUNDL(gidy + (i - y_axis_min), max_y_ix);
            #pragma unroll
            for(long j=0; j < x_range; j++){
                const long x = BOUNDL(gidx + (j - x_axis_min), max_x_ix);
                const long index = y * row_len + x;
                sum_acc += A[index];
            }
        }
        out[gindex] = sum_acc / (T)total_range;
    }
}

template<long x_axis_min, long x_axis_max, long y_axis_min, long y_axis_max>
__global__
void small_tile_2d_inline_reduce(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long row_len,
    const long col_len
    )
{
    __shared__ T tile[SQ_BLOCKSIZE][SQ_BLOCKSIZE];
    const long waste_x = x_axis_min + x_axis_max;
    const long waste_y = y_axis_min + y_axis_max;
    const long gidx = blockIdx.x*(SQ_BLOCKSIZE - waste_x) + threadIdx.x - x_axis_min;
    const long gidy = blockIdx.y*(SQ_BLOCKSIZE - waste_y) + threadIdx.y - y_axis_min;
    const long gindex = gidy * row_len + gidx;
    const long max_x_ix = row_len - 1;
    const long max_y_ix = col_len - 1;
    const long x = BOUNDL(gidx, max_x_ix);
    const long y = BOUNDL(gidy, max_y_ix);
    const long index = y * row_len + x;
    tile[threadIdx.y][threadIdx.x] = A[index];
    __syncthreads();

    if (    (0 <= gidx && gidx < row_len)
        &&  (x_axis_min <= threadIdx.x && threadIdx.x < SQ_BLOCKSIZE - x_axis_max)
        &&  (0 <= gidy && gidy < col_len)
        &&  (y_axis_min <= threadIdx.y && threadIdx.y < SQ_BLOCKSIZE - y_axis_max)
        )
    {
        const long x_range = x_axis_max + x_axis_min + 1;
        const long y_range = y_axis_max + y_axis_min + 1;
        const long total_range = x_range * y_range;
        T sum_acc = 0;

        for(long i=0; i < y_range; i++){
            for(long j=0; j < x_range; j++){
                const long y = threadIdx.y + (i - y_axis_min);
                const long x = threadIdx.x + (j - x_axis_min);
                sum_acc += tile[y][x];
            }
        }
        out[gindex] = sum_acc / (T)total_range;
    }
}

template<long x_axis_min, long x_axis_max, long y_axis_min, long y_axis_max>
__global__
void big_tile_2d_inline_reduce(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long row_len,
    const long col_len
    )
{
    const long waste_x = x_axis_min + x_axis_max;
    const long waste_y = y_axis_min + y_axis_max;
    const long block_offset_x = blockIdx.x*SQ_BLOCKSIZE;
    const long block_offset_y = blockIdx.y*SQ_BLOCKSIZE;
    const long gidx = block_offset_x + threadIdx.x;
    const long gidy = block_offset_y + threadIdx.y;
    const long gindex = gidy * row_len + gidx;
    const long max_x_ix = row_len - 1;
    const long max_y_ix = col_len - 1;

    const long shared_size_x = SQ_BLOCKSIZE + waste_x;
    const long shared_size_y = SQ_BLOCKSIZE + waste_y;
    __shared__ T tile[shared_size_y][shared_size_x];

    const long x_iters = (shared_size_x + (SQ_BLOCKSIZE-1)) / SQ_BLOCKSIZE;
    const long y_iters = (shared_size_y + (SQ_BLOCKSIZE-1)) / SQ_BLOCKSIZE;

    for(long i = 0; i < y_iters; i++){
        const long local_y = threadIdx.y + i*SQ_BLOCKSIZE;
        const long gy = BOUNDL( local_y + block_offset_y - y_axis_min, max_y_ix)
                     * row_len;

        for(long j = 0; j < x_iters; j++){
            const long local_x = threadIdx.x + j*SQ_BLOCKSIZE;
            const long gx = BOUNDL( local_x + block_offset_x - x_axis_min, max_x_ix);
            if(local_x < shared_size_x && local_y < shared_size_y){
                tile[local_y][local_x] = A[gx + gy];
            }
        }
    }
    __syncthreads();

    if((gidx < row_len) && (gidy < col_len))
    {
        const long x_range = x_axis_max + x_axis_min + 1;
        const long y_range = y_axis_max + y_axis_min + 1;
        const long total_range = x_range * y_range;
        T sum_acc = 0;

        for(long i=0; i < y_range; i++){

            for(long j=0; j < x_range; j++){
                const long y = threadIdx.y + i;
                const long x = threadIdx.x + j;
                sum_acc += tile[y][x];
            }
        }
        out[gindex] = sum_acc / (T)total_range;
    }
}

template<long x_axis_min, long x_axis_max, long y_axis_min, long y_axis_max>
__global__
void big_tile_2d_inline_reduce_flat(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long row_len,
    const long col_len
    )
{
    const long waste_x = x_axis_min + x_axis_max;
    const long waste_y = y_axis_min + y_axis_max;
    const long block_offset_x = blockIdx.x*SQ_BLOCKSIZE;
    const long block_offset_y = blockIdx.y*SQ_BLOCKSIZE;
    const long gidx = block_offset_x + threadIdx.x;
    const long gidy = block_offset_y + threadIdx.y;
    const long gindex = gidy * row_len + gidx;
    const long max_x_ix = row_len - 1;
    const long max_y_ix = col_len - 1;

    const long shared_size_x = SQ_BLOCKSIZE + waste_x;
    const long shared_size_y = SQ_BLOCKSIZE + waste_y;

    const long flatIndex = threadIdx.y*SQ_BLOCKSIZE + threadIdx.x;

    const long shared_size = shared_size_x*shared_size_y;
    __shared__ T tile[shared_size];

    const long flatBlock = SQ_BLOCKSIZE*SQ_BLOCKSIZE;
    const long iters = CEIL_DIV(shared_size, flatBlock);


    for(long i = 0; i < iters; i++){
        const long local_ix = flatIndex + i*flatBlock;
        const long local_x = local_ix % shared_size_x;
        const long local_y = (local_ix / shared_size_x);

        const long gx = BOUNDL( local_x + block_offset_x - x_axis_min, max_x_ix);
        const long gy = BOUNDL( local_y + block_offset_y - y_axis_min, max_y_ix)
                     * row_len;

        if(local_ix < shared_size){
            tile[local_ix] = A[gx + gy];
        }
    }
    __syncthreads();

    if((gidx < row_len) && (gidy < col_len))
    {
        const long x_range = x_axis_max + x_axis_min + 1;
        const long y_range = y_axis_max + y_axis_min + 1;
        const long total_range = x_range * y_range;
        T sum_acc = 0;

        for(long i=0; i < y_range; i++){

            for(long j=0; j < x_range; j++){
                const long y = threadIdx.y + i;
                const long x = threadIdx.x + j;
                sum_acc += tile[shared_size_x*y + x];
            }
        }
        out[gindex] = sum_acc / (T)total_range;
    }
}


/*
template<long ixs_len>
__global__
void global_reads_2d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long row_len,
    const long col_len
    )
{
    const long gidx = blockIdx.x*SQ_BLOCKSIZE + threadIdx.x;
    const long gidy = blockIdx.y*SQ_BLOCKSIZE + threadIdx.y;
    const long gindex = gidy * row_len + gidx;
    const long max_x_ix = row_len - 1;
    const long max_y_ix = col_len - 1;

    if (gidx < row_len && gidy < col_len)
    {
        T sum_acc = 0;

        for (long i = 0; i < ixs_len; i++ ){
            const long y = BOUNDL(gidy + ixs_2d[i].y, max_y_ix);
            const long x = BOUNDL(gidx + ixs_2d[i].x, max_x_ix);
            const long index = y * row_len + x;
            sum_acc += A[index];
        }
        out[gindex] = sum_acc / ixs_len;
    }
}


template<long ixs_len, long x_axis_min, long x_axis_max, long y_axis_min, long y_axis_max>
__global__
void small_tile_2d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long row_len,
    const long col_len
    )
{
    __shared__ T tile[SQ_BLOCKSIZE][SQ_BLOCKSIZE];
    const long waste_x = x_axis_min + x_axis_max;
    const long waste_y = y_axis_min + y_axis_max;
    const long gidx = blockIdx.x*(SQ_BLOCKSIZE - waste_x) + threadIdx.x - x_axis_min;
    const long gidy = blockIdx.y*(SQ_BLOCKSIZE - waste_y) + threadIdx.y - y_axis_min;
    const long gindex = gidy * row_len + gidx;
    const long max_x_ix = row_len - 1;
    const long max_y_ix = col_len - 1;
    const long x = BOUNDL(gidx, max_x_ix);
    const long y = BOUNDL(gidy, max_y_ix);
    const long index = y * row_len + x;
    tile[threadIdx.y][threadIdx.x] = A[index];
    __syncthreads();

    if (    (0 <= gidx && gidx < row_len)
        &&  (x_axis_min <= threadIdx.x && threadIdx.x < SQ_BLOCKSIZE - x_axis_max)
        &&  (0 <= gidy && gidy < col_len)
        &&  (y_axis_min <= threadIdx.y && threadIdx.y < SQ_BLOCKSIZE - y_axis_max)
        )
    {
        T sum_acc = 0;

        for (long i = 0; i < ixs_len; i++ ){
            const long y = threadIdx.y + ixs_2d[i].y;
            const long x = threadIdx.x + ixs_2d[i].x;
            sum_acc += tile[y][x];
        }
        out[gindex] = sum_acc / (T)ixs_len;

    }
}

template<long ixs_len, long x_axis_min, long x_axis_max, long y_axis_min, long y_axis_max>
__global__
void big_tile_2d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long row_len,
    const long col_len
    )
{
    const long waste_x = x_axis_min + x_axis_max;
    const long waste_y = y_axis_min + y_axis_max;
    const long block_offset_x = blockIdx.x*SQ_BLOCKSIZE;
    const long block_offset_y = blockIdx.y*SQ_BLOCKSIZE;
    const long gidx = block_offset_x + threadIdx.x;
    const long gidy = block_offset_y + threadIdx.y;
    const long gindex = gidy * row_len + gidx;
    const long max_x_ix = row_len - 1;
    const long max_y_ix = col_len - 1;

    const long shared_size_x = SQ_BLOCKSIZE + waste_x;
    const long shared_size_y = SQ_BLOCKSIZE + waste_y;
    __shared__ T tile[shared_size_y][shared_size_x];

    const long x_iters = (shared_size_x + (SQ_BLOCKSIZE-1)) / SQ_BLOCKSIZE;
    const long y_iters = (shared_size_y + (SQ_BLOCKSIZE-1)) / SQ_BLOCKSIZE;

    for(long i = 0; i < y_iters; i++){

        for(long j = 0; j < x_iters; j++){
            const long local_y = threadIdx.y + i*SQ_BLOCKSIZE;
            const long local_x = threadIdx.x + j*SQ_BLOCKSIZE;
            if(local_x < shared_size_x && local_y < shared_size_y){
                const long gx = BOUNDL( local_x + block_offset_x - x_axis_min, max_x_ix);
                const long gy = BOUNDL( local_y + block_offset_y - y_axis_min, max_y_ix);
                const long index = gy * row_len + gx;
                tile[local_y][local_x] = A[index];
            }
        }
    }
    __syncthreads();

    if((gidx < row_len) && (gidy < col_len))
    {
        T sum_acc = 0;

        for (long i = 0; i < ixs_len; i++ ){
            const long y = threadIdx.y + y_axis_min + ixs_2d[i].y;
            const long x = threadIdx.x + x_axis_min + ixs_2d[i].x;
            sum_acc += tile[y][x];
        }
        out[gindex] = sum_acc / (T)ixs_len;
    }
}
*/

#endif



