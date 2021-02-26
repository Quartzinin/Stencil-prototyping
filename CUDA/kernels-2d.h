#ifndef KERNELS2D
#define KERNELS2D

#include <cuda_runtime.h>
#include "constants.h"

template<int D, int y_l, int x_l, int2 qixs[D], class T>
__device__
inline T stencil_fun_inline_ix_2d(const T arr[y_l][x_l], const int y_off, const int x_off){
    T sum_acc = 0;
    for (int i = 0; i < D; i++ ){
        const int y = y_off + qixs[i].y;
        const int x = x_off + qixs[i].x;
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
    const int gidx = blockIdx.x*SQ_BLOCKSIZE + threadIdx.x;
    const int gidy = blockIdx.y*SQ_BLOCKSIZE + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    if (gidx < row_len && gidy < col_len)
    {
        T sum_acc = 0;
        for (int i = 0; i < ixs_len; i++ ){
            const int y = BOUND(gidy + ixs_2d[i].y, max_y_ix);
            const int x = BOUND(gidx + ixs_2d[i].x, max_x_ix);
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
        out[gindex] = stencil_fun_inline_ix_2d<ixs_len, SQ_BLOCKSIZE,SQ_BLOCKSIZE, ixs_2d>
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
    for(int i = 0; i < y_iters; i++){
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
        out[gindex] = stencil_fun_inline_ix_2d<ixs_len, shared_size_y, shared_size_x, ixs_2d>
                                              (tile, threadIdx.y + y_axis_min, threadIdx.x + x_axis_min);
    }
}


/*
 * versions with hardcoded index arrays:
 */

#define CONST_9_IXS \
    const int D = 9;\
    const int2 qixs[D] = { make_int2(-1,-1), make_int2(0,-1), make_int2(1,-1), make_int2(-1,0)\
        , make_int2(0,0),  make_int2(1,0),  make_int2(-1,1),  make_int2(0,1),  make_int2(1,1)};

#define CONST_9_WASTE \
    const int x_axis_min = 1;\
    const int x_axis_max = 1;\
    const int y_axis_min = 1;\
    const int y_axis_max = 1;\
    const int waste_x = x_axis_min + x_axis_max;\
    const int waste_y = y_axis_min + y_axis_max;


template<class T>
__global__
void global_reads_2d_9(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    CONST_9_IXS;
    const int gidx = blockIdx.x*SQ_BLOCKSIZE + threadIdx.x;
    const int gidy = blockIdx.y*SQ_BLOCKSIZE + threadIdx.y;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;

    if (gidx < row_len && gidy < col_len)
    {

        T sum_acc = 0;
        for (int i = 0; i < D; i++ ){
            const int y = BOUND(gidy + qixs[i].y, max_y_ix);
            const int x = BOUND(gidx + qixs[i].x, max_x_ix);
            const int index = y * row_len + x;
            sum_acc += A[index];
        }
        out[gindex] = sum_acc / (T)D;
    }
}

template<class T>
__global__
void small_tile_2d_9(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    CONST_9_IXS;
    CONST_9_WASTE;
    const int gidx = blockIdx.x*(SQ_BLOCKSIZE - waste_x) + threadIdx.x - x_axis_min;
    const int gidy = blockIdx.y*(SQ_BLOCKSIZE - waste_y) + threadIdx.y - y_axis_min;
    const int gindex = gidy * row_len + gidx;
    const int max_x_ix = row_len - 1;
    const int max_y_ix = col_len - 1;
    const int x = BOUND(gidx, max_x_ix);
    const int y = BOUND(gidy, max_y_ix);
    const int index = y * row_len + x;
    __shared__ T tile[SQ_BLOCKSIZE][SQ_BLOCKSIZE];
    tile[threadIdx.y][threadIdx.x] = A[index];
    __syncthreads();

    if (    (0 <= gidx && gidx < row_len)
        &&  (x_axis_min <= threadIdx.x && threadIdx.x < SQ_BLOCKSIZE - x_axis_max)
        &&  (0 <= gidy && gidy < col_len)
        &&  (y_axis_min <= threadIdx.y && threadIdx.y < SQ_BLOCKSIZE - y_axis_max)
        )
    {
        T sum_acc = 0;
        for (int i = 0; i < D; i++ ){
            const int y = threadIdx.y + qixs[i].y;
            const int x = threadIdx.x + qixs[i].x;
            sum_acc += tile[y][x];
        }
        const T lambda_res = sum_acc / (T)D;
        out[gindex] = lambda_res;
    }
}

template<class T>
__global__
void big_tile_2d_9(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned row_len,
    const unsigned col_len
    )
{
    CONST_9_IXS;
    CONST_9_WASTE;
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
    for(int i = 0; i < y_iters; i++){
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
        for (int i = 0; i < D; i++ ){
            const int y = y_axis_min + threadIdx.y + qixs[i].y;
            const int x = x_axis_min + threadIdx.x + qixs[i].x;
            sum_acc += tile[y][x];
        }
        const T lambda_res = sum_acc / (T)D;
        out[gindex] = lambda_res;
    }
}

#endif

