#ifndef KERNELS1D
#define KERNELS1D

#include <cuda_runtime.h>
#include "constants.h"

/*
template<int D, class T>
__device__
inline int stencil_fun_inline_ix_bounded(const T* arr, const int cix, const int max_ix){
    T sum_acc = 0;
    for (int i = 0; i < D; ++i){
        sum_acc += arr[BOUND(cix + ixs[i], max_ix)];
    }
    return sum_acc/D;
}

template<int D, int arr_len, class T>
__device__
inline int stencil_fun_inline_ix(const T arr[arr_len]){
    T sum_acc = 0;
    for (int i = 0; i < D; ++i){
        sum_acc += arr[ixs[i]];
    }
    return sum_acc/D;
}
*/


template<int ixs_len, class T>
__global__
void inlinedIndexes_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*BLOCKSIZE + threadIdx.x;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        T sum_acc = 0;
        #pragma unroll
        for (int i = 0; i < ixs_len; ++i){
            const int loc_x = BOUND(gid + ixs_1d[i], max_ix);
            sum_acc += A[loc_x];
        }
        out[gid] = sum_acc/ixs_len;
    }
}

template<int ixs_len, int ix_min, int ix_max, class T>
__global__
void inSharedtiled_1d_const_ixs_inline(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    __shared__ T tile[BLOCKSIZE];

    const int wasted = ix_min + ix_max;
    const int offset = (BLOCKSIZE-wasted)*blockIdx.x;
    const int gid = offset + threadIdx.x - ix_min;
    const int max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUND(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (ix_min <= threadIdx.x && threadIdx.x < BLOCKSIZE-ix_max))
    {
        T sum_acc = 0;
        #pragma unroll
        for (int i = 0; i < ixs_len; ++i){
            const int loc_x = threadIdx.x + ixs_1d[i];
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/ixs_len;
    }
}

template<int ixs_len, int ix_min, int ix_max, class T>
__global__
void big_tiled_1d_const_ixs_inline(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int block_offset = blockIdx.x*BLOCKSIZE;
    const int gid = block_offset + threadIdx.x;
    const int shared_size = ix_min + BLOCKSIZE + ix_max;
    const int max_ix = nx - 1;
    __shared__ T tile[shared_size];

    const int left_most = block_offset - ix_min;
    const int x_iters = (shared_size + (BLOCKSIZE-1)) / BLOCKSIZE;
    #pragma unroll
    for (int i = 0; i < x_iters; i++)
    {
        const int local_x = threadIdx.x + i*BLOCKSIZE;
        const int gx = local_x + left_most;
        if (local_x < shared_size)
        {
            tile[local_x] = A[BOUND(gx, max_ix)];
        }
    }
    __syncthreads();

    if (gid < nx)
    {
        T sum_acc = 0;
        #pragma unroll
        for (int i = 0; i < ixs_len; ++i){
            const int loc_x = threadIdx.x + ix_min + ixs_1d[i];
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/ixs_len;
    }
}

/*
 * CONST INDICES
 */

template<int D, int ix_min, int ix_max, class T>
__global__
void global_read_1d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*BLOCKSIZE + threadIdx.x;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        T sum_acc = 0;
        const int step = (ix_min + ix_max) / (D-1);
        #pragma unroll
        for (int i = 0; i < D; ++i){
            const int loc_x = BOUND(gid + i*step - ix_min , max_ix);
            sum_acc += A[loc_x];
        }
        out[gid] = sum_acc/D;
    }
}

template<int D, int ix_min, int ix_max, class T>
__global__
void small_tile_1d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int wasted = ix_min + ix_max;
    const int offset = (BLOCKSIZE-wasted)*blockIdx.x;
    const int gid = offset + threadIdx.x - ix_min;
    const int max_ix = nx - 1;
    __shared__ T tile[BLOCKSIZE];
    tile[threadIdx.x] = A[BOUND(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (ix_min <= threadIdx.x && threadIdx.x < BLOCKSIZE-ix_max))
    {
        T sum_acc = 0;
        const int step = (ix_min + ix_max) / (D-1);
        #pragma unroll
        for (int i = 0; i < D; ++i){
            const int loc_x = threadIdx.x + i*step - ix_min;
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/D;
    }
}

template<int D, int ix_min, int ix_max, class T>
__global__
void big_tile_1d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int block_offset = blockIdx.x*BLOCKSIZE;
    const int gid = block_offset + threadIdx.x;
    const int shared_size = ix_min + BLOCKSIZE + ix_max;
    const int max_ix = nx - 1;
    __shared__ T tile[shared_size];

    const int left_most = block_offset - ix_min;
    const int x_iters = (shared_size + (BLOCKSIZE-1)) / BLOCKSIZE;
    #pragma unroll
    for (int i = 0; i < x_iters; i++)
    {
        const int local_x = threadIdx.x + i*BLOCKSIZE;
        const int gx = local_x + left_most;
        if (local_x < shared_size)
        {
            tile[local_x] = A[BOUND(gx, max_ix)];
        }
    }
    __syncthreads();

    if (gid < nx)
    {
        T sum_acc = 0;
        const int step = (ix_min + ix_max) / (D-1);
        #pragma unroll
        for (int i = 0; i < D; ++i){
            const int loc_x = threadIdx.x + i*step;
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/D;
    }
}






















/*

template<int D, class T>
__device__
inline T stencil_fun(const T* arr){
    T sum_acc = 0;
    for (int i = 0; i < D; ++i){
        sum_acc += arr[i];
    }
    return sum_acc/(D);
}

template<int ixs_len, int ix_min, int ix_max, class T>
__global__
void big_tiled_1d(
    const T* __restrict__ A,
    const int* ixs,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int block_offset = blockIdx.x*blockDim.x;
    const int gid = block_offset + threadIdx.x;
    const int left_extra = ix_min;
    const int right_extra = ix_max;
    const int shared_size = left_extra + BLOCKSIZE + right_extra;
    const int max_ix = nx - 1;
    __shared__ int sixs[ixs_len];
    __shared__ T tile[shared_size];
    if(threadIdx.x < ixs_len){ sixs[threadIdx.x] = ixs[threadIdx.x]; }

    const int right_most = block_offset - left_extra + shared_size;
    int loc_ix = threadIdx.x;
    for (int i = gid - left_extra; i < right_most; i += blockDim.x)
    {
        if (loc_ix < shared_size)
        {
            tile[loc_ix] = A[BOUND(i, max_ix)];
            loc_ix += blockDim.x;
        }
    }
    __syncthreads();

    if (gid < nx)
    {
        T subtile[ixs_len];
        const int base = threadIdx.x + left_extra;
        for(int i = 0; i < ixs_len; i++){
            subtile[i] = tile[sixs[i] + base];
        }
        out[gid] = stencil_fun<ixs_len, T>(subtile);
    }
}

template<int ixs_len, int ix_min, int ix_max, class T>
__global__
void big_tiled_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int block_offset = blockIdx.x*blockDim.x;
    const int gid = block_offset + threadIdx.x;
    const int D = ixs_len;
    const int left_extra = ix_min;
    const int right_extra = ix_max;
    const int shared_size = left_extra + BLOCKSIZE + right_extra;
    const int max_ix = nx - 1;
    __shared__ T tile[shared_size];

    const int right_most = block_offset - left_extra + shared_size;
    int loc_ix = threadIdx.x;
    for (int i = gid - left_extra; i < right_most; i += blockDim.x)
    {
        if (loc_ix < shared_size)
        {
            tile[loc_ix] = A[BOUND(i, max_ix)];
            loc_ix += blockDim.x;
        }
    }
    __syncthreads();

    if (gid < nx)
    {
        T subtile[D];
        const int base = threadIdx.x + left_extra;
        for(int i = 0; i < D; i++){
            subtile[i] = tile[ixs[i] + base];
        }
        out[gid] = stencil_fun<D, T>(subtile);
    }
}
*/
/*
template<int ixs_len, class T>
__global__
void inlinedIndexes_1d(
    const T* __restrict__ A,
    const int* ixs,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int D = ixs_len;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        T sum_acc = 0;
        for (int i = 0; i < D; ++i)
        {
            sum_acc += A[BOUND(gid + ixs[i], max_ix)];
        }
        out[gid] = sum_acc/D;
    }
}
*/


/*
template<int ixs_len, class T>
__global__
void threadLocalArr_1d(
    const T* __restrict__ A,
    const int* ixs,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int D = ixs_len;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        T arr[D];
        for (int i = 0; i < D; ++i)
        {
            arr[i] = A[BOUND(gid + ixs[i], max_ix)];
        }
        out[gid] = stencil_fun<D, T>(arr);
    }
}
template<int ixs_len, class T>
__global__
void threadLocalArr_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int max_ix = nx - 1;
    const int D = ixs_len;
    if (gid < nx)
    {
        T arr[D];
        for (int i = 0; i < D; ++i)
        {
            arr[i] = A[BOUND(gid + ixs[i], max_ix)];
        }
        out[gid] = stencil_fun<D, T>(arr);
    }
}

template<int ixs_len, class T>
__global__
void outOfSharedtiled_1d(
    const T* __restrict__ A,
    const int* ixs,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int block_offset = blockDim.x*blockIdx.x;
    const int gid = block_offset + threadIdx.x;
    const int max_ix = nx - 1;
    const int D = ixs_len;
    __shared__ int sixs[D];
    if(threadIdx.x < D){ sixs[threadIdx.x] = ixs[threadIdx.x]; }
    __shared__ T tile[BLOCKSIZE];
    if (gid < nx){ tile[threadIdx.x] = A[gid]; }

    __syncthreads();

    if (gid < nx)
    {
        T arr[D];
        for (int i = 0; i < D; ++i)
        {
            int gix = BOUND(gid + sixs[i], max_ix);
            int lix = gix - block_offset;
            arr[i] = (0 <= lix && lix < BLOCKSIZE) ? tile[lix] : A[gix];
        }
        out[gid] = stencil_fun<D, T>(arr);
    }
}
template<int ixs_len, class T>
__global__
void outOfSharedtiled_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int D = ixs_len;
    const int block_offset = blockDim.x*blockIdx.x;
    const int gid = block_offset + threadIdx.x;
    const int max_ix = nx - 1;
    __shared__ T tile[BLOCKSIZE];
    if (gid < nx){ tile[threadIdx.x] = A[gid]; }

    __syncthreads();

    if (gid < nx)
    {
        T arr[D];
        for (int i = 0; i < D; ++i)
        {
            int gix = BOUND(gid + ixs[i], max_ix);
            int lix = gix - block_offset;
            arr[i] = (0 <= lix && lix < BLOCKSIZE) ? tile[lix] : A[gix];
        }
        out[gid] = stencil_fun<D, T>(arr);
    }
}

template<int ixs_len, int ix_min, int ix_max, class T>
__global__
void inSharedtiled_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    __shared__ T tile[BLOCKSIZE];

    const int D = ixs_len;
    const int wasted = ix_min + ix_max;
    const int offset = (blockDim.x-wasted)*blockIdx.x;
    const int gid = offset + threadIdx.x - ix_min;
    const int max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUND(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (ix_min <= threadIdx.x && threadIdx.x < BLOCKSIZE-ix_max))
    {
        T subtile[D];
        for(int i = 0; i < D; i++){
            subtile[i] = tile[ixs[i] + threadIdx.x];
        }
        out[gid] = stencil_fun<ixs_len, T>(subtile);
    }
}
*/


/*
template<int ixs_len, int ix_min, int ix_max, class T>
__global__
void inSharedtiled_1d(
    const T* __restrict__ A,
    const int* ixs,
    T* __restrict__ out,
    const unsigned nx
    )
{
    const int D = ixs_len;
    __shared__ int sixs[ixs_len];
    __shared__ T tile[BLOCKSIZE];
    if(threadIdx.x < D){ sixs[threadIdx.x] = ixs[threadIdx.x]; }

    const int wasted = ix_min + ix_max;
    const int offset = (blockDim.x-wasted)*blockIdx.x;
    const int gid = offset + threadIdx.x - ix_min;
    const int max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUND(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (ix_min <= threadIdx.x && threadIdx.x < BLOCKSIZE-ix_max))
    {
        T subtile[D];
        for(int i = 0; i < D; i++){
            subtile[i] = tile[sixs[i] + threadIdx.x];
        }
        out[gid] = stencil_fun<D, T>(subtile);
    }
}

template<int D, class T>
__global__
void global_temp__1d_to_temp(
    const T* __restrict__ A,
    const int* ixs,
    T* temp,
    const int nx
    ){
    const int max_ix = nx-1;
    const int gid = blockDim.x*blockIdx.x + threadIdx.x;
    const int chunk_idx = gid / D;
    const int chunk_off = gid % D;
    const int i = max(0, min(max_ix, (chunk_idx + ixs[chunk_off])));
    if(gid < nx*D){
        temp[gid] = A[i];
    }
}

template<int D, class T>
__global__
void global_temp__1d(
    const T* temp,
    T* __restrict__ out,
    const unsigned nx
    ){
    const int gid = blockDim.x*blockIdx.x + threadIdx.x;
    const int temp_i_start = gid * D;
    if(gid < nx){
        out[gid] = stencil_fun<D, T>(temp + temp_i_start);
    }
}
*/





/*
__global__
void sevenPointStencil_single_iter_tiled_sliding(
        const float* __restrict__ A,
        float * __restrict__ out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz
        );

__global__
void sevenPointStencil_single_iter_tiled_sliding_read(
        const float* __restrict__ A,
        float * __restrict__ out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz
        );

__global__
void sevenPointStencil_single_iter(
        const float* __restrict__ A,
        float * __restrict__ out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz
        );
*/

/*
template<int W>
__global__
void breathFirst(
    const T* __restrict__ A,
    int * __restrict__ out,
    const unsigned nx
    )
{
    const unsigned gid = blockIdx.x*blockDim.x + threadIdx.x;

    if (gid < nx)
    {
        const unsigned li = (gid <= W) ? 0 : gid - W;
        const unsigned lli = (gid <= 2*W) ? 0 : gid - 2*W;
        const unsigned llli = (gid <= 3*W) ? 0 : gid - 3*W;
        const unsigned hi = ((nx - W) <= gid) ? nx - 1 : gid + W;
        const unsigned hhi = ((nx - 2*W) <= gid) ? nx - 1 : gid + 2*W;
        const unsigned hhhi = ((nx - 3*W) <= gid) ? nx - 1 : gid + 3*W;
        out[gid] = A[llli] + A[lli] + A[li] + A[gid] + A[hi] + A[hhi] + A[hhhi];
    }
}
*/
#endif
