#ifndef KERNELS1D
#define KERNELS1D

#include <cuda_runtime.h>
#include "constants.h"

/*
 * inlined indices using a provided associative and commutative operator with a neutral element.
 */

template<long D, long ix_min, long ix_max>
__global__
void global_read_1d_inline_reduce(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    const long gid = blockIdx.x*BLOCKSIZE + threadIdx.x;
    const long max_ix = nx - 1;
    if (gid < nx)
    {
        T sum_acc = 0;
        const long step = (1 + ix_min + ix_max) / (D-1);

        #pragma unroll
        for (long i = 0; i < D; ++i){
            const long loc_x = BOUNDL(gid + i*step - ix_min , max_ix);
            sum_acc += A[loc_x];
        }
        out[gid] = sum_acc/D;
    }
}

template<long D, long ix_min, long ix_max>
__global__
void small_tile_1d_inline_reduce(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    const long wasted = ix_min + ix_max;
    const long offset = (BLOCKSIZE-wasted)*blockIdx.x;
    const long gid = offset + threadIdx.x - ix_min;
    const long max_ix = nx - 1;
    __shared__ T tile[BLOCKSIZE];
    tile[threadIdx.x] = A[BOUNDL(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (ix_min <= threadIdx.x && threadIdx.x < BLOCKSIZE-ix_max))
    {
        T sum_acc = 0;
        const long step = (1 + ix_min + ix_max) / (D-1);

        #pragma unroll
        for (long i = 0; i < D; ++i){
            const long loc_x = threadIdx.x + i*step - ix_min;
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/D;
    }
}

template<long D, long ix_min, long ix_max>
__global__
void big_tile_1d_inline_reduce(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    const long block_offset = blockIdx.x*BLOCKSIZE;
    const long gid = block_offset + threadIdx.x;
    const long shared_size = ix_min + BLOCKSIZE + ix_max;
    const long max_ix = nx - 1;
    __shared__ T tile[shared_size];

    const long left_most = block_offset - ix_min;
    const long x_iters = (shared_size + (BLOCKSIZE-1)) / BLOCKSIZE;

    for (long i = 0; i < x_iters; i++)
    {
        const long local_x = threadIdx.x + i*BLOCKSIZE;
        const long gx = local_x + left_most;
        if (local_x < shared_size)
        {
            tile[local_x] = A[BOUNDL(gx, max_ix)];
        }
    }
    __syncthreads();

    if (gid < nx)
    {
        T sum_acc = 0;
        const long step = (1 + ix_min + ix_max) / (D-1);

        #pragma unroll
        for (long i = 0; i < D; ++i){
            const long loc_x = threadIdx.x + i*step;
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/D;
    }
}

/*
 * Inlined indices but provided the elements to the lambda function using a local array.
 */

template<long D, long x_min, long x_max>
__global__
void global_read_1d_inline(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long max_ix_x
    )
{
    constexpr long step = (1 + x_max - x_min) / (D-1);
    const long gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid <= max_ix_x)
    {
        T arr[D];

        #pragma unroll
        for (long i = 0; i < D; ++i)
        {
            arr[i] = A[BOUNDL(gid + (i*step + x_min), max_ix_x)];
        }
        T res = lambdaFun<D>(arr);

        out[gid] = res;
    }
}
template<long D, long x_min, long x_max>
__global__
void big_tile_1d_inline(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long max_ix_x,
    const long shared_len
    )
{
    extern __shared__ T tile[];
    const long writeSet_offset = blockIdx.x*blockDim.x;
    const long gid = writeSet_offset + threadIdx.x;
    { // load tile
        const long readSet_off = writeSet_offset + x_min;
        const long x_iters = CEIL_DIV(shared_len, blockDim.x);

        for (long i = 0; i < x_iters; i++)
        {
            const long local_x = threadIdx.x + i*blockDim.x;
            const long gx = local_x + readSet_off;
            if (local_x < shared_len)
            {
                tile[local_x] = A[BOUNDL(gx, max_ix_x)];
            }
        }
    }
    __syncthreads();

    { // load local array an eval
        constexpr long step = (1 + x_max - x_min) / (D-1);
        if (gid <= max_ix_x)
        {
            T arr[D];
            #pragma unroll
            for (long i = 0; i < D; ++i)
            {
                const long off = i*step;
                arr[i] = tile[threadIdx.x + off];
            }
            T res = lambdaFun<D>(arr);

            out[gid] = res;
        }
    }
}

















/*
template<long ixs_len>
__global__
void inlinedIndexes_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    const long gid = blockIdx.x*BLOCKSIZE + threadIdx.x;
    const long max_ix = nx - 1;
    if (gid < nx)
    {
        T sum_acc = 0;

        for (long i = 0; i < ixs_len; ++i){
            const long loc_x = BOUNDL(gid + ixs_1d[i], max_ix);
            sum_acc += A[loc_x];
        }
        out[gid] = sum_acc/ixs_len;
    }
}

template<long ixs_len, long ix_min, long ix_max>
__global__
void inSharedtiled_1d_const_ixs_inline(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    __shared__ T tile[BLOCKSIZE];

    const long wasted = ix_min + ix_max;
    const long offset = (BLOCKSIZE-wasted)*blockIdx.x;
    const long gid = offset + threadIdx.x - ix_min;
    const long max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUNDL(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (ix_min <= threadIdx.x && threadIdx.x < BLOCKSIZE-ix_max))
    {
        T sum_acc = 0;

        for (long i = 0; i < ixs_len; ++i){
            const long loc_x = threadIdx.x + ixs_1d[i];
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/ixs_len;
    }
}

template<long ixs_len, long ix_min, long ix_max>
__global__
void big_tiled_1d_const_ixs_inline(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    const long block_offset = blockIdx.x*BLOCKSIZE;
    const long gid = block_offset + threadIdx.x;
    const long shared_size = ix_min + BLOCKSIZE + ix_max;
    const long max_ix = nx - 1;
    __shared__ T tile[shared_size];

    const long left_most = block_offset - ix_min;
    const long x_iters = (shared_size + (BLOCKSIZE-1)) / BLOCKSIZE;

    for (long i = 0; i < x_iters; i++)
    {
        const long local_x = threadIdx.x + i*BLOCKSIZE;
        const long gx = local_x + left_most;
        if (local_x < shared_size)
        {
            tile[local_x] = A[BOUNDL(gx, max_ix)];
        }
    }
    __syncthreads();

    if (gid < nx)
    {
        T sum_acc = 0;

        for (long i = 0; i < ixs_len; ++i){
            const long loc_x = threadIdx.x + ix_min + ixs_1d[i];
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/ixs_len;
    }
}
*/

/*

template<long D>
__device__
inline T stencil_fun(const T* arr){
    T sum_acc = 0;
    for (long i = 0; i < D; ++i){
        sum_acc += arr[i];
    }
    return sum_acc/(D);
}

template<long ixs_len, long ix_min, long ix_max>
__global__
void big_tiled_1d(
    const T* __restrict__ A,
    const long* ixs,
    T* __restrict__ out,
    const long nx
    )
{
    const long block_offset = blockIdx.x*blockDim.x;
    const long gid = block_offset + threadIdx.x;
    const long left_extra = ix_min;
    const long right_extra = ix_max;
    const long shared_size = left_extra + BLOCKSIZE + right_extra;
    const long max_ix = nx - 1;
    __shared__ long sixs[ixs_len];
    __shared__ T tile[shared_size];
    if(threadIdx.x < ixs_len){ sixs[threadIdx.x] = ixs[threadIdx.x]; }

    const long right_most = block_offset - left_extra + shared_size;
    long loc_ix = threadIdx.x;
    for (long i = gid - left_extra; i < right_most; i += blockDim.x)
    {
        if (loc_ix < shared_size)
        {
            tile[loc_ix] = A[BOUNDL(i, max_ix)];
            loc_ix += blockDim.x;
        }
    }
    __syncthreads();

    if (gid < nx)
    {
        T subtile[ixs_len];
        const long base = threadIdx.x + left_extra;
        for(long i = 0; i < ixs_len; i++){
            subtile[i] = tile[sixs[i] + base];
        }
        out[gid] = stencil_fun<ixs_len, T>(subtile);
    }
}

template<long ixs_len, long ix_min, long ix_max>
__global__
void big_tiled_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    const long block_offset = blockIdx.x*blockDim.x;
    const long gid = block_offset + threadIdx.x;
    const long D = ixs_len;
    const long left_extra = ix_min;
    const long right_extra = ix_max;
    const long shared_size = left_extra + BLOCKSIZE + right_extra;
    const long max_ix = nx - 1;
    __shared__ T tile[shared_size];

    const long right_most = block_offset - left_extra + shared_size;
    long loc_ix = threadIdx.x;
    for (long i = gid - left_extra; i < right_most; i += blockDim.x)
    {
        if (loc_ix < shared_size)
        {
            tile[loc_ix] = A[BOUNDL(i, max_ix)];
            loc_ix += blockDim.x;
        }
    }
    __syncthreads();

    if (gid < nx)
    {
        T subtile[D];
        const long base = threadIdx.x + left_extra;
        for(long i = 0; i < D; i++){
            subtile[i] = tile[ixs[i] + base];
        }
        out[gid] = stencil_fun<D, T>(subtile);
    }
}
*/
/*
template<long ixs_len>
__global__
void inlinedIndexes_1d(
    const T* __restrict__ A,
    const long* ixs,
    T* __restrict__ out,
    const long nx
    )
{
    const long gid = blockIdx.x*blockDim.x + threadIdx.x;
    const long D = ixs_len;
    const long max_ix = nx - 1;
    if (gid < nx)
    {
        T sum_acc = 0;
        for (long i = 0; i < D; ++i)
        {
            sum_acc += A[BOUNDL(gid + ixs[i], max_ix)];
        }
        out[gid] = sum_acc/D;
    }
}
*/


/*
template<long ixs_len>
__global__
void threadLocalArr_1d(
    const T* __restrict__ A,
    const long* ixs,
    T* __restrict__ out,
    const long nx
    )
{
    const long gid = blockIdx.x*blockDim.x + threadIdx.x;
    const long D = ixs_len;
    const long max_ix = nx - 1;
    if (gid < nx)
    {
        T arr[D];
        for (long i = 0; i < D; ++i)
        {
            arr[i] = A[BOUNDL(gid + ixs[i], max_ix)];
        }
        out[gid] = stencil_fun<D, T>(arr);
    }
}
template<long ixs_len>
__global__
void threadLocalArr_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    const long gid = blockIdx.x*blockDim.x + threadIdx.x;
    const long max_ix = nx - 1;
    const long D = ixs_len;
    if (gid < nx)
    {
        T arr[D];
        for (long i = 0; i < D; ++i)
        {
            arr[i] = A[BOUNDL(gid + ixs[i], max_ix)];
        }
        out[gid] = stencil_fun<D, T>(arr);
    }
}

template<long ixs_len>
__global__
void outOfSharedtiled_1d(
    const T* __restrict__ A,
    const long* ixs,
    T* __restrict__ out,
    const long nx
    )
{
    const long block_offset = blockDim.x*blockIdx.x;
    const long gid = block_offset + threadIdx.x;
    const long max_ix = nx - 1;
    const long D = ixs_len;
    __shared__ long sixs[D];
    if(threadIdx.x < D){ sixs[threadIdx.x] = ixs[threadIdx.x]; }
    __shared__ T tile[BLOCKSIZE];
    if (gid < nx){ tile[threadIdx.x] = A[gid]; }

    __syncthreads();

    if (gid < nx)
    {
        T arr[D];
        for (long i = 0; i < D; ++i)
        {
            long gix = BOUNDL(gid + sixs[i], max_ix);
            long lix = gix - block_offset;
            arr[i] = (0 <= lix && lix < BLOCKSIZE) ? tile[lix] : A[gix];
        }
        out[gid] = stencil_fun<D, T>(arr);
    }
}
template<long ixs_len>
__global__
void outOfSharedtiled_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    const long D = ixs_len;
    const long block_offset = blockDim.x*blockIdx.x;
    const long gid = block_offset + threadIdx.x;
    const long max_ix = nx - 1;
    __shared__ T tile[BLOCKSIZE];
    if (gid < nx){ tile[threadIdx.x] = A[gid]; }

    __syncthreads();

    if (gid < nx)
    {
        T arr[D];
        for (long i = 0; i < D; ++i)
        {
            long gix = BOUNDL(gid + ixs[i], max_ix);
            long lix = gix - block_offset;
            arr[i] = (0 <= lix && lix < BLOCKSIZE) ? tile[lix] : A[gix];
        }
        out[gid] = stencil_fun<D, T>(arr);
    }
}

template<long ixs_len, long ix_min, long ix_max>
__global__
void inSharedtiled_1d_const_ixs(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long nx
    )
{
    __shared__ T tile[BLOCKSIZE];

    const long D = ixs_len;
    const long wasted = ix_min + ix_max;
    const long offset = (blockDim.x-wasted)*blockIdx.x;
    const long gid = offset + threadIdx.x - ix_min;
    const long max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUNDL(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (ix_min <= threadIdx.x && threadIdx.x < BLOCKSIZE-ix_max))
    {
        T subtile[D];
        for(long i = 0; i < D; i++){
            subtile[i] = tile[ixs[i] + threadIdx.x];
        }
        out[gid] = stencil_fun<ixs_len, T>(subtile);
    }
}
*/


/*
template<long ixs_len, long ix_min, long ix_max>
__global__
void inSharedtiled_1d(
    const T* __restrict__ A,
    const long* ixs,
    T* __restrict__ out,
    const long nx
    )
{
    const long D = ixs_len;
    __shared__ long sixs[ixs_len];
    __shared__ T tile[BLOCKSIZE];
    if(threadIdx.x < D){ sixs[threadIdx.x] = ixs[threadIdx.x]; }

    const long wasted = ix_min + ix_max;
    const long offset = (blockDim.x-wasted)*blockIdx.x;
    const long gid = offset + threadIdx.x - ix_min;
    const long max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUNDL(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (ix_min <= threadIdx.x && threadIdx.x < BLOCKSIZE-ix_max))
    {
        T subtile[D];
        for(long i = 0; i < D; i++){
            subtile[i] = tile[sixs[i] + threadIdx.x];
        }
        out[gid] = stencil_fun<D, T>(subtile);
    }
}

template<long D>
__global__
void global_temp__1d_to_temp(
    const T* __restrict__ A,
    const long* ixs,
    T* temp,
    const long nx
    ){
    const long max_ix = nx-1;
    const long gid = blockDim.x*blockIdx.x + threadIdx.x;
    const long chunk_idx = gid / D;
    const long chunk_off = gid % D;
    const long i = max(0, min(max_ix, (chunk_idx + ixs[chunk_off])));
    if(gid < nx*D){
        temp[gid] = A[i];
    }
}

template<long D>
__global__
void global_temp__1d(
    const T* temp,
    T* __restrict__ out,
    const long nx
    ){
    const long gid = blockDim.x*blockIdx.x + threadIdx.x;
    const long temp_i_start = gid * D;
    if(gid < nx){
        out[gid] = stencil_fun<D, T>(temp + temp_i_start);
    }
}
*/





/*
__global__
void sevenPolongStencil_single_iter_tiled_sliding(
        const float* __restrict__ A,
        float * __restrict__ out,
        const long nx,
        const long ny,
        const long nz
        );

__global__
void sevenPolongStencil_single_iter_tiled_sliding_read(
        const float* __restrict__ A,
        float * __restrict__ out,
        const long nx,
        const long ny,
        const long nz
        );

__global__
void sevenPolongStencil_single_iter(
        const float* __restrict__ A,
        float * __restrict__ out,
        const long nx,
        const long ny,
        const long nz
        );
*/

/*
template<long W>
__global__
void breathFirst(
    const T* __restrict__ A,
    long * __restrict__ out,
    const long nx
    )
{
    const long gid = blockIdx.x*blockDim.x + threadIdx.x;

    if (gid < nx)
    {
        const long li = (gid <= W) ? 0 : gid - W;
        const long lli = (gid <= 2*W) ? 0 : gid - 2*W;
        const long llli = (gid <= 3*W) ? 0 : gid - 3*W;
        const long hi = ((nx - W) <= gid) ? nx - 1 : gid + W;
        const long hhi = ((nx - 2*W) <= gid) ? nx - 1 : gid + 2*W;
        const long hhhi = ((nx - 3*W) <= gid) ? nx - 1 : gid + 3*W;
        out[gid] = A[llli] + A[lli] + A[li] + A[gid] + A[hi] + A[hhi] + A[hhhi];
    }
}
*/
#endif
