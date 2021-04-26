#ifndef KERNELS1D
#define KERNELS1D

#include <cuda_runtime.h>
#include "constants.h"

/*
 * inlined indices using a provided associative and commutative operator with a neutral element.
 */

template<int amin_x, int sh_size_flat, int group_size>
__device__
__forceinline__
void bigtile_flat_loader(
    const T* A,
    T tile[sh_size_flat],
    const long lens,
    const int locals,
    const long block_offset)
{
    const long max_ix = lens - 1;
    const int x_iters = divUp(sh_size_flat,group_size);

    for (int i = 0; i < x_iters; i++)
    {
        const int local_x = locals + i*group_size;
        const long gx = long(local_x) + block_offset + long(amin_x);
        if (local_x < sh_size_flat)
        {
            tile[local_x] = A[BOUNDL(gx, max_ix)];
        }
    }
}

template<
    const int amin_x, const int amax_x
    ,const int sh_size_flat
    >
__device__
__forceinline__
void write_from_shared_flat(
    const T tile[],
    T* const out,
    const long lens,
    const int locals,
    const long block_offsets)
{
    constexpr int range = (amax_x - amin_x) + 1;

    const long gid_x = block_offsets + long(locals);

    T vals[range];

    const bool should_write = gid_x < lens;
    if(should_write){
        for(int k=0; k < range; k++){
            const int x = locals + k;
            vals[k] = tile[long(x)];
        }

        T sum_acc = 0;
        for(int k=0; k < range; k++){
            sum_acc += vals[k];
        }
        sum_acc /= (T)range;
        out[gid_x] = sum_acc;
    }
}

template<long ix_min, long ix_max, int group_size>
__global__
__launch_bounds__(BLOCKSIZE)
void global_read_1d_inline(
    const T* A,
    T* out,
    const long nx
    )
{
    const long gid = long(blockIdx.x)*long(group_size) + long(threadIdx.x);
    const long max_ix = nx - 1;
    const int range = (ix_max - ix_min) + 1;
    T vals[range];
    const bool should_write = gid < nx;
    if (should_write)
    {
        for (int i = 0; i < range; ++i){
            const long loc_x = BOUNDL(gid + long(i + ix_min), max_ix);
            vals[i] = A[loc_x];
        }

        T sum_acc = 0;
        for (int i = 0; i < range; ++i){
            sum_acc += vals[i];
        } 
        out[gid] = sum_acc/(T)range;
    }
}

template<long ix_min, long ix_max, int group_size, int strip_x>
__global__
__launch_bounds__(BLOCKSIZE)
void global_read_1d_inline_strip(
    const T* A,
    T* out,
    const long nx
    )
{
    const long max_ix = nx - 1;
    const int range = (ix_max - ix_min) + 1;
    const int strip_length = group_size*strip_x;
    const long start_gid_offset = long(blockIdx.x)*long(strip_length) + long(threadIdx.x); 
    for (int x__ = 0; x__ < strip_x; ++x__)
    {
        T vals[range];
        const long gid = start_gid_offset + long(x__*group_size);
        const bool should_write = gid < nx;
        if (should_write)
        {
            for (int i = 0; i < range; ++i){
                const long loc_x = BOUNDL(gid + long(i + ix_min), max_ix);
                vals[i] = A[loc_x];
            }

            T sum_acc = 0;
            for (int i = 0; i < range; ++i){
                sum_acc += vals[i];
            } 
            out[gid] = sum_acc/(T)range;
        }
    }
}

template<long ix_min, long ix_max, int group_size>
__global__
__launch_bounds__(BLOCKSIZE)
void small_tile_1d_inline(
    const T* A,
    T* out,
    const long nx
    )
{
    const int wasted = ix_max - ix_min;
    const long offset = long(group_size-wasted)*long(blockIdx.x);
    const long gid = offset + long(threadIdx.x + ix_min);
    const long max_ix = nx - 1;
    const int range = wasted + 1;
    extern __shared__ T tile[];
    tile[long(threadIdx.x)] = A[BOUNDL(gid, max_ix)];
    __syncthreads();
    T vals[range];
    if ((0 <= gid && gid < nx) && (-ix_min <= threadIdx.x && threadIdx.x < group_size-ix_max))
    {
        T sum_acc = 0;

        for (int i = 0; i < range; ++i){
            const int loc_x = threadIdx.x + i + ix_min;
            vals[i] = tile[long(loc_x)];
        }

        for (int i = 0; i < range; ++i){
            sum_acc += vals[i];
        } 

        out[gid] = sum_acc/(T)range;
    }
}

template<int amin_x, int amax_x, int group_size>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_1d_inline(
    const T* A,
    T* out,
    const long nx
    )
{
    const long block_offset = long(blockIdx.x)*long(group_size);
    const int shared_size = group_size + (amax_x - amin_x);
    extern __shared__ T tile[];

    bigtile_flat_loader
        <amin_x
        ,shared_size
        ,group_size>
        (A, tile
         , nx
         , threadIdx.x
         , block_offset);

    __syncthreads();

    write_from_shared_flat<amin_x,
                           amax_x,
                           shared_size>
        (tile,out,nx,threadIdx.x,block_offset);
}

template
< const int amin_x, const int amax_x
, const int group_size_x
, const int strip_x
>
__global__
__launch_bounds__(BLOCKSIZE)
void stripmine_big_tile_1d_inlined(
    const T* A,
    T* out,
    const long lens
    )
{
    constexpr int sh_size_x = strip_x*group_size_x + (amax_x - amin_x);
    constexpr int sh_size_flat = sh_size_x;
    __shared__ T tile[sh_size_flat];
    constexpr long strip_id_scaler_x = strip_x * group_size_x;

    const long strip_id_flat = blockIdx.x;

    const int loc_flat = threadIdx.x;

    const long block_offset_x = strip_id_flat * strip_id_scaler_x;

    bigtile_flat_loader
        <amin_x
        ,sh_size_flat
        ,group_size_x>
        (A, tile
         , lens
         , loc_flat
         , block_offset_x);
    
    // the tile has to be fully done being loaded before we start reading
    __syncthreads();
    for(int k__ = 0; k__ < strip_x; k__++){
        // tile_offsets implicitly also handle the change in group_id
        const int tile_offset_x = loc_flat + (k__ * group_size_x);

        write_from_shared_flat<amin_x,
                           amax_x,
                           sh_size_flat>
        (tile,out,lens,tile_offset_x,block_offset_x);
    }
}


/*
 * Inlined indices but provided the elements to the lambda function using a local array.
 */
/*
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




*/












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
