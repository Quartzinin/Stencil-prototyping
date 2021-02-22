#ifndef CUDA_PROJ_HELPER
#define CUDA_PROJ_HELPER

#include <cuda_runtime.h>

#define BOUND(i,max_ix) (min((max_ix),max(0,(i))))

__constant__ int ixs[501];

template<int D>
__device__
inline int stencil_fun(const int* arr){
    int sum_acc = 0;
    for (int i = 0; i < D; ++i){
        sum_acc += arr[i];
    }
    return sum_acc/(D);
}

template<int D>
__device__
inline int stencil_fun_inline_ix(const int* arr){
    int sum_acc = 0;
    for (int i = 0; i < D; ++i){
        sum_acc += arr[ixs[i]];
    }
    return sum_acc/D;
}

//sacrifice potential parallelism, to ensure that all threads are doing work
template<int W, int T> // assume T = blockDim.x
__global__
void big_tiled_1d(
    const int* A,
    const int* ixs,
    int* out,
    const unsigned nx
    )
{
    const int block_offset = blockIdx.x*blockDim.x;
    const int gid = block_offset + threadIdx.x;
    const int D = 2*W+1;
    const int left_extra = W;
    const int right_extra = W;
    const int shared_size = left_extra + T + right_extra;
    const int max_ix = nx - 1;
    __shared__ int sixs[D];
    __shared__ int tile[shared_size];
    if(threadIdx.x < D){ sixs[threadIdx.x] = ixs[threadIdx.x]; }

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
        int subtile[D];
        const int base = threadIdx.x + left_extra;
        for(int i = 0; i < D; i++){
            subtile[i] = tile[sixs[i] + base];
        }
        out[gid] = stencil_fun<D>(subtile);
    }
}

template<int D, int T> // assume T = blockDim.x
__global__
void big_tiled_1d_const_ixs(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int block_offset = blockIdx.x*blockDim.x;
    const int gid = block_offset + threadIdx.x;
    const int W = D / 2;
    const int left_extra = W;
    const int right_extra = W;
    const int shared_size = left_extra + T + right_extra;
    const int max_ix = nx - 1;
    __shared__ int tile[shared_size];

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
        int subtile[D];
        const int base = threadIdx.x + left_extra;
        for(int i = 0; i < D; i++){
            subtile[i] = tile[ixs[i] + base];
        }
        out[gid] = stencil_fun<D>(subtile);
    }
}
template<int D, int T> // assume T = blockDim.x
__global__
void big_tiled_1d_const_ixs_inline(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int block_offset = blockIdx.x*blockDim.x;
    const int gid = block_offset + threadIdx.x;
    const int W = D / 2;
    const int left_extra = W;
    const int right_extra = W;
    const int shared_size = left_extra + T + right_extra;
    const int max_ix = nx - 1;
    __shared__ int tile[shared_size];

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
        out[gid] = stencil_fun_inline_ix<D>(&(tile[left_extra + threadIdx.x]));
    }
}

template<int W>
__global__
void inlinedIndexes_1d(
    const int* A,
    const int* ixs,
    int* out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int D = 2*W + 1;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        int sum_acc = 0;
        for (int i = 0; i < D; ++i)
        {
            sum_acc += A[BOUND(gid + ixs[i], max_ix)];
        }
        out[gid] = sum_acc/D;
    }
}

template<int D>
__global__
void inlinedIndexes_1d_const_ixs(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        int sum_acc = 0;
        for (int i = 0; i < D; ++i)
        {
            sum_acc += A[BOUND(gid + ixs[i], max_ix)];
        }
        out[gid] = sum_acc/D;
    }
}

template<int W>
__global__
void threadLocalArr_1d(
    const int* A,
    const int* ixs,
    int* out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int D = 2*W + 1;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        int arr[D];
        for (int i = 0; i < D; ++i)
        {
            arr[i] = A[BOUND(gid + ixs[i], max_ix)];
        }
        out[gid] = stencil_fun<D>(arr);
    }
}
template<int D>
__global__
void threadLocalArr_1d_const_ixs(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        int arr[D];
        for (int i = 0; i < D; ++i)
        {
            arr[i] = A[BOUND(gid + ixs[i], max_ix)];
        }
        out[gid] = stencil_fun<D>(arr);
    }
}

template<int W, int T>
__global__
void outOfSharedtiled_1d(
    const int* A,
    const int* ixs,
    int* out,
    const unsigned nx
    )
{
    const int block_offset = blockDim.x*blockIdx.x;
    const int gid = block_offset + threadIdx.x;
    const int max_ix = nx - 1;
    const int D = 2*W + 1;
    __shared__ int sixs[D];
    if(threadIdx.x < D){ sixs[threadIdx.x] = ixs[threadIdx.x]; }
    __shared__ int tile[T];
    if (gid < nx){ tile[threadIdx.x] = A[gid]; }

    __syncthreads();

    if (gid < nx)
    {
        int arr[D];
        for (int i = 0; i < D; ++i)
        {
            int gix = BOUND(gid + sixs[i], max_ix);
            int lix = gix - block_offset;
            arr[i] = (0 <= lix && lix < T) ? tile[lix] : A[gix];
        }
        out[gid] = stencil_fun<D>(arr);
    }
}
template<int D, int T>
__global__
void outOfSharedtiled_1d_const_ixs(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int block_offset = blockDim.x*blockIdx.x;
    const int gid = block_offset + threadIdx.x;
    const int max_ix = nx - 1;
    __shared__ int tile[T];
    if (gid < nx){ tile[threadIdx.x] = A[gid]; }

    __syncthreads();

    if (gid < nx)
    {
        int arr[D];
        for (int i = 0; i < D; ++i)
        {
            int gix = BOUND(gid + ixs[i], max_ix);
            int lix = gix - block_offset;
            arr[i] = (0 <= lix && lix < T) ? tile[lix] : A[gix];
        }
        out[gid] = stencil_fun<D>(arr);
    }
}

template<int D, int T>
__global__
void inSharedtiled_1d_const_ixs(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    __shared__ int tile[T];

    const int W = D / 2;
    const int offset = (blockDim.x-2*W)*blockIdx.x;
    const int gid = offset + threadIdx.x - W;
    const int max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUND(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (W <= threadIdx.x && threadIdx.x < T-W))
    {
        int subtile[D];
        for(int i = 0; i < D; i++){
            subtile[i] = tile[ixs[i] + threadIdx.x];
        }
        out[gid] = stencil_fun<D>(subtile);
    }
}

template<int D, int T>
__global__
void inSharedtiled_1d_const_ixs_inline(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    __shared__ int tile[T];

    const int W = D / 2;
    const int offset = (blockDim.x-2*W)*blockIdx.x;
    const int gid = offset + threadIdx.x - W;
    const int max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUND(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (W <= threadIdx.x && threadIdx.x < T-W))
    {
        out[gid] = stencil_fun_inline_ix<D>(&(tile[threadIdx.x]));
    }
}


template<int D, int T>
__global__
void inSharedtiled_1d(
    const int* A,
    const int* ixs,
    int* out,
    const unsigned nx
    )
{
    __shared__ int sixs[D];
    __shared__ int tile[T];
    if(threadIdx.x < D){ sixs[threadIdx.x] = ixs[threadIdx.x]; }

    const int W = D / 2;
    const int offset = (blockDim.x-2*W)*blockIdx.x;
    const int gid = offset + threadIdx.x - W;
    const int max_ix = nx - 1;
    tile[threadIdx.x] = A[BOUND(gid, max_ix)];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (W <= threadIdx.x && threadIdx.x < T-W))
    {
        int subtile[D];
        for(int i = 0; i < D; i++){
            subtile[i] = tile[sixs[i] + threadIdx.x];
        }
        out[gid] = stencil_fun<D>(subtile);
    }
}

template<int D>
__global__
void global_temp__1d_to_temp(
    const int* A,
    const int* ixs,
    int* temp,
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

template<int D>
__global__
void global_temp__1d(
    const int* temp,
    int* out,
    const unsigned nx
    ){
    const int gid = blockDim.x*blockIdx.x + threadIdx.x;
    const int temp_i_start = gid * D;
    if(gid < nx){
        out[gid] = stencil_fun<D>(temp + temp_i_start);
    }
}



#endif



/*
__global__
void sevenPointStencil_single_iter_tiled_sliding(
        const float* A,
        float * out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz
        );

__global__
void sevenPointStencil_single_iter_tiled_sliding_read(
        const float* A,
        float * out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz
        );

__global__
void sevenPointStencil_single_iter(
        const float* A,
        float * out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz
        );
*/

/*
template<int W>
__global__
void breathFirst(
    const int* A,
    int * out,
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
