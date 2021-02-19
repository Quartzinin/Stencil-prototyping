#ifndef CUDA_PROJ_HELPER
#define CUDA_PROJ_HELPER

#include <cuda_runtime.h>

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

template<int D>
__device__
inline int stencil_fun(const int* arr){
    int sum_acc = 0;
    for (int i = 0; i < D; ++i){
        sum_acc += arr[i];
    }
    return sum_acc/(D);
}

template<int W>
__global__
void breathFirst_generic1d(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int idx = gid - W;
    int sum[2*W];
    if (gid < nx)
    {
        int sum_acc = 0;
        for (int i = 0; i < 2*W; ++i)
        {
            int newIdx = idx + i;
            newIdx = newIdx < 0 ? 0 : newIdx;
            newIdx = newIdx > nx - 1 ? nx - 1 : newIdx;
            sum[i] = A[newIdx];
        }
        //function
        for (int i = 0; i < 2*W; ++i)
        {
            sum_acc += sum[i];
        }
        int lambda_res = sum_acc/(2*W);
        //lambda_res
        out[gid] = lambda_res;
    }
}

//sacrifice potential parallelism, to ensure that all threads are doing work
template<int W, int T>
__global__
void big_tiled_generic1d(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int shared_size = 2*W+T;
    __shared__ int sum[shared_size];

    int varIdx = threadIdx.x;
    const int end = nx - 1;

    for (int i = gid-W; i < gid+W; i+=blockDim.x)
    {
        if (varIdx < shared_size)
        {
            if (i < 0)
            {
                sum[varIdx] = A[0];
            }
            else if (i > end)
            {
                sum[varIdx] = A[end];
            }
            else
            {
               sum[varIdx] = A[i];
            }
            varIdx += blockDim.x;
        }
    }
    __syncthreads();

    const int* pointer = &(sum[threadIdx.x]);
    const int w2 = 2*W;
    if (gid < nx)
    {
        const int lambda_res = stencil_fun<w2>(pointer);
        out[gid] = lambda_res;
    }
}


template<int W>
__global__
void inlinedIndexesBreathFirst_generic1d(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int idx = gid - W;
    if (gid < nx)
    {
        int sum_acc = 0;
        for (int i = 0; i < 2*W; ++i)
        {
            int newIdx = idx + i;
            newIdx = newIdx < 0 ? 0 : newIdx;
            newIdx = newIdx > nx - 1 ? nx - 1 : newIdx;
            sum_acc += A[newIdx];
        }
        int lambda_res = sum_acc/(2*W);
        //lambda_res
        out[gid] = lambda_res;
    }
}

template<int W, int T>
__global__
void outOfSharedtiled_generic1d(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int offset = blockDim.x*blockIdx.x;
    const int start = gid - W;
    __shared__ int tile[T];
    if (gid < nx)
    {
        tile[threadIdx.x] = A[gid];
    }
    __syncthreads();

    if (gid < nx)
    {
        int sum_acc = 0;

        for (int i = 0; i < 2*W; ++i)
        {
            int val;
            int Idx = start + i;
            Idx = Idx < 0 ? 0 : Idx;
            Idx = Idx > nx - 1 ? nx - 1 : Idx;

            int temp = Idx - offset;
            if (0 <= temp && temp < T)
            {
                val = tile[temp];
            }
            else
            {
                val = A[Idx];
            }

            sum_acc += val;
        }
        int lambda_res = sum_acc/(2*W);
        //lambda_res
        out[gid] = lambda_res;
    }
}

template<int D, int T>
__global__
void inSharedtiled_generic1d(
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
    const int offset = (blockDim.x-W*2)*blockIdx.x;
    const int gid = offset + threadIdx.x - W;
    const int max_ix = nx - 1;

    tile[threadIdx.x] = A[max(0,min(max_ix,gid))];
    __syncthreads();

    if ((0 <= gid && gid < nx) && (W <= threadIdx.x && threadIdx.x < T-W))
    {
        int subtile[D];
        for(int i = 0; i < D; i++){ subtile[i] = tile[sixs[i] + threadIdx.x]; }
        const int lambda_res = stencil_fun<D>(subtile);
        out[gid] = lambda_res;
    }
}

template<int D>
__global__
void global_temp_generic_1d_to_temp(
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
void global_temp_generic_1d(
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
