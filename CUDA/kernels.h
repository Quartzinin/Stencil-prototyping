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
        int sum_acc = 0;
        //function
        for (int i = 0; i < w2; ++i)
        {
            sum_acc += pointer[i];
        }
        int lambda_res = sum_acc/w2;
        //lambda_res
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

template<int W, int T>
__global__
void inSharedtiled_generic1d(
    const int* A,
    int* out,
    const unsigned nx
    )
{
    const int w2 = 2*W;
    const int offset = (blockDim.x-w2)*blockIdx.x;
    const int gid = offset + threadIdx.x - W;
    const int max_ix = nx - 1;

    __shared__ int tile[T];
    //if (0 <= gid && gid < nx)
    {
        tile[threadIdx.x] = A[max(0,min(max_ix,gid))];
    }
    __syncthreads();

    const int* tile_p = &(tile[threadIdx.x - W]);
    if ((0 <= gid && gid < nx) && (W <= threadIdx.x && threadIdx.x < T-W))
    {
        int sum_acc = 0;

        for (int i = 0; i < w2; ++i)
        {
            sum_acc += tile_p[i];
        }
        int lambda_res = sum_acc/w2;
        //lambda_res
        out[gid] = lambda_res;
    }
}




#endif
