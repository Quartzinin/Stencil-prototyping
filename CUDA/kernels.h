#ifndef CUDA_PROJ_HELPER
#define CUDA_PROJ_HELPER

#include <cuda_runtime.h>

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

#endif
