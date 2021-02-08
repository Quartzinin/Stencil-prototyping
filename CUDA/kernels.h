#ifndef CUDA_PROJ_HELPER
#define CUDA_PROJ_HELPER

#include <cuda_runtime.h>

template <int T>
__global__
void sevenPointStencil_single_iter(
        const float* A,
        float * out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz
        )
{
    const float c0 = 1.0/6.0;
    const float c1 = 1.0/6.0/6.0;

    const unsigned lidz = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned lidy = blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned total = ny*nz;

    __shared__ float TILE[3][T][T+1];

    //should current tile read from something 32 x 32, or write to something 32 x 32
    //write something 32x32 by adding extra reads.!!!

    // in this algorithm we do not write to the edges of the matrix

    TILE[0][threadIdx.y][threadIdx.x] = A[lidz+lidy*nz+0*total];

    __syncthreads();

    // for small y and z we should tune the schedule to utilize the parallelism of the x-axis
    for (int i = 0; i < nx; ++i)
    {
        TILE[(i+1) % 3][threadIdx.y][threadIdx.x] = A[lidz+lidy*nz+(i+1)*total];
        __syncthreads();
    
        const float mid = TILE[i % 3][threadIdx.y][threadIdx.x];
        float res;
        if (i == 0 || i == nx-1 || lidy == 0 || lidy == ny-1 || lidz == 0 || lidz == nz-1)
        {
            res = mid;
        }
        else {
            const float left_z  = (threadIdx.x == 0) ? A[(lidz-1) + lidy*nz+i*total]
                			            : TILE[i % 3][threadIdx.y][threadIdx.x-1];
            const float right_z = (threadIdx.x == T-1) ? A[(lidz+1) + lidy*nz+i*total]
                			               : TILE[i % 3][threadIdx.y][threadIdx.x+1];
            const float left_y  = (threadIdx.y == 0) ? A[lidz + (lidy-1)*nz+i*total]
                			             : TILE[i % 3][threadIdx.y-1][threadIdx.x];
            const float right_y = (threadIdx.y == T-1) ? A[lidz + (lidy+1)*nz+i*total]
                			               : TILE[i % 3][threadIdx.y+1][threadIdx.x];
            const float left_x  = TILE[(i-1) % 3][threadIdx.y][threadIdx.x];
            const float right_x = TILE[(i+1) % 3][threadIdx.y][threadIdx.x];

            res = left_z + right_z + left_y + right_y + left_x + right_x * c1 + mid * c0;
        }
        out[lidz+lidy*nz+i*total] = res;
    }

}


//z yyyy z
//y xxxx y
//y xxxx y
//y xxxx y
//y xxxx y
//z yyyy z

#endif
