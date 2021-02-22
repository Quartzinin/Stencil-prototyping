#include <cuda_runtime.h>

/*
__global__
void sevenPointStencil_single_iter_tiled_sliding(
        const float* A,
        float * out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz
        )
{
    const unsigned T = 32;
    const float c0 = 1.0/6.0;
    const float c1 = 1.0/6.0/6.0;

    const unsigned lidz = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned lidy = blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned total = ny*nz;

    __shared__ float TILE[3][T][T+1];

    //should current tile read from something 32 x 32, or write to something 32 x 32
    //write something 32x32 by adding extra reads.!!!

    // in this algorithm we do not write to the edges of the matrix

    unsigned toff = lidz + lidy*nz;
    if (lidz < nz && lidy < ny)
    {
        TILE[0][threadIdx.y][threadIdx.x] = A[toff];
    }

    for (int i = 0; i < nx; ++i,toff+=total)
    {
	const unsigned cx = i % 3;
	const unsigned nx = (i+1) % 3;

        if (lidz < nz && lidy < ny && i + 1 < nx)
        {
            TILE[nx][threadIdx.y][threadIdx.x] = A[toff + total];
        }
        __syncthreads();

        if (lidz < nz && lidy < ny)
        {
            const float mid = TILE[cx][threadIdx.y][threadIdx.x];
            float res;
            if (i == 0 || i == nx-1 || lidy == 0 || lidy == ny-1 || lidz == 0 || lidz == nz-1)
            {
                res = mid;
            }
            else {
                const float left_z  = (threadIdx.x == 0) ? A[toff - 1]
                    			                 : TILE[cx][threadIdx.y][threadIdx.x-1];
                const float right_z = (threadIdx.x == T-1) ? A[toff + 1]
                    			               	   : TILE[cx][threadIdx.y][threadIdx.x+1];
                const float left_y  = (threadIdx.y == 0) ? A[toff - nz]
                    			                 : TILE[cx][threadIdx.y-1][threadIdx.x];
                const float right_y = (threadIdx.y == T-1) ? A[toff + nz]
                    			                   : TILE[cx][threadIdx.y+1][threadIdx.x];
                const float left_x  = TILE[(i-1) % 3][threadIdx.y][threadIdx.x];
                const float right_x = TILE[nx][threadIdx.y][threadIdx.x];

                res = left_z + right_z + left_y + right_y + left_x + right_x * c1 + mid * c0;
            }
            out[toff] = res;
        }
    }

}

__global__
void sevenPointStencil_single_iter_tiled_sliding_read(
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
    const unsigned lidx = blockIdx.z*blockDim.z + threadIdx.z;
    const unsigned total = ny*nz;
    const unsigned toff = lidz + lidy*nz + lidx*total;

    __shared__ float TILE[6][6][32];

    float mid;
    if (lidx < nx && lidy < ny && lidz < nz)
    {
	mid = A[toff];
        TILE[threadIdx.z][threadIdx.y][threadIdx.x] = mid;
    }

    __syncthreads();

    float res;
    if (0 < lidz && lidz < nz-1 && 0 < lidy && lidy < ny-1 && 0 < lidx && lidx < nx-1)
    {
	const float left_x  = (threadIdx.x == 0) ? A[toff - 1]
					    : TILE[threadIdx.z][threadIdx.y][threadIdx.x-1];
	const float right_x = (threadIdx.x == 31) ? A[toff + 1]
					       : TILE[threadIdx.z][threadIdx.y][threadIdx.x+1];
	const float left_y  = (threadIdx.y == 0) ? A[toff - nz]
					     : TILE[threadIdx.z][threadIdx.y-1][threadIdx.x];
	const float right_y = (threadIdx.y == 5) ? A[toff + nz]
					       : TILE[threadIdx.z][threadIdx.y+1][threadIdx.x];
	const float left_z  = (threadIdx.z == 0) ? A[toff - total]
				                 : TILE[threadIdx.z-1][threadIdx.y][threadIdx.x];
	const float right_z  = (threadIdx.z == 5) ? A[toff + total]
                                                  : TILE[threadIdx.z+1][threadIdx.y][threadIdx.x];

	res = left_z + right_z + left_y + right_y + left_x + right_x * c1 + mid * c0;
    }
    else {
    	res = mid;
    }
    out[toff] = res;
}


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

    //should current tile read from something 32 x 32, or write to something 32 x 32
    //write something 32x32 by adding extra reads.!!!

    // in this algorithm we do not write to the edges of the matrix

    // for small y and z we should tune the schedule to utilize the parallelism of the x-axis
    //
    unsigned toff = lidz + lidy*nz;
    for (int i = 0; i < nx; ++i, toff+=total)
    {
        if (lidz < nz && lidy < ny)
        {
            const float mid = A[toff];
            float res;
            if (i == 0 || i == nx-1 || lidy == 0 || lidy == ny-1 || lidz == 0 || lidz == nz-1)
            {
                res = mid;
            }
            else {
                const float left_z  = A[toff - 1];
                const float right_z = A[toff + 1];
                const float left_y  = A[toff - nz];
                const float right_y = A[toff + nz];
                const float left_x  = A[toff - total];
                const float right_x = A[toff + total];

                res = left_z + right_z + left_y + right_y + left_x + right_x * c1 + mid * c0;
            }
            out[toff] = res;
        }
    }

}
*/

