#ifndef KERNELS1D
#define KERNELS1D

#include <cuda_runtime.h>
#include "constants.h"

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
        out[gid] = stencil_fun_inline_ix_bounded<ixs_len, T>(A, gid, max_ix);
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
        out[gid] = stencil_fun_inline_ix<ixs_len, BLOCKSIZE, T>(&(tile[threadIdx.x]));
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
        out[gid] = stencil_fun_inline_ix<ixs_len, shared_size, T>(&(tile[ix_min + threadIdx.x]));
    }
}

/*
 *
 */

#define CONST_QIXS \
    const int qixs[401] = {-200, -199, -198, -197, -196, -195, -194, -193, -192, -191, -190, -189, -188, -187, -186, -185, -184, -183, -182, -181, -180, -179, -178, -177, -176, -175, -174, -173, -172, -171, -170, -169, -168, -167, -166, -165, -164, -163, -162, -161, -160, -159, -158, -157, -156, -155, -154, -153, -152, -151, -150, -149, -148, -147, -146, -145, -144, -143, -142, -141, -140, -139, -138, -137, -136, -135, -134, -133, -132, -131, -130, -129, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -93, -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200};

#define CONST_QIXSB \
    const int qixs[401] = { 200, -199, 198, -197, 196, -195, 194, -193, 192, -191, 190, -189, 188, -187, 186, -185, 184, -183, 182, -181, 180, -179, 178, -177, 176, -175, 174, -173, 172, -171, 170, -169, 168, -167, 166, -165, 164, -163, 162, -161, 160, -159, 158, -157, 156, -155, 154, -153, 152, -151, 150, -149, 148, -147, 146, -145, 144, -143, 142, -141, 140, -139, 138, -137, 136, -135, 134, -133, 132, -131, 130, -129, 128, -127, 126, -125, 124, -123, 122, -121, 120, -119, 118, -117, 116, -115, 114, -113, 112, -111, 110, -109, 108, -107, 106, -105, 104, -103, 102, -101, 100, -99, 98, -97, 96, -95, 94, -93, 92, -91, 90, -89, 88, -87, 86, -85, 84, -83, 82, -81, 80, -79, 78, -77, 76, -75, 74, -73, 72, -71, 70, -69, 68, -67, 66, -65, 64, -63, 62, -61, 60, -59, 58, -57, 56, -55, 54, -53, 52, -51, 50, -49, 48, -47, 46, -45, 44, -43, 42, -41, 40, -39, 38, -37, 36, -35, 34, -33, 32, -31, 30, -29, 28, -27, 26, -25, 24, -23, 22, -21, 20, -19, 18, -17, 16, -15, 14, -13, 12, -11, 10, -9, 8, -7, 6, -5, 4, -3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32, 33, -34, 35, -36, 37, -38, 39, -40, 41, -42, 43, -44, 45, -46, 47, -48, 49, -50, 51, -52, 53, -54, 55, -56, 57, -58, 59, -60, 61, -62, 63, -64, 65, -66, 67, -68, 69, -70, 71, -72, 73, -74, 75, -76, 77, -78, 79, -80, 81, -82, 83, -84, 85, -86, 87, -88, 89, -90, 91, -92, 93, -94, 95, -96, 97, -98, 99, -100, 101, -102, 103, -104, 105, -106, 107, -108, 109, -110, 111, -112, 113, -114, 115, -116, 117, -118, 119, -120, 121, -122, 123, -124, 125, -126, 127, -128, 129, -130, 131, -132, 133, -134, 135, -136, 137, -138, 139, -140, 141, -142, 143, -144, 145, -146, 147, -148, 149, -150, 151, -152, 153, -154, 155, -156, 157, -158, 159, -160, 161, -162, 163, -164, 165, -166, 167, -168, 169, -170, 171, -172, 173, -174, 175, -176, 177, -178, 179, -180, 181, -182, 183, -184, 185, -186, 187, -188, 189, -190, 191, -192, 193, -194, 195, -196, 197, -198, 199, -200 };

template<int D,int q_off, int ix_min, int ix_max, class T>
__global__
void global_read_1d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    CONST_QIXS;

    const int gid = blockIdx.x*BLOCKSIZE + threadIdx.x;
    const int max_ix = nx - 1;
    if (gid < nx)
    {
        T sum_acc = 0;
        #pragma unroll
        for (int i = 0; i < D; ++i){
            const int loc_x = BOUND(gid + qixs[i + q_off], max_ix);
            sum_acc += A[loc_x];
        }
        out[gid] = sum_acc/D;
    }
}

template<int D,int q_off, int ix_min, int ix_max, class T>
__global__
void small_tile_1d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    CONST_QIXS;

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
        #pragma unroll
        for (int i = 0; i < D; ++i){
            const int loc_x = threadIdx.x + qixs[i + q_off];
            sum_acc += tile[loc_x];
        }
        out[gid] = sum_acc/D;
    }
}

template<int D,int q_off, int ix_min, int ix_max, class T>
__global__
void big_tile_1d_const(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned nx
    )
{
    CONST_QIXS;

    const int block_offset = blockIdx.x*BLOCKSIZE;
    const int gid = block_offset + threadIdx.x;
    const int shared_size = ix_min + BLOCKSIZE + ix_max;
    const int max_ix = nx - 1;
    __shared__ T tile[shared_size];

    const int left_most = block_offset - ix_min;
    const int x_iters = (shared_size + (BLOCKSIZE-1)) / BLOCKSIZE;
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
        for (int i = 0; i < D; ++i){
            const int loc_x = threadIdx.x + ix_min + qixs[i + q_off];
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
