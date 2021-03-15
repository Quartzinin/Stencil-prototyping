#ifndef KERNELS3D
#define KERNELS3D

#include <cuda_runtime.h>
#include "constants.h"

/*
 * inlined indice versions:
 */

template<
    int z_axis_min, int z_axis_max,
    int y_axis_min, int y_axis_max,
    int x_axis_min, int x_axis_max>
__global__
void global_reads_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len, //outermost
    const unsigned y_len, //middle
    const unsigned x_len) //innermost
{
    const int3 gid = {
        int(blockIdx.x*X_BLOCK + threadIdx.x),
        int(blockIdx.y*Y_BLOCK + threadIdx.y),
        int(blockIdx.z*Z_BLOCK + threadIdx.z)};

    if (gid.x < x_len && gid.y < y_len && gid.z < z_len)
    {
        const int gindex = (gid.z*y_len + gid.y)*x_len + gid.x;
        const int3 max_idx = {
            int(x_len - 1),
            int(y_len - 1),
            int(z_len - 1)};
        constexpr int3 range = {
            x_axis_max + x_axis_min + 1,
            y_axis_max + y_axis_min + 1,
            z_axis_max + z_axis_min + 1};
        constexpr int total_range = range.z * range.y * range.x;

        T sum_acc = 0;
        #pragma unroll
        for(int i=0; i < range.z; i++){
            const int z = BOUND(gid.z + i - z_axis_min, max_idx.z);
            #pragma unroll
            for(int j=0; j < range.y; j++){
                const int y = BOUND(gid.y + j - y_axis_min, max_idx.y);
                #pragma unroll
                for(int k=0; k < range.x; k++){
                    const int x = BOUND(gid.x + k - x_axis_min, max_idx.x);
                    const int index = (z*y_len + y)*x_len + x;
                    sum_acc += A[index];
                }
            }
        }
        sum_acc /= (T)total_range;
        out[gindex] = sum_acc;
    }
}

template<
    int z_axis_min, int z_axis_max,
    int y_axis_min, int y_axis_max,
    int x_axis_min, int x_axis_max>
__global__
void small_tile_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len,
    const unsigned y_len,
    const unsigned x_len
    )
{
    __shared__ T tile[Z_BLOCK][Y_BLOCK][X_BLOCK];
    constexpr int3 waste = {
        int(x_axis_min + x_axis_max),
        int(y_axis_min + y_axis_max),
        int(z_axis_min + z_axis_max)};
    const int3 gid = {
        int(blockIdx.x*(X_BLOCK - waste.x) + threadIdx.x - x_axis_min),
        int(blockIdx.y*(Y_BLOCK - waste.y) + threadIdx.y - y_axis_min),
        int(blockIdx.z*(Z_BLOCK - waste.z) + threadIdx.z - z_axis_min)};

    { // reading segment
        const int3 max_idx = {
            int(x_len - 1),
            int(y_len - 1),
            int(z_len - 1)};

        const int x = BOUND(gid.x, max_idx.x);
        const int y = BOUND(gid.y, max_idx.y);
        const int z = BOUND(gid.z, max_idx.z);

        const int index = (z * y_len + y) * x_len + x;
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = A[index];
    }

    __syncthreads();
    { // writing segment
        const int should_write =
            (       (0 <= gid.x && gid.x < x_len)
                &&  (x_axis_min <= threadIdx.x && threadIdx.x < X_BLOCK - x_axis_max)
                &&  (0 <= gid.y && gid.y < y_len)
                &&  (y_axis_min <= threadIdx.y && threadIdx.y < Y_BLOCK - y_axis_max)
                &&  (0 <= gid.z && gid.z < z_len)
                &&  (z_axis_min <= threadIdx.z && threadIdx.z < Z_BLOCK - z_axis_max)
            );
        if(should_write){
            const int gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr int3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr int total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            #pragma unroll
            for(int i=0; i < range.z; i++){
                const int z = threadIdx.z + i - z_axis_min;
                #pragma unroll
                for(int j=0; j < range.y; j++){
                    const int y = threadIdx.y + j - y_axis_min;
                    #pragma unroll
                    for(int k=0; k < range.x; k++){
                        const int x = threadIdx.x + k - x_axis_min;
                        sum_acc += tile[z][y][x];
                    }
                }
            }
            sum_acc /= (T)total_range;
            out[gid_flat] = sum_acc;
        }
    }
}

template<
    int z_axis_min, int z_axis_max,
    int y_axis_min, int y_axis_max,
    int x_axis_min, int x_axis_max>
__global__
void big_tile_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len,
    const unsigned y_len,
    const unsigned x_len
    )
{
    constexpr int3 waste = {
        int(x_axis_min + x_axis_max),
        int(y_axis_min + y_axis_max),
        int(z_axis_min + z_axis_max)};
    const int3 block_offset = {
        int(blockIdx.x*X_BLOCK),
        int(blockIdx.y*Y_BLOCK),
        int(blockIdx.z*Z_BLOCK)};
    constexpr int3 shared_size = {
        int(X_BLOCK + waste.x),
        int(Y_BLOCK + waste.y),
        int(Z_BLOCK + waste.z)};

    __shared__ T tile[shared_size.z][shared_size.y][shared_size.x];

    { // reading segment
        const int3 max_idx = {
            int(x_len - 1),
            int(y_len - 1),
            int(z_len - 1)};
        const int3 view_offset = {
            block_offset.x - x_axis_min,
            block_offset.y - y_axis_min,
            block_offset.z - z_axis_min};
        constexpr int3 iters = {
            CEIL_DIV(shared_size.x, X_BLOCK),
            CEIL_DIV(shared_size.y, Y_BLOCK),
            CEIL_DIV(shared_size.z, Z_BLOCK)};

        #pragma unroll
        for(int i = 0; i < iters.z; i++){
            const int local_z = threadIdx.z + i*Z_BLOCK;
            const int gid_z = y_len * BOUND((local_z + view_offset.z), max_idx.z);
            #pragma unroll
            for(int j = 0; j < iters.y; j++){
                const int local_y = threadIdx.y + j*Y_BLOCK;
                const int gid_zy = x_len * (gid_z + BOUND((local_y + view_offset.y), max_idx.y));
                #pragma unroll
                for (int k = 0; k < iters.x; k++){
                    const int local_x = threadIdx.x + k*X_BLOCK;
                    const int gid_zyx = gid_zy + BOUND((local_x + view_offset.x), max_idx.x);
                    if((local_z < shared_size.z) && (local_y < shared_size.y) && (local_x < shared_size.x)){
                        tile[local_z][local_y][local_x] = A[gid_zyx];
                    }
                }
            }
        }
    }
    __syncthreads();

    { // writing segment
        const int3 gid = {
            int(block_offset.x + threadIdx.x),
            int(block_offset.y + threadIdx.y),
            int(block_offset.z + threadIdx.z)};
        const int should_write = ((gid.x < x_len) && (gid.y < y_len) && (gid.z < z_len));
        if(should_write){
            const int gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr int3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr int total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            #pragma unroll
            for(int i=0; i < range.z; i++){
                const int z = threadIdx.z + i;
                #pragma unroll
                for(int j=0; j < range.y; j++){
                    const int y = threadIdx.y + j;
                    #pragma unroll
                    for(int k=0; k < range.x; k++){
                        const int x = threadIdx.x + k;
                        sum_acc += tile[z][y][x];
                    }
                }
            }
            sum_acc /= (T)total_range;
            out[gid_flat] = sum_acc;
        }
    }
}

template<
    int z_axis_min, int z_axis_max,
    int y_axis_min, int y_axis_max,
    int x_axis_min, int x_axis_max>
__global__
void big_tile_3d_inlined_flat(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len,
    const unsigned y_len,
    const unsigned x_len
    )
{
    constexpr int3 waste = {
        int(x_axis_min + x_axis_max),
        int(y_axis_min + y_axis_max),
        int(z_axis_min + z_axis_max)};
    const int3 block_offset = {
        int(blockIdx.x*X_BLOCK),
        int(blockIdx.y*Y_BLOCK),
        int(blockIdx.z*Z_BLOCK)};
    constexpr int3 shared_size = {
        int(X_BLOCK + waste.x),
        int(Y_BLOCK + waste.y),
        int(Z_BLOCK + waste.z)};

    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;
    __shared__ T tile[shared_size_zyx];
    { // reading segment
        const int3 max_idx = {
            int(x_len - 1),
            int(y_len - 1),
            int(z_len - 1)};
        const int3 view_offset = {
            block_offset.x - x_axis_min,
            block_offset.y - y_axis_min,
            block_offset.z - z_axis_min};

        constexpr int blockDimFlat = X_BLOCK * Y_BLOCK * Z_BLOCK;
        const int blockIdxFlat = (threadIdx.z * Y_BLOCK + threadIdx.y) * X_BLOCK + threadIdx.x;

        constexpr int lz_span = shared_size.y * shared_size.x;
        constexpr int ly_span = shared_size.x;

        constexpr int iters = CEIL_DIV(shared_size_zyx, blockDimFlat);
        #pragma unroll
        for(int i = 0; i < iters; i++){
            const int local_ix = (i * blockDimFlat) + blockIdxFlat;

            const int rem_z = local_ix % lz_span;
            const int rem_y = rem_z % ly_span;
            const int local_z = local_ix / lz_span;
            const int local_y = rem_z / ly_span;
            const int local_x = rem_y;

            const int gz = BOUND((local_z + view_offset.z), max_idx.z);
            const int gy = BOUND((local_y + view_offset.y), max_idx.y);
            const int gx = BOUND((local_x + view_offset.x), max_idx.x);

            const int index = (gz * y_len + gy) * x_len + gx;
            if(local_ix < shared_size_zyx){
                tile[local_ix] = A[index];
            }
        }
    }
    __syncthreads();

    { // writing segment
        const int3 gid = {
            int(block_offset.x + threadIdx.x),
            int(block_offset.y + threadIdx.y),
            int(block_offset.z + threadIdx.z)};
        const int should_write = ((gid.x < x_len) && (gid.y < y_len) && (gid.z < z_len));
        if(should_write){
            const int gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr int3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr int total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            #pragma unroll
            for(int i=0; i < range.z; i++){
                const int z = (threadIdx.z + i)*shared_size.y;
                #pragma unroll
                for(int j=0; j < range.y; j++){
                    const int y = (z + threadIdx.y + j)*shared_size.x;
                    #pragma unroll
                    for(int k=0; k < range.x; k++){
                        const int x = y + threadIdx.x + k;
                        sum_acc += tile[x];
                    }
                }
            }
            sum_acc /= (T)total_range;
            out[gid_flat] = sum_acc;
        }
    }
}

template<
    int z_axis_min, int z_axis_max,
    int y_axis_min, int y_axis_max,
    int x_axis_min, int x_axis_max>
__global__
void big_tile_3d_inlined_layered(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len,
    const unsigned y_len,
    const unsigned x_len
    )
{
    constexpr int3 waste = {
        int(x_axis_min + x_axis_max),
        int(y_axis_min + y_axis_max),
        int(z_axis_min + z_axis_max)};
    const int3 block_offset = {
        int(blockIdx.x*X_BLOCK),
        int(blockIdx.y*Y_BLOCK),
        int(blockIdx.z*Z_BLOCK)};
    constexpr int3 shared_size = {
        int(X_BLOCK + waste.x),
        int(Y_BLOCK + waste.y),
        int(Z_BLOCK + waste.z)};

    __shared__ T tile[shared_size.z][shared_size.y][shared_size.x];
    { // reading segment
        const int3 max_idx = {
            int(x_len - 1),
            int(y_len - 1),
            int(z_len - 1)};
        const int3 view_offset = {
            block_offset.x - x_axis_min,
            block_offset.y - y_axis_min,
            block_offset.z - z_axis_min};

        const int BLOCK_Xm = X_BLOCK * (Z_BLOCK / 2);
        const int BLOCK_Ym = Y_BLOCK * (Z_BLOCK / 2);
        //const int BLOCK_Xm = X_BLOCK;
        //const int BLOCK_Ym = Y_BLOCK * Z_BLOCK;
        const int x_iters = CEIL_DIV(shared_size.x, BLOCK_Xm);
        const int y_iters = CEIL_DIV(shared_size.y, BLOCK_Ym);
        const int blockIdxFlat = (threadIdx.z * Y_BLOCK + threadIdx.y) * X_BLOCK + threadIdx.x;
        const int thread_y = blockIdxFlat / BLOCK_Xm;
        const int thread_x = blockIdxFlat % BLOCK_Xm;

        #pragma unroll
        for(int local_z = 0; local_z < shared_size.z; local_z++){
            const int gz = y_len * BOUND( local_z + view_offset.z, max_idx.z);
            #pragma unroll
            for(int j = 0; j < y_iters; j++){
                const int local_y = thread_y + j*BLOCK_Ym;
                if(local_y < shared_size.y){
                    #pragma unroll
                    for (int k = 0; k < x_iters; k++){
                        const int local_x = thread_x + k*BLOCK_Xm;
                        const int gy = x_len * (gz + BOUND( local_y + view_offset.y, max_idx.y));
                        const int gx = gy + BOUND( local_x + view_offset.x, max_idx.x);

                        if(local_x < shared_size.x){
                            tile[local_z][local_y][local_x] = A[gx];
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    { // writing segment
        const int3 gid = {
            int(block_offset.x + threadIdx.x),
            int(block_offset.y + threadIdx.y),
            int(block_offset.z + threadIdx.z)};
        const int should_write = ((gid.x < x_len) && (gid.y < y_len) && (gid.z < z_len));
        if(should_write){
            const int gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr int3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr int total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            #pragma unroll
            for(int i=0; i < range.z; i++){
                const int z = threadIdx.z + i;
                #pragma unroll
                for(int j=0; j < range.y; j++){
                    const int y = threadIdx.y + j;
                    #pragma unroll
                    for(int k=0; k < range.x; k++){
                        const int x = threadIdx.x + k;
                        sum_acc += tile[z][y][x];
                    }
                }
            }
            sum_acc /= (T)total_range;
            out[gid_flat] = sum_acc;
        }
    }
}

/*
 * version reading indices from constant memory
 */

/*
template<int D, int z_l, int y_l, int x_l>
__device__
inline T stencil_fun_inline_ix_3d(const T arr[z_l][y_l][x_l], const int z_off, const int y_off, const int x_off){
    T sum_acc = 0;
    for (int i = 0; i < D; i++ ){
        const int z = z_off + ixs_3d[i].z;
        const int y = y_off + ixs_3d[i].y;
        const int x = x_off + ixs_3d[i].x;
        sum_acc += arr[z][y][x];
    }
    return sum_acc / (T)D;
}


template<int D>
__global__
void global_reads_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len, //outermost
    const unsigned y_len, //middle
    const unsigned x_len) //innermost
{
    const int gidz = blockIdx.z*Z_BLOCK + threadIdx.z;
    const int gidy = blockIdx.y*Y_BLOCK + threadIdx.y;
    const int gidx = blockIdx.x*X_BLOCK + threadIdx.x;

    const int gindex = gidz*y_len*x_len + gidy*x_len + gidx;
    const int y_len_maxIdx = y_len - 1;
    const int x_len_maxIdx = x_len - 1;
    const int z_len_maxIdx = z_len - 1;

    if (gidx < x_len && gidy < y_len && gidz < z_len)
    {
        T sum_acc = 0;
        for (int i = 0; i < D; i++)
        {
            const int z = BOUND(gidz + ixs_3d[i].z, z_len_maxIdx);
            const int y = BOUND(gidy + ixs_3d[i].y, y_len_maxIdx);
            const int x = BOUND(gidx + ixs_3d[i].x, x_len_maxIdx);
            const int index = z*y_len*x_len + y*x_len + x;
            sum_acc += A[index];
        }
        out[gindex] = sum_acc / (T)D;
    }
}


template<int ixs_len,
    int z_axis_min, int z_axis_max,
    int y_axis_min, int y_axis_max,
    int x_axis_min, int x_axis_max>
__global__
void small_tile_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len,
    const unsigned y_len,
    const unsigned x_len
    )
{
    __shared__ T tile[Z_BLOCK][Y_BLOCK][X_BLOCK];
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int waste_z = z_axis_min + z_axis_max;
    const int gidx = blockIdx.x*(X_BLOCK - waste_x) + threadIdx.x - x_axis_min;
    const int gidy = blockIdx.y*(Y_BLOCK - waste_y) + threadIdx.y - y_axis_min;
    const int gidz = blockIdx.z*(Z_BLOCK - waste_z) + threadIdx.z - z_axis_min;
    const int gindex = gidz * y_len * x_len + gidy * x_len + gidx;
    const int max_x_idx = x_len - 1;
    const int max_y_idx = y_len - 1;
    const int max_z_idx = z_len - 1;

    const int x = BOUND(gidx, max_x_idx);
    const int y = BOUND(gidy, max_y_idx);
    const int z = BOUND(gidz, max_z_idx);

    const int index = z * y_len * x_len + y * x_len + x;

    tile[threadIdx.z][threadIdx.y][threadIdx.x] = A[index];

    __syncthreads();

    if (    (0 <= gidx && gidx < x_len)
        &&  (x_axis_min <= threadIdx.x && threadIdx.x < X_BLOCK - x_axis_max)
        &&  (0 <= gidy && gidy < y_len)
        &&  (y_axis_min <= threadIdx.y && threadIdx.y < Y_BLOCK - y_axis_max)
        &&  (0 <= gidz && gidz < z_len)
        &&  (z_axis_min <= threadIdx.z && threadIdx.z < Z_BLOCK - z_axis_max)
        )
    {
        out[gindex] = stencil_fun_inline_ix_3d<ixs_len, Z_BLOCK, Y_BLOCK, X_BLOCK>
                                              (tile, threadIdx.z, threadIdx.y, threadIdx.x);
    }
}

template<int ixs_len,
    int z_axis_min, int z_axis_max,
    int y_axis_min, int y_axis_max,
    int x_axis_min, int x_axis_max>
__global__
void big_tile_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const unsigned z_len,
    const unsigned y_len,
    const unsigned x_len
    )
{
    const int waste_x = x_axis_min + x_axis_max;
    const int waste_y = y_axis_min + y_axis_max;
    const int waste_z = z_axis_min + z_axis_max;

    const int block_offset_x = blockIdx.x*X_BLOCK;
    const int block_offset_y = blockIdx.y*Y_BLOCK;
    const int block_offset_z = blockIdx.z*Z_BLOCK;
    const int gidz = block_offset_z + threadIdx.z;
    const int gidy = block_offset_y + threadIdx.y;
    const int gidx = block_offset_x + threadIdx.x;
    const int gindex = gidz * y_len * x_len + gidy * x_len + gidx;
    const int max_z_idx = z_len - 1;
    const int max_y_idx = y_len - 1;
    const int max_x_idx = x_len - 1;

    const int shared_size_z = Z_BLOCK + waste_z;
    const int shared_size_y = Y_BLOCK + waste_y;
    const int shared_size_x = X_BLOCK + waste_x;

    __shared__ T tile[shared_size_z][shared_size_y][shared_size_x];

    const int z_iters = (shared_size_z + (Z_BLOCK-1)) / Z_BLOCK;
    const int y_iters = (shared_size_y + (Y_BLOCK-1)) / Y_BLOCK;
    const int x_iters = (shared_size_x + (X_BLOCK-1)) / X_BLOCK;

    for(int i = 0; i < z_iters; i++){
        for(int j = 0; j < y_iters; j++){
            for (int k = 0; k < x_iters; k++){
                const int local_z = threadIdx.z + i*Z_BLOCK;
                const int local_y = threadIdx.y + j*Y_BLOCK;
                const int local_x = threadIdx.x + k*X_BLOCK;

                if(local_x < shared_size_x && local_y < shared_size_y && local_z < shared_size_z){
                    const int gx = BOUND( local_x + block_offset_x - x_axis_min, max_x_idx);
                    const int gy = BOUND( local_y + block_offset_y - y_axis_min, max_y_idx);
                    const int gz = BOUND( local_z + block_offset_z - z_axis_min, max_z_idx);
                    const int index = gz * y_len * x_len + gy * x_len + gx;
                    tile[local_z][local_y][local_x] = A[index];
                }
            }
        }
    }
    __syncthreads();

    if((gidx < x_len) && (gidy < y_len) && (gidz < z_len))
    {
        out[gindex] = stencil_fun_inline_ix_3d<ixs_len, shared_size_z, shared_size_y, shared_size_x>
                                              (tile, threadIdx.z + z_axis_min, threadIdx.y + y_axis_min, threadIdx.x + x_axis_min);
    }
}
*/

#endif
