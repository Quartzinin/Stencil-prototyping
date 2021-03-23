#ifndef KERNELS3D
#define KERNELS3D

#include <cuda_runtime.h>
#include "constants.h"

/*
 * inlined indices using a provided associative and commutative operator with a neutral element.
 */

template<
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void global_reads_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len,
    const long y_len,
    const long x_len)
{
    constexpr long3 axis_min = { long(x_axis_min),long(y_axis_min),long(z_axis_min) };
    constexpr long3 axis_max = { long(x_axis_max),long(y_axis_max),long(z_axis_max) };
    const long3 gid = {
        long(blockIdx.x*X_BLOCK + threadIdx.x),
        long(blockIdx.y*Y_BLOCK + threadIdx.y),
        long(blockIdx.z*Z_BLOCK + threadIdx.z)};

    if (gid.x < x_len && gid.y < y_len && gid.z < z_len)
    {
        const long gindex = (gid.z*y_len + gid.y)*x_len + gid.x;
        const long3 max_idx = {
            long(x_len - 1),
            long(y_len - 1),
            long(z_len - 1)};
        constexpr long3 range = {
            axis_max.x + axis_min.x + 1,
            axis_max.y + axis_min.y + 1,
            axis_max.z + axis_min.z + 1};
        constexpr long total_range = range.z * range.y * range.x;

        T sum_acc = 0;
#pragma unroll
        for(long i=0; i < range.z; i++){
                    const long z = BOUNDL(gid.z + (i - axis_min.z), max_idx.z);
#pragma unroll
            for(long j=0; j < range.y; j++){
                    const long y = BOUNDL(gid.y + (j - axis_min.y), max_idx.y);
#pragma unroll
                for(long k=0; k < range.x; k++){
                    const long x = BOUNDL(gid.x + (k - axis_min.x), max_idx.x);
                    const long index = (z*y_len + y)*x_len + x;
                    sum_acc += A[index];
                }
            }
        }
        sum_acc /= (T)total_range;
        out[gindex] = sum_acc;
    }
}

template<
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void small_tile_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len,
    const long y_len,
    const long x_len
    )
{
    __shared__ T tile[Z_BLOCK][Y_BLOCK][X_BLOCK];
    constexpr long3 waste = {
        long(x_axis_min + x_axis_max),
        long(y_axis_min + y_axis_max),
        long(z_axis_min + z_axis_max)};
    const long3 gid = {
        long(blockIdx.x*(X_BLOCK - waste.x) + threadIdx.x - x_axis_min),
        long(blockIdx.y*(Y_BLOCK - waste.y) + threadIdx.y - y_axis_min),
        long(blockIdx.z*(Z_BLOCK - waste.z) + threadIdx.z - z_axis_min)};

    { // reading segment
        const long3 max_idx = {
            long(x_len - 1),
            long(y_len - 1),
            long(z_len - 1)};

        const long x = BOUNDL(gid.x, max_idx.x);
        const long y = BOUNDL(gid.y, max_idx.y);
        const long z = BOUNDL(gid.z, max_idx.z);

        const long index = (z * y_len + y) * x_len + x;
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = A[index];
    }

    __syncthreads();
    { // writing segment
        const long should_write =
            (       (0 <= gid.x && gid.x < x_len)
                &&  (x_axis_min <= threadIdx.x && threadIdx.x < X_BLOCK - x_axis_max)
                &&  (0 <= gid.y && gid.y < y_len)
                &&  (y_axis_min <= threadIdx.y && threadIdx.y < Y_BLOCK - y_axis_max)
                &&  (0 <= gid.z && gid.z < z_len)
                &&  (z_axis_min <= threadIdx.z && threadIdx.z < Z_BLOCK - z_axis_max)
            );
        if(should_write){
            const long gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr long3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr long total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            for(long i=0; i < range.z; i++){
                const long z = threadIdx.z + i - z_axis_min;
                for(long j=0; j < range.y; j++){
                    const long y = threadIdx.y + j - y_axis_min;
                    for(long k=0; k < range.x; k++){
                        const long x = threadIdx.x + k - x_axis_min;
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
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void big_tile_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len,
    const long y_len,
    const long x_len
    )
{
    constexpr long3 waste = {
        long(x_axis_min + x_axis_max),
        long(y_axis_min + y_axis_max),
        long(z_axis_min + z_axis_max)};
    const long3 block_offset = {
        long(blockIdx.x*X_BLOCK),
        long(blockIdx.y*Y_BLOCK),
        long(blockIdx.z*Z_BLOCK)};
    constexpr long3 shared_size = {
        long(X_BLOCK + waste.x),
        long(Y_BLOCK + waste.y),
        long(Z_BLOCK + waste.z)};

    __shared__ T tile[shared_size.z][shared_size.y][shared_size.x];

    { // reading segment
        const long3 max_idx = {
            long(x_len - 1),
            long(y_len - 1),
            long(z_len - 1)};
        const long3 view_offset = {
            block_offset.x - x_axis_min,
            block_offset.y - y_axis_min,
            block_offset.z - z_axis_min};
        constexpr long3 iters = {
            CEIL_DIV(shared_size.x, X_BLOCK),
            CEIL_DIV(shared_size.y, Y_BLOCK),
            CEIL_DIV(shared_size.z, Z_BLOCK)};

        for(long i = 0; i < iters.z; i++){
            const long local_z = threadIdx.z + i*Z_BLOCK;
            const long gid_z = y_len * BOUNDL((local_z + view_offset.z), max_idx.z);
            for(long j = 0; j < iters.y; j++){
                const long local_y = threadIdx.y + j*Y_BLOCK;
                const long gid_zy = x_len * (gid_z + BOUNDL((local_y + view_offset.y), max_idx.y));
                for (long k = 0; k < iters.x; k++){
                    const long local_x = threadIdx.x + k*X_BLOCK;
                    const long gid_zyx = gid_zy + BOUNDL((local_x + view_offset.x), max_idx.x);
                    if(local_z < shared_size.z && local_y < shared_size.y && local_x < shared_size.x){
                        tile[local_z][local_y][local_x] = A[gid_zyx];
                    }
                }
            }
        }
    }
    __syncthreads();

    { // writing segment
        const long3 gid = {
            long(block_offset.x + threadIdx.x),
            long(block_offset.y + threadIdx.y),
            long(block_offset.z + threadIdx.z)};
        const long should_write = ((gid.x < x_len) && (gid.y < y_len) && (gid.z < z_len));
        if(should_write){
            const long gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr long3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr long total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            for(long i=0; i < range.z; i++){
                const long z = threadIdx.z + i;
                for(long j=0; j < range.y; j++){
                    const long y = threadIdx.y + j;
                    for(long k=0; k < range.x; k++){
                        const long x = threadIdx.x + k;
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
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void big_tile_3d_inlined_flat(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len,
    const long y_len,
    const long x_len
    )
{
    constexpr long3 waste = {
        long(x_axis_min + x_axis_max),
        long(y_axis_min + y_axis_max),
        long(z_axis_min + z_axis_max)};
    const long3 block_offset = {
        long(blockIdx.x*X_BLOCK),
        long(blockIdx.y*Y_BLOCK),
        long(blockIdx.z*Z_BLOCK)};
    constexpr long3 shared_size = {
        long(X_BLOCK + waste.x),
        long(Y_BLOCK + waste.y),
        long(Z_BLOCK + waste.z)};

    constexpr long shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;
    __shared__ T tile[shared_size_zyx];
    { // reading segment
        const long3 max_idx = {
            long(x_len - 1),
            long(y_len - 1),
            long(z_len - 1)};
        const long3 view_offset = {
            block_offset.x - x_axis_min,
            block_offset.y - y_axis_min,
            block_offset.z - z_axis_min};

        constexpr long blockDimFlat = X_BLOCK * Y_BLOCK * Z_BLOCK;
        const long blockIdxFlat = (threadIdx.z * Y_BLOCK + threadIdx.y) * X_BLOCK + threadIdx.x;

        constexpr long lz_span = shared_size.y * shared_size.x;
        constexpr long ly_span = shared_size.x;
        constexpr long iters = CEIL_DIV(shared_size_zyx, blockDimFlat);

        for(long i = 0; i < iters; i++){
            const long local_ix = (i * blockDimFlat) + blockIdxFlat;

            const long rem_z = local_ix % lz_span;
            const long rem_y = rem_z % ly_span;
            const long local_z = local_ix / lz_span;
            const long local_y = rem_z / ly_span;
            const long local_x = rem_y;

            const long gz = BOUNDL((local_z + view_offset.z), max_idx.z);
            const long gy = BOUNDL((local_y + view_offset.y), max_idx.y);
            const long gx = BOUNDL((local_x + view_offset.x), max_idx.x);

            const long index = (gz * y_len + gy) * x_len + gx;
            if(local_ix < shared_size_zyx){
                tile[local_ix] = A[index];
            }
        }
    }
    __syncthreads();

    { // writing segment
        const long3 gid = {
            long(block_offset.x + threadIdx.x),
            long(block_offset.y + threadIdx.y),
            long(block_offset.z + threadIdx.z)};
        const long should_write = ((gid.x < x_len) && (gid.y < y_len) && (gid.z < z_len));
        if(should_write){
            const long gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr long3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr long total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            for(long i=0; i < range.z; i++){
                const long z = (threadIdx.z + i)*shared_size.y;
                for(long j=0; j < range.y; j++){
                    const long y = (z + threadIdx.y + j)*shared_size.x;
                    for(long k=0; k < range.x; k++){
                        const long x = y + threadIdx.x + k;
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
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max,
    long BNx       , long BNy>
__global__
void big_tile_3d_inlined_flat_singleDim(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len,
    const long y_len,
    const long x_len
    )
{
    constexpr long3 waste = {
        long(x_axis_min + x_axis_max),
        long(y_axis_min + y_axis_max),
        long(z_axis_min + z_axis_max)};


    const long BNxy = BNx*BNy;
    const long block_index = blockIdx.x;

    const long block_index_z = block_index / (BNxy);
    const long block_index_y = (block_index % BNxy) / BNx;
    const long block_index_x = block_index % BNx;

    const long3 block_offset = {
        long(block_index_x*X_BLOCK),
        long(block_index_y*Y_BLOCK),
        long(block_index_z*Z_BLOCK)};

    constexpr long3 shared_size = {
        long(X_BLOCK + waste.x),
        long(Y_BLOCK + waste.y),
        long(Z_BLOCK + waste.z)};

    const long XY_SIZE = X_BLOCK*Y_BLOCK;

    const long lid = threadIdx.x;
    const long z = lid / (XY_SIZE);
    const long y = (lid % XY_SIZE) / X_BLOCK;
    const long x = lid % X_BLOCK;

    constexpr long shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;
    __shared__ T tile[shared_size_zyx];
    { // reading segment
        const long3 max_idx = {
            long(x_len - 1),
            long(y_len - 1),
            long(z_len - 1)};
        const long3 view_offset = {
            block_offset.x - x_axis_min,
            block_offset.y - y_axis_min,
            block_offset.z - z_axis_min};

        constexpr long blockDimFlat = BLOCKSIZE;
        const long blockIdxFlat = (z * Y_BLOCK + y) * X_BLOCK + x;

        constexpr long lz_span = shared_size.y * shared_size.x;
        constexpr long ly_span = shared_size.x;

        constexpr long iters = CEIL_DIV(shared_size_zyx, blockDimFlat);

        long3 start;
        {
            const long lz = blockIdxFlat / lz_span;
            const long rz = blockIdxFlat % lz_span;
            const long ly = rz / ly_span;
            const long lx = rz % ly_span;
            start = { lx, ly, lz };
        }
        constexpr long added_lz = blockDimFlat / lz_span;
        constexpr long added_rz = blockDimFlat % lz_span;
        constexpr long added_ly = added_rz / ly_span;
        constexpr long added_lx = added_rz % ly_span;
        constexpr long3 added = { added_lx, added_ly, added_lz };

        long start_flat = blockIdxFlat;
        constexpr long added_flat = blockDimFlat;

        for(long i = 0; i < iters; i++){
            const long gz = BOUNDL((start.z + view_offset.z), max_idx.z);
            const long gy = BOUNDL((start.y + view_offset.y), max_idx.y);
            const long gx = BOUNDL((start.x + view_offset.x), max_idx.x);

            const long index = (gz * y_len + gy) * x_len + gx;
            if(start_flat < shared_size_zyx){
                tile[start_flat] = A[index];
            }

            start_flat += added_flat;
            start.x += added.x;
            start.y += added.y;
            start.z += added.z;
            if(start.x >= shared_size.x){
                start.x -= shared_size.x;
                start.y += 1;
            }
            if(start.y >= shared_size.y){
                start.y -= shared_size.y;
                start.z += 1;
            }
        }
    }
    __syncthreads();

    { // writing segment
        const long3 gid = {
            long(block_offset.x + x),
            long(block_offset.y + y),
            long(block_offset.z + z)};
        const long should_write = ((gid.x < x_len) && (gid.y < y_len) && (gid.z < z_len));
        if(should_write){
            const long gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr long3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr long total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            for(long i=0; i < range.z; i++){
                const long zIdx = (z + i)*shared_size.y;
                for(long j=0; j < range.y; j++){
                    const long yIdx = (zIdx + y + j)*shared_size.x;
                    for(long k=0; k < range.x; k++){
                        const long idx = yIdx + x + k;
                        sum_acc += tile[idx];
                    }
                }
            }
            sum_acc /= (T)total_range;
            out[gid_flat] = sum_acc;
        }
    }
}


template<
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void big_tile_3d_inlined_layered(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len,
    const long y_len,
    const long x_len
    )
{
    constexpr long3 waste = {
        long(x_axis_min + x_axis_max),
        long(y_axis_min + y_axis_max),
        long(z_axis_min + z_axis_max)};
    const long3 block_offset = {
        long(blockIdx.x*X_BLOCK),
        long(blockIdx.y*Y_BLOCK),
        long(blockIdx.z*Z_BLOCK)};
    constexpr long3 shared_size = {
        long(X_BLOCK + waste.x),
        long(Y_BLOCK + waste.y),
        long(Z_BLOCK + waste.z)};

    __shared__ T tile[shared_size.z][shared_size.y][shared_size.x];
    { // reading segment
        const long3 max_idx = {
            long(x_len - 1),
            long(y_len - 1),
            long(z_len - 1)};
        const long3 view_offset = {
            block_offset.x - x_axis_min,
            block_offset.y - y_axis_min,
            block_offset.z - z_axis_min};

        constexpr long BLOCK_Xm = X_BLOCK * (Z_BLOCK / 2);
        constexpr long BLOCK_Ym = Y_BLOCK * (Z_BLOCK / 2);
        constexpr long x_iters = CEIL_DIV(shared_size.x, BLOCK_Xm);
        constexpr long y_iters = CEIL_DIV(shared_size.y, BLOCK_Ym);
        const long blockIdxFlat = (threadIdx.z * Y_BLOCK + threadIdx.y) * X_BLOCK + threadIdx.x;
        const long thread_y = blockIdxFlat / BLOCK_Xm;
        const long thread_x = blockIdxFlat % BLOCK_Xm;

        for(long local_z = 0; local_z < shared_size.z; local_z++){
            const long gz = y_len * BOUNDL( local_z + view_offset.z, max_idx.z);
            for(long j = 0; j < y_iters; j++){
                const long local_y = thread_y + j*BLOCK_Ym;
                for (long k = 0; k < x_iters; k++){
                    const long local_x = thread_x + k*BLOCK_Xm;
                    const long gy = x_len * (gz + BOUNDL( local_y + view_offset.y, max_idx.y));
                    const long gx = gy + BOUNDL( local_x + view_offset.x, max_idx.x);

                    if(local_x < shared_size.x && local_y < shared_size.y){
                        tile[local_z][local_y][local_x] = A[gx];
                    }
                }
            }
        }
    }
    __syncthreads();

    { // writing segment
        const long3 gid = {
            long(block_offset.x + threadIdx.x),
            long(block_offset.y + threadIdx.y),
            long(block_offset.z + threadIdx.z)};
        const long should_write = ((gid.x < x_len) && (gid.y < y_len) && (gid.z < z_len));
        if(should_write){
            const long gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr long3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr long total_range = range.x * range.y * range.z;

            T sum_acc = 0;

            for(long i=0; i < range.z; i++){
                const long z = threadIdx.z + i;
                for(long j=0; j < range.y; j++){
                    const long y = threadIdx.y + j;
                    for(long k=0; k < range.x; k++){
                        const long x = threadIdx.x + k;
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
template<long D, long z_l, long y_l, long x_l>
__device__
inline T stencil_fun_inline_ix_3d(const T arr[z_l][y_l][x_l], const long z_off, const long y_off, const long x_off){
    T sum_acc = 0;
    for (long i = 0; i < D; i++ ){
        const long z = z_off + ixs_3d[i].z;
        const long y = y_off + ixs_3d[i].y;
        const long x = x_off + ixs_3d[i].x;
        sum_acc += arr[z][y][x];
    }
    return sum_acc / (T)D;
}


template<long D>
__global__
void global_reads_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len, //outermost
    const long y_len, //middle
    const long x_len) //innermost
{
    const long gidz = blockIdx.z*Z_BLOCK + threadIdx.z;
    const long gidy = blockIdx.y*Y_BLOCK + threadIdx.y;
    const long gidx = blockIdx.x*X_BLOCK + threadIdx.x;

    const long gindex = gidz*y_len*x_len + gidy*x_len + gidx;
    const long y_len_maxIdx = y_len - 1;
    const long x_len_maxIdx = x_len - 1;
    const long z_len_maxIdx = z_len - 1;

    if (gidx < x_len && gidy < y_len && gidz < z_len)
    {
        T sum_acc = 0;
        for (long i = 0; i < D; i++)
        {
            const long z = BOUNDL(gidz + ixs_3d[i].z, z_len_maxIdx);
            const long y = BOUNDL(gidy + ixs_3d[i].y, y_len_maxIdx);
            const long x = BOUNDL(gidx + ixs_3d[i].x, x_len_maxIdx);
            const long index = z*y_len*x_len + y*x_len + x;
            sum_acc += A[index];
        }
        out[gindex] = sum_acc / (T)D;
    }
}


template<long ixs_len,
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void small_tile_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len,
    const long y_len,
    const long x_len
    )
{
    __shared__ T tile[Z_BLOCK][Y_BLOCK][X_BLOCK];
    const long waste_x = x_axis_min + x_axis_max;
    const long waste_y = y_axis_min + y_axis_max;
    const long waste_z = z_axis_min + z_axis_max;
    const long gidx = blockIdx.x*(X_BLOCK - waste_x) + threadIdx.x - x_axis_min;
    const long gidy = blockIdx.y*(Y_BLOCK - waste_y) + threadIdx.y - y_axis_min;
    const long gidz = blockIdx.z*(Z_BLOCK - waste_z) + threadIdx.z - z_axis_min;
    const long gindex = gidz * y_len * x_len + gidy * x_len + gidx;
    const long max_x_idx = x_len - 1;
    const long max_y_idx = y_len - 1;
    const long max_z_idx = z_len - 1;

    const long x = BOUNDL(gidx, max_x_idx);
    const long y = BOUNDL(gidy, max_y_idx);
    const long z = BOUNDL(gidz, max_z_idx);

    const long index = z * y_len * x_len + y * x_len + x;

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

template<long ixs_len,
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void big_tile_3d(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long z_len,
    const long y_len,
    const long x_len
    )
{
    const long waste_x = x_axis_min + x_axis_max;
    const long waste_y = y_axis_min + y_axis_max;
    const long waste_z = z_axis_min + z_axis_max;

    const long block_offset_x = blockIdx.x*X_BLOCK;
    const long block_offset_y = blockIdx.y*Y_BLOCK;
    const long block_offset_z = blockIdx.z*Z_BLOCK;
    const long gidz = block_offset_z + threadIdx.z;
    const long gidy = block_offset_y + threadIdx.y;
    const long gidx = block_offset_x + threadIdx.x;
    const long gindex = gidz * y_len * x_len + gidy * x_len + gidx;
    const long max_z_idx = z_len - 1;
    const long max_y_idx = y_len - 1;
    const long max_x_idx = x_len - 1;

    const long shared_size_z = Z_BLOCK + waste_z;
    const long shared_size_y = Y_BLOCK + waste_y;
    const long shared_size_x = X_BLOCK + waste_x;

    __shared__ T tile[shared_size_z][shared_size_y][shared_size_x];

    const long z_iters = (shared_size_z + (Z_BLOCK-1)) / Z_BLOCK;
    const long y_iters = (shared_size_y + (Y_BLOCK-1)) / Y_BLOCK;
    const long x_iters = (shared_size_x + (X_BLOCK-1)) / X_BLOCK;

    for(long i = 0; i < z_iters; i++){
        for(long j = 0; j < y_iters; j++){
            for (long k = 0; k < x_iters; k++){
                const long local_z = threadIdx.z + i*Z_BLOCK;
                const long local_y = threadIdx.y + j*Y_BLOCK;
                const long local_x = threadIdx.x + k*X_BLOCK;

                if(local_x < shared_size_x && local_y < shared_size_y && local_z < shared_size_z){
                    const long gx = BOUNDL( local_x + block_offset_x - x_axis_min, max_x_idx);
                    const long gy = BOUNDL( local_y + block_offset_y - y_axis_min, max_y_idx);
                    const long gz = BOUNDL( local_z + block_offset_z - z_axis_min, max_z_idx);
                    const long index = gz * y_len * x_len + gy * x_len + gx;
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
