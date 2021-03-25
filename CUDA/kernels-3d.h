#ifndef KERNELS3D
#define KERNELS3D

#include <cuda_runtime.h>
#include "constants.h"

/*******************************************************************************
 * Helper functions
 */
template<const int axis_min_x, const int axis_min_y, const int axis_min_z, const int axis_max_x, const int axis_max_y, const int axis_max_z, const int sh_size_x,  const int sh_size_y,  const int sh_size_flat>
__device__
__forceinline__
void write_from_shared_flat(
    const T tile[sh_size_flat],
    T* __restrict__ const out,
    const long len_x, const long len_y, const long len_z,
    const int local_x, const int local_y, const int local_z,
    const long block_offset_x, const long block_offset_y, const long block_offset_z){
    constexpr int3 range = {
        axis_max_x + axis_min_x + 1,
        axis_max_y + axis_min_y + 1,
        axis_max_z + axis_min_z + 1};
    constexpr int total_range = range.x * range.y * range.z;

    const long gid_x = block_offset_x + local_x;
    const long gid_y = block_offset_y + local_y;
    const long gid_z = block_offset_z + local_z;

    const long should_write = ((gid_x < len_x) && (gid_y < len_y) && (gid_z < len_z));
    if(should_write){
        const long gid_flat = (gid_z * len_y + gid_y) * len_x + gid_x;

        T sum_acc = 0;
        for(int i=0; i < range.z; i++){
            const int zIdx = (local_z + i)*sh_size_y;
            for(int j=0; j < range.y; j++){
                const int yIdx = (zIdx + local_y + j)*sh_size_x;
                for(int k=0; k < range.x; k++){
                    const int idx = yIdx + local_x + k;
                    sum_acc += tile[idx];
                }
            }
        }
        sum_acc /= (T)total_range;
        out[gid_flat] = sum_acc;
    }
}

template<
    const int axis_min_x, const int axis_min_y, const int axis_min_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_flat>
__device__
__forceinline__
void bigtile_flat_loader_divrem(
    const T* __restrict__ A,
    T tile[sh_size_flat],
    const long len_x, const long len_y, const long len_z,
    const int local_flat,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    constexpr int blockDimFlat = X_BLOCK * Y_BLOCK * Z_BLOCK;
    constexpr int lz_span = sh_size_y * sh_size_x;
    constexpr int ly_span = sh_size_x;
    constexpr int iters = CEIL_DIV(sh_size_flat, blockDimFlat);

    const long max_ix_x = len_x - 1;
    const long max_ix_y = len_y - 1;
    const long max_ix_z = len_z - 1;

    const long view_offset_x = block_offset_x - axis_min_x;
    const long view_offset_y = block_offset_y - axis_min_y;
    const long view_offset_z = block_offset_z - axis_min_z;


    for(int i = 0; i < iters; i++){
        const int local_ix = (i * blockDimFlat) + local_flat;

        // div/rem
        const int local_z = local_ix / lz_span;
        const int rem_z   = local_ix % lz_span;
        const int local_y = rem_z / ly_span;
        const int local_x = rem_z % ly_span;

        const long gz = BOUNDL((local_z + view_offset_z), max_ix_z);
        const long gy = BOUNDL((local_y + view_offset_y), max_ix_y);
        const long gx = BOUNDL((local_x + view_offset_x), max_ix_x);

        const long index = (gz * len_y + gy) * len_x + gx;
        if(local_ix < sh_size_flat){
            tile[local_ix] = A[index];
        }
    }
}

template<
    const int axis_min_x, const int axis_min_y, const int axis_min_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_flat>
__device__
__forceinline__
void bigtile_flat_loader_addcarry(
    const T* __restrict__ A,
    T tile[sh_size_flat],
    const long len_x, const long len_y, const long len_z,
    const int loc_flat,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    constexpr int blockDimFlat = X_BLOCK * Y_BLOCK * Z_BLOCK;
    constexpr int sh_span_z = sh_size_y * sh_size_x;
    constexpr int sh_span_y = sh_size_x;
    constexpr int iters = CEIL_DIV(sh_size_flat, blockDimFlat);

    const long max_ix_x = len_x - 1;
    const long max_ix_y = len_y - 1;
    const long max_ix_z = len_z - 1;

    const long view_offset_x = block_offset_x - axis_min_x;
    const long view_offset_y = block_offset_y - axis_min_y;
    const long view_offset_z = block_offset_z - axis_min_z;

    int local_flat = loc_flat;
    int local_z   = loc_flat / sh_span_z;
    const int lrz = loc_flat % sh_span_z;
    int local_y = lrz / sh_span_y;
    int local_x = lrz % sh_span_y;

    const int add_z = blockDimFlat / sh_span_z;
    const int arz   = blockDimFlat % sh_span_z;
    const int add_y = arz / sh_span_y;
    const int add_x = arz % sh_span_y;

    for(int i = 0; i < iters; i++){
        const long gz = BOUNDL((local_z + view_offset_z), max_ix_z);
        const long gy = BOUNDL((local_y + view_offset_y), max_ix_y);
        const long gx = BOUNDL((local_x + view_offset_x), max_ix_x);

        const long index = (gz * len_y + gy) * len_x + gx;
        if(local_flat < sh_size_flat){
            tile[local_flat] = A[index];
        }

        // add
        local_flat += blockDimFlat;
        local_x += add_x;
        local_y += add_y;
        local_z += add_z;

        // carry
        if(local_x >= sh_size_x){
            local_x -= sh_size_x;
            local_y += 1;
        }
        if(local_y >= sh_size_y){
            local_y -= sh_size_y;
            local_z += 1;
        }
    }
}

/*******************************************************************************
 * Versions where the indices are inlined and we are provided a
 * associative and commutative operator with a neutral element
 * and then a map
 * (here summation followed by division, aka taking the average).
 */
template<
    int axis_min_z, int axis_max_z,
    int axis_min_y, int axis_max_y,
    int axis_min_x, int axis_max_x>
__global__
__launch_bounds__(BLOCKSIZE)
void global_reads_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens)
{
    constexpr int3 axis_min = { axis_min_x, axis_min_y, axis_min_z };
    constexpr int3 axis_max = { axis_max_x, axis_max_y, axis_max_z };
    constexpr int3 range = {
        axis_max.x + axis_min.x + 1,
        axis_max.y + axis_min.y + 1,
        axis_max.z + axis_min.z + 1};
    constexpr int total_range = range.z * range.y * range.x;

    const long3 gid = {
        long(blockIdx.x*X_BLOCK + threadIdx.x),
        long(blockIdx.y*Y_BLOCK + threadIdx.y),
        long(blockIdx.z*Z_BLOCK + threadIdx.z)};

    if (gid.x < lens.x && gid.y < lens.y && gid.z < lens.z)
    {
        const long gindex = (gid.z*lens.y + gid.y)*lens.x + gid.x;
        const long3 max_idx = {
            lens.x - 1L,
            lens.y - 1L,
            lens.z - 1L};

        T sum_acc = 0;
        for(int i=-axis_min_z; i <= axis_max_z; i++){
            const long z = BOUNDL(gid.z + i, max_idx.z);
            for(int j=-axis_min_y; j <= axis_max_y; j++){
                const long y = BOUNDL(gid.y + j, max_idx.y);
                for(int k=-axis_min_x; k <= axis_max_x; k++){
                    const long x = BOUNDL(gid.x + k, max_idx.x);
                    const long index = (z*lens.y + y)*lens.x + x;
                    sum_acc += A[index];
                }
            }
        }
        sum_acc /= (T)total_range;
        out[gindex] = sum_acc;
    }
}

template<
    int axis_min_z, int axis_max_z,
    int axis_min_y, int axis_max_y,
    int axis_min_x, int axis_max_x>
__global__
__launch_bounds__(BLOCKSIZE)
void global_reads_3d_inlined_singleDim_gridSpan(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens,
    const int3 grid_spans)
{
    constexpr int3 axis_min = { axis_min_x, axis_min_y, axis_min_z };
    constexpr int3 axis_max = { axis_max_x, axis_max_y, axis_max_z };
    constexpr int3 range = {
        axis_max.x + axis_min.x + 1,
        axis_max.y + axis_min.y + 1,
        axis_max.z + axis_min.z + 1};
    constexpr int total_range = range.z * range.y * range.x;

    const int group_id_flat = blockIdx.x;

    const int group_id_z = group_id_flat / grid_spans.z;
    const int grz        = group_id_flat % grid_spans.z;
    const int group_id_y = grz / grid_spans.y;
    const int group_id_x = grz % grid_spans.y;

    const int locz = threadIdx.x / (X_BLOCK * Y_BLOCK);
    const int lrz =  threadIdx.x % (X_BLOCK * Y_BLOCK);
    const int locy = lrz / X_BLOCK;
    const int locx = lrz % X_BLOCK;

    const long gidz = long(group_id_z) * Z_BLOCK + long(locz);
    const long gidy = long(group_id_y) * Y_BLOCK + long(locy);
    const long gidx = long(group_id_x) * X_BLOCK + long(locx);
    const long3 gid = { gidx, gidy, gidz };

    if (gid.x < lens.x && gid.y < lens.y && gid.z < lens.z)
    {
        const long gindex = (gid.z*lens.y + gid.y)*lens.x + gid.x;
        const long3 max_idx = {
            lens.x - 1L,
            lens.y - 1L,
            lens.z - 1L};

        T sum_acc = 0;
        for(int i=-axis_min_z; i <= axis_max_z; i++){
            for(int j=-axis_min_y; j <= axis_max_y; j++){
                for(int k=-axis_min_x; k <= axis_max_x; k++){
                    const long z = BOUNDL(gid.z + i, max_idx.z);
                    const long y = BOUNDL(gid.y + j, max_idx.y);
                    const long x = BOUNDL(gid.x + k, max_idx.x);
                    const long index = (z*lens.y + y)*lens.x + x;
                    sum_acc += A[index];
                }
            }
        }
        sum_acc /= (T)total_range;
        out[gindex] = sum_acc;
    }
}

template<
    int axis_min_z, int axis_max_z,
    int axis_min_y, int axis_max_y,
    int axis_min_x, int axis_max_x>
__global__
__launch_bounds__(BLOCKSIZE)
void global_reads_3d_inlined_singleDim_lensSpan(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens,
    const int3 place_holder)
{
    const long3 lens_spans = { 1, lens.x, lens.x*lens.y };
    constexpr int3 axis_min = { axis_min_x, axis_min_y, axis_min_z };
    constexpr int3 axis_max = { axis_max_x, axis_max_y, axis_max_z };
    constexpr int3 range = {
        axis_max.x + axis_min.x + 1,
        axis_max.y + axis_min.y + 1,
        axis_max.z + axis_min.z + 1};
    constexpr int total_range = range.z * range.y * range.x;
    constexpr int blockdim_flat = X_BLOCK * Y_BLOCK * Z_BLOCK;
    const long gid_flat = blockIdx.x * blockdim_flat + threadIdx.x;
    const long gidz = gid_flat / lens_spans.z;
    const long rgid = gid_flat % lens_spans.z;
    const long gidy = rgid / lens_spans.y;
    const long gidx = rgid % lens_spans.y;
    const long3 gid = { gidx, gidy, gidz };

    if (gid.x < lens.x && gid.y < lens.y && gid.z < lens.z)
    {
        const long gindex = (gid.z*lens.y + gid.y)*lens.x + gid.x;
        const long3 max_idx = {
            lens.x - 1L,
            lens.y - 1L,
            lens.z - 1L};

        T sum_acc = 0;
        for(int i=-axis_min_z; i <= axis_max_z; i++){
            for(int j=-axis_min_y; j <= axis_max_y; j++){
                for(int k=-axis_min_x; k <= axis_max_x; k++){
                    const long z = BOUNDL(gid.z + i, max_idx.z);
                    const long y = BOUNDL(gid.y + j, max_idx.y);
                    const long x = BOUNDL(gid.x + k, max_idx.x);
                    const long index = (z*lens.y + y)*lens.x + x;
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
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens)
{
    const long z_len = lens.z;
    const long y_len = lens.y;
    const long x_len = lens.x;
    constexpr int3 waste = {
        int(x_axis_min + x_axis_max),
        int(y_axis_min + y_axis_max),
        int(z_axis_min + z_axis_max)};
    const long3 block_offset = {
        long(blockIdx.x*X_BLOCK),
        long(blockIdx.y*Y_BLOCK),
        long(blockIdx.z*Z_BLOCK)};
    constexpr int3 shared_size = {
        int(X_BLOCK + waste.x),
        int(Y_BLOCK + waste.y),
        int(Z_BLOCK + waste.z)};

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
        constexpr int3 iters = {
            CEIL_DIV(shared_size.x, X_BLOCK),
            CEIL_DIV(shared_size.y, Y_BLOCK),
            CEIL_DIV(shared_size.z, Z_BLOCK)};

        for(int i = 0; i < iters.z; i++){
            const int local_z = threadIdx.z + i*Z_BLOCK;
            const long gid_z = y_len * BOUNDL((local_z + view_offset.z), max_idx.z);
            for(int j = 0; j < iters.y; j++){
                const int local_y = threadIdx.y + j*Y_BLOCK;
                const long gid_zy = x_len * (gid_z + BOUNDL((local_y + view_offset.y), max_idx.y));
                for (int k = 0; k < iters.x; k++){
                    const int local_x = threadIdx.x + k*X_BLOCK;
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
        const bool should_write = ((gid.x < x_len) && (gid.y < y_len) && (gid.z < z_len));
        if(should_write){
            const long gid_flat = (gid.z * y_len + gid.y) * x_len + gid.x;
            constexpr int3 range = {
                waste.x + 1,
                waste.y + 1,
                waste.z + 1};
            constexpr int total_range = range.x * range.y * range.z;

            T sum_acc = 0;
            for(int i=0; i < range.z; i++){
                const int z = threadIdx.z + i;
                for(int j=0; j < range.y; j++){
                    const int y = threadIdx.y + j;
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
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_flat(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens)
{
    constexpr int3 waste = {
        int(x_axis_min + x_axis_max),
        int(y_axis_min + y_axis_max),
        int(z_axis_min + z_axis_max)};
    constexpr int3 shared_size = {
        int(X_BLOCK + waste.x),
        int(Y_BLOCK + waste.y),
        int(Z_BLOCK + waste.z)};
    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;
    __shared__ T tile[shared_size_zyx];

    const long block_offset_x = blockIdx.x * X_BLOCK;
    const long block_offset_y = blockIdx.y * Y_BLOCK;
    const long block_offset_z = blockIdx.z * Z_BLOCK;

    const int3 local = { int(threadIdx.x), int(threadIdx.y), int(threadIdx.z) };
    const int local_flat = (threadIdx.z * Y_BLOCK + threadIdx.y) * X_BLOCK + threadIdx.x;

    bigtile_flat_loader_divrem
        <x_axis_min,y_axis_min,z_axis_min
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (A, tile
         , lens.x, lens.y, lens.z
         , local_flat
         , block_offset_x, block_offset_y, block_offset_z);

    __syncthreads();

    write_from_shared_flat
        <x_axis_min,y_axis_min,z_axis_min
        ,x_axis_max,y_axis_max,z_axis_max
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         local.x,local.y,local.z,
         block_offset_x,block_offset_y,block_offset_z);
}


template<
    int z_axis_min, int z_axis_max,
    int y_axis_min, int y_axis_max,
    int x_axis_min, int x_axis_max>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_flat_singleDim(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens,
    const int3 grid_spans)
{
    constexpr int3 waste = {
        int(x_axis_min + x_axis_max),
        int(y_axis_min + y_axis_max),
        int(z_axis_min + z_axis_max)};
    constexpr int3 shared_size = {
        int(X_BLOCK + waste.x),
        int(Y_BLOCK + waste.y),
        int(Z_BLOCK + waste.z)};
    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;
    __shared__ T tile[shared_size_zyx];

    const int block_index_z = blockIdx.x / grid_spans.z;
    const int rblock        = blockIdx.x % grid_spans.z;
    const int block_index_y = rblock / grid_spans.y;
    const int block_index_x = rblock % grid_spans.y;

    const long block_offset_x = block_index_x * X_BLOCK;
    const long block_offset_y = block_index_y * Y_BLOCK;
    const long block_offset_z = block_index_z * Z_BLOCK;

    constexpr int zb_span = X_BLOCK*Y_BLOCK;
    constexpr int yb_span = X_BLOCK;
    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / zb_span;
    const int rloc  = loc_flat % zb_span;
    const int loc_y = rloc / yb_span;
    const int loc_x = rloc % yb_span;

    bigtile_flat_loader_divrem
        <x_axis_min,y_axis_min,z_axis_min
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (A, tile
         , lens.x, lens.y, lens.z
         , loc_flat
         , block_offset_x, block_offset_y, block_offset_z);

    __syncthreads();

    write_from_shared_flat
        <x_axis_min,y_axis_min,z_axis_min
        ,x_axis_max,y_axis_max,z_axis_max
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         loc_x,loc_y,loc_z,
         block_offset_x,block_offset_y,block_offset_z);

}



/*******************************************************************************
 * Versions where the indices are inlined and we are provided a
 * associative and commutative operator with a neutral element
 * and then a map
 * (here summation followed by division, aka taking the average).
 * Futhermore these version use virtual groups
 */
template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void virtual_addcarry_global_read_3d_inlined_grid_span_singleDim(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens,
    const int num_phys_groups,
    const int iters_per_phys,
    const int3 id_add,
    const int3 virtual_grid,
    const int3 virtual_grid_spans,
    const int virtual_grid_flat_length
    )
{
    constexpr int3 range = {
        amax_x + amin_x + 1,
        amax_y + amin_y + 1,
        amax_z + amin_z + 1};
    constexpr int total_range = range.x * range.y * range.z;

    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / (group_size_x * group_size_y);
    const int rloc  = loc_flat % (group_size_x * group_size_y);
    const int loc_y = rloc / group_size_x;
    const int loc_x = rloc % group_size_x;

    int group_id_z = blockIdx.x / virtual_grid_spans.z;
    int rblock     = blockIdx.x % virtual_grid_spans.z;
    int group_id_y = rblock / virtual_grid_spans.y;
    int group_id_x = rblock % virtual_grid_spans.y;

    const long3 max_idx = {
        lens.x - 1L,
        lens.y - 1L,
        lens.z - 1L};

    int virtual_group_id_flat = blockIdx.x;
    for(int i=0; i<iters_per_phys;i++){
        if(virtual_group_id_flat < virtual_grid_flat_length){
            const long gidx = long(group_id_x) * long(group_size_x) + long(loc_x);
            const long gidy = long(group_id_y) * long(group_size_y) + long(loc_y);
            const long gidz = long(group_id_z) * long(group_size_z) + long(loc_z);

            if (gidx < lens.x && gidy < lens.y && gidz < lens.z){
                const long gindex = (gidz * lens.y + gidy)*lens.x + gidx;

                T sum_acc = 0;
                for(int i=-amin_z; i <= amax_z; i++){
                    for(int j=-amin_y; j <= amax_y; j++){
                        for(int k=-amin_x; k <= amax_x; k++){
                            const long z = BOUNDL(gidz + i, max_idx.z);
                            const long y = BOUNDL(gidy + j, max_idx.y);
                            const long x = BOUNDL(gidx + k, max_idx.x);
                            const long index = (z*lens.y + y)*lens.x + x;
                            sum_acc += A[index];
                        }
                    }
                }
                sum_acc /= (T)total_range;
                out[gindex] = sum_acc;
            }

            // add
            virtual_group_id_flat += num_phys_groups;
            group_id_x += id_add.x;
            group_id_y += id_add.y;
            group_id_z += id_add.z;

            // carry
            if(group_id_x >= virtual_grid.x){
                group_id_x -= virtual_grid.x;
                group_id_y += 1;
            }
            if(group_id_y >= virtual_grid.y){
                group_id_y -= virtual_grid.y;
                group_id_z += 1;
            }
        }
    }
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void virtual_addcarry_big_tile_3d_inlined_flat_divrem_MultiDim(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens,
    const int num_phys_groups,
    const int iters_per_phys,
    const int3 id_add,
    const int3 virtual_grid,
    const int3 virtual_grid_spans,
    const int virtual_grid_flat_length
    )
{
    constexpr int sh_size_x = amin_x + group_size_x + amax_x;
    constexpr int sh_size_y = amin_y + group_size_y + amax_y;
    constexpr int sh_size_z = amin_z + group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;
    __shared__ T tile[sh_size_flat];

    const int loc_z = threadIdx.z;
    const int loc_y = threadIdx.y;
    const int loc_x = threadIdx.x;
    const int loc_flat = (loc_z * group_size_y + loc_y)*group_size_x + loc_x;

    int group_id_z = blockIdx.x / virtual_grid_spans.z;
    int rblock     = blockIdx.x % virtual_grid_spans.z;
    int group_id_y = rblock / virtual_grid_spans.y;
    int group_id_x = rblock % virtual_grid_spans.y;

    int virtual_group_id_flat = blockIdx.x;
    for(int i=0; i<iters_per_phys;i++){
        if(virtual_group_id_flat < virtual_grid_flat_length){
            const long writeSet_offset_x = long(group_id_x) * group_size_x;
            const long writeSet_offset_y = long(group_id_y) * group_size_y;
            const long writeSet_offset_z = long(group_id_z) * group_size_z;

            bigtile_flat_loader_divrem
                <amin_x,amin_y,amin_z
                ,sh_size_x,sh_size_y,sh_size_flat>
                (A, tile
                 , lens.x, lens.y, lens.z
                 , loc_flat
                 , writeSet_offset_x, writeSet_offset_y, writeSet_offset_z);

            __syncthreads();

            write_from_shared_flat
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,sh_size_x,sh_size_y,sh_size_flat>
                (tile, out,
                 lens.x, lens.y, lens.z,
                 loc_x,loc_y,loc_z,
                 writeSet_offset_x, writeSet_offset_y, writeSet_offset_z);


            // add
            virtual_group_id_flat += num_phys_groups;
            group_id_x += id_add.x;
            group_id_y += id_add.y;
            group_id_z += id_add.z;

            // carry
            if(group_id_x >= virtual_grid.x){
                group_id_x -= virtual_grid.x;
                group_id_y += 1;
            }
            if(group_id_y >= virtual_grid.y){
                group_id_y -= virtual_grid.y;
                group_id_z += 1;
            }
        }
        // need to sync so there are no inter-iteration tile issues
        __syncthreads();
    }
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void virtual_divrem_big_tile_3d_inlined_flat_divrem_singleDim(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens,
    const int num_phys_groups,
    const int iters_per_phys,
    const int3 id_add,
    const int3 virtual_grid,
    const int3 virtual_grid_spans,
    const int virtual_grid_flat_length
    )
{
    constexpr int sh_size_x = amin_x + group_size_x + amax_x;
    constexpr int sh_size_y = amin_y + group_size_y + amax_y;
    constexpr int sh_size_z = amin_z + group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;
    __shared__ T tile[sh_size_flat];

    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / (group_size_x * group_size_y);
    const int rloc  = loc_flat % (group_size_x * group_size_y);
    const int loc_y = rloc / group_size_x;
    const int loc_x = rloc % group_size_x;

    const int start_id = blockIdx.x;
    for(int i=0; i<iters_per_phys;i++){
        const int virtual_group_id_flat = (i * num_phys_groups) + start_id;
        if(virtual_group_id_flat < virtual_grid_flat_length){

            int group_id_z = virtual_group_id_flat / virtual_grid_spans.z;
            int rblock     = virtual_group_id_flat % virtual_grid_spans.z;
            int group_id_y = rblock / virtual_grid_spans.y;
            int group_id_x = rblock % virtual_grid_spans.y;

            const long writeSet_offset_x = long(group_id_x) * group_size_x;
            const long writeSet_offset_y = long(group_id_y) * group_size_y;
            const long writeSet_offset_z = long(group_id_z) * group_size_z;

            bigtile_flat_loader_divrem
                <amin_x,amin_y,amin_z
                ,sh_size_x,sh_size_y,sh_size_flat>
                (A, tile
                 , lens.x, lens.y, lens.z
                 , loc_flat
                 , writeSet_offset_x, writeSet_offset_y, writeSet_offset_z);

            __syncthreads();

            write_from_shared_flat
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,sh_size_x,sh_size_y,sh_size_flat>
                (tile, out,
                 lens.x, lens.y, lens.z,
                 loc_x,loc_y,loc_z,
                 writeSet_offset_x, writeSet_offset_y, writeSet_offset_z);
        }
        // need to sync so there are no inter-iteration tile issues
        __syncthreads();
    }
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void virtual_addcarry_big_tile_3d_inlined_flat_divrem_singleDim(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens,
    const int num_phys_groups,
    const int iters_per_phys,
    const int3 id_add,
    const int3 virtual_grid,
    const int3 virtual_grid_spans,
    const int virtual_grid_flat_length
    )
{
    constexpr int sh_size_x = amin_x + group_size_x + amax_x;
    constexpr int sh_size_y = amin_y + group_size_y + amax_y;
    constexpr int sh_size_z = amin_z + group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;
    __shared__ T tile[sh_size_flat];

    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / (group_size_x * group_size_y);
    const int rloc  = loc_flat % (group_size_x * group_size_y);
    const int loc_y = rloc / group_size_x;
    const int loc_x = rloc % group_size_x;

    int group_id_z = blockIdx.x / virtual_grid_spans.z;
    int rblock     = blockIdx.x % virtual_grid_spans.z;
    int group_id_y = rblock / virtual_grid_spans.y;
    int group_id_x = rblock % virtual_grid_spans.y;

    int virtual_group_id_flat = blockIdx.x;
    for(int i=0; i<iters_per_phys;i++){
        if(virtual_group_id_flat < virtual_grid_flat_length){
            const long writeSet_offset_x = long(group_id_x) * group_size_x;
            const long writeSet_offset_y = long(group_id_y) * group_size_y;
            const long writeSet_offset_z = long(group_id_z) * group_size_z;

            bigtile_flat_loader_divrem
                <amin_x,amin_y,amin_z
                ,sh_size_x,sh_size_y,sh_size_flat>
                (A, tile
                 , lens.x, lens.y, lens.z
                 , loc_flat
                 , writeSet_offset_x, writeSet_offset_y, writeSet_offset_z);

            __syncthreads();

            write_from_shared_flat
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,sh_size_x,sh_size_y,sh_size_flat>
                (tile, out,
                 lens.x, lens.y, lens.z,
                 loc_x,loc_y,loc_z,
                 writeSet_offset_x, writeSet_offset_y, writeSet_offset_z);

            // add
            virtual_group_id_flat += num_phys_groups;
            group_id_x += id_add.x;
            group_id_y += id_add.y;
            group_id_z += id_add.z;

            // carry
            if(group_id_x >= virtual_grid.x){
                group_id_x -= virtual_grid.x;
                group_id_y += 1;
            }
            if(group_id_y >= virtual_grid.y){
                group_id_y -= virtual_grid.y;
                group_id_z += 1;
            }
        }
        // need to sync so there are no inter-iteration tile issues
        __syncthreads();
    }
}


template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void virtual_addcarry_big_tile_3d_inlined_flat_addcarry_singleDim(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens,
    const int num_phys_groups,
    const int iters_per_phys,
    const int3 id_add,
    const int3 virtual_grid,
    const int3 virtual_grid_spans,
    const int virtual_grid_flat_length
    )
{
    constexpr int sh_size_x = amin_x + group_size_x + amax_x;
    constexpr int sh_size_y = amin_y + group_size_y + amax_y;
    constexpr int sh_size_z = amin_z + group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;
    __shared__ T tile[sh_size_flat];

    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / (group_size_x * group_size_y);
    const int rloc  = loc_flat % (group_size_x * group_size_y);
    const int loc_y = rloc / group_size_x;
    const int loc_x = rloc % group_size_x;

    int group_id_z = blockIdx.x / virtual_grid_spans.z;
    int rblock     = blockIdx.x % virtual_grid_spans.z;
    int group_id_y = rblock / virtual_grid_spans.y;
    int group_id_x = rblock % virtual_grid_spans.y;

    int virtual_group_id_flat = blockIdx.x;
    for(int i=0; i<iters_per_phys;i++){
        if(virtual_group_id_flat < virtual_grid_flat_length){
            const long writeSet_offset_x = long(group_id_x) * group_size_x;
            const long writeSet_offset_y = long(group_id_y) * group_size_y;
            const long writeSet_offset_z = long(group_id_z) * group_size_z;

            bigtile_flat_loader_addcarry
                <amin_x,amin_y,amin_z
                ,sh_size_x,sh_size_y,sh_size_flat>
                (A, tile
                 , lens.x, lens.y, lens.z
                 , loc_flat
                 , writeSet_offset_x, writeSet_offset_y, writeSet_offset_z);

            __syncthreads();

            write_from_shared_flat
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,sh_size_x,sh_size_y,sh_size_flat>
                (tile, out,
                 lens.x, lens.y, lens.z,
                 loc_x,loc_y,loc_z,
                 writeSet_offset_x, writeSet_offset_y, writeSet_offset_z);

            // add
            virtual_group_id_flat += num_phys_groups;
            group_id_x += id_add.x;
            group_id_y += id_add.y;
            group_id_z += id_add.z;

            // carry
            if(group_id_x >= virtual_grid.x){
                group_id_x -= virtual_grid.x;
                group_id_y += 1;
            }
            if(group_id_y >= virtual_grid.y){
                group_id_y -= virtual_grid.y;
                group_id_z += 1;
            }
        }
        // need to sync so there are no inter-iteration tile issues
        __syncthreads();
    }
}

/*
 * Version that are unused for whatever reason
 *

template<
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void big_tile_3d_inlined_layered(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens)
{
    const long z_len = lens.z;
    const long y_len = lens.y;
    const long x_len = lens.x;
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

template<
    long z_axis_min, long z_axis_max,
    long y_axis_min, long y_axis_max,
    long x_axis_min, long x_axis_max>
__global__
void small_tile_3d_inlined(
    const T* __restrict__ A,
    T* __restrict__ out,
    const long3 lens)
{
    const long z_len = lens.z;
    const long y_len = lens.y;
    const long x_len = lens.x;
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
*/


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
