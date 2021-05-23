#ifndef KERNELS3D
#define KERNELS3D

#include <cuda_runtime.h>
#include "constants.h"

//#define Jacobi3D
/*******************************************************************************
 * Helper functions
 */
template<
    const int amin_x, const int amin_y, const int amin_z,
    const int amax_x, const int amax_y, const int amax_z>
__device__ __host__
__forceinline__
T stencil_fun_3d(const T arr[]){
    constexpr int3 range = {
        amax_x - amin_x + 1,
        amax_y - amin_y + 1,
        amax_z - amin_z + 1};
    constexpr int total_range = range.x * range.y * range.z;

    T sum_acc = 0;
    for(int i=0; i < range.z; i++){
        for(int j=0; j < range.y; j++){
            for(int k=0; k < range.x; k++){
#ifdef Jacobi3D
                constexpr int zc = range.z / 2;
                constexpr int yc = range.y / 2;
                constexpr int xc = range.x / 2;
                const bool zn = i == zc;
                const bool yn = j == yc;
                const bool xn = k == xc;
                if((zn && yn) || (zn && xn) || (yn && xn))
#endif
                    sum_acc += arr[i*range.y*range.x + j*range.x + k];
            }
        }
    }
    sum_acc /= (T)total_range;
    return sum_acc;
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int amax_x, const int amax_y, const int amax_z>
__device__
__forceinline__
void read_write_from_global(
    const T* A,
    T* out,
    const long lens_x, const long lens_y, const long lens_z,
    const long gid_x, const long gid_y, const long gid_z)
{
    constexpr int3 range = {
        amax_x - amin_x + 1,
        amax_y - amin_y + 1,
        amax_z - amin_z + 1};
    constexpr int total_range = range.x * range.y * range.z;

    const long gindex = (gid_z*lens_y + gid_y)*lens_x + gid_x;
    const long max_idx_z = lens_z - 1L;
    const long max_idx_y = lens_y - 1L;
    const long max_idx_x = lens_x - 1L;

    T vals[total_range];
    for(int i=0; i < range.z; i++){
        for(int j=0; j < range.y; j++){
            for(int k=0; k < range.x; k++){
                const long z = bound<(amin_z<0),long>(gid_z + (i + amin_z), max_idx_z);
                const long y = bound<(amin_y<0),long>(gid_y + (j + amin_y), max_idx_y);
                const long x = bound<(amin_x<0),long>(gid_x + (k + amin_x), max_idx_x);
                const long index = (z*lens_y + y)*lens_x + x;
                const int flat_idx = (i*range.y + j)*range.x + k;
                vals[flat_idx] = A[index];
            }
        }
    }
    out[gindex] = stencil_fun_3d<amin_x, amin_y, amin_z, amax_x, amax_y, amax_z>(vals);
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int amax_x, const int amax_y, const int amax_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_flat>
__device__
__forceinline__
void write_from_shared_flat(
    const T tile[sh_size_flat],
    T* out,
    const long len_x, const long len_y, const long len_z,
    const int local_x, const int local_y, const int local_z,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    constexpr int3 range = {
        amax_x - amin_x + 1,
        amax_y - amin_y + 1,
        amax_z - amin_z + 1};
    constexpr int total_range = range.x * range.y * range.z;

    T vals[total_range];

    const long gid_x = block_offset_x + local_x;
    const long gid_y = block_offset_y + local_y;
    const long gid_z = block_offset_z + local_z;

    const long should_write = ((gid_x < len_x) && (gid_y < len_y) && (gid_z < len_z));
    if(should_write){
        const long gid_flat = (gid_z * len_y + gid_y) * len_x + gid_x;

        for(int i=0; i < range.z; i++){
            const int zIdx = (local_z + i)*sh_size_y;
            for(int j=0; j < range.y; j++){
                const int yIdx = (zIdx + local_y + j)*sh_size_x;
                for(int k=0; k < range.x; k++){
                    const int idx = yIdx + local_x + k;
                    const int flat_idx = i*range.y*range.x + j*range.x + k;
                    vals[flat_idx] = tile[idx];
                }
            }
        }
        out[gid_flat] = stencil_fun_3d<amin_x, amin_y, amin_z, amax_x, amax_y, amax_z>(vals);
    }
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_z,
    const int group_size_x,  const int group_size_y, const int group_size_z>
__device__
__forceinline__
void bigtile_cube_block_loader(
    const T* A,
    T tile[],
    const long len_x, const long len_y, const long len_z,
    const int local_x, const int local_y, const int local_z,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    const long3 max_idx = {
        long(len_x - 1),
        long(len_y - 1),
        long(len_z - 1)};
    const long3 view_offset = {
        block_offset_x + amin_x,
        block_offset_y + amin_y,
        block_offset_z + amin_z};
    constexpr int3 iters = {
        divUp(sh_size_x, group_size_x),
        divUp(sh_size_y, group_size_y),
        divUp(sh_size_z, group_size_z)};

    for(int i = 0; i < iters.z; i++){
        const int lz = local_z + i*group_size_z;
        const long gid_z = len_y * bound<(amin_z<0),long>((lz + view_offset.z), max_idx.z);
        for(int j = 0; j < iters.y; j++){
            const int ly = local_y + j*group_size_y;
            const long gid_zy = len_x * (gid_z + bound<(amin_y<0),long>((ly + view_offset.y), max_idx.y));
            for (int k = 0; k < iters.x; k++){
                const int lx = local_x + k*group_size_x;
                const long gid_zyx = gid_zy + bound<(amin_x<0),long>((lx + view_offset.x), max_idx.x);
                const int local_flat = (lz * sh_size_y + ly) * sh_size_x + lx;
                if(lz < sh_size_z && ly < sh_size_y && lx < sh_size_x){
                    tile[local_flat] = A[gid_zyx];
                }
            }
        }
    }
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_flat,
    const int group_size_x,  const int group_size_y, const int group_size_z>
__device__
__forceinline__
void bigtile_flat_loader_divrem(
    const T* A,
    T tile[sh_size_flat],
    const long len_x, const long len_y, const long len_z,
    const int local_flat,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    constexpr int blockDimFlat = group_size_x * group_size_y * group_size_z;
    constexpr int lz_span = sh_size_y * sh_size_x;
    constexpr int ly_span = sh_size_x;
    constexpr int iters = divUp(sh_size_flat, blockDimFlat);

    const long max_ix_x = len_x - 1;
    const long max_ix_y = len_y - 1;
    const long max_ix_z = len_z - 1;

    const long view_offset_x = block_offset_x + amin_x;
    const long view_offset_y = block_offset_y + amin_y;
    const long view_offset_z = block_offset_z + amin_z;


    for(int i = 0; i < iters; i++){
        const int local_ix = (i * blockDimFlat) + local_flat;

        // div/rem
        const int local_z = local_ix / lz_span;
        const int rem_z   = local_ix % lz_span;
        const int local_y = rem_z / ly_span;
        const int local_x = rem_z % ly_span;

        const long gz = bound<(amin_z<0),long>((local_z + view_offset_z), max_ix_z);
        const long gy = bound<(amin_y<0),long>((local_y + view_offset_y), max_ix_y);
        const long gx = bound<(amin_x<0),long>((local_x + view_offset_x), max_ix_x);

        const long index = (gz * len_y + gy) * len_x + gx;
        if(local_ix < sh_size_flat){
            tile[local_ix] = A[index];
        }
    }
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_flat,
    const int group_size_x,  const int group_size_y, const int group_size_z>
__device__
__forceinline__
void bigtile_flat_loader_addcarry(
    const T* A,
    T tile[sh_size_flat],
    const long len_x, const long len_y, const long len_z,
    const int loc_flat,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    constexpr int blockDimFlat = group_size_x * group_size_y * group_size_z;
    constexpr int sh_span_z = sh_size_y * sh_size_x;
    constexpr int sh_span_y = sh_size_x;
    constexpr int iters = divUp(sh_size_flat, blockDimFlat);

    const long max_ix_x = len_x - 1;
    const long max_ix_y = len_y - 1;
    const long max_ix_z = len_z - 1;

    constexpr int loader_cap_x = sh_size_x + amin_x;
    constexpr int loader_cap_y = sh_size_y + amin_y;

    int local_flat = loc_flat;
    int local_z   = (loc_flat / sh_span_z) + amin_z;
    const int lrz = loc_flat % sh_span_z;
    int local_y = (lrz / sh_span_y) + amin_y;
    int local_x = (lrz % sh_span_y) + amin_x;

    constexpr int add_z = blockDimFlat / sh_span_z;
    constexpr int arz   = blockDimFlat % sh_span_z;
    constexpr int add_y = arz / sh_span_y;
    constexpr int add_x = arz % sh_span_y;

    for(int i = 0; i < iters; i++){
        const long gx = bound<(amin_x<0),long>((local_x + block_offset_x), max_ix_x);
        const long gy = bound<(amin_y<0),long>((local_y + block_offset_y), max_ix_y);
        const long gz = bound<(amin_z<0),long>((local_z + block_offset_z), max_ix_z);

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
        if(local_x >= loader_cap_x){
            local_x -= sh_size_x;
            local_y += 1;
        }
        if(local_y >= loader_cap_y){
            local_y -= sh_size_y;
            local_z += 1;
        }
    }
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_z,
    const int group_size_x,  const int group_size_y, const int group_size_z>
__device__
__forceinline__
void bigtile_flat_loader_transactionAligned(
    const T* A,
    T tile[],
    const long len_x, const long len_y, const long len_z,
    const int loc_flat,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    constexpr int blockDimFlat = group_size_x * group_size_y * group_size_z;

    const long max_ix_x = len_x - 1;
    const long max_ix_y = len_y - 1;
    const long max_ix_z = len_z - 1;

    constexpr int elements_per_warp = 32;
    constexpr int nr_of_warps = blockDimFlat / 32;
    constexpr int chunks_per_row = 1 + divUp((sh_size_x - 1), elements_per_warp);
    constexpr int nr_of_rows = sh_size_y * sh_size_z;
    constexpr int max_chunks = nr_of_rows * chunks_per_row;
    constexpr int iters = divUp(max_chunks, nr_of_warps);

    constexpr int chunk_span_z = sh_size_y * chunks_per_row;
    constexpr int chunk_span_y = chunks_per_row;

    constexpr int loader_cap_x = chunks_per_row;
    constexpr int loader_cap_y = sh_size_y;

    constexpr int and_round_down_32 = ~(32-1);

    const int warp_id = loc_flat / elements_per_warp;
    const int lane_id = loc_flat & (32-1);

    int tnx_id_z = warp_id / chunk_span_z;
    int tnx_id__ = warp_id % chunk_span_z;
    int tnx_id_y = tnx_id__ / chunk_span_y;
    int tnx_id_x = tnx_id__ % chunk_span_y;

    constexpr int add_z = nr_of_warps / chunk_span_z;
    constexpr int add__ = nr_of_warps % chunk_span_z;
    constexpr int add_y = add__ / chunk_span_y;
    constexpr int add_x = add__ % chunk_span_y;

    const long boam_z = block_offset_z + amin_z;
    const long boam_y = block_offset_y + amin_y;
    const long boam_x = block_offset_x + amin_x;

    for(int i = 0; i < iters; i++){
        const long gz = bound<(amin_z<0),long>((tnx_id_z + boam_z), max_ix_z);
        const long gy = bound<(amin_y<0),long>((tnx_id_y + boam_y), max_ix_y);
        const long index_zy = (gz * len_y + gy) * len_x;
        const long index_bzyx = index_zy + boam_x;
        const int gxtr_diff = index_bzyx - (index_bzyx & and_round_down_32);

        const int sh_id_x_noff = lane_id + (tnx_id_x * elements_per_warp);
        const int sh_id_x = sh_id_x_noff - gxtr_diff;

        const long ugx = boam_x + sh_id_x;
        const long gx = bound<(amin_x<0),long>(ugx, max_ix_x);
        const long index = index_zy + gx;

        const int sh_id_flat = (tnx_id_z * sh_size_y + tnx_id_y) * sh_size_x + sh_id_x;
        if(0 <= sh_id_x && sh_id_x < sh_size_x){
            tile[sh_id_flat] = A[index];
        }

        // add
        tnx_id_x += add_x;
        tnx_id_y += add_y;
        tnx_id_z += add_z;

        // carry
        if(tnx_id_x >= loader_cap_x){
            tnx_id_x -= loader_cap_x;
            tnx_id_y += 1;
        }
        if(tnx_id_y >= loader_cap_y){
            tnx_id_y -= loader_cap_y;
            tnx_id_z += 1;
        }
    }
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_z,
    const int group_size_x,  const int group_size_y, const int group_size_z>
__device__
__forceinline__
void big_tile_3d_inlined_flat_forced_coalesced_loader(
    const T* A,
    T tile[],
    const long len_x, const long len_y, const long len_z,
    const int loc_flat,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    constexpr int blockDimFlat = group_size_x * group_size_y * group_size_z;

    const long max_ix_x = len_x - 1;
    const long max_ix_y = len_y - 1;
    const long max_ix_z = len_z - 1;

    constexpr int warp_size = 32;
    constexpr int n_warps = blockDimFlat / 32;
    constexpr int chunks_per_row = divUp(sh_size_x, warp_size);
    constexpr int n_rows = sh_size_y * sh_size_z;
    constexpr int n_chunks = n_rows * chunks_per_row;
    constexpr int iters = divUp(n_chunks, n_warps);

    constexpr int chunk_span_z = sh_size_y * chunks_per_row;
    constexpr int chunk_span_y = chunks_per_row;

    const int warp_id = loc_flat / warp_size;
    const int lane_id = loc_flat & (32-1);

    const long view_offset_z = block_offset_z + amin_z;
    const long view_offset_y = block_offset_y + amin_y;
    const long view_offset_x = block_offset_x + amin_x;

    for(int i = 0; i < iters; i++){
        const int chunk_ix = (i * warp_size) + warp_id;
        if(chunk_ix >= n_chunks){ break; }

        // div/rem
        const int tnx_id_z = chunk_ix / chunk_span_z;
        const int tnx_id__ = chunk_ix % chunk_span_z;
        const int tnx_id_y = tnx_id__ / chunk_span_y;
        const int tnx_id_x = tnx_id__ % chunk_span_y;

        const int sh_id_x = lane_id + (tnx_id_x * warp_size);
        const int sh_id_flat = (tnx_id_z * sh_size_y + tnx_id_y) * sh_size_x + sh_id_x;

        const long gz = bound<(amin_z<0),long>((tnx_id_z + view_offset_z), max_ix_z);
        const long gy = bound<(amin_y<0),long>((tnx_id_y + view_offset_y), max_ix_y);
        const long gx = bound<(amin_x<0),long>(( sh_id_x + view_offset_x), max_ix_x);
        const long index = (gz * len_y + gy) * len_x + gx;

        if(sh_id_x < sh_size_x){
            tile[sh_id_flat] = A[index];
        }
    }
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int sh_size_x,  const int sh_size_y,  const int sh_size_z,
    const int group_size_x,  const int group_size_y, const int group_size_z>
__device__
__forceinline__
void bigtile_cube_reshape_loader(
    const T* A,
    T tile[],
    const long len_x, const long len_y, const long len_z,
    const int loc_flat,
    const long block_offset_x, const long block_offset_y, const long block_offset_z)
{
    constexpr int blockDimFlat = group_size_x * group_size_y * group_size_z;

    const long max_ix_x = len_x - 1;
    const long max_ix_y = len_y - 1;
    const long max_ix_z = len_z - 1;

    constexpr int warp_size = 32;
    constexpr int n_warps = blockDimFlat / 32; // assume this is 30 or 32
    constexpr int chunks_per_row = divUp(sh_size_x, warp_size);;
    constexpr int n_rows = sh_size_y * sh_size_z;

    constexpr int row_size = chunks_per_row * warp_size;
    constexpr int rows_per_iter = n_warps / chunks_per_row;
    constexpr int iters = divUp(n_rows, rows_per_iter);

    const int loc_x = loc_flat & (row_size-1);
    const int row_id = loc_flat / row_size;

    const long view_offset_z = block_offset_z + amin_z;
    const long view_offset_y = block_offset_y + amin_y;
    const long view_offset_x = block_offset_x + amin_x;

    for(int i = 0; i < iters; i++){
        const int row_flat = (i * rows_per_iter) + row_id;
        if(row_flat >= n_rows){ break; }

        // div/rem
        const int row_z = row_flat / sh_size_y;
        const int row_y = row_flat % sh_size_y;

        const int sh_id_flat = (row_z * sh_size_y + row_y) * sh_size_x + loc_x;

        const long gz = bound<(amin_z<0),long>((row_z + view_offset_z), max_ix_z);
        const long gy = bound<(amin_y<0),long>((row_y + view_offset_y), max_ix_y);
        const long gx = bound<(amin_x<0),long>((loc_x + view_offset_x), max_ix_x);
        const long index = (gz * len_y + gy) * len_x + gx;

        if(loc_x < sh_size_x){
            tile[sh_id_flat] = A[index];
        }
    }
}

/*******************************************************************************
 * Versions where the indices are inlined and we are provided a
 * associative and commutative operator with a neutral element
 * and then a map
 * (here summation followed by division, aka taking the average).
 */
template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void global_reads_3d_inlined(
    const T* A,
    T* out,
    const long3 lens)
{
    const long3 gid = {
        long(blockIdx.x)*group_size_x + threadIdx.x,
        long(blockIdx.y)*group_size_y + threadIdx.y,
        long(blockIdx.z)*group_size_z + threadIdx.z};

    if (gid.x < lens.x && gid.y < lens.y && gid.z < lens.z)
    {
        read_write_from_global
            <amin_x,amin_y,amin_z
            ,amax_x,amax_y,amax_z>
            (A, out, lens.x, lens.y, lens.z, gid.x, gid.y, gid.z);
    }
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void global_reads_3d_inlined_singleDim_gridSpan(
    const T* A,
    T* out,
    const long3 lens,
    const int3 grid_spans)
{
    const int group_id_flat = blockIdx.x;

    const int group_id_z = group_id_flat / grid_spans.z;
    const int grz        = group_id_flat % grid_spans.z;
    const int group_id_y = grz / grid_spans.y;
    const int group_id_x = grz % grid_spans.y;
    const int locz = threadIdx.x / (group_size_x * group_size_y);
    const int lrz =  threadIdx.x % (group_size_x * group_size_y);
    const int locy = lrz / group_size_x;
    const int locx = lrz % group_size_x;

    const long gidz = long(group_id_z) * long(group_size_z) + long(locz);
    const long gidy = long(group_id_y) * long(group_size_y) + long(locy);
    const long gidx = long(group_id_x) * long(group_size_x) + long(locx);

    if (gidx < lens.x && gidy < lens.y && gidz < lens.z)
    {
        read_write_from_global
            <amin_x,amin_y,amin_z
            ,amax_x,amax_y,amax_z>
            (A, out, lens.x, lens.y, lens.z, gidx, gidy, gidz);
    }
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void global_reads_3d_inlined_singleDim_lensSpan(
    const T* A,
    T* out,
    const long3 lens,
    const int3 place)
{
    const long3 lens_spans = { 1L, long(place.y), long(place.z) };

    constexpr long blockdim_flat = group_size_x * group_size_y * group_size_z;

    const long group_id_flat = blockIdx.x;
    const long gid_flat = group_id_flat * blockdim_flat + threadIdx.x;
    const long gidz = gid_flat / lens_spans.z;
    const long gid_ = gid_flat % lens_spans.z;
    const long gidy = gid_ / lens_spans.y;
    const long gidx = gid_ % lens_spans.y;

    if (gidx < lens.x && gidy < lens.y && gidz < lens.z){
        read_write_from_global
            <amin_x,amin_y,amin_z
            ,amax_x,amax_y,amax_z>
            (A, out, lens.x, lens.y, lens.z, gidx, gidy, gidz);
    }
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined(
    const T* A,
    T* out,
    const long3 lens)
{
    extern __shared__ T tile[];
    constexpr int3 waste = {
        int(-amin_x + amax_x),
        int(-amin_y + amax_y),
        int(-amin_z + amax_z)};
    const long3 block_offset = {
        long(blockIdx.x*group_size_x),
        long(blockIdx.y*group_size_y),
        long(blockIdx.z*group_size_z)};
    constexpr int3 shared_size = {
        int(group_size_x + waste.x),
        int(group_size_y + waste.y),
        int(group_size_z + waste.z)};

    constexpr int shared_size_zyx = product(shared_size);
    const int3 local = { int(threadIdx.x),int(threadIdx.y),int(threadIdx.z) };

    bigtile_cube_block_loader
        <amin_x,amin_y,amin_z
        ,shared_size.x,shared_size.y,shared_size.z
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , local.x,local.y,local.z
         , block_offset.x, block_offset.y, block_offset.z);

    __syncthreads();

    write_from_shared_flat
        <amin_x,amin_y,amin_z
        ,amax_x,amax_y,amax_z
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         local.x,local.y,local.z,
         block_offset.x,block_offset.y,block_offset.z);
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_flat(
    const T* A,
    T* out,
    const long3 lens)
{
    extern __shared__ T tile[];
    constexpr int3 shared_size = {
        int(group_size_x + (-amin_x + amax_x)),
        int(group_size_y + (-amin_y + amax_y)),
        int(group_size_z + (-amin_z + amax_z))};
    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;

    const long block_offset_x = blockIdx.x * group_size_x;
    const long block_offset_y = blockIdx.y * group_size_y;
    const long block_offset_z = blockIdx.z * group_size_z;

    const int3 local = { int(threadIdx.x), int(threadIdx.y), int(threadIdx.z) };
    const int local_flat = (threadIdx.z * group_size_y + threadIdx.y) * group_size_x + threadIdx.x;

    bigtile_flat_loader_divrem
        <amin_x,amin_y,amin_z
        ,shared_size.x,shared_size.y,shared_size_zyx
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , local_flat
         , block_offset_x, block_offset_y, block_offset_z);

    __syncthreads();

    write_from_shared_flat
        <amin_x,amin_y,amin_z
        ,amax_x,amax_y,amax_z
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         local.x,local.y,local.z,
         block_offset_x,block_offset_y,block_offset_z);
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_trx_align(
    const T* A,
    T* out,
    const long3 lens)
{
    extern __shared__ T tile[];
    constexpr int3 shared_size = {
        int(group_size_x + (-amin_x + amax_x)),
        int(group_size_y + (-amin_y + amax_y)),
        int(group_size_z + (-amin_z + amax_z))};
    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;

    const long block_offset_x = blockIdx.x * group_size_x;
    const long block_offset_y = blockIdx.y * group_size_y;
    const long block_offset_z = blockIdx.z * group_size_z;

    const int3 local = { int(threadIdx.x), int(threadIdx.y), int(threadIdx.z) };
    const int local_flat = (threadIdx.z * group_size_y + threadIdx.y) * group_size_x + threadIdx.x;

    bigtile_flat_loader_transactionAligned
        <amin_x,amin_y,amin_z
        ,shared_size.x,shared_size.y,shared_size.z
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , local_flat
         , block_offset_x, block_offset_y, block_offset_z);

    __syncthreads();

    write_from_shared_flat
        <amin_x,amin_y,amin_z
        ,amax_x,amax_y,amax_z
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         local.x,local.y,local.z,
         block_offset_x,block_offset_y,block_offset_z);
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_flat_forced_coalesced(
    const T* A,
    T* out,
    const long3 lens)
{
    extern __shared__ T tile[];
    constexpr int3 shared_size = {
        int(group_size_x + (-amin_x + amax_x)),
        int(group_size_y + (-amin_y + amax_y)),
        int(group_size_z + (-amin_z + amax_z))};
    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;

    const long block_offset_x = blockIdx.x * group_size_x;
    const long block_offset_y = blockIdx.y * group_size_y;
    const long block_offset_z = blockIdx.z * group_size_z;

    const int3 local = { int(threadIdx.x), int(threadIdx.y), int(threadIdx.z) };
    const int local_flat = (threadIdx.z * group_size_y + threadIdx.y) * group_size_x + threadIdx.x;

    big_tile_3d_inlined_flat_forced_coalesced_loader
        <amin_x,amin_y,amin_z
        ,shared_size.x,shared_size.y,shared_size.z
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , local_flat
         , block_offset_x, block_offset_y, block_offset_z);

    __syncthreads();

    write_from_shared_flat
        <amin_x,amin_y,amin_z
        ,amax_x,amax_y,amax_z
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         local.x,local.y,local.z,
         block_offset_x,block_offset_y,block_offset_z);
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_cube_reshape(
    const T* A,
    T* out,
    const long3 lens)
{
    extern __shared__ T tile[];
    constexpr int3 shared_size = {
        int(group_size_x + (-amin_x + amax_x)),
        int(group_size_y + (-amin_y + amax_y)),
        int(group_size_z + (-amin_z + amax_z))};
    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;

    const long block_offset_x = blockIdx.x * group_size_x;
    const long block_offset_y = blockIdx.y * group_size_y;
    const long block_offset_z = blockIdx.z * group_size_z;

    const int3 local = { int(threadIdx.x), int(threadIdx.y), int(threadIdx.z) };
    const int local_flat = (threadIdx.z * group_size_y + threadIdx.y) * group_size_x + threadIdx.x;

    bigtile_cube_reshape_loader
        <amin_x,amin_y,amin_z
        ,shared_size.x,shared_size.y,shared_size.z
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , local_flat
         , block_offset_x, block_offset_y, block_offset_z);

    __syncthreads();

    write_from_shared_flat
        <amin_x,amin_y,amin_z
        ,amax_x,amax_y,amax_z
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         local.x,local.y,local.z,
         block_offset_x,block_offset_y,block_offset_z);
}


template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_flat_singleDim(
    const T* A,
    T* out,
    const long3 lens,
    const int3 grid)
{
    extern __shared__ T tile[];
    constexpr int3 shared_size = {
        int(group_size_x + (-amin_x + amax_x)),
        int(group_size_y + (-amin_y + amax_y)),
        int(group_size_z + (-amin_z + amax_z))};
    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;

    const int grid_spans_z = grid.x * grid.y;
    const int grid_spans_y = grid.x;
    const int block_index_z = blockIdx.x / grid_spans_z;
    const int rblock        = blockIdx.x % grid_spans_z;
    const int block_index_y = rblock / grid_spans_y;
    const int block_index_x = rblock % grid_spans_y;

    const long block_offset_x = block_index_x * group_size_x;
    const long block_offset_y = block_index_y * group_size_y;
    const long block_offset_z = block_index_z * group_size_z;

    constexpr int zb_span = group_size_x*group_size_y;
    constexpr int yb_span = group_size_x;
    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / zb_span;
    const int rloc  = loc_flat % zb_span;
    const int loc_y = rloc / yb_span;
    const int loc_x = rloc % yb_span;

    bigtile_flat_loader_divrem
        <amin_x,amin_y,amin_z
        ,shared_size.x,shared_size.y,shared_size_zyx
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , loc_flat
         , block_offset_x, block_offset_y, block_offset_z);

    __syncthreads();

    write_from_shared_flat
        <amin_x,amin_y,amin_z
        ,amax_x,amax_y,amax_z
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         loc_x,loc_y,loc_z,
         block_offset_x,block_offset_y,block_offset_z);

}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_flat_addcarry_singleDim(
    const T* A,
    T* out,
    const long3 lens,
    const int3 grid)
{
    extern __shared__ T tile[];
    constexpr int3 shared_size = {
        int(group_size_x + (-amin_x + amax_x)),
        int(group_size_y + (-amin_y + amax_y)),
        int(group_size_z + (-amin_z + amax_z))};
    constexpr int shared_size_zyx = shared_size.z * shared_size.y * shared_size.x;

    const int grid_spans_z = grid.x * grid.y;
    const int grid_spans_y = grid.x;
    const int block_index_z = blockIdx.x / grid_spans_z;
    const int rblock        = blockIdx.x % grid_spans_z;
    const int block_index_y = rblock / grid_spans_y;
    const int block_index_x = rblock % grid_spans_y;

    const long block_offset_x = block_index_x * group_size_x;
    const long block_offset_y = block_index_y * group_size_y;
    const long block_offset_z = block_index_z * group_size_z;

    constexpr int zb_span = group_size_x*group_size_y;
    constexpr int yb_span = group_size_x;
    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / zb_span;
    const int rloc  = loc_flat % zb_span;
    const int loc_y = rloc / yb_span;
    const int loc_x = rloc % yb_span;

    bigtile_flat_loader_addcarry
        <amin_x,amin_y,amin_z
        ,shared_size.x,shared_size.y,shared_size_zyx
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , loc_flat
         , block_offset_x, block_offset_y, block_offset_z);

    __syncthreads();

    write_from_shared_flat
        <amin_x,amin_y,amin_z
        ,amax_x,amax_y,amax_z
        ,shared_size.x,shared_size.y,shared_size_zyx>
        (tile, out,
         lens.x, lens.y, lens.z,
         loc_x,loc_y,loc_z,
         block_offset_x,block_offset_y,block_offset_z);

}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
, const int strip_x, const int strip_y, const int strip_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void stripmine_big_tile_3d_inlined_flat_addcarry_singleDim(
    const T* A,
    T* out,
    const long3 lens,
    const int3 strip_grid
    )
{
    extern __shared__ T tile[];
    constexpr int sh_size_x = -amin_x + strip_x*group_size_x + amax_x;
    constexpr int sh_size_y = -amin_y + strip_y*group_size_y + amax_y;
    constexpr int sh_size_z = -amin_z + strip_z*group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;

    constexpr long strip_id_scaler_x = strip_x * group_size_x;
    constexpr long strip_id_scaler_y = strip_y * group_size_y;
    constexpr long strip_id_scaler_z = strip_z * group_size_z;

    const int strip_grid_spans_z = strip_grid.x * strip_grid.y;
    const int strip_grid_spans_y = strip_grid.x;

    const long strip_id_flat = blockIdx.x;
    const long strip_id_z = strip_id_flat / strip_grid_spans_z;
    const long strip_id__ = strip_id_flat % strip_grid_spans_z;
    const long strip_id_y = strip_id__ / strip_grid_spans_y;
    const long strip_id_x = strip_id__ % strip_grid_spans_y;

    constexpr int grp_span_z = group_size_x*group_size_y;
    constexpr int grp_span_y = group_size_x;
    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / grp_span_z;
    const int loc__ = loc_flat % grp_span_z;
    const int loc_y = loc__ / grp_span_y;
    const int loc_x = loc__ % grp_span_y;

    const long block_offset_x = strip_id_x * strip_id_scaler_x;
    const long block_offset_y = strip_id_y * strip_id_scaler_y;
    const long block_offset_z = strip_id_z * strip_id_scaler_z;

    bigtile_flat_loader_addcarry
        <amin_x,amin_y,amin_z
        ,sh_size_x,sh_size_y,sh_size_flat
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , loc_flat
         , block_offset_x, block_offset_y, block_offset_z);

    // the tile has to be fully done being loaded before we start reading
    __syncthreads();
    for(int i__ = 0; i__ < strip_z; i__++){
        for(int j__ = 0; j__ < strip_y; j__++){
            for(int k__ = 0; k__ < strip_x; k__++){
                // tile_offsets implicitly also handle the change in group_id
                const int tile_offset_x = loc_x + (k__ * group_size_x);
                const int tile_offset_y = loc_y + (j__ * group_size_y);
                const int tile_offset_z = loc_z + (i__ * group_size_z);

                write_from_shared_flat
                    <amin_x,amin_y,amin_z
                    ,amax_x,amax_y,amax_z
                    ,sh_size_x,sh_size_y,sh_size_flat>
                    (tile, out,
                     lens.x, lens.y, lens.z,
                     tile_offset_x,tile_offset_y,tile_offset_z,
                     block_offset_x, block_offset_y, block_offset_z);
            }
        }
    }
}

template
< const int amin_x, const int amin_y, const int amin_z
, const int amax_x, const int amax_y, const int amax_z
, const int group_size_x,  const int group_size_y, const int group_size_z
, const int strip_x, const int strip_y, const int strip_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void stripmine_big_tile_3d_inlined_cube_singleDim(
    const T* A,
    T* out,
    const long3 lens,
    const int3 strip_grid
    )
{
    extern __shared__ T tile[];
    constexpr int sh_size_x = -amin_x + strip_x*group_size_x + amax_x;
    constexpr int sh_size_y = -amin_y + strip_y*group_size_y + amax_y;
    constexpr int sh_size_z = -amin_z + strip_z*group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;

    constexpr long strip_id_scaler_x = strip_x * group_size_x;
    constexpr long strip_id_scaler_y = strip_y * group_size_y;
    constexpr long strip_id_scaler_z = strip_z * group_size_z;

    const int strip_grid_spans_z = strip_grid.x * strip_grid.y;
    const int strip_grid_spans_y = strip_grid.x;

    const long strip_id_flat = blockIdx.x;
    const long strip_id_z = strip_id_flat / strip_grid_spans_z;
    const long strip_id__ = strip_id_flat % strip_grid_spans_z;
    const long strip_id_y = strip_id__ / strip_grid_spans_y;
    const long strip_id_x = strip_id__ % strip_grid_spans_y;

    constexpr int grp_span_z = group_size_x*group_size_y;
    constexpr int grp_span_y = group_size_x;
    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / grp_span_z;
    const int loc__ = loc_flat % grp_span_z;
    const int loc_y = loc__ / grp_span_y;
    const int loc_x = loc__ % grp_span_y;

    const long block_offset_x = strip_id_x * strip_id_scaler_x;
    const long block_offset_y = strip_id_y * strip_id_scaler_y;
    const long block_offset_z = strip_id_z * strip_id_scaler_z;

    bigtile_cube_block_loader
        <amin_x,amin_y,amin_z
        ,sh_size_x,sh_size_y,sh_size_z
        ,group_size_x,group_size_y,group_size_z>
        (A, tile
         , lens.x, lens.y, lens.z
         , loc_x, loc_y, loc_z
         , block_offset_x, block_offset_y, block_offset_z);

    // the tile has to be fully done being loaded before we start reading
    __syncthreads();
    for(int i__ = 0; i__ < strip_z; i__++){
        for(int j__ = 0; j__ < strip_y; j__++){
            for(int k__ = 0; k__ < strip_x; k__++){
                // tile_offsets implicitly also handle the change in group_id
                const int tile_offset_x = loc_x + (k__ * group_size_x);
                const int tile_offset_y = loc_y + (j__ * group_size_y);
                const int tile_offset_z = loc_z + (i__ * group_size_z);

                write_from_shared_flat
                    <amin_x,amin_y,amin_z
                    ,amax_x,amax_y,amax_z
                    ,sh_size_x,sh_size_y,sh_size_flat>
                    (tile, out,
                     lens.x, lens.y, lens.z,
                     tile_offset_x,tile_offset_y,tile_offset_z,
                     block_offset_x, block_offset_y, block_offset_z);
            }
        }
    }
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
    const T* A,
    T* out,
    const long3 lens,
    const int num_phys_groups,
    const int3 virtual_grid
    )
{
    const int3 virtual_grid_spans = create_spans(virtual_grid);
    const int virtual_grid_flat = product(virtual_grid);
    const int iters_per_phys = divUp(virtual_grid_flat, num_phys_groups);

    const int id_add_z = num_phys_groups / virtual_grid_spans.z;
    const int id_add_r = num_phys_groups % virtual_grid_spans.z;
    const int id_add_y = id_add_r / virtual_grid_spans.y;
    const int id_add_x = id_add_r % virtual_grid_spans.y;
    const int3 id_add = { id_add_x, id_add_y, id_add_z };

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
        if(virtual_group_id_flat < virtual_grid_flat){
            const long gidx = long(group_id_x) * long(group_size_x) + long(loc_x);
            const long gidy = long(group_id_y) * long(group_size_y) + long(loc_y);
            const long gidz = long(group_id_z) * long(group_size_z) + long(loc_z);

            if (gidx < lens.x && gidy < lens.y && gidz < lens.z){
                read_write_from_global
                    <amin_x,amin_y,amin_z
                    ,amax_x,amax_y,amax_z>
                    (A, out, lens.x, lens.y, lens.z, gidx, gidy, gidz);
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
    const T* A,
    T* out,
    const long3 lens,
    const int num_phys_groups,
    const int3 virtual_grid
    )
{
    extern __shared__ T tile[];
    constexpr int sh_size_x = -amin_x + group_size_x + amax_x;
    constexpr int sh_size_y = -amin_y + group_size_y + amax_y;
    constexpr int sh_size_z = -amin_z + group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;

    const int3 virtual_grid_spans = create_spans(virtual_grid);
    const int virtual_grid_flat = product(virtual_grid);
    const int iters_per_phys = divUp(virtual_grid_flat, num_phys_groups);

    const int id_add_z = num_phys_groups / virtual_grid_spans.z;
    const int id_add_r = num_phys_groups % virtual_grid_spans.z;
    const int id_add_y = id_add_r / virtual_grid_spans.y;
    const int id_add_x = id_add_r % virtual_grid_spans.y;
    const int3 id_add = { id_add_x, id_add_y, id_add_z };

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
        if(virtual_group_id_flat < virtual_grid_flat){
            const long writeSet_offset_x = long(group_id_x) * group_size_x;
            const long writeSet_offset_y = long(group_id_y) * group_size_y;
            const long writeSet_offset_z = long(group_id_z) * group_size_z;

            bigtile_flat_loader_divrem
                <amin_x,amin_y,amin_z
                ,sh_size_x,sh_size_y,sh_size_flat
                ,group_size_x,group_size_y,group_size_z>
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
    const T* A,
    T* out,
    const long3 lens,
    const int num_phys_groups,
    const int3 virtual_grid
    )
{
    extern __shared__ T tile[];
    constexpr int sh_size_x = -amin_x + group_size_x + amax_x;
    constexpr int sh_size_y = -amin_y + group_size_y + amax_y;
    constexpr int sh_size_z = -amin_z + group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;

    const int3 virtual_grid_spans = create_spans(virtual_grid);
    const int virtual_grid_flat = product(virtual_grid);
    const int iters_per_phys = divUp(virtual_grid_flat, num_phys_groups);

    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / (group_size_x * group_size_y);
    const int rloc  = loc_flat % (group_size_x * group_size_y);
    const int loc_y = rloc / group_size_x;
    const int loc_x = rloc % group_size_x;

    const int start_id = blockIdx.x;
    for(int i=0; i<iters_per_phys;i++){
        const int virtual_group_id_flat = (i * num_phys_groups) + start_id;
        if(virtual_group_id_flat < virtual_grid_flat){

            int group_id_z = virtual_group_id_flat / virtual_grid_spans.z;
            int rblock     = virtual_group_id_flat % virtual_grid_spans.z;
            int group_id_y = rblock / virtual_grid_spans.y;
            int group_id_x = rblock % virtual_grid_spans.y;

            const long writeSet_offset_x = long(group_id_x) * group_size_x;
            const long writeSet_offset_y = long(group_id_y) * group_size_y;
            const long writeSet_offset_z = long(group_id_z) * group_size_z;

            bigtile_flat_loader_divrem
                <amin_x,amin_y,amin_z
                ,sh_size_x,sh_size_y,sh_size_flat
                ,group_size_x,group_size_y,group_size_z>
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
    const T* A,
    T* out,
    const long3 lens,
    const int num_phys_groups,
    const int3 virtual_grid
    )
{
    extern __shared__ T tile[];
    constexpr int sh_size_x = -amin_x + group_size_x + amax_x;
    constexpr int sh_size_y = -amin_y + group_size_y + amax_y;
    constexpr int sh_size_z = -amin_z + group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;

    const int3 virtual_grid_spans = create_spans(virtual_grid);
    const int virtual_grid_flat = product(virtual_grid);
    const int iters_per_phys = divUp(virtual_grid_flat, num_phys_groups);

    const int id_add_z = num_phys_groups / virtual_grid_spans.z;
    const int id_add_r = num_phys_groups % virtual_grid_spans.z;
    const int id_add_y = id_add_r / virtual_grid_spans.y;
    const int id_add_x = id_add_r % virtual_grid_spans.y;
    const int3 id_add = { id_add_x, id_add_y, id_add_z };

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
        if(virtual_group_id_flat < virtual_grid_flat){
            const long writeSet_offset_x = long(group_id_x) * group_size_x;
            const long writeSet_offset_y = long(group_id_y) * group_size_y;
            const long writeSet_offset_z = long(group_id_z) * group_size_z;

            bigtile_flat_loader_divrem
                <amin_x,amin_y,amin_z
                ,sh_size_x,sh_size_y,sh_size_flat
                ,group_size_x,group_size_y,group_size_z>
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
    const T* A,
    T* out,
    const long3 lens,
    const int num_phys_groups,
    const int3 virtual_grid
    )
{
    extern __shared__ T tile[];
    constexpr int sh_size_x = -amin_x + group_size_x + amax_x;
    constexpr int sh_size_y = -amin_y + group_size_y + amax_y;
    constexpr int sh_size_z = -amin_z + group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;

    const int3 virtual_grid_spans = create_spans(virtual_grid);
    const int virtual_grid_flat = product(virtual_grid);
    const int iters_per_phys = divUp(virtual_grid_flat, num_phys_groups);

    const int id_add_z = num_phys_groups / virtual_grid_spans.z;
    const int id_add_r = num_phys_groups % virtual_grid_spans.z;
    const int id_add_y = id_add_r / virtual_grid_spans.y;
    const int id_add_x = id_add_r % virtual_grid_spans.y;
    const int3 id_add = { id_add_x, id_add_y, id_add_z };

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
        if(virtual_group_id_flat < virtual_grid_flat){
            const long writeSet_offset_x = long(group_id_x) * group_size_x;
            const long writeSet_offset_y = long(group_id_y) * group_size_y;
            const long writeSet_offset_z = long(group_id_z) * group_size_z;

            bigtile_flat_loader_addcarry
                <amin_x,amin_y,amin_z
                ,sh_size_x,sh_size_y,sh_size_flat
                ,group_size_x,group_size_y,group_size_z>
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
, const int strip_x, const int strip_y, const int strip_z
>
__global__
__launch_bounds__(BLOCKSIZE)
void virtual_addcarry_stripmine_big_tile_3d_inlined_flat_addcarry_singleDim(
    const T* A,
    T* out,
    const long3 lens,
    const int num_phys_groups,
    const int3 virtual_strip
    )
{
    extern __shared__ T tile[];
    constexpr int sh_size_x = -amin_x + strip_x*group_size_x + amax_x;
    constexpr int sh_size_y = -amin_y + strip_y*group_size_y + amax_y;
    constexpr int sh_size_z = -amin_z + strip_z*group_size_z + amax_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;

    constexpr long strip_id_scaler_x = strip_x * group_size_x;
    constexpr long strip_id_scaler_y = strip_y * group_size_y;
    constexpr long strip_id_scaler_z = strip_z * group_size_z;

    //--
    const int virtual_strip_spans_z = virtual_strip.x *  virtual_strip.y;
    const int virtual_strip_spans_y = virtual_strip.x;
    const int virtual_strip_flat = product(virtual_strip);
    const int iters_per_phys = divUp(virtual_strip_flat, num_phys_groups);

    const int strip_id_add_z = num_phys_groups / virtual_strip_spans_z;
    const int strip_id_add__ = num_phys_groups % virtual_strip_spans_z;
    const int strip_id_add_y = strip_id_add__ / virtual_strip_spans_y;
    const int strip_id_add_x = strip_id_add__ % virtual_strip_spans_y;

    // --
    const int loc_flat = threadIdx.x;
    const int loc_z = loc_flat / (group_size_x * group_size_y);
    const int rloc  = loc_flat % (group_size_x * group_size_y);
    const int loc_y = rloc / group_size_x;
    const int loc_x = rloc % group_size_x;

    const int start_group_id_flat = blockIdx.x;
    int strip_id_z   = start_group_id_flat / virtual_strip_spans_z;
    const int rblock = start_group_id_flat % virtual_strip_spans_z;
    int strip_id_y   = rblock / virtual_strip_spans_y;
    int strip_id_x   = rblock % virtual_strip_spans_y;

    for(int i=0; i<iters_per_phys;i++){
        const long base_block_offset_x = strip_id_x * strip_id_scaler_x;
        const long base_block_offset_y = strip_id_y * strip_id_scaler_y;
        const long base_block_offset_z = strip_id_z * strip_id_scaler_z;

        bigtile_flat_loader_addcarry
            <amin_x,amin_y,amin_z
            ,sh_size_x,sh_size_y,sh_size_flat
            ,group_size_x,group_size_y,group_size_z>
            (A, tile
             , lens.x, lens.y, lens.z
             , loc_flat
             , base_block_offset_x, base_block_offset_y, base_block_offset_z);

        // the tile has to be fully done being loaded before we start reading
        __syncthreads();
        for(int i__ = 0; i__ < strip_z; i__++){
            for(int j__ = 0; j__ < strip_y; j__++){
                for(int k__ = 0; k__ < strip_x; k__++){
                    // tile_offsets implicitly also handle the change in group_id
                    const int tile_offset_x = loc_x + (k__ * group_size_x);
                    const int tile_offset_y = loc_y + (j__ * group_size_y);
                    const int tile_offset_z = loc_z + (i__ * group_size_z);

                    write_from_shared_flat
                        <amin_x,amin_y,amin_z
                        ,amax_x,amax_y,amax_z
                        ,sh_size_x,sh_size_y,sh_size_flat>
                        (tile, out,
                         lens.x, lens.y, lens.z,
                         tile_offset_x,tile_offset_y,tile_offset_z,
                         base_block_offset_x, base_block_offset_y, base_block_offset_z);
                }
            }
        }
        // add
        strip_id_x += strip_id_add_x;
        strip_id_y += strip_id_add_y;
        strip_id_z += strip_id_add_z;

        // carry
        if(strip_id_x >= virtual_strip.x){
            strip_id_x -= virtual_strip.x;
            strip_id_y += 1;
        }
        if(strip_id_y >= virtual_strip.y){
            strip_id_y -= virtual_strip.y;
            strip_id_z += 1;
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
__launch_bounds__(BLOCKSIZE)
void big_tile_3d_inlined_layered(
    const T* A,
    T* out,
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
__launch_bounds__(BLOCKSIZE)
void small_tile_3d_inlined(
    const T* A,
    T* out,
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
__launch_bounds__(BLOCKSIZE)
void global_reads_3d(
    const T* A,
    T* out,
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
__global__\n__launch_bounds__(BLOCKSIZE)
void small_tile_3d(
    const T* A,
    T* out,
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
__global__\n__launch_bounds__(BLOCKSIZE)
void big_tile_3d(
    const T* A,
    T* out,
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
