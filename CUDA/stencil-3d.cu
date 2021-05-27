#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>
using namespace std;
using std::cout;
using std::endl;

#include "runners.h"
#include "kernels-3d.h"

static constexpr long3 lens = {
    ((1 << 8) + 2),
    ((1 << 8) + 4),
    ((1 << 8) + 8)};
static constexpr long lens_flat = lens.x * lens.y * lens.z;
static constexpr long n_runs = 100;
static Globs
    <long3,int3
    ,Kernel3dVirtual
    ,Kernel3dPhysMultiDim
    ,Kernel3dPhysSingleDim
    > G(lens, lens_flat, n_runs);

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int amax_x, const int amax_y, const int amax_z>
__host__
void stencil_3d_cpu(
    const T* start,
    T* out)
{
    constexpr int3 range = {
        amax_x - amin_x + 1,
        amax_y - amin_y + 1,
        amax_z - amin_z + 1};
    constexpr int total_range = range.x * range.y * range.z;

    const int max_x_idx = lens.x - 1;
    const int max_y_idx = lens.y - 1;
    const int max_z_idx = lens.z - 1;
    for (int gidz = 0; gidz < lens.z; ++gidz){
        for (int gidy = 0; gidy < lens.y; ++gidy){
            for (int gidx = 0; gidx < lens.x; ++gidx){
                T arr[total_range];
                for(int i=0; i < range.z; i++){
                    for(int j=0; j < range.y; j++){
                        for(int k=0; k < range.x; k++){
                            const long z = bound<(amin_z<0),long>(gidz + (i + amin_z), max_z_idx);
                            const long y = bound<(amin_y<0),long>(gidy + (j + amin_y), max_y_idx);
                            const long x = bound<(amin_x<0),long>(gidx + (k + amin_x), max_x_idx);
                            const long index = (z*lens.y + y)*lens.x + x;
                            const int flat_idx = (i*range.y + j)*range.x + k;
                            arr[flat_idx] = start[index];
                        }
                    }
                }
                T lambda_res = stencil_fun_3d<amin_x, amin_y, amin_z, amax_x, amax_y, amax_z>(arr);
                out[(gidz*lens.y + gidy)*lens.x + gidx] = lambda_res;
            }
        }
    }
}

template<
    const int amin_x, const int amin_y, const int amin_z,
    const int amax_x, const int amax_y, const int amax_z>
__host__
void run_cpu_3d(T* cpu_out)
{
    T* cpu_in = (T*)malloc(lens_flat*sizeof(T));
    srand(1);
    for (int i = 0; i < lens_flat; ++i)
    {
        cpu_in[i] = (T)rand();
    }

    struct timeval t_startpar, t_endpar, t_diffpar;
    gettimeofday(&t_startpar, NULL);
    {
        stencil_3d_cpu<amin_x, amin_y, amin_z, amax_x, amax_y, amax_z>(cpu_in,cpu_out);
    }
    gettimeofday(&t_endpar, NULL);
    timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
    const unsigned long elapsed = (t_diffpar.tv_sec*1e6+t_diffpar.tv_usec) / 1000;
    const unsigned long seconds = elapsed / 1000;
    const unsigned long microseconds = elapsed % 1000;
    printf("cpu c 3d for 1 run : %lu.%03lu seconds\n", seconds, microseconds);

    free(cpu_in);
}

template<
    const int amin_z, const int amax_z,
    const int amin_y, const int amax_y,
    const int amin_x, const int amax_x,
    const int group_size_x,  const int group_size_y, const int group_size_z,
    const int strip_pow_x, const int strip_pow_y, const int strip_pow_z>
__host__
void doTest_3D(const int physBlocks)
{
    static_assert(amin_z <= amax_z, "invalid setup");
    static_assert(amin_y <= amax_y, "invalid setup");
    static_assert(amin_x <= amax_x, "invalid setup");
    const int z_range = (amax_z + 1) - amin_z;
    const int y_range = (amax_y + 1) - amin_y;
    const int x_range = (amax_x + 1) - amin_x;

#ifdef Jacobi3D
    const int ixs_len = z_range + y_range + x_range - 2;
#else
    const int ixs_len = z_range * y_range * x_range;
#endif
    cout << "ixs[" << ixs_len << "] = (zr,yr,xr) = (" << amin_z << "..." << amax_z << ", " << amin_y << "..." << amax_y << ", " << amin_x << "..." << amax_x << ")\n";

    constexpr long len = lens_flat;

    T* cpu_out = (T*)malloc(len*sizeof(T));
    run_cpu_3d<amin_x, amin_y, amin_z, amax_x, amax_y, amax_z>(cpu_out);

    constexpr int blockDim_flat = group_size_x * group_size_y * group_size_z;
    constexpr int3 virtual_grid = {
        divUp((int)lens.x, group_size_x),
        divUp((int)lens.y, group_size_y),
        divUp((int)lens.z, group_size_z)};
    constexpr dim3 block_3d(group_size_x,group_size_y,group_size_z);
    constexpr dim3 block_3d_flat(group_size_x*group_size_y*group_size_z,1,1);
    constexpr dim3 grid_3d(virtual_grid.x, virtual_grid.y, virtual_grid.z);
    constexpr int virtual_grid_flat = virtual_grid.x * virtual_grid.y * virtual_grid.z;
    constexpr int3 virtual_grid_spans = { 1, virtual_grid.x, virtual_grid.x * virtual_grid.y };

    constexpr int sh_size_x = group_size_x + amax_x - amin_x;
    constexpr int sh_size_y = group_size_y + amax_y - amin_y;
    constexpr int sh_size_z = group_size_z + amax_z - amin_z;
    constexpr int sh_size_flat = sh_size_x * sh_size_y * sh_size_z;
    constexpr int sh_mem_size_flat = sh_size_flat * sizeof(T);

    cout << "Blockdim z,y,x = " << group_size_z << ", " << group_size_y << ", " << group_size_x << endl;
    //printf("virtual number of blocks = %d\n", virtual_grid_flat);
    {
        /*{
            cout << "## Benchmark 3d global read - inlined ixs - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = global_reads_3d_inlined
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            for(int i=0;i<4;i++){
                G.do_run_multiDim(kfun, cpu_out, grid_3d, block_3d, 1, false); // warmup as it is first kernel
            }
            G.do_run_multiDim(kfun, cpu_out, grid_3d, block_3d, 1);
        }
        {
            cout << "## Benchmark 3d global read - inlined ixs - singleDim grid - grid span ##";
            Kernel3dPhysSingleDim kfun = global_reads_3d_inlined_singleDim_gridSpan
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_singleDim(kfun, cpu_out, virtual_grid_flat, blockDim_flat, virtual_grid_spans, 1);
        }

        {
            constexpr long lens_grid = divUp(lens_flat, long(blockDim_flat));
            constexpr int3 lens_spans = { 1, int(lens.x), int(lens.x*lens.y) };
            cout << "## Benchmark 3d global read - inlined ixs - singleDim grid - lens span ##";
            Kernel3dPhysSingleDim kfun = global_reads_3d_inlined_singleDim_lensSpan
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_singleDim(kfun, cpu_out, lens_grid, blockDim_flat, lens_spans, 1);
        }

        {
            cout << "## Benchmark 3d global read - inlined idxs - virtual (add/carry) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_global_read_3d_inlined_grid_span_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid, 1);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - cube load - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_multiDim(kfun, cpu_out, grid_3d, block_3d, sh_mem_size_flat);
        }*/

        /*{
            cout << "## Benchmark 3d big tile - inlined idxs - transaction aligned loads - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined_trx_align
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_multiDim(kfun, cpu_out, grid_3d, block_3d, sh_mem_size_flat);
        }*/
        /*
        {
            cout << "## Benchmark 3d big tile - inlined idxs - forced coalesced flat load (div/rem) - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined_flat_forced_coalesced
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_multiDim(kfun, cpu_out, grid_3d, block_3d, sh_mem_size_flat);
        }
        */
        /*{
            cout << "## Benchmark 3d big tile - inlined idxs - cube reshape (div/rem) - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined_cube_reshape
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_multiDim(kfun, cpu_out, grid_3d, block_3d, sh_mem_size_flat);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - flat load (div/rem) - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined_flat
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_multiDim(kfun, cpu_out, grid_3d, block_3d, sh_mem_size_flat);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - flat load (div/rem) - singleDim grid ##";
            Kernel3dPhysSingleDim kfun = big_tile_3d_inlined_flat_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_singleDim(kfun, cpu_out, virtual_grid_flat, blockDim_flat, virtual_grid, sh_mem_size_flat);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - flat load (add/carry) - singleDim grid ##";
            Kernel3dPhysSingleDim kfun = big_tile_3d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_singleDim(kfun, cpu_out, virtual_grid_flat, blockDim_flat, virtual_grid, sh_mem_size_flat);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - virtual (add/carry) - flat load (div/rem) - multiDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_divrem_MultiDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual(kfun, cpu_out, physBlocks, block_3d, virtual_grid, sh_mem_size_flat);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - virtual (add/carry) - flat load (div/rem) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_divrem_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual(kfun, cpu_out, physBlocks, block_3d_flat, virtual_grid, sh_mem_size_flat);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - virtual (rem/div) - flat load (div/rem) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_divrem_big_tile_3d_inlined_flat_divrem_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual(kfun, cpu_out, physBlocks, block_3d_flat, virtual_grid, sh_mem_size_flat);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - virtual (add/carry) - flat load (add/carry) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual(kfun, cpu_out, physBlocks, block_3d_flat, virtual_grid, sh_mem_size_flat);
        }*/

        constexpr int strip_x = 1 << strip_pow_x;
        constexpr int strip_y = 1 << strip_pow_y;
        constexpr int strip_z = 1 << strip_pow_z;
        constexpr int strip_size_x = group_size_x*strip_x;
        constexpr int strip_size_y = group_size_y*strip_y;
        constexpr int strip_size_z = group_size_z*strip_z;

        constexpr int sh_x = strip_size_x + amax_x - amin_x;
        constexpr int sh_y = strip_size_y + amax_y - amin_y;
        constexpr int sh_z = strip_size_z + amax_z - amin_z;
        constexpr int strip_sh_total = sh_x * sh_y * sh_z;
        constexpr int strip_sh_total_mem_usage = strip_sh_total * sizeof(T);
        //printf("shared memory used = %d B\n", strip_sh_total_mem_usage);
        constexpr int max_shared_mem = 0xc000; // 48KiB
        static_assert(strip_sh_total_mem_usage < max_shared_mem,
                "Current configuration requires too much shared memory\n");

        // this should technically be measured but it it div by (2^n) so it is very fast, and won't matter much.
        const int3 strip_grid = {
            int(divUp(lens.x, long(strip_size_x))),
            int(divUp(lens.y, long(strip_size_y))),
            int(divUp(lens.z, long(strip_size_z)))};
        const int strip_grid_flat = product(strip_grid);

        {
            //cout << "## Benchmark 3d big tile - inlined idxs - stripmined: ";
            //printf("strip_size=[%d][%d][%d]f32 ", strip_size_z, strip_size_y, strip_size_x);
            //cout << "- flat load (add/carry) - singleDim grid ##";
            Kernel3dPhysSingleDim kfun = stripmine_big_tile_3d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z
                ,strip_x,strip_y,strip_z
                >;
            G.do_run_singleDim(kfun, cpu_out, strip_grid_flat, blockDim_flat, strip_grid, strip_sh_total_mem_usage,false);
            G.do_run_singleDim(kfun, cpu_out, strip_grid_flat, blockDim_flat, strip_grid, strip_sh_total_mem_usage);
        }
        /*{
            cout << "## Benchmark 3d big tile - inlined idxs - stripmined: ";
            printf("strip_size=[%d][%d][%d]f32 ", strip_size_z, strip_size_y, strip_size_x);
            cout << "- cube loader - singleDim grid ##";
            Kernel3dPhysSingleDim kfun = stripmine_big_tile_3d_inlined_cube_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z
                ,strip_x,strip_y,strip_z
                >;
            G.do_run_singleDim(kfun, cpu_out, strip_grid_flat, blockDim_flat, strip_grid, strip_sh_total_mem_usage);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - stripmined: ";
            printf("strip_size=[%d][%d][%d]f32 ", strip_size_z, strip_size_y, strip_size_x);
            cout << "- virtual (add/carry) - flat load (add/carry) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_stripmine_big_tile_3d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z
                ,strip_x,strip_y,strip_z
                >;
            G.do_run_virtual(kfun, cpu_out, physBlocks, block_3d_flat, strip_grid, strip_sh_total_mem_usage);
        }*/
    }

    free(cpu_out);

    (void)block_3d;
    (void)block_3d_flat;
    (void)grid_3d;
    (void)virtual_grid_spans;
    (void)virtual_grid_flat;
    (void)sh_mem_size_flat;
}

template
    <const int gx, const int gy, const int gz
    ,const int sx, const int sy, const int sz>
__host__
void testStrips(const int physBlocks){
    doTest_3D<-1,0, -1,0, -1,0, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,1, -1,0, -1,0, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,1, -1,1, -1,0, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,1, -1,0, -1,1, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,1, -1,1, -1,1, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,2, -1,1, -1,1, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,2, -1,2, -1,1, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,2, -1,1, -1,2, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,2, -1,2, -1,2, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,3, -1,2, -1,2, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,3, -1,3, -1,2, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,3, -1,2, -1,3, gx,gy,gz,sx,sy,sz>(physBlocks);
    doTest_3D<-1,3, -1,3, -1,3, gx,gy,gz,sx,sy,sz>(physBlocks);
}

__host__
int main()
{

    constexpr int gps_x = 32;
    constexpr int gps_y = 4;
    constexpr int gps_z = 2;

    constexpr int gps_flat = gps_x * gps_y * gps_z;
    int physBlocks = getPhysicalBlockCount<gps_flat>();
#ifdef Jacobi3D
    cout << "running Jacobi 3D" << endl;
#else
    cout << "running Dense stencil with mean" << endl;
#endif

    // small test samples.
    /*
    doTest_3D<0,1,0,1,0,1, gps_x,gps_y,gps_z,1,1,1>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, gps_x,gps_y,gps_z,1,1,1>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, gps_x,gps_y,gps_z,1,1,1>(physBlocks);
    // 0 < amin
    doTest_3D<1,2,1,2,1,2, gps_x,gps_y,gps_z,0,1,2>(physBlocks);
    // 0 > amax
    doTest_3D<-2,-1,-2,-1,-2,-1, gps_x,gps_y,gps_z,0,1,2>(physBlocks);

    // z axis is only in use
    doTest_3D<-1,1,0,0,0,0, gps_x,gps_y,gps_z,0,0,3>(physBlocks);
    doTest_3D<-2,2,0,0,0,0, gps_x,gps_y,gps_z,0,0,3>(physBlocks);
    doTest_3D<-3,3,0,0,0,0, gps_x,gps_y,gps_z,0,0,3>(physBlocks);
    doTest_3D<-4,4,0,0,0,0, gps_x,gps_y,gps_z,0,0,3>(physBlocks);
    doTest_3D<-5,5,0,0,0,0, gps_x,gps_y,gps_z,0,0,3>(physBlocks);

    // z-y axis are only in use
    doTest_3D<-1,1,-1,1,0,0, gps_x,gps_y,gps_z,0,1,1>(physBlocks);
    doTest_3D<-2,2,-2,2,0,0, gps_x,gps_y,gps_z,0,1,1>(physBlocks);
    doTest_3D<-3,3,-3,3,0,0, gps_x,gps_y,gps_z,0,0,1>(physBlocks);
    doTest_3D<-4,4,-4,4,0,0, gps_x,gps_y,gps_z,0,0,1>(physBlocks);
    doTest_3D<-5,5,-5,5,0,0, gps_x,gps_y,gps_z,0,0,0>(physBlocks);

    // z-x axis are only in use
    doTest_3D<-1,1,0,0,-1,1, gps_x,gps_y,gps_z,1,0,1>(physBlocks);
    doTest_3D<-2,2,0,0,-2,2, gps_x,gps_y,gps_z,1,0,1>(physBlocks);
    doTest_3D<-3,3,0,0,-3,3, gps_x,gps_y,gps_z,0,0,1>(physBlocks);
    doTest_3D<-4,4,0,0,-4,4, gps_x,gps_y,gps_z,0,0,1>(physBlocks);
    doTest_3D<-5,5,0,0,-5,5, gps_x,gps_y,gps_z,0,0,0>(physBlocks);
    */
    // all axis are in use
    doTest_3D<0,1,0,1,0,1, 32,2,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, 32,2,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,0,1, 32,2,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,-1,1, 32,2,2,0,0,0>(physBlocks);

    doTest_3D<0,1,0,1,0,1, 32,4,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, 32,4,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,0,1, 32,4,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,-1,1, 32,4,2,0,0,0>(physBlocks);

    doTest_3D<0,1,0,1,0,1, 32,8,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, 32,8,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,0,1, 32,8,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,-1,1, 32,8,2,0,0,0>(physBlocks);

    doTest_3D<0,1,0,1,0,1, 32,8,4,0,0,0>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, 32,8,4,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,0,1, 32,8,4,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,-1,1, 32,8,4,0,0,0>(physBlocks);
    //blocksize test
    /*
    doTest_3D<0,1,0,1,0,1, 32,8,1,0,0,0>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, 32,8,1,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,0,1, 32,8,1,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,-1,1, 32,8,1,0,0,0>(physBlocks);

    doTest_3D<0,1,0,1,0,1, 32,4,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, 32,4,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,0,1, 32,4,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,-1,1, 32,4,2,0,0,0>(physBlocks);

    doTest_3D<0,1,0,1,0,1, 32,8,4,0,0,0>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, 32,8,4,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,0,1, 32,8,4,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,-1,1, 32,8,4,0,0,0>(physBlocks);

    doTest_3D<0,1,0,1,0,1, 32,16,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,0,1,0,1, 32,16,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,0,1, 32,16,2,0,0,0>(physBlocks);
    doTest_3D<-1,1,-1,1,-1,1, 32,16,2,0,0,0>(physBlocks);
    */
    /*
    doTest_3D<-2,2,-2,2,-2,2, gps_x,gps_y,gps_z,0,0,1>(physBlocks);
    doTest_3D<-3,3,-3,3,-3,3, gps_x,gps_y,gps_z,0,0,1>(physBlocks);
    doTest_3D<-4,4,-4,4,-4,4, gps_x,gps_y,gps_z,0,0,1>(physBlocks);
    doTest_3D<-5,5,-5,5,-5,5, gps_x,gps_y,gps_z,0,0,0>(physBlocks);
    */

    constexpr int gx=32;
    constexpr int gy=8;
    constexpr int gz=4;
    //printf("strip_size=[%d][%d][%d]f32 ", 0, 0, 0);
    //testStrips<gx,gy,gz,0,0,0>(physBlocks);
    //testStrips<gx,gy,gz,1,0,0>(physBlocks);
    //testStrips<gx,gy,gz,0,1,0>(physBlocks);
    //testStrips<gx,gy,gz,0,0,1>(physBlocks);
    //testStrips<gx,gy,gz,1,0,1>(physBlocks);
    //testStrips<gx,gy,gz,1,1,0>(physBlocks);
    //testStrips<gx,gy,gz,0,1,1>(physBlocks);
    //testStrips<gx,gy,gz,1,1,1>(physBlocks);
    //testStrips<gx,gy,gz,0,1,1>(physBlocks);
    //testStrips<gx,gy,gz,0,0,1>(physBlocks);
    //testStrips<gx,gy,gz,1,0,2>(physBlocks);
    //testStrips<gx,gy,gz,0,2,1>(physBlocks);
    //testStrips<gx,gy,gz,1,2,0>(physBlocks);
    //testStrips<gx,gy,gz,2,1,0>(physBlocks);
    //testStrips<gx,gy,gz,2,0,1>(physBlocks);
    return 0;
}
