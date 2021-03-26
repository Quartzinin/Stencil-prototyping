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
    ((1 << 7) - 1),
    ((1 << 8) - 1),
    ((1 << 9) - 1)};
static constexpr long lens_flat = lens.x * lens.y * lens.z;
static constexpr long n_runs = 100;
static Globs<long3> G(lens, lens_flat, n_runs);

template<int D>
__host__
void stencil_3d_cpu(
    const T* start,
    const int3* idxs,
    T* out)
{
    const int max_x_idx = lens.x - 1;
    const int max_y_idx = lens.y - 1;
    const int max_z_idx = lens.z - 1;
    for (int i = 0; i < lens.z; ++i)
    {
        for (int j = 0; j < lens.y; ++j)
        {
            for (int k = 0; k < lens.x; ++k)
            {
                T arr[D];
                for (int p = 0; p < D; ++p)
                {
                    int z = BOUND(i + idxs[p].z, max_z_idx);
                    int y = BOUND(j + idxs[p].y, max_y_idx);
                    int x = BOUND(k + idxs[p].x, max_x_idx);
                    int index = (z * lens.y + y) * lens.x + x;
                    arr[p] = start[index];
                }

                T lambda_res = stencil_fun_cpu<D>(arr);
                out[(i*lens.y + j)*lens.x + k] = lambda_res;
            }
        }
    }
}

template<int D>
__host__
void run_cpu_3d(const int3* idxs, T* cpu_out)
{
    T* cpu_in = (T*)malloc(lens_flat*sizeof(T));

    for (int i = 0; i < lens_flat; ++i)
    {
        cpu_in[i] = (T)(i+1);
    }

    struct timeval t_startpar, t_endpar, t_diffpar;
    gettimeofday(&t_startpar, NULL);
    {
        stencil_3d_cpu<D>(cpu_in,idxs,cpu_out);
    }
    gettimeofday(&t_endpar, NULL);
    timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
    const unsigned long elapsed = (t_diffpar.tv_sec*1e6+t_diffpar.tv_usec) / 1000;
    const unsigned long seconds = elapsed / 1000;
    const unsigned long microseconds = elapsed % 1000;
    printf("cpu c 3d for 1 run : %lu.%03lu seconds\n", seconds, microseconds);

    free(cpu_in);
}

int getPhysicalBlockCount(void){
    cudaDeviceProp dprop;
    // assume that device 0 exists and is the desired one.
    cudaGetDeviceProperties(&dprop, 0);
    int maxThreadsPerSM = dprop.maxThreadsPerMultiProcessor;
    int SM_count = dprop.multiProcessorCount;
    printf("Setting:\n");
    printf("\tmaxThreadsPerSM = %d\n", maxThreadsPerSM);
    printf("\tSM_count = %d\n", SM_count);

    int runningBlocksPerSM = maxThreadsPerSM / BLOCKSIZE;
    printf("\trunningBlocksPerSM = %d\n", runningBlocksPerSM);
    int runningBlocksTotal = runningBlocksPerSM * SM_count;
    printf("\trunningBlocksTotal = %d\n", runningBlocksTotal);
    int runningThreadsTotal = runningBlocksTotal * BLOCKSIZE;
    printf("\trunningThreadsTotal = %d\n", runningThreadsTotal);
    printf("\tphysical number of blocks = %d\n", runningBlocksTotal);
    printf("\n");
    cout << "{ z_len = " << lens.z << ", y_len = " << lens.y << ", x_len = " << lens.x << ", total_len = " << lens_flat << " }" << endl;
    printf("\n");
    printf("\n");

    return runningBlocksTotal;
}

template<
    const int amin_z, const int amax_z,
    const int amin_y, const int amax_y,
    const int amin_x, const int amax_x,
    const int group_size_x,  const int group_size_y, const int group_size_z>
__host__
void doTest_3D(const int physBlocks)
{
    const int z_range = (amin_z + amax_z + 1);
    const int y_range = (amin_y + amax_y + 1);
    const int x_range = (amin_x + amax_x + 1);

    const int ixs_len = z_range  * y_range * x_range;
    const int ixs_size = ixs_len*sizeof(int3);
    int3* cpu_ixs = (int3*)malloc(ixs_size);

    {
        int q = 0;
        for(int i=0; i < z_range; i++){
            for(int j=0; j < y_range; j++){
                for(int k=0; k < x_range; k++){
                    cpu_ixs[q++] = make_int3(k-amin_x, j-amin_y, i-amin_z);
                }
            }
        }
    }
    cout << "ixs[" << ixs_len << "] = (zr,yr,xr) = (" << -amin_z << "..." << amax_z << ", " << -amin_y << "..." << amax_y << ", " << -amin_x << "..." << amax_x << ")" << endl;

    constexpr long len = lens_flat;

    T* cpu_out = (T*)malloc(len*sizeof(T));
    run_cpu_3d<ixs_len>(cpu_ixs, cpu_out);


    constexpr int blockDim_flat = group_size_x * group_size_y * group_size_z;
    constexpr int3 virtual_grid = {
        CEIL_DIV(lens.x, group_size_x),
        CEIL_DIV(lens.y, group_size_y),
        CEIL_DIV(lens.z, group_size_z)};
    constexpr dim3 block_3d(group_size_x,group_size_y,group_size_z);
    constexpr dim3 grid_3d(virtual_grid.x, virtual_grid.y, virtual_grid.z);
    constexpr int virtual_grid_flat = virtual_grid.x * virtual_grid.y * virtual_grid.z;
    constexpr int lens_grid = CEIL_DIV(lens_flat, blockDim_flat);
    constexpr int3 lens_spans = { 0, 0, 0 }; // void
    constexpr int3 virtual_grid_spans = { 1, virtual_grid.x, virtual_grid.x*virtual_grid.y };


    cout << "Blockdim z,y,x = " << group_size_z << ", " << group_size_y << ", " << group_size_x << endl;
    printf("virtual number of blocks = %d\n", virtual_grid_flat);
    {
        {
            cout << "## Benchmark 3d global read - inlined ixs - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = global_reads_3d_inlined
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d, false); // warmup as it is first kernel
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            cout << "## Benchmark 3d global read - inlined ixs - singleDim grid - grid span ##";
            Kernel3dPhysSingleDim kfun = global_reads_3d_inlined_singleDim_gridSpan
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_singleDim(kfun, cpu_out, virtual_grid_flat, blockDim_flat, virtual_grid_spans);
        }
        {
            cout << "## Benchmark 3d global read - inlined ixs - singleDim grid - lens span ##";
            Kernel3dPhysSingleDim kfun = global_reads_3d_inlined_singleDim_lensSpan
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_singleDim(kfun, cpu_out, lens_grid, blockDim_flat, lens_spans);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - cube load - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - flat load (div/rem) - multiDim grid ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined_flat
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - flat load (div/rem) - singleDim grid ##";
            Kernel3dPhysSingleDim kfun = big_tile_3d_inlined_flat_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_singleDim(kfun, cpu_out, virtual_grid_flat, blockDim_flat, virtual_grid_spans);
        }
        {
            cout << "## Benchmark 3d virtual (add/carry) - global read - inlined idxs - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_global_read_3d_inlined_grid_span_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid);
        }
        {
            cout << "## Benchmark 3d virtual (add/carry) - big tile - inlined idxs - flat load (div/rem) - multiDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_divrem_MultiDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual_MultiDim(kfun, cpu_out, physBlocks, block_3d, virtual_grid);
        }
        {
            cout << "## Benchmark 3d virtual (add/carry) - big tile - inlined idxs - flat load (div/rem) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_divrem_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid);
        }
        {
            cout << "## Benchmark 3d virtual (rem/div) - big tile - inlined idxs - flat load (div/rem) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_divrem_big_tile_3d_inlined_flat_divrem_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid);
        }
        {
            cout << "## Benchmark 3d virtual (add/carry) - big tile - inlined idxs - flat load (add/carry) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z>;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid);
        }
        {
            constexpr int strip_x = 2;
            constexpr int strip_y = 1;
            constexpr int strip_z = 4;
            constexpr int strip_size_x = group_size_x*strip_x;
            constexpr int strip_size_y = group_size_y*strip_y;
            constexpr int strip_size_z = group_size_z*strip_z;

            constexpr int3 virtual_grid_strip = {
                CEIL_DIV(lens.x, strip_size_x),
                CEIL_DIV(lens.y, strip_size_y),
                CEIL_DIV(lens.z, strip_size_z)};
            cout << "## Benchmark 3d virtual (add/carry) - stripmined big tile, ";
            printf("strip_size=[%d][%d][%d]f32 ", strip_size_z, strip_size_y, strip_size_x);
            cout << "- inlined idxs - flat load (add/carry) - singleDim grid ##";
            Kernel3dVirtual kfun = virtual_addcarry_stripmine_big_tile_3d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y,amin_z
                ,amax_x,amax_y,amax_z
                ,group_size_x,group_size_y,group_size_z
                ,strip_x,strip_y,strip_z
                >;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid_strip);
        }
    }

    free(cpu_out);
    free(cpu_ixs);
}

__host__
int main()
{
    int physBlocks = getPhysicalBlockCount();

    constexpr int group_size_x = 32;
    constexpr int group_size_y = 8;
    constexpr int group_size_z = 4;

    doTest_3D<1,1,0,0,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<2,2,0,0,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<3,3,0,0,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<4,4,0,0,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<5,5,0,0,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);

    doTest_3D<1,1,1,1,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<2,2,2,2,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<3,3,3,3,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<4,4,4,4,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<5,5,5,5,0,0, group_size_x,group_size_y,group_size_z>(physBlocks);

    doTest_3D<1,1,0,0,1,1, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<2,2,0,0,2,2, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<3,3,0,0,3,3, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<4,4,0,0,4,4, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<5,5,0,0,5,5, group_size_x,group_size_y,group_size_z>(physBlocks);

    doTest_3D<1,1,1,1,1,1, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<2,2,2,2,2,2, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<3,3,3,3,3,3, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<4,4,4,4,4,4, group_size_x,group_size_y,group_size_z>(physBlocks);
    doTest_3D<5,5,5,5,5,5, group_size_x,group_size_y,group_size_z>(physBlocks);

    return 0;
}
