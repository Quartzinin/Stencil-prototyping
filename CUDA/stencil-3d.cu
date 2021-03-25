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
    constexpr int3 virtual_grid = {
        CEIL_DIV(lens.x, X_BLOCK),
        CEIL_DIV(lens.y, Y_BLOCK),
        CEIL_DIV(lens.z, Z_BLOCK)};
    constexpr int virtual_grid_flat = virtual_grid.x * virtual_grid.y * virtual_grid.z;
    printf("\n");
    cout << "Blockdim z,y,x = " << Z_BLOCK << ", " << Y_BLOCK << ", " << X_BLOCK << endl;
    cout << "{ z_len = " << lens.z << ", y_len = " << lens.y << ", x_len = " << lens.x << ", total_len = " << lens_flat << " }" << endl;
    printf("virtual number of blocks = %d\n", virtual_grid_flat);
    printf("\n");
    printf("\n");

    return runningBlocksTotal;
}

template<
    const int z_min, const int z_max,
    const int y_min, const int y_max,
    const int x_min, const int x_max>
__host__
void doTest_3D(const int physBlocks)
{
    const int z_range = (z_min + z_max + 1);
    const int y_range = (y_min + y_max + 1);
    const int x_range = (x_min + x_max + 1);

    const int ixs_len = z_range  * y_range * x_range;
    const int ixs_size = ixs_len*sizeof(int3);
    int3* cpu_ixs = (int3*)malloc(ixs_size);

    {
        int q = 0;
        for(int i=0; i < z_range; i++){
            for(int j=0; j < y_range; j++){
                for(int k=0; k < x_range; k++){
                    cpu_ixs[q++] = make_int3(k-x_min, j-y_min, i-z_min);
                }
            }
        }
    }
    {
        cout << "ixs[" << ixs_len << "] = (zr,yr,xr) = (" << -z_min << "..." << z_max << ", " << -y_min << "..." << y_max << ", " << -x_min << "..." << x_max << ")" << endl;
    }


    constexpr long len = lens_flat;

    T* cpu_out = (T*)malloc(len*sizeof(T));
    run_cpu_3d<ixs_len>(cpu_ixs, cpu_out);

    constexpr int blockDim_flat = X_BLOCK * Y_BLOCK * Z_BLOCK;
    constexpr int3 virtual_grid = {
        CEIL_DIV(lens.x, X_BLOCK),
        CEIL_DIV(lens.y, Y_BLOCK),
        CEIL_DIV(lens.z, Z_BLOCK)};
    constexpr dim3 block_3d(X_BLOCK,Y_BLOCK,Z_BLOCK);
    constexpr dim3 grid_3d(virtual_grid.x, virtual_grid.y, virtual_grid.z);
    constexpr int virtual_grid_flat = virtual_grid.x * virtual_grid.y * virtual_grid.z;
    constexpr int lens_grid = CEIL_DIV(lens_flat, blockDim_flat);
    constexpr int3 lens_spans = { 0, 0, 0 }; // void
    constexpr int3 virtual_grid_spans = { 1, virtual_grid.x, virtual_grid.x*virtual_grid.y };

    {
        {
            cout << "## Benchmark 3d global read - inlined ixs - multiDim block ##";
            Kernel3dPhysMultiDim kfun = global_reads_3d_inlined<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d, false); // warmup as it is first kernel
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            cout << "## Benchmark 3d global read - inlined ixs - singleDim block - grid span ##";
            Kernel3dPhysSingleDim kfun = global_reads_3d_inlined_singleDim_gridSpan<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_singleDim(kfun, cpu_out, virtual_grid_flat, blockDim_flat, virtual_grid_spans);
        }
        {
            cout << "## Benchmark 3d global read - inlined ixs - singleDim block - lens span ##";
            Kernel3dPhysSingleDim kfun = global_reads_3d_inlined_singleDim_lensSpan<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_singleDim(kfun, cpu_out, lens_grid, blockDim_flat, lens_spans);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - flat load (div/rem) ##";
            Kernel3dPhysMultiDim kfun = big_tile_3d_inlined_flat<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - flat load (div/rem) - singleDim block ##";
            Kernel3dPhysSingleDim kfun = big_tile_3d_inlined_flat_singleDim<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_singleDim(kfun, cpu_out, virtual_grid_flat, blockDim_flat, virtual_grid_spans);
        }
        {
            cout << "## Benchmark 3d virtual (add/carry) - global read - inlined idxs - singleDim block ##";
            Kernel3dVirtual kfun = virtual_addcarry_global_read_3d_inlined_grid_span_singleDim
                <x_min,y_min,z_min
                ,x_max,y_max,z_max
                ,X_BLOCK,Y_BLOCK,Z_BLOCK>;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid);
        }
        {
            cout << "## Benchmark 3d virtual (add/carry) - big tile - inlined idxs - flat load (div/rem) - multiDim block ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_divrem_MultiDim
                <x_min,y_min,z_min
                ,x_max,y_max,z_max
                ,X_BLOCK,Y_BLOCK,Z_BLOCK>;
            G.do_run_virtual_MultiDim(kfun, cpu_out, physBlocks, block_3d, virtual_grid);
        }
        {
            cout << "## Benchmark 3d virtual (add/carry) - big tile - inlined idxs - flat load (div/rem) - singleDim block ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_divrem_singleDim
                <x_min,y_min,z_min
                ,x_max,y_max,z_max
                ,X_BLOCK,Y_BLOCK,Z_BLOCK>;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid);
        }
        {
            cout << "## Benchmark 3d virtual (rem/div) - big tile - inlined idxs - flat load (div/rem) - singleDim block ##";
            Kernel3dVirtual kfun = virtual_divrem_big_tile_3d_inlined_flat_divrem_singleDim
                <x_min,y_min,z_min
                ,x_max,y_max,z_max
                ,X_BLOCK,Y_BLOCK,Z_BLOCK>;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid);
        }
        {
            cout << "## Benchmark 3d virtual (add/carry) - big tile - inlined idxs - flat load (add/carry) - singleDim block ##";
            Kernel3dVirtual kfun = virtual_addcarry_big_tile_3d_inlined_flat_addcarry_singleDim
                <x_min,y_min,z_min
                ,x_max,y_max,z_max
                ,X_BLOCK,Y_BLOCK,Z_BLOCK>;
            G.do_run_virtual_singleDim(kfun, cpu_out, physBlocks, blockDim_flat, virtual_grid);
        }
    }

    free(cpu_out);
    free(cpu_ixs);
}

__host__
int main()
{
    int physBlocks = getPhysicalBlockCount();

    doTest_3D<1,1,0,0,0,0>(physBlocks);
    doTest_3D<2,2,0,0,0,0>(physBlocks);
    doTest_3D<3,3,0,0,0,0>(physBlocks);
    doTest_3D<4,4,0,0,0,0>(physBlocks);
    doTest_3D<5,5,0,0,0,0>(physBlocks);

    doTest_3D<1,1,1,1,0,0>(physBlocks);
    doTest_3D<2,2,2,2,0,0>(physBlocks);
    doTest_3D<3,3,3,3,0,0>(physBlocks);
    doTest_3D<4,4,4,4,0,0>(physBlocks);
    doTest_3D<5,5,5,5,0,0>(physBlocks);

    doTest_3D<1,1,0,0,1,1>(physBlocks);
    doTest_3D<2,2,0,0,2,2>(physBlocks);
    doTest_3D<3,3,0,0,3,3>(physBlocks);
    doTest_3D<4,4,0,0,4,4>(physBlocks);
    doTest_3D<5,5,0,0,5,5>(physBlocks);

    doTest_3D<1,1,1,1,1,1>(physBlocks);
    doTest_3D<2,2,2,2,2,2>(physBlocks);
    doTest_3D<3,3,3,3,3,3>(physBlocks);
    doTest_3D<4,4,4,4,4,4>(physBlocks);
    doTest_3D<5,5,5,5,5,5>(physBlocks);

    return 0;
}
