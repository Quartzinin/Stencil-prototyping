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


using KernelFunType3d = void(*)(const T*, T*, const long3);
using KernelFunType3dF = void(*)(const T*, T*, const long3, const long3);
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


template<int z_min, int z_max, int y_min, int y_max, int x_min, int x_max>
__host__
void doTest_3D()
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
    //if(ixs_len <= BLOCKSIZE){
    //    CUDASSERT(cudaMemcpyToSymbol(ixs_3d, cpu_ixs, ixs_size));
    //}
    {
        cout << "Blockdim z,y,x = " << Z_BLOCK << ", " << Y_BLOCK << ", " << X_BLOCK << endl;
        cout << "ixs[" << ixs_len << "] = (zr,yr,xr) = (" << -z_min << "..." << z_max << ", " << -y_min << "..." << y_max << ", " << -x_min << "..." << x_max << ")" << endl;
    }

    constexpr long len = lens_flat;
    cout << "{ z_len = " << lens.z << ", y_len = " << lens.y << ", x_len = " << lens.x << ", total_len = " << len << " }" << endl;

    T* cpu_out = (T*)malloc(len*sizeof(T));
    run_cpu_3d<ixs_len>(cpu_ixs, cpu_out);

    constexpr long blockDim_flat = X_BLOCK * Y_BLOCK * Z_BLOCK;
    constexpr dim3 block_3d(X_BLOCK,Y_BLOCK,Z_BLOCK);
    constexpr dim3 grid_3d(
        CEIL_DIV(lens.x, X_BLOCK),
        CEIL_DIV(lens.y, Y_BLOCK),
        CEIL_DIV(lens.z, Z_BLOCK));
    constexpr dim3 block_3d_singledim(X_BLOCK*Y_BLOCK*Z_BLOCK,1,1);
    constexpr dim3 grid_3d_singledim(
        CEIL_DIV(lens.x, X_BLOCK)*
        CEIL_DIV(lens.y, Y_BLOCK)*
        CEIL_DIV(lens.z, Z_BLOCK),1,1);
    constexpr dim3 grid_3d_singledim_flat(CEIL_DIV(lens_flat, blockDim_flat),1,1);
    constexpr long3 lens_spans = { 1, lens.x, lens.x*lens.y };
    {
        {
            cout << "## Benchmark 3d global read inlined ixs ##";
            KernelFunType3d kfun = global_reads_3d_inlined<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d, false); // warmup as it is first kernel
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            cout << "## Benchmark 3d global read inlined ixs - singleDim block ##";
            KernelFunType3dF kfun = global_reads_3d_inlined_singleDim<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_singleDim(kfun, cpu_out, grid_3d_singledim_flat, block_3d_singledim, lens_spans);
        }
/*
        if(!(z_range > Z_BLOCK || y_range > Y_BLOCK || x_range > X_BLOCK))
        {
            cout << "## Benchmark 3d small tile inlined ixs ##";
            constexpr int working_block_z = Z_BLOCK - (z_min + z_max);\
            constexpr int working_block_y = Y_BLOCK - (y_min + y_max);\
            constexpr int working_block_x = X_BLOCK - (x_min + x_max);\
            constexpr int BNx = CEIL_DIV(lens.x, working_block_x);\
            constexpr int BNy = CEIL_DIV(lens.y, working_block_y);\
            constexpr int BNz = CEIL_DIV(lens.z, working_block_z);\
            constexpr dim3 small_grid_3d(BNx, BNy, BNz);
            KernelFunType3d kfun = small_tile_3d_inlined<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_cube(kfun, cpu_out, small_grid_3d, block_3d);
        }
*/
        {
            cout << "## Benchmark 3d big tile - inlined idxs ##";
            KernelFunType3d kfun = big_tile_3d_inlined<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            cout << "## Benchmark 3d big tile - inlined idxs - flat load ##";
            KernelFunType3d kfun = big_tile_3d_inlined_flat<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_cube(kfun, cpu_out, grid_3d, block_3d);
        }
        {
            const long3 grid_spans = { 1, grid_3d.x, grid_3d.y*grid_3d.x};
            cout << "## Benchmark 3d big tile - inlined idxs - flat load - singleDim block ##";
            KernelFunType3dF kfun = big_tile_3d_inlined_flat_singleDim<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run_singleDim(kfun, cpu_out, grid_3d_singledim, block_3d_singledim, grid_spans);
        }
/*
        {
            cout << "## Benchmark 3d big tile inlined ixs layered ##";
            KernelFunType3d kfun = big_tile_3d_inlined_layered<z_min,z_max,y_min,y_max,x_min,x_max>;
            G.do_run(kfun, cpu_out, grid_3d, block_3d);
        }
*/
    }

    free(cpu_out);
    free(cpu_ixs);
}

__host__
int main()
{
    doTest_3D<1,1,0,0,0,0>();
    doTest_3D<2,2,0,0,0,0>();
    doTest_3D<3,3,0,0,0,0>();
    doTest_3D<4,4,0,0,0,0>();
    doTest_3D<5,5,0,0,0,0>();

    doTest_3D<1,1,1,1,0,0>();
    doTest_3D<2,2,2,2,0,0>();
    doTest_3D<3,3,3,3,0,0>();
    doTest_3D<4,4,4,4,0,0>();
    doTest_3D<5,5,5,5,0,0>();

    doTest_3D<1,1,0,0,1,1>();
    doTest_3D<2,2,0,0,2,2>();
    doTest_3D<3,3,0,0,3,3>();
    doTest_3D<4,4,0,0,4,4>();
    doTest_3D<5,5,0,0,5,5>();

    doTest_3D<1,1,1,1,1,1>();
    doTest_3D<2,2,2,2,2,2>();
    doTest_3D<3,3,3,3,3,3>();
    doTest_3D<4,4,4,4,4,4>();
    doTest_3D<5,5,5,5,5,5>();

    return 0;
}
