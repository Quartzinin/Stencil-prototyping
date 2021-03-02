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

template<int D>
void stencil_3d_cpu(
    const T* start,
    const int3* idxs,
    T* out,
    const int z_len,
    const int y_len,
    const int x_len)
{
    const int max_x_idx = x_len - 1;
    const int max_y_idx = y_len - 1;
    const int max_z_idx = z_len - 1;
    for (int i = 0; i < z_len; ++i)
    {
        for (int j = 0; j < y_len; ++j)
        {
            for (int k = 0; k < x_len; ++k)
            {
                T arr[D];
                for (int p = 0; p < D; ++p)
                {
                    int z = BOUND(i + idxs[p].z, max_z_idx);
                    int y = BOUND(j + idxs[p].y, max_y_idx);
                    int x = BOUND(k + idxs[p].x, max_x_idx);
                    int index = z * y_len * x_len + y * x_len + x;
                    arr[p] = start[index];
                }

                T lambda_res = stencil_fun_cpu<D>(arr);
                out[i*y_len*x_len + j*x_len + k] = lambda_res;
            }
        }
    }
}

#define call_kernel_3d(kernel) {\
    const int x_block = SQ_BLOCKSIZE; \
    const int y_block = x_block/4; \
    const int z_block = x_block/y_block; \
    const dim3 block(x_block,y_block,z_block);\
    const int BNx = CEIL_DIV(x_len, x_block);\
    const int BNy = CEIL_DIV(y_len, y_block);\
    const int BNz = CEIL_DIV(z_len, z_block);\
    const dim3 grid(BNx, BNy, BNz);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

#define call_small_tile_3d(kernel) {\
    const int x_block = X_BLOCK; \
    const int y_block = Y_BLOCK; \
    const int z_block = Z_BLOCK; \
    const dim3 block(x_block,y_block,z_block);\
    const int working_block_z = z_block - (z_min + z_max);\
    const int working_block_y = y_block - (y_min + y_max);\
    const int working_block_x = x_block - (x_min + x_max);\
    const int BNx = CEIL_DIV(x_len, working_block_x);\
    const int BNy = CEIL_DIV(y_len, working_block_y);\
    const int BNz = CEIL_DIV(z_len, working_block_z);\
    const dim3 grid(BNx, BNy, BNz);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

template<int D>
T* run_cpu_3d(const int3* idxs, const int z_len, const int y_len, const int x_len)
{
    int len = x_len*y_len*z_len;
    T* cpu_in = (T*)malloc(len*sizeof(T));
    T* cpu_out = (T*)malloc(len*sizeof(T));

    for (int i = 0; i < len; ++i)
    {
        cpu_in[i] = (T)(i+1);
    }

    stencil_3d_cpu<D>(cpu_in,idxs,cpu_out,z_len,y_len,x_len);
    free(cpu_in);
    return cpu_out;
}


template<int z_min, int z_max, int y_min, int y_max, int x_min, int x_max>
void doTest_3D()
{
    const int RUNS = 100;
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
    if(ixs_len <= BLOCKSIZE){
        CUDASSERT(cudaMemcpyToSymbol(ixs_3d, cpu_ixs, ixs_size));
    }
    {
        const int x_block = SQ_BLOCKSIZE;
        const int y_block = x_block/4;
        const int z_block = x_block/y_block;
        cout << "Blockdim z,y,x = " << z_block << ", " << y_block << ", " << x_block << endl;
        cout << "ixs[" << ixs_len << "] = (zr,yr,xr) = (-" << z_min << "..." << z_max << ", -" << y_min << "..." << y_max << ", -" << x_min << "..." << x_max << ")" << endl;
    }
    /*
    cout << "const int ixs[" << ixs_len << "] = [";
    for(int i=0; i < ixs_len ; i++){
        cout << " (" << cpu_ixs[i].z << "," << cpu_ixs[i].y << "," << cpu_ixs[i].x << ")";
        if(i == ixs_len-1)
        { cout << "]" << endl; }
        else{ cout << ", "; }
    }
    */

    const int z_len = 2 << 8; //outermost
    const int y_len = 2 << 7; //middle
    const int x_len = 2 << 6; //innermost

    const int len = z_len * y_len * x_len;
    cout << "{ z = " << z_len << ", y = " << y_len << ", x = " << x_len << ", total_len = " << len << " }" << endl;

    T* cpu_out = run_cpu_3d<ixs_len>(cpu_ixs,z_len,y_len,x_len);

    {
        GPU_RUN_INIT;

        /*
        if(ixs_len <= BLOCKSIZE){
            GPU_RUN(call_kernel_3d(
                        (global_reads_3d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                    ,"## Benchmark 3d global read ##",(void)0,(void)0);
            if (!(z_range > Z_BLOCK || y_range > Y_BLOCK || x_range > X_BLOCK))
            {
                GPU_RUN(call_small_tile_3d(
                            (small_tile_3d<ixs_len,z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                        ,"## Benchmark 3d small tile ##",(void)0,(void)0);
            }
            GPU_RUN(call_kernel_3d(
                        (big_tile_3d<ixs_len,z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                    ,"## Benchmark 3d big tile ##",(void)0,(void)0);
        }
        */
        GPU_RUN(call_kernel_3d(
                    (global_reads_3d_const<z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d global read const ##",(void)0,(void)0);
        if (!(z_range > Z_BLOCK || y_range > Y_BLOCK || x_range > X_BLOCK))
        {
            GPU_RUN(call_small_tile_3d(
                        (small_tile_3d_const<z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                    ,"## Benchmark 3d small tile const ##",(void)0,(void)0);
        }
        GPU_RUN(call_kernel_3d(
                    (big_tile_3d_const<z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d big tile const ##",(void)0,(void)0);

        GPU_RUN_END;
    }
}

int main()
{
    doTest_3D<1,1,0,0,0,0>();
    doTest_3D<1,1,1,1,0,0>();
    doTest_3D<1,1,0,0,1,1>();
    doTest_3D<1,1,1,1,1,1>();

    doTest_3D<2,2,0,0,0,0>();
    doTest_3D<2,2,2,2,0,0>();
    doTest_3D<2,2,0,0,2,2>();
    doTest_3D<2,2,2,2,2,2>();

    doTest_3D<3,3,0,0,0,0>();
    doTest_3D<3,3,3,3,0,0>();
    doTest_3D<3,3,0,0,3,3>();
    doTest_3D<3,3,3,3,3,3>();

    doTest_3D<4,4,0,0,0,0>();
    doTest_3D<4,4,4,4,0,0>();
    doTest_3D<4,4,0,0,4,4>();
    doTest_3D<4,4,4,4,4,4>();

    doTest_3D<5,5,0,0,0,0>();
    doTest_3D<5,5,5,5,0,0>();
    doTest_3D<5,5,0,0,5,5>();
    doTest_3D<5,5,5,5,5,5>();

    return 0;
}
