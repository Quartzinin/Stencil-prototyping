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
    const dim3 block(X_BLOCK,Y_BLOCK,Z_BLOCK);\
    const int BNx = CEIL_DIV(x_len, X_BLOCK);\
    const int BNy = CEIL_DIV(y_len, Y_BLOCK);\
    const int BNz = CEIL_DIV(z_len, Z_BLOCK);\
    const dim3 grid(BNx, BNy, BNz);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

#define call_kernel_3d_singleDim(kernel) {\
    const int block = BLOCKSIZE;\
    const int BNx = CEIL_DIV(x_len, X_BLOCK);\
    const int BNy = CEIL_DIV(y_len, Y_BLOCK);\
    const int BNz = CEIL_DIV(z_len, Z_BLOCK);\
    const int grid = BNx * BNy * BNz;\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

#define call_small_tile_3d(kernel) {\
    const dim3 block(X_BLOCK,Y_BLOCK,Z_BLOCK);\
    const int working_block_z = Z_BLOCK - (z_min + z_max);\
    const int working_block_y = Y_BLOCK - (y_min + y_max);\
    const int working_block_x = X_BLOCK - (x_min + x_max);\
    const int BNx = CEIL_DIV(x_len, working_block_x);\
    const int BNy = CEIL_DIV(y_len, working_block_y);\
    const int BNz = CEIL_DIV(z_len, working_block_z);\
    const dim3 grid(BNx, BNy, BNz);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

template<int D>
void run_cpu_3d(const int3* idxs, const int z_len, const int y_len, const int x_len, T* cpu_out)
{
    int len = x_len*y_len*z_len;
    T* cpu_in = (T*)malloc(len*sizeof(T));

    for (int i = 0; i < len; ++i)
    {
        cpu_in[i] = (T)(i+1);
    }

    struct timeval t_startpar, t_endpar, t_diffpar;
    gettimeofday(&t_startpar, NULL);
    {
        stencil_3d_cpu<D>(cpu_in,idxs,cpu_out,z_len,y_len,x_len);
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
    //if(ixs_len <= BLOCKSIZE){
    //    CUDASSERT(cudaMemcpyToSymbol(ixs_3d, cpu_ixs, ixs_size));
    //}
    {
        cout << "Blockdim z,y,x = " << Z_BLOCK << ", " << Y_BLOCK << ", " << X_BLOCK << endl;
        cout << "ixs[" << ixs_len << "] = (zr,yr,xr) = (" << -z_min << "..." << z_max << ", " << -y_min << "..." << y_max << ", " << -x_min << "..." << x_max << ")" << endl;
    }

    const long z_len = (1 << 9) - 1; //outermost
    const long y_len = (1 << 8) - 1; //middle
    const long x_len = (1 << 7) - 1; //innermost

    const int BNx_in = CEIL_DIV(x_len, X_BLOCK);\
    const int BNy_in = CEIL_DIV(y_len, Y_BLOCK);\

    const int len = z_len * y_len * x_len;
    cout << "{ z_len = " << z_len << ", y_len = " << y_len << ", x_len = " << x_len << ", total_len = " << len << " }" << endl;

    T* cpu_out = (T*)malloc(len*sizeof(T));
    run_cpu_3d<ixs_len>(cpu_ixs,z_len,y_len,x_len, cpu_out);

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
                    (global_reads_3d_inlined<z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d global read inlined ixs ##",(void)0,(void)0);
/*
        if (!(z_range > Z_BLOCK || y_range > Y_BLOCK || x_range > X_BLOCK))
        {
            GPU_RUN(call_small_tile_3d(
                        (small_tile_3d_inlined<z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                    ,"## Benchmark 3d small tile inlined ixs ##",(void)0,(void)0);
        }
*/
        GPU_RUN(call_kernel_3d(
                    (big_tile_3d_inlined<z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d big tile - inlined idxs ##",(void)0,(void)0);
        GPU_RUN(call_kernel_3d(
                    (big_tile_3d_inlined_flat<z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d big tile - inlined idxs - flat load ##",(void)0,(void)0);
        GPU_RUN(call_kernel_3d_singleDim(
                    (big_tile_3d_inlined_flat_singleDim<z_min,z_max,y_min,y_max,x_min,x_max,BNx_in,BNy_in><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d big tile - inlined idxs - flat load - singleDim block ##",(void)0,(void)0)

        //GPU_RUN(call_kernel_3d(
        //            (big_tile_3d_inlined_layered<z_min,z_max,y_min,y_max,x_min,x_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
        //        ,"## Benchmark 3d big tile inlined ixs layered ##",(void)0,(void)0);

        GPU_RUN_END;
    }

    free(cpu_out);
    free(cpu_ixs);
}

int main()
{
/*
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
*/
    doTest_3D<1,1,1,1,1,1>();
    doTest_3D<2,2,2,2,2,2>();
    doTest_3D<3,3,3,3,3,3>();
    doTest_3D<4,4,4,4,4,4>();
    doTest_3D<5,5,5,5,5,5>();

    return 0;
}
