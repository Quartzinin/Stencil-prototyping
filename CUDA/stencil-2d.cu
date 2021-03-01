#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include "runners.h"
#include "kernels-2d.h"
using namespace std;
#include <iostream>
using std::cout;
using std::endl;


template<int D>
void stencil_2d_cpu(
    const T* start,
    const int2* idxs,
    T* out,
    const int y_len,
    const int x_len)
{
    const int max_y_ix = y_len - 1;
    const int max_x_ix = x_len - 1;
    for (int i = 0; i < y_len; ++i)
    {
        for (int k = 0; k < x_len; ++k)
        {
            T arr[D];
            for (int j = 0; j < D; ++j)
            {
                int y = BOUND(i + idxs[j].y, max_y_ix);
                int x = BOUND(k + idxs[j].x, max_x_ix);
                int index = y * x_len + x;
                arr[j] = start[index];
            }
            T lambda_res = stencil_fun_cpu<D>(arr);
            out[i * x_len + k] = lambda_res;
        }
    }
}

#define call_kernel_2d(kernel) {\
    const dim3 block(SQ_BLOCKSIZE,SQ_BLOCKSIZE,1);\
    const int BNx = CEIL_DIV(x_len, SQ_BLOCKSIZE);\
    const int BNy = CEIL_DIV(y_len, SQ_BLOCKSIZE);\
    const dim3 grid(BNx, BNy, 1);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}
#define call_small_tile_2d(kernel) {\
    const dim3 block(SQ_BLOCKSIZE,SQ_BLOCKSIZE,1);\
    const int wasted_x = ix_min + ix_max;\
    const int wasted_y = ix_min + ix_max;\
    const int working_block_x = SQ_BLOCKSIZE-wasted_x;\
    const int working_block_y = SQ_BLOCKSIZE-wasted_y;\
    const int BNx = CEIL_DIV(x_len, working_block_x);\
    const int BNy = CEIL_DIV(y_len   , working_block_y);\
    const dim3 grid(BNx, BNy, 1);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

template<int D>
T* run_cpu_2d(const int2* idxs, const int y_len, const int x_len)
{
    int len = y_len*x_len;
    T* cpu_in = (T*)malloc(len*sizeof(T));
    T* cpu_out = (T*)malloc(len*sizeof(T));

    for (int i = 0; i < len; ++i)
    {
        cpu_in[i] = (T)(i+1);
    }

    stencil_2d_cpu<D>(cpu_in,idxs,cpu_out,y_len,x_len);
    free(cpu_in);
    return cpu_out;
}

template<int sq_ixs_len, int ix_min, int ix_max>
void doTest_2D()
{
    const int RUNS = 100;

    const int ixs_len = sq_ixs_len * sq_ixs_len;
    //const int W = D / 2;
    const int ixs_size = ixs_len*sizeof(int2);
    int2* cpu_ixs = (int2*)malloc(ixs_size);
    {
        int q = 0;
        for(int i=0; i < sq_ixs_len; i++){
            for(int j=0; j < sq_ixs_len; j++){
                cpu_ixs[q++] = make_int2(j-ix_min, i-ix_min);
            }
        }
    }
    CUDASSERT(cudaMemcpyToSymbol(ixs_2d, cpu_ixs, ixs_size));

    cout << "const int ixs[" << ixs_len << "] = [";
    for(int i=0; i < ixs_len ; i++){
        cout << " (" << cpu_ixs[i].x << "," << cpu_ixs[i].y << ")";
        if(i == ixs_len-1)
        { cout << "]" << endl; }
        else{ cout << ", "; }
    }

    const int y_len = 2 << 12;
    const int x_len = 2 << 8;
    const int len = y_len * x_len;
    cout << "{ row_len = " << x_len << ", col_len = " << y_len
         << ", total_len = " << len << " }" << endl;
    T* cpu_out = run_cpu_2d<ixs_len>(cpu_ixs,y_len,x_len);

    measure_memset_bandwidth(len * sizeof(T));

    {
        GPU_RUN_INIT;
        GPU_RUN(call_kernel_2d(
                    (global_reads_2d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                ,"## Benchmark 2d global read ##",(void)0,(void)0);
        GPU_RUN(call_small_tile_2d(
                    (small_tile_2d<ixs_len,ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                ,"## Benchmark 2d small tile ##",(void)0,(void)0);
        GPU_RUN(call_kernel_2d(
                    (big_tile_2d<ixs_len,ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                ,"## Benchmark 2d big tile ##",(void)0,(void)0);
        if(ixs_len == (ix_min + ix_max + 1) * (ix_min + ix_max + 1)){
            GPU_RUN(call_kernel_2d(
                        (global_reads_2d_const<ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                    ,"## Benchmark 2d global read constant ixs ##",(void)0,(void)0);
            GPU_RUN(call_small_tile_2d(
                        (small_tile_2d_const<ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                    ,"## Benchmark 2d small tile constant ixs ##",(void)0,(void)0);
            GPU_RUN(call_kernel_2d(
                        (big_tile_2d_const<ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                    ,"## Benchmark 2d big tile constant ixs ##",(void)0,(void)0);
        }
        GPU_RUN_END;
    }
}


int main()
{
    doTest_2D<3,1,1>();
    doTest_2D<5,2,2>();
    doTest_2D<7,3,3>();

    return 0;
}


