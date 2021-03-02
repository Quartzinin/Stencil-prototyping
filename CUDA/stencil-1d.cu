#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

#include "runners.h"
#include "kernels-1d.h"

using namespace std;
#include <iostream>
using std::cout;
using std::endl;


template<int D>
void stencil_1d_cpu(
    const T* start,
    const int* idxs,
    T* out,
    const int len)
{
    for (int i = 0; i < len; ++i)
    {
        T arr[D];
        for (int j = 0; j < D; ++j)
        {
            int idx = idxs[j];
            int bound = min(len-1,max(0,i+idx));
            arr[j] = start[bound];
        }
        T lambda_res = stencil_fun_cpu<D>(arr);
        out[i] = lambda_res;
    }
}

#define call_inSharedKernel_1d(kernel) {\
    const int block = BLOCKSIZE;\
    const int wasted = ix_min + ix_max;\
    const int working_block = BLOCKSIZE-wasted;\
    const int grid = (wasted + len + (working_block-1)) / working_block;\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

#define call_kernel_1d(kernel) {\
    const int block = BLOCKSIZE;\
    const int grid = (len + (block-1)) / block;\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

template<int D>
T* run_cpu_1d(const int* idxs, const int len)
{
    T* cpu_in  = (T*)malloc(len*sizeof(T));
    T* cpu_out = (T*)malloc(len*sizeof(T));

    for (int i = 0; i < len; ++i)
    {
        cpu_in[i] = (T)(i+1);
    }

    stencil_1d_cpu<D>(cpu_in,idxs,cpu_out,len);
    free(cpu_in);
    return cpu_out;
}

template<int ixs_len, int ix_min, int ix_max>
void doTest_1D()
{
    const int RUNS = 100;

    const int D = ixs_len;
    const int ixs_size = D*sizeof(int);
    int* cpu_ixs = (int*)malloc(ixs_size);
    const int step = (ix_min + ix_max) / (ixs_len-1);
    {
        int s = -ix_min;
        for(int i=0; i < D ; i++){ cpu_ixs[i] = s; s += step; }
    }
    for(int i=0; i < D ; i++){
        const int V = cpu_ixs[i];
        if(-ix_min <= V && V <= ix_max)
        {}
        else { printf("index array contains indexes not in range\n"); exit(1);}
    }
    CUDASSERT(cudaMemcpyToSymbol(ixs_1d, cpu_ixs, ixs_size));

    const int lenp = 23;
    const int len = 2 << lenp;
    cout << "{ x_len = " << len << " }" << endl;
    T* cpu_out = run_cpu_1d<D>(cpu_ixs,len);

    //cout << "input[2^" << lenp << "]" << endl;
    //cout << "ixs[" << D << "]" << endl;
    cout << "ixs[" << D << "] = [";
    for(int i=0; i < 3 ; i++){
        cout << " " << cpu_ixs[i];
        if(i != D-1)
        { cout << ", "; }
    }
    cout << endl;

    {
        GPU_RUN_INIT;


        /*
        GPU_RUN(call_kernel_1d(
                    (inlinedIndexes_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d global reads ##",(void)0,(void)0);
        GPU_RUN(call_inSharedKernel_1d(
                    (inSharedtiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d small tile ##",(void)0,(void)0);
        GPU_RUN(call_kernel_1d(
                    (big_tiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d big tile ##",(void)0,(void)0);
        */

        GPU_RUN(call_kernel_1d(
                    (global_read_1d_const<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d global reads constant ixs ##",(void)0,(void)0);
        GPU_RUN(call_inSharedKernel_1d(
                    (small_tile_1d_const<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d small tile constant ixs  ##",(void)0,(void)0);
        GPU_RUN(call_kernel_1d(
                    (big_tile_1d_const<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d big tile constant ixs ##",(void)0,(void)0);

        GPU_RUN_END;
    }

    free(cpu_out);
    free(cpu_ixs);
}


int main()
{
    doTest_1D<3,1,1>();
    doTest_1D<5,2,2>();
    doTest_1D<7,3,3>();
    doTest_1D<9,4,4>();
    doTest_1D<11,5,5>();
    doTest_1D<13,6,6>();
    doTest_1D<15,7,7>();
    doTest_1D<17,8,8>();
    doTest_1D<19,9,9>();
    doTest_1D<21,10,10>();
    doTest_1D<23,11,11>();
    doTest_1D<25,12,12>();
    doTest_1D<27,13,13>();
    doTest_1D<29,14,14>();
    doTest_1D<31,15,15>();
    doTest_1D<33,16,16>();
    doTest_1D<35,17,17>();
    doTest_1D<37,18,18>();
    doTest_1D<39,19,19>();
    doTest_1D<41,20,20>();
    doTest_1D<43,21,21>();
    doTest_1D<45,22,22>();
    doTest_1D<47,23,23>();
    doTest_1D<49,24,24>();
    doTest_1D<51,25,25>();

    doTest_1D<101,50,50>();
    doTest_1D<201,100,100>();
    doTest_1D<301,150,150>();
    doTest_1D<401,200,200>();
    doTest_1D<501,250,250>();
    doTest_1D<601,300,300>();
    doTest_1D<701,350,350>();
    doTest_1D<801,400,400>();
    doTest_1D<901,450,450>();
    doTest_1D<1001,500,500>();

    doTest_1D<3,2,2>();
    doTest_1D<3,3,3>();
    doTest_1D<3,4,4>();
    doTest_1D<3,5,5>();
    doTest_1D<3,6,6>();
    doTest_1D<3,7,7>();
    doTest_1D<3,8,8>();
    doTest_1D<3,9,9>();
    doTest_1D<3,10,10>();
    doTest_1D<3,11,11>();
    doTest_1D<3,12,12>();
    doTest_1D<3,13,13>();
    doTest_1D<3,14,14>();
    doTest_1D<3,15,15>();
    doTest_1D<3,16,16>();
    doTest_1D<3,17,17>();
    doTest_1D<3,18,18>();
    doTest_1D<3,19,19>();
    doTest_1D<3,20,20>();
    doTest_1D<3,21,21>();
    doTest_1D<3,22,22>();
    doTest_1D<3,23,23>();
    doTest_1D<3,24,24>();
    doTest_1D<3,25,25>();

    doTest_1D<3,51,51>();
    doTest_1D<3,100,100>();
    doTest_1D<3,150,150>();
    doTest_1D<3,200,200>();
    doTest_1D<3,250,250>();
    doTest_1D<3,300,300>();
    doTest_1D<3,350,350>();
    doTest_1D<3,400,400>();
    doTest_1D<3,450,450>();
    doTest_1D<3,500,500>();

    return 0;
}

