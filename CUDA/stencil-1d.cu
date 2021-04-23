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

static constexpr long n_runs = 100;
static constexpr long lens = (1 << 24) - 1;

static Globs
    <long,long
    ,Kernel1dVirtual
    ,Kernel1dPhysMultiDim
    ,Kernel1dPhysStripDim
    > G(lens, lens, n_runs);

template<int D, int ix_min, int ix_max>
void stencil_1d_cpu(
    const T* A,
    const int* idxs,
    T* out,
    const int len)
{
    const int max_ix = len-1;
    constexpr int step = (1 + ix_max - ix_min) / (D-1);
    T tmp[D];
    for (int gid = 0; gid < len; ++gid)
    {
        #pragma unroll
        for (int j = 0; j < D; ++j)
        {
            tmp[j] = A[BOUND((gid + (j*step + ix_min)), max_ix)];
        }

        T res = stencil_fun_cpu<D>((const T*)(tmp));
        out[gid] = res;
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

template<int D, int ix_min, int ix_max>
void run_cpu_1d(const int* idxs, const int len, T* cpu_out)
{
    T* cpu_in  = (T*)malloc(len*sizeof(T));
    srand(1);
    for (int i = 0; i < len; ++i)
    {
        cpu_in[i] = (T)rand();
    }

    struct timeval t_startpar, t_endpar, t_diffpar;
    gettimeofday(&t_startpar, NULL);
    {
        stencil_1d_cpu<D, ix_min, ix_max>(cpu_in,idxs,cpu_out,len);
    }
    gettimeofday(&t_endpar, NULL);
    timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
    const unsigned long elapsed = (t_diffpar.tv_sec*1e6+t_diffpar.tv_usec) / 1000;
    const unsigned long seconds = elapsed / 1000;
    const unsigned long microseconds = elapsed % 1000;
    printf("cpu c 1d for 1 run : %lu.%03lu seconds\n", seconds, microseconds);

    free(cpu_in);
}

template<int ixs_len, int gps_x, int ix_min, int ix_max, int strip_pow_x>
void doTest_1D()
{

    const int D = ixs_len;
    const int ixs_size = D*sizeof(int);
    int* cpu_ixs = (int*)malloc(ixs_size);
    const int step = (1 + (ix_max - ix_min)) / (ixs_len-1);
    {
        int s = ix_min;
        for(int i=0; i < D ; i++){ cpu_ixs[i] = s; s += step; }
    }
    for(int i=0; i < D ; i++){
        const int V = cpu_ixs[i];
        if(ix_min <= V && V <= ix_max)
        {}
        else { printf("index array contains indexes not in range\n"); exit(1);}
    }
    //CUDASSERT(cudaMemcpyToSymbol(ixs_1d, cpu_ixs, ixs_size));

    const long len = lens;
    cout << "{ x_len = " << len << " }" << endl;
    T* cpu_out = (T*)malloc(len*sizeof(T));
    run_cpu_1d<D, ix_min, ix_max>(cpu_ixs,len, cpu_out);

    cout << "ixs[" << D << "] = [";
    cout << cpu_ixs[0] << ", ";
    cout << cpu_ixs[1] << ", ";
    if(D == 3){ cout << cpu_ixs[2]; }
    else{ cout << "... , " << cpu_ixs[D-1]; }
    cout << "]" << endl;

    const long shared_len = (gps_x + (ix_max - ix_min));
    const long shared_size = shared_len * sizeof(T);
    const long small_shared_size = gps_x * sizeof(T);

    constexpr int singleDim_block = gps_x;
    constexpr int singleDim_grid = CEIL_DIV(len, singleDim_block); 
    constexpr int smallWork = len+(ix_max - ix_min);
    constexpr int smallBlock = singleDim_block-(ix_max - ix_min);
    constexpr int smallSingleDim_grid = divUp(smallWork,smallBlock); // the flattening happens in the before the kernel call.

    {

        {
            cout << "## Benchmark 1d global read inline ixs ##";
            Kernel1dPhysMultiDim kfun = global_read_1d_inline
                <ix_min,ix_max,gps_x>;
            G.do_run_multiDim(kfun, cpu_out, singleDim_grid, singleDim_block, 1, false); // warmup as it is the first kernel
            G.do_run_multiDim(kfun, cpu_out, singleDim_grid, singleDim_block, 1);
        }
        {
            cout << "## Benchmark 1d big tile inline ixs ##";
            Kernel1dPhysMultiDim kfun = big_tile_1d_inline
                <ix_min,ix_max,gps_x>;
            G.do_run_multiDim(kfun, cpu_out, singleDim_grid, singleDim_block, shared_size);
        }

        {
            cout << "## Benchmark 1d small tile inline ixs ##";
            Kernel1dPhysMultiDim kfun = small_tile_1d_inline
                <ix_min,ix_max,gps_x>;
            G.do_run_multiDim(kfun, cpu_out, smallSingleDim_grid, singleDim_block, small_shared_size);
        }

        {

            constexpr int strip_x = 1 << strip_pow_x;

            constexpr int strip_size_x = gps_x*strip_x;

            constexpr int sh_x = strip_size_x + (ix_max - ix_min);
            constexpr int sh_total = sh_x;
            constexpr int sh_total_mem_usage = sh_total * sizeof(T);
            const int strip_grid = int(divUp(len, long(strip_size_x)));
            const int strip_grid_flat = strip_grid;
            constexpr int max_shared_mem = 0xc000;
            static_assert(sh_total_mem_usage <= max_shared_mem,
                    "Current configuration requires too much shared memory\n");

            {
                cout << "## Benchmark 1d big tile - inlined idxs - stripmined: ";
                printf("strip_size=[%d]f32 \n", strip_size_x);
                Kernel1dPhysStripDim kfun = stripmine_big_tile_1d_inlined
                    <ix_min
                    ,ix_max
                    ,gps_x
                    ,strip_x
                    >;
                G.do_run_1d_stripmine(kfun, cpu_out, strip_grid_flat, singleDim_block);
            }
        }
        /*{
            cout << "## Benchmark 2d global read - inlined ixs - singleDim grid ##";
            Kernel2dPhysSingleDim kfun = global_reads_2d_inline_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_singleDim(kfun, cpu_out, singleDim_grid_flat, singleDim_block, singleDim_grid, 1);
        }*/
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
        
        /* THIS
        GPU_RUN(call_kernel_1d(
                    (global_read_1d_inline_reduce<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d global read inline ixs reduce ##",(void)0,(void)0);

        */
/*          
        const int width = ix_min + ix_max + 1;
        if(width < BLOCKSIZE-20){
            GPU_RUN(call_inSharedKernel_1d(
                        (small_tile_1d_inline_reduce<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                    ,"## Benchmark 1d small tile inline ixs reduce ##",(void)0,(void)0);
        }
*/      
        /* THIS
        GPU_RUN(call_kernel_1d(
                    (big_tile_1d_inline_reduce<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d big tile inline ixs reduce ##",(void)0,(void)0);
        GPU_RUN(call_kernel_1d(
                    (global_read_1d_inline<ixs_len,(-ix_min),ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, max_ix_x)))
                ,"## Benchmark 1d global read inline ixs ##",(void)0,(void)0);
        GPU_RUN(call_kernel_1d(
                    (big_tile_1d_inline<ixs_len,(-ix_min),ix_max><<<grid,block,shared_size>>>(gpu_array_in, gpu_array_out, max_ix_x, shared_len)))
                ,"## Benchmark 1d big tile thread inline ixs ##",(void)0,(void)0);

        GPU_RUN_END;
        */
    }   

    free(cpu_out);
    free(cpu_ixs);
}


int main()
{

    constexpr int gps_x = 256;
    doTest_1D<3,gps_x,-1,1,3>();
    doTest_1D<5,gps_x,-2,2,3>();
    doTest_1D<7,gps_x,-3,3,3>();
    doTest_1D<9,gps_x,-4,4,3>();
    doTest_1D<11,gps_x,-5,5,3>();
    doTest_1D<13,gps_x,-6,6,3>();
    doTest_1D<15,gps_x,-7,7,3>();
    doTest_1D<17,gps_x,-8,8,3>();
    /*
    doTest_1D<19,gps_x,9,9>();
    doTest_1D<21,gps_x,10,10>();
    doTest_1D<23,gps_x,11,11>();
    doTest_1D<25,gps_x,12,12>();
    doTest_1D<27,gps_x,13,13>();
    doTest_1D<29,gps_x,14,14>();
    doTest_1D<31,gps_x,15,15>();
    doTest_1D<33,gps_x,16,16>();
    doTest_1D<35,gps_x,17,17>();
    doTest_1D<37,gps_x,18,18>();
    doTest_1D<39,gps_x,19,19>();
    doTest_1D<41,gps_x,20,20>();
    doTest_1D<43,gps_x,21,21>();
    doTest_1D<45,gps_x,22,22>();
    doTest_1D<47,gps_x,23,23>();
    doTest_1D<49,gps_x,24,24>();
    doTest_1D<51,gps_x,25,25>();
    doTest_1D<101,gps_x,50,50>();
    doTest_1D<201,gps_x,100,100>();
    doTest_1D<301,gps_x,150,150>();
    doTest_1D<401,gps_x,200,200>();
    doTest_1D<501,gps_x,250,250>();

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

    doTest_1D<3,50,50>();
    doTest_1D<3,100,100>();
    doTest_1D<3,200,200>();
    doTest_1D<3,300,300>();
    doTest_1D<3,400,400>();
    doTest_1D<3,450,450>();
    doTest_1D<3,500,500>();
    doTest_1D<3,600,600>();
    doTest_1D<3,700,700>();
    doTest_1D<3,800,800>();
    doTest_1D<3,900,900>();
    doTest_1D<3,1000,1000>();
    */
    return 0;
}

