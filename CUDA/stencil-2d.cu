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
#include "kernels-2d.h"

static constexpr long2 lens = {
   (1 << 14) - 1,
   (1 << 10) - 1};
static constexpr int lens_flat = lens.x * lens.y;
static constexpr long n_runs = 100;
static Globs
    <long2,int2
    ,Kernel2dVirtual
    ,Kernel2dPhysMultiDim
    ,Kernel2dPhysSingleDim
    > G(lens, lens_flat, n_runs);

template<int D>
void stencil_2d_cpu(
    const T* start,
    const int2* idxs,
    T* out)
{
    const int max_y_ix = lens.y - 1;
    const int max_x_ix = lens.x - 1;
    for (int i = 0; i < lens.y; ++i)
    {
        for (int k = 0; k < lens.x; ++k)
        {
            T arr[D];
            for (int j = 0; j < D; ++j)
            {
                int y = BOUND(i + idxs[j].y, max_y_ix);
                int x = BOUND(k + idxs[j].x, max_x_ix);
                int index = y * lens.x + x;
                arr[j] = start[index];
            }
            T lambda_res = stencil_fun_cpu<D>(arr);
            out[i * lens.x + k] = lambda_res;
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
    const int wasted_x = x_min + x_max;\
    const int wasted_y = y_min + y_max;\
    const int working_block_x = SQ_BLOCKSIZE-wasted_x;\
    const int working_block_y = SQ_BLOCKSIZE-wasted_y;\
    const int BNx = CEIL_DIV(x_len, working_block_x);\
    const int BNy = CEIL_DIV(y_len   , working_block_y);\
    const dim3 grid(BNx, BNy, 1);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

template<int D>
void run_cpu_2d(const int2* idxs, T* cpu_out)
{
    T* cpu_in = (T*)malloc(lens_flat*sizeof(T));

    for (int i = 0; i < lens_flat; ++i)
    {
        cpu_in[i] = (T)(i+1);
    }

    struct timeval t_startpar, t_endpar, t_diffpar;
    gettimeofday(&t_startpar, NULL);
    {
        stencil_2d_cpu<D>(cpu_in,idxs,cpu_out);
    }
    gettimeofday(&t_endpar, NULL);
    timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
    const unsigned long elapsed = (t_diffpar.tv_sec*1e6+t_diffpar.tv_usec) / 1000;
    const unsigned long seconds = elapsed / 1000;
    const unsigned long microseconds = elapsed % 1000;
    printf("cpu c 2d for 1 run : %lu.%03lu seconds\n", seconds, microseconds);

    free(cpu_in);
}

template<
    const int amin_y, const int amax_y,
    const int amin_x, const int amax_x,
    const int group_size_x,  const int group_size_y,
    const int strip_x, const int strip_y
    >
void doTest_2D(const int physBlocks)
{
    const int y_range = (amin_y + amax_y + 1);
    const int x_range = (amin_x + amax_x + 1);
    const int ixs_len = y_range * x_range;
    //const int W = D / 2;
    const int ixs_size = ixs_len*sizeof(int2);
    int2* cpu_ixs = (int2*)malloc(ixs_size);
    {
        int q = 0;
        for(int i=0; i < y_range; i++){
            for(int j=0; j < x_range; j++){
                cpu_ixs[q++] = make_int2(j-amin_x, i-amin_y);
            }
        }
    }

    //CUDASSERT(cudaMemcpyToSymbol(ixs_2d, cpu_ixs, ixs_size));

    cout << "const int ixs[" << ixs_len << "]: ";
    cout << "y= " << -amin_y << "..." << amax_y << ", x= " << -amin_x << "..." << amax_x << endl;

    T* cpu_out = (T*)malloc(lens_flat*sizeof(T));
    run_cpu_2d<ixs_len>(cpu_ixs, cpu_out);

    constexpr int  singleDim_block = group_size_x * group_size_y;
    constexpr int2 singleDim_grid = {
        CEIL_DIV(lens.x, group_size_x),
        CEIL_DIV(lens.y, group_size_y)
        }; // the flattening happens in the before the kernel call.
    constexpr int singleDim_grid_flat = singleDim_grid.x * singleDim_grid.y;
    constexpr dim3 multiDim_grid(singleDim_grid.x, singleDim_grid.y, 1);
    constexpr dim3 multiDim_block( group_size_x , group_size_y, 1);

    constexpr int std_sh_size_x = amin_x + group_size_x + amax_x;
    constexpr int std_sh_size_y = amin_y + group_size_y + amax_y;
    constexpr int std_sh_size_flat = std_sh_size_x * std_sh_size_y;
    constexpr int std_sh_size_bytes = std_sh_size_flat * sizeof(T);

    {
        {
            cout << "## Benchmark 2d global read - inlined ixs - multiDim grid ##";
            Kernel2dPhysMultiDim kfun = global_reads_2d_inline_multiDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_multiDim(kfun, cpu_out, multiDim_grid, multiDim_block, false); // warmup as it is the first kernel
            G.do_run_multiDim(kfun, cpu_out, multiDim_grid, multiDim_block);
        }
        {
            cout << "## Benchmark 2d global read - inlined ixs - singleDim grid ##";
            Kernel2dPhysSingleDim kfun = global_reads_2d_inline_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_singleDim(kfun, cpu_out, singleDim_grid_flat, singleDim_block, singleDim_grid, 1);
        }
        {
            cout << "## Benchmark 2d big tile - inlined idxs - flat load (add/carry) - singleDim grid ##";
            Kernel2dPhysSingleDim kfun = big_tile_2d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_singleDim(kfun, cpu_out, singleDim_grid_flat, singleDim_block, singleDim_grid, std_sh_size_bytes);
        }
        {
            cout << "## Benchmark 2d virtual (add/carry) - big tile - inlined idxs - flat load (add/carry) - singleDim grid ##";
            Kernel2dVirtual kfun = virtual_addcarry_big_tile_2d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_virtual(kfun, cpu_out, physBlocks, singleDim_block, singleDim_grid, std_sh_size_bytes);
        }
        {
            constexpr int strip_size_x = group_size_x*strip_x;
            constexpr int strip_size_y = group_size_y*strip_y;

            constexpr int sh_x = strip_size_x + amin_x + amax_x;
            constexpr int sh_y = strip_size_y + amin_y + amax_y;
            constexpr int sh_total = sh_x * sh_y;
            constexpr int sh_total_mem_usage = sh_total * sizeof(T);

            //constexpr int2 strips = { strip_x, strip_y };

            //printf("shared memory used = %d B\n", sh_total_mem_usage);
            constexpr int max_shared_mem = 0xc000;
            static_assert(sh_total_mem_usage <= max_shared_mem,
                    "Current configuration requires too much shared memory\n");
            cout << "## Benchmark 2d virtual (add/carry) - stripmined big tile, ";
            printf("strip_size=[%d][%d]f32 ", strip_size_y, strip_size_x);
            cout << "- inlined idxs - flat load (add/carry) - singleDim grid ##";
            Kernel2dVirtual kfun = virtual_addcarry_stripmine_big_tile_2d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y
                ,strip_x,strip_y
                >;
            G.do_run_virtual(kfun, cpu_out, physBlocks, singleDim_block, singleDim_grid, sh_total_mem_usage
                    //, strips
                    );
        }

        //GPU_RUN_INIT;
        /*
        GPU_RUN(call_kernel_2d(
                    (global_reads_2d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                ,"## Benchmark 2d global read ##",(void)0,(void)0);
        GPU_RUN(call_small_tile_2d(
                    (small_tile_2d<ixs_len,amin_x,amax_x,amin_y,amax_y><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                ,"## Benchmark 2d small tile ##",(void)0,(void)0);
        GPU_RUN(call_kernel_2d(
                    (big_tile_2d<ixs_len,amin_x,amax_x,amin_y,amax_y><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                ,"## Benchmark 2d big tile ##",(void)0,(void)0);
        */
        //GPU_RUN(call_kernel_2d(
        //            (global_reads_2d_inline_reduce<amin_x,amax_x,amin_y,amax_y><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
        //        ,"## Benchmark 2d global read inline ixs reduce ##",(void)0,(void)0);
/*
        GPU_RUN(call_small_tile_2d(
                    (small_tile_2d_inline_reduce<amin_x,amax_x,amin_y,amax_y><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
                ,"## Benchmark 2d small tile inline ixs reduce ##",(void)0,(void)0);
*/
        //GPU_RUN(call_kernel_2d(
        //            (big_tile_2d_inline_reduce<amin_x,amax_x,amin_y,amax_y><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
        //        ,"## Benchmark 2d big tile inline ixs reduce ##",(void)0,(void)0);
        //GPU_RUN(call_kernel_2d(
        //            (big_tile_2d_inline_reduce_flat<amin_x,amax_x,amin_y,amax_y><<<grid,block>>>(gpu_array_in, gpu_array_out, x_len, y_len)))
        //        ,"## Benchmark 2d big tile inline ixs reduce flat ##",(void)0,(void)0);
        //GPU_RUN_END;
    }

    free(cpu_out);
    free(cpu_ixs);
}


int main()
{

    int physBlocks = getPhysicalBlockCount();

    // group sizes
    constexpr int gps_x = 1 << 5;
    constexpr int gps_y = 1 << 5;

    constexpr int group_size_flat = gps_x * gps_y;
    static_assert(
            32 <= group_size_flat
        &&  group_size_flat <= 1024
        &&  (group_size_flat % 32) == 0
        , "invalid group size"
    );

    cout << "{ x_len = " << lens.x << ", y_len = " << lens.y
         << ", total_len = " << lens_flat << " }" << endl;
    cout << "Blockdim y,x = " << gps_y << ", " << gps_x << endl;

    doTest_2D<1,1,0,0, gps_x,gps_y,1,4>(physBlocks);
    doTest_2D<2,2,0,0, gps_x,gps_y,1,4>(physBlocks);
    doTest_2D<3,3,0,0, gps_x,gps_y,1,4>(physBlocks);
    doTest_2D<4,4,0,0, gps_x,gps_y,1,4>(physBlocks);
    doTest_2D<5,5,0,0, gps_x,gps_y,1,4>(physBlocks);

    doTest_2D<1,1,1,1, gps_x,gps_y,3,3>(physBlocks);
    doTest_2D<2,2,2,2, gps_x,gps_y,3,3>(physBlocks);
    doTest_2D<3,3,3,3, gps_x,gps_y,3,2>(physBlocks);
    doTest_2D<4,4,4,4, gps_x,gps_y,3,2>(physBlocks);
    doTest_2D<5,5,5,5, gps_x,gps_y,3,2>(physBlocks);

    return 0;
}


