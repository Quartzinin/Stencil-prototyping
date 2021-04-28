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
   (1 << 12)+1,
   (1 << 12)+1};
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
    srand(1);
    for (int i = 0; i < lens_flat; ++i)
    {
        cpu_in[i] = (T)rand();
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
    const int strip_pow_x, const int strip_pow_y
    >
void doTest_2D(const int physBlocks)
{
    const int y_range = (amax_y - amin_y) + 1;
    const int x_range = (amax_x - amin_x) + 1;
    const int ixs_len = y_range * x_range;
    //const int W = D / 2;
    const int ixs_size = ixs_len*sizeof(int2);
    int2* cpu_ixs = (int2*)malloc(ixs_size);
    {
        int q = 0;
        for(int i=0; i < y_range; i++){
            for(int j=0; j < x_range; j++){
                cpu_ixs[q++] = make_int2(j+amin_x, i+amin_y);
            }
        }
    }

    //CUDASSERT(cudaMemcpyToSymbol(ixs_2d, cpu_ixs, ixs_size));

    cout << "const int ixs[" << ixs_len << "]: ";
    cout << "y= " << amin_y << "..." << amax_y << ", x= " << amin_x << "..." << amax_x << endl;

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

    constexpr int std_sh_size_x = group_size_x + amax_x - amin_x;
    constexpr int std_sh_size_y = group_size_y + amax_y - amin_y;
    constexpr int std_sh_size_flat = std_sh_size_x * std_sh_size_y;
    constexpr int std_sh_size_bytes = std_sh_size_flat * sizeof(T);

    {
        /*
        {
            cout << "## Benchmark 2d global read - inlined ixs - multiDim grid ##";
            Kernel2dPhysMultiDim kfun = global_reads_2d_inline_multiDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_multiDim(kfun, cpu_out, multiDim_grid, multiDim_block, 1, false); // warmup as it is the first kernel
            G.do_run_multiDim(kfun, cpu_out, multiDim_grid, multiDim_block, 1);
        }
        */
        {
            cout << "## Benchmark 2d global read - inlined ixs - singleDim grid ##";
            Kernel2dPhysSingleDim kfun = global_reads_2d_inline_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_singleDim(kfun, cpu_out, singleDim_grid_flat, singleDim_block, singleDim_grid, 1);
        }
        /*
        {
            cout << "## Benchmark 2d big tile - inlined idxs - cube2d load - singleDim grid ##";
            Kernel2dPhysSingleDim kfun = big_tile_2d_inlined_cube_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_singleDim(kfun, cpu_out, singleDim_grid_flat, singleDim_block, singleDim_grid, std_sh_size_bytes);
        }
        {
            cout << "## Benchmark 2d big tile - inlined idxs - flat load (div/rem) - singleDim grid ##";
            Kernel2dPhysSingleDim kfun = big_tile_2d_inlined_flat_divrem_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_singleDim(kfun, cpu_out, singleDim_grid_flat, singleDim_block, singleDim_grid, std_sh_size_bytes);
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
        */
        {
            constexpr int strip_x = 1 << strip_pow_x;
            constexpr int strip_y = 1 << strip_pow_y;

            constexpr int strip_size_x = group_size_x*strip_x;
            constexpr int strip_size_y = group_size_y*strip_y;

            constexpr int sh_x = strip_size_x + (amax_x - amin_x);
            constexpr int sh_y = strip_size_y + (amax_y - amin_y);
            constexpr int sh_total = sh_x * sh_y;
            constexpr int sh_total_mem_usage = sh_total * sizeof(T);

            const int2 strip_grid = {
            int(divUp(lens.x, long(strip_size_x))),
            int(divUp(lens.y, long(strip_size_y)))};
            const int strip_grid_flat = product(strip_grid);

            //constexpr int2 strips = { strip_x, strip_y };

            printf("shared memory per block: stripmine = %d B\n", sh_total_mem_usage);
            constexpr int max_shared_mem = 0xc000;
            static_assert(sh_total_mem_usage <= max_shared_mem,
                    "Current configuration requires too much shared memory\n");

            {
            cout << "## Benchmark 2d big tile - inlined idxs - stripmined: ";
            printf("strip_size=[%d][%d]f32 ", strip_size_y, strip_size_x);
            cout << "- flat load (add/carry) - singleDim grid ##";
            Kernel2dPhysSingleDim kfun = stripmine_big_tile_2d_inlined_flat_addcarry_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y
                ,strip_x,strip_y
                >;
            G.do_run_singleDim(kfun, cpu_out, strip_grid_flat, singleDim_block, strip_grid, sh_total_mem_usage);
            }

            /*
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
        */
        }
        {
            constexpr int window_length_y = 64;
            constexpr int group_size_flat = group_size_y * group_size_x;
            constexpr int sh_x = group_size_flat;
            constexpr int range_exc_y = amax_y - amin_y;
            constexpr int range_exc_x = amax_x - amin_x;
            constexpr int range_y = range_exc_y + 1;

            constexpr int sh_y = range_y;
            constexpr int working_x = sh_x - range_exc_x;
            constexpr int sh_total = sh_x * sh_y;
            constexpr int sh_total_mem_usage = sh_total * sizeof(T);
            const int2 strip_grid = {
                int(divUp(lens.x, long(working_x))),
                int(divUp(lens.y, long(window_length_y)))};
            const int strip_grid_flat = product(strip_grid);
            //printf("range_y=%d, sh_y=%d\n",range_y,sh_y);
            printf("shared memory per block: sliding = %d B\n", sh_total_mem_usage);

            cout << "## Benchmark 2d sliding (small-)tile - flat - inlined idxs: ";
            printf("strip_size=[%d][%d]f32 ", window_length_y, working_x);
            cout << "- singleDim grid ##";
            Kernel2dPhysSingleDim kfun = sliding_tile_flat_smalltile_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_flat
                ,window_length_y
                >;
            G.do_run_singleDim(kfun, cpu_out, strip_grid_flat, singleDim_block, strip_grid, sh_total_mem_usage);
        }
        /*
        {
            constexpr int gpx = group_size_x;
            constexpr int gpy = group_size_y;
            constexpr int windows_y = 64/gpy;
            constexpr int work_y = windows_y * gpy;
            constexpr int sh_x = gpx;
            constexpr int range_exc_y = amax_y - amin_y;
            constexpr int range_exc_x = amax_x - amin_x;
            constexpr int range_y = range_exc_y + 1;
            constexpr int sh_used_spac_y = range_y + gpy;
            // magic to get next power of 2
            constexpr int r0 = sh_used_spac_y-1;
            constexpr int r1 = r0 | (r0 >> 1);
            constexpr int r2 = r1 | (r1 >> 2);
            constexpr int r3 = r2 | (r2 >> 4);
            constexpr int r4 = r3 | (r3 >> 8);
            constexpr int r5 = r4 | (r4 >> 16);
            constexpr int r6 = r5+1;
            //

            constexpr int sh_y = r6;
            constexpr int working_x = sh_x - range_exc_x;
            constexpr int sh_total = sh_x * sh_y;
            constexpr int sh_total_mem_usage = sh_total * sizeof(T);
            const int2 strip_grid = {
                int(divUp(lens.x, long(working_x))),
                int(divUp(lens.y, long(work_y)))};
            const int strip_grid_flat = product(strip_grid);
            //printf("range_y=%d, sh_y=%d\n",range_y,sh_y);
            printf("shared memory per block: sliding = %d B\n", sh_total_mem_usage);
            cout << "## Benchmark 2d sliding (small-)tile - inlined idxs: ";
            printf("strip_size=[%d][%d]f32 ", work_y, working_x);
            cout << "- singleDim grid ##";
            Kernel2dPhysSingleDim kfun = sliding_tile_smalltile_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,gpx,gpy
                ,windows_y
                >;
            G.do_run_singleDim(kfun, cpu_out, strip_grid_flat, singleDim_block, strip_grid, sh_total_mem_usage);
        }
        */

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
    constexpr int gps_x = 32;
    constexpr int gps_y = 8;

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


    //blockdim tests
    cout << "Blockdim y,x = " << 8 << ", " << 32 << endl;
    doTest_2D< 0,1, 0,1, 32,8,1,1>(physBlocks);
    doTest_2D<-1,1, 0,1, 32,8,1,1>(physBlocks);
    doTest_2D<-1,1,-1,1, 32,8,1,1>(physBlocks);
    doTest_2D<-1,2,-1,1, 32,8,1,1>(physBlocks);
    doTest_2D<-1,2,-1,2, 32,8,1,1>(physBlocks);
    doTest_2D<-2,2,-1,2, 32,8,1,1>(physBlocks);
    doTest_2D<-2,2,-2,2, 32,8,1,1>(physBlocks);

//    cout << "Blockdim y,x = " << 32 << ", " << 32 << endl;
//    doTest_2D< 0,1, 0,1, 32,32,1,1>(physBlocks);
//    doTest_2D<-1,1, 0,1, 32,32,1,1>(physBlocks);
//    doTest_2D<-1,1,-1,1, 32,32,1,1>(physBlocks);
//    doTest_2D<-1,2,-1,1, 32,32,1,1>(physBlocks);
//    doTest_2D<-1,2,-1,2, 32,32,1,1>(physBlocks);
//    doTest_2D<-2,2,-1,2, 32,32,1,1>(physBlocks);
//    doTest_2D<-2,2,-2,2, 32,32,1,1>(physBlocks);

//    cout << "Blockdim y,x = " << 16 << ", " << 64 << endl;
//    doTest_2D< 0,1, 0,1, 64,16,1,1>(physBlocks);
//    doTest_2D<-1,1, 0,1, 64,16,1,1>(physBlocks);
//    doTest_2D<-1,1,-1,1, 64,16,1,1>(physBlocks);
//    doTest_2D<-1,2,-1,1, 64,16,1,1>(physBlocks);
//    doTest_2D<-1,2,-1,2, 64,16,1,1>(physBlocks);
//    doTest_2D<-2,2,-1,2, 64,16,1,1>(physBlocks);
//    doTest_2D<-2,2,-2,2, 64,16,1,1>(physBlocks);

    // tests for amins > 0 and (but not at same time) amaxs < 0
    //doTest_2D<2,5,3,6, 32,8,1,1>(physBlocks);
    //doTest_2D<-5,-2,-6,-3, 32,8,1,1>(physBlocks);

    /*
    //stripmine tests
    doTest_2D<0,1,0,1, gps_x,gps_y,0,0>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,0,0>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,0,0>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,0,0>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,0,0>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,0,0>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,0,0>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,0,1>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,0,1>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,0,1>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,0,1>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,0,1>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,0,1>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,0,1>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,1,0>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,1,0>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,1,0>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,1,0>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,1,0>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,1,0>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,1,0>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,2,0>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,2,0>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,2,0>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,2,0>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,2,0>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,2,0>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,2,0>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,0,3>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,0,3>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,0,3>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,0,3>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,0,3>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,0,3>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,0,3>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,3,0>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,3,0>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,3,0>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,3,0>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,3,0>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,3,0>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,3,0>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,1,2>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,1,2>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,1,2>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,1,2>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,1,2>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,1,2>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,1,2>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,2,1>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,2,1>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,2,1>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,2,1>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,2,1>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,2,1>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,2,1>(physBlocks);
    doTest_2D<0,1,0,1, gps_x,gps_y,2,2>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,2,2>(physBlocks);
    doTest_2D<-1,1,-1,1, gps_x,gps_y,2,2>(physBlocks);
    doTest_2D<-1,2,-1,1, gps_x,gps_y,2,2>(physBlocks);
    doTest_2D<-1,2,-1,2, gps_x,gps_y,2,2>(physBlocks);
    doTest_2D<-2,2,-1,2, gps_x,gps_y,2,2>(physBlocks);
    doTest_2D<-2,2,-2,2, gps_x,gps_y,2,2>(physBlocks);
    */
    return 0;
}


