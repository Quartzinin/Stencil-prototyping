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
   (1 << 12)+2,
   (1 << 12)+4};
static constexpr int lens_flat = lens.x * lens.y;

static constexpr long n_runs = 100;

static Globs
    <long2,int2
    ,Kernel2dVirtual
    ,Kernel2dPhysMultiDim
    ,Kernel2dPhysSingleDim
    > G(lens, lens_flat, n_runs);

template<
    const int amin_x, const int amin_y,
    const int amax_x, const int amax_y>
__host__
void stencil_2d_cpu(
    const T* start,
    T* out)
{
    constexpr int2 range = {
        amax_x - amin_x + 1,
        amax_y - amin_y + 1};
    constexpr int total_range = range.x * range.y;

    const int max_ix_y = lens.y - 1;
    const int max_ix_x = lens.x - 1;
    for (int gidy = 0; gidy < lens.y; ++gidy){
        for (int gidx = 0; gidx < lens.x; ++gidx){
            T arr[total_range];
            for(int j=0; j < range.y; j++){
                for(int k=0; k < range.x; k++){
                    const int y = bound<(amin_y<0),int>(gidy + (j + amin_y), max_ix_y);
                    const int x = bound<(amin_x<0),int>(gidx + (k + amin_x), max_ix_x);
                    const int index = y*lens.x + x;
                    const int flat_idx = j*range.x + k;
                    arr[flat_idx] = start[index];
                }
            }
            out[gidy* lens.x + gidx] = stencil_fun_2d<amin_x,amin_y,amax_x,amax_y>(arr);
        }
    }
}

template<
    const int amin_x, const int amin_y,
    const int amax_x, const int amax_y>
__host__
void run_cpu_2d(T* cpu_out)
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
        stencil_2d_cpu<amin_x,amin_y,amax_x,amax_y>(cpu_in,cpu_out);
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
    const int strip_pow_x_pre, const int strip_pow_y_pre
    >
void doTest_2D(const int physBlocks)
{
    constexpr int strip_x_pre = 1 << strip_pow_x_pre;
    constexpr int strip_y = 1 << strip_pow_y_pre;
    constexpr int fsz = sizeof(float);
    constexpr int esz = sizeof(T);
    constexpr int ub_strip_x = (fsz <= esz) ? (strip_x_pre/(esz/fsz)) : (strip_x_pre*(fsz/esz));
    constexpr int strip_x = (1 <= ub_strip_x) ? ub_strip_x : 1;
    const int y_range = (amax_y - amin_y) + 1;
    const int x_range = (amax_x - amin_x) + 1;
#ifdef Jacobi2D
    cout << "running Jacobi2D" << endl;
    const int ixs_len = y_range + x_range - 1;
#else
    const int ixs_len = y_range * x_range;
#endif

    cout << "const int ixs[" << ixs_len << "]: ";
    cout << "y= " << amin_y << "..." << amax_y << ", x= " << amin_x << "..." << amax_x << endl;

    T* cpu_out = (T*)malloc(lens_flat*sizeof(T));
    run_cpu_2d<amin_x,amin_y,amax_x,amax_y>(cpu_out);

    constexpr int  singleDim_block = group_size_x * group_size_y;
    constexpr int2 singleDim_grid = {
        divUp(lens.x, (long) group_size_x),
        divUp(lens.y, (long) group_size_y)
        }; // the flattening happens in the before the kernel call.
    constexpr int singleDim_grid_flat = singleDim_grid.x * singleDim_grid.y;
    constexpr dim3 multiDim_grid(singleDim_grid.x, singleDim_grid.y, 1);
    constexpr dim3 multiDim_block( group_size_x , group_size_y, 1);

    constexpr int std_sh_size_x = group_size_x + amax_x - amin_x;
    constexpr int std_sh_size_y = group_size_y + amax_y - amin_y;
    constexpr int std_sh_size_flat = std_sh_size_x * std_sh_size_y;
    constexpr int std_sh_size_bytes = std_sh_size_flat * sizeof(T);

    {

        /*{
            cout << "## Benchmark 2d global read - inlined ixs - multiDim grid ##";
            Kernel2dPhysMultiDim kfun = global_reads_2d_inline_multiDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_multiDim(kfun, cpu_out, multiDim_grid, multiDim_block, 1, false); // warmup as it is the first kernel
            G.do_run_multiDim(kfun, cpu_out, multiDim_grid, multiDim_block, 1);
        }

        {
            cout << "## Benchmark 2d global read - inlined ixs - singleDim grid ##";
            Kernel2dPhysSingleDim kfun = global_reads_2d_inline_singleDim
                <amin_x,amin_y
                ,amax_x,amax_y
                ,group_size_x,group_size_y>;
            G.do_run_singleDim(kfun, cpu_out, singleDim_grid_flat, singleDim_block, singleDim_grid, 1,false);
            G.do_run_singleDim(kfun, cpu_out, singleDim_grid_flat, singleDim_block, singleDim_grid, 1);
        }

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
        }*/
        {

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
            //static_assert(sh_total_mem_usage <= max_shared_mem,
            //      "Current configuration requires too much shared memory\n");

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
                G.do_run_singleDim(kfun, cpu_out, strip_grid_flat, singleDim_block, strip_grid, sh_total_mem_usage,false);
                G.do_run_singleDim(kfun, cpu_out, strip_grid_flat, singleDim_block, strip_grid, sh_total_mem_usage);
            }


            /*{
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
            }*/

        }

        
        /*{
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
            //printf("shared memory per block: sliding = %d B\n", sh_total_mem_usage);

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


        {
            constexpr int group_size_flat = group_size_y * group_size_x;
            constexpr int gpx = 128;
            constexpr int gpy = group_size_flat/gpx;
            constexpr int windows_y = 64/gpy;
            constexpr int work_y = windows_y * gpy;
            constexpr int sh_x = gpx;
            constexpr int range_exc_y = amax_y - amin_y;
            constexpr int range_exc_x = amax_x - amin_x;
            constexpr int sh_y = range_exc_y + gpy;

            constexpr int working_x = sh_x - range_exc_x;
            constexpr int sh_total = sh_x * sh_y;
            constexpr int sh_total_mem_usage = sh_total * sizeof(T);
            const int2 strip_grid = {
                int(divUp(lens.x, long(working_x))),
                int(divUp(lens.y, long(work_y)))};
            const int strip_grid_flat = product(strip_grid);
            //printf("range_y=%d, sh_y=%d\n",range_y,sh_y);
            //printf("shared memory per block: sliding = %d B\n", sh_total_mem_usage);
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
        }*/


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

    // to avoid unused varible warning.
    (void)singleDim_grid_flat;
    (void)std_sh_size_bytes;
}


int main()
{


    // group sizes
    constexpr int gps_x = 32;
    constexpr int gps_y = 8;
    constexpr int group_size_flat = gps_x * gps_y;
    int physBlocks = getPhysicalBlockCount<group_size_flat>();

    cout << "{ x_len = " << lens.x << ", y_len = " << lens.y
         << ", total_len = " << lens_flat << " }" << endl;
#ifdef Jacobi2D
    cout << "running Jacobi 2D" << endl;
#else
    cout << "running Dense stencil with mean" << endl;
#endif


    //blockdim tests
    


    cout << "Blockdim y,x = " << 32 << ", " << 32 << endl;
    doTest_2D< 0,1, 0,1, 32,32,0,0>(physBlocks);
    doTest_2D<-1,1, 0,1, 32,32,0,0>(physBlocks);
    doTest_2D<-1,1,-1,1, 32,32,0,0>(physBlocks);
    doTest_2D<-1,2,-1,1, 32,32,0,0>(physBlocks);
    doTest_2D<-1,2,-1,2, 32,32,0,0>(physBlocks);
    doTest_2D<-2,2,-1,2, 32,32,0,0>(physBlocks);
    doTest_2D<-2,2,-2,2, 32,32,0,0>(physBlocks);

    cout << "Blockdim y,x = " << 16 << ", " << 32 << endl;
    doTest_2D< 0,1, 0,1, 32,16,0,0>(physBlocks);
    doTest_2D<-1,1, 0,1, 32,16,0,0>(physBlocks);
    doTest_2D<-1,1,-1,1, 32,16,0,0>(physBlocks);
    doTest_2D<-1,2,-1,1, 32,16,0,0>(physBlocks);
    doTest_2D<-1,2,-1,2, 32,16,0,0>(physBlocks);
    doTest_2D<-2,2,-1,2, 32,16,0,0>(physBlocks);
    doTest_2D<-2,2,-2,2, 32,16,0,0>(physBlocks);

    cout << "Blockdim y,x = " << 8 << ", " << 32 << endl;
    doTest_2D< 0,1, 0,1, 32,8,0,0>(physBlocks);
    doTest_2D<-1,1, 0,1, 32,8,0,0>(physBlocks);
    doTest_2D<-1,1,-1,1, 32,8,0,0>(physBlocks);
    doTest_2D<-1,2,-1,1, 32,8,0,0>(physBlocks);
    doTest_2D<-1,2,-1,2, 32,8,0,0>(physBlocks);
    doTest_2D<-2,2,-1,2, 32,8,0,0>(physBlocks);
    doTest_2D<-2,2,-2,2, 32,8,0,0>(physBlocks);

    cout << "Blockdim y,x = " << 4 << ", " << 32 << endl;
    doTest_2D< 0,1, 0,1, 32,4,0,0>(physBlocks);
    doTest_2D<-1,1, 0,1, 32,4,0,0>(physBlocks);
    doTest_2D<-1,1,-1,1, 32,4,0,0>(physBlocks);
    doTest_2D<-1,2,-1,1, 32,4,0,0>(physBlocks);
    doTest_2D<-1,2,-1,2, 32,4,0,0>(physBlocks);
    doTest_2D<-2,2,-1,2, 32,4,0,0>(physBlocks);
    doTest_2D<-2,2,-2,2, 32,4,0,0>(physBlocks);
    

    //testing reuse

    /*doTest_2D<-1,1,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-1,2,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-2,2,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-2,3,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-3,3,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-3,4,0,1, gps_x,gps_y,0,2>(physBlocks);*/
    //doTest_2D<-1,1,-1,1, 32,4,3,3>(physBlocks);
    //doTest_2D<-1,1,-1,1, 32,8,1,1>(physBlocks);
    //doTest_2D<-1,1,-1,1, 32,8,2,2>(physBlocks);
    //doTest_2D<-1,1,-1,1, 32,16,0,0>(physBlocks);
    //doTest_2D<-1,1,-1,1, 32,32,0,0>(physBlocks);
    /*doTest_2D<-4,5,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-5,5,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-5,6,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-6,6,0,1, gps_x,gps_y,0,2>(physBlocks);
    doTest_2D<-1,1,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-1,2,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-2,2,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-2,3,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-3,3,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-3,4,0,1, gps_x,gps_y,1,1>(physBlocks);*/
    /*doTest_2D<-4,5,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-5,5,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-5,6,0,1, gps_x,gps_y,1,1>(physBlocks);
    doTest_2D<-6,6,0,1, gps_x,gps_y,1,1>(physBlocks);*/

//    doTest_2D<-1,1,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-1,2,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-2,2,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-2,3,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-3,3,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-3,4,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-4,4,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-4,5,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-5,5,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-5,6,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-6,6,0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-1,1,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-1,2,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-2,2,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-2,3,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-3,3,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-3,4,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-4,4,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-4,5,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-5,5,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-5,6,0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-6,6,0,1, gps_x,gps_y,1,1>(physBlocks);

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


    //stripmine tests
    //doTest_2D< 0,1, 0,1, gps_x,gps_y,0,0>(physBlocks);
    //doTest_2D<-1,1, 0,1, gps_x,gps_y,0,0>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,0,0>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,0,0>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,0,0>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,0,0>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,0,0>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,0,1>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,0,1>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,0,1>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,0,1>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,0,1>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,0,1>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,0,1>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,0,2>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,1,0>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,1,0>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,1,0>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,1,0>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,1,0>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,1,0>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,1,0>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,2,0>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,2,0>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,2,0>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,2,0>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,2,0>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,2,0>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,2,0>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,0,3>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,0,3>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,0,3>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,0,3>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,0,3>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,0,3>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,0,3>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,3,0>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,3,0>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,3,0>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,3,0>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,3,0>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,3,0>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,3,0>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,1,1>(physBlocks);
      //inst_fp_32, inst_integer, inst_bit_convert, inst_control, inst_compute_ld_st, inst_misc

    //Stripmine
      //doTest_2D<-1,0,-1,0, 32,8,0,0>(physBlocks);
    //doTest_2D<-1,0,-1,0, 32,8,0,0>(physBlocks);
    //doTest_2D<-1,0,-1,0, 32,8,1,0>(physBlocks);
    //doTest_2D<-1,0,-1,0, 32,8,1,1>(physBlocks);
    //doTest_2D<-1,0,-1,0, 32,8,2,2>(physBlocks);
    //doTest_2D<-1,0,-1,0, 32,8,3,3>(physBlocks);
      //doTest_2D<-1,0,-1,1, 32,8,3,3>(physBlocks);
      //doTest_2D<-1,1,-1,1, 32,8,0,0>(physBlocks);

    //Blocksize testing
      //doTest_2D<-1,0,-1,0, 32,8,0,0>(physBlocks);
      //doTest_2D<-1,0,-1,1, 32,8,0,0>(physBlocks);
      //doTest_2D<-1,1,-1,1, 32,8,0,0>(physBlocks);
      
      //doTest_2D<-1,0,-1,0, 32,16,0,0>(physBlocks);
      //doTest_2D<-1,0,-1,1, 32,16,0,0>(physBlocks);
      //doTest_2D<-1,1,-1,1, 32,16,0,0>(physBlocks);

      //doTest_2D<-1,0,-1,0, 32,32,0,0>(physBlocks);
      //doTest_2D<-1,0,-1,1, 32,32,0,0>(physBlocks);
      //doTest_2D<-1,1,-1,1, 32,32,0,0>(physBlocks);

//    doTest_2D<-1,2,-1,1, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,1,1>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,1,2>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,1,2>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,1,2>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,1,2>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,1,2>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,1,2>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,1,2>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,2,1>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,2,1>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,2,1>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,2,1>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,2,1>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,2,1>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,2,1>(physBlocks);
//    doTest_2D< 0,1, 0,1, gps_x,gps_y,2,2>(physBlocks);
//    doTest_2D<-1,1, 0,1, gps_x,gps_y,2,2>(physBlocks);
//    doTest_2D<-1,1,-1,1, gps_x,gps_y,2,2>(physBlocks);
//    doTest_2D<-1,2,-1,1, gps_x,gps_y,2,2>(physBlocks);
//    doTest_2D<-1,2,-1,2, gps_x,gps_y,2,2>(physBlocks);
//    doTest_2D<-2,2,-1,2, gps_x,gps_y,2,2>(physBlocks);
//    doTest_2D<-2,2,-2,2, gps_x,gps_y,2,2>(physBlocks);

    return 0;
}


