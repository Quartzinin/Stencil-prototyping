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

static constexpr long n_runs = 1000;
static constexpr long lens = (1 << 24) + 2;

static Globs
    <long,long
    ,Kernel1dVirtual
    ,Kernel1dPhysMultiDim
    ,Kernel1dPhysStripDim
    > G(lens, lens, n_runs);

template<
    const int amin_x,
    const int amax_x>
__host__
void stencil_1d_cpu(
    const T* A,
    T* out)
{
    constexpr int range = amax_x - amin_x + 1;

    const int max_ix_x = lens - 1;
    for (int gidx = 0; gidx < lens; ++gidx){
        T arr[range];
        for(int k=0; k < range; k++){
            const int x = bound<(amin_x<0),int>(gidx + (k + amin_x), max_ix_x);
            arr[k] = A[x];
        }
        out[gidx] = stencil_fun_1d<amin_x,amax_x>(arr);

    }
}

template<int ix_min, int ix_max>
void run_cpu_1d(T* cpu_out)
{
    T* cpu_in  = (T*)malloc(lens*sizeof(T));
    srand(1);
    for (int i = 0; i < lens; ++i)
    {
        cpu_in[i] = (T)rand();
    }

    struct timeval t_startpar, t_endpar, t_diffpar;
    gettimeofday(&t_startpar, NULL);
    {
        stencil_1d_cpu<ix_min, ix_max>(cpu_in,cpu_out);
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
    T* cpu_out = (T*)malloc(lens*sizeof(T));
    run_cpu_1d<ix_min, ix_max>(cpu_out);

    cout << "ixs[" << ix_min << "..." << ix_max << "]" << endl;

    const long shared_len = (gps_x + (ix_max - ix_min));
    const long shared_size = shared_len * sizeof(T);
    const long small_shared_size = gps_x * sizeof(T);

    constexpr int singleDim_block = gps_x;
    constexpr int singleDim_grid = divUp((int)lens, singleDim_block);
    constexpr int smallWork = lens+(ix_max - ix_min);
    constexpr int smallBlock = singleDim_block-(ix_max - ix_min);
    constexpr int smallSingleDim_grid = divUp(smallWork,smallBlock); // the flattening happens in the before the kernel call.

    {

        /*{

            cout << "## Benchmark 1d global read inline ixs ##";
            Kernel1dPhysMultiDim kfun = global_read_1d_inline
                <ix_min,ix_max,gps_x>;
            G.do_run_multiDim(kfun, cpu_out, singleDim_grid, singleDim_block, 1, false); // warmup as it is the first kernel
            G.do_run_multiDim(kfun, cpu_out, singleDim_grid, singleDim_block, 1);

        }*/
        /*


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
        }*/


        {

            constexpr int strip_x = 1 << strip_pow_x;

            constexpr int strip_size_x = gps_x*strip_x;

            constexpr int sh_x = strip_size_x + (ix_max - ix_min);
            constexpr int sh_total = sh_x;
            constexpr int sh_total_mem_usage = sh_total * sizeof(T);
            const int strip_grid = int(divUp(lens, long(strip_size_x)));
            const int strip_grid_flat = strip_grid;
            constexpr int max_shared_mem = 0xc000;
            static_assert(sh_total_mem_usage <= max_shared_mem,
                    "Current configuration requires too much shared memory\n");

            {
                cout << "## Benchmark 1d big tile - inlined idxs - stripmined: ";
                printf("strip_size=[%d]f32 ", strip_size_x);
                Kernel1dPhysStripDim kfun = stripmine_big_tile_1d_inlined
                    <ix_min
                    ,ix_max
                    ,gps_x
                    ,strip_x
                    >;
                G.do_run_1d_stripmine(kfun, cpu_out, strip_grid_flat, singleDim_block,false);
                G.do_run_1d_stripmine(kfun, cpu_out, strip_grid_flat, singleDim_block);
            }
            {
                cout << "## Benchmark 1d global read unrolled/stripmined - inlined idxs: ";
                printf("strip_size=[%d]f32 \n", strip_size_x);
                Kernel1dPhysStripDim kfun = global_read_1d_inline_strip
                    <ix_min
                    ,ix_max
                    ,gps_x
                    ,strip_x
                    >;
                G.do_run_1d_stripmine(kfun, cpu_out, strip_grid_flat, singleDim_block);
            }
        }
    }

    free(cpu_out);
}


int main()
{


    cout << "{ x_len = " << lens << " }" << endl;
    constexpr int gps_x = 256;

    //stripmine test
    /*doTest_1D<1,gps_x,0,0,0>();
    doTest_1D<1,gps_x,0,0,0>();
    doTest_1D<2,gps_x,0,1,0>();
    doTest_1D<3,gps_x,-1,1,0>();
    doTest_1D<5,gps_x,-2,2,0>();
    doTest_1D<7,gps_x,-3,3,0>();
    doTest_1D<9,gps_x,-4,4,0>();
    doTest_1D<11,gps_x,-5,5,0>();
    doTest_1D<13,gps_x,-6,6,0>();
    doTest_1D<15,gps_x,-7,7,0>();
    doTest_1D<17,gps_x,-8,8,0>();
    doTest_1D<25,gps_x,-12,12,0>();

    doTest_1D<1,gps_x,0,0,1>();
    doTest_1D<2,gps_x,0,1,1>();
    doTest_1D<3,gps_x,-1,1,1>();
    doTest_1D<5,gps_x,-2,2,1>();
    doTest_1D<7,gps_x,-3,3,1>();
    doTest_1D<9,gps_x,-4,4,1>();
    doTest_1D<11,gps_x,-5,5,1>();
    doTest_1D<13,gps_x,-6,6,1>();
    doTest_1D<15,gps_x,-7,7,1>();
    doTest_1D<17,gps_x,-8,8,1>();
    doTest_1D<25,gps_x,-12,12,1>(); 

    */
    //doTest_1D<1,128,0,0,0>();
    //doTest_1D<3,128,-1,1,0>();
    //doTest_1D<9,128,-4,4,0>();
    //doTest_1D<21,128,-10,10,0>();


    //doTest_1D<1,256,0,0,0>();
    //doTest_1D<3,256,-1,1,0>();
    //doTest_1D<9,256,-4,4,0>();
    //doTest_1D<9,128,-4,4,0>();
    //doTest_1D<5,256,-2,2,0>();
    //doTest_1D<9,512,-4,4,0>();
    //doTest_1D<9,1024,-4,4,0>();
    //doTest_1D<21,256,-10,10,0>();

    //doTest_1D<1,512,0,0,0>();
    //doTest_1D<3,512,-1,1,0>();
    //doTest_1D<9,512,-4,4,0>();
    //doTest_1D<21,512,-10,10,0>();

    //doTest_1D<1,1024,0,0,0>();
    //doTest_1D<3,1024,-1,1,0>();
    //doTest_1D<9,1024,-4,4,0>();
    //doTest_1D<21,1024,-10,10,0>();

    //normal runs
    //doTest_1D<1,gps_x,0,0,2>();
    //doTest_1D<2,gps_x,0,1,2>();
    //doTest_1D<3,gps_x,-1,1,2>();
    //doTest_1D<5,gps_x,-2,2,2>();
    //doTest_1D<7,gps_x,-3,3,2>();
    //doTest_1D<9,gps_x,-4,4,2>();
    //doTest_1D<11,gps_x,-5,5,2>();
    //doTest_1D<13,gps_x,-6,6,2>();
    //doTest_1D<17,gps_x,-8,8,2>();
    //doTest_1D<25,gps_x,-12,12,2>();


    //blocksize tests
    /*
    doTest_1D<2,128,0,1,0>();
    doTest_1D<3,128,-1,1,0>();
    doTest_1D<5,128,-2,2,0>();
    doTest_1D<7,128,-3,3,0>();
    doTest_1D<9,128,-4,4,0>();
    doTest_1D<11,128,-5,5,0>();
    doTest_1D<13,128,-6,6,0>();
    doTest_1D<15,128,-7,7,0>();
    doTest_1D<17,128,-8,8,0>();
    doTest_1D<25,128,-12,12,0>();

    doTest_1D<2,256,0,1,0>();
    doTest_1D<3,256,-1,1,0>();
    doTest_1D<5,256,-2,2,0>();
    doTest_1D<7,256,-3,3,0>();
    doTest_1D<9,256,-4,4,0>();
    doTest_1D<11,256,-5,5,0>();
    doTest_1D<13,256,-6,6,0>();
    doTest_1D<15,256,-7,7,0>();
    doTest_1D<17,256,-8,8,0>();
    doTest_1D<25,256,-12,12,0>();

    doTest_1D<2,512,0,1,0>();
    doTest_1D<3,512,-1,1,0>();
    doTest_1D<5,512,-2,2,0>();
    doTest_1D<7,512,-3,3,0>();
    doTest_1D<9,512,-4,4,0>();
    doTest_1D<11,512,-5,5,0>();
    doTest_1D<13,512,-6,6,0>();
    doTest_1D<15,512,-7,7,0>();
    doTest_1D<17,512,-8,8,0>();
    doTest_1D<25,512,-12,12,0>();
    
    doTest_1D<2,1024,0,1,0>();
    doTest_1D<3,1024,-1,1,0>();
    doTest_1D<5,1024,-2,2,0>();
    doTest_1D<7,1024,-3,3,0>();
    doTest_1D<9,1024,-4,4,0>();
    doTest_1D<11,1024,-5,5,0>();
    doTest_1D<13,1024,-6,6,0>();
    doTest_1D<15,1024,-7,7,0>();
    doTest_1D<17,1024,-8,8,0>();
    doTest_1D<25,1024,-12,12,0>();
    */

    doTest_1D<2,256,0,1,0>();
    doTest_1D<3,256,-1,1,0>();
    doTest_1D<5,256,-2,2,0>();
    doTest_1D<7,256,-3,3,0>();
    doTest_1D<9,256,-4,4,0>();
    doTest_1D<11,256,-5,5,0>();
    doTest_1D<13,256,-6,6,0>();
    doTest_1D<15,256,-7,7,0>();
    doTest_1D<17,256,-8,8,0>();
    doTest_1D<25,256,-12,12,0>();

    doTest_1D<2,256,0,1,1>();
    doTest_1D<3,256,-1,1,1>();
    doTest_1D<5,256,-2,2,1>();
    doTest_1D<7,256,-3,3,1>();
    doTest_1D<9,256,-4,4,1>();
    doTest_1D<11,256,-5,5,1>();
    doTest_1D<13,256,-6,6,1>();
    doTest_1D<15,256,-7,7,1>();
    doTest_1D<17,256,-8,8,1>();
    doTest_1D<25,256,-12,12,1>();

    doTest_1D<2,256,0,1,2>();
    doTest_1D<3,256,-1,1,2>();
    doTest_1D<5,256,-2,2,2>();
    doTest_1D<7,256,-3,3,2>();
    doTest_1D<9,256,-4,4,2>();
    doTest_1D<11,256,-5,5,2>();
    doTest_1D<13,256,-6,6,2>();
    doTest_1D<15,256,-7,7,2>();
    doTest_1D<17,256,-8,8,2>();
    doTest_1D<25,256,-12,12,2>();

    doTest_1D<2,256,0,1,3>();
    doTest_1D<3,256,-1,1,3>();
    doTest_1D<5,256,-2,2,3>();
    doTest_1D<7,256,-3,3,3>();
    doTest_1D<9,256,-4,4,3>();
    doTest_1D<11,256,-5,5,3>();
    doTest_1D<13,256,-6,6,3>();
    doTest_1D<15,256,-7,7,3>();
    doTest_1D<17,256,-8,8,3>();
    doTest_1D<25,256,-12,12,3>();

    return 0;
}

