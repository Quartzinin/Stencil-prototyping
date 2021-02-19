#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include "kernels.h"
using namespace std;

#include <iostream>
using std::cout;
using std::endl;


#define GPU_RUN(call,benchmark_name, preproc, destroy) {\
    const int mem_size = len*sizeof(int); \
    int* arr_in  = (int*)malloc(mem_size*2); \
    int* arr_out = arr_in + len; /*(int*)malloc(mem_size);*/ \
    for(int i=0; i<len; i++){ arr_in[i] = i+1; } \
    int* gpu_array_in; \
    int* gpu_array_out; \
    CUDASSERT(cudaMalloc((void **) &gpu_array_in, 2*mem_size)); \
    gpu_array_out = gpu_array_in + len; \
    /*CUDASSERT(cudaMalloc((void **) &gpu_array_out, mem_size));*/ \
    CUDASSERT(cudaMemcpy(gpu_array_in, arr_in, mem_size, cudaMemcpyHostToDevice));\
    CUDASSERT(cudaMemset(gpu_array_out, 0, mem_size));\
    (preproc);\
    CUDASSERT(cudaDeviceSynchronize());\
    cout << (benchmark_name) << endl; \
    gettimeofday(&t_startpar, NULL); \
    for(unsigned x = 0; x < RUNS; x++){ \
        (call); \
    }\
    CUDASSERT(cudaDeviceSynchronize());\
    gettimeofday(&t_endpar, NULL);\
    CUDASSERT(cudaMemcpy(arr_out, gpu_array_out, mem_size, cudaMemcpyDeviceToHost));\
    CUDASSERT(cudaDeviceSynchronize());\
    timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);\
    unsigned long elapsed = t_diffpar.tv_sec*1e6+t_diffpar.tv_usec;\
    elapsed /= RUNS;\
    printf("    mean elapsed time was: %lu microseconds\n", elapsed);\
    printf("%d %d %d %d %d %d\n", arr_out[0], arr_out[1], arr_out[2], arr_out[3],arr_out[10], arr_out[len-1]); \
    free(arr_in);\
    /*free(arr_out);*/\
    cudaFree(gpu_array_in);\
    /*cudaFree(gpu_array_out);*/\
    (destroy);\
}


static int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


static inline void cudAssert(cudaError_t exit_code,
        const char *file,
        int         line) {
    if (exit_code != cudaSuccess) {
        fprintf(stderr, ">>> Cuda run-time error: %s, at %s:%d\n",
                cudaGetErrorString(exit_code), file, line);
        exit(exit_code);
    }
}
#define CUDASSERT(exit_code) { cudAssert((exit_code), __FILE__, __LINE__); }

template<int W>
void stencil_1d_tiled(
    const int * start,
    int * out,
    const unsigned len
    )
{
    const int block = 1024;
    const int grid = (len + (block-1)) / block;

    big_tiled_generic1d<W,block><<<grid,block>>>(start, out, len);
    CUDASSERT(cudaDeviceSynchronize());
}

template<int W>
void stencil_1d_global_read(
    const int * start,
    int * out,
    const unsigned len
    )
{
    const int block = 1024;
    const int grid = (len + (block-1)) / block;

    breathFirst_generic1d<W><<<grid,block>>>(start, out, len);
    CUDASSERT(cudaDeviceSynchronize());
}
template<int D, int block>
void stencil_1d_inSharedtiled(
    const int * start,
    const int * ixs,
    int * out,
    const unsigned len
    )
{
    const int working_block = block-D;
    const int grid = (D + len + (working_block-1)) / working_block;

    inSharedtiled_generic1d<D,block><<<grid,block>>>(start, ixs, out, len);
    CUDASSERT(cudaDeviceSynchronize());
}

#define call_kernel(kernel,blocksize) {\
    const int block = blocksize;\
    const int grid = (len + (block-1)) / block;\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}


template<int D, int block_size>
void stencil_1d_global_temp(
    const int * start,
    const int * ixs,
    int * temp,
    int * out,
    const int len
    )
{
    const int grid1 = (len*D + (block_size-1)) / block_size;
    const int grid2 = (len + (block_size-1)) / block_size;

    global_temp_generic_1d_to_temp<D><<<grid1,block_size>>>(start, ixs, temp, len);
    global_temp_generic_1d<D><<<grid2,block_size>>>(temp, out, len);
    CUDASSERT(cudaDeviceSynchronize());
}


template<int W, int RUNS, int standard_block_size>
void doTest()
{
    struct timeval t_startpar, t_endpar, t_diffpar;
    int* temp;

    const int D = (2*W+1);
    const int ixs_size = D*sizeof(int);
    int* ixs = (int*)malloc(ixs_size);
    for(int i=0; i < D ; i++){ ixs[i] = i-W; } \
    int* gpu_ixs;
    CUDASSERT(cudaMalloc((void **) &gpu_ixs, ixs_size));
    CUDASSERT(cudaMemcpy(gpu_ixs, ixs, ixs_size, cudaMemcpyHostToDevice));

    {
        const int len = 5000000;

        GPU_RUN(call_kernel(
                    (breathFirst_generic1d<W><<<grid,block>>>(gpu_array_in, gpu_array_out, len))
                    ,standard_block_size)
                ,"## Benchmark GPU 1d global-mem ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (big_tiled_generic1d<W,block><<<grid,block>>>(gpu_array_in, gpu_array_out, len))
                    ,standard_block_size)
                ,"## Benchmark GPU 1d tiled ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (inlinedIndexesBreathFirst_generic1d<W><<<grid,block>>>(gpu_array_in, gpu_array_out, len))
                    ,standard_block_size)
                ,"## Benchmark GPU 1d inlined global read indxs ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (outOfSharedtiled_generic1d<W,block><<<grid,block>>>(gpu_array_in, gpu_array_out, len))
                    ,standard_block_size)
                ,"## Benchmark GPU 1d out of shared tiled ##",(void)0,(void)0);
        GPU_RUN((stencil_1d_inSharedtiled<D, standard_block_size>(gpu_array_in, gpu_ixs, gpu_array_out, len)),
                "## Benchmark GPU 1d in shared tiled ##",(void)0,(void)0);
        GPU_RUN((stencil_1d_global_temp<D, standard_block_size>(gpu_array_in, gpu_ixs, temp, gpu_array_out, len)),
                "## Benchmark GPU 1d global temp ##"
                ,(CUDASSERT(cudaMalloc((void **) &temp, D*mem_size)))
                ,(cudaFree(temp)));
    }

    cudaFree(gpu_ixs);
    free(ixs);
}

int main()
{
    doTest<9,1000,1024>();
    return 0;
}

















/*static void sevenPointStencil(
        float * start,
        float * swap_out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz,
        const unsigned iterations // must be odd
        )
{
    const int T = 32;
    const int dimx = (nz + (T-1))/T;
    const int dimy = (ny + (T-1))/T;
    dim3 block(T,T,1);
    dim3 grid(dimx, dimy, 1);

    for (unsigned i = 0; i < iterations; ++i){
        if(i & 1){
            sevenPointStencil_single_iter<<< grid,block >>>(swap_out, start, nx, ny, nz);
        }
        else {
            sevenPointStencil_single_iter<<< grid,block >>>(start, swap_out, nx, ny, nz);
        }
    }
    CUDASSERT(cudaDeviceSynchronize());

}

static void sevenPointStencil_tiledSliding(
        float * start,
        float * swap_out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz,
        const unsigned iterations // must be odd
        )
{
    const int T = 32;
    const int dimx = (nz + (T-1))/T;
    const int dimy = (ny + (T-1))/T;
    dim3 block(T,T,1);
    dim3 grid(dimx, dimy, 1);

    for (unsigned i = 0; i < iterations; ++i){
        if(i & 1){
            sevenPointStencil_single_iter_tiled_sliding <<< grid,block >>>(swap_out, start, nx, ny, nz);
        }
        else {
            sevenPointStencil_single_iter_tiled_sliding <<< grid,block >>>(start, swap_out, nx, ny, nz);
        }
    }
    CUDASSERT(cudaDeviceSynchronize());

}
static void sevenPointStencil_tiledSliding_fully(
        float * start,
        float * swap_out,
        const unsigned nx,
        const unsigned ny,
        const unsigned nz,
        const unsigned iterations // must be odd
        )
{
    const unsigned T = 32;
    const unsigned Ts = 6;
    const unsigned dimx = (nx + (T-1))/T;
    const unsigned dimy = (ny + (Ts-1))/Ts;
    const unsigned dimz = (nz + (Ts-1))/Ts;
    dim3 block(32,6,6);
    dim3 grid(dimx, dimy, dimz);

    for (unsigned i = 0; i < iterations; ++i){
        if(i & 1){
            sevenPointStencil_single_iter_tiled_sliding_read<<<grid,block>>>(swap_out, start, nx, ny, nz);
        }
        else {
            sevenPointStencil_single_iter_tiled_sliding_read<<<grid,block>>>(start, swap_out, nx, ny, nz);
        }
    }
    CUDASSERT(cudaDeviceSynchronize());

}*/
