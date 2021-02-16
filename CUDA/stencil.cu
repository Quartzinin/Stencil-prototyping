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

static void sevenPointStencil(
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

    breathFirst<W><<<grid,block>>>(start, out, len);
    CUDASSERT(cudaDeviceSynchronize());
}


int main()
{
    struct timeval t_startpar, t_endpar, t_diffpar;
    int RUNS = 100;

    const unsigned nx = 1000;
    const unsigned ny = 1000;
    const unsigned nz = 3;
    const unsigned iterations = 30; // must be an even number

    const unsigned total_size = nx*ny*nz;

    float* array3d_s = (float*)malloc(total_size*sizeof(float));
    float* array3d_ts = (float*)malloc(total_size*sizeof(float));
    float* array3d_tsr = (float*)malloc(total_size*sizeof(float));

    float* gpu_array3d;
    float* gpu_array3d_2;

    CUDASSERT(cudaMalloc((void **) &gpu_array3d, 2*nx*ny*nz*sizeof(float)));
    gpu_array3d_2 = &(gpu_array3d[nx*ny*nz]);
    const float et_godt_primtal = 7.0;

    CUDASSERT(cudaMemset(gpu_array3d, et_godt_primtal, nx*ny*nz*sizeof(float)));
    CUDASSERT(cudaMemset(gpu_array3d_2, et_godt_primtal, nx*ny*nz*sizeof(float)));

/*
    {
        CUDASSERT(cudaDeviceSynchronize());
        cout << "## Benchmark GPU stupid ##" << endl;

        gettimeofday(&t_startpar, NULL);

        for(unsigned x = 0; x < RUNS; x++){
            sevenPointStencil(gpu_array3d, gpu_array3d_2, nx, ny, nz, iterations);
        }

        gettimeofday(&t_endpar, NULL);
	CUDASSERT(cudaMemcpy(array3d_s, gpu_array3d_2, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost))

        timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
        unsigned long elapsed = t_diffpar.tv_sec*1e6+t_diffpar.tv_usec;
            elapsed /= RUNS;
        unsigned long el_sec = elapsed / 1000000;
        unsigned long el_mil_sec = (elapsed / 1000) % 1000;
        printf("    mean elapsed time was: %lu.%03lu seconds\n", el_sec, el_mil_sec);

    }

    {
        CUDASSERT(cudaDeviceSynchronize());
        cout << "## Benchmark GPU tiled sliding ##" << endl;

        gettimeofday(&t_startpar, NULL);

        for(unsigned x = 0; x < RUNS; x++){
            sevenPointStencil_tiledSliding(gpu_array3d, gpu_array3d_2, nx, ny, nz, iterations);
        }

        gettimeofday(&t_endpar, NULL);
	CUDASSERT(cudaMemcpy(array3d_ts, gpu_array3d_2, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost))

        timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
        unsigned long elapsed = t_diffpar.tv_sec*1e6+t_diffpar.tv_usec;
            elapsed /= RUNS;
        unsigned long el_sec = elapsed / 1000000;
        unsigned long el_mil_sec = (elapsed / 1000) % 1000;
        printf("    mean elapsed time was: %lu.%03lu seconds\n", el_sec, el_mil_sec);

    }
    {
        CUDASSERT(cudaDeviceSynchronize());
        cout << "## Benchmark GPU fully-tiled ##" << endl;

        gettimeofday(&t_startpar, NULL);

        for(unsigned x = 0; x < RUNS; x++){
            sevenPointStencil_tiledSliding_fully(gpu_array3d, gpu_array3d_2, nx, ny, nz, iterations);
        }

        gettimeofday(&t_endpar, NULL);
	CUDASSERT(cudaMemcpy(array3d_tsr, gpu_array3d_2, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost))

        timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
        unsigned long elapsed = t_diffpar.tv_sec*1e6+t_diffpar.tv_usec;
            elapsed /= RUNS;
        unsigned long el_sec = elapsed / 1000000;
        unsigned long el_mil_sec = (elapsed / 1000) % 1000;
        printf("    mean elapsed time was: %lu.%03lu seconds\n", el_sec, el_mil_sec);

    }
*/
    CUDASSERT(cudaDeviceSynchronize());
    {
        const int len = 10000000;
        //int* arr_in  = (int*)malloc(len*sizeof(int));
        int* arr_out = (int*)malloc(len*sizeof(int));
        const int a_number = 7;

        int* gpu_array_in;
        int* gpu_array_out;
        CUDASSERT(cudaMalloc((void **) &gpu_array_in, 2*len*sizeof(int)));
        gpu_array_out = &(gpu_array_in[len]);
        CUDASSERT(cudaMemset(gpu_array_in, a_number, 2*len*sizeof(float)));

        CUDASSERT(cudaDeviceSynchronize());
        cout << "## Benchmark GPU 1d global-mem ##" << endl;

        gettimeofday(&t_startpar, NULL);

        for(unsigned x = 0; x < RUNS; x++){
            stencil_1d_global_read<10000>(gpu_array_in, gpu_array_out, len);
        }

        CUDASSERT(cudaDeviceSynchronize());
        gettimeofday(&t_endpar, NULL);
	CUDASSERT(cudaMemcpy(arr_out, gpu_array_out, len*sizeof(float), cudaMemcpyDeviceToHost))

        timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
        unsigned long elapsed = t_diffpar.tv_sec*1e6+t_diffpar.tv_usec;
        elapsed /= RUNS;
        printf("    mean elapsed time was: %lu microseconds\n", elapsed);

        free(arr_out);
        cudaFree(gpu_array_in);
        cudaFree(gpu_array_out);
    }


    free(array3d_s);
    free(array3d_ts);
    free(array3d_tsr);

    cudaFree(gpu_array3d);
    cudaFree(gpu_array3d_2);

    return 0;
}
