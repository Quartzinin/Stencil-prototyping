#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include "kernels.h"
#include "kernels-2d.h"
#include "kernels-3d.h"
using namespace std;

#include <iostream>
using std::cout;
using std::endl;

#define T int
#define CEIL_DIV(x,d) (((x)+(d)-1)/(d))

#define GPU_RUN(call,benchmark_name, preproc, destroy) {\
    struct timeval t_startpar, t_endpar, t_diffpar;\
    const int mem_size = len*sizeof(T); \
    T* arr_in  = (T*)malloc(mem_size*2); \
    T* arr_out = arr_in + len; \
    for(int i=0; i<len; i++){ arr_in[i] = (T)(i+1); } \
    T* gpu_array_in; \
    T* gpu_array_out; \
    CUDASSERT(cudaMalloc((void **) &gpu_array_in, 2*mem_size)); \
    gpu_array_out = gpu_array_in + len; \
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
    const int n_reads_writes = ixs_len + 1;\
    const double GBperSec = len * sizeof(T) * n_reads_writes / elapsed / 1e3; \
    printf("    mean elapsed time was : %lu microseconds\n", elapsed);\
    printf("    mean GigaBytes per sec: %lf GiBs\n", GBperSec);\
    if (validate(cpu_out,arr_out,len)) \
    { \
        printf("%s\n", "    VALIDATED");\
    }\
    free(arr_in);\
    CUDASSERT(cudaFree(gpu_array_in));\
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

void measure_memset_bandwidth(const int mem_size){
    struct timeval t_startpar, t_endpar, t_diffpar;
    T* gpu_array_in;
    T* gpu_array_out;
    CUDASSERT(cudaMalloc((void **) &gpu_array_in, 2*mem_size));
    gpu_array_out = gpu_array_in + (mem_size / sizeof(T)); \
    gettimeofday(&t_startpar, NULL);
    const unsigned RUNS = 100;
    for(unsigned x = 0; x < RUNS; x++){
        CUDASSERT(cudaMemcpy(gpu_array_out, gpu_array_in, mem_size, cudaMemcpyDeviceToDevice));
    }
    CUDASSERT(cudaDeviceSynchronize());
    gettimeofday(&t_endpar, NULL);\
    timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
    unsigned long elapsed = t_diffpar.tv_sec*1e6+t_diffpar.tv_usec;
    elapsed /= RUNS;
    const int n_reads_writes = 1 + 1;
    const double GBperSec = mem_size * n_reads_writes / elapsed / 1e3;
    printf("## Benchmark memcpy device to device ##\n");
    printf("    mean elapsed time was : %lu microseconds\n", elapsed);
    printf("    mean GigaBytes per sec: %lf GiBs\n", GBperSec);
    CUDASSERT(cudaFree(gpu_array_in));\
}

bool validate(const T* A, const T* B, unsigned int sizeAB){
    int c = 0;
    for(unsigned i = 0; i < sizeAB; i++){
        const int va = A[i];
        const int vb = B[i];
        if (va != vb){
            printf("INVALID RESULT at index %d: (expected, actual) == (%d, %d)\n",
                    i, va, vb);
            c++;
            if(c > 20)
                return false;
        }
    }
    return c == 0;
}

template<int D>
T stencil_fun_cpu(const T* arr)
{
    T sum_acc = 0;
    for (int i = 0; i < D; ++i){
        sum_acc += arr[i];
    }
    return sum_acc / (T)D;
}

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

template<int D>
void stencil_2d_cpu(
    const T* start,
    const int2* idxs,
    T* out,
    const int n_rows,
    const int n_columns)
{
    const int max_y_ix = n_rows - 1;
    const int max_x_ix = n_columns - 1;
    for (int i = 0; i < n_rows; ++i)
    {
        for (int k = 0; k < n_columns; ++k)
        {
            T arr[D];
            for (int j = 0; j < D; ++j)
            {
                int y = BOUND(i + idxs[j].y, max_y_ix);
                int x = BOUND(k + idxs[j].x, max_x_ix);
                int index = y * n_columns + x;
                arr[j] = start[index];
            }
            T lambda_res = stencil_fun_cpu<D>(arr);
            out[i * n_columns + k] = lambda_res;
        }
    }
}

template<int D>
void stencil_3d_cpu(
    const T* start,
    const int* idxs,
    T* out,
    const int x_len,
    const int y_len,
    const int z_len)
{
    const int max_x_idx = x_len - 1;
    const int max_y_idx = y_len - 1;
    const int max_z_idx = z_len - 1;
    for (int i = 0; i < x_len; ++i)
    {
        for (int j = 0; j < y_len; ++j)
        {
            for (int k = 0; k < z_len; ++k)
            {
                T arr[D];
                for (int p = 0; p < D; ++p)
                {
                    int x = BOUND(i + idxs[p*3  ], max_x_idx);
                    int y = BOUND(j + idxs[p*3+1], max_y_idx);
                    int z = BOUND(k + idxs[p*3+2], max_z_idx);

                    int index = x * y_len * z_len + y * z_len + z;
                    arr[p] = start[index];
                }

                T lambda_res = stencil_fun_cpu<D>(arr);
                out[i*y_len*z_len + j*z_len + k] = lambda_res;
            }
        }
    }
}

#define call_inSharedKernel(kernel) {\
    const int wasted = ix_min + ix_max;\
    const int working_block = BLOCKSIZE-wasted;\
    const int grid = (wasted + len + (working_block-1)) / working_block;\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

#define call_kernel(kernel) {\
    const int block = BLOCKSIZE;\
    const int grid = (len + (block-1)) / block;\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}
#define call_kernel_2d(kernel) {\
    const dim3 block(SQ_BLOCKSIZE,SQ_BLOCKSIZE,1);\
    const int BNx = CEIL_DIV(n_columns, SQ_BLOCKSIZE);\
    const int BNy = CEIL_DIV(n_rows, SQ_BLOCKSIZE);\
    const dim3 grid(BNx, BNy, 1);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}
#define call_kernel_3d(kernel) {\
    const int x_block = SQ_BLOCKSIZE; \
    const int y_block = x_block/4; \
    const int z_block = x_block/y_block; \
    const dim3 block(x_block,y_block,z_block);\
    const int BNx = CEIL_DIV(x_len, x_block);\
    const int BNy = CEIL_DIV(y_len, y_block);\
    const int BNz = CEIL_DIV(z_len, z_block);\
    const dim3 grid(BNx, BNy, BNz);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}
#define call_small_tile_2d(kernel) {\
    const dim3 block(SQ_BLOCKSIZE,SQ_BLOCKSIZE,1);\
    const int wasted_x = ix_min + ix_max;\
    const int wasted_y = ix_min + ix_max;\
    const int working_block_x = SQ_BLOCKSIZE-wasted_x;\
    const int working_block_y = SQ_BLOCKSIZE-wasted_y;\
    const int BNx = CEIL_DIV(n_columns, working_block_x);\
    const int BNy = CEIL_DIV(n_rows   , working_block_y);\
    const dim3 grid(BNx, BNy, 1);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

#define call_small_tile_3d(kernel) {\
    const int x_block = SQ_BLOCKSIZE; \
    const int y_block = x_block/4; \
    const int z_block = x_block/y_block; \
    const dim3 block(x_block,y_block,z_block);\
    const int working_block_z = z_block - (ix_min + ix_max);\
    const int working_block_y = y_block - (ix_min + ix_max);\
    const int working_block_x = x_block - (ix_min + ix_max);\
    const int BNx = CEIL_DIV(x_len, working_block_x);\
    const int BNy = CEIL_DIV(y_len, working_block_y);\
    const int BNz = CEIL_DIV(z_len, working_block_z);\
    const dim3 grid(BNx, BNy, BNz);\
    kernel;\
    CUDASSERT(cudaDeviceSynchronize());\
}

/*
template<int D>
void stencil_1d_global_temp(
    const T* start,
    const int* ixs,
    T* temp,
    T* out,
    const int len
    )
{
    const int grid1 = (len*D + (BLOCKSIZE-1)) / BLOCKSIZE;
    const int grid2 = (len + (BLOCKSIZE-1)) / BLOCKSIZE;

    global_temp__1d_to_temp<D><<<grid1,BLOCKSIZE>>>(start, ixs, temp, len);
    global_temp__1d<D><<<grid2,BLOCKSIZE>>>(temp, out, len);
    CUDASSERT(cudaDeviceSynchronize());
}
*/

template<int D>
T* run_cpu(const int* idxs, const int len)
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

template<int D>
T* run_cpu_2d(const int2* idxs, const int n_rows, const int n_columns)
{
    int len = n_rows*n_columns;
    T* cpu_in = (T*)malloc(len*sizeof(T));
    T* cpu_out = (T*)malloc(len*sizeof(T));

    for (int i = 0; i < len; ++i)
    {
        cpu_in[i] = (T)(i+1);
    }

    stencil_2d_cpu<D>(cpu_in,idxs,cpu_out,n_rows,n_columns);
    free(cpu_in);
    return cpu_out;
}

template<int D>
T* run_cpu_3d(const int* idxs, const int x_len, const int y_len, const int z_len)
{
    int len = x_len*y_len*z_len;
    T* cpu_in = (T*)malloc(len*sizeof(T));
    T* cpu_out = (T*)malloc(len*sizeof(T));

    for (int i = 0; i < len; ++i)
    {
        cpu_in[i] = (T)(i+1);
    }

    stencil_3d_cpu<D>(cpu_in,idxs,cpu_out,x_len,y_len,z_len);
    free(cpu_in);
    return cpu_out;
}

template<int ixs_len, int ix_min, int ix_max>
void doAllTest()
{
    const int RUNS = 100;

    struct timeval t_startpar, t_endpar, t_diffpar;

    const int D = ixs_len;
    const int ixs_size = D*sizeof(int);
    int* cpu_ixs = (int*)malloc(ixs_size);
    for(int i=0; i < D ; i++){ cpu_ixs[i] = i; }

    for(int i=0; i < D ; i++){
        const int V = cpu_ixs[i];
        if(-ix_min <= V && V <= ix_max)
        {}
        else { printf("index array contains indexes not in range\n"); exit(1); }
    }
    int* gpu_ixs;
    CUDASSERT(cudaMalloc((void **) &gpu_ixs, ixs_size));
    CUDASSERT(cudaMemcpy(gpu_ixs, cpu_ixs, ixs_size, cudaMemcpyHostToDevice));
    CUDASSERT(cudaMemcpyToSymbol(ixs, cpu_ixs, ixs_size));

    const int len = 2 << 20;
    T* cpu_out = run_cpu<D>(cpu_ixs,len);

    cout << "const int ixs[" << D << "] = [";
    for(int i=0; i < D ; i++){
        cout << " " << cpu_ixs[i];
        if(i == D-1)
        { cout << "]" << endl; }
        else{ cout << ", "; }
    }
    {
//        GPU_RUN(call_kernel(
//                    (big_tiled_1d<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d big-tiled ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (big_tiled_1d_const_ixs<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d big-tiled const ixs ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (big_tiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d big-tiled const inline ixs ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (inlinedIndexes_1d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d inlined idxs with global reads ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (inlinedIndexes_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d inlined idxs with global reads const ixs ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (threadLocalArr_1d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d local temp-array w/ global reads ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (threadLocalArr_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d local temp-array const ixs w/ global reads ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (outOfSharedtiled_1d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d out of shared tiled /w local temp-array ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (outOfSharedtiled_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d out of shared tiled const ixs /w local temp-array ##",(void)0,(void)0);
//        GPU_RUN(call_inSharedKernel(
//                    (inSharedtiled_1d<ixs_len,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d in shared tiled /w local temp-array ##",(void)0,(void)0);
//        GPU_RUN(call_inSharedKernel(
//                    (inSharedtiled_1d_const_ixs<ixs_len,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d in shared tiled const ixs /w local temp-array ##",(void)0,(void)0);
        GPU_RUN(call_inSharedKernel(
                    (inSharedtiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d in shared tiled const inline ixs ##",(void)0,(void)0);
        /*GPU_RUN((stencil_1d_global_temp<D, BLOCKSIZE>(gpu_array_in, gpu_ixs, temp, gpu_array_out, len)),
                "## Benchmark GPU 1d global temp ##"
                ,(CUDASSERT(cudaMalloc((void **) &temp, D*mem_size)))
                ,(cudaFree(temp)));*/
    }

    free(cpu_out);
    cudaFree(gpu_ixs);
    free(cpu_ixs);
}

template<int ixs_len, int ix_min, int ix_max>
void doTest()
{
    const int RUNS = 100;


    const int D = ixs_len;
    const int ixs_size = D*sizeof(int);
    int* cpu_ixs = (int*)malloc(ixs_size);
    for(int i=0; i < D ; i++){ cpu_ixs[i] = i; }

    for(int i=0; i < D ; i++){
        const int V = cpu_ixs[i];
        if(-ix_min <= V && V <= ix_max)
        {}
        else { printf("index array contains indexes not in range\n"); exit(1);}
    }
    CUDASSERT(cudaMemcpyToSymbol(ixs, cpu_ixs, ixs_size));

    const int len = 2 << 20;
    T* cpu_out = run_cpu<D>(cpu_ixs,len);

    cout << "const int ixs[" << D << "] \n";
    /*cout << "const int ixs[" << D << "] = [";
    for(int i=0; i < D ; i++){
        cout << " " << cpu_ixs[i];
        if(i == D-1)
        { cout << "]" << endl; }
        else{ cout << ", "; }
    }*/

    {
        GPU_RUN(call_kernel(
                    (big_tiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d big-tiled const inline ixs ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (inlinedIndexes_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d inlined idxs with global reads const ixs ##",(void)0,(void)0);
        GPU_RUN(call_inSharedKernel(
                    (inSharedtiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d in shared tiled const inline ixs ##",(void)0,(void)0);
    }

    free(cpu_out);
    free(cpu_ixs);
}

template<int ixs_len, int ix_min, int ix_max>
void doWideTest()
{
    const int RUNS = 1000;

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
    CUDASSERT(cudaMemcpyToSymbol(ixs, cpu_ixs, ixs_size));

    const int len = 2 << 20;
    T* cpu_out = run_cpu<D>(cpu_ixs,len);

    cout << "const int ixs[" << D << "]" << endl;
    /*
    cout << "const int ixs[" << D << "] = [";
    for(int i=0; i < D ; i++){
        cout << " " << cpu_ixs[i];
        if(i == D-1)
        { cout << "]" << endl; }
        else{ cout << ", "; }
    }
    */

    {
        /*
        GPU_RUN(call_kernel(
                    (inlinedIndexes_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d global reads ##",(void)0,(void)0);
        GPU_RUN(call_inSharedKernel(
                    (inSharedtiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d small tile ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (big_tiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark 1d big tile ##",(void)0,(void)0);
        */
        const int cap = 401;
        const int q_off = (cap - ixs_len)/2;
        if(3 <= ixs_len && ixs_len <= cap && ixs_len & 1 > 0){
            GPU_RUN(call_kernel(
                        (global_read_1d_const<ixs_len,q_off,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                    ,"## Benchmark 1d global reads constant ixs ##",(void)0,(void)0);
            GPU_RUN(call_inSharedKernel(
                        (small_tile_1d_const<ixs_len,q_off,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_array_out, len)))
                    ,"## Benchmark 1d small tile constant ixs  ##",(void)0,(void)0);
            GPU_RUN(call_kernel(
                        (big_tile_1d_const<ixs_len,q_off,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                    ,"## Benchmark 1d big tile constant ixs ##",(void)0,(void)0);
        }
    }

    free(cpu_out);
    free(cpu_ixs);
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

    const int n_rows = 2 << 12;
    const int n_columns = 2 << 8;
    const int len = n_rows * n_columns;
    cout << "{ row_len = " << n_columns << ", col_len = " << n_rows
         << ", total_len = " << len << " }" << endl;
    T* cpu_out = run_cpu_2d<ixs_len>(cpu_ixs,n_rows,n_columns);

    measure_memset_bandwidth(len * sizeof(T));

    {
        GPU_RUN(call_kernel_2d(
                    (global_reads_2d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, n_columns, n_rows)))
                ,"## Benchmark 2d global read ##",(void)0,(void)0);
        GPU_RUN(call_small_tile_2d(
                    (small_tile_2d<ixs_len,ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, n_columns, n_rows)))
                ,"## Benchmark 2d small tile ##",(void)0,(void)0);
        GPU_RUN(call_kernel_2d(
                    (big_tile_2d<ixs_len,ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, n_columns, n_rows)))
                ,"## Benchmark 2d big tile ##",(void)0,(void)0);
        if(ixs_len == 9){
            GPU_RUN(call_kernel_2d(
                        (global_reads_2d_9<<<grid,block>>>(gpu_array_in, gpu_array_out, n_columns, n_rows)))
                    ,"## Benchmark 2d global read constant ixs ##",(void)0,(void)0);
            GPU_RUN(call_small_tile_2d(
                        (small_tile_2d_9<<<grid,block>>>(gpu_array_in, gpu_array_out, n_columns, n_rows)))
                    ,"## Benchmark 2d small tile constant ixs ##",(void)0,(void)0);
            GPU_RUN(call_kernel_2d(
                        (big_tile_2d_9<<<grid,block>>>(gpu_array_in, gpu_array_out, n_columns, n_rows)))
                    ,"## Benchmark 2d big tile constant ixs ##",(void)0,(void)0);
        }
    }
}

template<int sq_ixs_len, int ix_min, int ix_max>
void doTest_3D()
{
    const int RUNS = 200;

    const int ixs_len = sq_ixs_len * sq_ixs_len * sq_ixs_len;
    const int ixs_size = ixs_len*sizeof(int)*3;
    int* cpu_ixs = (int*)malloc(ixs_size);
    {
        int q = 0;
        for(int i=0; i < sq_ixs_len; i++){
            for(int j=0; j < sq_ixs_len; j++){
                for(int k=0; k < sq_ixs_len; k++){
                    cpu_ixs[q++] = i-ix_min;
                    cpu_ixs[q++] = j-ix_min;
                    cpu_ixs[q++] = k-ix_min;
                }
            }
        }
    }
    CUDASSERT(cudaMemcpyToSymbol(ixs, cpu_ixs, ixs_size));

    cout << "const int ixs[" << ixs_len << "] = [";
    for(int i=0; i < ixs_len ; i++){
        cout << " (" << cpu_ixs[i*2] << "," << cpu_ixs[i*2+1] << "," << cpu_ixs[i*2+2] << ")";
        if(i == ixs_len-1)
        { cout << "]" << endl; }
        else{ cout << ", "; }
    }

    const int z_len = (2 << 7); //outermost
    const int y_len = 2 << 7; //middle
    const int x_len = 2 << 6; //innermost

    const int len = z_len * y_len * x_len;
    cout << "{ z = " << z_len << ", y = " << y_len << ", x = " << x_len << ", total_len = " << len << " }" << endl;

    T* cpu_out = run_cpu_3d<ixs_len>(cpu_ixs,z_len,y_len,x_len);

    measure_memset_bandwidth(len * sizeof(T));

    {
        GPU_RUN(call_kernel_3d(
                    (global_reads_3d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d global read ##",(void)0,(void)0);
        GPU_RUN(call_small_tile_3d(
                    (small_tile_3d<ixs_len,ix_min,ix_max,ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d small tile ##",(void)0,(void)0);
        GPU_RUN(call_kernel_3d(
                    (big_tile_3d<ixs_len,ix_min,ix_max,ix_min,ix_max,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, z_len, y_len, x_len)))
                ,"## Benchmark 3d big tile ##",(void)0,(void)0);

    }
}

int main()
{
    // tests all kernels
    //doAllTest<3,0,2>();

    // find limits for a small iota pattern stencil

    //doTest<1,0,0>();
    //doTest<2,0,1>();
    //doTest<3,1,2>();
    //doTest<4,0,3>();
    //doTest<5,0,4>();
    //doTest<11,0,10>();
    //doTest<801,0,800>();
    //doTest<980,0,979>();
    //doTest<985,0,984>();
    //doTest<990,0,989>();


    //Try with small length ixs, but with a large gap between indices.

    //doWideTest<3,256,256>();
    doWideTest<3,1,1>();
    doWideTest<5,2,2>();
    doWideTest<7,3,3>();
    doWideTest<9,4,4>();
    doWideTest<21,10,10>();
    doWideTest<23,11,11>();
    doWideTest<25,12,12>();
    doWideTest<27,13,13>();
    doWideTest<29,14,14>();
    doWideTest<31,15,15>();
    doWideTest<41,20,20>();

    doTest_2D<3,1,1>();
    doTest_2D<5,2,2>();
    //doTest_3D<3,1,1>();


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
