#ifndef RUNNERS
#define RUNNERS

#include"constants.h"


#define GPU_RUN_INIT \
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

#define GPU_RUN_END \
    free(arr_in);\
    CUDASSERT(cudaFree(gpu_array_in));\

#define GPU_RUN(call,benchmark_name, preproc, destroy) {\
    CUDASSERT(cudaMemset(gpu_array_out, 0, mem_size));\
    CUDASSERT(cudaDeviceSynchronize());\
    cout << (benchmark_name); \
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
    printf(" : mean %lu microseconds\n", elapsed);\
    if (!validate(cpu_out,arr_out,len)) \
    { \
        printf("%s\n", "   FAILED TO VALIDATE");\
    }\
}
// const int n_reads_writes = ixs_len + 1;\
// const double GBperSec = len * sizeof(T) * n_reads_writes / elapsed / 1e3; \

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
        const T va = A[i];
        const T vb = B[i];
        if (fabs(va - vb) > 0.00001 || std::isnan(va) || std::isinf(va) || std::isnan(vb) || std::isinf(vb)){
                    printf("INVALID RESULT at index %d: (expected, actual) == (%f, %f)\n",
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

#endif
