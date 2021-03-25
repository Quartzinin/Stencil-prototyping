#ifndef RUNNERS
#define RUNNERS

#include"constants.h"

#include <nvfunctional>


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
    CUDASSERT(cudaPeekAtLastError());\
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
inline __host__
T stencil_fun_cpu(const T* tmp)
{
#if 1
    T acc = 0;
    #pragma unroll
    for (int j = 0; j < D; ++j)
    {
        acc += tmp[j];
    }
    acc /= (T(D));
#else
    T acc = 0;
    #pragma unroll
    for (int j = 0; j < D; ++j)
    {
        const T v = tmp[j];
        acc -= v;
        acc /= v;
    }
    #pragma unroll
    for (int j = 0; j < D; ++j)
    {
        const T v = tmp[j];
        acc -= v;
        acc /= v;
    }
#endif
    return acc;
}


using Kernel3dVirtual = void (*)(const T*, T*, const long3, const int, const int, const int3, const int3, const int3, const int);
using Kernel3dPhysMultiDim = void(*)(const T*, T*, const long3);
using Kernel3dPhysSingleDim = void(*)(const T*, T*, const long3, const int3);
template<typename L>
class Globs {
    public :
        struct timeval start_stamp, end_stamp;
        long RUNS;
        long mem_size;
        long tlen;
        L lens;
        T* arr_out;
        T* gpu_array_out;
        T* arr_in;
        T* gpu_array_in;

        __host__
        Globs(L arrlens, const long totallen, const long runsv){
            lens = arrlens;
            tlen = totallen;
            RUNS = runsv;
            mem_size = tlen*sizeof(T);
            const long out_start = 2*tlen;
            const long alloc_sizes = mem_size*3;
            arr_in = (T*)malloc(alloc_sizes);
            for(int i=0; i<tlen; i++){ arr_in[i] = (T)(i+1); }
            CUDASSERT(cudaMalloc((void **) &gpu_array_in, alloc_sizes));
            arr_out = &arr_in[out_start];
            gpu_array_out = gpu_array_in + out_start;
            CUDASSERT(cudaMemcpy(gpu_array_in, arr_in, mem_size, cudaMemcpyHostToDevice));
            CUDASSERT(cudaMemset(gpu_array_out, 0, mem_size));
            CUDASSERT(cudaDeviceSynchronize());
        }
        __host__
        ~Globs(void){
            free(arr_in);
            CUDASSERT(cudaFree(gpu_array_in));
        }
        __host__
        void reset_output(){
            CUDASSERT(cudaMemset(gpu_array_out, 0, mem_size));
            CUDASSERT(cudaDeviceSynchronize());
        }

        __host__
        inline
        void startTimer(){
            gettimeofday(&start_stamp, NULL);
        }
        __host__
        inline
        long endTimer(){
            gettimeofday(&end_stamp, NULL);
            struct timeval time_diff;
            timeval_subtract(&time_diff, &end_stamp, &start_stamp);
            return time_diff.tv_sec*1e6+time_diff.tv_usec;
        }

        __host__
        void check_output(const T* cpu_out, const bool should_print, const long elapsed){
            CUDASSERT(cudaMemcpy(arr_out, gpu_array_out, mem_size, cudaMemcpyDeviceToHost));
            CUDASSERT(cudaDeviceSynchronize());
            const long average_elapsed = elapsed / RUNS;
            if(should_print){
                printf(" : mean %ld microseconds\n", average_elapsed);
                if (!validate(cpu_out,arr_out,tlen)){
                    printf("%s\n", "   FAILED TO VALIDATE");
                }
            }

        }
        __host__
        void do_run_cube(
                Kernel3dPhysMultiDim call
                , const T* cpu_out
                , const dim3 grid
                , const dim3 block
                , const bool should_print=true){
            reset_output();
            long time_acc = 0;
            for(unsigned x = 0; x < RUNS; x++){
                startTimer();
                call<<<grid,block>>>(gpu_array_in, gpu_array_out, lens);
                CUDASSERT(cudaGetLastError()); // check cuda for errors
                CUDASSERT(cudaDeviceSynchronize());
                time_acc += endTimer();
            }
            check_output(cpu_out, should_print, time_acc);
        };
        __host__
        void do_run_singleDim(
                Kernel3dPhysSingleDim call
                , const T* cpu_out
                , const int grid_flat
                , const int block_flat
                , const int3 spans
                , bool should_print=true){
            reset_output();
            long time_acc = 0;
            for(unsigned x = 0; x < RUNS; x++){
                startTimer();
                call<<<grid_flat,block_flat>>>(gpu_array_in, gpu_array_out, lens, spans);
                CUDASSERT(cudaGetLastError()); // check cuda for errors
                CUDASSERT(cudaDeviceSynchronize());
                time_acc += endTimer();
            }
            check_output(cpu_out, should_print, time_acc);
        };

        __host__
        void do_run_virtual_MultiDim(
                Kernel3dVirtual call
                , const T* cpu_out
                , const int num_phys_groups
                , const dim3 block
                , const int3 virtual_grid
                , bool should_print=true){
            reset_output();
            long time_acc = 0;
            for(unsigned x = 0; x < RUNS; x++){
                startTimer();
                const int3 virtual_grid_spans = { 1, virtual_grid.x, virtual_grid.x * virtual_grid.y };
                const int virtual_grid_flat_length = virtual_grid.x * virtual_grid.y * virtual_grid.z;

                const int iters_per_phys = CEIL_DIV(virtual_grid_flat_length, num_phys_groups);

                const int add_z = num_phys_groups / virtual_grid_spans.z;
                const int radd  = num_phys_groups % virtual_grid_spans.z;
                const int add_y = radd / virtual_grid_spans.y;
                const int add_x = radd % virtual_grid_spans.y;
                const int3 id_add = { add_x, add_y, add_z };

                call<<<num_phys_groups,block>>>(
                        gpu_array_in, gpu_array_out,
                        lens, num_phys_groups,
                        iters_per_phys, id_add, virtual_grid,
                        virtual_grid_spans, virtual_grid_flat_length);
                CUDASSERT(cudaGetLastError()); // check cuda for errors
                CUDASSERT(cudaDeviceSynchronize());
                time_acc += endTimer();
            }
            check_output(cpu_out, should_print, time_acc);
        };
        __host__
        void do_run_virtual_singleDim(
                Kernel3dVirtual call
                , const T* cpu_out
                , const int num_phys_groups
                , const int block_flat
                , const int3 virtual_grid
                , bool should_print=true){
            reset_output();
            long time_acc = 0;
            for(unsigned x = 0; x < RUNS; x++){
                startTimer();
                const int3 virtual_grid_spans = { 1, virtual_grid.x, virtual_grid.x * virtual_grid.y };
                const int virtual_grid_flat_length = virtual_grid.x * virtual_grid.y * virtual_grid.z;

                const int iters_per_phys = CEIL_DIV(virtual_grid_flat_length, num_phys_groups);

                const int add_z = num_phys_groups / virtual_grid_spans.z;
                const int radd  = num_phys_groups % virtual_grid_spans.z;
                const int add_y = radd / virtual_grid_spans.y;
                const int add_x = radd % virtual_grid_spans.y;
                const int3 id_add = { add_x, add_y, add_z };

                call<<<num_phys_groups,block_flat>>>(
                        gpu_array_in, gpu_array_out,
                        lens, num_phys_groups,
                        iters_per_phys, id_add, virtual_grid,
                        virtual_grid_spans, virtual_grid_flat_length);
                CUDASSERT(cudaGetLastError()); // check cuda for errors
                CUDASSERT(cudaDeviceSynchronize());
                time_acc += endTimer();
            }
            check_output(cpu_out, should_print, time_acc);
        };
};

#endif
