#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
using namespace std;

#include <iostream>
using std::cout;
using std::endl;

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


inline void cudAssert(cudaError_t exit_code,
        const char *file,
        int         line) {
    if (exit_code != cudaSuccess) {
        fprintf(stderr, ">>> Cuda run-time error: %s, at %s:%d\n",
                cudaGetErrorString(exit_code), file, line);
        exit(exit_code);
    }
}

#define CUDASSERT(exit_code) { cudAssert((exit_code), __FILE__, __LINE__); }

int main()
{
	struct timeval t_startpar, t_endpar, t_diffpar;
	int RUNS = 10;

	unsigned int x = 1000;
	unsigned int y = 1000;
	unsigned int z = 1000;

	float* array1d = (float*)malloc(x*sizeof(float));
	float* array2d = (float*)malloc(x*y*sizeof(float));
	float* array3d = (float*)malloc(x*y*z*sizeof(float));

	float* gpu_array1d;
	float* gpu_array2d;
	float* gpu_array3d;

	CUDASSERT(cudaMalloc((void **) &gpu_array1d, x*sizeof(float)));
	CUDASSERT(cudaMalloc((void **) &gpu_array2d, x*y*sizeof(float)));
	CUDASSERT(cudaMalloc((void **) &gpu_array1d, x*y*z*sizeof(float)));

	int et_godt_primtal = 7;

	CUDASSERT(cudaMemset(gpu_array1d, et_godt_primtal, x*sizeof(float)));
	CUDASSERT(cudaMemset(gpu_array2d, et_godt_primtal, x*y*sizeof(float)));
	CUDASSERT(cudaMemset(gpu_array3d, et_godt_primtal, x*y*z*sizeof(float)));

	{
		cout << "## Benchmark GPU ##" << endl;

        gettimeofday(&t_startpar, NULL);

        for(unsigned x = 0; x < RUNS; x++){

        }

        gettimeofday(&t_endpar, NULL);
        timeval_subtract(&t_diffpar, &t_endpar, &t_startpar);
        unsigned long elapsed = t_diffpar.tv_sec*1e6+t_diffpar.tv_usec;
            elapsed /= RUNS;
        unsigned long el_sec = elapsed / 1000000;
        unsigned long el_mil_sec = (elapsed / 1000) % 1000;
        printf("    mean elapsed time was: %lu.%03lu seconds\n", el_sec, el_mil_sec);

	}
	return 1;
}
