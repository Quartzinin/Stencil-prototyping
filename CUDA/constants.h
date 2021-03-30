#ifndef CONSTANTS
#define CONSTANTS

#define T float

#define BLOCKSIZE 1024
#define SQ_BLOCKSIZE 32

#define BOUND(i,max_ix) (min((max_ix),max(0,(i))))
#define BOUNDL(i,max_ix) (min((max_ix),max(0l,(i))))
#define CEIL_DIV(x,d) (((x)+(d)-1)/(d))

//__constant__ int ixs_1d[BLOCKSIZE];
//__constant__ int2 ixs_2d[BLOCKSIZE];
//__constant__ int3 ixs_3d[BLOCKSIZE];

template<long D>
__device__
__forceinline__
T lambdaFun(const T* tmp){
    T acc = 0;
    #pragma unroll
    for (long j = 0; j < D; ++j)
    {
        acc += tmp[j];
    }
    acc /= T(D);
    return acc;
}

#define MACROLIKE __device__ __host__ __forceinline__

MACROLIKE long bound(long i, long max_i){ return min(max_i, max(0l, i)); }
MACROLIKE int bound(int i, int max_i){ return min(max_i, max(0, i)); }
MACROLIKE int divUp(int i, int d){ return (i + (d-1))/d; }
MACROLIKE long divUp(long i, long d){ return (i + (d-1))/d; }

MACROLIKE constexpr int3 create_spans(const int3 lens){ return { 1, lens.x, lens.x*lens.y }; }
MACROLIKE constexpr int2 create_spans(const int2 lens){ return { 1, lens.x, }; }
MACROLIKE constexpr int product(const int3 lens){ return lens.x * lens.y * lens.z; }
MACROLIKE constexpr int product(const int2 lens){ return lens.x * lens.y; }

#endif
