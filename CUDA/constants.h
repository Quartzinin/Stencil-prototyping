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

#endif
