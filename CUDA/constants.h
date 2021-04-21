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


#define MACROLIKE __device__ __host__ __forceinline__

template<typename L> MACROLIKE constexpr L bound(L i, L max_i){ return min(max_i, max(L(0), i)); }
template<typename L> MACROLIKE constexpr L divUp(L i, L d){ return (i + (d- (L(1))))/d; }

MACROLIKE constexpr int3 create_spans(const int3 lens){ return { 1, lens.x, lens.x*lens.y }; }
MACROLIKE constexpr int2 create_spans(const int2 lens){ return { 1, lens.x, }; }
MACROLIKE constexpr int product(const int3 lens){ return lens.x * lens.y * lens.z; }
MACROLIKE constexpr int product(const int2 lens){ return lens.x * lens.y; }

#endif
