#ifndef CONSTANTS
#define CONSTANTS

#define T float

#define BLOCKSIZE 1024
#define SQ_BLOCKSIZE 32

#define MACROLIKE __device__ __host__ __forceinline__

template<bool lowBound,typename L> MACROLIKE constexpr
L bound(const L i, const L max_i){ return min(max_i, lowBound ? max(L(0), i) : i); }
template<typename L> MACROLIKE constexpr L divUp(L i, L d){ return (i + (d- (L(1))))/d; }

MACROLIKE constexpr int3 create_spans(const int3 lens){ return { 1, lens.x, lens.x*lens.y }; }
MACROLIKE constexpr int2 create_spans(const int2 lens){ return { 1, lens.x, }; }
MACROLIKE constexpr int product(const int3 lens){ return lens.x * lens.y * lens.z; }
MACROLIKE constexpr int product(const int2 lens){ return lens.x * lens.y; }

#endif
