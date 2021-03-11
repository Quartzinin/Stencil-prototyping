#ifndef CONSTANTS
#define CONSTANTS

#define T float

#define BLOCKSIZE 1024
#define SQ_BLOCKSIZE 32

//3d blocksizes
#define X_BLOCK 32
#define Y_BLOCK 8
#define Z_BLOCK 4

#define BOUND(i,max_ix) (min((max_ix),max(0,(i))))
#define SUBBOUND(i,min_ix,max_ix) (min((max_ix),max((min_ix),(i))-(min_ix)))
#define CEIL_DIV(x,d) (((x)+(d)-1)/(d))

__constant__ int ixs_1d[BLOCKSIZE];
__constant__ int2 ixs_2d[BLOCKSIZE];
__constant__ int3 ixs_3d[BLOCKSIZE];

#endif
