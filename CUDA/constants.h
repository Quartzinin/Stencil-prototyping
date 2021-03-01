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

__constant__ int ixs_1d[BLOCKSIZE];
__constant__ int2 ixs_2d[BLOCKSIZE];
__constant__ int3 ixs_3d[BLOCKSIZE];

#endif
