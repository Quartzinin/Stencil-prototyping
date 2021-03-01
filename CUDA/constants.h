#ifndef CONSTANTS
#define CONSTANTS

#define BLOCKSIZE 1024
#define SQ_BLOCKSIZE 32
#define BOUND(i,max_ix) (min((max_ix),max(0,(i))))

__constant__ int ixs_1d[BLOCKSIZE];
__constant__ int2 ixs_2d[BLOCKSIZE];
__constant__ int3 ixs_3d[BLOCKSIZE];

#endif
