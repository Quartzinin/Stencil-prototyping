#ifndef CONSTANTS
#define CONSTANTS

#define BLOCKSIZE 1024
#define SQ_BLOCKSIZE 32
#define BOUND(i,max_ix) (min((max_ix),max(0,(i))))

__constant__ int ixs[BLOCKSIZE*3];

#endif
