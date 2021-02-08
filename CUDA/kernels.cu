template <int T>
__global__
void sevenPointStencil(
      float * out,
      const float* A,
      const unsigned nx,
      const unsigned ny,
      const unsigned nz
)
{
	const float c0 = 1.0/6.0;
	const float c1 = 1.0/6.0/6.0;

	const unsigned lidz = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned lidy = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned total = ny*nz;

	__shared__ float TILE[3][T][T];

	//should current tile read from something 32 x 32, or write to something 32 x 32
	//write something 32x32 by adding extra reads.!!!

	// in this algorithm we do not write to the edges of the matrix
	if (threadIdx.y == 0 && lidy > 0)
	{
		TILE[0][0][threadIdx.x+1] = A[lidz+(lidy-1)*nz+0*total];
	}
	if (threadIdx.y == 1 && lidy < ny-1)
	{
		TILE[0][T+1][threadIdx.x+1] = A[lidz+(lidy+1)*nz+0*total];
	}

	if (threadIdx.x == 0 && lidz > 0)
	{
		TILE[0][threadIdx.y+1][0] = A[(lidz-1)+lidy*nz+0*total];
	}
	if (threadIdx.x == 1 && lidz < nx-1)
	{
		TILE[0][threadIdx.y+1][T+1] = A[(lidz+1)+lidy*nz+0*total];
	}

	TILE[0][threadIdx.y][threadIdx.x] = A[lidz+lidy*nz+0*total];
    TILE[1][threadIdx.y][threadIdx.x] = A[lidz+lidy*nz+1*total];

	__syncthreads();
	
	// for small y and z we should tune the schedule to utilize the parallelism of the x-axis
	for (int i = 1; i < nx-1; ++i)
	{
		TILE[(i+1) % 3][threadIdx.y][threadIdx.x] = A[lidz+lidy*nz+(i+1)*total];
		__syncthreads();

		if (lidy == 0 || lidy == ny-1 || lidz == 0 || lidz == nz-1)
		{
			out[lidz+lidy*nz+i*total] = A[lidz+lidy*nz+i*total];
		}
		//if (threadIdx.y == 0 && lidy > 0)

		//if (threadIdx.y == 1 && lidy < ny-1)
	}

}


 //z yyyy z
 //y xxxx y
 //y xxxx y
 //y xxxx y
 //y xxxx y
 //z yyyy z