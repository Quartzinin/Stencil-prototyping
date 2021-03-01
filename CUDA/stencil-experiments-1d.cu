template<int D>
void stencil_1d_global_temp(
    const T* start,
    const int* ixs,
    T* temp,
    T* out,
    const int len
    )
{
    const int grid1 = (len*D + (BLOCKSIZE-1)) / BLOCKSIZE;
    const int grid2 = (len + (BLOCKSIZE-1)) / BLOCKSIZE;

    global_temp__1d_to_temp<D><<<grid1,BLOCKSIZE>>>(start, ixs, temp, len);
    global_temp__1d<D><<<grid2,BLOCKSIZE>>>(temp, out, len);
    CUDASSERT(cudaDeviceSynchronize());
}

template<int ixs_len, int ix_min, int ix_max>
void doAllTest()
{
    const int RUNS = 100;

    struct timeval t_startpar, t_endpar, t_diffpar;

    const int D = ixs_len;
    const int ixs_size = D*sizeof(int);
    int* cpu_ixs = (int*)malloc(ixs_size);
    for(int i=0; i < D ; i++){ cpu_ixs[i] = i; }

    for(int i=0; i < D ; i++){
        const int V = cpu_ixs[i];
        if(-ix_min <= V && V <= ix_max)
        {}
        else { printf("index array contains indexes not in range\n"); exit(1); }
    }
    int* gpu_ixs;
    CUDASSERT(cudaMalloc((void **) &gpu_ixs, ixs_size));
    CUDASSERT(cudaMemcpy(gpu_ixs, cpu_ixs, ixs_size, cudaMemcpyHostToDevice));
    CUDASSERT(cudaMemcpyToSymbol(ixs_1d, cpu_ixs, ixs_size));

    const int len = 2 << 20;
    T* cpu_out = run_cpu<D>(cpu_ixs,len);

    cout << "const int ixs[" << D << "] = [";
    for(int i=0; i < D ; i++){
        cout << " " << cpu_ixs[i];
        if(i == D-1)
        { cout << "]" << endl; }
        else{ cout << ", "; }
    }
    {
//        GPU_RUN(call_kernel(
//                    (big_tiled_1d<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d big-tiled ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (big_tiled_1d_const_ixs<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d big-tiled const ixs ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (big_tiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d big-tiled const inline ixs ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (inlinedIndexes_1d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d inlined idxs with global reads ##",(void)0,(void)0);
        GPU_RUN(call_kernel(
                    (inlinedIndexes_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d inlined idxs with global reads const ixs ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (threadLocalArr_1d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d local temp-array w/ global reads ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (threadLocalArr_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d local temp-array const ixs w/ global reads ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (outOfSharedtiled_1d<ixs_len><<<grid,block>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d out of shared tiled /w local temp-array ##",(void)0,(void)0);
//        GPU_RUN(call_kernel(
//                    (outOfSharedtiled_1d_const_ixs<ixs_len><<<grid,block>>>(gpu_array_in, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d out of shared tiled const ixs /w local temp-array ##",(void)0,(void)0);
//        GPU_RUN(call_inSharedKernel(
//                    (inSharedtiled_1d<ixs_len,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_ixs, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d in shared tiled /w local temp-array ##",(void)0,(void)0);
//        GPU_RUN(call_inSharedKernel(
//                    (inSharedtiled_1d_const_ixs<ixs_len,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_array_out, len)))
//                ,"## Benchmark GPU 1d in shared tiled const ixs /w local temp-array ##",(void)0,(void)0);
        GPU_RUN(call_inSharedKernel(
                    (inSharedtiled_1d_const_ixs_inline<ixs_len,ix_min,ix_max><<<grid,BLOCKSIZE>>>(gpu_array_in, gpu_array_out, len)))
                ,"## Benchmark GPU 1d in shared tiled const inline ixs ##",(void)0,(void)0);
        /*GPU_RUN((stencil_1d_global_temp<D, BLOCKSIZE>(gpu_array_in, gpu_ixs, temp, gpu_array_out, len)),
                "## Benchmark GPU 1d global temp ##"
                ,(CUDASSERT(cudaMalloc((void **) &temp, D*mem_size)))
                ,(cudaFree(temp)));*/
    }

    free(cpu_out);
    cudaFree(gpu_ixs);
    free(cpu_ixs);
}
