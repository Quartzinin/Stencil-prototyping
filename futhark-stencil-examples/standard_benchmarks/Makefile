
Bases="`ls | grep '\.fut$$' | sed 's/\.fut$$//'`"

default: benchCuda

benchOpencl:
	futhark bench --backend=opencl -r 30 -p --default-group-size=256  ./*.fut > benchmark_opencl_256_.txt
benchCuda256:
	futhark bench --backend=cuda   -r 30 -p --default-group-size=256  ./*.fut > benchmark_cuda_256_.txt
benchCuda1024:
	futhark bench --backend=cuda   -r 30 -p --default-group-size=1024 ./*.fut > benchmark_cuda_1024_.txt

clean:
	rm -fr ./data `ls | grep '.fut' | sed "s/.fut$$/.c/"` `ls | grep '.fut' | sed "s/.fut$$//"`
