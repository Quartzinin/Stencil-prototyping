
Bases="`ls | grep '\.fut$$' | sed 's/\.fut$$//'`"

default: benchCuda

benchOpencl:
	futhark bench --backend=opencl -r 30 ./*.fut > benchmark_opencl_.txt
benchCuda:
	futhark bench --backend=cuda   -r 30 ./*.fut > benchmark_cuda_.txt

clean:
	rm -fr ./data `ls | grep '.fut' | sed "s/.fut$$/.c/"` `ls | grep '.fut' | sed "s/.fut$$//"`
