
big_3d:
	futhark dataset --f32-bounds=1:10000 -g [511][255][127]f32 -b > big.3D

test_3d:
	cat $(data).3D | ./stencil_3d_benchmark --load-cuda 3Dcud -r 50 -t $(out).txt > /dev/null

test_original_3d:
	cat $(data).3D | ./stencil_3d_benchmark -r 50 -t original.txt > /dev/null

dump_3d:
	futhark cuda stencil_3d_benchmark.fut
	./stencil_3d_benchmark --dump-cuda 3Dcud

clean:
	rm -f 3Dcud *.txt stencil_3d_benchmark *.c
