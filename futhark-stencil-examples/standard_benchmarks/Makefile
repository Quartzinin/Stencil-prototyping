
sources=gaussian2d_map_nests.fut\
  gaussian2d_stencil.fut\
  gradiant2d_map_nests.fut\
  gradiant2d_stencil.fut\
  heat3d_map_nests.fut\
  heat3d_stencil.fut\
  hotspot2d_map_nests.fut\
  hotspot2d_stencil.fut\
  hotspot3d_map_nests.fut\
  hotspot3d_stencil.fut\
  jacobi2d5p_map_nests.fut\
  jacobi2d5p_stencil.fut\
  jacobi2d9p_map_nests.fut\
  jacobi2d9p_stencil.fut\
  jacobi3d13p_map_nests.fut\
  jacobi3d13p_stencil.fut\
  jacobi3d7p_map_nests.fut\
  jacobi3d7p_stencil.fut\
  poission_map_nests.fut\
  poission_stencil.fut\
  srad_nest_maps.fut\
  srad_stencil.fut\
  tsize_3d7p_sum.fut
cudas=$(sources:.fut=.execuda)
opencls=$(sources:.fut=.exeopencl)

bases2d=gaussian2d gradiant2d hotspot2d jacobi2d5p jacobi2d9p
bases3d=heat3d hotspot3d jacobi3d13p jacobi3d7p poission
sizes2d=4095x4095 8191x8191
sizes3d=255x255x255 511x511x511
srad_sizes=2047x2047 4095x4095

runs=50

datasets:
	mkdir datasets
	futhark dataset -s 1337 -b -g '[2047][2047]u8' > datasets/2047x2047xu8.bin
	futhark dataset -s 1337 -b -g '[4095][4095]u8' > datasets/4095x4095xu8.bin
	futhark dataset -s 1337 -b -g '[4095][4095]f32' > datasets/4095x4095xf32.bin
	futhark dataset -s 1337 -b -g '[8191][8191]f32' > datasets/8191x8191xf32.bin
	futhark dataset -s 1337 -b -g '[255][255][255]f32' > datasets/255x255x255xf32.bin
	futhark dataset -s 1337 -b -g '[511][511][511]f32' > datasets/511x511x511xf32.bin
	futhark dataset -s 1337 -b -g '[255][255][255]f64' > datasets/255x255x255xf64.bin
	futhark dataset -s 1337 -b -g '[255][255][255]i8' > datasets/255x255x255xi8.bin

%.execuda: %.fut
	futhark cuda -o $@ $<
%.exeopencl: %.fut
	futhark opencl -o $@ $<

benchmarkCuda: $(cudas) datasets
	mkdir -p runtimesCUDA
	for g in ${srad_sizes}; do \
	    cat ./datasets/$${g}xu8.bin | ./srad_nest_maps.execuda -b -r ${runs} -t ./runtimesCUDA/srad_map_nests_$${g}xu8.txt > /dev/null; \
	    cat ./datasets/$${g}xu8.bin | ./srad_stencil.execuda -b -r ${runs} -t ./runtimesCUDA/srad_stencil_$${g}xu8.txt > /dev/null; \
	done;
	for f in ${bases2d}; do \
	    for g in ${sizes2d}; do \
	        cat ./datasets/$${g}xf32.bin | ./$${f}_map_nests.execuda -b -r ${runs} -t ./runtimesCUDA/$${f}_map_nests_$${g}xf32.txt > /dev/null; \
	        cat ./datasets/$${g}xf32.bin | ./$${f}_stencil.execuda -b -r ${runs} -t ./runtimesCUDA/$${f}_stencil_$${g}xf32.txt > /dev/null; \
	    done; \
	done;
	for f in ${bases3d}; do \
	    for g in ${sizes3d}; do \
	        cat ./datasets/$${g}xf32.bin | ./$${f}_map_nests.execuda -b -r ${runs} -t ./runtimesCUDA/$${f}_map_nests_$${g}xf32.txt > /dev/null; \
	        cat ./datasets/$${g}xf32.bin | ./$${f}_stencil.execuda -b -r ${runs} -t ./runtimesCUDA/$${f}_stencil_$${g}xf32.txt > /dev/null; \
	    done; \
	done;
	for g in ${sizes3d}; do \
	    cat ./datasets/$${g}xi8.bin | ./tsize_3d7p_sum.execuda --entry-point main_nest_maps_i8 -b -r ${runs} -t ./runtimesCUDA/tsize_map_nests_$${g}xi8.txt > /dev/null; \
	    cat ./datasets/$${g}xi8.bin | ./tsize_3d7p_sum.execuda --entry-point main_nest_maps_i8 -b -r ${runs} -t ./runtimesCUDA/tsize_stencil_$${g}xi8.txt > /dev/null; \
	    cat ./datasets/$${g}xf64.bin | ./tsize_3d7p_sum.execuda --entry-point main_nest_maps_f64 -b -r ${runs} -t ./runtimesCUDA/tsize_map_nests_$${g}xf64.txt > /dev/null; \
	    cat ./datasets/$${g}xf64.bin | ./tsize_3d7p_sum.execuda --entry-point main_nest_maps_f64 -b -r ${runs} -t ./runtimesCUDA/tsize_stencil_$${g}xf64.txt > /dev/null; \
	    break; \
	done;

benchmarkOpenCL: $(opencls) datasets
	mkdir -p runtimesOpenCL
	for g in ${srad_sizes}; do \
	    cat ./datasets/$${g}xu8.bin | ./srad_nest_maps.exeopencl -b -r ${runs} -t ./runtimesOpenCL/srad_map_nests_$${g}xu8.txt > /dev/null; \
	    cat ./datasets/$${g}xu8.bin | ./srad_stencil.exeopencl -b -r ${runs} -t ./runtimesOpenCL/srad_stencil_$${g}xu8.txt > /dev/null; \
	done;
	for f in ${bases2d}; do \
	    for g in ${sizes2d}; do \
	        cat ./datasets/$${g}xf32.bin | ./$${f}_map_nests.exeopencl -b -r ${runs} -t ./runtimesOpenCL/$${f}_map_nests_$${g}xf32.txt > /dev/null; \
	        cat ./datasets/$${g}xf32.bin | ./$${f}_stencil.exeopencl -b -r ${runs} -t ./runtimesOpenCL/$${f}_stencil_$${g}xf32.txt > /dev/null; \
	    done; \
	done;
	for f in ${bases3d}; do \
	    for g in ${sizes3d}; do \
	        cat ./datasets/$${g}xf32.bin | ./$${f}_map_nests.exeopencl -b -r ${runs} -t ./runtimesOpenCL/$${f}_map_nests_$${g}xf32.txt > /dev/null; \
	        cat ./datasets/$${g}xf32.bin | ./$${f}_stencil.exeopencl -b -r ${runs} -t ./runtimesOpenCL/$${f}_stencil_$${g}xf32.txt > /dev/null; \
	    done; \
	done;
	for g in ${sizes3d}; do \
	    cat ./datasets/$${g}xi8.bin | ./tsize_3d7p_sum.exeopencl --entry-point main_nest_maps_i8 -b -r ${runs} -t ./runtimesOpenCL/tsize_map_nests_$${g}xi8.txt > /dev/null; \
	    cat ./datasets/$${g}xi8.bin | ./tsize_3d7p_sum.exeopencl --entry-point main_nest_maps_i8 -b -r ${runs} -t ./runtimesOpenCL/tsize_stencil_$${g}xi8.txt > /dev/null; \
	    cat ./datasets/$${g}xf64.bin | ./tsize_3d7p_sum.exeopencl --entry-point main_nest_maps_f64 -b -r ${runs} -t ./runtimesOpenCL/tsize_map_nests_$${g}xf64.txt > /dev/null; \
	    cat ./datasets/$${g}xf64.bin | ./tsize_3d7p_sum.exeopencl --entry-point main_nest_maps_f64 -b -r ${runs} -t ./runtimesOpenCL/tsize_stencil_$${g}xf64.txt > /dev/null; \
	    break; \
	done;

clean:
	rm -f *.c *.exe *.bin *.execuda *.exeopencl
