CXX        = nvcc -O3 -arch=compute_35 -D_FORCE_INLINES -Wno-deprecated-gpu-targets -std=c++11 -maxrregcount=63

SRC = stencil-1d.cu stencil-2d.cu stencil-2d.cu
OBJECTS     =stencil-1d.o stencil-2d.o stencil-3d.o
EXECUTABLES  =runproject-1d runproject-2d runproject-3d

default: compile run1d run2d run3d

.o: $(SRC)
	$(CXX) -c $@ $<

compile: $(EXECUTABLES)

runproject-1d: stencil-1d.cu kernels-1d.h constants.h runners.h Makefile
	$(CXX) -o runproject-1d stencil-1d.cu
runproject-2d: stencil-2d.cu kernels-2d.h constants.h runners.h Makefile
	$(CXX) -o runproject-2d stencil-2d.cu
runproject-3d: stencil-3d.cu kernels-3d.h constants.h runners.h Makefile
	$(CXX) -o runproject-3d stencil-3d.cu

#$(EXECUTABLE): $(SRC) kernels-1d.h kernels-2d.h kernels-3d.h constants.h
#	$(CXX) -o $(EXECUTABLE) $(SRC)

run1d: runproject-1d
	./runproject-1d
run2d: runproject-2d
	./runproject-2d
run3d: runproject-3d
	./runproject-3d

runall: runproject-1d runproject-2d runproject-3d
	./runproject-1d
	./runproject-2d
	./runproject-3d

clean:
	rm -f Debug.txt $(EXECUTABLES) $(OBJECTS)
