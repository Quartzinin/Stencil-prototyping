CXX        = nvcc -arch=compute_35
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL
CXXFLAGS   = -O3 -DWITH_FLOATS=0 -D_FORCE_INLINES -Xcompiler -fopenmp

INCLUDES    += -I ../include
GPU_OPTS   = -D lgWARP=5

SOURCES_CPP =stencil.cpp
HELPERS     =/include/Constants.h /include/ParseInput.h /include/ParserC.h /include/OpenmpUtil.h kernels.h
OBJECTS     =stencil.o kernels.o
EXECUTABLE  =runproject
SOURCES_CU  =kernels.h

default: cpu

kernels.o: $(SOURCES_CU)
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) kernels.h


.cpp.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<


cpu: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXECUTABLE) $(OBJECTS)


run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt


clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)
