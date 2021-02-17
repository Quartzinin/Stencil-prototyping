CXX        = nvcc -O3 -arch=compute_35 -D_FORCE_INLINES -Wno-deprecated-gpu-targets

SRC = stencil.cu kernels.cu
OBJECTS     =stencil.o kernels.o
EXECUTABLE  =runproject

default: compile run

.o: $(SRC)
	$(CXX) -c $@ $<

compile: $(EXECUTABLE)


$(EXECUTABLE): $(SRC) kernels.h
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE) 
	./$(EXECUTABLE)

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)
