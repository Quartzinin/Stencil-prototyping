CXX        = nvcc -O3 

SRC = stencil.cu
SOURCES_CPP =stencil.cu
HELPERS     =kernels.h
OBJECTS     =stencil.o kernels.o
EXECUTABLE  =runproject

default: compile run

.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<

compile: $(EXECUTABLE)


$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE) 
	./$(EXECUTABLE)

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)
