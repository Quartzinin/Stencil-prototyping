CXX        = nvcc -O3 

SRC = stencil.cu kernels.cu
OBJECTS     =stencil.o kernels.o
EXECUTABLE  =runproject

default: compile run

.o: $(SRC)
	$(CXX) -c $@ $<

compile: $(EXECUTABLE)


$(EXECUTABLE): $(SRC) 
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE) 
	./$(EXECUTABLE)

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)
