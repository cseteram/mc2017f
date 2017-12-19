CC=mpicc
TARGET=main
OBJECTS=photomosaic.o qdbmp.o timer.o

CFLAGS=-std=c99 -O3 -Wall -lOpenCL -fopenmp
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --mode mpi --nodes 4 --slots 1 --device gpu/7970 ./$(TARGET) $(INPUT) $(OUTPUT)
