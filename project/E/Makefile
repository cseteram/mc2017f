CC=mpicc
TARGET=main
OBJECTS=photomosaic.o qdbmp.o timer.o

CFLAGS=-std=c99 -O3 -Wall -L$(SNUCLROOT)/lib -lsnucl_cluster -fopenmp
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --mode snucl --nodes 4 --device gpu/7970 ./$(TARGET) $(INPUT) $(OUTPUT)
