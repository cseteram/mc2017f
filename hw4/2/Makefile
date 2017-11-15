CC=mpicc
TARGET=mat_mul
OBJECTS=timer.o

CFLAGS=-std=c99 -O3 -Wall -lpthread -funroll-loops
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --mode mpi --nodes 1 --slots $(SLOTS) ./$(TARGET)

verify: $(TARGET)
	thorq --add --mode mpi --nodes 1 --slots $(SLOTS) ./$(TARGET) -v
