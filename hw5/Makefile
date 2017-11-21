CC=mpicc
TARGET=mat_mul
OBJECTS=timer.o

CFLAGS=-std=c99 -O3 -Wall -fopenmp
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --mode single ./$(TARGET)

verify: $(TARGET)
	thorq --add --mode single ./$(TARGET) -v
