TARGET=mat_mul
OBJECTS=timer.o

CFLAGS=-std=c99 -O3 -Wall -lpthread -lOpenCL
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --mode single --device gpu/7970 ./$(TARGET)

verify: $(TARGET)
	thorq --add --mode single --device gpu/7970 ./$(TARGET) -v
