TARGET=mat_mul
OBJECTS=timer.o

CFLAGS=-std=c99 -O3 -Wall -lpthread -mavx -g -funroll-loops
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add ./$(TARGET)
