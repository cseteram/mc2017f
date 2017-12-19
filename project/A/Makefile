TARGET=main
OBJECTS=photomosaic.o qdbmp.o timer.o

CFLAGS=-std=c99 -O3 -Wall -mavx -lpthread -fopenmp
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add ./$(TARGET) $(INPUT) $(OUTPUT)
