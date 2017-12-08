CC=nvcc
TARGET=mat_mul

all: $(TARGET)

$(TARGET): mat_mul.cu timer.cpp
	$(CC) -o mat_mul mat_mul.cu timer.cpp

clean:
	rm -rf $(TARGET)

run: $(TARGET)
	thorq --add --device gpu/1080 ./$(TARGET)

verify: $(TARGET)
	thorq --add --device gpu/1080 ./$(TARGET) -v
