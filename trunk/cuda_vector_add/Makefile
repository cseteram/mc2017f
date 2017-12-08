CC=nvcc
TARGET=vec_add

all: $(TARGET)

$(TARGET): vec_add.cu
	nvcc vec_add.cu -o vec_add

clean:
	rm -rf $(TARGET)

run: $(TARGET)
	thorq --add --mode single --device gpu/1080 ./$(TARGET)
