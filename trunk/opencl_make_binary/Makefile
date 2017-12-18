all: vector_add kernel_compile

vector_add: vector_add.c
	gcc -o vector_add vector_add.c -std=c99 -Wall -lOpenCL

kernel_compile: compile.c
	gcc -o compile compile.c -std=c99 -Wall -lOpenCL

clean:
	rm vector_add compile

run: vector_add
	thorq --add --device gpu/7970 ./vector_add

compile: kernel_compile
	thorq --add --device gpu/7970 ./compile
