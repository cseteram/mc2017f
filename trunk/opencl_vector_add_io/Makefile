all: vector_add

vector_add: vector_add.c
	gcc -o vector_add vector_add.c -std=c99 -Wall -lOpenCL

clean:
	rm vector_add

run:
	thorq --add --mode single --device gpu/7970 ./vector_add
