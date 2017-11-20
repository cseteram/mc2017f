all: vector_add

vector_add: vector_add.c
	gcc -o vector_add vector_add.c -std=c99 -Wall -fopenmp

clean:
	rm vector_add

run:
	thorq --add --mode single ./vector_add
