all: hello

hello: hello.c
	mpicc -o hello hello.c -std=c99 -Wall

clean:
	rm hello

run:
	thorq --add --mode mpi --nodes 1 --slots 8 ./hello
