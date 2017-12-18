all: collect

collect: collect.c
	mpicc -o collect collect.c -std=c99 -Wall

clean:
	rm collect

run:
	thorq --add --mode mpi --nodes 4 --slots 1 ./collect
