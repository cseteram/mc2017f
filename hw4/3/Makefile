all: 1 2 3

1: case_1.c
	mpicc -o case_1 case_1.c -std=c99 -Wall
2: case_2.c
	mpicc -o case_2 case_2.c -std=c99 -Wall
3: case_3.c
	mpicc -o case_3 case_3.c -std=c99 -Wall

clean:
	rm case_1 case_2 case_3

run:
	thorq --add --mode mpi --nodes 1 --slots 2 ./case_1
	thorq --add --mode mpi --nodes 1 --slots 2 ./case_2
	thorq --add --mode mpi --nodes 1 --slots 2 ./case_3
