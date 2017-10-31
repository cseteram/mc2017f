#include <stdio.h>
#include <mpi.h>

int main()
{
    int size, rank;

    // Initialize MPI
    MPI_Init(NULL, NULL);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // print "Hello, World!"
    printf("Hello world from rank %d of %d processors\n", rank, size);

    // Finalize MPI
    MPI_Finalize();
}
