#include <stdio.h>
#include <mpi.h>

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2) {
        printf("Not enough processors\n");
        return 1;
    }

    int num1 = 123456789, num2 = 987654321;
    if (rank == 0) num1 = 0;
    if (rank == 1) num2 = 1;

    MPI_Status status;
    if (rank == 0) {
        MPI_Recv(&num2, 1, MPI_INT, 1, 1001, MPI_COMM_WORLD, &status);
        MPI_Send(&num1, 1, MPI_INT, 1, 1001, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        MPI_Recv(&num1, 1, MPI_INT, 0, 1001, MPI_COMM_WORLD, &status);
        MPI_Send(&num2, 1, MPI_INT, 0, 1001, MPI_COMM_WORLD);
    }

    printf("Processor %d says hello\n", rank);
    
    MPI_Finalize();
    return 0;
}
