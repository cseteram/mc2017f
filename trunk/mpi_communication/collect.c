#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#define T 8

typedef unsigned char uchar;
const int N = 312 * 312;
const int num_filters = 60000;
const int filter_size = 3 * 32 * 32;

double start_time[T];
double get_time();
void timer_start(int i);
double timer_stop(int i);

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uchar *img_t_all;
    uchar *dataset = (uchar*)malloc(sizeof(uchar) * num_filters * filter_size);
    if (rank == 0) {
        img_t_all = (uchar*)malloc(sizeof(uchar) * N * filter_size);
        for (int i = 0; i < N * filter_size; ++i) {
            img_t_all[i] = rand() % 256;
        }
        for (int i = 0; i < num_filters * filter_size; ++i) {
            dataset[i] = rand() % 256;
        }
    }

    if (rank == 0)
        timer_start(0);

    int *num_tiles_per_node, *num_tiles_offset;
    int *img_t_size, *img_t_offset;
    if (rank == 0) {
        num_tiles_per_node = (int*)malloc(sizeof(int) * size);
        num_tiles_offset = (int*)malloc(sizeof(int) * size);
        img_t_size = (int*)malloc(sizeof(int) * size);
        img_t_offset = (int*)malloc(sizeof(int) * size);
        for (int i = 0; i < size; ++i) {
            num_tiles_per_node[i] = N / size;
            if (i < N % size) ++num_tiles_per_node[i];
            num_tiles_offset[i] = (i > 0) ? (num_tiles_offset[i - 1] + num_tiles_per_node[i - 1]) : 0;

            img_t_size[i] = num_tiles_per_node[i] * filter_size;
            img_t_offset[i] = (i > 0) ? (img_t_offset[i - 1] + img_t_size[i - 1]) : 0;
        }
    }

    if (rank == 0) {
        printf("(N, num_filters, filter_size) = (%d, %d, %d)\n", N, num_filters, filter_size);
        printf("Number of processors = %d\n", size);
        printf("num_tiles_per_node = {");
        for (int i = 0; i < size; ++i) printf(" %d", num_tiles_per_node[i]);
        printf(" }\n");
        printf("num_tiles_offset = {");
        for (int i = 0; i < size; ++i) printf(" %d", num_tiles_offset[i]);
        printf(" }\n");
        printf("img_t_size = {");
        for (int i = 0; i < size; ++i) printf(" %d", img_t_size[i]);
        printf(" }\n");
        printf("img_t_offset = {");
        for (int i = 0; i < size; ++i) printf(" %d", img_t_offset[i]);
        printf(" }\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int num_tiles;
    if (rank == 0)
        timer_start(1);
    MPI_Scatter(num_tiles_per_node, 1, MPI_INT, &num_tiles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Scatter num_tiles : %f seconds\n", timer_stop(1));

    uchar *img_t = (uchar*)malloc(sizeof(uchar) * num_tiles * filter_size);
    if (rank == 0)
        timer_start(2);
    MPI_Scatterv(img_t_all, img_t_size, img_t_offset, MPI_UNSIGNED_CHAR, img_t, num_tiles * filter_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Scatterv img_t : %f seconds\n", timer_stop(2));
    if (rank == 0)
        timer_start(3);
    MPI_Bcast(dataset, num_filters * filter_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Bcast dataset : %f seconds\n", timer_stop(3));

    int *idx = (int*)malloc(sizeof(int) * N);
    int *sub_idx = (int*)malloc(sizeof(int) * num_tiles);
    for (int i = 0; i < num_tiles; ++i)
        sub_idx[i] = rank;

    if (rank == 0)
        timer_start(4);
    MPI_Gatherv(sub_idx, num_tiles, MPI_INT, idx, num_tiles_per_node, num_tiles_offset, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Gatherv idx : %f seconds\n", timer_stop(4));

    if (rank == 0) {
        printf("Elapsed time = %f seconds\n", timer_stop(0));
    }

    MPI_Finalize();
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) {
    start_time[i] = get_time();
}

double timer_stop(int i) {
    return get_time() - start_time[i];
}
