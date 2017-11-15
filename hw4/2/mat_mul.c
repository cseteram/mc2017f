#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "timer.h"

#include <mpi.h>
#include <string.h>

#define N 2048
#define TSIZE 16
#define NUM_TILES (N / TSIZE)

// #define TILING
// #define NON_BLOCKING

bool print_matrix = false;
bool validation = false;

float a[N][N], b[N][N], c[N][N];

void mat_mul() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int id = rank;
    int elements_per_proc = (N / size) * N;

#ifdef NON_BLOCKING
    MPI_Request request_a, request_b;
    MPI_Status status_a, status_b;
    
    MPI_Iscatter(
        (float*)a, elements_per_proc, MPI_FLOAT,
        (float*)a + rank * elements_per_proc, elements_per_proc, MPI_FLOAT,
        0, MPI_COMM_WORLD, &request_a
    );
    MPI_Ibcast((float*)b, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &request_b);
    
    MPI_Wait(&request_a, &status_a);
    MPI_Wait(&request_b, &status_b);
#else
    MPI_Scatter(
        (float*)a, elements_per_proc, MPI_FLOAT,
        (float*)a + rank * elements_per_proc, elements_per_proc, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );
    MPI_Bcast((float*)b, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif

#ifdef TILING
    int ii_chunk = N / TSIZE / size;
    float asub[TSIZE][TSIZE], bsub[TSIZE][TSIZE], csub[TSIZE][TSIZE];
    
    for (int ii = id * ii_chunk; ii < (id + 1) * ii_chunk; ++ii) {
        for (int jj = 0; jj < NUM_TILES; ++jj) {
            // init csub <- 0
            memset(csub, 0, sizeof(csub));

            for (int t = 0; t < NUM_TILES; ++t) {
                // load asub, bsub <- a, b
                for (int i = 0; i < TSIZE; ++i) {
                    for (int j = 0; j < TSIZE; ++j) {
                        asub[i][j] = a[ii * TSIZE + i][t * TSIZE + j];
                        bsub[i][j] = b[t * TSIZE + i][jj * TSIZE + j];
                    }
                }
                // calculate csub
                for (int i = 0; i < TSIZE; ++i) {
                    for (int j = 0; j < TSIZE; ++j) {
                        for (int k = 0; k < TSIZE; ++k) {
                            csub[i][j] += asub[i][k] * bsub[k][j];
                        }
                    }
                }
            }

            // store csub -> c
            for (int i = 0; i < TSIZE; ++i) {
                for (int j = 0; j < TSIZE; ++j) {
                    c[ii * TSIZE + i][jj * TSIZE + j] = csub[i][j];
                }
            }
        }
    }
#else
    int i_chunk = N / size;
    for (int i = id * i_chunk; i < (id + 1) * i_chunk; ++i) {
        for (int k = 0; k < N; ++k) {
            register float val = a[i][k];

            for (int j = 0; j < N; ++j) {
                c[i][j] += val * b[k][j];
            }
        }
    }
#endif

#ifdef NON_BLOCKING
    MPI_Request request_c;
    MPI_Status status_c;

    MPI_Igather(
        (float*)c + rank * elements_per_proc, elements_per_proc, MPI_FLOAT,
        (float*)c, elements_per_proc, MPI_FLOAT,
        0, MPI_COMM_WORLD, &request_c
    );

    MPI_Wait(&request_c, &status_c);
#else
    MPI_Gather(
        (float*)c + rank * elements_per_proc, elements_per_proc, MPI_FLOAT,
        (float*)c, elements_per_proc, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );
#endif
}

/*
 * ==================================================================
 *                      DO NOT EDIT BELOW THIS LINE
 * ==================================================================
 */

void check_mat_mul() {
    printf("Validating...\n");

    bool is_valid = true;
    float eps = 1e-3;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0;
            for (int k = 0; k < N; ++k) {
                s += a[i][k] * b[k][j];
            }
            if (fabsf(c[i][j] - s) > eps && (s == 0 || fabsf((c[i][j] - s) / s) > eps)) {
                printf("c[%d][%d] : correct_value = %f, your_value = %f\n", i, j, s, c[i][j]);
                is_valid = false;
            }
        }
    }

    if (is_valid) {
        printf("result: VALID\n");
    } else {
        printf("result: INVALID\n");
    }
}

void generate_mat(float (*a)[N]) {
    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < N; ++j) {
            a[i][j] = (float)rand() / RAND_MAX - 0.5;
        }
    }
}

void print_mat(float (*a)[N]) {
    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < N; ++j) {
            printf("%+.3f ", a[i][j]);
        }
        printf("\n");
    }
}

void print_help(const char* prog_name) {
    printf("Usage: %s [-pvh]\n", prog_name);
    printf("OPTIONS\n");
    printf("  -p : print matrix data.\n");
    printf("  -v : validate matrix multiplication.\n");
    printf("  -h : print this page.\n");
}

void parse_opt(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "pvh")) != -1) {
        switch(opt) {
            case 'p':
                print_matrix = true;
                break;
            case 'v':
                validation = true;
                break;
            case 'h':
            default:
                print_help(argv[0]);
                exit(0);
        }
    }
}

int main(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    
    // Find out rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        parse_opt( argc, argv );

        generate_mat(a);
        generate_mat(b);

        printf("Calculating..."); fflush(stdout);
        timer_start(0);
    }

    mat_mul();

    if (rank == 0) {
        double elapsed_time = timer_stop(0);
        printf(" done!\n");

        if (print_matrix) {
            printf("MATRIX A:\n"); print_mat(a);
            printf("MATRIX B:\n"); print_mat(b);
            printf("MATRIX C:\n"); print_mat(c);
        }

        if (validation) {
            check_mat_mul();
        } else {
            printf("Validation is skipped.\n");
        }

        printf("Elapsed time: %f sec\n", elapsed_time);
    }

    MPI_Finalize();
    return 0;
}
