#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "timer.h"

#include <string.h>
#include <pthread.h>
#include <immintrin.h>

#define N 2048
#define NUM_THREADS 16
#define BSIZE 16
#define BLOCK_SIZE (N / NUM_THREADS)
#define NUM_TILES (N / BSIZE)

#define TILING
// #define BASIC

bool print_matrix = false;
bool validation = false;

float a[N][N], b[N][N], c[N][N];
pthread_t threads[NUM_THREADS];

void *thread_mat_mul(void *thread_id)
{
    long id = (long)thread_id;

#ifdef TILING
    int ii_chunk = BLOCK_SIZE / BSIZE;
    float asub[BSIZE][BSIZE], bsub[BSIZE][BSIZE], csub[BSIZE][BSIZE];
    
    for (int ii = id * ii_chunk; ii < (id + 1) * ii_chunk; ++ii) {
        for (int jj = 0; jj < NUM_TILES; ++jj) {
            // init csub <- 0
            memset(csub, 0, sizeof(csub));

            for (int t = 0; t < NUM_TILES; ++t) {
                // load asub, bsub <- a, b
                for (int i = 0; i < BSIZE; ++i) {
                    for (int j = 0; j < BSIZE; ++j) {
                        asub[i][j] = a[ii * BSIZE + i][t * BSIZE + j];
                        bsub[i][j] = b[t * BSIZE + i][jj * BSIZE + j];
                    }
                }
                // calculate csub
                for (int i = 0; i < BSIZE; ++i) {
                    for (int j = 0; j < BSIZE; ++j) {
                        for (int k = 0; k < BSIZE; ++k) {
                            csub[i][j] += asub[i][k] * bsub[k][j];
                        }
                    }
                }
            }

            // store csub -> c
            for (int i = 0; i < BSIZE; ++i) {
                for (int j = 0; j < BSIZE; ++j) {
                    c[ii * BSIZE + i][jj * BSIZE + j] = csub[i][j];
                }
            }
        }
    }
#endif

#ifdef BASIC
    for (int i = id * BLOCK_SIZE; i < (id + 1) * BLOCK_SIZE; ++i) {
        for (int k = 0; k < N; ++k) {
            register float val = a[i][k];

            for (int j = 0; j < N; ++j) {
                c[i][j] += val * b[k][j];
            }
        }
    }
#endif

    pthread_exit(NULL);
}

void mat_mul() {
    for (long i = 0; i < NUM_THREADS; ++i)
        pthread_create(&threads[i], NULL, &thread_mat_mul, (void*)i);
    for (long i = 0; i < NUM_THREADS; ++i)
        pthread_join(threads[i], NULL);
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
    parse_opt( argc, argv );

    generate_mat(a);
    generate_mat(b);

    printf("Calculating..."); fflush(stdout);
    timer_start(0);
    mat_mul();
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

    return 0;
}
