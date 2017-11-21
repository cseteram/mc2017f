#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "timer.h"

#include <omp.h>
#include <string.h>

#define N 2048
#define NUM_THREADS 32

bool print_matrix = false;
bool validation = false;

float a[N][N], b[N][N], c[N][N];

void mat_mul() {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                register float val = a[i][k];

                for (int j = 0; j < N; ++j) {
                    c[i][j] += val * b[k][j];
                }
            }
        }
    }
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
