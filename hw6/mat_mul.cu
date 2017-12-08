#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "timer.hpp"

#define N 2048

bool print_matrix = false;
bool validation = false;

float a[N][N], b[N][N], c[N][N];
float *d_a, *d_b, *d_c;

__global__ void mat_mul(
    float *A, float *B, float *C,
    int ROW_A, int COL_A, int COL_B)
{
    const int i = blockDim.y * blockIdx.y + threadIdx.y;
    const int j = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
    if (i < ROW_A && j < COL_B) {
        for (int k = 0; k < COL_A; ++k) {
            sum += A[i * COL_A + k] * B[k * COL_B + j];
        }
        C[i * COL_B + j] = sum;
    }
}

void mat_mul() {
    cudaMalloc(&d_a, sizeof(float) * N * N);
    cudaMalloc(&d_b, sizeof(float) * N * N);
    cudaMalloc(&d_c, sizeof(float) * N * N);

    cudaMemcpy(d_a, a, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    dim3 num_of_blocks(N / 32, N / 32);
    dim3 threads_per_block(32, 32);
    mat_mul<<<num_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N, N, N);

    cudaMemcpy(c, d_c, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
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
