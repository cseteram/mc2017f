#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "timer.h"

#include <CL/cl.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

#define N 8192
#define K 4

// #define BASIC
// #define TILING_16
#define TILING_64

bool print_matrix = false;
bool validation = false;

float a[N][N], b[N][N], c[N][N];

/* OpenCL variables */
cl_platform_id platform;
cl_device_id device[K];
cl_context context;
cl_command_queue queue[K];
cl_program program;

cl_kernel kernel[K];

cl_mem bufA[K], bufB[K], bufC[K];
cl_int err;

char *get_source_code(const char *file_name, size_t *len)
{
    char *source_code;
    size_t length;
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    length = (size_t)ftell(file);
    rewind(file);

    source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';

    fclose(file);

    *len = length;
    return source_code;
}

void setup_opencl()
{
    /* Get platform, device, context, command_queue */
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, K, device, NULL);
    context = clCreateContext(NULL, K, device, NULL, NULL, &err);
    for (int i = 0; i < K; i++)
        queue[i] = clCreateCommandQueue(context, device[i], 0, NULL);

    /* Compile the kernel code */
    size_t source_size;
    const char *source_code = get_source_code("kernel.cl.c", &source_size);
    program = clCreateProgramWithSource(
        context, 1, &source_code, &source_size, NULL);
    err = clBuildProgram(program, K, device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        char *log;
        size_t log_size;
        clGetProgramBuildInfo(
            program, device[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(
            program, device[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        printf("Compile error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < K; i++) {
#ifdef BASIC
        kernel[i] = clCreateKernel(program, "mat_mul", NULL);
#endif
#ifdef TILING_16
        kernel[i] = clCreateKernel(program, "mat_mul_t16", NULL);
#endif
#ifdef TILING_64
        kernel[i] = clCreateKernel(program, "mat_mul_t64", NULL);
#endif
    }
 
    /* Create buffer */
    for (int i = 0; i < K; i++) {
        bufA[i] = clCreateBuffer(
            context, CL_MEM_READ_ONLY, sizeof(float) * (N / K) * N, NULL, NULL);
        bufB[i] = clCreateBuffer(
            context, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
        bufC[i] = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY, sizeof(float) * (N / K) * N, NULL, NULL);
    }
}

void release_opencl()
{
    /* Release OpenCL object */
    for (int i = 0; i < K; i++) {
        clReleaseMemObject(bufA[i]);
        clReleaseMemObject(bufB[i]);
        clReleaseMemObject(bufC[i]);
        clReleaseCommandQueue(queue[i]);
        clReleaseKernel(kernel[i]);
    }
    clReleaseContext(context);
    clReleaseProgram(program);
}

void mat_mul()
{
    timer_start(1);
    setup_opencl();
    double setup_time = timer_stop(1);

    /* Write buffer */
    timer_start(2);
    for (int i = 0; i < K; i++) {
        clEnqueueWriteBuffer(
            queue[i], bufA[i], CL_FALSE, 0,
            sizeof(float) * (N / K) * N, (float*)a + i * (N / K) * N,
            0, NULL, NULL
        );
        clEnqueueWriteBuffer(
            queue[i], bufB[i], CL_FALSE, 0,
            sizeof(float) * N * N, b,
            0, NULL, NULL
        );
    }
    for (int i = 0; i < K; i++) {
        clFinish(queue[i]);
    }
    double write_time = timer_stop(2);

    /* Set kernel arguments */
    const int P = N / K, Q = N, R = N;
    for (int i = 0; i < K; i++) {
        clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &bufA[i]);
        clSetKernelArg(kernel[i], 1, sizeof(cl_mem), &bufB[i]);
        clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &bufC[i]);
        clSetKernelArg(kernel[i], 3, sizeof(int), &P);
        clSetKernelArg(kernel[i], 4, sizeof(int), &Q);
        clSetKernelArg(kernel[i], 5, sizeof(int), &R);
    }

#ifdef BASIC
    size_t global_size[] = {R, P};
    size_t local_size[] = {16, 16};
#endif
#ifdef TILING_16
    size_t global_size[] = {R, P};
    size_t local_size[] = {16, 16};
#endif
#ifdef TILING_64
    size_t global_size[] = {R >> 2, P >> 2};
    size_t local_size[] = {16, 16};
#endif

    /* Launch the kernel */
    timer_start(3);
    for (int i = 0; i < K; i++) {
        clEnqueueNDRangeKernel(
            queue[i], kernel[i], 2, NULL, global_size, local_size, 0, NULL, NULL);
    }
    for (int i = 0; i < K; i++) {
        clFinish(queue[i]);
    }
    double kernel_time = timer_stop(3);

    /* Read buffer */
    timer_start(4);
    for (int i = 0; i < K; i++) {
        clEnqueueReadBuffer(
            queue[i], bufC[i], CL_FALSE, 0,
            sizeof(float) * (N / K) * N, (float*)c + i * (N / K) * N,
            0, NULL, NULL
        );
    }
    for (int i = 0; i < K; i++) {
        clFinish(queue[i]);
    }
    double read_time = timer_stop(4);

    timer_start(5);
    release_opencl();    
    double release_time = timer_stop(5);

    printf("\n- setup_opencl: %f sec\n"
        "- release_opencl: %f sec\n"
        "- write_buffer: %f sec ----|\n"
        "- kernel: %f sec ----------|=> %f sec\n"
        "- read_buffer: %f sec -----|\n",
        setup_time, release_time, write_time, kernel_time,
        write_time + kernel_time + read_time, read_time
    );
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
