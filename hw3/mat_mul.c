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
// #define BASIC
// #define TILING_16
#define TILING_64

bool print_matrix = false;
bool validation = false;

float a[N][N], b[N][N], c[N][N];

/* OpenCL variables */
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

cl_kernel kernel;

cl_mem bufA, bufB, bufC;
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
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    /* Compile the kernel code */
    size_t source_size;
    const char *source_code = get_source_code("kernel.cl.c", &source_size);
    program = clCreateProgramWithSource(
        context, 1, &source_code, &source_size, NULL);
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        char *log;
        size_t log_size;
        clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        printf("Compile error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
#ifdef BASIC
    kernel = clCreateKernel(program, "mat_mul", NULL);
#endif
#ifdef TILING_16
    kernel = clCreateKernel(program, "mat_mul_t16", NULL);
#endif
#ifdef TILING_64
    kernel = clCreateKernel(program, "mat_mul_t64", NULL);
#endif
 
    /* Create buffer */
    bufA = clCreateBuffer(
        context, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
    bufB = clCreateBuffer(
        context, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
    bufC = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N, NULL, NULL);
}

void release_opencl()
{
    /* Release OpenCL object */
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
}

void mat_mul()
{
    setup_opencl();

    /* Write buffer */
    clEnqueueWriteBuffer(
        queue, bufA, CL_FALSE, 0, sizeof(float) * N * N, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(
        queue, bufB, CL_FALSE, 0, sizeof(float) * N * N, b, 0, NULL, NULL);

    /* Set kernel arguments */
    const int P = N, Q = N, R = N;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &P);
    clSetKernelArg(kernel, 4, sizeof(int), &Q);
    clSetKernelArg(kernel, 5, sizeof(int), &R);

    /* Launch the kernel */
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
    clEnqueueNDRangeKernel(
        queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    clFinish(queue);

    /* Read buffer */
    clEnqueueReadBuffer(
        queue, bufC, CL_TRUE, 0, sizeof(float) * N * N, c, 0, NULL, NULL);

    release_opencl();    
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
