#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <CL/cl.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

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

/* OpenCL variables */
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;

cl_program program;
cl_kernel kernel;

cl_mem bufA, bufB, bufC;

cl_int err;

#define N (1 << 24)
int A[N], B[N], C[N];

double write_time, kernel_time, read_time, t;

int main()
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
    kernel = clCreateKernel(program, "vec_add", NULL);

    /* Initialize */
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
 
    /* Create buffer */
    bufA = clCreateBuffer(
        context, CL_MEM_READ_ONLY, sizeof(int) * N, NULL, NULL);
    bufB = clCreateBuffer(
        context, CL_MEM_READ_ONLY, sizeof(int) * N, NULL, NULL);
    bufC = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, sizeof(int) * N, NULL, NULL);

    /* Write buffer */
    t = get_time();
    clEnqueueWriteBuffer(
        queue, bufA, CL_TRUE, 0, sizeof(int) * N, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(
        queue, bufB, CL_TRUE, 0, sizeof(int) * N, B, 0, NULL, NULL);
    write_time = get_time() - t;

    /* Set kernel arguments */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    /* Launch the kernel */
    t = get_time();
    size_t global_size = N;
    size_t local_size = 1;
    clEnqueueNDRangeKernel(
        queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    clFinish(queue);
    kernel_time = get_time() - t;

    /* Read buffer */
    t = get_time();
    clEnqueueReadBuffer(
        queue, bufC, CL_TRUE, 0, sizeof(int) * N, C, 0, NULL, NULL);
    read_time = get_time() - t;

    /* Verification */
    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) {
            printf("Incorrect (i = %d : %d != %d + %d)\n",
                i, C[i], A[i], B[i]);
            break;
        }
    }

    /* Release OpenCL object */
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    printf("write buffer: %f seconds\n", write_time);
    printf("kernel: %f seconds\n", kernel_time);
    printf("read buffer: %f seconds\n\n", read_time);

    printf("Finished!\n");
    return 0;
}
