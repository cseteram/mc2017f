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

#define N (1 << 24)
#define D 2
#define S (N / D)

/* OpenCL variables */
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue[3];

cl_program program;
cl_kernel kernel;
cl_event write_event[D], kernel_event[D], read_event[D];

cl_mem bufA[D], bufB[D], bufC[D];

cl_int err;

int A[N], B[N], C[N];

double total_time, t;

int main()
{
    /* Get platform, device, context, command_queue */
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    for (int i = 0; i < 3; i++)
        queue[i] = clCreateCommandQueue(context, device, 0, NULL);

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
    for (int i = 0; i < D; i++) {
        bufA[i] = clCreateBuffer(
            context, CL_MEM_READ_ONLY, sizeof(int) * S, NULL, NULL);
        bufB[i] = clCreateBuffer(
            context, CL_MEM_READ_ONLY, sizeof(int) * S, NULL, NULL);
        bufC[i] = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY, sizeof(int) * S, NULL, NULL);
    }

    t = get_time();

    /* Write buffer */
    for (int i = 0; i < D; i++) {
        clEnqueueWriteBuffer(
            queue[0], bufA[i], CL_FALSE, 0, sizeof(int) * S, A + i * S,
            0, NULL, &write_event[i]
        );
        clEnqueueWriteBuffer(
            queue[0], bufB[i], CL_FALSE, 0, sizeof(int) * S, B + i * S,
            0, NULL, &write_event[i]
        );
    }

    /* Set kernel arguments & Launch the kernel */
    for (int i = 0; i < D; i++) {
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA[i]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB[i]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC[i]);

        /* Launch the kernel */
        size_t global_size = S;
        size_t local_size = 256;
        clEnqueueNDRangeKernel(
            queue[1], kernel, 1, NULL, &global_size, &local_size,
            1, &write_event[i], &kernel_event[i]
        );
    }

    /* Read buffer */
    for (int i = 0; i < D; i++) {
        clEnqueueReadBuffer(
            queue[2], bufC[i], CL_TRUE, 0, sizeof(int) * S, C + i * S,
            1, &kernel_event[i], &read_event[i]
        );
    }

    total_time = get_time() - t;

    /* Verification */
    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) {
            printf("Incorrect (i = %d : %d != %d + %d)\n",
                i, C[i], A[i], B[i]);
            break;
        }
    }

    /* Release OpenCL object */
    for (int i = 0; i < D; i++) {
        clReleaseMemObject(bufA[i]);
        clReleaseMemObject(bufB[i]);
        clReleaseMemObject(bufC[i]);

        clReleaseEvent(write_event[i]);
        clReleaseEvent(kernel_event[i]);
        clReleaseEvent(read_event[i]);
    }
    clReleaseContext(context);
    for (int i = 0; i < 3; i++)
        clReleaseCommandQueue(queue[i]);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    printf("total elapsed time: %f seconds\n", total_time);
    printf("Finished!\n");
    return 0;
}
