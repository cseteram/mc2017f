#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

unsigned char *get_binary(const char *file_name, size_t *len)
{
    unsigned char *binary;
    size_t binary_size;
    FILE *file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    binary_size = (size_t)ftell(file);
    rewind(file);

    binary = (unsigned char*)malloc(binary_size);
    fread(binary, binary_size, 1, file);
    fclose(file);

    *len = binary_size;
    return binary;
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

#define N (1 << 26)
int A[N], B[N], C[N];

int main()
{
    /* Get platform, device, context, command_queue */
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    /* Compile the kernel code */
    size_t binary_size;
    const unsigned char *binary = get_binary("kernel.bin", &binary_size);
    program = clCreateProgramWithBinary(
        context, 1, &device, &binary_size, &binary, NULL, &err);
    clBuildProgram(program, 1, &device, "", NULL, NULL);
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
    clEnqueueWriteBuffer(
        queue, bufA, CL_FALSE, 0, sizeof(int) * N, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(
        queue, bufB, CL_FALSE, 0, sizeof(int) * N, B, 0, NULL, NULL);
    clFinish(queue);

    /* Set kernel arguments */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    /* Launch the kernel */
    size_t global_size = N;
    size_t local_size = 256;
    clEnqueueNDRangeKernel(
        queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    clFinish(queue);

    /* Read buffer */
    clEnqueueReadBuffer(
        queue, bufC, CL_TRUE, 0, sizeof(int) * N, C, 0, NULL, NULL);

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

    printf("Finished!\n");
    return 0;
}
