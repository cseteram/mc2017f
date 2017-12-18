#include <stdio.h>
#include <CL/cl.h>

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
cl_int err;

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
    
    size_t binary_size;
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
    unsigned char *binary = (unsigned char*)malloc(binary_size);
    clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary, NULL);
    
    FILE *out = fopen("kernel.bin", "wb");
    fwrite(binary, sizeof(unsigned char), binary_size, out);
    fclose(out);

    /* Release OpenCL object */
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);

    printf("Finished!\n");
    return 0;
}
