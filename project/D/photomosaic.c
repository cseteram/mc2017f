#include "photomosaic.h"
#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <CL/cl.h>
#include <omp.h>
#include <mpi.h>

#define BATCH_SIZE 1024
#define K 4
#define NUM_THREADS 32

typedef unsigned char uchar;
#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

/* OpenCL variables */
cl_platform_id platform;
cl_device_id device[K];
cl_context context;
cl_command_queue queue[K];
cl_program program;

size_t binaries_size[K];
const uchar *binaries[K];

cl_kernel kernel_conv[K], kernel_reduce[K];
cl_kernel kernel_transpose[K];

cl_mem buf_img[K], buf_dataset[K];
cl_mem buf_img_t[K], buf_dataset_t[K];
cl_mem buf_diff[K], buf_idx[K];
cl_mem buf_diff_reduced[K], buf_idx_reduced[K];
cl_int err;

int *diff_reduced[K], *idx_reduced[K];

char *get_source_code(const char *file_name, size_t *len);
uchar *get_binary(const char *file_name, size_t *len);
void setup_opencl(int width, int height);
void release_opencl();
size_t round_work_size(size_t work_size, size_t group_size);
void set_work_size_rounded(size_t *work_size, size_t *group_size, int n);

void photomosaic(unsigned char *img, int width, int height, unsigned char *dataset, int *idx)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int swidth = width / 32, sheight = height / 32;
    const int batch_size = BATCH_SIZE;
    const int num_tiles_all = sheight * swidth;
    const int filter_size = 3 * 32 * 32;
    const int num_filters = 60000;
    const int reduction_count = (num_filters + 255) / 256;
    for (int k = 0; k < K; ++k) {
        diff_reduced[k] = (int*)malloc(sizeof(int) * batch_size * reduction_count);
        idx_reduced[k] = (int*)malloc(sizeof(int) * batch_size * reduction_count);
    }

    uchar *img_t = (uchar*)malloc(sizeof(uchar) * num_tiles_all * filter_size);
    if (rank == 0) {
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            #pragma omp for schedule(guided) collapse(2)
            for (int sh = 0; sh < sheight; ++sh) {
                for (int sw = 0; sw < swidth; ++sw) {
                    for (int c = 0; c < 3; ++c) {
                        for (int h = 0; h < 32; ++h) {
                            for (int w = 0; w < 32; ++w) {
                                // img_t[sh][sw][c][h][w] = img[sh * 32 + h][sw * 32 + w][c]
                                img_t[(sh * swidth + sw) * filter_size + (c * 32 + h) * 32 + w] = img[(sh * 32 + h) * width * 3 + (sw * 32 + w) * 3 + c];
                            }
                        }
                    }
                }
            }
        }
    }

    int *num_tiles_per_node = (int*)malloc(sizeof(int) * size);
    int *num_tiles_offset = (int*)malloc(sizeof(int) * size);
    int *img_t_size = (int*)malloc(sizeof(int) * size);
    int *img_t_offset = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; ++i) {
        num_tiles_per_node[i] = num_tiles_all / size;
        if (i < num_tiles_all % size) ++num_tiles_per_node[i];
        num_tiles_offset[i] = (i > 0) ? (num_tiles_offset[i - 1] + num_tiles_per_node[i - 1]) : 0;

        img_t_size[i] = num_tiles_per_node[i] * filter_size;
        img_t_offset[i] = (i > 0) ? (img_t_offset[i - 1] + img_t_size[i - 1]) : 0;
    }
    int num_tiles = num_tiles_per_node[rank];

    MPI_Request request_img_t;
    MPI_Status status_img_t;
    MPI_Iscatterv(img_t, img_t_size, img_t_offset, MPI_UNSIGNED_CHAR, img_t, num_tiles * filter_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD, &request_img_t);

    if (rank == 0) {
        printf("\nNumber of tiles = %d x %d = %d\n", sheight, swidth, num_tiles_all);
        printf("Number of processors = %d\n", size);
        for (int i = 0; i < size; ++i) {
            printf(" - rank %d : %d tiles ( tiles[%d ... %d) )\n", i, num_tiles_per_node[i], num_tiles_offset[i], num_tiles_offset[i] + num_tiles_per_node[i]);
        }
        printf("\n");
    }

    if (rank == 0)
        timer_start(1);
    setup_opencl(batch_size, num_filters + 32);
    if (rank == 0)
        printf("setup opencl : %f seconds\n\n", timer_stop(1));
    MPI_Wait(&request_img_t, &status_img_t);

    for (int k = 0; k < K; ++k) {
        clEnqueueWriteBuffer(
            queue[k], buf_dataset[k], CL_FALSE,
            0, sizeof(uchar) * num_filters * filter_size,
            dataset, 0, NULL, NULL
        );
    }

    const int Q = filter_size, R = num_filters + 32;
    for (int k = 0; k < K; ++k) {
        size_t gws_trans2[] = {Q, R};
        size_t lws_trans2[] = {16, 16};
        err  = clSetKernelArg(kernel_transpose[k], 0, sizeof(cl_mem), &buf_dataset[k]);
        err |= clSetKernelArg(kernel_transpose[k], 1, sizeof(cl_mem), &buf_dataset_t[k]);
        err |= clSetKernelArg(kernel_transpose[k], 2, sizeof(int), &R);
        err |= clSetKernelArg(kernel_transpose[k], 3, sizeof(int), &Q);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(
            queue[k], kernel_transpose[k], 2, NULL, gws_trans2, lws_trans2, 0, NULL, NULL
        );
        CHECK_ERROR(err);
    }

    for (int i = 0; i < num_tiles; i += K * batch_size) {
        const int ntiles = (i + K * batch_size < num_tiles) ? K * batch_size : num_tiles - i;
        printf("[rank %d] Calculate tiles[%d ... %d) (%d, %d / %d)\n",
            rank, i, i + ntiles, ntiles, i, num_tiles
        );

        int ntiles_per_device[K], ntiles_offset[K];
        for (int k = 0; k < K; ++k) {
            ntiles_per_device[k] = ntiles / K;
            if (k < ntiles % K) ++ntiles_per_device[k];
            ntiles_offset[k] = (k > 0) ? (ntiles_offset[k - 1] + ntiles_per_device[k - 1]) : 0;
        }

        for (int k = 0; k < K; ++k) {
            if (ntiles_per_device[k] == 0) continue;
            const int P = (ntiles_per_device[k] + 63) / 64 * 64;

            clEnqueueWriteBuffer(
                queue[k], buf_img_t[k], CL_FALSE,
                0, sizeof(uchar) * ntiles_per_device[k] * filter_size,
                img_t + (i + ntiles_offset[k]) * filter_size, 0, NULL, NULL
            );
        
            size_t gws_conv[] = {R >> 2, P >> 2};
            size_t lws_conv[] = {16, 16};
            err  = clSetKernelArg(kernel_conv[k], 0, sizeof(cl_mem), &buf_img_t[k]);
            err |= clSetKernelArg(kernel_conv[k], 1, sizeof(cl_mem), &buf_dataset_t[k]);
            err |= clSetKernelArg(kernel_conv[k], 2, sizeof(cl_mem), &buf_diff[k]);
            err |= clSetKernelArg(kernel_conv[k], 3, sizeof(int), &P);
            err |= clSetKernelArg(kernel_conv[k], 4, sizeof(int), &Q);
            err |= clSetKernelArg(kernel_conv[k], 5, sizeof(int), &R);
            CHECK_ERROR(err);
            err = clEnqueueNDRangeKernel(
                queue[k], kernel_conv[k], 2, NULL, gws_conv, lws_conv, 0, NULL, NULL
            );
            CHECK_ERROR(err);

            size_t gws_reduce[] = {num_filters, ntiles_per_device[k]};
            size_t lws_reduce[] = {256, 1};
            set_work_size_rounded(gws_reduce, lws_reduce, 2);
            err  = clSetKernelArg(kernel_reduce[k], 0, sizeof(cl_mem), &buf_diff[k]);
            err |= clSetKernelArg(kernel_reduce[k], 1, sizeof(cl_mem), &buf_diff_reduced[k]);
            err |= clSetKernelArg(kernel_reduce[k], 2, sizeof(cl_mem), &buf_idx_reduced[k]);
            err |= clSetKernelArg(kernel_reduce[k], 3, sizeof(int) * 256, NULL);
            err |= clSetKernelArg(kernel_reduce[k], 4, sizeof(int) * 256, NULL);
            err |= clSetKernelArg(kernel_reduce[k], 5, sizeof(int), &ntiles_per_device[k]);
            err |= clSetKernelArg(kernel_reduce[k], 6, sizeof(int), &num_filters);
            err |= clSetKernelArg(kernel_reduce[k], 7, sizeof(int), &R);
            CHECK_ERROR(err);
            err = clEnqueueNDRangeKernel(
                queue[k], kernel_reduce[k], 2, NULL, gws_reduce, lws_reduce, 0, NULL, NULL
            );
            CHECK_ERROR(err);

            clEnqueueReadBuffer(
                queue[k], buf_diff_reduced[k], CL_FALSE,
                0, sizeof(int) * ntiles_per_device[k] * reduction_count,
                diff_reduced[k], 0, NULL, NULL
            );
            clEnqueueReadBuffer(
                queue[k], buf_idx_reduced[k], CL_FALSE,
                0, sizeof(int) * ntiles_per_device[k] * reduction_count,
                idx_reduced[k], 0, NULL, NULL
            );
        }

        for (int k = 0; k < K; ++k) {
            clFinish(queue[k]);
            #pragma omp parallel num_threads(NUM_THREADS)
            {
                #pragma omp for schedule(guided)
                for (int t = 0; t < ntiles_per_device[k]; ++t) {
                    int diff = INT_MAX, min_j = -1;
                    for (int j = 0; j < reduction_count; ++j) {
                        if (diff_reduced[k][t * reduction_count + j] < diff) {
                            diff = diff_reduced[k][t * reduction_count + j];
                            min_j = idx_reduced[k][t * reduction_count + j];
                        }
                    }
                    idx[i + ntiles_offset[k] + t] = min_j;
                }
            }
        }
    }

    for (int i = 0; i < num_tiles; ++i) {
        if (idx[i] < 0 || idx[i] >= 60000) {
            printf("Invalid index at i = %d (idx[%d] = %d)\n", i, i, idx[i]);
            idx[i] = 0;
        }
    }

    if (rank == 0)
        printf("\n");
    MPI_Gatherv(idx, num_tiles, MPI_INT, idx, num_tiles_per_node, num_tiles_offset, MPI_INT, 0, MPI_COMM_WORLD);
    release_opencl();
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

uchar *get_binary(const char *file_name, size_t *len)
{
    uchar *binary;
    size_t binary_size;
    FILE *file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    binary_size = (size_t)ftell(file);
    rewind(file);

    binary = (uchar*)malloc(binary_size);
    fread(binary, binary_size, 1, file);
    fclose(file);

    *len = binary_size;
    return binary;
}

void setup_opencl(int batch_size, int num_filters)
{
    double t9, t10, t11, t12, t13, t14, t15;

    /* Get platform, device, context, command_queue */
    timer_start(9);
    clGetPlatformIDs(1, &platform, NULL);
    t9 = timer_stop(9);
    timer_start(10);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, K, device, NULL);
    t10 = timer_stop(10);
    timer_start(11);
    context = clCreateContext(NULL, K, device, NULL, NULL, NULL);
    t11 = timer_stop(11);
    timer_start(12);
    for (int i = 0; i < K; ++i)
        queue[i] = clCreateCommandQueue(context, device[i], 0, NULL);
    t12 = timer_stop(12);

    /* Compile the kernel code */
    timer_start(13);
    /*
    size_t source_size;
    const char *source_code = get_source_code("kernel.cl.c", &source_size);
    program = clCreateProgramWithSource(
        context, 1, &source_code, &source_size, NULL);
    err = clBuildProgram(program, K, device, "", NULL, NULL);
    */
    size_t binary_size;
    uchar *kernel_binary = get_binary("kernel.bin", &binary_size);
    for (int k = 0; k < K; ++k) {
        binaries_size[k] = binary_size;
        binaries[k] = kernel_binary;
    }
    program = clCreateProgramWithBinary(
        context, 4, device, binaries_size, binaries, NULL, &err);
    err = clBuildProgram(program, 4, device, "", NULL, NULL);
    t13 = timer_stop(13);
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
    timer_start(14);
    for (int i = 0; i < K; ++i) {
        kernel_conv[i] = clCreateKernel(program, "conv", NULL);
        kernel_reduce[i] = clCreateKernel(program, "reduction", NULL);
        kernel_transpose[i] = clCreateKernel(program, "transpose", NULL);
    }
    t14 = timer_stop(14);
 
    /* Create buffer */
    timer_start(15);
    int filter_size = 3 * 32 * 32;
    int reduction_count = (num_filters + 255) / 256;
    for (int i = 0; i < K; ++i) {
        buf_img_t[i] = clCreateBuffer(
            context, CL_MEM_READ_ONLY, sizeof(uchar) * batch_size * filter_size, NULL, NULL);
        buf_dataset[i] = clCreateBuffer(
            context, CL_MEM_READ_ONLY, sizeof(uchar) * num_filters * filter_size, NULL, NULL);
        buf_dataset_t[i] = clCreateBuffer(
            context, CL_MEM_READ_ONLY, sizeof(uchar) * filter_size * num_filters, NULL, NULL);
        buf_diff[i] = clCreateBuffer(
            context, CL_MEM_READ_WRITE, sizeof(int) * batch_size * num_filters, NULL, NULL);
        buf_idx[i] = clCreateBuffer(
            context, CL_MEM_READ_WRITE, sizeof(int) * batch_size, NULL, NULL);
        buf_diff_reduced[i] = clCreateBuffer(
            context, CL_MEM_READ_WRITE, sizeof(int) * batch_size * reduction_count, NULL, NULL);
        buf_idx_reduced[i] = clCreateBuffer(
            context, CL_MEM_READ_WRITE, sizeof(int) * batch_size * reduction_count, NULL, NULL);
    }
    t15 = timer_stop(15);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("\nGetPlatformIDs : %f seconds\n", t9);
        printf("GetDeviceIDs : %f seconds\n", t10);
        printf("CreateContext : %f seconds\n", t11);
        printf("CreateCommandQueue : %f seconds\n", t12);
        printf("BuildProgram : %f seconds\n", t13);
        printf("CreateKernel : %f seconds\n", t14);
        printf("CreateBuffer : %f seconds\n\n", t15);
    }
}

void release_opencl()
{
    /* Release OpenCL object */
    for (int i = 0; i < K; ++i) {
        clReleaseMemObject(buf_dataset[i]);
        clReleaseMemObject(buf_img_t[i]);
        clReleaseMemObject(buf_dataset_t[i]);
        clReleaseMemObject(buf_diff[i]);
        clReleaseMemObject(buf_idx[i]);
        clReleaseCommandQueue(queue[i]);
        clReleaseKernel(kernel_transpose[i]);
        clReleaseKernel(kernel_conv[i]);
        clReleaseKernel(kernel_reduce[i]);
    }
    clReleaseContext(context);
    clReleaseProgram(program);
}

size_t round_work_size(size_t work_size, size_t group_size)
{
    size_t rem = work_size % group_size;
    return (rem == 0) ? work_size : (work_size + group_size - rem);
}

void set_work_size_rounded(size_t *work_size, size_t *group_size, int n)
{
    for (int i = 0; i < n; ++i) {
        work_size[i] = round_work_size(work_size[i], group_size[i]);
    }
}
