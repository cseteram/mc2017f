#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

__global__ void vec_add(int *x, int *y, int *z, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

const int n = 1 << 24;
int *h_x, *h_y, *h_z;
int *d_x, *d_y, *d_z;

double write_time, kernel_time, read_time, t;

int main()
{
    /* host buffer setup */
    h_x = (int*)malloc(sizeof(int) * n);
    h_y = (int*)malloc(sizeof(int) * n);
    h_z = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = rand() % 100;
        h_y[i] = rand() % 100;
    }

    /* device buffer setup */
    cudaMalloc(&d_x, sizeof(int) * n);
    cudaMalloc(&d_y, sizeof(int) * n);
    cudaMalloc(&d_z, sizeof(int) * n);

    /* host to device memory transfer */
    t = get_time();
    cudaMemcpy(d_x, h_x, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(int) * n, cudaMemcpyHostToDevice);
    write_time = get_time() - t;

    /* kernel execution */
    int threads_per_block = 1024;
    int num_of_blocks = (n + threads_per_block - 1) / threads_per_block;
    t = get_time();
    vec_add<<<num_of_blocks, threads_per_block>>>(d_x, d_y, d_z, n);
    cudaDeviceSynchronize();
    kernel_time = get_time() - t;

    /* device to host memory transfer */
    t = get_time();
    cudaMemcpy(h_z, d_z, sizeof(int) * n, cudaMemcpyDeviceToHost);
    read_time = get_time() - t;

    /* verification */
    for (int i = 0; i < n; ++i) {
        if (h_x[i] + h_y[i] != h_z[i]) {
            printf("Incorrect (i = %d : %d + %d != %d)\n",
                i, h_x[i], h_y[i], h_z[i]);
            break;
        }
    }

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    printf("write buffer: %f seconds\n", write_time);
    printf("kernel: %f seconds\n", kernel_time);
    printf("read buffer: %f seconds\n\n", read_time);

    printf("Finished!\n");
    return 0;
}
