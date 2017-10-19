#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define N (1 << 24)

double gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main()
{
    float sum0 = 0.1f, sum1 = 0.1f, sum2 = 0.1f, sum3 = 0.1f;
    float sum4 = 0.1f, sum5 = 0.1f, sum6 = 0.1f, sum7 = 0.1f;
    float sum8 = 0.1f, sum9 = 0.1f, sum10 = 0.1f, sum11 = 0.1f;
    float sum12 = 0.1f, sum13 = 0.1f, sum14 = 0.1f, sum15 = 0.1f;

    float d = 0.001f, D = 0.002f;
    double start_t, end_t;

    start_t = gettime();
    for (int i = 0; i < N; i++) {
        sum0 += d, sum1 += d, sum2 += d, sum3 += d;
        sum4 += d, sum5 += d, sum6 += d, sum7 += d;
        sum8 += d, sum9 += d, sum10 += d, sum11 += d;
        sum12 += d, sum13 += d, sum14 += d, sum15 += d;

        sum0 += D, sum1 += D, sum2 += D, sum3 += D;
        sum4 += D, sum5 += D, sum6 += D, sum7 += D;
        sum8 += D, sum9 += D, sum10 += D, sum11 += D;
        sum12 += D, sum13 += D, sum14 += D, sum15 += D;
    }
    end_t = gettime();

    printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n",
        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, 
        sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15);
    printf("# of floating point operations = %d flop\n", 32 * N);
    printf("total elapsed time = %f seconds\n", end_t - start_t);
    printf("performance = %f FLOPS\n", 32 * N / (end_t - start_t));
    return 0;
}

