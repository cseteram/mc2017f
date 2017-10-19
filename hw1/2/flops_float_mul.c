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

    float m = 30859.0f / 18190.0f, M = 18190.0f / 30859.0f;
    double start_t, end_t;

    start_t = gettime();
    for (int i = 0; i < N; i++) {
        sum0 *= m, sum1 *= m, sum2 *= m, sum3 *= m;
        sum4 *= m, sum5 *= m, sum6 *= m, sum7 *= m;
        sum8 *= m, sum9 *= m, sum10 *= m, sum11 *= m;
        sum12 *= m, sum13 *= m, sum14 *= m, sum15 *= m;

        sum0 *= M, sum1 *= M, sum2 *= M, sum3 *= M;
        sum4 *= M, sum5 *= M, sum6 *= M, sum7 *= M;
        sum8 *= M, sum9 *= M, sum10 *= M, sum11 *= M;
        sum12 *= M, sum13 *= M, sum14 *= M, sum15 *= M;
    }
    end_t = gettime();

    printf("%e %e %e %e\n%e %e %e %e\n%e %e %e %e\n%e %e %e %e\n\n",
        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, 
        sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15);
    printf("# of floating point operations = %d flop\n", 32 * N);
    printf("total elapsed time = %f seconds\n", end_t - start_t);
    printf("performance = %f FLOPS\n", 32 * N / (end_t - start_t));
    return 0;
}

