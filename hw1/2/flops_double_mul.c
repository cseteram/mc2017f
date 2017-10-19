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
    double sum0 = 0.1, sum1 = 0.1, sum2 = 0.1, sum3 = 0.1;
    double sum4 = 0.1, sum5 = 0.1, sum6 = 0.1, sum7 = 0.1;
    double sum8 = 0.1, sum9 = 0.1, sum10 = 0.1, sum11 = 0.1;
    double sum12 = 0.1, sum13 = 0.1, sum14 = 0.1, sum15 = 0.1;

    double m = 30859.0 / 18190.0, M = 18190.0 / 30859.0;
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

