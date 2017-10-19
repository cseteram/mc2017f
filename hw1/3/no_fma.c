#include <immintrin.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#define N (1 << 26)

double gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main()
{
    double start_t, end_t;

    double res00 = 0.0, res01 = 0.0, res02 = 0.0, res03 = 0.0;
    double veca00 = 1.5, veca01 = 1.2, veca02 = 2.0, veca03 = 1.8;
    double vecb00 = 0.7, vecb01 = 0.8, vecb02 = 0.5, vecb03 = 0.6;

    double res10 = 0.0, res11 = 0.0, res12 = 0.0, res13 = 0.0;
    double veca10 = 1.3, veca11 = 1.4, veca12 = 1.1, veca13 = 2.2;
    double vecb10 = 0.9, vecb11 = 0.7, vecb12 = 0.8, vecb13 = 0.4;

    double res20 = 0.0, res21 = 0.0, res22 = 0.0, res23 = 0.0;
    double veca20 = 1.6, veca21 = 1.9, veca22 = 2.1, veca23 = 1.4;
    double vecb20 = 0.6, vecb21 = 0.6, vecb22 = 0.5, vecb23 = 0.8;

    double res30 = 0.0, res31 = 0.0, res32 = 0.0, res33 = 0.0;
    double veca30 = 1.1, veca31 = 2.4, veca32 = 1.8, veca33 = 1.6;
    double vecb30 = 0.9, vecb31 = 0.4, vecb32 = 0.6, vecb33 = 0.4;

    start_t = gettime();
    for (int i = 0; i < N; i++) {
        res00 = veca00 * vecb00 + res00;
        res01 = veca01 * vecb01 + res01;
        res02 = veca02 * vecb02 + res02;
        res03 = veca03 * vecb03 + res03;

        res10 = veca10 * vecb10 + res10;
        res11 = veca11 * vecb11 + res11;
        res12 = veca12 * vecb12 + res12;
        res13 = veca13 * vecb13 + res13;

        res20 = veca20 * vecb20 + res20;
        res21 = veca21 * vecb21 + res21;
        res22 = veca22 * vecb22 + res22;
        res23 = veca23 * vecb23 + res23;

        res30 = veca30 * vecb30 + res30;
        res31 = veca31 * vecb31 + res31;
        res32 = veca32 * vecb32 + res32;
        res33 = veca33 * vecb33 + res33;
    }
    end_t = gettime();
  
    printf("%f %f %f %f\n", res00, res01, res02, res03);
    printf("%f %f %f %f\n", res10, res11, res12, res13);
    printf("%f %f %f %f\n", res20, res21, res22, res23);
    printf("%f %f %f %f\n", res30, res31, res32, res33);
    
    printf("Elapsed time: %f seconds\n", end_t - start_t);
    return 0;
}
