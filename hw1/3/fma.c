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
    double *res;

    __m256d result0 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
    __m256d veca0 = _mm256_setr_pd(1.5, 1.2, 2.0, 1.8);
    __m256d vecb0 = _mm256_setr_pd(0.7, 0.8, 0.5, 0.6);

    __m256d result1 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
    __m256d veca1 = _mm256_setr_pd(1.3, 1.4, 1.1, 2.2);
    __m256d vecb1 = _mm256_setr_pd(0.9, 0.7, 0.8, 0.4);

    __m256d result2 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
    __m256d veca2 = _mm256_setr_pd(1.6, 1.9, 2.1, 1.4);
    __m256d vecb2 = _mm256_setr_pd(0.6, 0.6, 0.5, 0.8);

    __m256d result3 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
    __m256d veca3 = _mm256_setr_pd(1.1, 2.4, 1.8, 1.6);
    __m256d vecb3 = _mm256_setr_pd(0.9, 0.4, 0.6, 0.4);

    start_t = gettime();
    for (int i = 0; i < N; i++) {
      result0 = _mm256_fmadd_pd(veca0, vecb0, result0);
      result1 = _mm256_fmadd_pd(veca1, vecb1, result1);
      result2 = _mm256_fmadd_pd(veca2, vecb2, result2);
      result3 = _mm256_fmadd_pd(veca3, vecb3, result3);
    }
    end_t = gettime();
  
    res = (double*)&result0;
    printf("%f %f %f %f\n", res[0], res[1], res[2], res[3]);
    res = (double*)&result1;
    printf("%f %f %f %f\n", res[0], res[1], res[2], res[3]);
    res = (double*)&result2;
    printf("%f %f %f %f\n", res[0], res[1], res[2], res[3]);
    res = (double*)&result3;
    printf("%f %f %f %f\n", res[0], res[1], res[2], res[3]);

    printf("Elapsed time: %f seconds\n", end_t - start_t);
    return 0;
}
