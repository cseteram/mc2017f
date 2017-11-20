#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

#define N (1 << 26)

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

int A[N], B[N], C[N];

int main()
{
    int i;

    /* Initialize */
    for (i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    double start_time = get_time();
 
    /* Parallel section */
    #pragma omp parallel shared(A, B, C) private(i)
    {
        #pragma omp for schedule(static) nowait
        for (i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
    }

    double end_time = get_time();

    /* Verification */
    for (i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) {
            printf("Incorrect (i = %d : %d != %d + %d)\n",
                i, C[i], A[i], B[i]);
            break;
        }
    }

    printf("Elasped time: %f seconds\n", end_time - start_time);
    printf("Finished!\n");
    return 0;
}
