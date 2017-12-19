#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#include "photomosaic.h"
#include "timer.h"

typedef unsigned char uchar;
#define NUM_THREADS 32
#define TSIZE 16

void photomosaic(unsigned char *img, int width, int height, unsigned char *dataset, int *idx) {
    int swidth = width / 32, sheight = height / 32;
    const int P = 60416;
    const int Q = 3 * 32 * 32;
    const int R = (swidth * sheight + TSIZE - 1) / TSIZE * TSIZE;
    
    timer_start(1);
    int *diff_all = (int*)malloc(sizeof(int) * P * R);
    int *min_diff = (int*)malloc(sizeof(int) * R);
    uchar *img_t = (uchar*)malloc(sizeof(uchar) * Q * R);
    uchar *dataset_p = (uchar*)malloc(sizeof(uchar) * P * Q);
    printf("P = %d, Q = %d, R = %d\n", P, Q, R);
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp for collapse(2) nowait
        for (int sh = 0; sh < sheight; ++sh) {
            for (int sw = 0; sw < swidth; ++sw) {
                for (int h = 0; h < 32; ++h) {
                    for (int w = 0; w < 32; ++w) {
                        for (int c = 0; c < 3; ++c) {
                            img_t[(c * 32 * 32 + h * 32 + w) * R + (sh * swidth + sw)] = img[(sh * 32 + h) * width * 3 + (sw * 32 + w) * 3 + c];
                        }
                    }
                }
            }
        }

        #pragma omp for schedule(guided) collapse(2) nowait
        for (int i = 0; i < Q; ++i) {
            for (int j = swidth * sheight; j < R; ++j) {
                img_t[i * R + j] = 0;
            }
        }
        #pragma omp for schedule(guided) nowait
        for (int i = 0; i < 60000 * Q; ++i)
            dataset_p[i] = dataset[i];
        #pragma omp for schedule(guided) nowait
        for (int i = 60000 * Q; i < P * Q; ++i)
            dataset_p[i] = 0;
        #pragma omp for schedule(guided) nowait
        for (int i = 0; i < R; ++i)
            min_diff[i] = INT_MAX;
    }
    printf("\nprepare diff_all & img_t & dataset_p: %f seconds\n", timer_stop(1));

    timer_start(2);
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int asub[TSIZE][TSIZE], bsub[TSIZE][TSIZE], csub[TSIZE][TSIZE];
        const int ii_chunk = P / TSIZE / NUM_THREADS;
        const int NUM_TILES = Q / TSIZE;
    
        int id = omp_get_thread_num();
        for (int ii = id * ii_chunk; ii < (id + 1) * ii_chunk; ++ii) {
            for (int jj = 0; jj < R / TSIZE; ++jj) {
                // init csub <- 0
                memset(csub, 0, sizeof(csub));

                for (int t = 0; t < NUM_TILES; ++t) {
                    // load asub, bsub <- a, b
                    for (int i = 0; i < TSIZE; ++i) {
                        for (int j = 0; j < TSIZE; ++j) {
                            asub[i][j] = (int)dataset_p[(ii * TSIZE + i) * Q + (t * TSIZE + j)];
                            bsub[i][j] = (int)img_t[(t * TSIZE + i) * R + (jj * TSIZE + j)];
                        }
                    }
                    // calculate csub
                    for (int i = 0; i < TSIZE; ++i) {
                        for (int j = 0; j < TSIZE; ++j) {
                            for (int k = 0; k < TSIZE; ++k) {
                                int a = asub[i][k];
                                int b = bsub[k][j];
                                csub[i][j] += (a - b) * (a - b);
                            }
                        }
                    }
                }

                // store csub -> c
                for (int i = 0; i < TSIZE; ++i) {
                    for (int j = 0; j < TSIZE; ++j) {
                        size_t li = ii * TSIZE + i;
                        size_t lj = jj * TSIZE + j;
                        diff_all[li * R + lj] = csub[i][j];
                    }
                }
            }
        }
    }
    printf("mat_mul: %f seconds\n", timer_stop(2));

    timer_start(3);
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp for schedule(guided)
        for (int i = 0; i < 60000; ++i) {
            for (int sh = 0; sh < sheight; ++sh) {
                for (int sw = 0; sw < swidth; ++sw) {
                    size_t li = i;
                    size_t lj = sh * swidth + sw;

                    int diff = diff_all[li * R + lj];
                    if (diff < min_diff[lj]) {
                        min_diff[lj] = diff;
                        idx[lj] = i;
                    }
                }
            }
        }
    }
    printf("set idx: %f seconds\n\n", timer_stop(3));

    free(diff_all);
    free(min_diff);
    free(img_t);
    free(dataset_p);
}
