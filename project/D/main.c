#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "photomosaic.h"
#include "timer.h"
#include "qdbmp.h"

int main(int argc, char **argv) {
    MPI_Init(NULL, NULL);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (argc != 3) {
        if (rank == 0)   
            printf("Usage : %s [input.bmp] [output.bmp]\n", argv[0]);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    /*
     * read input image
     */
    BMP *bmp;
    int width = 0, height = 0, depth = 0;
    if (rank == 0) {
        bmp = BMP_ReadFile(argv[1]);
        if (BMP_GetError() != BMP_OK) {
            fprintf(stderr, "BMP error: %s\n", BMP_GetErrorDescription());
        }
        else {
            width = BMP_GetWidth(bmp);
            height = BMP_GetHeight(bmp);
            depth = BMP_GetDepth(bmp);
            printf("image read success; image = %s, width = %d, height = %d, depth = %d\n", argv[1], width, height, depth);
            if (width % 32 != 0 || height % 32 != 0) {
                printf("width and height should be multiple of 32.\n");
            }
            if (depth != 24) {
                printf("depth should be 24.\n");
            }
        }
    }

    int info[3] = {width, height, depth};
    MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
    width = info[0], height = info[1], depth = info[2];
    if (width % 32 != 0 || height % 32 != 0 || depth != 24) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    unsigned char *img = (unsigned char*)malloc(height * width * 3), *it = img;
    if (rank == 0) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                BMP_GetPixelRGB(bmp, j, i, it, it + 1, it + 2);
                it += 3;
            }
        }

        BMP_Free(bmp);
    }

    /*
     * read cifar-10 dataset
     */
    unsigned char *dataset = (unsigned char*)malloc(60000 * 3 * 32 * 32);
    FILE *fin = fopen("data/cifar-10.bin", "rb");
    if (!fin) {
        if (rank == 0)
            printf("cifar-10.bin not found\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    fread(dataset, 1, 60000 * 3 * 32 * 32, fin);
    fclose(fin);

    if (rank == 0)
        printf("dataset read success\n");

    /*
     * photomosaic computation
     */

    int swidth = width / 32, sheight = height / 32;
    int *idx = (int*)malloc(sheight * swidth * sizeof(int));
    if (rank == 0)
        timer_start(0);
    photomosaic(img, width, height, dataset, idx);
    if (rank == 0) 
        printf("Elapsed time: %f sec\n", timer_stop(0));

    /*
     * construct output image
     */
    if (rank == 0) {
        bmp = BMP_Create(width, height, depth);
        for (int sh = 0; sh < sheight; ++sh) {
            for (int sw = 0; sw < swidth; ++sw) {
                for (int h = 0; h < 32; ++h) {
                    for (int w = 0; w < 32; ++w) {
                        unsigned char rgb[3];
                        for (int c = 0; c < 3; ++c) {
                            rgb[c] = dataset[((idx[sh * swidth + sw] * 3 + c) * 32 + h) * 32 + w];
                        }
                        BMP_SetPixelRGB(bmp, sw * 32 + w, sh * 32 + h, rgb[0], rgb[1], rgb[2]);
                    }
                }
            }
        }
        BMP_WriteFile(bmp, argv[2]);
        BMP_Free(bmp);
        printf("image write success\n");
    }

    /*
     * free resources
     */
    if (rank == 0) {
        free(img);
        free(dataset);
        free(idx);
    }

    MPI_Finalize();
    return 0;
}
