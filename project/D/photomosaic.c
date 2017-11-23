#include "photomosaic.h"

#include <limits.h>

void photomosaic(unsigned char *img, int width, int height, unsigned char *dataset, int *idx) {
    int swidth = width / 32, sheight = height / 32;
    for (int sh = 0; sh < sheight; ++sh) {
        for (int sw = 0; sw < swidth; ++sw) {
            int min_diff = INT_MAX, min_i = -1;
            for (int i = 0; i < 60000; ++i) {
                int diff = 0;
                for (int h = 0; h < 32; ++h) {
                    for (int w = 0; w < 32; ++w) {
                        for (int c = 0; c < 3; ++c) {
                            int pixel_diff = (int)img[((sh * 32 + h) * width + (sw * 32 + w)) * 3 + c] - (int)dataset[((i * 3 + c) * 32 + h) * 32 + w];
                            diff += pixel_diff * pixel_diff;
                        }
                    }
                }
                if (min_diff > diff) {
                    min_diff = diff;
                    min_i = i;
                }
            }
            idx[sh * swidth + sw] = min_i;
        }
    }
}
