// See if I can export a PyTorch model to low level.

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define byte unsigned char
#define max(x, y) ((x)>(y) ? (x) : (y))


float* load_weights(const char* path) {
    FILE* fp = fopen(path, "rb");
    fseek(fp, 0L, SEEK_END);
    int nw = ftell(fp) / sizeof(float);
    rewind(fp);
    float* w = malloc(sizeof(float) * nw);
    fread(w, sizeof(float), nw, fp);
    fclose(fp);
    printf("%d weights loaded from %s.\n", nw, path);
    return w;
}


#define M_IMAGES 8
#define N_IMAGES (M_IMAGES*M_IMAGES)


int main() {

    // load weights
    float* w01 = load_weights("raw-weights-64x64/w01_256_32.bin");
    float* w02 = load_weights("raw-weights-64x64/w02_16_16_3_3.bin");
    float* w03 = load_weights("raw-weights-64x64/w03_16_16_3_3.bin");
    float* w04 = load_weights("raw-weights-64x64/w04_8_16_5_5.bin");
    float* w05 = load_weights("raw-weights-64x64/w05_3_8_5_5.bin");
    printf("All weights loaded.\n");

    // latent - N_IMAGES x 32
    float* z = malloc(N_IMAGES * 32 * sizeof(float));
    for (int imgi = 0; imgi < N_IMAGES; imgi++) {
        float* imgp = &z[imgi * 32];
        for (int i = 0; i < 32; i++) {
            // initialize to standard normal distribution
            float u1 = (float)rand() / (float)RAND_MAX;
            float u2 = (float)rand() / (float)RAND_MAX;
            imgp[i] = sqrt(-2.0 * log(1.0 - u1)) * sin(6.283185 * u2);
        }
    }

    // first layer - (32) => (256) => (16, 4, 4)
    float *l01 = malloc(N_IMAGES * 256 * sizeof(float));
    for (int imgi = 0; imgi < N_IMAGES; imgi++) {
        float* imgsrc = &z[imgi * 32];
        float* imgres = &l01[imgi * 256];
        for (int i = 0; i < 256; i++) {
            float s = 0.0;
            for (int j = 0; j < 32; j++) {
                s += w01[i * 32 + j] * imgsrc[j];
            }
            imgres[i] = max(0.1 * s, s);
        }
    }

    // second layer - (16, 4, 4) => (16, 8, 8), 3x3 filter
    float *l02 = malloc(N_IMAGES * 1024 * sizeof(float));
    for (int imgi = 0; imgi < N_IMAGES; imgi++) {
        float* imgsrc = &l01[imgi * 256];
        float* imgres = &l02[imgi * 1024];
        for (int chr = 0; chr < 16; chr++) {  // res channel
            for (int ri = 0; ri < 8; ri++) for (int rj = 0; rj < 8; rj++) {  // res pixel
                float s = 0.0;
                for (int chs = 0; chs < 16; chs++) {  // src channel
                    float* w = &w02[(chr * 16 + chs) * 9];
                    for (int ci = 0; ci < 3; ci++) for (int cj = 0; cj < 3; cj++) {  // filter
                        int si = ri + ci - 1, sj = rj + cj - 1;
                        if (si >= 0 && si < 8 && sj >= 0 && sj < 8) {
                            si /= 2, sj /= 2;  // reverse upscale
                            float wv = w[ci * 3 + cj];
                            float sv = imgsrc[chs * 16 + (si * 4 + sj)];
                            s += wv * sv;
                        }
                    }
                }
                imgres[chr * 64 + (ri * 8 + rj)] = max(0.1 * s, s);;
            }
        }
    }

    // third layer - (16, 8, 8) -> (16, 16, 16), 3x3 filter
    float *l03 = malloc(N_IMAGES * 4096 * sizeof(float));
    for (int imgi = 0; imgi < N_IMAGES; imgi++) {
        float* imgsrc = &l02[imgi * 1024];
        float* imgres = &l03[imgi * 4096];
        for (int chr = 0; chr < 16; chr++) {  // res channel
            for (int ri = 0; ri < 16; ri++) for (int rj = 0; rj < 16; rj++) {  // res pixel
                float s = 0.0;
                for (int chs = 0; chs < 16; chs++) {  // src channel
                    float* w = &w03[(chr * 16 + chs) * 9];
                    for (int ci = 0; ci < 3; ci++) for (int cj = 0; cj < 3; cj++) {  // filter
                        int si = ri + ci - 1, sj = rj + cj - 1;
                        if (si >= 0 && si < 16 && sj >= 0 && sj < 16) {
                            si /= 2, sj /= 2;  // reverse upscale
                            float wv = w[ci * 3 + cj];
                            float sv = imgsrc[chs * 64 + (si * 8 + sj)];
                            s += wv * sv;
                        }
                    }
                }
                imgres[chr * 256 + (ri * 16 + rj)] = max(0.1 * s, s);;
            }
        }
    }

    // fourth layer - (16, 16, 16) -> (8, 32, 32), 5x5 filter
    float *l04 = malloc(N_IMAGES * 8192 * sizeof(float));
    for (int imgi = 0; imgi < N_IMAGES; imgi++) {
        float* imgsrc = &l03[imgi * 4096];
        float* imgres = &l04[imgi * 8192];
        for (int chr = 0; chr < 8; chr++) {  // res channel
            for (int ri = 0; ri < 32; ri++) for (int rj = 0; rj < 32; rj++) {  // res pixel
                float s = 0.0;
                for (int chs = 0; chs < 16; chs++) {  // src channel
                    float* w = &w04[(chr * 16 + chs) * 25];
                    for (int ci = 0; ci < 5; ci++) for (int cj = 0; cj < 5; cj++) {  // filter
                        int si = ri + ci - 2, sj = rj + cj - 2;
                        if (si >= 0 && si < 32 && sj >= 0 && sj < 32) {
                            si /= 2, sj /= 2;  // reverse upscale
                            float wv = w[ci * 5 + cj];
                            float sv = imgsrc[chs * 256 + (si * 16 + sj)];
                            s += wv * sv;
                        }
                    }
                }
                imgres[chr * 1024 + (ri * 32 + rj)] = max(0.1 * s, s);;
            }
        }
    }

    // fifth layer - (8, 32, 32) -> (3, 64, 64), 5x5 filter
    float *l05 = malloc(N_IMAGES * 12288 * sizeof(float));
    for (int imgi = 0; imgi < N_IMAGES; imgi++) {
        float* imgsrc = &l04[imgi * 8192];
        float* imgres = &l05[imgi * 12288];
        for (int chr = 0; chr < 3; chr++) {  // res channel
            for (int ri = 0; ri < 64; ri++) for (int rj = 0; rj < 64; rj++) {  // res pixel
                float s = 0.0;
                for (int chs = 0; chs < 8; chs++) {  // src channel
                    float* w = &w05[(chr * 8 + chs) * 25];
                    for (int ci = 0; ci < 5; ci++) for (int cj = 0; cj < 5; cj++) {  // filter
                        int si = ri + ci - 2, sj = rj + cj - 2;
                        if (si >= 0 && si < 64 && sj >= 0 && sj < 64) {
                            si /= 2, sj /= 2;  // reverse upscale
                            float wv = w[ci * 5 + cj];
                            float sv = imgsrc[chs * 1024 + (si * 32 + sj)];
                            s += wv * sv;
                        }
                    }
                }
                imgres[chr * 4096 + (ri * 64 + rj)] = 1.0 / (1.0 + exp(-s));
            }
        }
    }

    // output images
    byte* image = malloc(N_IMAGES * 12288);
    for (int mi = 0; mi < M_IMAGES; mi++) {
        for (int mj = 0; mj < M_IMAGES; mj++) {
            float* imgsrc = &l05[(mi * M_IMAGES + mj) * 12288];
            for (int i = 0; i < 64; i++)  for (int j = 0; j < 64; j++) {
                for (int c = 0; c < 3; c++) {
                    byte v = (byte)(imgsrc[c * 4096 + i * 64 + j] * 255.0 + 0.5);
                    image[((mi * 64 + i) * (M_IMAGES * 64) + (mj * 64 + j)) * 3 + c] = v;
                }
            }
        }
    }
    stbi_write_png("generator_64x64.c.png",
        M_IMAGES * 64, M_IMAGES * 64,
        3, image, 3 * M_IMAGES * 64);

    return 0;
}
