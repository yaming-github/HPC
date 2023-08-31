#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512
#define LOOPS 10

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main() {
    double a[N][N]; /* input matrix */
    double b[N][N]; /* input matrix */
    double c[N][N] = {0}; /* result matrix */
    int i, j, k, kk, jj, l, num_zeros, block_size = 32;
    double start, finish, total, sum;


    /* initialize a dense matrix */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i][j] = (double) (i + j);
            b[i][j] = (double) (i - j);
        }
    }

    printf("starting dense matrix multiply \n");
    start = CLOCK();
    for (l = 0; l < LOOPS; l++) {
        for (kk = 0; kk < N; kk += block_size) {
            for (jj = 0; jj < N; jj += block_size) {
#pragma omp parallel for private(i, j, k, sum) shared(kk, jj, a, b, c)
                for (i = 0; i < N; i++) {
                    for (j = jj; j < jj + block_size; j++) {
                        sum = c[i][j];
                        for (k = kk; k < kk + block_size; k++) {
                            sum += a[i][k] * b[k][j];
                        }
                        c[i][j] = sum;
                    }
                }
            }
        }
    }
    finish = CLOCK();
    total = finish - start;
    printf("a result %g \n", c[7][8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with dense matrices = %f ms\n", total);

    /* initialize a sparse matrix */
    num_zeros = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if ((i < j) && (i % 2 > 0)) {
                a[i][j] = (double) (i + j);
                b[i][j] = (double) (i - j);
            } else {
                num_zeros++;
                a[i][j] = 0.0;
                b[i][j] = 0.0;
            }
            c[i][j] = 0;
        }
    }

    printf("starting sparse matrix multiply \n");
    start = CLOCK();
    for (l = 0; l < LOOPS; l++) {
        for (kk = 0; kk < N; kk += block_size) {
            for (jj = 0; jj < N; jj += block_size) {
#pragma omp parallel for private(i, j, k, sum) shared(kk, jj, a, b, c)
                for (i = 0; i < N; i++) {
                    for (j = jj; j < jj + block_size; j++) {
                        sum = c[i][j];
                        for (k = kk; k < kk + block_size; k++) {
                            sum += a[i][k] * b[k][j];
                        }
                        c[i][j] = sum;
                    }
                }
            }
        }
    }
    finish = CLOCK();
    total = finish - start;
    printf("A result %g \n", c[1][510]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with sparse matrices = %f ms\n", total);
    printf("The sparsity of the a and b matrices = %f \n", (float) num_zeros / (float) (N * N));

    return 0;
}

