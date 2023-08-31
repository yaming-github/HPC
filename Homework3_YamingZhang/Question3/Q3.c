#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#define N 512
#define LOOPS 10

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main() {
    int i, j, k, kk, jj, l, num_zeros, block_size = 32;
    double start, finish, total, sum;
    double *a = (double *) malloc(N * N * sizeof(double));
    double *b = (double *) malloc(N * N * sizeof(double));
    double *c = (double *) malloc(N * N * sizeof(double));

    /* initialize a dense matrix */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i * N + j] = (double) (i + j);
            b[i * N + j] = (double) (i - j);
            c[i * N + j] = 0;
        }
    }

    printf("starting dense matrix multiply \n");
    start = CLOCK();
    for (l = 0; l < LOOPS; l++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, a,
                    N, b, N, 0.0,
                    c, N);
    }
    finish = CLOCK();
    total = finish - start;
    printf("a result %g \n", c[7 * N + 8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with dense matrices = %f ms\n", total);

    /* initialize a sparse matrix */
    num_zeros = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if ((i < j) && (i % 2 > 0)) {
                a[i * N + j] = (double) (i + j);
                b[i * N + j] = (double) (i - j);
            } else {
                num_zeros++;
                a[i * N + j] = 0.0;
                b[i * N + j] = 0.0;
            }
            c[i * N + j] = 0;
        }
    }

    printf("starting sparse matrix multiply \n");
    start = CLOCK();
    for (l = 0; l < LOOPS; l++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, a,
                    N, b, N, 0.0,
                    c, N);
    }
    finish = CLOCK();
    total = finish - start;
    printf("A result %g \n", c[1 * N + 510]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with sparse matrices = %f ms\n", total);
    printf("The sparsity of the a and b matrices = %f \n", (float) num_zeros / (float) (N * N));

    free(a);
    free(b);
    free(c);

    return 0;
}

