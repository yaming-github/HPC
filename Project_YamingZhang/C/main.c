#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>
#include <cblas.h>

#define VECTOR_N 1000000
#define N 512
#define BLOCK_SIZE 16
#define THREAD_NUM 10
double *a, *b, *c;
int part = 0;
pthread_mutex_t mutex;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

void *vecAdd(void *arg) {
    pthread_mutex_lock(&mutex);
    int thread_part = part++;
    pthread_mutex_unlock(&mutex);

    int i;
    for (i = thread_part * VECTOR_N / THREAD_NUM; i < (thread_part + 1) * VECTOR_N / THREAD_NUM; i++)
        c[i] = a[i] + b[i];
}

int main() {
    int i, j, k, kk, jj;
    double start, duration, sum;

    a = (double *) malloc(VECTOR_N * sizeof(double));
    b = (double *) malloc(VECTOR_N * sizeof(double));
    c = (double *) malloc(VECTOR_N * sizeof(double));

    for (i = 0; i < VECTOR_N; i++) {
        a[i] = sin(i) * sin(i);
        b[i] = cos(i) * cos(i);
    }

    start = CLOCK();
    for (i = 0; i < VECTOR_N; i++) {
        c[i] = a[i] + b[i];
    }
    duration = CLOCK() - start;
    sum = 0;
    for (i = 0; i < VECTOR_N; i++) {
        sum += c[i];
    }
    printf("C VECTOR: %f\n", sum / VECTOR_N);
    printf("C VECTOR: %f ms\n", duration);
    for (i = 0; i < VECTOR_N; i++) {
        c[i] = 0;
    }

    start = CLOCK();
    pthread_t threads[THREAD_NUM];

    for (i = 0; i < THREAD_NUM; i++) {
        pthread_create(&threads[i], NULL, vecAdd, (void *) NULL);
    }

    for (i = 0; i < THREAD_NUM; i++) {
        pthread_join(threads[i], NULL);
    }
    duration = CLOCK() - start;
    sum = 0;
    for (i = 0; i < VECTOR_N; i++) {
        sum += c[i];
    }
    printf("C VECTOR PTHREADS: %f\n", sum / VECTOR_N);
    printf("C VECTOR PTHREADS: %f ms\n", duration);
    for (i = 0; i < VECTOR_N; i++) {
        c[i] = 0;
    }

    start = CLOCK();
#pragma omp parallel num_threads(THREAD_NUM)
    {
#pragma omp for
        for (i = 0; i < VECTOR_N; i++) {
            c[i] = a[i] + b[i];
        }
    }
    duration = CLOCK() - start;
    sum = 0;
    for (i = 0; i < VECTOR_N; i++) {
        sum += c[i];
    }
    printf("C VECTOR OMP: %f\n", sum / VECTOR_N);
    printf("C VECTOR OMP: %f ms\n", duration);

    double aa[N][N];
    double bb[N][N];
    double cc[N][N] = {0};

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            aa[i][j] = (double) (i + j);
            bb[i][j] = (double) (i - j);
        }
    }

    start = CLOCK();
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = cc[i][j];
            for (k = 0; k < N; k++) {
                sum += aa[i][k] * bb[k][j];
            }
            cc[i][j] = sum;
        }
    }
    duration = CLOCK() - start;
    printf("C MATMUL: %g \n", cc[7][8]);
    printf("C MATMUL: %f ms\n", duration);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            cc[i][j] = 0;
        }
    }

    start = CLOCK();
    for (kk = 0; kk < N; kk += BLOCK_SIZE) {
        for (jj = 0; jj < N; jj += BLOCK_SIZE) {
#pragma omp parallel for private(i, j, k, sum) shared(kk, jj, a, b, c)
            for (i = 0; i < N; i++) {
                for (j = jj; j < jj + BLOCK_SIZE; j++) {
                    sum = cc[i][j];
                    for (k = kk; k < kk + BLOCK_SIZE; k++) {
                        sum += aa[i][k] * bb[k][j];
                    }
                    cc[i][j] = sum;
                }
            }
        }
    }
    duration = CLOCK() - start;
    printf("C MATMUL OMP: %g \n", cc[7][8]);
    printf("C MATMUL OMP: %f ms\n", duration);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            cc[i][j] = 0;
        }
    }

    double *aaa = (double *) malloc(N * N * sizeof(double));
    double *bbb = (double *) malloc(N * N * sizeof(double));
    double *ccc = (double *) malloc(N * N * sizeof(double));

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            aaa[i * N + j] = (double) (i + j);
            bbb[i * N + j] = (double) (i - j);
            ccc[i * N + j] = 0;
        }
    }

    start = CLOCK();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, aaa,
                N, bbb, N, 0.0,
                ccc, N);
    duration = CLOCK() - start;
    printf("C MATMUL BLAS: %g \n", ccc[7 * N + 8]);
    printf("C MATMUL BLAS: %f ms\n", duration);

    free(a);
    free(b);
    free(c);
    free(aaa);
    free(bbb);
    free(ccc);
}
