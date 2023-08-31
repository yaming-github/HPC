#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 512
#define LOOPS 10

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

__global__ void matmul(double *A, double *B, double *C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        double tmp = 0;
        for (int i = 0; i < n; i++) {
            tmp += A[r * n + i] * B[i * n + c];
        }
        C[r * n + c] = tmp;
    }
}

int main() {
    int i, j, l;
    double start, finish, total;
    double *d_a;
    double *d_b;
    double *d_c;

    size_t bytes = N * N * sizeof(double);

    double *h_a = (double *) malloc(bytes);
    double *h_b = (double *) malloc(bytes);
    double *h_c = (double *) malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    /* initialize a dense matrix */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            h_a[i * N + j] = (double) (i + j);
            h_b[i * N + j] = (double) (i - j);
        }
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of blocks and threads in 3 dimension
    dim3 gridSize(16, 16);
    dim3 blockSize(32, 32);

    printf("starting dense matrix multiply \n");
    start = CLOCK();
    for (l = 0; l < LOOPS; l++) {
        matmul<<< gridSize, blockSize >>>(d_a, d_b, d_c, N);
    }
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    finish = CLOCK();
    total = finish - start;
    printf("a result %g \n", h_c[7 * N + 8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with dense matrices = %f ms\n", total);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
}
