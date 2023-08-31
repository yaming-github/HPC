#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 128
#define LOOPS 10

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

__global__ void mul(double *A, double *B, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x > 0 && x < n - 1 && y > 0 && y < n - 1 && z > 0 && z < n - 1) {
        A[x * n * n + y * n + z] = 0.8 *
                                   (B[(x - 1) * n * n + y * n + z] + B[(x + 1) * n * n + y * n + z]
                                    + B[x * n * n + (y - 1) * n + z] + B[x * n * n + (y + 1) * n + z]
                                    + B[x * n * n + y * n + z - 1] + B[x * n * n + y * n + z + 1]);
    }
}

int main() {
    int i, j, k, l;
    double start, finish, total;
    double *d_a;
    double *d_b;

    size_t bytes = N * N * N * sizeof(double);

    double *h_a = (double *) malloc(bytes);
    double *h_b = (double *) malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    /* initialize a dense matrix */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                h_b[i * N * N + j * N + k] = (double) (i + j + k);
            }
        }
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of blocks and threads in 3 dimension
    int blockNum = 16;
    dim3 gridSize(blockNum, blockNum, blockNum);
    dim3 blockSize(N / blockNum, N / blockNum, N / blockNum);

    printf("starting CUDA code... \n");
    start = CLOCK();
    for (l = 0; l < LOOPS; l++) {
        mul<<< gridSize, blockSize >>>(d_a, d_b, N);
    }
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    finish = CLOCK();
    total = finish - start;
    printf("a result %g \n", h_a[7 * N * N + 7 * N + 8]);
    printf("The total time = %f ms\n", total);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);

    // Release host memory
    free(h_a);
    free(h_b);
}
