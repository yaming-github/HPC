#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 128
#define LOOPS 10
#define X_TILE_SIZE 32
#define Y_TILE_SIZE 32
#define Z_TILE_SIZE 2
#define RADIUS 1

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

__device__ int l_offset(int x, int y, int z, int r) {
    int dimX = 2 * r + blockDim.x;
    int dimY = 2 * r + blockDim.y;
    return (z + r) * dimX * dimY + (y + r) * dimX + x + r;
}

__global__ void tiling_mul(double *a, double *b, int n) {
    extern __shared__ float s[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int local_idx = l_offset(threadIdx.x, threadIdx.y, threadIdx.z, RADIUS);

    s[local_idx] = b[x * n * n + y * n + z];

    if (x < 1 || x > n - 2 || y < 1 || y > n - 2 || z < 1 || z > n - 2) return;

    if (threadIdx.x < RADIUS) {
        int l_idx1 = l_offset(threadIdx.x - RADIUS, threadIdx.y, threadIdx.z, RADIUS);
        int g_idx1 = (x - RADIUS) * n * n + y * n + z;

        s[l_idx1] = b[g_idx1];

        if (blockIdx.x < gridDim.x - 1) {
            int l_idx2 = l_offset(threadIdx.x + blockDim.x, threadIdx.y, threadIdx.z, RADIUS);
            int g_idx2 = (x + blockDim.x) * n * n + y * n + z;
            s[l_idx2] = b[g_idx2];
        }
    }

    if (threadIdx.y < RADIUS) {
        int l_idx1 = l_offset(threadIdx.x, threadIdx.y - RADIUS, threadIdx.z, RADIUS);
        int g_idx1 = x * n * n + (y - RADIUS) * n + z;

        s[l_idx1] = b[g_idx1];

        if (blockIdx.y < gridDim.y - 1) {
            int l_idx2 = l_offset(threadIdx.x, threadIdx.y + blockDim.y, threadIdx.z, RADIUS);
            int g_idx2 = x * n * n + (y + blockDim.y) * n + z;
            s[l_idx2] = b[g_idx2];
        }
    }

    if (threadIdx.z < RADIUS) {
        int l_idx1 = l_offset(threadIdx.x, threadIdx.y, threadIdx.z - RADIUS, RADIUS);
        int g_idx1 = x * n * n + y * n + z - RADIUS;

        s[l_idx1] = b[g_idx1];

        if (blockIdx.z < gridDim.z - 1) {
            int l_idx2 = l_offset(threadIdx.x, threadIdx.y, threadIdx.z + blockDim.z, RADIUS);
            int g_idx2 = x * n * n + y * n + z + blockDim.z;
            s[l_idx2] = b[g_idx2];
        }
    }

    __syncthreads();

    a[x * n * n + y * n + z] = 0.8 *
                               (s[l_offset(threadIdx.x - 1, threadIdx.y, threadIdx.z, RADIUS)]
                                + s[l_offset(threadIdx.x + 1, threadIdx.y, threadIdx.z, RADIUS)]
                                + s[l_offset(threadIdx.x, threadIdx.y - 1, threadIdx.z, RADIUS)]
                                + s[l_offset(threadIdx.x, threadIdx.y + 1, threadIdx.z, RADIUS)]
                                + s[l_offset(threadIdx.x, threadIdx.y, threadIdx.z - 1, RADIUS)]
                                + s[l_offset(threadIdx.x, threadIdx.y, threadIdx.z + 1, RADIUS)]);
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
    int xGrid = N / X_TILE_SIZE;
    int yGrid = N / Y_TILE_SIZE;
    int zGrid = N / Z_TILE_SIZE;

    dim3 gridSize(X_TILE_SIZE, Y_TILE_SIZE, Z_TILE_SIZE);
    dim3 blockSize(xGrid, yGrid, zGrid);

    int shared_block_size = (X_TILE_SIZE + (2 * RADIUS))
                            * (Y_TILE_SIZE + (2 * RADIUS))
                            * (Z_TILE_SIZE + (2 * RADIUS)) * sizeof(double);

    printf("starting CUDA code... \n");
    start = CLOCK();
    for (l = 0; l < LOOPS; l++) {
        tiling_mul<<< gridSize, blockSize, shared_block_size >>>(d_a, d_b, N);
    }
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
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
