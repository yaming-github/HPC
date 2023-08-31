#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>

using namespace std;
#define R 10000000
#define GRID_SIZE 8
#define BLOCK_SIZE 8

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int *get_random_data(long long N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> value(1, R);

    int *data = (int *) malloc(sizeof(int) * N);
    for (int i = 0; i < N; ++i) data[i] = value(gen);

    return data;
}

__global__ void histogram(int *A, long long *B, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    long long sum = 0;
    for (int i = x * n; i < (x + 1) * n; i++) sum += A[i];
    B[x] = sum;
}

int main() {
    long long N = (long long) pow(2, 10);
    int i;
    double start, finish, total;
    int *d_a;
    long long *d_b;

    size_t int_bytes = N * sizeof(int);
    size_t long_long_bytes = GRID_SIZE * BLOCK_SIZE * sizeof(long long);

    int *h_a = get_random_data(N);
    long long *h_b = (long long *) malloc(long_long_bytes);

    cudaMalloc(&d_a, int_bytes);
    cudaMalloc(&d_b, long_long_bytes);

    cudaMemcpy(d_a, h_a, int_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, long_long_bytes, cudaMemcpyHostToDevice);

    printf("starting CUDA code... \n");
    start = CLOCK();
    histogram<<< GRID_SIZE, BLOCK_SIZE >>>(d_a, d_b, N / GRID_SIZE / BLOCK_SIZE);
    cudaMemcpy(h_b, d_b, long_long_bytes, cudaMemcpyDeviceToHost);
    finish = CLOCK();
    total = finish - start;
    printf("The total time = %f ms\n", total);
    for (i = 0; i < GRID_SIZE * BLOCK_SIZE; i++)
        printf("The sum of class %d is %lld\n", i, h_b[i]);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);

    // Release host memory
    free(h_a);
    free(h_b);
}
