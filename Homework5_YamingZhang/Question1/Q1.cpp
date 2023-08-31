#include <stdio.h>
#include <random>
#include <omp.h>

using namespace std;
#define R 10000000
#define THREAD_NUM 64

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int *get_random_data(int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> value(1, R);

    int *data = (int *) malloc(sizeof(int) * N);
    for (int i = 0; i < N; ++i) data[i] = value(gen);

    return data;
}

int main() {
    long long N = (long long) pow(2, 25);
    int *data = get_random_data(N);
    int i;
    long long sum = 0;

    double start = CLOCK();
#pragma omp parallel num_threads(THREAD_NUM) reduction(+:sum)
    {
#pragma omp for private(i)
        for (i = 0; i < N; i++) {
            sum += data[i];
        }
        printf("ThreadID = %d; Sum = %lld\n", omp_get_thread_num(), sum);
    }
    printf("Total sum: %lld\n", sum);
    double duration = CLOCK() - start;
    printf("Finished in %f ms\n", duration);

    free(data);
}
