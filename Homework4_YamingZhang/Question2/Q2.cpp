#include <stdio.h>
#include <random>
#include <mpi.h>

using namespace std;
#define R 1000000
#define N 2000000

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int *get_random_data() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> value(1, R);

    int *data = (int *) malloc(sizeof(int) * N);
    for (int i = 0; i < N; ++i) data[i] = value(gen);

    return data;
}

int main() {
    int world_size;
    int world_rank;
    int *data;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        data = get_random_data();
    }

    int num_per_bin = N / world_size;
    int *bin_data = (int *) malloc(sizeof(int) * num_per_bin);

    double start = CLOCK();

    MPI_Scatter(data, num_per_bin, MPI_INT, bin_data, num_per_bin, MPI_INT, 0, MPI_COMM_WORLD);\

    long long sum = 0;
    for (int i = 0; i < num_per_bin; ++i) sum += bin_data[i];
    printf("Process %d has %d values with total sum %lld\n", world_rank, num_per_bin, sum);

    long long total_sum;
    MPI_Reduce(&sum, &total_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double duration = CLOCK() - start;
    if (world_rank == 0) printf("Total sum = %lld; Finished in %f ms\n", total_sum, duration);

    free(bin_data);
    if (world_rank == 0) free(data);

    MPI_Finalize();
}
