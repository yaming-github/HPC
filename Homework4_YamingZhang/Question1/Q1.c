#include <stdio.h>
#include <mpi.h>

int main() {
    int world_rank;
    int world_size;
    int pass_value = 0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        pass_value++;
        printf("Process %d has incremented pass_value to %d and is now sending...\n", world_rank, pass_value);
        MPI_Send(&pass_value, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == world_size - 1) {
        MPI_Recv(&pass_value, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d has got pass_value %d\n", world_rank, pass_value);
    } else {
        MPI_Recv(&pass_value, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d has got pass_value %d\n", world_rank, pass_value);
        pass_value++;
        printf("Process %d has incremented pass_value to %d and is now sending...\n", world_rank, pass_value);
        MPI_Send(&pass_value, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
    }

    if (world_rank == world_size - 1) {
        pass_value--;
        printf("Process %d has decremented pass_value to %d and is now sending...\n", world_rank, pass_value);
        MPI_Send(&pass_value, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 0) {
        MPI_Recv(&pass_value, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d has got pass_value %d\n", world_rank, pass_value);
    } else {
        MPI_Recv(&pass_value, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d has got pass_value %d\n", world_rank, pass_value);
        pass_value--;
        printf("Process %d has decremented pass_value to %d and is now sending...\n", world_rank, pass_value);
        MPI_Send(&pass_value, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
