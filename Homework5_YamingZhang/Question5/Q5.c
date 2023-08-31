#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main(int argc, char *argv[]) {
    int iteration = atoi(argv[1]);
    double start, finish, total;

    // Allocate memory for each vector on host
    float *h_a = (float *) malloc(iteration * sizeof(float));

    int i;

    start = CLOCK();

#pragma acc kernels
    for (i = 0; i < iteration; i++) {
        int s = 1;
        if (i % 2 == 1) s = -s;

        h_a[i] = (float) s * 4 / (i * 2 + 1);
    }

    // Sum up vector c and print result divided by n, this should equal 1 within error
    float sum = 0;
    for (i = 0; i < iteration; i++)
        sum += h_a[i];

    finish = CLOCK();

    printf("Iteration: %d; Pi: %F\n", iteration, sum);
    total = finish - start;
    printf("Time for the loop = %4.2f milliseconds\n", total);

    // Release host memory
    free(h_a);

    return 0;
}
