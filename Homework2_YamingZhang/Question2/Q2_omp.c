//g++ -fopenmp dart.c -o dart && ./dart 1000000
#include <cstdlib>

#include <stdio.h>

#include <omp.h>

#include <time.h>

int threadNum = 24;
int dartNum = 10000;
int radius = 1;
int inCircle;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, & t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

double randDouble() {
    return radius * -1 + (double) rand() * radius * 2 / RAND_MAX;
}

bool isInCircle(double i, double j) {
    return i * i + j * j <= radius;
}

int main(int argc, char * argv[]) {
    if (argc >= 3) {
		threadNum = atoi(argv[1]);
		dartNum = atoi(argv[2]);
    }
    printf("Using %d threads and %d darts...\n", threadNum, dartNum);
    srand(time(NULL));

    int i;
    omp_set_num_threads(threadNum);
    double start = CLOCK();
    #pragma omp parallel shared(threadNum) private(i) reduction(+: inCircle)
    {
        #pragma omp parallel for reduction(+: inCircle)
        for (i = 0; i < dartNum / threadNum; i++) {
            if (isInCircle(randDouble(), randDouble())) {
                inCircle++;
            }
        }
    }
    double finish = CLOCK();
    printf("Running time: %fms\n", finish - start);

    printf("Pi is %f \n", (double) inCircle / (dartNum / threadNum * threadNum) * 4);

    return 0;
}