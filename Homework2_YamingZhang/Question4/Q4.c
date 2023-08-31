//g++ -pthread dart.c -o dart && ./dart 16 1000000
#include <cstdlib>

#include <stdio.h>

#include <pthread.h>

#include <semaphore.h>

#include <time.h>

int threadNum = 24;
int dartNum = 10000;
int radius = 1;
int inCircle = 0;
sem_t mutex;

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

void * dartThrow(void * arg) {
    for (int i = 0; i < dartNum / threadNum; i++) {
        if (isInCircle(randDouble(), randDouble())) {
            sem_wait(&mutex);
            inCircle++;
            sem_post(&mutex);
        }
    }
}

int main(int argc, char * argv[]) {
    if (argc >= 3) {
        threadNum = atoi(argv[1]);
        dartNum = atoi(argv[2]);
    }
    printf("Using %d threads and %d darts...\n", threadNum, dartNum);

    srand(time(NULL));

    pthread_t threads[threadNum];

    int i;
    sem_init(&mutex, 0, 1);
    double start = CLOCK();
    for (i = 0; i < threadNum; i++) {
        pthread_create( & threads[i], NULL, dartThrow, (void * ) NULL);
    }

    for (i = 0; i < threadNum; i++) {
        pthread_join(threads[i], NULL);
    }

    double finish = CLOCK();
    printf("Running time: %fms\n", finish - start);

    printf("Pi is %f \n", (double) inCircle / (dartNum / threadNum * threadNum) * 4);

    return 0;
}