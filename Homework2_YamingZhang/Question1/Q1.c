#include <cstdlib>

#include <stdio.h>

#include <unistd.h>

#include <pthread.h>

#include <semaphore.h>

// state
enum {
    THINKING = 0,
    HUNGRY = 1,
    EATING = 2
};

#define LEFT (i + threadNum - 1) % threadNum
#define RIGHT (i + 1) % threadNum

// default number of threads, which is number of philosophers in this case
int threadNum = 3;
// number of round as indicated in the requirement in the assignment is 12
int round = 12;
// mutex for getting forks try, which means only 1 philosopher could try to get or put forks
sem_t mutex;
// mutexs for each philosopher to determine whether they could eat now
sem_t * mutexs;
// philosopher states in enum value
int * state;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, & t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

// return random int among (1, 2, 3) for THINKING and EATING
double randInt() {
    return rand() % 3 + 1;
}

// try to test if the forks could be got
void test(int i) {
    if (state[i] == HUNGRY && state[LEFT] != EATING && state[RIGHT] != EATING) {
        state[i] = EATING;
        sem_post( & mutexs[i]);
    }
}

// try to take the left and right fork, if successful, eat!
void take_fork(int i) {
    // only 1 philosopher could get the mutex to try to get the forks
    sem_wait( & mutex);
    state[i] = HUNGRY;
    test(i);
    sem_post( & mutex);
    // if not get into the if statement in the test, here the thread(philosopher) will wait
    // until the adjacent philosopher release the forks
    sem_wait( & mutexs[i]);
}

// notify the LEFT and RIGHT philosopher to get the forks
void put_fork(int i) {
    sem_wait( & mutex);
    state[i] = THINKING;
    // notify the LEFT philosopher, if he is blocked and waiting, he has a chance to get the
    // forks now
    test(LEFT);
    // notify the RIGHT philosopher, the same
    test(RIGHT);
    sem_post( & mutex);
}

// just for print info and sleep(thinking or eating)
void doSomething(int i, int state, int round) {
    printf("Philosopher %d is %s for the %d (st/nd/rd/th) time\n", i, state == 0 ? "thinking" : "eating", round);
    sleep(randInt());
}

// entrance for multi threads, the philosopher is either thinking or eating
void * philosopher(void * num) {
    int * i = (int * ) num;
    int myRound = round;
    // 12 rounds as indicated in the requirement
    while (myRound-- > 0) {
        doSomething( * i, THINKING, round - myRound);
        take_fork( * i);
        doSomething( * i, EATING, round - myRound);
        put_fork( * i);
    }
}

int main(int argc, char * argv[]) {
    if (argc >= 2) {
        threadNum = atoi(argv[1]);
    }
    printf("There are %d philosophers on the table...\n", threadNum);

    srand(time(NULL));

    pthread_t threads[threadNum];
    // dynamic allocating cuz the threadNum is indicated on runtime
    int * threadNums = (int * ) malloc(threadNum * sizeof(int));
    mutexs = (sem_t * ) malloc(threadNum * sizeof(sem_t));
    state = (int * ) malloc(threadNum * sizeof(int));

    sem_init( & mutex, 0, 1);

    for (int i = 0; i < threadNum; i++) {
        sem_init( & mutexs[i], 0, 0);
        threadNums[i] = i;
    }

    double start = CLOCK();

    for (int i = 0; i < threadNum; i++) {
        pthread_create( & threads[i], NULL, philosopher, & threadNums[i]);
    }

    for (int i = 0; i < threadNum; i++) {
        pthread_join(threads[i], NULL);
    }

    double end = CLOCK();
    printf("Running time: %f ms\n", end - start);

    free(threadNums);
    free(mutexs);
    free(state);

    return 0;
}