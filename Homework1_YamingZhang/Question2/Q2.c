// CITE: The code below for multi-threading merge sort is aspired by
// https://geeksforgeeks.org/merge-sort-using-multi-threading/
// CPP Program to implement merge sort using multi-threading
// script: g++ -std=c++11 -pthread merge_sort.c -o mergesort && ./mergesort 8 && rm -f mergesort
#include <cstdlib>

#include <iostream>

#include <random>

#include <chrono>

// number of elements in array
#define MAX 80000

// random maximum value
#define RANDOM_MAX 1000000

// number of printing items 4 certification
#define CERTIFY_NUM 20

using namespace std;

int threadNum = 1;
// array of size MAX
int a[MAX];
int part = 0;
pthread_mutex_t mutex;

// CITE: The code below for generate uniformly distributed random integers is aspired by
// https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
void genRandom() {
    std::mt19937 generator(time(NULL));
    std::uniform_int_distribution < int > distribution(0, RANDOM_MAX);
    for (int i = 0; i < MAX; i++) {
        a[i] = distribution(generator); // generate random int flat in [min, max]
    }
}

// merge function for merging two parts
void merge(int low, int mid, int high) {
    // n1 is size of left part and n2 is size of right part
    int n1 = mid - low + 1, n2 = high - mid, i, j, k;

    int * left = new int[n1];
    int * right = new int[n2];

    // storing values in left part
    for (i = 0; i < n1; i++) {
        left[i] = a[i + low];
    }

    // storing values in right part
    for (i = 0; i < n2; i++) {
        right[i] = a[i + mid + 1];
    }

    k = low;
    i = j = 0;

    // merge left and right in ascending order
    while (i < n1 && j < n2) {
        if (left[i] <= right[j]) {
            a[k++] = left[i++];
        } else {
            a[k++] = right[j++];
        }
    }

    // insert remaining values from left
    while (i < n1) {
        a[k++] = left[i++];
    }

    // insert remaining values from right
    while (j < n2) {
        a[k++] = right[j++];
    }
}

// merge sort function
void merge_sort(int low, int high) {
    // calculating mid point of array
    int mid = low + (high - low) / 2;
    if (low < high) {
        // calling first half
        merge_sort(low, mid);

        // calling second half
        merge_sort(mid + 1, high);

        // merging the two halves
        merge(low, mid, high);
    }
}

// thread function for multi-threading merge sort
void * merge_sort(void * arg) {
    // which part out of 4 parts
    // We need mutex lock here
    pthread_mutex_lock( & mutex);
    int thread_part = part++;
    pthread_mutex_unlock( & mutex);

    // calculating low and high
    int low = thread_part * (MAX / threadNum);
    int high = (thread_part + 1) * (MAX / threadNum) - 1;
    // Here is the tricky thing: if MAX / threadNum has remaining, we will lost last several items
    if (thread_part == threadNum - 1) {
        high = MAX - 1;
    }

    // evaluating mid point
    int mid = low + (high - low) / 2;
    if (low < high) {
        merge_sort(low, mid);
        merge_sort(mid + 1, high);
        merge(low, mid, high);
    }
}

// thread function for multi-threading merge
void * merge(void * arg) {
    // which part out of threadNum parts
    // We need mutex lock here
    pthread_mutex_lock( & mutex);
    int thread_part = part++;
    pthread_mutex_unlock( & mutex);

    // calculating low and high
    int low = (thread_part * 2) * (MAX / threadNum);
    int high = (thread_part + 1) * 2 * (MAX / threadNum) - 1;

    // evaluating mid point
    int mid = low + (high - low) / 2;

    // tricky thing said above
    if (thread_part == threadNum / 2 - 1) {
        high = MAX - 1;
    }
    if (low < high) {
        merge(low, mid, high);
    }
}

// thread function for multi-threading merge
void * merge8(void * arg) {
    // which part out of threadNum parts
    // We need mutex lock here
    pthread_mutex_lock( & mutex);
    int thread_part = part++;
    pthread_mutex_unlock( & mutex);

    // calculating low and high
    int low = (thread_part * 4) * (MAX / threadNum);
    int high = (thread_part + 1) * 4 * (MAX / threadNum) - 1;

    // evaluating mid point
    int mid = low + (high - low) / 2;

    if (thread_part == 1) {
        high = MAX - 1;
    }
    if (low < high) {
        merge(low, mid, high);
    }
}

// Driver Code
int main(int argc, char * argv[]) {
    if (argc >= 2) {
        threadNum = atoi(argv[1]);
    }
    cout << "Using " << threadNum << " threads..." << endl;
    // generating random values in array
    genRandom();

    // Elaped time
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    pthread_t threads[threadNum];

    // creating threadNum threads
    for (int i = 0; i < threadNum; i++) {
        pthread_create( & threads[i], NULL, merge_sort, (void * ) NULL);
    }

    // joining all threads
    for (int i = 0; i < threadNum; i++) {
        pthread_join(threads[i], NULL);
    }

    // if thread number is 4 or 8, we use pthread for parallel merge
    if (threadNum == 4 || threadNum == 8) {
        part = 0;
        // creating (threadNum / 2) threads for parallel merge
        for (int i = 0; i < threadNum / 2; i++) {
            pthread_create( & threads[i], NULL, merge, (void * ) NULL);
        }
        // joining all threads
        for (int i = 0; i < threadNum / 2; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    // parallel merge
    if (threadNum == 8) {
        part = 0;
        // creating (threadNum / 4) threads for parallel merge
        for (int i = 0; i < threadNum / 4; i++) {
            pthread_create( & threads[i], NULL, merge8, (void * ) NULL);
        }
        // joining all threads
        for (int i = 0; i < threadNum / 4; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    if (threadNum != 1) {
        // merging the final 2 parts
        merge(0, (threadNum / 2) * (MAX / threadNum) - 1, MAX - 1);
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    // time taken by merge sort in seconds
    cout << "Time taken: " << chrono::duration_cast < std::chrono::microseconds > (end - begin).count() << "[µs]" << endl;

    bool isSorted = true;
    for (int i = 1; i < MAX; i++) {
        if (a[i] < a[i - 1]) {
            isSorted = false;
            break;
        }
    }

    if (isSorted) {
        // displaying sorted array
        cout << "Sorted array: ";
        for (int i = 0; i < CERTIFY_NUM; i++) {
            cout << a[(MAX / CERTIFY_NUM) * i] << " ";
        }
        cout << endl;
    } else {
        cout << "Sorry! The merge sort does NOT lead to the correct order";
    }

    return 0;
}