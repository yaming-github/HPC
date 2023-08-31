#include <bits/stdc++.h>

#include <iostream>

#include <random>

#include <chrono>

// number of elements in array
#define MAX 100000000

using namespace std;

// array of size MAX
int a[MAX];

// CITE: The code below for generate uniformly distributed random integers is aspired by
// https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
void genRandom() {
    std::mt19937 generator(time(NULL));
    std::uniform_int_distribution < int > distribution(0, INT_MAX);
    for (int i = 0; i < MAX; i++) {
        a[i] = distribution(generator); // generate random int flat in [min, max]
    }
}

int main() {
    genRandom();
    long long int sum = 0;
    int i;
    // Elaped time
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    for (i = 0; i < MAX; i++) {
        sum += a[i];
    }
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    // time taken by merge sort in seconds
    cout << "Time taken: " << chrono::duration_cast < std::chrono::milliseconds > (end - begin).count() << "[ms]" << endl;
    cout << "sum: " << sum << endl;
}