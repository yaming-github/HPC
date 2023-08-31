#include <iostream>
#include <cmath>
#include <random>

using namespace std;
#define N 4
typedef numeric_limits<double> fll;
typedef numeric_limits<double> dbl;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

class sinCal {
public:
    static float randFloat() {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<float> dist(3.14 / 2, 3.14 * 1.5);

        return dist(mt);
    }

    static double randDouble() {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(3.14 / 2, 3.14 * 1.5);

        return dist(mt);
    }

    static float sin(float x) {
        return ((float) 12671 / 4363920 * powf(x, 5) -
                (float) 2363 / 18183 * powf(x, 3) + x) /
               (1 + (float) 445 / 12122 * powf(x, 2) + (float) 601 / 872784 * powf(x, 4) +
                (float) 121 / 16662240 * powf(x, 6));
    }

    static double sin(double x) {
        return ((double) 12671 / 4363920 * pow(x, 5) -
                (double) 2363 / 18183 * pow(x, 3) + x) /
               (1 + (double) 445 / 12122 * pow(x, 2) + (double) 601 / 872784 * pow(x, 4) +
                (double) 121 / 16662240 * pow(x, 6));
    }
};

int main() {
    float f[N], resultF[N];
    for (float &i: f) {
        i = sinCal::randFloat();
    }
    double d[N], resultD[N];
    for (int i = 0; i < N; ++i) {
        d[i] = f[i];
    }
    double start = CLOCK();
    for (int i = 0; i < N; ++i) {
        resultF[i] = sinCal::sin(f[i]);
        resultD[i] = sinCal::sin(d[i]);
    }
    cout << "Running time: " << CLOCK() - start << " ms" << endl;
    cout.precision(fll::max_digits10);
    cout.precision(dbl::max_digits10);
    cout << "x: " << d[0] << "; float: " << resultF[0] << "; double: " << resultD[0] << endl;
}