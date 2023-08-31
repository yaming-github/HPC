#include <iostream>
#include <cmath>
#include <random>

using namespace std;
#define N 40000
typedef numeric_limits<double> fll;
typedef numeric_limits<double> dbl;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

class sinCal {
private:
    const static int terms = 28;

    static float factorialF(int x) {
        float res = (float) x;
        for (int i = x - 1; i >= 2; --i) {
            res *= (float) i;
        }
        return res;
    }

    static double factorialD(int x) {
        double res = x;
        for (int i = x - 1; i >= 2; --i) {
            res *= i;
        }
        return res;
    }

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
        float res = x;
        for (int i = 1; i <= terms; ++i) {
            res += powf(-1, (float) i) * powf(x, (float) (2 * i + 1)) / factorialF(2 * i + 1);
        }
        return res;
    }

    static double sin(double x) {
        double res = x;
        for (int i = 1; i <= terms; ++i) {
            res += pow(-1, i) * pow(x, 2 * i + 1) / factorialD(2 * i + 1);
        }
        return res;
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