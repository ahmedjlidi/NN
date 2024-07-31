#pragma once
#include <chrono>

class Timer {
public:
    Timer() {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_time() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        return elapsed_seconds.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

