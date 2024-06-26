#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <chrono>
#include <iostream>

#define START_STOPWATCH() Stopwatch::start()
#define STOP_STOPWATCH() Stopwatch::stop()
#define PRINT_ELAPSED_TIME() Stopwatch::printElapsed()
#define GET_ELAPSED_TIME() Stopwatch::getElapsed()

class Stopwatch
{
public:
    static void start()
    {
        startTime = std::chrono::high_resolution_clock::now();
    }

    static void stop()
    {
        stopTime = std::chrono::high_resolution_clock::now();
    }

    static void printElapsed()
    {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
        std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;
    }

    // Returns the elapsed time in milliseconds
    static long long getElapsed()
    {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
        return duration.count();
    }

private:
    static std::chrono::time_point<std::chrono::high_resolution_clock> startTime, stopTime;
};

std::chrono::time_point<std::chrono::high_resolution_clock> Stopwatch::startTime;
std::chrono::time_point<std::chrono::high_resolution_clock> Stopwatch::stopTime;

#endif // STOPWATCH_H
