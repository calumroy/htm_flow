#include <utilities/stopwatch.hpp>

// Define the static member variables here (moved from header to avoid multiple definitions)
std::chrono::time_point<std::chrono::high_resolution_clock> Stopwatch::startTime;
std::chrono::time_point<std::chrono::high_resolution_clock> Stopwatch::stopTime; 