#ifndef TIMER_H
#define TIMER_H

#include <chrono>

/**
 * Simple timer for performance measurement
 */
class Timer {
public:
    Timer() : running_(false) {}
    
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }
    
    void stop() {
        if (running_) {
            end_time_ = std::chrono::high_resolution_clock::now();
            running_ = false;
        }
    }
    
    double elapsed_ms() const {
        if (running_) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
            return duration.count() / 1000.0;
        } else {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
            return duration.count() / 1000.0;
        }
    }
    
    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_;
};

#endif // TIMER_H

