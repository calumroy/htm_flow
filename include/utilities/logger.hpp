#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>

///////////////////////////////////////////////////////////////////////////////
// Set the log level here for all files that include this header file.
#define LOG_LEVEL 1 // Only show messages from this log level and higher.
///////////////////////////////////////////////////////////////////////////////

enum LogLevel
{
    DEBUG = 1,
    INFO,
    WARN,
    ERROR
};

class Logger
{
public:
    static Logger &getInstance()
    {
        static Logger instance;
        return instance;
    }

    void log(LogLevel level, std::string message, const char *func, int line)
    {
        if (level < LOG_LEVEL)
            return; // Don't log the message if the log level is too low
        switch (level)
        {
        case DEBUG:
            std::cout << "[DEBUG] " << func << " (" << line << "): " << message << std::endl;
            break;
        case INFO:
            std::cout << "[INFO] " << func << " (" << line << "): " << message << std::endl;
            break;
        case WARN:
            std::cout << "[WARN] " << func << " (" << line << "): " << message << std::endl;
            break;
        case ERROR:
            std::cout << "[ERROR] " << func << " (" << line << "): " << message << std::endl;
            break;
        }
    }

private:
    Logger() {}
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;
};

#define LOG(level, message) Logger::getInstance().log(level, message, __func__, __LINE__)

#endif