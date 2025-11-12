#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>

namespace Tool {
    // 时间戳工具类
    class Timestamp {
    public:
        // 获取当前时间戳（格式：YYYY-MM-DD HH:MM）
        static std::string getCurrentTimestamp() {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            std::stringstream ss;
            ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M");
            return ss.str();
        }
        
        // 获取简单时间戳（格式：HH:MM:SS.mmm）
        static std::string getSimpleTimestamp() {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()) % 1000;
            
            std::stringstream ss;
            ss << std::put_time(std::localtime(&time_t), "%H:%M:%S.%f");
            ss << '.' << std::setfill('0') << std::setw(6) << us.count();
            return ss.str();
        }
        
        // 获取日期戳（格式：YYYY-MM-DD）
        static std::string getDateStamp() {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            std::stringstream ss;
            ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d");
            return ss.str();
        }
        
        // 获取文件名时间戳（格式：mmddhhmmss）
        static std::string getFilenameTimestamp() {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            std::stringstream ss;
            ss << std::put_time(std::localtime(&time_t), "%m%d%H%M%S");
            return ss.str();
        }

        // 计算时间差（秒）
        static double difftime(const std::string& start_time, const std::string& end_time) {
            auto start = std::chrono::system_clock::from_time_t(std::stoll(start_time));
            auto end = std::chrono::system_clock::from_time_t(std::stoll(end_time));
            return std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        }
    };

    // 高精度计时器类
    class HighPrecisionTimer {
    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        std::chrono::high_resolution_clock::time_point end_time_;
        bool is_running_;

    public:
        HighPrecisionTimer() : is_running_(false) {}
        
        // 开始计时
        void start() {
            start_time_ = std::chrono::high_resolution_clock::now();
            is_running_ = true;
        }
        
        // 停止计时
        void stop() {
            if (is_running_) {
                end_time_ = std::chrono::high_resolution_clock::now();
                is_running_ = false;
            }
        }
        
        // 获取耗时（微秒）
        long long getElapsedMicroseconds() const {
            auto end = is_running_ ? std::chrono::high_resolution_clock::now() : end_time_;
            return std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_).count();
        }
        
        // 获取耗时（毫秒）
        long long getElapsedMilliseconds() const {
            return getElapsedMicroseconds() / 1000;
        }
        
        // 获取耗时（秒，double精度）
        double getElapsedSeconds() const {
            return getElapsedMicroseconds() / 1000000.0;
        }
        
        // 获取耗时（纳秒）
        long long getElapsedNanoseconds() const {
            auto end = is_running_ ? std::chrono::high_resolution_clock::now() : end_time_;
            return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_time_).count();
        }
        
        // 重置计时器
        void reset() {
            is_running_ = false;
        }
        
        // 获取格式化的耗时字符串
        std::string getFormattedElapsed() const {
            long long us = getElapsedMicroseconds();
            long long ms = us / 1000;
            long long s = ms / 1000;
            long long m = s / 60;
            long long h = m / 60;
            
            std::stringstream ss;
            if (h > 0) {
                ss << h << "h " << (m % 60) << "m " << (s % 60) << "s " << (ms % 1000) << "ms " << (us % 1000) << "μs";
            } else if (m > 0) {
                ss << m << "m " << (s % 60) << "s " << (ms % 1000) << "ms " << (us % 1000) << "μs";
            } else if (s > 0) {
                ss << s << "s " << (ms % 1000) << "ms " << (us % 1000) << "μs";
            } else if (ms > 0) {
                ss << ms << "ms " << (us % 1000) << "μs";
            } else {
                ss << us << "μs";
            }
            return ss.str();
        }
    };

    // 静态耗时计算函数
    class TimingUtils {
    public:
        // 测量函数执行时间（微秒）
        template<typename Func>
        static long long measureMicroseconds(Func&& func) {
            HighPrecisionTimer timer;
            timer.start();
            func();
            timer.stop();
            return timer.getElapsedMicroseconds();
        }
        
        // 测量函数执行时间（毫秒）
        template<typename Func>
        static long long measureMilliseconds(Func&& func) {
            return measureMicroseconds(func) / 1000;
        }
        
        // 测量函数执行时间（秒）
        template<typename Func>
        static double measureSeconds(Func&& func) {
            return measureMicroseconds(func) / 1000000.0;
        }
        
        // 测量函数执行时间并打印结果
        template<typename Func>
        static void measureAndPrint(const std::string& operation_name, Func&& func) {
            HighPrecisionTimer timer;
            timer.start();
            func();
            timer.stop();
            
            std::cout << operation_name << " 耗时: " << timer.getFormattedElapsed() 
                      << " (" << timer.getElapsedMicroseconds() << " μs)" << std::endl;
        }
        
        // 测量函数执行时间并返回结果和耗时
        template<typename Func>
        static auto measureWithResult(Func&& func) -> std::pair<decltype(func()), long long> {
            HighPrecisionTimer timer;
            timer.start();
            auto result = func();
            timer.stop();
            return std::make_pair(result, timer.getElapsedMicroseconds());
        }
        
        // 多次测量取平均值
        template<typename Func>
        static double measureAverage(int iterations, Func&& func) {
            long long total_us = 0;
            for (int i = 0; i < iterations; ++i) {
                total_us += measureMicroseconds(func);
            }
            return total_us / static_cast<double>(iterations);
        }
        
        // 多次测量并统计
        template<typename Func>
        static void measureStatistics(int iterations, const std::string& operation_name, Func&& func) {
            std::vector<long long> times;
            times.reserve(iterations);
            
            for (int i = 0; i < iterations; ++i) {
                times.push_back(measureMicroseconds(func));
            }
            
            // 计算统计信息
            std::sort(times.begin(), times.end());
            long long min_time = times[0];
            long long max_time = times[iterations - 1];
            long long total_time = std::accumulate(times.begin(), times.end(), 0LL);
            double avg_time = total_time / static_cast<double>(iterations);
            long long median_time = times[iterations / 2];
            
            std::cout << operation_name << " 统计 (" << iterations << " 次):" << std::endl;
            std::cout << "  最小: " << min_time << " μs" << std::endl;
            std::cout << "  最大: " << max_time << " μs" << std::endl;
            std::cout << "  平均: " << avg_time << " μs" << std::endl;
            std::cout << "  中位数: " << median_time << " μs" << std::endl;
            std::cout << "  总计: " << total_time << " μs" << std::endl;
        }
    };

    // 便捷的宏定义
    #define TIMER_START(name) HighPrecisionTimer name; name.start()
    #define TIMER_STOP(name) name.stop()
    #define TIMER_ELAPSED_US(name) name.getElapsedMicroseconds()
    #define TIMER_ELAPSED_MS(name) name.getElapsedMilliseconds()
    #define TIMER_ELAPSED_S(name) name.getElapsedSeconds()
    #define TIMER_PRINT(name, desc) std::cout << desc << " 耗时: " << name.getFormattedElapsed() << std::endl
    
    #define MEASURE_TIME_US(func) TimingUtils::measureMicroseconds([&](){func})
    #define MEASURE_TIME_MS(func) TimingUtils::measureMilliseconds([&](){func})
    #define MEASURE_TIME_S(func) TimingUtils::measureSeconds([&](){func})
    #define MEASURE_AND_PRINT(name, func) TimingUtils::measureAndPrint(name, [&](){func})
}

#endif // TIMESTAMP_H
