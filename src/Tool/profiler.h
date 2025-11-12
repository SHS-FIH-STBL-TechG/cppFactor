#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <climits>

namespace Tool {
    // 性能分析数据
    struct ProfileData {
        std::string name;
        long long total_us = 0;      // 总耗时（微秒）
        long long min_us = LLONG_MAX; // 最小耗时
        long long max_us = 0;         // 最大耗时
        size_t call_count = 0;        // 调用次数
        double avg_us = 0.0;          // 平均耗时
        
        void update(long long elapsed_us) {
            total_us += elapsed_us;
            min_us = std::min(min_us, elapsed_us);
            max_us = std::max(max_us, elapsed_us);
            call_count++;
            avg_us = total_us / static_cast<double>(call_count);
        }
    };

    // 性能分析器（单例）
    class Profiler {
    private:
        static Profiler* instance_;
        std::map<std::string, ProfileData> profiles_;
        mutable std::mutex mutex_;
        bool enabled_ = true;

        Profiler() = default;
        ~Profiler() = default;

    public:
        static Profiler& getInstance() {
            static Profiler instance;
            return instance;
        }

        // 启用/禁用性能分析
        void enable() { enabled_ = true; }
        void disable() { enabled_ = false; }
        bool isEnabled() const { return enabled_; }

        // 记录性能数据
        void record(const std::string& name, long long elapsed_us) {
            if (!enabled_) return;
            
            std::lock_guard<std::mutex> lock(mutex_);
            profiles_[name].name = name;
            profiles_[name].update(elapsed_us);
        }

        // 获取性能数据
        ProfileData getProfile(const std::string& name) const {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = profiles_.find(name);
            if (it != profiles_.end()) {
                return it->second;
            }
            return ProfileData();
        }

        // 获取所有性能数据
        std::map<std::string, ProfileData> getAllProfiles() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return profiles_;
        }

        // 清空所有数据
        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            profiles_.clear();
        }

        // 打印报告到控制台
        void printReport() const {
            if (!enabled_) return;
            
            std::lock_guard<std::mutex> lock(mutex_);
            
            if (profiles_.empty()) {
                std::cout << "\n========== 性能分析报告 ==========\n";
                std::cout << "无性能数据记录\n";
                std::cout << "====================================\n\n";
                return;
            }
            
            std::cout << "\n========== 性能分析报告 ==========\n";
            std::cout << std::left << std::setw(40) << "函数名"
                      << std::right << std::setw(12) << "调用次数"
                      << std::setw(15) << "总耗时(ms)"
                      << std::setw(15) << "平均耗时(ms)"
                      << std::setw(15) << "最小耗时(ms)"
                      << std::setw(15) << "最大耗时(ms)" << "\n";
            std::cout << std::string(110, '-') << "\n";

            // 按总耗时排序
            std::vector<std::pair<std::string, ProfileData>> sorted_profiles;
            for (const auto& pair : profiles_) {
                sorted_profiles.push_back(pair);
            }
            std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                [](const auto& a, const auto& b) {
                    return a.second.total_us > b.second.total_us;
                });

            for (const auto& pair : sorted_profiles) {
                const auto& data = pair.second;
                std::cout << std::left << std::setw(40) << data.name
                          << std::right << std::setw(12) << data.call_count
                          << std::setw(15) << std::fixed << std::setprecision(2) 
                          << data.total_us / 1000.0
                          << std::setw(15) << data.avg_us / 1000.0
                          << std::setw(15) << (data.min_us == LLONG_MAX ? 0 : data.min_us) / 1000.0
                          << std::setw(15) << data.max_us / 1000.0 << "\n";
            }
            std::cout << "====================================\n\n";
        }

        // 导出到CSV文件
        void exportToCSV(const std::string& filename) const {
            if (!enabled_) return;
            
            std::lock_guard<std::mutex> lock(mutex_);
            
            if (profiles_.empty()) {
                std::cout << "无性能数据可导出\n";
                return;
            }
            
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "无法打开文件: " << filename << std::endl;
                return;
            }

            // CSV头部
            file << "函数名,调用次数,总耗时(ms),平均耗时(ms),最小耗时(ms),最大耗时(ms)\n";

            // 按总耗时排序
            std::vector<std::pair<std::string, ProfileData>> sorted_profiles;
            for (const auto& pair : profiles_) {
                sorted_profiles.push_back(pair);
            }
            std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                [](const auto& a, const auto& b) {
                    return a.second.total_us > b.second.total_us;
                });

            for (const auto& pair : sorted_profiles) {
                const auto& data = pair.second;
                file << data.name << ","
                     << data.call_count << ","
                     << std::fixed << std::setprecision(2) << data.total_us / 1000.0 << ","
                     << data.avg_us / 1000.0 << ","
                     << (data.min_us == LLONG_MAX ? 0 : data.min_us) / 1000.0 << ","
                     << data.max_us / 1000.0 << "\n";
            }

            file.close();
            std::cout << "性能报告已导出到: " << filename << std::endl;
        }

        // 导出特定前缀的记录到CSV文件
        void exportToCSVWithPrefix(const std::string& filename, const std::string& prefix) const {
            if (!enabled_) return;
            
            std::lock_guard<std::mutex> lock(mutex_);
            
            std::vector<std::pair<std::string, ProfileData>> filtered_profiles;
            for (const auto& pair : profiles_) {
                if (pair.first.find(prefix) == 0) {  // 检查是否以prefix开头
                    filtered_profiles.push_back(pair);
                }
            }
            
            if (filtered_profiles.empty()) {
                std::cout << "无匹配前缀 \"" << prefix << "\" 的性能数据可导出\n";
                return;
            }
            
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "无法打开文件: " << filename << std::endl;
                return;
            }

            // CSV头部
            file << "函数名,调用次数,总耗时(ms),平均耗时(ms),最小耗时(ms),最大耗时(ms)\n";

            // 按总耗时排序
            std::sort(filtered_profiles.begin(), filtered_profiles.end(),
                [](const auto& a, const auto& b) {
                    return a.second.total_us > b.second.total_us;
                });

            for (const auto& pair : filtered_profiles) {
                const auto& data = pair.second;
                file << data.name << ","
                     << data.call_count << ","
                     << std::fixed << std::setprecision(2) << data.total_us / 1000.0 << ","
                     << data.avg_us / 1000.0 << ","
                     << (data.min_us == LLONG_MAX ? 0 : data.min_us) / 1000.0 << ","
                     << data.max_us / 1000.0 << "\n";
            }

            file.close();
            std::cout << "性能报告（前缀: \"" << prefix << "\"）已导出到: " << filename << std::endl;
        }
    };

    // RAII自动计时器
    class AutoTimer {
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_time_;
        bool active_;

    public:
        explicit AutoTimer(const std::string& name) 
            : name_(name), active_(true) {
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        ~AutoTimer() {
            if (active_) {
                stop();
            }
        }

        // 停止计时并记录
        void stop() {
            if (!active_) return;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time_).count();
            
            Profiler::getInstance().record(name_, elapsed);
            active_ = false;
        }

        // 获取当前耗时（不停止计时）
        long long getElapsed() const {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                now - start_time_).count();
        }
    };

    // 便捷宏定义
    #define PROFILE_SCOPE(name) Tool::AutoTimer _auto_timer_##__LINE__(name)
    #define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
    #define PROFILE_FUNCTION_FULL() PROFILE_SCOPE(std::string(__FILE__) + "::" + __FUNCTION__)
}

#endif // PROFILER_H

