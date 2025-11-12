#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;

namespace Tool {

class ConfigReader {
private:
    std::map<std::string, std::string> config_;
    std::string filename_;
    
public:
    // 构造函数
    ConfigReader(const std::string& filename) : filename_(filename) {
        loadConfig();
    }
    
    // 加载配置文件
    bool loadConfig() {
        return loadConfig(filename_, config_);
    }
    
    // 静态方法：加载配置文件
    static bool loadConfig(const std::string& filename, std::map<std::string, std::string>& config) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        std::string currentSection;
        
        while (std::getline(file, line)) {
            // 去除首尾空格
            line = trim(line);
            
            // 跳过空行和注释
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }
            
            // 解析节（section）
            if (line[0] == '[' && line[line.size()-1] == ']') {
                currentSection = line.substr(1, line.size()-2);
                continue;
            }
            
            // 解析键值对
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                
                // 如果有节名，添加前缀
                if (!currentSection.empty()) {
                    key = currentSection + "." + key;
                }
                
                config[key] = value;
            }
        }
        
        file.close();
        return true;
    }
    
    // 获取配置值
    static std::string getValue(const std::map<std::string, std::string>& config, 
                               const std::string& key, 
                               const std::string& defaultValue = "") {
        auto it = config.find(key);
        if (it != config.end()) {
            return it->second;
        }
        return defaultValue;
    }
    
    // 实例方法：获取字符串配置值
    std::string getString(const std::string& section, const std::string& key, const std::string& defaultValue = "") {
        std::string fullKey = section + "." + key;
        auto it = config_.find(fullKey);
        if (it != config_.end()) {
            return it->second;
        }
        return defaultValue;
    }
    
    // 实例方法：获取布尔配置值
    bool getBool(const std::string& section, const std::string& key, bool defaultValue = false) {
        std::string value = getString(section, key, "");
        if (value.empty()) {
            cout << "getBool: " << section << "." << key << " is empty, return default value: " << defaultValue << '\n';
            return defaultValue;
        }
        
        // 转换为小写
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        
        // 检查真值
        if (value == "true" || value == "1" || value == "yes" || value == "on") {
            return true;
        }
        
        // 检查假值
        if (value == "false" || value == "0" || value == "no" || value == "off") {
            return false;
        }
        
        // 默认返回默认值
        return defaultValue;
    }
    
    // 实例方法：获取整数配置值
    int getInt(const std::string& section, const std::string& key, int defaultValue = 0) {
        std::string value = getString(section, key, "");
        if (value.empty()) {
            return defaultValue;
        }
        
        try {
            return std::stoi(value);
        } catch (const std::exception&) {
            return defaultValue;
        }
    }
    
    // 实例方法：获取浮点数配置值
    double getDouble(const std::string& section, const std::string& key, double defaultValue = 0.0) {
        std::string value = getString(section, key, "");
        if (value.empty()) {
            return defaultValue;
        }
        
        try {
            return std::stod(value);
        } catch (const std::exception&) {
            return defaultValue;
        }
    }
    
private:
    // 去除字符串首尾空格
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }
};

} // namespace Tool

#endif // CONFIG_READER_H

