#pragma once
#include <memory>
#include "OnlineUtils.h"

// 在线因子基类：提供统一的创建接口和多态支持
class OnlineBaseFactor {
    public:
        virtual ~OnlineBaseFactor() = default;
        
        // 创建在线因子实例（应用层统一管理共享）
        template<typename T, typename... Args>
        static std::shared_ptr<T> createOnlineBaseF(Args&&... args){
            return std::make_shared<T>(std::forward<Args>(args)...);
        }
    
    protected:
        OnlineBaseFactor() = default; // 防止直接实例化
};
