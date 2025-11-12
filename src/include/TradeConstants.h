#pragma once
// =============================================
// 市场交易规则常量
// =============================================

// 涨跌停幅度
#define PRICE_LIMIT_UP_PCT       20.0     // 涨停幅度 10%
#define PRICE_LIMIT_DOWN_PCT    -20.0     // 跌停幅度 -10%

// 分位数缩尾（winsorize）默认阈值
#define WINSOR_QUANTILE_LOW      0.005
#define WINSOR_QUANTILE_HIGH     0.995

