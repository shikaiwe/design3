# 股票数据分析项目

这是一个用于股票数据分析的 Python 项目，包含股价数据收集、财务指标分析、数据清洗和因子分析等功能。项目提供了完整的量化投资分析工具链，从数据收集到因子计算再到数据分析。

## 项目结构

```
design3/
├── collector/                    # 数据收集模块
│   ├── financial/               # 财务数据收集
│   │   ├── financial_data.py    # 财务数据获取
│   │   └── merge_all_financial_data.py # 财务数据合并
│   └── price/                   # 股价数据收集
│       ├── price_data.py        # 股价数据获取
│       └── merge_all_price_data.py # 股价数据合并
├── data/                        # 数据存储目录
│   └── EPU/                     # 经济政策不确定性数据
│       └── EPU_factors_description.txt # EPU因子说明
├── data_clean/                  # 数据清洗模块
│   ├── data_clean_result/       # 数据清洗结果目录
│   └── optimized_data_cleaning.py # 优化的数据清洗脚本
├── data_clean_result/           # 数据清洗结果
│   ├── cleaned_data/           # 清洗后的数据
│   ├── logs/                   # 日志文件
│   ├── raw_data/               # 原始数据
│   ├── reports/                # 清洗报告
│   └── visualizations/         # 可视化图表
├── factor/                      # 因子分析文档
│   └── factors_analysis.md     # 因子分析文档
├── factor_calculators/          # 因子计算器模块
│   ├── __init__.py
│   ├── base_calculator.py      # 基础计算器
│   ├── basics_calculator.py    # 基础科目及衍生因子计算器
│   ├── epu_factor_calculator.py # EPU因子计算器
│   ├── factor_calculation.py   # 因子计算主程序
│   ├── growth_calculator.py    # 成长类因子计算器
│   ├── momentum_calculator.py  # 动量类因子计算器
│   ├── new_style_calculator.py # 新风格因子计算器
│   ├── per_share_calculator.py # 每股指标因子计算器
│   ├── quality_calculator.py   # 质量类因子计算器
│   ├── risk_calculator.py      # 风险类因子计算器
│   ├── sentiment_calculator.py # 情绪类因子计算器
│   ├── style_calculator.py     # 风格因子计算器
│   └── technical_calculator.py # 技术指标类因子计算器
├── process_stock_index.py      # 股票指数处理
├── data_analysis_report.md     # 数据分析报告
├── requirements.txt             # 项目依赖
├── .gitignore                  # Git忽略文件
└── README.md                   # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 主要功能

### 1. 数据收集
- **股价数据收集**: 收集股票的日线价格数据，包括开盘价、收盘价、最高价、最低价、成交量等
- **财务数据收集**: 收集公司的财务报表数据，包括资产负债表、利润表、现金流量表等

### 2. 数据清洗
- **缺失值处理**: 使用拉格朗日线性插值等方法处理缺失数据
- **异常值检测**: 检测和处理数据中的异常值
- **数据标准化**: 对数据进行标准化处理，便于后续分析
- **可视化分析**: 生成数据清洗过程的可视化图表

### 3. 因子计算
项目支持计算9大类共276个因子，包括：

- **动量类因子** (32个): ROC、BIAS、CCI、CR、TRIX等
- **情绪类因子** (35个): AR、BR、ATR、VOL、VR等
- **风险类因子** (12个): 夏普比率、方差、偏度、峰度等
- **技术指标类因子** (17个): EMA、MACD、布林线等
- **质量类因子** (68个): ROE、ROA、资产周转率等
- **风格因子** (28个): 市值、动量、流动性等
- **成长类因子** (9个): 营收增长率、净利润增长率等
- **每股指标因子** (15个): 每股收益、每股净资产等
- **基础科目及衍生类因子** (37个): 市值、市盈率等
- **EPU因子** (10个): 经济政策不确定性相关因子，包括：
  - 移动平均因子：EPU_MA3、EPU_MA6、EPU_MA12
  - 变化率因子：EPU_MOM、EPU_QOQ、EPU_YOY
  - 波动率因子：EPU_VOL3、EPU_VOL6、EPU_VOL12
  - 相对位置因子：EPU_RANK12
  - 极端值因子：EPU_HIGH_FLAG、EPU_LOW_FLAG
  - 趋势因子：EPU_TREND12
  - 周期性因子：EPU_MONTHLY_AVG、EPU_SEASONAL
  - 加速度因子：EPU_ACCELERATION
  - 离散度因子：EPU_DEVIATION_MA12
  - 复合因子：EPU_COMPOSITE

### 4. 数据分析
- **因子比较分析**: 比较计算的因子与标准因子的差异
- **股票指数处理**: 处理和分析股票指数数据
- **可视化分析**: 生成各类分析图表

## 使用方法

### 1. 数据收集
```bash
# 收集财务数据
python collector/financial/financial_data.py

# 合并财务数据
python collector/financial/merge_all_financial_data.py

# 收集股价数据
python collector/price/price_data.py

# 合并股价数据
python collector/price/merge_all_price_data.py
```

### 2. 数据清洗
```bash
# 运行数据清洗脚本
python data_cleaning.py

# 或运行优化的数据清洗脚本
python optimized_data_cleaning.py
```

### 3. 因子计算
```bash
# 计算各类因子
python factor_calculators/factor_calculation.py

# 计算EPU因子
python factor_calculators/epu_factor_calculator.py
```

### 4. 股票指数处理
```bash
# 处理股票指数
python process_stock_index.py
```

## 数据说明

- **EPU数据**: 存储在 `data/EPU/` 目录下，包含经济政策不确定性指数数据
- **清洗后的数据**: 存储在 `data_clean_result/cleaned_data/` 目录下
- **原始数据**: 存储在 `data_clean_result/raw_data/` 目录下
- **清洗报告**: 存储在 `data_clean_result/reports/` 目录下
- **可视化图表**: 存储在 `data_clean_result/visualizations/` 目录下
- **因子分析文档**: 详见 `factor/factors_analysis.md`
- **数据分析报告**: 详见 `data_analysis_report.md`

## 项目依赖

主要依赖包括：
- **akshare**: 股票数据获取
- **pandas**: 数据处理
- **numpy**: 数值计算
- **scikit-learn**: 机器学习算法
- **tensorflow**: 深度学习框架
- **matplotlib/seaborn**: 数据可视化
- **streamlit**: 数据可视化应用
- **scipy**: 科学计算
- **backtrader**: 回测框架

## 注意事项

- 本项目使用的数据为示例数据，实际使用时请替换为真实数据
- 请确保已安装所有必要的依赖包
- 数据文件较大，已使用 .gitignore 排除数据文件和可视化输出
- 因子计算需要较长时间，建议在性能较好的机器上运行

## 项目报告

- **因子分析**: 详见 `factor/factors_analysis.md`
- **数据分析报告**: 详见 `data_analysis_report.md`
- **数据清洗报告**: 存储在 `data_clean_result/reports/` 目录下

## 更新日志

- 2025-06-18: 更新项目结构，添加EPU因子计算功能
- 2025-11-03: 完成数据清洗和因子计算功能，支持9大类276个因子
- 添加了数据清洗可视化和报告生成功能
- 优化了因子计算性能，支持分类存储