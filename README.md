# 股票数据分析项目

这是一个用于股票数据分析的 Python 项目，包含股价数据收集、财务指标分析和因子分析等功能。

## 项目结构

```
design3/
├── collector/           # 数据收集模块
│   ├── financial/       # 财务数据收集
│   ├── price/          # 股价数据收集
│   └── data/           # 数据存储
├── data/               # 数据目录
│   ├── daliy_prices/   # 日线股价数据
│   ├── financial_data/ # 财务数据
│   ├── factors/        # 因子数据
│   └── EPU/            # 经济政策不确定性数据
├── process_stock_index.py # 股票指数处理
└── requirements.txt    # 项目依赖
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 主要功能

1. **数据收集**: 收集股票的日线价格数据和财务数据
2. **数据分析**: 对股票数据进行比较和分析
3. **因子分析**: 分析各种因子对股票表现的影响

## 使用方法

1. 运行主程序:
   ```bash
   python 1.py
   ```

2. 比较股票分析:
   ```bash
   python compare_stocks.py
   ```

3. 处理股票指数:
   ```bash
   python process_stock_index.py
   ```

## 数据说明

- 股价数据存储在 `data/daliy_prices/` 目录下，格式为 `price_股票代码.csv`
- 财务数据存储在 `data/financial_data/` 目录下
- 因子数据存储在 `data/factors/` 目录下

## 注意事项

- 本项目使用的数据为示例数据，实际使用时请替换为真实数据
- 请确保已安装所有必要的依赖包
- 数据文件较大，建议使用 .gitignore 排除数据文件