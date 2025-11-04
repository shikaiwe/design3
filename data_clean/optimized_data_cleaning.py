#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版系统性数据清洗脚本
对股票价格数据和财务数据进行清洗处理，采用更优的清洗方法和文件组织结构
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, Union

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

class OptimizedDataCleaner:
    """优化版数据清洗类"""
    
    def __init__(self, price_data_path: str, financial_data_path: str, trading_days_path: str = None, output_dir: str = "data_clean_result"):
        """
        初始化数据清洗器
        
        参数:
        price_data_path: 价格数据文件路径
        financial_data_path: 财务数据文件路径
        trading_days_path: 交易日数据文件路径
        output_dir: 输出目录
        """
        self.price_data_path = price_data_path
        self.financial_data_path = financial_data_path
        self.trading_days_path = trading_days_path
        self.output_dir = output_dir
        
        # 创建输出目录结构
        self._create_output_dirs()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化数据变量
        self.price_data = None
        self.financial_data = None
        self.trading_days = None
        self.cleaned_price_data = None
        self.cleaned_financial_data = None
        
        # 定义标识符列，这些列不应被当作数值数据处理
        self.identifier_columns = ['stock_code', '股票代码', 'REPORT_DATE', 'REPORT_TYPE', 'report_type', 
                                  'has_balance_sheet', 'has_profit_sheet', 'has_cash_flow', '日期']
        
        # 初始化清洗报告
        self.cleaning_report = {
            "price_data": {
                "original_shape": None,
                "cleaned_shape": None,
                "missing_values": {},
                "outliers": {},
                "data_quality_score": None
            },
            "financial_data": {
                "original_shape": None,
                "cleaned_shape": None,
                "missing_values": {},
                "outliers": {},
                "data_quality_score": None
            },
            "processing_time": None,
            "cleaning_methods": {
                "missing_value": "时间序列插值/行业均值填充",
                "outlier_detection": "3σ准则/IQR/分位数组合方法"
            }
        }
        
    def _create_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            self.output_dir,
            os.path.join(self.output_dir, "raw_data"),
            os.path.join(self.output_dir, "cleaned_data"),
            os.path.join(self.output_dir, "reports"),
            os.path.join(self.output_dir, "visualizations"),
            os.path.join(self.output_dir, "logs")
        ]
        
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"创建目录: {dir_path}")
    
    def _setup_logging(self):
        """设置日志记录"""
        log_file = os.path.join(self.output_dir, "logs", f"data_cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("数据清洗日志初始化完成")
        
    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """加载数据"""
        self.logger.info("开始加载数据...")
        
        # 加载价格数据
        try:
            self.price_data = pd.read_csv(self.price_data_path)
            self.logger.info(f"价格数据加载成功，形状: {self.price_data.shape}")
        except Exception as e:
            self.logger.error(f"价格数据加载失败: {e}")
            
        # 加载财务数据
        try:
            self.financial_data = pd.read_csv(self.financial_data_path)
            self.logger.info(f"财务数据加载成功，形状: {self.financial_data.shape}")
        except Exception as e:
            self.logger.error(f"财务数据加载失败: {e}")
            
        # 加载交易日数据
        if self.trading_days_path:
            try:
                self.trading_days = pd.read_csv(self.trading_days_path)
                self.logger.info(f"成功加载交易日数据，形状: {self.trading_days.shape}")
                
                # 尝试将日期列转换为datetime格式
                date_col = None
                for col in self.trading_days.columns:
                    if 'date' in col.lower() or '日期' in col or '时间' in col:
                        date_col = col
                        break
                
                if date_col:
                    self.trading_days[date_col] = pd.to_datetime(self.trading_days[date_col])
                    # 提取日期部分，去除时间
                    if hasattr(self.trading_days[date_col].dt, 'date'):
                        self.trading_days['trading_date'] = self.trading_days[date_col].dt.date
                    else:
                        self.trading_days['trading_date'] = self.trading_days[date_col].apply(lambda x: x.date() if hasattr(x, 'date') else x)
                    
                    self.logger.info(f"交易日数据日期列: {date_col}")
                else:
                    self.logger.warning("未找到交易日数据的日期列")
                    
            except Exception as e:
                self.logger.error(f"加载交易日数据失败: {e}")
                self.trading_days = None
        else:
            self.logger.info("未提供交易日数据路径，跳过交易日数据加载")
            self.trading_days = None
            
        # 记录原始数据形状
        if self.price_data is not None:
            self.cleaning_report["price_data"]["original_shape"] = self.price_data.shape
            
        if self.financial_data is not None:
            self.cleaning_report["financial_data"]["original_shape"] = self.financial_data.shape
            
        return self.price_data, self.financial_data
    
    def analyze_data_structure(self):
        """分析数据结构"""
        self.logger.info("===== 数据结构分析 =====")
        
        # 价格数据分析
        if self.price_data is not None:
            self.logger.info("\n价格数据信息:")
            self.logger.info(f"数据形状: {self.price_data.shape}")
            self.logger.info(f"列名: {list(self.price_data.columns)}")
            self.logger.info("\n数据类型:")
            self.logger.info(self.price_data.dtypes)
            
            # 保存原始数据到raw_data目录
            raw_price_path = os.path.join(self.output_dir, "raw_data", "original_price_data.csv")
            self.price_data.to_csv(raw_price_path, index=False)
            self.logger.info(f"原始价格数据已保存至: {raw_price_path}")
            
        # 财务数据分析
        if self.financial_data is not None:
            self.logger.info("\n财务数据信息:")
            self.logger.info(f"数据形状: {self.financial_data.shape}")
            self.logger.info(f"列名: {list(self.financial_data.columns)}")
            self.logger.info("\n数据类型:")
            self.logger.info(self.financial_data.dtypes)
            
            # 保存原始数据到raw_data目录
            raw_financial_path = os.path.join(self.output_dir, "raw_data", "original_financial_data.csv")
            self.financial_data.to_csv(raw_financial_path, index=False)
            self.logger.info(f"原始财务数据已保存至: {raw_financial_path}")
        
        # 分析交易日数据结构
        if self.trading_days is not None:
            self.logger.info(f"\n交易日数据信息:")
            self.logger.info(f"数据形状: {self.trading_days.shape}")
            self.logger.info(f"列名: {list(self.trading_days.columns)}")
            self.logger.info("\n数据类型:")
            self.logger.info(self.trading_days.dtypes)
    
    def filter_non_trading_days(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        过滤非交易日数据
        
        参数:
        data: 原始数据
        
        返回:
        过滤后的数据
        """
        self.logger.info("开始过滤非交易日数据")
        
        if self.trading_days is None:
            self.logger.warning("未提供交易日数据，跳过非交易日过滤")
            return data.copy()
        
        data_copy = data.copy()
        
        # 尝试找到日期列
        date_col = None
        for col in data_copy.columns:
            if 'date' in col.lower() or '日期' in col:
                date_col = col
                break
        
        if date_col is None:
            self.logger.warning("未找到日期列，跳过非交易日过滤")
            return data_copy
        
        # 确保日期列是datetime格式
        try:
            data_copy[date_col] = pd.to_datetime(data_copy[date_col])
        except Exception as e:
            self.logger.error(f"转换日期列失败: {e}")
            return data_copy
        
        # 提取日期部分（去除时间）
        if hasattr(data_copy[date_col].dt, 'date'):
            data_dates = data_copy[date_col].dt.date
        else:
            data_dates = data_copy[date_col].apply(lambda x: x.date() if hasattr(x, 'date') else x)
        
        # 获取非交易日集合
        # 非交易日文件没有列名，第一行是0，后面是日期
        non_trading_dates = set()
        if len(self.trading_days) > 0:
            # 跳过第一行（值为0），从第二行开始读取日期
            for i in range(1, len(self.trading_days)):
                date_str = self.trading_days.iloc[i, 0]  # 使用第一列的值
                if isinstance(date_str, str):
                    try:
                        date_obj = pd.to_datetime(date_str).date()
                        non_trading_dates.add(date_obj)
                    except:
                        continue
        
        # 过滤非交易日数据
        is_non_trading_day = data_dates.isin(non_trading_dates)
        filtered_data = data_copy[~is_non_trading_day].copy()
        
        # 记录过滤信息
        removed_count = len(data_copy) - len(filtered_data)
        self.logger.info(f"移除了 {removed_count} 条非交易日记录")
        self.logger.info(f"过滤前数据形状: {data_copy.shape}")
        self.logger.info(f"过滤后数据形状: {filtered_data.shape}")
        
        # 记录到清洗报告
        self.cleaning_report["price_data"]["non_trading_days_removed"] = int(removed_count)
        
        return filtered_data
    
    def analyze_missing_pattern(self, data: pd.DataFrame, data_type: str = "price") -> Dict:
        """
        分析缺失值模式
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        missing_pattern: 缺失值模式分析结果
        """
        self.logger.info(f"\n===== 分析{data_type}数据缺失值模式 =====")
        
        # 计算缺失值比例
        missing_values = data.isnull().sum()
        total_records = len(data)
        missing_percentage = (missing_values / total_records) * 100
        
        # 分析缺失值模式
        missing_pattern = {
            "total_records": total_records,
            "missing_counts": missing_values.to_dict(),
            "missing_percentages": missing_percentage.to_dict(),
            "complete_records": total_records - data.dropna().shape[0],
            "complete_record_percentage": (data.dropna().shape[0] / total_records) * 100
        }
        
        # 判断缺失值类型
        if data_type == "price" and "日期" in data.columns:
            # 对于价格数据，检查时间序列特性
            data_sorted = data.sort_values("日期")
            missing_by_time = {}
            for col in data.select_dtypes(include=[np.number]).columns:
                if col != "stock_code":  # 跳过股票代码
                    # 检查连续缺失情况
                    missing_series = data_sorted[col].isnull()
                    consecutive_missing = self._find_consecutive_missing(missing_series)
                    missing_by_time[col] = consecutive_missing
            
            missing_pattern["time_series_analysis"] = missing_by_time
        
        # 打印缺失值信息
        self.logger.info(f"总记录数: {total_records}")
        self.logger.info(f"完整记录数: {data.dropna().shape[0]} ({missing_pattern['complete_record_percentage']:.2f}%)")
        self.logger.info("\n缺失值统计:")
        missing_df = pd.DataFrame({
            "缺失数量": missing_values,
            "缺失比例(%)": missing_percentage
        })
        self.logger.info(missing_df[missing_df["缺失数量"] > 0])
        
        # 可视化缺失值模式
        if missing_values.sum() > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
            plt.title(f"{data_type}数据缺失值模式")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "visualizations", f"{data_type}_missing_pattern.png"))
            plt.close()
        
        return missing_pattern
    
    def _find_consecutive_missing(self, missing_series: pd.Series) -> Dict:
        """查找连续缺失值"""
        consecutive_groups = []
        current_group = []
        
        for i, is_missing in enumerate(missing_series):
            if is_missing:
                current_group.append(i)
            else:
                if current_group:
                    consecutive_groups.append(current_group)
                    current_group = []
        
        if current_group:
            consecutive_groups.append(current_group)
        
        return {
            "consecutive_groups": consecutive_groups,
            "max_consecutive": max([len(group) for group in consecutive_groups]) if consecutive_groups else 0,
            "total_consecutive_groups": len(consecutive_groups)
        }
    
    def handle_missing_values(self, data: pd.DataFrame, is_price_data: bool = True) -> pd.DataFrame:
        """
        处理缺失值 - 修改为更加温和的处理方式，避免数据过于统一化
        
        参数:
        data: 原始数据
        is_price_data: 是否为价格数据
        
        返回:
        处理后的数据
        """
        self.logger.info("开始处理缺失值（使用温和的处理方式）")
        data_copy = data.copy()
        missing_info = {}
        
        # 获取数值列（排除标识符列）
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in self.identifier_columns]
        
        for col in numeric_cols:
            missing_count = data_copy[col].isnull().sum()
            if missing_count > 0:
                missing_ratio = missing_count / len(data_copy)
                missing_info[col] = {
                    "count": missing_count,
                    "ratio": missing_ratio
                }
                
                # 根据缺失比例选择处理方法 - 使用更温和的方式
                if missing_ratio < 0.01:  # 只处理缺失比例小于1%的列
                    # 缺失比例很小，使用时间序列插值（对价格数据）或前向填充（对财务数据）
                    if is_price_data and '日期' in data_copy.columns:
                        # 按股票代码分组后进行时间序列插值
                        for stock in data_copy['stock_code'].unique():
                            mask = data_copy['stock_code'] == stock
                            if mask.sum() > 1:  # 确保每个股票至少有两条记录
                                data_copy.loc[mask, col] = data_copy.loc[mask, col].interpolate(method='linear')
                    else:
                        # 使用前向填充，保留数据的原始特征
                        data_copy[col] = data_copy[col].fillna(method='ffill')
                        # 如果开头就是缺失，用后向填充
                        data_copy[col] = data_copy[col].fillna(method='bfill')
                elif missing_ratio < 0.05:  # 缺失比例在1%-5%之间
                    # 使用按股票分组的前向填充，保留更多原始特征
                    if 'stock_code' in data_copy.columns:
                        # 按股票代码分组，使用组内前向填充
                        data_copy[col] = data_copy.groupby('stock_code')[col].transform(
                            lambda x: x.fillna(method='ffill').fillna(method='bfill')
                        )
                        # 如果仍有缺失值，使用整体前向填充
                        data_copy[col] = data_copy[col].fillna(method='ffill').fillna(method='bfill')
                    else:
                        data_copy[col] = data_copy[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # 缺失比例大于5%，记录但不处理，避免引入过多人为数据
                    self.logger.warning(f"列 {col} 缺失比例过高 ({missing_ratio:.2%})，保留原始缺失状态")
                    # 不处理，保留原始缺失状态
        
        # 记录缺失值处理信息
        if is_price_data:
            self.cleaning_report["price_data"]["missing_values"] = missing_info
        else:
            self.cleaning_report["financial_data"]["missing_values"] = missing_info
        
        self.logger.info(f"缺失值处理完成，处理了 {len(missing_info)} 列")
        return data_copy
    
    def _time_series_interpolation(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """时间序列插值"""
        # 按日期排序
        if "日期" in data.columns:
            data_sorted = data.sort_values("日期")
            
            # 获取非缺失值的索引和值
            valid_indices = data_sorted[col].notna()
            valid_data = data_sorted.loc[valid_indices, col]
            
            if len(valid_data) > 1:  # 确保有足够的点进行插值
                # 创建插值函数
                x = valid_data.index
                y = valid_data.values
                
                # 使用时间加权插值
                f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
                
                # 填充缺失值
                missing_indices = data_sorted[col].isna()
                if missing_indices.any():
                    data_sorted.loc[missing_indices, col] = f(data_sorted.loc[missing_indices].index)
            else:
                # 如果没有足够的点进行插值，使用均值填充
                data_sorted[col].fillna(data_sorted[col].mean(), inplace=True)
                self.logger.info(f"  - 数据点不足，使用均值填充")
            
            # 恢复原始顺序
            return data_sorted.sort_index()
        else:
            # 如果没有日期列，使用线性插值
            data[col] = data[col].interpolate(method='linear')
            return data
    
    def _industry_mean_imputation(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """行业均值填充"""
        # 如果有股票代码，尝试按行业分组填充
        if "stock_code" in data.columns:
            # 这里简化处理，使用整体均值
            # 实际应用中可以根据股票代码映射到行业
            data[col].fillna(data[col].mean(), inplace=True)
        else:
            # 使用整体均值填充
            data[col].fillna(data[col].mean(), inplace=True)
        
        return data
    
    def detect_outliers(self, data: pd.DataFrame, is_price_data: bool = True) -> pd.DataFrame:
        """
        检测和处理极端异常值 - 修改为只处理真正的极端值，避免数据过于统一化
        
        参数:
        data: 原始数据
        is_price_data: 是否为价格数据
        
        返回:
        处理后的数据
        """
        self.logger.info("开始检测和处理极端异常值（只处理真正的极端值）")
        data_copy = data.copy()
        outlier_info = {}
        
        # 获取数值列（排除标识符列）
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in self.identifier_columns]
        
        for col in numeric_cols:
            # 检查零值情况
            zero_count = (data_copy[col] == 0).sum()
            if zero_count > 0:
                zero_ratio = zero_count / len(data_copy)
                self.logger.info(f"列 '{col}' 中有 {zero_count} 个零值 ({zero_ratio:.2%})")
                
                # 对于价格数据，零值通常是异常的，需要处理
                if is_price_data and col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额']:
                    # 用前一个非零值替换零值
                    data_copy[col] = data_copy[col].replace(0, np.nan)
                    data_copy[col] = data_copy[col].fillna(method='ffill')
                    # 如果开头就是零，用后一个值填充
                    data_copy[col] = data_copy[col].fillna(method='bfill')
                    outlier_info[col] = {
                        "zero_values": zero_count,
                        "zero_ratio": zero_ratio
                    }
            
            # 检查缺失值情况
            missing_count = data_copy[col].isnull().sum()
            if missing_count > 0:
                missing_ratio = missing_count / len(data_copy)
                self.logger.info(f"列 '{col}' 中有 {missing_count} 个缺失值 ({missing_ratio:.2%})")
                
                # 对于价格数据，使用线性插值填充
                if is_price_data and '日期' in data_copy.columns:
                    data_copy = self._time_series_interpolation(data_copy, col)
                else:
                    # 使用中位数填充
                    data_copy[col].fillna(data_copy[col].median(), inplace=True)
                
                if col not in outlier_info:
                    outlier_info[col] = {}
                outlier_info[col]["missing_values"] = missing_count
                outlier_info[col]["missing_ratio"] = missing_ratio
            
            # 只处理真正的极端异常值（使用非常严格的阈值）
            if is_price_data and col in ['开盘', '收盘', '最高', '最低']:
                # 对于价格数据，使用非常严格的异常值检测
                # 只处理明显不合理的数据（如价格为负数或极端离群值）
                if '股票代码' in data_copy.columns:
                    extreme_outliers = []
                    for stock_code, group in data_copy.groupby('股票代码'):
                        # 只检测极端异常值（使用Z-score > 7）
                        stock_extreme_outliers = self._detect_extreme_outliers(group[col])
                        extreme_outliers.extend(stock_extreme_outliers)
                    
                    extreme_outliers = list(set(extreme_outliers))  # 去重
                else:
                    # 如果没有股票代码列，只检测极端异常值
                    extreme_outliers = self._detect_extreme_outliers(data_copy[col])
            else:
                # 对于非价格数据，只检测极端异常值
                extreme_outliers = self._detect_extreme_outliers(data_copy[col])
            
            if len(extreme_outliers) > 0:
                extreme_ratio = len(extreme_outliers) / len(data_copy)
                self.logger.info(f"列 '{col}' 中有 {len(extreme_outliers)} 个极端异常值 ({extreme_ratio:.2%})")
                
                if col not in outlier_info:
                    outlier_info[col] = {}
                outlier_info[col]["extreme_outliers"] = len(extreme_outliers)
                outlier_info[col]["extreme_outlier_ratio"] = extreme_ratio
                
                # 处理极端异常值 - 使用非常温和的方法
                if is_price_data and col in ['开盘', '收盘', '最高', '最低']:
                    # 对于价格数据，使用非常宽松的边界
                    q1, q3 = data_copy[col].quantile([0.01, 0.99])  # 使用1%和99%分位数
                    iqr = q3 - q1
                    lower_bound = max(0, q1 - 5 * iqr)  # 使用5倍IQR，且下界不小于0
                    upper_bound = q3 + 5 * iqr  # 使用5倍IQR
                else:
                    # 对于非价格数据，使用更宽松的边界
                    q1, q3 = data_copy[col].quantile([0.01, 0.99])
                    iqr = q3 - q1
                    lower_bound = q1 - 5 * iqr
                    upper_bound = q3 + 5 * iqr
                
                # 将极端异常值替换为边界值
                data_copy.loc[data_copy[col] < lower_bound, col] = lower_bound
                data_copy.loc[data_copy[col] > upper_bound, col] = upper_bound
                
                self.logger.info(f"  - 处理极端异常值，下界: {lower_bound:.4f}, 上界: {upper_bound:.4f}")
        
        # 记录异常值处理信息
        if is_price_data:
            self.cleaning_report["price_data"]["outliers"] = outlier_info
        else:
            self.cleaning_report["financial_data"]["outliers"] = outlier_info
        
        self.logger.info(f"极端异常值检测和处理完成，处理了 {len(outlier_info)} 列")
        return data_copy
    
    def _detect_extreme_outliers(self, series: pd.Series) -> list:
        """检测真正的极端异常值（使用非常严格的阈值）"""
        if len(series) < 5:  # 数据点太少，不检测异常值
            return []
        
        # 使用非常严格的Z-score阈值（7而不是3）
        z_scores = np.abs(stats.zscore(series.dropna()))
        extreme_outliers = series.dropna().index[z_scores > 7].tolist()
        
        # 额外检查：对于价格数据，负值通常是异常的
        price_cols = ['开盘', '收盘', '最高', '最低']
        if series.name in price_cols:
            negative_values = series[series < 0].index.tolist()
            extreme_outliers.extend(negative_values)
            extreme_outliers = list(set(extreme_outliers))  # 去重
        
        return extreme_outliers
    
    def _detect_outliers_iqr(self, series: pd.Series) -> list:
        """使用IQR方法检测异常值"""
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outlier_indices
    
    def _detect_outliers_zscore(self, series: pd.Series) -> list:
        """使用Z-score方法检测异常值"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_indices = series.dropna().index[z_scores > 3].tolist()
        return outlier_indices
    
    def _detect_outliers_quantile(self, series: pd.Series) -> list:
        """使用分位数方法检测异常值"""
        lower_bound = series.quantile(0.01)
        upper_bound = series.quantile(0.99)
        outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outlier_indices
    
    def _detect_outliers_stock_specific(self, series: pd.Series) -> list:
        """针对单只股票的异常值检测"""
        if len(series) < 5:  # 数据点太少，不检测异常值
            return []
        
        # 使用更宽松的Z-score阈值（5而不是3）
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_indices = series.dropna().index[z_scores > 5].tolist()
        return outlier_indices
    
    def _detect_outliers_loose(self, series: pd.Series) -> list:
        """使用更宽松的异常值检测方法"""
        # 使用1%和99%分位数作为边界，而不是IQR方法
        lower_bound = series.quantile(0.005)  # 更宽松的下界
        upper_bound = series.quantile(0.995)  # 更宽松的上界
        outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outlier_indices
    
    def handle_outliers(self, data: pd.DataFrame, outliers_info: Dict, data_type: str = "price") -> pd.DataFrame:
        """
        优化的异常值处理方法
        
        参数:
        data: 数据框
        outliers_info: 异常值信息字典
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        cleaned_data: 处理后的数据框
        """
        self.logger.info(f"\n===== 优化处理{data_type}数据异常值 =====")
        
        # 复制数据
        cleaned_data = data.copy()
        
        # 处理每个列的异常值
        for col, info in outliers_info["combined_outliers"].items():
            if col in ['stock_code']:  # 跳过股票代码等非数值型数据
                continue
                
            outlier_indices = info["indices"]
            
            if len(outlier_indices) > 0:
                self.logger.info(f"处理列 '{col}' 的 {len(outlier_indices)} 个异常值:")
                
                # 记录原始异常值
                original_values = cleaned_data.loc[outlier_indices, col].copy()
                
                # 使用Winsorization方法处理异常值（将异常值替换为边界值）
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 将异常值替换为边界值
                cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
                
                self.logger.info(f"  - 使用Winsorization方法处理异常值")
                self.logger.info(f"  - 下界值: {lower_bound:.4f}, 上界值: {upper_bound:.4f}")
                self.logger.info(f"  - 替换前部分值: {original_values.head().tolist()}")
        
        self.logger.info(f"\n异常值处理前数据形状: {data.shape}")
        self.logger.info(f"异常值处理后数据形状: {cleaned_data.shape}")
        
        return cleaned_data
    
    def calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """
        计算数据质量分数
        
        参数:
        data: 数据框
        
        返回:
        数据质量分数 (0-100)
        """
        # 计算完整性分数
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness_score = (1 - missing_cells / total_cells) * 100
        
        # 计算一致性分数（基于数值列的标准差）
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # 计算各数值列的变异系数，取平均值
            cv_scores = []
            for col in numeric_cols:
                if data[col].std() > 0:
                    cv = data[col].std() / data[col].mean()
                    cv_scores.append(min(cv, 1))  # 限制在0-1之间
            
            consistency_score = (1 - np.mean(cv_scores)) * 100 if cv_scores else 100
        else:
            consistency_score = 100
        
        # 计算唯一性分数（基于重复行）
        duplicate_rows = data.duplicated().sum()
        uniqueness_score = (1 - duplicate_rows / len(data)) * 100
        
        # 综合分数（加权平均）
        quality_score = 0.5 * completeness_score + 0.3 * consistency_score + 0.2 * uniqueness_score
        
        return quality_score
    
    def save_cleaned_data(self):
        """保存清洗后的数据到对应的分类文件夹"""
        self.logger.info("\n===== 保存清洗后的数据 =====")
        
        # 保存清洗后的价格数据
        if self.cleaned_price_data is not None:
            price_output_path = os.path.join(self.output_dir, "cleaned_data", "cleaned_price_data.csv")
            self.cleaned_price_data.to_csv(price_output_path, index=False)
            self.logger.info(f"清洗后的价格数据已保存至: {price_output_path}")
        
        # 保存清洗后的财务数据
        if self.cleaned_financial_data is not None:
            financial_output_path = os.path.join(self.output_dir, "cleaned_data", "cleaned_financial_data.csv")
            self.cleaned_financial_data.to_csv(financial_output_path, index=False)
            self.logger.info(f"清洗后的财务数据已保存至: {financial_output_path}")
        
        # 保存清洗报告
        report_output_path = os.path.join(self.output_dir, "reports", f"data_cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_report, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info(f"数据清洗报告已保存至: {report_output_path}")
    
    def generate_cleaning_report(self):
        """生成数据清洗报告"""
        self.logger.info("\n===== 生成数据清洗报告 =====")
        
        # 计算数据质量分数
        if self.cleaned_price_data is not None:
            self.cleaning_report["price_data"]["data_quality_score"] = float(self.calculate_data_quality_score(self.cleaned_price_data))
            
        if self.cleaned_financial_data is not None:
            self.cleaning_report["financial_data"]["data_quality_score"] = float(self.calculate_data_quality_score(self.cleaned_financial_data))
        
        # 记录处理时间
        self.cleaning_report["processing_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 将numpy类型转换为Python原生类型，以便JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 转换报告中的numpy类型
        serializable_report = convert_numpy_types(self.cleaning_report)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "reports", "data_cleaning_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"数据清洗报告已保存至: {report_path}")
        
        # 打印报告摘要
        self.logger.info("\n===== 数据清洗报告摘要 =====")
        
        if "price_data" in self.cleaning_report:
            price_report = self.cleaning_report["price_data"]
            self.logger.info(f"\n价格数据:")
            self.logger.info(f"原始形状: {price_report.get('original_shape', 'N/A')}")
            self.logger.info(f"清洗后形状: {price_report.get('cleaned_shape', 'N/A')}")
            self.logger.info(f"移除非交易日记录数: {price_report.get('non_trading_days_removed', 0)}")
            quality_score = price_report.get('data_quality_score')
            if quality_score is not None:
                self.logger.info(f"数据质量分数: {quality_score:.2f}")
            else:
                self.logger.info("数据质量分数: N/A")
            
        if "financial_data" in self.cleaning_report:
            financial_report = self.cleaning_report["financial_data"]
            self.logger.info(f"\n财务数据:")
            self.logger.info(f"原始形状: {financial_report.get('original_shape', 'N/A')}")
            self.logger.info(f"清洗后形状: {financial_report.get('cleaned_shape', 'N/A')}")
            quality_score = financial_report.get('data_quality_score')
            if quality_score is not None:
                self.logger.info(f"数据质量分数: {quality_score:.2f}")
            else:
                self.logger.info("数据质量分数: N/A")
            
        self.logger.info(f"\n处理时间: {self.cleaning_report.get('processing_time', 'N/A')}")
        
        return self.cleaning_report
    
    def clean_price_data(self) -> pd.DataFrame:
        """清洗价格数据"""
        self.logger.info("\n===== 开始清洗价格数据 =====")
        
        if self.price_data is None:
            self.logger.error("价格数据未加载")
            return None
            
        # 1. 过滤非交易日数据
        self.logger.info("步骤1: 过滤非交易日数据")
        self.cleaned_price_data = self.filter_non_trading_days(self.price_data)
        
        # 2. 分析缺失值模式
        self.logger.info("步骤2: 分析缺失值模式")
        self.analyze_missing_pattern(self.cleaned_price_data, "price")
        
        # 3. 处理缺失值
        self.logger.info("步骤3: 处理缺失值")
        self.cleaned_price_data = self.handle_missing_values(self.cleaned_price_data, is_price_data=True)
        
        # 4. 检测和处理异常值
        self.logger.info("步骤4: 检测和处理异常值")
        self.cleaned_price_data = self.detect_outliers(self.cleaned_price_data, is_price_data=True)
        
        # 记录清洗后的数据形状
        self.cleaning_report["price_data"]["cleaned_shape"] = self.cleaned_price_data.shape
        
        # 保存清洗后的数据
        cleaned_price_path = os.path.join(self.output_dir, "cleaned_data", "cleaned_price_data.csv")
        self.cleaned_price_data.to_csv(cleaned_price_path, index=False)
        self.logger.info(f"清洗后的价格数据已保存至: {cleaned_price_path}")
        
        return self.cleaned_price_data
    
    def clean_financial_data(self) -> pd.DataFrame:
        """清洗财务数据"""
        self.logger.info("\n===== 开始清洗财务数据 =====")
        
        if self.financial_data is None:
            self.logger.error("财务数据未加载")
            return None
            
        # 1. 分析缺失值模式
        self.logger.info("步骤1: 分析缺失值模式")
        self.analyze_missing_pattern(self.financial_data, "financial")
        
        # 2. 处理缺失值
        self.logger.info("步骤2: 处理缺失值")
        self.cleaned_financial_data = self.handle_missing_values(self.financial_data, is_price_data=False)
        
        # 3. 检测和处理异常值
        self.logger.info("步骤3: 检测和处理异常值")
        self.cleaned_financial_data = self.detect_outliers(self.cleaned_financial_data, is_price_data=False)
        
        # 记录清洗后的数据形状
        self.cleaning_report["financial_data"]["cleaned_shape"] = self.cleaned_financial_data.shape
        
        # 保存清洗后的数据
        cleaned_financial_path = os.path.join(self.output_dir, "cleaned_data", "cleaned_financial_data.csv")
        self.cleaned_financial_data.to_csv(cleaned_financial_path, index=False)
        self.logger.info(f"清洗后的财务数据已保存至: {cleaned_financial_path}")
        
        return self.cleaned_financial_data
    
    def run_cleaning_pipeline(self):
        """运行完整的数据清洗流程"""
        self.logger.info("===== 开始数据清洗流程 =====")
        start_time = datetime.now()
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 分析数据结构
        self.analyze_data_structure()
        
        # 3. 清洗价格数据
        self.clean_price_data()
        
        # 4. 清洗财务数据
        self.clean_financial_data()
        
        # 5. 生成清洗报告
        self.generate_cleaning_report()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        self.logger.info(f"\n数据清洗流程完成，总耗时: {processing_time:.2f} 秒")
        
        return self.cleaned_price_data, self.cleaned_financial_data


def main():
    """主函数"""
    # 设置文件路径
    price_data_path = "data/daliy_prices_merged/all_price_data.csv"
    financial_data_path = "data/financial_data_merged/all_financial_data.csv"
    non_trading_days_path = "data/components/non_trading_days.csv"  # 使用非交易日文件
    
    # 创建数据清洗器实例
    cleaner = OptimizedDataCleaner(
        price_data_path=price_data_path,
        financial_data_path=financial_data_path,
        trading_days_path=non_trading_days_path,  # 传入非交易日路径
        output_dir="data_clean_result"
    )
    
    # 运行清洗流程
    start_time = datetime.now()
    print(f"开始数据清洗: {start_time}")
    
    try:
        cleaned_price_data, cleaned_financial_data = cleaner.run_cleaning_pipeline()
        
        end_time = datetime.now()
        print(f"数据清洗完成: {end_time}")
        print(f"总耗时: {end_time - start_time}")
        
        print(f"\n清洗结果已保存至: {cleaner.output_dir}")
        
    except Exception as e:
        print(f"数据清洗过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()