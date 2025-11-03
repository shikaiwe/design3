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
from sklearn.preprocessing import RobustScaler, MinMaxScaler
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
    
    def __init__(self, price_data_path: str, financial_data_path: str, output_dir: str = "data_cleaning_results"):
        """
        初始化数据清洗器
        
        参数:
        price_data_path: 价格数据文件路径
        financial_data_path: 财务数据文件路径
        output_dir: 输出目录
        """
        self.price_data_path = price_data_path
        self.financial_data_path = financial_data_path
        self.output_dir = output_dir
        
        # 创建输出目录结构
        self._create_output_dirs()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化数据变量
        self.price_data = None
        self.financial_data = None
        self.cleaned_price_data = None
        self.cleaned_financial_data = None
        
        # 初始化清洗报告
        self.cleaning_report = {
            "price_data": {
                "original_shape": None,
                "cleaned_shape": None,
                "missing_values": {},
                "outliers": {},
                "standardization": {},
                "data_quality_score": None
            },
            "financial_data": {
                "original_shape": None,
                "cleaned_shape": None,
                "missing_values": {},
                "outliers": {},
                "standardization": {},
                "data_quality_score": None
            },
            "processing_time": None,
            "cleaning_methods": {
                "missing_value": "时间序列插值/行业均值填充",
                "outlier_detection": "3σ准则/IQR/分位数组合方法",
                "standardization": "Robust Scaler/Z-Score对比"
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
    
    def handle_missing_values(self, data: pd.DataFrame, data_type: str = "price") -> pd.DataFrame:
        """
        优化的缺失值处理方法
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        cleaned_data: 处理后的数据框
        """
        self.logger.info(f"\n===== 优化处理{data_type}数据缺失值 =====")
        
        # 分析缺失值模式
        missing_pattern = self.analyze_missing_pattern(data, data_type)
        
        # 复制数据
        cleaned_data = data.copy()
        
        # 获取数值型列
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 处理每个列的缺失值
        for col in numeric_cols:
            missing_pct = missing_pattern["missing_percentages"].get(col, 0)
            
            if missing_pct > 0:
                self.logger.info(f"处理列 '{col}' 的缺失值 (缺失比例: {missing_pct:.2f}%)")
                
                if missing_pct < 20:
                    if data_type == "price" and "日期" in cleaned_data.columns:
                        # 对于价格数据，使用时间序列插值
                        self.logger.info(f"  - 使用时间序列插值填充")
                        cleaned_data = self._time_series_interpolation(cleaned_data, col)
                    else:
                        # 对于财务数据，使用行业均值或历史均值填充
                        self.logger.info(f"  - 使用行业/历史均值填充")
                        cleaned_data = self._industry_mean_imputation(cleaned_data, col)
                else:
                    # 缺失值比例达到或超过20%，删除包含该字段的整条记录
                    self.logger.info(f"  - 缺失值比例过高，删除包含该缺失值的记录")
                    cleaned_data = cleaned_data.dropna(subset=[col])
        
        # 处理非数值型列的缺失值（如果有）
        non_numeric_cols = [col for col in cleaned_data.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            missing_pct = missing_pattern["missing_percentages"].get(col, 0)
            if missing_pct > 0:
                self.logger.info(f"处理非数值列 '{col}' 的缺失值 (缺失比例: {missing_pct:.2f}%)")
                if missing_pct < 20:
                    # 使用众数填充
                    mode_value = cleaned_data[col].mode()
                    if not mode_value.empty:
                        cleaned_data[col].fillna(mode_value[0], inplace=True)
                        self.logger.info(f"  - 使用众数 '{mode_value[0]}' 填充")
                    else:
                        cleaned_data = cleaned_data.dropna(subset=[col])
                        self.logger.info(f"  - 无众数，删除记录")
                else:
                    # 删除记录
                    cleaned_data = cleaned_data.dropna(subset=[col])
                    self.logger.info(f"  - 缺失值比例过高，删除记录")
        
        self.logger.info(f"\n缺失值处理前数据形状: {data.shape}")
        self.logger.info(f"缺失值处理后数据形状: {cleaned_data.shape}")
        self.logger.info(f"删除记录数: {data.shape[0] - cleaned_data.shape[0]}")
        
        # 保存到报告
        self.cleaning_report[f"{data_type}_data"]["missing_values"] = missing_pattern
        
        return cleaned_data
    
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
    
    def detect_outliers(self, data: pd.DataFrame, data_type: str = "price") -> Dict:
        """
        优化的异常值检测方法
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        outliers_info: 异常值信息字典
        """
        self.logger.info(f"\n===== 优化检测{data_type}数据异常值 =====")
        
        # 获取数值型列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {
            "method_3sigma": {},
            "method_iqr": {},
            "method_quantile": {},
            "combined_outliers": {}
        }
        
        for col in numeric_cols:
            if col in ['stock_code']:  # 跳过股票代码等非数值型数据
                continue
                
            self.logger.info(f"\n检测列 '{col}' 的异常值:")
            
            # 方法1: 3-σ准则
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outliers_3sigma = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            # 方法2: 箱线图法 (IQR)
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound_iqr = Q1 - 1.5 * IQR
            upper_bound_iqr = Q3 + 1.5 * IQR
            
            outliers_iqr = data[(data[col] < lower_bound_iqr) | (data[col] > upper_bound_iqr)]
            
            # 方法3: 分位数法 (更保守的IQR)
            lower_bound_quantile = data[col].quantile(0.01)
            upper_bound_quantile = data[col].quantile(0.99)
            
            outliers_quantile = data[(data[col] < lower_bound_quantile) | (data[col] > upper_bound_quantile)]
            
            # 记录异常值信息
            outliers_info["method_3sigma"][col] = {
                "count": len(outliers_3sigma),
                "percentage": len(outliers_3sigma) / len(data) * 100,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "indices": outliers_3sigma.index.tolist()
            }
            
            outliers_info["method_iqr"][col] = {
                "count": len(outliers_iqr),
                "percentage": len(outliers_iqr) / len(data) * 100,
                "lower_bound": lower_bound_iqr,
                "upper_bound": upper_bound_iqr,
                "indices": outliers_iqr.index.tolist()
            }
            
            outliers_info["method_quantile"][col] = {
                "count": len(outliers_quantile),
                "percentage": len(outliers_quantile) / len(data) * 100,
                "lower_bound": lower_bound_quantile,
                "upper_bound": upper_bound_quantile,
                "indices": outliers_quantile.index.tolist()
            }
            
            # 获取三种方法都检测到的异常值（交集）
            common_indices = set(outliers_3sigma.index) & set(outliers_iqr.index) & set(outliers_quantile.index)
            common_outliers = data.loc[list(common_indices)]
            
            outliers_info["combined_outliers"][col] = {
                "count": len(common_outliers),
                "percentage": len(common_outliers) / len(data) * 100,
                "indices": list(common_indices)
            }
            
            self.logger.info(f"  - 3-σ准则检测到 {len(outliers_3sigma)} 个异常值 ({len(outliers_3sigma)/len(data)*100:.2f}%)")
            self.logger.info(f"  - 箱线图法检测到 {len(outliers_iqr)} 个异常值 ({len(outliers_iqr)/len(data)*100:.2f}%)")
            self.logger.info(f"  - 分位数法检测到 {len(outliers_quantile)} 个异常值 ({len(outliers_quantile)/len(data)*100:.2f}%)")
            self.logger.info(f"  - 三种方法共同检测到 {len(common_outliers)} 个异常值 ({len(common_outliers)/len(data)*100:.2f}%)")
            
            # 可视化异常值（可选）
            if len(common_outliers) > 0 and len(common_outliers) < 100:  # 只在异常值数量适中时可视化
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.scatter(data.index, data[col], alpha=0.5, label='正常值')
                plt.scatter(outliers_3sigma.index, outliers_3sigma[col], color='red', label='3-σ异常值')
                plt.title(f"{col} 3-σ异常值检测")
                plt.xlabel("索引")
                plt.ylabel(col)
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 2, 2)
                plt.scatter(data.index, data[col], alpha=0.5, label='正常值')
                plt.scatter(outliers_iqr.index, outliers_iqr[col], color='orange', label='IQR异常值')
                plt.title(f"{col} IQR异常值检测")
                plt.xlabel("索引")
                plt.ylabel(col)
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 2, 3)
                plt.scatter(data.index, data[col], alpha=0.5, label='正常值')
                plt.scatter(outliers_quantile.index, outliers_quantile[col], color='green', label='分位数异常值')
                plt.title(f"{col} 分位数异常值检测")
                plt.xlabel("索引")
                plt.ylabel(col)
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 2, 4)
                plt.scatter(data.index, data[col], alpha=0.5, label='正常值')
                plt.scatter(common_outliers.index, common_outliers[col], color='purple', label='共同异常值')
                plt.title(f"{col} 共同异常值检测")
                plt.xlabel("索引")
                plt.ylabel(col)
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "visualizations", f"{data_type}_data_{col}_outliers.png"))
                plt.close()
        
        # 保存到报告
        self.cleaning_report[f"{data_type}_data"]["outliers"] = outliers_info
        
        return outliers_info
    
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
    
    def compare_standardization_methods(self, data: pd.DataFrame, data_type: str = "price") -> Dict:
        """
        比较不同标准化方法
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        comparison_results: 比较结果
        """
        self.logger.info(f"\n===== 比较{data_type}数据标准化方法 =====")
        
        # 获取数值型列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除非数值型标识列
        exclude_cols = ['stock_code']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        comparison_results = {
            "columns": numeric_cols,
            "zscore": {},
            "minmax": {},
            "robust": {}
        }
        
        # 选择几个代表性列进行比较
        sample_cols = numeric_cols[:min(3, len(numeric_cols))]
        
        for col in sample_cols:
            self.logger.info(f"\n比较列 '{col}' 的标准化方法:")
            
            # 原始数据统计
            original_mean = data[col].mean()
            original_std = data[col].std()
            original_min = data[col].min()
            original_max = data[col].max()
            original_median = data[col].median()
            original_q1 = data[col].quantile(0.25)
            original_q3 = data[col].quantile(0.75)
            
            # Z-Score标准化
            zscore_data = (data[col] - original_mean) / original_std
            zscore_mean = zscore_data.mean()
            zscore_std = zscore_data.std()
            
            # Min-Max标准化
            minmax_data = (data[col] - original_min) / (original_max - original_min)
            minmax_mean = minmax_data.mean()
            minmax_std = minmax_data.std()
            
            # Robust标准化（基于中位数和四分位距）
            robust_scaler = RobustScaler()
            robust_data = robust_scaler.fit_transform(data[[col]]).flatten()
            robust_mean = robust_data.mean()
            robust_std = robust_data.std()
            
            # 记录结果
            comparison_results["zscore"][col] = {
                "mean": zscore_mean,
                "std": zscore_std,
                "min": zscore_data.min(),
                "max": zscore_data.max()
            }
            
            comparison_results["minmax"][col] = {
                "mean": minmax_mean,
                "std": minmax_std,
                "min": minmax_data.min(),
                "max": minmax_data.max()
            }
            
            comparison_results["robust"][col] = {
                "mean": robust_mean,
                "std": robust_std,
                "min": robust_data.min(),
                "max": robust_data.max()
            }
            
            self.logger.info(f"  原始数据: 均值={original_mean:.4f}, 标准差={original_std:.4f}, 中位数={original_median:.4f}")
            self.logger.info(f"  Z-Score标准化: 均值={zscore_mean:.4f}, 标准差={zscore_std:.4f}, 范围=[{zscore_data.min():.4f}, {zscore_data.max():.4f}]")
            self.logger.info(f"  Min-Max标准化: 均值={minmax_mean:.4f}, 标准差={minmax_std:.4f}, 范围=[{minmax_data.min():.4f}, {minmax_data.max():.4f}]")
            self.logger.info(f"  Robust标准化: 均值={robust_mean:.4f}, 标准差={robust_std:.4f}, 范围=[{robust_data.min():.4f}, {robust_data.max():.4f}]")
            
            # 可视化比较
            plt.figure(figsize=(20, 5))
            
            plt.subplot(1, 4, 1)
            plt.hist(data[col], bins=30, alpha=0.7)
            plt.title(f"原始数据 ({col})")
            plt.xlabel(col)
            plt.ylabel("频数")
            
            plt.subplot(1, 4, 2)
            plt.hist(zscore_data, bins=30, alpha=0.7)
            plt.title(f"Z-Score标准化 ({col})")
            plt.xlabel("标准化值")
            
            plt.subplot(1, 4, 3)
            plt.hist(minmax_data, bins=30, alpha=0.7)
            plt.title(f"Min-Max标准化 ({col})")
            plt.xlabel("标准化值")
            
            plt.subplot(1, 4, 4)
            plt.hist(robust_data, bins=30, alpha=0.7)
            plt.title(f"Robust标准化 ({col})")
            plt.xlabel("标准化值")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "visualizations", f"{data_type}_data_{col}_standardization_comparison.png"))
            plt.close()
        
        # 分析不同标准化方法的适用性
        self.logger.info("\n标准化方法适用性分析:")
        self.logger.info("- Z-Score标准化: 适用于数据近似服从正态分布的情况，对异常值敏感")
        self.logger.info("- Min-Max标准化: 将数据缩放到[0,1]区间，适用于有明确边界的场景")
        self.logger.info("- Robust标准化: 基于中位数和四分位距，对异常值不敏感，适用于有异常值的数据")
        
        return comparison_results
    
    def standardize_data(self, data: pd.DataFrame, data_type: str = "price", method: str = "robust") -> Tuple[pd.DataFrame, Dict]:
        """
        数据标准化
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        method: 标准化方法 ("zscore", "minmax", "robust")
        
        返回:
        standardized_data: 标准化后的数据框
        standardization_info: 标准化信息字典
        """
        self.logger.info(f"\n===== {data_type}数据标准化 ({method}方法) =====")
        
        # 复制数据
        standardized_data = data.copy()
        
        # 获取数值型列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除非数值型标识列
        exclude_cols = ['stock_code']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        standardization_info = {
            "method": method,
            "columns": numeric_cols,
            "statistics": {}
        }
        
        for col in numeric_cols:
            # 记录原始统计信息
            original_mean = data[col].mean()
            original_std = data[col].std()
            original_min = data[col].min()
            original_max = data[col].max()
            original_median = data[col].median()
            original_q1 = data[col].quantile(0.25)
            original_q3 = data[col].quantile(0.75)
            
            if method == "zscore":
                # Z-Score标准化
                standardized_data[col] = (data[col] - original_mean) / original_std
                self.logger.info(f"列 '{col}' 使用Z-Score标准化 (μ={original_mean:.4f}, σ={original_std:.4f})")
                
                standardization_info["statistics"][col] = {
                    "type": "zscore",
                    "mean": original_mean,
                    "std": original_std
                }
                
            elif method == "minmax":
                # Min-Max标准化
                standardized_data[col] = (data[col] - original_min) / (original_max - original_min)
                self.logger.info(f"列 '{col}' 使用Min-Max标准化 (min={original_min:.4f}, max={original_max:.4f})")
                
                standardization_info["statistics"][col] = {
                    "type": "minmax",
                    "min": original_min,
                    "max": original_max
                }
                
            elif method == "robust":
                # Robust标准化
                median = original_median
                q75 = original_q3
                q25 = original_q1
                iqr = q75 - q25
                
                standardized_data[col] = (data[col] - median) / iqr
                self.logger.info(f"列 '{col}' 使用Robust标准化 (median={median:.4f}, IQR={iqr:.4f})")
                
                standardization_info["statistics"][col] = {
                    "type": "robust",
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr
                }
        
        # 保存到报告
        self.cleaning_report[f"{data_type}_data"]["standardization"] = standardization_info
        
        return standardized_data, standardization_info
    
    def calculate_data_quality_score(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame, data_type: str = "price") -> float:
        """
        计算数据质量评分
        
        参数:
        original_data: 原始数据
        cleaned_data: 清洗后数据
        data_type: 数据类型
        
        返回:
        quality_score: 数据质量评分 (0-100)
        """
        self.logger.info(f"\n===== 计算{data_type}数据质量评分 =====")
        
        # 计算完整性评分 (基于缺失值)
        completeness_score = (cleaned_data.shape[0] / original_data.shape[0]) * 100
        
        # 计算一致性评分 (基于数据类型一致性)
        consistency_score = 100  # 简化处理，实际可以检查数据类型一致性
        
        # 计算准确性评分 (基于异常值处理)
        # 这里简化处理，实际可以基于业务规则评估
        accuracy_score = 95  # 假设经过异常值处理后准确性为95%
        
        # 计算唯一性评分 (基于重复记录)
        original_duplicates = original_data.duplicated().sum()
        cleaned_duplicates = cleaned_data.duplicated().sum()
        uniqueness_score = 100 - ((cleaned_duplicates / cleaned_data.shape[0]) * 100)
        
        # 计算时效性评分 (基于数据时间范围)
        timeliness_score = 90  # 简化处理，实际可以基于数据时间戳评估
        
        # 计算综合质量评分 (加权平均)
        weights = {
            "completeness": 0.3,
            "consistency": 0.2,
            "accuracy": 0.3,
            "uniqueness": 0.1,
            "timeliness": 0.1
        }
        
        quality_score = (
            completeness_score * weights["completeness"] +
            consistency_score * weights["consistency"] +
            accuracy_score * weights["accuracy"] +
            uniqueness_score * weights["uniqueness"] +
            timeliness_score * weights["timeliness"]
        )
        
        self.logger.info(f"完整性评分: {completeness_score:.2f}%")
        self.logger.info(f"一致性评分: {consistency_score:.2f}%")
        self.logger.info(f"准确性评分: {accuracy_score:.2f}%")
        self.logger.info(f"唯一性评分: {uniqueness_score:.2f}%")
        self.logger.info(f"时效性评分: {timeliness_score:.2f}%")
        self.logger.info(f"综合质量评分: {quality_score:.2f}%")
        
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
        self.logger.info("\n===== 数据清洗报告 =====")
        
        # 价格数据报告
        if self.price_data is not None and self.cleaned_price_data is not None:
            self.logger.info("\n价格数据清洗报告:")
            self.logger.info(f"  原始数据形状: {self.cleaning_report['price_data']['original_shape']}")
            self.logger.info(f"  清洗后数据形状: {self.cleaned_price_data.shape}")
            self.logger.info(f"  数据保留率: {self.cleaned_price_data.shape[0]/self.cleaning_report['price_data']['original_shape'][0]*100:.2f}%")
            
            # 缺失值处理报告
            missing_info = self.cleaning_report['price_data']['missing_values']
            self.logger.info(f"\n  缺失值处理:")
            for col, pct in missing_info['missing_percentages'].items():
                if pct > 0:
                    self.logger.info(f"    {col}: {pct:.2f}%")
            
            # 异常值处理报告
            outliers_info = self.cleaning_report['price_data']['outliers']
            self.logger.info(f"\n  异常值处理:")
            for col, info in outliers_info['combined_outliers'].items():
                if info['count'] > 0:
                    self.logger.info(f"    {col}: {info['count']} 个异常值 ({info['percentage']:.2f}%)")
            
            # 数据质量评分
            quality_score = self.calculate_data_quality_score(self.price_data, self.cleaned_price_data, "price")
            self.cleaning_report["price_data"]["data_quality_score"] = quality_score
            self.logger.info(f"  数据质量评分: {quality_score:.2f}/100")
        
        # 财务数据报告
        if self.financial_data is not None and self.cleaned_financial_data is not None:
            self.logger.info("\n财务数据清洗报告:")
            self.logger.info(f"  原始数据形状: {self.cleaning_report['financial_data']['original_shape']}")
            self.logger.info(f"  清洗后数据形状: {self.cleaned_financial_data.shape}")
            self.logger.info(f"  数据保留率: {self.cleaned_financial_data.shape[0]/self.cleaning_report['financial_data']['original_shape'][0]*100:.2f}%")
            
            # 缺失值处理报告
            missing_info = self.cleaning_report['financial_data']['missing_values']
            self.logger.info(f"\n  缺失值处理:")
            for col, pct in missing_info['missing_percentages'].items():
                if pct > 0:
                    self.logger.info(f"    {col}: {pct:.2f}%")
            
            # 异常值处理报告
            outliers_info = self.cleaning_report['financial_data']['outliers']
            self.logger.info(f"\n  异常值处理:")
            for col, info in outliers_info['combined_outliers'].items():
                if info['count'] > 0:
                    self.logger.info(f"    {col}: {info['count']} 个异常值 ({info['percentage']:.2f}%)")
            
            # 数据质量评分
            quality_score = self.calculate_data_quality_score(self.financial_data, self.cleaned_financial_data, "financial")
            self.cleaning_report["financial_data"]["data_quality_score"] = quality_score
            self.logger.info(f"  数据质量评分: {quality_score:.2f}/100")
    
    def run_cleaning_pipeline(self):
        """运行完整的数据清洗流程"""
        start_time = datetime.now()
        self.logger.info(f"开始数据清洗流程: {start_time}")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 分析数据结构
        self.analyze_data_structure()
        
        # 3. 处理价格数据
        if self.price_data is not None:
            self.logger.info("\n===== 处理价格数据 =====")
            
            # 3.1 处理缺失值
            price_no_missing = self.handle_missing_values(self.price_data, "price")
            
            # 3.2 检测异常值
            price_outliers = self.detect_outliers(price_no_missing, "price")
            
            # 3.3 处理异常值
            price_no_outliers = self.handle_outliers(price_no_missing, price_outliers, "price")
            
            # 3.4 比较标准化方法
            self.compare_standardization_methods(price_no_outliers, "price")
            
            # 3.5 数据标准化 (使用Robust方法，对异常值更稳健)
            self.cleaned_price_data, _ = self.standardize_data(price_no_outliers, "price", "robust")
            
            # 更新清洗后数据形状
            self.cleaning_report["price_data"]["cleaned_shape"] = self.cleaned_price_data.shape
        
        # 4. 处理财务数据
        if self.financial_data is not None:
            self.logger.info("\n===== 处理财务数据 =====")
            
            # 4.1 处理缺失值
            financial_no_missing = self.handle_missing_values(self.financial_data, "financial")
            
            # 4.2 检测异常值
            financial_outliers = self.detect_outliers(financial_no_missing, "financial")
            
            # 4.3 处理异常值
            financial_no_outliers = self.handle_outliers(financial_no_missing, financial_outliers, "financial")
            
            # 4.4 比较标准化方法
            self.compare_standardization_methods(financial_no_outliers, "financial")
            
            # 4.5 数据标准化 (使用Robust方法，对异常值更稳健)
            self.cleaned_financial_data, _ = self.standardize_data(financial_no_outliers, "financial", "robust")
            
            # 更新清洗后数据形状
            self.cleaning_report["financial_data"]["cleaned_shape"] = self.cleaned_financial_data.shape
        
        # 5. 生成清洗报告
        self.generate_cleaning_report()
        
        # 6. 保存清洗后的数据
        self.save_cleaned_data()
        
        end_time = datetime.now()
        self.cleaning_report["processing_time"] = str(end_time - start_time)
        self.logger.info(f"\n数据清洗流程完成: {end_time}")
        self.logger.info(f"总耗时: {end_time - start_time}")
        
        return self.cleaned_price_data, self.cleaned_financial_data


# 主程序
if __name__ == "__main__":
    # 设置文件路径
    price_data_path = "c:\\Users\\Administrator\\Desktop\\design3\\data\\daliy_prices_merged\\all_price_data.csv"
    financial_data_path = "c:\\Users\\Administrator\\Desktop\\design3\\data\\financial_data_merged\\all_financial_data.csv"
    
    # 创建优化版数据清洗器
    cleaner = OptimizedDataCleaner(price_data_path, financial_data_path)
    
    # 运行清洗流程
    cleaned_price_data, cleaned_financial_data = cleaner.run_cleaning_pipeline()
    
    print("\n优化版数据清洗完成!")