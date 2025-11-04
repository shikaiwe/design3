#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统性数据清洗脚本
对股票价格数据和财务数据进行清洗处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy import stats
import warnings
import os
import json
from datetime import datetime

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

class DataCleaner:
    """数据清洗类"""
    
    def __init__(self, price_data_path, financial_data_path):
        """
        初始化数据清洗器
        
        参数:
        price_data_path: 价格数据文件路径
        financial_data_path: 财务数据文件路径
        """
        self.price_data_path = price_data_path
        self.financial_data_path = financial_data_path
        self.price_data = None
        self.financial_data = None
        self.cleaned_price_data = None
        self.cleaned_financial_data = None
        self.cleaning_report = {
            "price_data": {
                "original_shape": None,
                "cleaned_shape": None,
                "missing_values": {},
                "outliers": {},
                "standardization": {}
            },
            "financial_data": {
                "original_shape": None,
                "cleaned_shape": None,
                "missing_values": {},
                "outliers": {},
                "standardization": {}
            },
            "processing_time": None
        }
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载价格数据
        try:
            self.price_data = pd.read_csv(self.price_data_path)
            print(f"价格数据加载成功，形状: {self.price_data.shape}")
        except Exception as e:
            print(f"价格数据加载失败: {e}")
            
        # 加载财务数据
        try:
            self.financial_data = pd.read_csv(self.financial_data_path)
            print(f"财务数据加载成功，形状: {self.financial_data.shape}")
        except Exception as e:
            print(f"财务数据加载失败: {e}")
            
        # 记录原始数据形状
        if self.price_data is not None:
            self.cleaning_report["price_data"]["original_shape"] = self.price_data.shape
            
        if self.financial_data is not None:
            self.cleaning_report["financial_data"]["original_shape"] = self.financial_data.shape
            
        return self.price_data, self.financial_data
    
    def analyze_data_structure(self):
        """分析数据结构"""
        print("\n===== 数据结构分析 =====")
        
        # 价格数据分析
        if self.price_data is not None:
            print("\n价格数据信息:")
            print(f"数据形状: {self.price_data.shape}")
            print(f"列名: {list(self.price_data.columns)}")
            print("\n数据类型:")
            print(self.price_data.dtypes)
            print("\n前5行数据:")
            print(self.price_data.head())
            
        # 财务数据分析
        if self.financial_data is not None:
            print("\n财务数据信息:")
            print(f"数据形状: {self.financial_data.shape}")
            print(f"列名: {list(self.financial_data.columns)}")
            print("\n数据类型:")
            print(self.financial_data.dtypes)
            print("\n前5行数据:")
            print(self.financial_data.head())
            
    def calculate_missing_values(self, data, data_type="price"):
        """
        计算缺失值比例
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        missing_info: 缺失值信息字典
        """
        print(f"\n===== 计算{data_type}数据缺失值比例 =====")
        
        missing_values = data.isnull().sum()
        total_records = len(data)
        missing_percentage = (missing_values / total_records) * 100
        
        missing_info = {
            "total_records": total_records,
            "missing_counts": missing_values.to_dict(),
            "missing_percentages": missing_percentage.to_dict()
        }
        
        # 打印缺失值信息
        print(f"总记录数: {total_records}")
        print("\n缺失值统计:")
        missing_df = pd.DataFrame({
            "缺失数量": missing_values,
            "缺失比例(%)": missing_percentage
        })
        print(missing_df[missing_df["缺失数量"] > 0])
        
        # 保存到报告
        self.cleaning_report[f"{data_type}_data"]["missing_values"] = missing_info
        
        return missing_info
    
    def handle_missing_values(self, data, data_type="price"):
        """
        处理缺失值
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        cleaned_data: 处理后的数据框
        """
        print(f"\n===== 处理{data_type}数据缺失值 =====")
        
        # 计算缺失值比例
        missing_info = self.calculate_missing_values(data, data_type)
        
        # 复制数据
        cleaned_data = data.copy()
        
        # 获取数值型列
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 处理每个列的缺失值
        for col in numeric_cols:
            missing_pct = missing_info["missing_percentages"].get(col, 0)
            
            if missing_pct > 0:
                print(f"处理列 '{col}' 的缺失值 (缺失比例: {missing_pct:.2f}%)")
                
                if missing_pct < 20:
                    # 缺失值比例低于20%，使用拉格朗日线性插值
                    print(f"  - 使用拉格朗日线性插值填充")
                    
                    # 获取非缺失值的索引和值
                    valid_indices = cleaned_data[col].notna()
                    valid_data = cleaned_data.loc[valid_indices, col]
                    
                    if len(valid_data) > 1:  # 确保有足够的点进行插值
                        # 创建插值函数
                        x = valid_data.index
                        y = valid_data.values
                        
                        # 使用线性插值（拉格朗日插值在数据点较多时可能不稳定）
                        f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
                        
                        # 填充缺失值
                        missing_indices = cleaned_data[col].isna()
                        if missing_indices.any():
                            cleaned_data.loc[missing_indices, col] = f(cleaned_data.loc[missing_indices].index)
                    else:
                        # 如果没有足够的点进行插值，使用均值填充
                        cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
                        print(f"  - 数据点不足，使用均值填充")
                else:
                    # 缺失值比例达到或超过20%，删除包含该字段的整条记录
                    print(f"  - 缺失值比例过高，删除包含该缺失值的记录")
                    cleaned_data = cleaned_data.dropna(subset=[col])
        
        # 处理非数值型列的缺失值（如果有）
        non_numeric_cols = [col for col in cleaned_data.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            missing_pct = missing_info["missing_percentages"].get(col, 0)
            if missing_pct > 0:
                print(f"处理非数值列 '{col}' 的缺失值 (缺失比例: {missing_pct:.2f}%)")
                if missing_pct < 20:
                    # 使用众数填充
                    mode_value = cleaned_data[col].mode()
                    if not mode_value.empty:
                        cleaned_data[col].fillna(mode_value[0], inplace=True)
                        print(f"  - 使用众数 '{mode_value[0]}' 填充")
                    else:
                        cleaned_data = cleaned_data.dropna(subset=[col])
                        print(f"  - 无众数，删除记录")
                else:
                    # 删除记录
                    cleaned_data = cleaned_data.dropna(subset=[col])
                    print(f"  - 缺失值比例过高，删除记录")
        
        print(f"\n缺失值处理前数据形状: {data.shape}")
        print(f"缺失值处理后数据形状: {cleaned_data.shape}")
        print(f"删除记录数: {data.shape[0] - cleaned_data.shape[0]}")
        
        return cleaned_data
    
    def detect_outliers(self, data, data_type="price"):
        """
        检测异常值
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        outliers_info: 异常值信息字典
        """
        print(f"\n===== 检测{data_type}数据异常值 =====")
        
        # 获取数值型列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {
            "method_3sigma": {},
            "method_iqr": {},
            "combined_outliers": {}
        }
        
        for col in numeric_cols:
            if col in ['stock_code']:  # 跳过股票代码等非数值型数据
                continue
                
            print(f"\n检测列 '{col}' 的异常值:")
            
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
            
            # 获取两种方法都检测到的异常值（交集）
            common_indices = set(outliers_3sigma.index) & set(outliers_iqr.index)
            common_outliers = data.loc[list(common_indices)]
            
            outliers_info["combined_outliers"][col] = {
                "count": len(common_outliers),
                "percentage": len(common_outliers) / len(data) * 100,
                "indices": list(common_indices)
            }
            
            print(f"  - 3-σ准则检测到 {len(outliers_3sigma)} 个异常值 ({len(outliers_3sigma)/len(data)*100:.2f}%)")
            print(f"  - 箱线图法检测到 {len(outliers_iqr)} 个异常值 ({len(outliers_iqr)/len(data)*100:.2f}%)")
            print(f"  - 两种方法共同检测到 {len(common_outliers)} 个异常值 ({len(common_outliers)/len(data)*100:.2f}%)")
            
            # 可视化异常值（可选）
            if len(common_outliers) > 0 and len(common_outliers) < 100:  # 只在异常值数量适中时可视化
                plt.figure(figsize=(10, 6))
                plt.scatter(data.index, data[col], alpha=0.5, label='正常值')
                plt.scatter(common_outliers.index, common_outliers[col], color='red', label='异常值')
                plt.title(f"{col} 异常值检测 (两种方法共同检测)")
                plt.xlabel("索引")
                plt.ylabel(col)
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{data_type}_data_{col}_outliers.png")
                plt.close()
        
        # 保存到报告
        self.cleaning_report[f"{data_type}_data"]["outliers"] = outliers_info
        
        return outliers_info
    
    def handle_outliers(self, data, outliers_info, data_type="price"):
        """
        处理异常值
        
        参数:
        data: 数据框
        outliers_info: 异常值信息字典
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        cleaned_data: 处理后的数据框
        """
        print(f"\n===== 处理{data_type}数据异常值 =====")
        
        # 复制数据
        cleaned_data = data.copy()
        
        # 处理每个列的异常值
        for col, info in outliers_info["combined_outliers"].items():
            if col in ['stock_code']:  # 跳过股票代码等非数值型数据
                continue
                
            outlier_indices = info["indices"]
            
            if len(outlier_indices) > 0:
                print(f"处理列 '{col}' 的 {len(outlier_indices)} 个异常值:")
                
                # 记录原始异常值
                original_values = cleaned_data.loc[outlier_indices, col].copy()
                
                # 使用中位数替换异常值（较为稳健的方法）
                median_value = cleaned_data[col].median()
                cleaned_data.loc[outlier_indices, col] = median_value
                
                print(f"  - 使用中位数 {median_value:.4f} 替换异常值")
                print(f"  - 替换前部分值: {original_values.head().tolist()}")
        
        print(f"\n异常值处理前数据形状: {data.shape}")
        print(f"异常值处理后数据形状: {cleaned_data.shape}")
        
        return cleaned_data
    
    def standardize_data(self, data, data_type="price", method="zscore"):
        """
        数据标准化
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        method: 标准化方法 ("zscore", "minmax")
        
        返回:
        standardized_data: 标准化后的数据框
        standardization_info: 标准化信息字典
        """
        print(f"\n===== {data_type}数据标准化 =====")
        
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
            
            if method == "zscore":
                # Z-Score标准化
                standardized_data[col] = (data[col] - original_mean) / original_std
                print(f"列 '{col}' 使用Z-Score标准化 (μ={original_mean:.4f}, σ={original_std:.4f})")
                
                standardization_info["statistics"][col] = {
                    "type": "zscore",
                    "mean": original_mean,
                    "std": original_std
                }
                
            elif method == "minmax":
                # Min-Max标准化
                standardized_data[col] = (data[col] - original_min) / (original_max - original_min)
                print(f"列 '{col}' 使用Min-Max标准化 (min={original_min:.4f}, max={original_max:.4f})")
                
                standardization_info["statistics"][col] = {
                    "type": "minmax",
                    "min": original_min,
                    "max": original_max
                }
        
        # 保存到报告
        self.cleaning_report[f"{data_type}_data"]["standardization"] = standardization_info
        
        return standardized_data, standardization_info
    
    def compare_standardization_methods(self, data, data_type="price"):
        """
        比较不同标准化方法
        
        参数:
        data: 数据框
        data_type: 数据类型 ("price" 或 "financial")
        
        返回:
        comparison_results: 比较结果
        """
        print(f"\n===== 比较{data_type}数据标准化方法 =====")
        
        # 获取数值型列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除非数值型标识列
        exclude_cols = ['stock_code']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        comparison_results = {
            "columns": numeric_cols,
            "zscore": {},
            "minmax": {}
        }
        
        # 选择几个代表性列进行比较
        sample_cols = numeric_cols[:min(3, len(numeric_cols))]
        
        for col in sample_cols:
            print(f"\n比较列 '{col}' 的标准化方法:")
            
            # 原始数据统计
            original_mean = data[col].mean()
            original_std = data[col].std()
            original_min = data[col].min()
            original_max = data[col].max()
            
            # Z-Score标准化
            zscore_data = (data[col] - original_mean) / original_std
            zscore_mean = zscore_data.mean()
            zscore_std = zscore_data.std()
            
            # Min-Max标准化
            minmax_data = (data[col] - original_min) / (original_max - original_min)
            minmax_mean = minmax_data.mean()
            minmax_std = minmax_data.std()
            
            # 记录结果
            comparison_results["zscore"][col] = {
                "mean": zscore_mean,
                "std": zscore_std
            }
            
            comparison_results["minmax"][col] = {
                "mean": minmax_mean,
                "std": minmax_std
            }
            
            print(f"  原始数据: 均值={original_mean:.4f}, 标准差={original_std:.4f}")
            print(f"  Z-Score标准化: 均值={zscore_mean:.4f}, 标准差={zscore_std:.4f}")
            print(f"  Min-Max标准化: 均值={minmax_mean:.4f}, 标准差={minmax_std:.4f}")
            
            # 可视化比较
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.hist(data[col], bins=30, alpha=0.7)
            plt.title(f"原始数据 ({col})")
            plt.xlabel(col)
            plt.ylabel("频数")
            
            plt.subplot(1, 3, 2)
            plt.hist(zscore_data, bins=30, alpha=0.7)
            plt.title(f"Z-Score标准化 ({col})")
            plt.xlabel("标准化值")
            
            plt.subplot(1, 3, 3)
            plt.hist(minmax_data, bins=30, alpha=0.7)
            plt.title(f"Min-Max标准化 ({col})")
            plt.xlabel("标准化值")
            
            plt.tight_layout()
            plt.savefig(f"{data_type}_data_{col}_standardization_comparison.png")
            plt.close()
        
        # 分析Z-Score标准化的适用性
        print("\nZ-Score标准化适用性分析:")
        print("- Z-Score标准化适用于数据近似服从正态分布的情况")
        print("- 标准化后的数据均值为0，标准差为1，便于不同量纲指标的比较")
        print("- 对异常值较为敏感，但在我们已处理异常值后影响较小")
        print("- 适用于后续使用欧氏距离、马氏距离等需要数据标准差一致的算法")
        
        return comparison_results
    
    def save_cleaned_data(self, output_dir="cleaned_data"):
        """保存清洗后的数据"""
        print("\n===== 保存清洗后的数据 =====")
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存清洗后的价格数据
        if self.cleaned_price_data is not None:
            price_output_path = os.path.join(output_dir, "cleaned_price_data.csv")
            self.cleaned_price_data.to_csv(price_output_path, index=False)
            print(f"清洗后的价格数据已保存至: {price_output_path}")
        
        # 保存清洗后的财务数据
        if self.cleaned_financial_data is not None:
            financial_output_path = os.path.join(output_dir, "cleaned_financial_data.csv")
            self.cleaned_financial_data.to_csv(financial_output_path, index=False)
            print(f"清洗后的财务数据已保存至: {financial_output_path}")
        
        # 保存清洗报告
        report_output_path = os.path.join(output_dir, "data_cleaning_report.json")
        with open(report_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_report, f, ensure_ascii=False, indent=2, default=str)
        print(f"数据清洗报告已保存至: {report_output_path}")
    
    def generate_cleaning_report(self):
        """生成数据清洗报告"""
        print("\n===== 数据清洗报告 =====")
        
        # 价格数据报告
        if self.price_data is not None and self.cleaned_price_data is not None:
            print("\n价格数据清洗报告:")
            print(f"  原始数据形状: {self.cleaning_report['price_data']['original_shape']}")
            print(f"  清洗后数据形状: {self.cleaned_price_data.shape}")
            print(f"  数据保留率: {self.cleaned_price_data.shape[0]/self.cleaning_report['price_data']['original_shape'][0]*100:.2f}%")
            
            # 缺失值处理报告
            missing_info = self.cleaning_report['price_data']['missing_values']
            print(f"\n  缺失值处理:")
            for col, pct in missing_info['missing_percentages'].items():
                if pct > 0:
                    print(f"    {col}: {pct:.2f}%")
            
            # 异常值处理报告
            outliers_info = self.cleaning_report['price_data']['outliers']
            print(f"\n  异常值处理:")
            for col, info in outliers_info['combined_outliers'].items():
                if info['count'] > 0:
                    print(f"    {col}: {info['count']} 个异常值 ({info['percentage']:.2f}%)")
        
        # 财务数据报告
        if self.financial_data is not None and self.cleaned_financial_data is not None:
            print("\n财务数据清洗报告:")
            print(f"  原始数据形状: {self.cleaning_report['financial_data']['original_shape']}")
            print(f"  清洗后数据形状: {self.cleaned_financial_data.shape}")
            print(f"  数据保留率: {self.cleaned_financial_data.shape[0]/self.cleaning_report['financial_data']['original_shape'][0]*100:.2f}%")
            
            # 缺失值处理报告
            missing_info = self.cleaning_report['financial_data']['missing_values']
            print(f"\n  缺失值处理:")
            for col, pct in missing_info['missing_percentages'].items():
                if pct > 0:
                    print(f"    {col}: {pct:.2f}%")
            
            # 异常值处理报告
            outliers_info = self.cleaning_report['financial_data']['outliers']
            print(f"\n  异常值处理:")
            for col, info in outliers_info['combined_outliers'].items():
                if info['count'] > 0:
                    print(f"    {col}: {info['count']} 个异常值 ({info['percentage']:.2f}%)")
    
    def run_cleaning_pipeline(self):
        """运行完整的数据清洗流程"""
        start_time = datetime.now()
        print(f"开始数据清洗流程: {start_time}")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 分析数据结构
        self.analyze_data_structure()
        
        # 3. 处理价格数据
        if self.price_data is not None:
            print("\n===== 处理价格数据 =====")
            
            # 3.1 处理缺失值
            price_no_missing = self.handle_missing_values(self.price_data, "price")
            
            # 3.2 检测异常值
            price_outliers = self.detect_outliers(price_no_missing, "price")
            
            # 3.3 处理异常值
            price_no_outliers = self.handle_outliers(price_no_missing, price_outliers, "price")
            
            # 3.4 比较标准化方法
            self.compare_standardization_methods(price_no_outliers, "price")
            
            # 3.5 数据标准化
            self.cleaned_price_data, _ = self.standardize_data(price_no_outliers, "price", "zscore")
            
            # 更新清洗后数据形状
            self.cleaning_report["price_data"]["cleaned_shape"] = self.cleaned_price_data.shape
        
        # 4. 处理财务数据
        if self.financial_data is not None:
            print("\n===== 处理财务数据 =====")
            
            # 4.1 处理缺失值
            financial_no_missing = self.handle_missing_values(self.financial_data, "financial")
            
            # 4.2 检测异常值
            financial_outliers = self.detect_outliers(financial_no_missing, "financial")
            
            # 4.3 处理异常值
            financial_no_outliers = self.handle_outliers(financial_no_missing, financial_outliers, "financial")
            
            # 4.4 比较标准化方法
            self.compare_standardization_methods(financial_no_outliers, "financial")
            
            # 4.5 数据标准化
            self.cleaned_financial_data, _ = self.standardize_data(financial_no_outliers, "financial", "zscore")
            
            # 更新清洗后数据形状
            self.cleaning_report["financial_data"]["cleaned_shape"] = self.cleaned_financial_data.shape
        
        # 5. 生成清洗报告
        self.generate_cleaning_report()
        
        # 6. 保存清洗后的数据
        self.save_cleaned_data()
        
        end_time = datetime.now()
        self.cleaning_report["processing_time"] = str(end_time - start_time)
        print(f"\n数据清洗流程完成: {end_time}")
        print(f"总耗时: {end_time - start_time}")
        
        return self.cleaned_price_data, self.cleaned_financial_data


# 主程序
if __name__ == "__main__":
    # 设置文件路径
    price_data_path = "c:\\Users\\Administrator\\Desktop\\design3\\data\\daliy_prices_merged\\all_price_data.csv"
    financial_data_path = "c:\\Users\\Administrator\\Desktop\\design3\\data\\financial_data_merged\\all_financial_data.csv"
    
    # 创建数据清洗器
    cleaner = DataCleaner(price_data_path, financial_data_path)
    
    # 运行清洗流程
    cleaned_price_data, cleaned_financial_data = cleaner.run_cleaning_pipeline()
    
    print("\n数据清洗完成!")