#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子计算主脚本
根据财务数据、价格数据和指数数据计算各类因子
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入因子计算模块
from .basics_calculator import BasicsCalculator
from .sentiment_calculator import SentimentCalculator
from .growth_calculator import GrowthCalculator
from .momentum_calculator import MomentumCalculator
from .per_share_calculator import PerShareCalculator
from .quality_calculator import QualityCalculator
from .risk_calculator import RiskCalculator
from .style_calculator import StyleCalculator
from .new_style_calculator import NewStyleCalculator
from .technical_calculator import TechnicalCalculator

class FactorCalculator:
    """因子计算主类"""
    
    def __init__(self):
        """初始化因子计算器"""
        # 数据文件路径
        self.financial_data_path = r"c:\Users\Administrator\Desktop\design3\data_clean_result\cleaned_data\cleaned_financial_data.csv"
        self.price_data_path = r"c:\Users\Administrator\Desktop\design3\data_clean_result\cleaned_data\cleaned_price_data.csv"
        self.index_data_path = r"c:\Users\Administrator\Desktop\design3\data\components\000300_SH_2019_2024.csv"
        
        # 结果保存路径
        self.output_path = r"c:\Users\Administrator\Desktop\design3\data\factories"
        
        # 初始化数据
        self.financial_data = None
        self.price_data = None
        self.index_data = None
        
        # 初始化因子计算器
        self.calculators = {
            'basics': BasicsCalculator(),
            'sentiment': SentimentCalculator(),
            'growth': GrowthCalculator(),
            'momentum': MomentumCalculator(),
            'per_share': PerShareCalculator(),
            'quality': QualityCalculator(),
            'risk': RiskCalculator(),
            'style': StyleCalculator(),
            'new_style': NewStyleCalculator(),
            'technical': TechnicalCalculator()
        }
    
    def load_data(self):
        """加载数据"""
        print("开始加载数据...")
        
        # 加载财务数据
        self.financial_data = pd.read_csv(self.financial_data_path)
        self.financial_data['REPORT_DATE'] = pd.to_datetime(self.financial_data['REPORT_DATE'])
        
        # 加载价格数据
        self.price_data = pd.read_csv(self.price_data_path)
        self.price_data['date'] = pd.to_datetime(self.price_data['日期'])
        
        # 添加英文列名以兼容因子计算器
        self.price_data['open'] = self.price_data['开盘']
        self.price_data['high'] = self.price_data['最高']
        self.price_data['low'] = self.price_data['最低']
        self.price_data['close'] = self.price_data['收盘']
        self.price_data['volume'] = self.price_data['成交量']
        self.price_data['amount'] = self.price_data['成交额']
        
        # 加载指数数据
        self.index_data = pd.read_csv(self.index_data_path)
        self.index_data['date'] = pd.to_datetime(self.index_data['date'])
        
        print("数据加载完成")
        print(f"财务数据: {self.financial_data.shape}")
        print(f"价格数据: {self.price_data.shape}")
        print(f"指数数据: {self.index_data.shape}")
    
    def calculate_all_factors(self):
        """计算所有因子"""
        print("开始计算所有因子...")
        
        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)
        
        # 计算各类因子
        for factor_type, calculator in self.calculators.items():
            print(f"正在计算 {factor_type} 类因子...")
            
            try:
                # 计算因子
                factor_data = calculator.calculate(
                    self.financial_data, 
                    self.price_data, 
                    self.index_data
                )
                
                # 保存结果
                output_file = os.path.join(self.output_path, f"{factor_type}_factors.csv")
                factor_data.to_csv(output_file, index=False)
                
                print(f"{factor_type} 类因子计算完成，已保存至 {output_file}")
                print(f"计算了 {len(factor_data.columns) - 2} 个因子")  # 减去stock_code和date列
                
            except Exception as e:
                print(f"计算 {factor_type} 类因子时出错: {str(e)}")
                continue
        
        print("所有因子计算完成")
    
    def run(self):
        """运行因子计算"""
        print("开始因子计算流程...")
        
        # 加载数据
        self.load_data()
        
        # 计算因子
        self.calculate_all_factors()
        
        print("因子计算流程结束")

if __name__ == "__main__":
    # 创建因子计算器
    calculator = FactorCalculator()
    
    # 运行因子计算
    calculator.run()