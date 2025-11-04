#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风险类因子计算器
计算风险类因子，共12个因子
"""

import pandas as pd
import numpy as np
from .base_calculator import BaseFactorCalculator

class RiskCalculator(BaseFactorCalculator):
    """风险类因子计算器"""
    
    def __init__(self):
        """初始化风险类因子计算器"""
        super().__init__()
        self.factor_names = [
            'Kurtosis_20',          # 20日峰度
            'Kurtosis_60',          # 60日峰度
            'Kurtosis_120',         # 120日峰度
            'Skewness_20',          # 20日偏度
            'Skewness_60',          # 60日偏度
            'Skewness_120',         # 120日偏度
            'sharpe_ratio_20',      # 20日夏普比率
            'sharpe_ratio_60',      # 60日夏普比率
            'sharpe_ratio_120',     # 120日夏普比率
            'maxdrawdown_20',        # 20日最大回撤
            'maxdrawdown_60',        # 60日最大回撤
            'maxdrawdown_120'        # 120日最大回撤
        ]
    
    def calculate(self, financial_data, price_data, index_data):
        """
        计算风险类因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            风险类因子DataFrame
        """
        # 准备数据
        financial_df, price_df, index_df = self.prepare_data(financial_data, price_data, index_data)
        
        # 计算收益率
        price_df = self._calculate_returns(price_df)
        
        # 获取最新日期的数据
        latest_date = price_df['date'].max()
        latest_prices = price_df[price_df['date'] == latest_date]
        
        # 初始化结果DataFrame
        result = latest_prices[['stock_code', 'date']].copy()
        
        # 计算风险类因子
        result = self._calculate_skewness_kurtosis(result, price_df)
        result = self._calculate_sharpe_ratio(result, price_df)
        result = self._calculate_max_drawdown(result, price_df)
        
        return result
    
    def _calculate_returns(self, price_df):
        """计算收益率"""
        # 按股票代码分组，计算收益率
        price_df = price_df.sort_values(['stock_code', 'date'])
        price_df['return'] = price_df.groupby('stock_code')['close'].pct_change()
        
        # 计算对数收益率
        price_df['log_return'] = np.log(price_df['close'] / price_df['close'].shift(1))
        
        return price_df
    
    def _calculate_skewness_kurtosis(self, result_df, price_df):
        """计算偏度和峰度"""
        # 计算不同周期的偏度和峰度
        periods = [20, 60, 120]
        
        for period in periods:
            # 计算滚动偏度
            skewness = price_df.groupby('stock_code')['return'].rolling(window=period).skew().reset_index()
            skewness = skewness.rename(columns={'return': f'Skewness_{period}'})
            
            # 获取最新日期的偏度值
            latest_skewness = skewness.loc[skewness.groupby('stock_code')['level_1'].idxmax()]
            latest_skewness = latest_skewness[['stock_code', f'Skewness_{period}']]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_skewness, on='stock_code', how='left')
            
            # 计算滚动峰度
            kurtosis = price_df.groupby('stock_code')['return'].rolling(window=period).kurt().reset_index()
            kurtosis = kurtosis.rename(columns={'return': f'Kurtosis_{period}'})
            
            # 获取最新日期的峰度值
            latest_kurtosis = kurtosis.loc[kurtosis.groupby('stock_code')['level_1'].idxmax()]
            latest_kurtosis = latest_kurtosis[['stock_code', f'Kurtosis_{period}']]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_kurtosis, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_sharpe_ratio(self, result_df, price_df):
        """计算夏普比率"""
        # 计算不同周期的夏普比率
        periods = [20, 60, 120]
        risk_free_rate = 0.03 / 252  # 假设年化无风险利率为3%，转换为日化
        
        for period in periods:
            # 计算滚动夏普比率
            def calculate_sharpe(group):
                returns = group['return'].dropna()
                if len(returns) < period:
                    return pd.Series([np.nan], [f'sharpe_ratio_{period}'])
                
                excess_returns = returns - risk_free_rate
                if excess_returns.std() == 0:
                    return pd.Series([0], [f'sharpe_ratio_{period}'])
                
                sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # 年化
                return pd.Series([sharpe], [f'sharpe_ratio_{period}'])
            
            sharpe_ratio = price_df.groupby('stock_code').apply(calculate_sharpe).reset_index()
            
            # 合并到结果
            result_df = pd.merge(result_df, sharpe_ratio, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_max_drawdown(self, result_df, price_df):
        """计算最大回撤"""
        # 计算不同周期的最大回撤
        periods = [20, 60, 120]
        
        for period in periods:
            # 计算滚动最大回撤
            def calculate_max_dd(group):
                prices = group['close'].dropna()
                if len(prices) < period:
                    return pd.Series([np.nan], [f'maxdrawdown_{period}'])
                
                # 计算累计收益
                cumulative = (1 + prices.pct_change()).cumprod()
                
                # 计算滚动最高点
                rolling_max = cumulative.rolling(window=period, min_periods=1).max()
                
                # 计算回撤
                drawdown = (cumulative - rolling_max) / rolling_max
                
                # 最大回撤
                max_dd = drawdown.min()
                
                return pd.Series([max_dd], [f'maxdrawdown_{period}'])
            
            max_dd = price_df.groupby('stock_code').apply(calculate_max_dd).reset_index()
            
            # 合并到结果
            result_df = pd.merge(result_df, max_dd, on='stock_code', how='left')
        
        return result_df