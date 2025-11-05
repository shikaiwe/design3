#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动量类因子计算器
计算动量类因子，共32个因子
"""

import pandas as pd
import numpy as np
from base_calculator import BaseFactorCalculator

class MomentumCalculator(BaseFactorCalculator):
    """动量类因子计算器"""
    
    def __init__(self):
        """初始化动量类因子计算器"""
        super().__init__()
        self.factor_names = [
            'HIGH_1D', 'HIGH_5D', 'HIGH_20D', 'HIGH_60D', 'HIGH_120D',  # 最高价
            'LOW_1D', 'LOW_5D', 'LOW_20D', 'LOW_60D', 'LOW_120D',      # 最低价
            'RETURN_1D', 'RETURN_5D', 'RETURN_20D', 'RETURN_60D', 'RETURN_120D',  # 收益率
            'MAX_RETURN_5D', 'MAX_RETURN_20D', 'MAX_RETURN_60D',         # 最大收益率
            'RETURN_VOL_5D', 'RETURN_VOL_20D', 'RETURN_VOL_60D',         # 收益波动率
            'SWING_VOL_5D', 'SWING_VOL_20D', 'SWING_VOL_60D',           # 摆动波动率
            'TURNOVER_5D', 'TURNOVER_20D', 'TURNOVER_60D',              # 换手率
            'TURNOVER_VOL_5D', 'TURNOVER_VOL_20D', 'TURNOVER_VOL_60D',  # 换手率波动率
            'VOLUME_RATIO_5D', 'VOLUME_RATIO_20D', 'VOLUME_RATIO_60D'    # 成交量比率
        ]
    
    def calculate(self, financial_data, price_data, index_data):
        """
        计算动量类因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            动量类因子DataFrame
        """
        # 准备数据
        price_df = price_data.copy()
        if 'date' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['date'])
        
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 获取最新日期的数据
        latest_date = price_df['date'].max()
        latest_prices = price_df[price_df['date'] == latest_date]
        
        # 初始化结果DataFrame
        result = latest_prices[['stock_code', 'date']].copy()
        
        # 计算动量类因子
        result = self._calculate_high_low_prices(result, price_df)
        result = self._calculate_returns(result, price_df)
        result = self._calculate_max_returns(result, price_df)
        result = self._calculate_return_volatility(result, price_df)
        result = self._calculate_swing_volatility(result, price_df)
        result = self._calculate_turnover(result, price_df)
        result = self._calculate_turnover_volatility(result, price_df)
        result = self._calculate_volume_ratio(result, price_df)
        
        return result
    
    def _calculate_high_low_prices(self, result_df, price_df):
        """计算最高价和最低价相对于当前收盘价的比率"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 获取每只股票的最新收盘价
        latest_close = price_df.groupby('stock_code')['close'].last().reset_index()
        latest_close.columns = ['stock_code', 'latest_close']
        
        # 将最新收盘价合并回价格数据
        price_df = pd.merge(price_df, latest_close, on='stock_code', how='left')
        
        # 计算不同周期的最高价和最低价比率
        for period in [1, 5, 20, 60, 120]:
            high_col = f'HIGH_{period}D'
            low_col = f'LOW_{period}D'
            
            # 计算滚动最高价，使用transform避免索引对齐问题
            price_df[high_col] = price_df.groupby('stock_code')['high'].transform(
                lambda x: x.rolling(window=period, min_periods=1).max()
            )
            
            # 计算滚动最低价，使用transform避免索引对齐问题
            price_df[low_col] = price_df.groupby('stock_code')['low'].transform(
                lambda x: x.rolling(window=period, min_periods=1).min()
            )
            
            # 计算相对于最新收盘价的比率
            price_df[high_col] = price_df[high_col] / price_df['latest_close']
            price_df[low_col] = price_df[low_col] / price_df['latest_close']
            
            # 获取最新日期的数据
            latest_data = price_df[price_df['date'] == price_df['date'].max()]
            latest_high_low = latest_data[['stock_code', high_col, low_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_high_low, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_returns(self, result_df, price_df):
        """计算收益率"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算不同周期的收益率
        for period in [1, 5, 20, 60, 120]:
            return_col = f'RETURN_{period}D'
            
            # 计算收益率，使用transform确保索引对齐
            price_df[return_col] = price_df.groupby('stock_code')['close'].transform(
                lambda x: x.pct_change(period)
            )
            
            # 获取最新日期的收益率
            latest_return = price_df.groupby('stock_code').tail(1)[['stock_code', return_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_return, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_max_returns(self, result_df, price_df):
        """计算最大收益率"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算日收益率，使用transform确保索引对齐
        price_df['daily_return'] = price_df.groupby('stock_code')['close'].transform(
            lambda x: x.pct_change()
        )
        
        # 计算不同周期的最大收益率
        for period in [5, 20, 60]:
            max_return_col = f'MAX_RETURN_{period}D'
            
            # 计算滚动最大收益率，使用transform避免索引对齐问题
            price_df[max_return_col] = price_df.groupby('stock_code')['daily_return'].transform(
                lambda x: x.rolling(window=period, min_periods=1).max()
            )
            
            # 获取最新日期的最大收益率
            latest_data = price_df[price_df['date'] == price_df['date'].max()]
            latest_max_return = latest_data[['stock_code', max_return_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_max_return, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_return_volatility(self, result_df, price_df):
        """计算收益波动率"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算日收益率，使用transform确保索引对齐
        price_df['daily_return'] = price_df.groupby('stock_code')['close'].transform(
            lambda x: x.pct_change()
        )
        
        # 计算不同周期的收益波动率
        for period in [5, 20, 60]:
            return_vol_col = f'RETURN_VOL_{period}D'
            
            # 计算收益波动率，使用transform避免索引对齐问题
            price_df[return_vol_col] = price_df.groupby('stock_code')['daily_return'].transform(
                lambda x: x.rolling(window=period, min_periods=1).std()
            )
            
            # 获取最新日期的收益波动率
            latest_data = price_df[price_df['date'] == price_df['date'].max()]
            latest_return_vol = latest_data[['stock_code', return_vol_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_return_vol, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_swing_volatility(self, result_df, price_df):
        """计算摆动波动率"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算日摆动率
        price_df['daily_swing'] = (price_df['high'] - price_df['low']) / price_df['close']
        
        # 计算不同周期的摆动波动率
        for period in [5, 20, 60]:
            swing_vol_col = f'SWING_VOL_{period}D'
            
            # 计算摆动波动率，使用transform避免索引对齐问题
            price_df[swing_vol_col] = price_df.groupby('stock_code')['daily_swing'].transform(
                lambda x: x.rolling(window=period, min_periods=1).std()
            )
            
            # 获取最新日期的摆动波动率
            latest_data = price_df[price_df['date'] == price_df['date'].max()]
            latest_swing_vol = latest_data[['stock_code', swing_vol_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_swing_vol, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_turnover(self, result_df, price_df):
        """计算换手率"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算不同周期的换手率
        for period in [5, 20, 60]:
            turnover_col = f'TURNOVER_{period}D'
            
            # 计算换手率，使用transform避免索引对齐问题
            price_df[turnover_col] = price_df.groupby('stock_code')['volume'].transform(
                lambda x: x.rolling(window=period, min_periods=1).mean()
            )
            
            # 获取最新日期的换手率
            latest_data = price_df[price_df['date'] == price_df['date'].max()]
            latest_turnover = latest_data[['stock_code', turnover_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_turnover, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_turnover_volatility(self, result_df, price_df):
        """计算换手率波动率"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算不同周期的换手率波动率
        for period in [5, 20, 60]:
            turnover_vol_col = f'TURNOVER_VOL_{period}D'
            
            # 计算换手率波动率，使用transform避免索引对齐问题
            price_df[turnover_vol_col] = price_df.groupby('stock_code')['volume'].transform(
                lambda x: x.rolling(window=period, min_periods=1).std()
            )
            
            # 获取最新日期的换手率波动率
            latest_data = price_df[price_df['date'] == price_df['date'].max()]
            latest_turnover_vol = latest_data[['stock_code', turnover_vol_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_turnover_vol, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_volume_ratio(self, result_df, price_df):
        """计算成交量比率"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算不同周期的成交量比率
        for period in [5, 20, 60]:
            volume_ratio_col = f'VOLUME_RATIO_{period}D'
            
            # 计算成交量比率（当日成交量/周期内平均成交量），使用transform避免索引对齐问题
            price_df['volume_ma'] = price_df.groupby('stock_code')['volume'].transform(
                lambda x: x.rolling(window=period, min_periods=1).mean()
            )
            price_df[volume_ratio_col] = price_df['volume'] / price_df['volume_ma']
            
            # 获取最新日期的成交量比率
            latest_data = price_df[price_df['date'] == price_df['date'].max()]
            latest_volume_ratio = latest_data[['stock_code', volume_ratio_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_volume_ratio, on='stock_code', how='left')
        
        return result_df