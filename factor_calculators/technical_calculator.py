#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
技术指标因子计算器
计算技术指标因子，共8个因子
"""

import pandas as pd
import numpy as np
from .base_calculator import BaseFactorCalculator

class TechnicalCalculator(BaseFactorCalculator):
    """技术指标因子计算器"""
    
    def __init__(self):
        """初始化技术指标因子计算器"""
        super().__init__()
        self.factor_names = [
            'rsi',                          # 相对强弱指数
            'macd',                         # MACD指标
            'bollinger_upper',              # 布林带上轨
            'bollinger_lower',              # 布林带下轨
            'bollinger_width',              # 布林带宽度
            'williams_r',                   # 威廉指标
            'cci',                          # 商品通道指数
            'stoch_k',                      # 随机指标K值
            'stoch_d'                       # 随机指标D值
        ]
    
    def calculate(self, financial_data, price_data, index_data):
        """
        计算技术指标因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            技术指标因子DataFrame
        """
        # 准备数据
        financial_df, price_df, index_df = self.prepare_data(financial_data, price_data, index_data)
        
        # 获取最新日期的数据
        latest_date = price_df['date'].max()
        latest_prices = price_df[price_df['date'] == latest_date]
        
        # 初始化结果DataFrame
        result = latest_prices[['stock_code', 'date']].copy()
        
        # 计算技术指标因子
        result = self._calculate_rsi(result, price_df)
        result = self._calculate_macd(result, price_df)
        result = self._calculate_bollinger_bands(result, price_df)
        result = self._calculate_williams_r(result, price_df)
        result = self._calculate_cci(result, price_df)
        result = self._calculate_stochastic(result, price_df)
        
        return result
    
    def _calculate_rsi(self, result_df, price_df):
        """计算相对强弱指数(RSI)"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算收益率
        price_df['return'] = price_df.groupby('stock_code')['close'].pct_change()
        
        # 计算RSI
        def calculate_rsi(group, period=14):
            if len(group) < period + 1:
                return pd.Series([np.nan], ['rsi'])
            
            # 计算涨跌
            gains = group['return'].apply(lambda x: x if x > 0 else 0)
            losses = group['return'].apply(lambda x: -x if x < 0 else 0)
            
            # 计算平均涨跌幅
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # 计算RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            # 返回最新的RSI值
            return pd.Series([rsi.iloc[-1]], ['rsi'])
        
        rsi = price_df.groupby('stock_code').apply(calculate_rsi).reset_index()
        
        # 合并到结果
        result_df = pd.merge(result_df, rsi, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_macd(self, result_df, price_df):
        """计算MACD指标"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算MACD
        def calculate_macd(group, fast=12, slow=26, signal=9):
            if len(group) < slow:
                return pd.Series([np.nan], ['macd'])
            
            # 计算EMA
            ema_fast = group['close'].ewm(span=fast).mean()
            ema_slow = group['close'].ewm(span=slow).mean()
            
            # 计算MACD线
            macd_line = ema_fast - ema_slow
            
            # 计算信号线
            signal_line = macd_line.ewm(span=signal).mean()
            
            # 计算MACD柱状图
            macd_histogram = macd_line - signal_line
            
            # 返回最新的MACD柱状图值
            return pd.Series([macd_histogram.iloc[-1]], ['macd'])
        
        macd = price_df.groupby('stock_code').apply(calculate_macd).reset_index()
        
        # 合并到结果
        result_df = pd.merge(result_df, macd, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_bollinger_bands(self, result_df, price_df):
        """计算布林带指标"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算布林带
        def calculate_bollinger_bands(group, period=20, std_dev=2):
            if len(group) < period:
                return pd.Series([np.nan, np.nan, np.nan], ['bollinger_upper', 'bollinger_lower', 'bollinger_width'])
            
            # 计算移动平均线
            sma = group['close'].rolling(window=period).mean()
            
            # 计算标准差
            std = group['close'].rolling(window=period).std()
            
            # 计算上轨和下轨
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # 计算带宽
            width = upper_band - lower_band
            
            # 返回最新的值
            return pd.Series([upper_band.iloc[-1], lower_band.iloc[-1], width.iloc[-1]], 
                            ['bollinger_upper', 'bollinger_lower', 'bollinger_width'])
        
        bollinger = price_df.groupby('stock_code').apply(calculate_bollinger_bands).reset_index()
        
        # 合并到结果
        result_df = pd.merge(result_df, bollinger, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_williams_r(self, result_df, price_df):
        """计算威廉指标(%R)"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算威廉指标
        def calculate_williams_r(group, period=14):
            if len(group) < period:
                return pd.Series([np.nan], ['williams_r'])
            
            # 计算最高价和最低价的滚动窗口
            high_max = group['high'].rolling(window=period).max()
            low_min = group['low'].rolling(window=period).min()
            
            # 计算威廉指标
            williams_r = -100 * (high_max - group['close']) / (high_max - low_min)
            
            # 返回最新的威廉指标值
            return pd.Series([williams_r.iloc[-1]], ['williams_r'])
        
        williams_r = price_df.groupby('stock_code').apply(calculate_williams_r).reset_index()
        
        # 合并到结果
        result_df = pd.merge(result_df, williams_r, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_cci(self, result_df, price_df):
        """计算商品通道指数(CCI)"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算CCI
        def calculate_cci(group, period=20):
            if len(group) < period:
                return pd.Series([np.nan], ['cci'])
            
            # 计算典型价格
            typical_price = (group['high'] + group['low'] + group['close']) / 3
            
            # 计算典型价格的移动平均线
            sma_tp = typical_price.rolling(window=period).mean()
            
            # 计算平均绝对偏差
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            # 计算CCI
            cci = (typical_price - sma_tp) / (0.015 * mad)
            
            # 返回最新的CCI值
            return pd.Series([cci.iloc[-1]], ['cci'])
        
        cci = price_df.groupby('stock_code').apply(calculate_cci).reset_index()
        
        # 合并到结果
        result_df = pd.merge(result_df, cci, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_stochastic(self, result_df, price_df):
        """计算随机指标(Stochastic Oscillator)"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算随机指标
        def calculate_stochastic(group, k_period=14, d_period=3):
            if len(group) < k_period:
                return pd.Series([np.nan, np.nan], ['stoch_k', 'stoch_d'])
            
            # 计算最高价和最低价的滚动窗口
            high_max = group['high'].rolling(window=k_period).max()
            low_min = group['low'].rolling(window=k_period).min()
            
            # 计算K值
            k_percent = 100 * (group['close'] - low_min) / (high_max - low_min)
            
            # 计算D值(K值的移动平均)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            # 返回最新的K值和D值
            return pd.Series([k_percent.iloc[-1], d_percent.iloc[-1]], ['stoch_k', 'stoch_d'])
        
        stochastic = price_df.groupby('stock_code').apply(calculate_stochastic).reset_index()
        
        # 合并到结果
        result_df = pd.merge(result_df, stochastic, on='stock_code', how='left')
        
        return result_df