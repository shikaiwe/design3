#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子计算器基础类
提供因子计算的基础功能和通用方法
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

class BaseFactorCalculator(ABC):
    """因子计算器基础类"""
    
    def __init__(self):
        """初始化因子计算器"""
        self.factor_names = []
    
    @abstractmethod
    def calculate(self, financial_data, price_data, index_data):
        """
        计算因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            因子数据DataFrame
        """
        pass
    
    def prepare_data(self, financial_data, price_data, index_data):
        """
        准备和预处理数据
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            处理后的数据元组 (financial_data, price_data, index_data)
        """
        # 复制数据以避免修改原始数据
        financial_df = financial_data.copy()
        price_df = price_data.copy()
        index_df = index_data.copy()
        
        # 确保日期格式正确
        if 'REPORT_DATE' in financial_df.columns:
            financial_df['REPORT_DATE'] = pd.to_datetime(financial_df['REPORT_DATE'])
        
        # 处理价格数据列名
        if '日期' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['日期'])
        
        if '开盘价' in price_df.columns:
            price_df['open'] = price_df['开盘价']
        
        if '最高价' in price_df.columns:
            price_df['high'] = price_df['最高价']
        
        if '最低价' in price_df.columns:
            price_df['low'] = price_df['最低价']
        
        if '收盘价' in price_df.columns:
            price_df['close'] = price_df['收盘价']
        
        if '成交量' in price_df.columns:
            price_df['volume'] = price_df['成交量']
        
        if '成交额' in price_df.columns:
            price_df['amount'] = price_df['成交额']
        
        # 确保指数数据日期格式正确
        if 'date' in index_df.columns:
            index_df['date'] = pd.to_datetime(index_df['date'])
        
        # 确保数值列为数值类型
        numeric_cols = financial_df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if col not in ['stock_code', 'REPORT_DATE', 'REPORT_TYPE', 'report_type']:
                financial_df[col] = pd.to_numeric(financial_df[col], errors='coerce')
        
        numeric_cols = price_df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if col not in ['stock_code', '日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额']:
                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
        
        numeric_cols = index_df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if col not in ['instrument', 'date']:
                index_df[col] = pd.to_numeric(index_df[col], errors='coerce')
        
        return financial_df, price_df, index_df
    
    def calculate_ttm(self, data, value_col, date_col='REPORT_DATE', periods=4):
        """
        计算TTM (Trailing Twelve Months) 值
        
        参数:
            data: 数据DataFrame
            value_col: 要计算TTM的列名
            date_col: 日期列名
            periods: TTM周期，默认为4个季度
            
        返回:
            包含TTM值的DataFrame
        """
        # 按股票代码和日期排序
        data = data.sort_values(['stock_code', date_col])
        
        # 计算TTM
        data[f'{value_col}_ttm'] = data.groupby('stock_code')[value_col].rolling(
            window=periods, min_periods=1
        ).sum().reset_index(level=0, drop=True)
        
        return data
    
    def calculate_growth_rate(self, data, value_col, date_col='REPORT_DATE', periods=4):
        """
        计算增长率
        
        参数:
            data: 数据DataFrame
            value_col: 要计算增长率的列名
            date_col: 日期列名
            periods: 比较周期，默认为4个季度（年同比增长）
            
        返回:
            包含增长率的DataFrame
        """
        # 按股票代码和日期排序
        data = data.sort_values(['stock_code', date_col])
        
        # 计算同比增长率
        data[f'{value_col}_growth_rate'] = data.groupby('stock_code')[value_col].pct_change(
            periods=periods
        )
        
        return data
    
    def calculate_ma(self, data, value_col, window):
        """
        计算移动平均
        
        参数:
            data: 数据DataFrame
            value_col: 要计算移动平均的列名
            window: 窗口大小
            
        返回:
            包含移动平均的DataFrame
        """
        # 按股票代码排序
        data = data.sort_values(['stock_code', '日期'])
        
        # 计算移动平均
        data[f'MA{window}_{value_col}'] = data.groupby('stock_code')[value_col].rolling(
            window=window, min_periods=1
        ).mean().reset_index(level=0, drop=True)
        
        return data
    
    def calculate_ema(self, data, value_col, window):
        """
        计算指数移动平均
        
        参数:
            data: 数据DataFrame
            value_col: 要计算指数移动平均的列名
            window: 窗口大小
            
        返回:
            包含指数移动平均的DataFrame
        """
        # 按股票代码排序
        data = data.sort_values(['stock_code', '日期'])
        
        # 计算指数移动平均
        data[f'EMA{window}_{value_col}'] = data.groupby('stock_code')[value_col].ewm(
            span=window, adjust=False
        ).mean().reset_index(level=0, drop=True)
        
        return data
    
    def calculate_std(self, data, value_col, window):
        """
        计算滚动标准差
        
        参数:
            data: 数据DataFrame
            value_col: 要计算标准差的列名
            window: 窗口大小
            
        返回:
            包含标准差的DataFrame
        """
        # 按股票代码排序
        data = data.sort_values(['stock_code', '日期'])
        
        # 计算滚动标准差
        data[f'STD{window}_{value_col}'] = data.groupby('stock_code')[value_col].rolling(
            window=window, min_periods=1
        ).std().reset_index(level=0, drop=True)
        
        return data
    
    def calculate_pct_change(self, data, value_col, periods):
        """
        计算百分比变化
        
        参数:
            data: 数据DataFrame
            value_col: 要计算百分比变化的列名
            periods: 周期数
            
        返回:
            包含百分比变化的DataFrame
        """
        # 按股票代码排序
        data = data.sort_values(['stock_code', '日期'])
        
        # 计算百分比变化
        data[f'pct_change_{periods}_{value_col}'] = data.groupby('stock_code')[value_col].pct_change(
            periods=periods
        )
        
        return data
    
    def merge_data(self, left_df, right_df, on=None, left_on=None, right_on=None, how='left'):
        """
        合并数据
        
        参数:
            left_df: 左DataFrame
            right_df: 右DataFrame
            on: 共同的连接键
            left_on: 左DataFrame的连接键
            right_on: 右DataFrame的连接键
            how: 合并方式，默认为'left'
            
        返回:
            合并后的DataFrame
        """
        if on is not None:
            return pd.merge(left_df, right_df, on=on, how=how)
        else:
            return pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how)
    
    def safe_divide(self, numerator, denominator, fillna=0):
        """
        安全除法，避免除以零
        
        参数:
            numerator: 分子
            denominator: 分母
            fillna: 除零时填充的值
            
        返回:
            除法结果
        """
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(fillna)
        return result