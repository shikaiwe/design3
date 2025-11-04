#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成长类因子计算器
计算成长类因子，共9个因子
"""

import pandas as pd
import numpy as np
from .base_calculator import BaseFactorCalculator

class GrowthCalculator(BaseFactorCalculator):
    """成长类因子计算器"""
    
    def __init__(self):
        """初始化成长类因子计算器"""
        super().__init__()
        self.factor_names = [
            'netprofit_growth_rate',        # 净利润增长率
            'netprofit_yoy',                # 净利润同比增长率
            'operateincome_yoy',            # 营业收入同比增长率
            'profit_to_gr',                 # 净利润增长率与营业收入增长率之比
            'revenue_growth_rate',          # 营业收入增长率
            'revenue_yoy',                  # 营业收入同比增长率
            'ROA',                          # 资产收益率
            'ROE',                          # 净资产收益率
            'ROIC'                          # 投入资本回报率
        ]
    
    def calculate(self, financial_data, price_data, index_data):
        """
        计算成长类因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            成长类因子DataFrame
        """
        # 准备数据
        financial_df, price_df, index_df = self.prepare_data(financial_data, price_data, index_data)
        
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 初始化结果DataFrame
        result = latest_financial[['stock_code', 'REPORT_DATE']].copy()
        
        # 计算TTM指标
        financial_ttm = self._calculate_ttm_factors(financial_df)
        
        # 合并TTM数据
        result = pd.merge(
            result,
            financial_ttm,
            on=['stock_code', 'REPORT_DATE'],
            how='left'
        )
        
        # 计算增长率指标
        result = self._calculate_growth_rates(result, financial_df)
        
        # 计算ROA、ROE和ROIC
        result = self._calculate_profitability_ratios(result, financial_df)
        
        # 重命名列
        result = result.rename(columns={'REPORT_DATE': 'date'})
        
        return result
    
    def _calculate_ttm_factors(self, financial_df):
        """计算TTM因子"""
        # 需要计算TTM的列
        ttm_columns = [
            'NETPROFIT',      # 净利润
            'OPERATE_INCOME', # 营业收入
            'PARENT_NETPROFIT' # 归属于母公司股东的净利润
        ]
        
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 初始化结果DataFrame
        result = latest_financial[['stock_code', 'REPORT_DATE']].copy()
        
        # 计算各列的TTM值
        for col in ttm_columns:
            if col in financial_df.columns:
                # 计算TTM
                ttm_data = self.calculate_ttm(financial_df, col, 'REPORT_DATE')
                # 获取最新日期的TTM值
                latest_ttm = ttm_data.loc[ttm_data.groupby('stock_code')['REPORT_DATE'].idxmax()]
                # 合并到结果
                result = pd.merge(
                    result,
                    latest_ttm[['stock_code', 'REPORT_DATE', f'{col}_ttm']],
                    on=['stock_code', 'REPORT_DATE'],
                    how='left'
                )
        
        # 重命名列
        column_mapping = {
            'NETPROFIT_ttm': 'netprofit_ttm',
            'OPERATE_INCOME_ttm': 'operateincome_ttm',
            'PARENT_NETPROFIT_ttm': 'parent_netprofit_ttm'
        }
        
        result = result.rename(columns=column_mapping)
        
        return result
    
    def _calculate_growth_rates(self, result_df, financial_df):
        """计算增长率指标"""
        # 计算净利润同比增长率
        netprofit_growth = self._calculate_year_over_year_growth(financial_df, 'NETPROFIT')
        result_df = pd.merge(result_df, netprofit_growth, on='stock_code', how='left')
        
        # 计算营业收入同比增长率
        operateincome_growth = self._calculate_year_over_year_growth(financial_df, 'OPERATE_INCOME')
        result_df = pd.merge(result_df, operateincome_growth, on='stock_code', how='left')
        
        # 计算净利润增长率 (基于TTM)
        if 'netprofit_ttm' in result_df.columns:
            # 获取当前TTM和一年前TTM
            current_ttm = result_df[['stock_code', 'netprofit_ttm']].copy()
            
            # 计算一年前的TTM
            financial_df['year'] = pd.to_datetime(financial_df['REPORT_DATE']).dt.year
            latest_year = financial_df['year'].max()
            prev_year = latest_year - 1
            
            # 获取前一年同期的财务数据
            prev_year_data = financial_df[financial_df['year'] == prev_year]
            
            if not prev_year_data.empty:
                # 计算前一年的TTM
                prev_ttm_data = self._calculate_ttm_for_year(financial_df, 'NETPROFIT', prev_year)
                
                # 合并当前和前一年的TTM
                growth_data = pd.merge(
                    current_ttm,
                    prev_ttm_data,
                    on='stock_code',
                    how='left',
                    suffixes=('_current', '_prev')
                )
                
                # 修正列名 - 合并后的列名应该是 netprofit_ttm 和 NETPROFIT_ttm
                # 而不是 netprofit_ttm_current 和 netprofit_ttm_prev
                if 'NETPROFIT_ttm' in growth_data.columns:
                    growth_data = growth_data.rename(columns={
                        'netprofit_ttm': 'netprofit_ttm_current',
                        'NETPROFIT_ttm': 'netprofit_ttm_prev'
                    })
                
                # 计算增长率
                growth_data['netprofit_growth_rate'] = self.safe_divide(
                    growth_data['netprofit_ttm_current'] - growth_data['netprofit_ttm_prev'],
                    growth_data['netprofit_ttm_prev']
                )
                
                # 合并到结果
                result_df = pd.merge(
                    result_df,
                    growth_data[['stock_code', 'netprofit_growth_rate']],
                    on='stock_code',
                    how='left'
                )
        
        # 计算营业收入增长率 (基于TTM)
        if 'operateincome_ttm' in result_df.columns:
            # 获取当前TTM和一年前TTM
            current_ttm = result_df[['stock_code', 'operateincome_ttm']].copy()
            
            # 计算一年前的TTM
            if not prev_year_data.empty:
                # 计算前一年的TTM
                prev_ttm_data = self._calculate_ttm_for_year(financial_df, 'OPERATE_INCOME', prev_year)
                
                # 合并当前和前一年的TTM
                growth_data = pd.merge(
                    current_ttm,
                    prev_ttm_data,
                    on='stock_code',
                    how='left',
                    suffixes=('_current', '_prev')
                )
                
                # 修正列名 - 合并后的列名应该是 operateincome_ttm 和 OPERATE_INCOME_ttm
                # 而不是 operateincome_ttm_current 和 operateincome_ttm_prev
                if 'OPERATE_INCOME_ttm' in growth_data.columns:
                    growth_data = growth_data.rename(columns={
                        'operateincome_ttm': 'operateincome_ttm_current',
                        'OPERATE_INCOME_ttm': 'operateincome_ttm_prev'
                    })
                
                # 计算增长率
                growth_data['revenue_growth_rate'] = self.safe_divide(
                    growth_data['operateincome_ttm_current'] - growth_data['operateincome_ttm_prev'],
                    growth_data['operateincome_ttm_prev']
                )
                
                # 合并到结果
                result_df = pd.merge(
                    result_df,
                    growth_data[['stock_code', 'revenue_growth_rate']],
                    on='stock_code',
                    how='left'
                )
        
        # 计算净利润增长率与营业收入增长率之比
        if 'netprofit_growth_rate' in result_df.columns and 'revenue_growth_rate' in result_df.columns:
            result_df['profit_to_gr'] = self.safe_divide(
                result_df['netprofit_growth_rate'],
                result_df['revenue_growth_rate']
            )
        
        # 重命名列
        column_mapping = {
            'NETPROFIT_yoy': 'netprofit_yoy',
            'OPERATE_INCOME_yoy': 'operateincome_yoy',
            'OPERATE_INCOME_yoy': 'revenue_yoy'
        }
        
        result_df = result_df.rename(columns=column_mapping)
        
        return result_df
    
    def _calculate_year_over_year_growth(self, financial_df, column):
        """计算同比增长率"""
        # 按股票代码和年份分组
        financial_df['year'] = pd.to_datetime(financial_df['REPORT_DATE']).dt.year
        financial_df['month'] = pd.to_datetime(financial_df['REPORT_DATE']).dt.month
        
        # 获取每年的数据
        yearly_data = financial_df.groupby(['stock_code', 'year'])[column].sum().reset_index()
        
        # 计算同比增长率
        yearly_data['prev_year_value'] = yearly_data.groupby('stock_code')[column].shift(1)
        yearly_data[f'{column}_yoy'] = self.safe_divide(
            yearly_data[column] - yearly_data['prev_year_value'],
            yearly_data['prev_year_value']
        )
        
        # 获取最新年份的增长率
        latest_year = yearly_data['year'].max()
        latest_growth = yearly_data[yearly_data['year'] == latest_year][['stock_code', f'{column}_yoy']]
        
        return latest_growth
    
    def _calculate_ttm_for_year(self, financial_df, column, year):
        """计算指定年份的TTM"""
        # 筛选指定年份的数据
        year_data = financial_df[financial_df['year'] == year].copy()
        
        # 按季度分组
        year_data['quarter'] = pd.to_datetime(year_data['REPORT_DATE']).dt.quarter
        
        # 计算TTM (假设一年有4个季度)
        ttm_data = year_data.groupby(['stock_code', 'quarter'])[column].sum().reset_index()
        ttm_data = ttm_data.groupby('stock_code')[column].sum().reset_index()
        ttm_data = ttm_data.rename(columns={column: f'{column}_ttm'})
        
        return ttm_data
    
    def _calculate_profitability_ratios(self, result_df, financial_df):
        """计算盈利能力比率"""
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 合并基础财务数据
        result = pd.merge(
            result_df,
            latest_financial[['stock_code', 'TOTAL_ASSETS', 'TOTAL_LIABILITIES', 'TOTAL_EQUITY', 
                            'NETPROFIT', 'OPERATE_PROFIT', 'FINANCE_EXPENSE']],
            on='stock_code',
            how='left'
        )
        
        # 计算ROA (资产收益率) = 净利润 / 总资产
        if 'TOTAL_ASSETS' in result.columns and 'NETPROFIT' in result.columns:
            result['ROA'] = self.safe_divide(result['NETPROFIT'], result['TOTAL_ASSETS'])
        
        # 计算ROE (净资产收益率) = 净利润 / 股东权益
        if 'TOTAL_EQUITY' in result.columns and 'NETPROFIT' in result.columns:
            result['ROE'] = self.safe_divide(result['NETPROFIT'], result['TOTAL_EQUITY'])
        
        # 计算ROIC (投入资本回报率) = NOPAT / 投入资本
        # NOPAT (税后净营业利润) = 营业利润 - 所得税费用
        # 投入资本 = 总资产 - 无息流动负债
        
        # 计算NOPAT
        if 'OPERATE_PROFIT' in result.columns:
            # 如果缺少所得税费用，使用营业利润的75%作为近似
            result['NOPAT'] = result['OPERATE_PROFIT'] * 0.75
        
        # 计算无息流动负债
        if 'TOTAL_CURRENT_LIAB' in result.columns and 'FINANCE_EXPENSE' in result.columns:
            # 假设有息负债占总负债的30%，无息负债占70%
            result['interest_free_current_liability'] = result['TOTAL_CURRENT_LIAB'] * 0.7
        else:
            # 如果缺少数据，使用流动负债的70%作为近似
            if 'TOTAL_CURRENT_LIAB' in result.columns:
                result['interest_free_current_liability'] = result['TOTAL_CURRENT_LIAB'] * 0.7
        
        # 计算投入资本
        if 'TOTAL_ASSETS' in result.columns and 'interest_free_current_liability' in result.columns:
            result['invested_capital'] = result['TOTAL_ASSETS'] - result['interest_free_current_liability']
        else:
            # 如果缺少数据，使用总资产的80%作为近似
            if 'TOTAL_ASSETS' in result.columns:
                result['invested_capital'] = result['TOTAL_ASSETS'] * 0.8
        
        # 计算ROIC
        if 'NOPAT' in result.columns and 'invested_capital' in result.columns:
            result['ROIC'] = self.safe_divide(result['NOPAT'], result['invested_capital'])
        
        # 选择需要的列
        columns_to_keep = result_df.columns.tolist() + ['ROA', 'ROE', 'ROIC']
        result = result[columns_to_keep]
        
        return result