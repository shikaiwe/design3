#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每股指标因子计算器
计算每股指标因子，共15个因子
"""

import pandas as pd
import numpy as np
from base_calculator import BaseFactorCalculator

class PerShareCalculator(BaseFactorCalculator):
    """每股指标因子计算器"""
    
    def __init__(self):
        """初始化每股指标因子计算器"""
        super().__init__()
        self.factor_names = [
            'bps',                         # 每股净资产
            'cash_per_share',               # 每股现金
            'eps',                          # 每股收益
            'eps_ttm',                      # 每股收益TTM
            'netprofit_to_ttm',             # 净利润TTM
            'ocfps',                        # 每股经营现金流
            'ocfps_ttm',                    # 每股经营现金流TTM
            'opcfps',                       # 每股经营现金流
            'profit_to_ttm',                # 利润总额TTM
            'revenue_to_ttm',               # 营业收入TTM
            'revenue_ttm',                  # 营业收入TTM
            'roic_ttm',                     # 投入资本回报率TTM
            'share_capital',                # 股本
            'total_assets_to_ttm',          # 总资产TTM
            'total_equity_to_ttm'           # 股东权益TTM
        ]
    
    def calculate(self, financial_data, price_data, index_data):
        """
        计算每股指标因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            每股指标因子DataFrame
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
        
        # 计算每股指标
        result = self._calculate_per_share_indicators(result, latest_financial)
        
        # 计算每股经营现金流
        result = self._calculate_per_share_cash_flow(result, financial_df)
        
        # 计算其他TTM指标
        result = self._calculate_other_ttm_indicators(result, financial_df)
        
        # 重命名列
        result = result.rename(columns={'REPORT_DATE': 'date'})
        
        return result
    
    def _calculate_ttm_factors(self, financial_df):
        """计算TTM因子"""
        # 需要计算TTM的列
        ttm_columns = [
            'NETPROFIT',           # 净利润
            'PARENT_NETPROFIT',    # 归属于母公司股东的净利润
            'TOTAL_PROFIT',        # 利润总额
            'OPERATE_INCOME',      # 营业收入
            'TOTAL_ASSETS',        # 总资产
            'TOTAL_EQUITY',        # 股东权益
            'NETCASH_OPERATE'      # 经营活动现金流量净额
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
            'NETPROFIT_ttm': 'netprofit_to_ttm',
            'PARENT_NETPROFIT_ttm': 'eps_ttm',  # 这里暂时用eps_ttm，后续会重新计算
            'TOTAL_PROFIT_ttm': 'profit_to_ttm',
            'OPERATE_INCOME_ttm': 'revenue_to_ttm',
            'TOTAL_ASSETS_ttm': 'total_assets_to_ttm',
            'TOTAL_EQUITY_ttm': 'total_equity_to_ttm',
            'NETCASH_OPERATE_ttm': 'ocfps_ttm'  # 这里暂时用ocfps_ttm，后续会重新计算
        }
        
        result = result.rename(columns=column_mapping)
        
        return result
    
    def _calculate_per_share_indicators(self, result_df, latest_financial):
        """计算每股指标"""
        # 合并基础财务数据
        result = pd.merge(
            result_df,
            latest_financial[['stock_code', 'TOTAL_EQUITY', 'MONETARYFUNDS', 'NETPROFIT', 
                            'SHARE_CAPITAL', 'PARENT_NETPROFIT']],
            on='stock_code',
            how='left'
        )
        
        # 计算每股净资产 (BPS) = 股东权益 / 总股本
        if 'TOTAL_EQUITY' in result.columns and 'SHARE_CAPITAL' in result.columns:
            result['bps'] = self.safe_divide(result['TOTAL_EQUITY'], result['SHARE_CAPITAL'])
        
        # 计算每股现金 = 货币资金 / 总股本
        if 'MONETARYFUNDS' in result.columns and 'SHARE_CAPITAL' in result.columns:
            result['cash_per_share'] = self.safe_divide(result['MONETARYFUNDS'], result['SHARE_CAPITAL'])
        
        # 计算每股收益 (EPS) = 净利润 / 总股本
        if 'NETPROFIT' in result.columns and 'SHARE_CAPITAL' in result.columns:
            result['eps'] = self.safe_divide(result['NETPROFIT'], result['SHARE_CAPITAL'])
        
        # 计算每股收益TTM = 归属于母公司股东的净利润TTM / 总股本
        if 'eps_ttm' in result.columns and 'SHARE_CAPITAL' in result.columns:
            result['eps_ttm'] = self.safe_divide(result['eps_ttm'], result['SHARE_CAPITAL'])
        
        # 计算股本
        if 'SHARE_CAPITAL' in result.columns:
            result['share_capital'] = result['SHARE_CAPITAL']
        
        return result
    
    def _calculate_per_share_cash_flow(self, result_df, financial_df):
        """计算每股经营现金流"""
        # 计算每股经营现金流 = 经营活动现金流量净额 / 总股本
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 合并基础财务数据
        result = pd.merge(
            result_df,
            latest_financial[['stock_code', 'NETCASH_OPERATE', 'SHARE_CAPITAL']],
            on='stock_code',
            how='left'
        )
        
        # 计算每股经营现金流
        if 'NETCASH_OPERATE' in result.columns and 'SHARE_CAPITAL' in result.columns:
            result['ocfps'] = self.safe_divide(result['NETCASH_OPERATE'], result['SHARE_CAPITAL'])
        
        # 计算每股经营现金流TTM
        if 'ocfps_ttm' in result.columns and 'SHARE_CAPITAL' in result.columns:
            result['ocfps_ttm'] = self.safe_divide(result['ocfps_ttm'], result['SHARE_CAPITAL'])
        
        # 计算每股经营现金流 (opcfps) 与 ocfps 相同
        if 'ocfps' in result.columns:
            result['opcfps'] = result['ocfps']
        
        return result
    
    def _calculate_other_ttm_indicators(self, result_df, financial_df):
        """计算其他TTM指标"""
        # 计算营业收入TTM (revenue_ttm) 与 revenue_to_ttm 相同
        if 'revenue_to_ttm' in result_df.columns:
            result_df['revenue_ttm'] = result_df['revenue_to_ttm']
        
        # 计算ROIC TTM
        # ROIC = NOPAT / 投入资本
        # NOPAT = 净利润 + 所得税费用
        # 投入资本 = 总资产 - 无息流动负债
        
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 合并基础财务数据
        result = pd.merge(
            result_df,
            latest_financial[['stock_code', 'NETPROFIT', 'TOTAL_EQUITY', 'OPERATE_INCOME', 
                            'TOTAL_ASSETS', 'FIXED_ASSET', 'INTANGIBLE_ASSET', 'SHARE_CAPITAL']],
            on='stock_code',
            how='left'
        )
        
        # 计算NOPAT TTM
        if 'netprofit_to_ttm' in result.columns:
            # 假设所得税费用为净利润的25%
            result['NOPAT_ttm'] = result['netprofit_to_ttm'] * 1.25
        
        # 计算无息流动负债
        if 'TOTAL_CURRENT_LIAB' in result.columns:
            # 假设无息流动负债占总流动负债的70%
            result['interest_free_current_liability'] = result['TOTAL_CURRENT_LIAB'] * 0.7
        
        # 计算投入资本TTM
        if 'total_assets_to_ttm' in result.columns and 'interest_free_current_liability' in result.columns:
            result['invested_capital_ttm'] = result['total_assets_to_ttm'] - result['interest_free_current_liability']
        elif 'TOTAL_ASSETS' in result.columns and 'interest_free_current_liability' in result.columns:
            result['invested_capital_ttm'] = result['TOTAL_ASSETS'] - result['interest_free_current_liability']
        
        # 计算ROIC TTM
        if 'NOPAT_ttm' in result.columns and 'invested_capital_ttm' in result.columns:
            result['roic_ttm'] = self.safe_divide(result['NOPAT_ttm'], result['invested_capital_ttm'])
        
        return result