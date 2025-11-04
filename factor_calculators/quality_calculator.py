#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
质量类因子计算器
计算质量类因子，共68个因子
"""

import pandas as pd
import numpy as np
from .base_calculator import BaseFactorCalculator

class QualityCalculator(BaseFactorCalculator):
    """质量类因子计算器"""
    
    def __init__(self):
        """初始化质量类因子计算器"""
        super().__init__()
        self.factor_names = [
            'debt_to_tangible_equity_ratio',      # 有形净值债务率
            'gross_income_ratio',                  # 毛利润率
            'netprofit_margin',                   # 净利润率
            'netprofit_margin_ttm',               # 净利润率TTM
            'operating_profit_margin',           # 经营利润率
            'operating_profit_margin_ttm',       # 经营利润率TTM
            'profit_margin',                     # 利润率
            'profit_margin_ttm',                 # 利润率TTM
            'roa',                               # 资产收益率
            'roa_ttm',                           # 资产收益率TTM
            'roa_average',                       # 平均资产收益率
            'roa_average_ttm',                   # 平均资产收益率TTM
            'roa_change',                        # 资产收益率变化
            'roa_change_ttm',                    # 资产收益率变化TTM
            'roa_delta',                         # 资产收益率差值
            'roa_delta_ttm',                     # 资产收益率差值TTM
            'roe',                               # 净资产收益率
            'roe_ttm',                           # 净资产收益率TTM
            'roe_average',                       # 平均净资产收益率
            'roe_average_ttm',                   # 平均净资产收益率TTM
            'roe_change',                        # 净资产收益率变化
            'roe_change_ttm',                    # 净资产收益率变化TTM
            'roe_delta',                         # 净资产收益率差值
            'roe_delta_ttm',                     # 净资产收益率差值TTM
            'roic',                              # 投入资本回报率
            'roic_ttm',                          # 投入资本回报率TTM
            'roic_average',                      # 平均投入资本回报率
            'roic_average_ttm',                  # 平均投入资本回报率TTM
            'roic_change',                       # 投入资本回报率变化
            'roic_change_ttm',                   # 投入资本回报率变化TTM
            'roic_delta',                        # 投入资本回报率差值
            'roic_delta_ttm',                    # 投入资本回报率差值TTM
            'ebit_to_assets',                    # EBIT/资产
            'ebit_to_assets_ttm',                # EBIT/资产TTM
            'ebit_to_average_assets',            # EBIT/平均资产
            'ebit_to_average_assets_ttm',        # EBIT/平均资产TTM
            'ebit_to_average_assets',            # EBIT/平均资产
            'ebit_to_average_assets_ttm',        # EBIT/平均资产TTM
            'ebitda_to_assets',                 # EBITDA/资产
            'ebitda_to_assets_ttm',              # EBITDA/资产TTM
            'ebitda_to_average_assets',         # EBITDA/平均资产
            'ebitda_to_average_assets_ttm',     # EBITDA/平均资产TTM
            'ebitda_to_average_assets',         # EBITDA/平均资产
            'ebitda_to_average_assets_ttm',     # EBITDA/平均资产TTM
            'ebitda_margin',                    # EBITDA利润率
            'ebitda_margin_ttm',                # EBITDA利润率TTM
            'ebitda_to_equity',                 # EBITDA/股东权益
            'ebitda_to_equity_ttm',             # EBITDA/股东权益TTM
            'ebitda_to_average_equity',         # EBITDA/平均股东权益
            'ebitda_to_average_equity_ttm',     # EBITDA/平均股东权益TTM
            'ebitda_to_interest',              # EBITDA/利息支出
            'ebitda_to_interest_ttm',          # EBITDA/利息支出TTM
            'ebit_to_interest',                # EBIT/利息支出
            'ebit_to_interest_ttm',            # EBIT/利息支出TTM
            'ebit_to_sales',                   # EBIT/营业收入
            'ebit_to_sales_ttm',               # EBIT/营业收入TTM
            'ebitda_to_sales',                # EBITDA/营业收入
            'ebitda_to_sales_ttm',            # EBITDA/营业收入TTM
            'ebit_to_equity',                 # EBIT/股东权益
            'ebit_to_equity_ttm',             # EBIT/股东权益TTM
            'ebit_to_average_equity',         # EBIT/平均股东权益
            'ebit_to_average_equity_ttm',     # EBIT/平均股东权益TTM
            'ebit_to_interest_expense',       # EBIT/利息支出
            'ebit_to_interest_expense_ttm',   # EBIT/利息支出TTM
            'ebitda_to_interest_expense',     # EBITDA/利息支出
            'ebitda_to_interest_expense_ttm', # EBITDA/利息支出TTM
            'ebitda_to_fixed_assets',         # EBITDA/固定资产
            'ebitda_to_fixed_assets_ttm',     # EBITDA/固定资产TTM
            'ebitda_to_current_assets',       # EBITDA/流动资产
            'ebitda_to_current_assets_ttm',   # EBITDA/流动资产TTM
            'ebitda_to_tangible_assets',      # EBITDA/有形资产
            'ebitda_to_tangible_assets_ttm',  # EBITDA/有形资产TTM
            'ebitda_to_working_capital',      # EBITDA/营运资本
            'ebitda_to_working_capital_ttm'   # EBITDA/营运资本TTM
        ]
    
    def calculate(self, financial_data, price_data, index_data):
        """
        计算质量类因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            质量类因子DataFrame
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
        
        # 合并基础财务数据
        result = pd.merge(
            result,
            latest_financial[['stock_code', 'TOTAL_EQUITY', 'TOTAL_ASSETS', 'NETPROFIT', 
                            'OPERATE_INCOME', 'TOTAL_PROFIT', 'PARENT_NETPROFIT',
                            'TOTAL_LIABILITIES', 'FIXED_ASSET', 'INTANGIBLE_ASSET',
                            'TOTAL_CURRENT_ASSETS', 'TOTAL_CURRENT_LIAB',
                            'FINANCE_EXPENSE']],
            on='stock_code',
            how='left'
        )
        
        # 计算质量类因子
        result = self._calculate_profitability_ratios(result)
        result = self._calculate_return_ratios(result)
        result = self._calculate_ebitda_ratios(result)
        result = self._calculate_ebit_ratios(result)
        result = self._calculate_debt_ratios(result)
        
        # 重命名列
        result = result.rename(columns={'REPORT_DATE': 'date'})
        
        return result
    
    def _calculate_ttm_factors(self, data):
        """计算TTM（过去12个月）因子"""
        ttm_columns = [
            'TOTAL_ASSETS', 'TOTAL_LIABILITIES', 'TOTAL_EQUITY', 'TOTAL_PARENT_EQUITY',
            'ACCOUNTS_RECE', 'FIXED_ASSET', 'INTANGIBLE_ASSET', 'GOODWILL',
            'SHARE_CAPITAL', 'CAPITAL_RESERVE', 'SURPLUS_RESERVE', 'UNASSIGN_RPOFIT',
            'OPERATE_INCOME', 'ASSET_IMPAIRMENT_LOSS', 'OPERATE_PROFIT', 'TOTAL_PROFIT',
            'NETPROFIT', 'PARENT_NETPROFIT', 'NETCASH_OPERATE', 'NETCASH_INVEST',
            'NETCASH_FINANCE', 'TOTAL_CURRENT_ASSETS', 'TOTAL_NONCURRENT_ASSETS', 
            'TOTAL_CURRENT_LIAB', 'TOTAL_NONCURRENT_LIAB', 'MONETARYFUNDS', 'INVENTORY',
            'TOTAL_OPERATE_INCOME', 'OPERATE_COST', 'SALE_EXPENSE', 'MANAGE_EXPENSE',
            'FINANCE_EXPENSE', 'RESEARCH_EXPENSE', 'SALES_SERVICES', 'BUY_SERVICES'
        ]
        
        # 为每个股票计算TTM指标
        result = []
        for stock_code in data['stock_code'].unique():
            stock_data = data[data['stock_code'] == stock_code].copy()
            stock_data = stock_data.sort_values('REPORT_DATE')
            
            # 计算TTM指标（过去4个季度的总和）
            for col in ttm_columns:
                if col in stock_data.columns:
                    stock_data[f'{col}_ttm'] = stock_data[col].rolling(window=4, min_periods=1).sum()
            
            result.append(stock_data)
        
        # 合并所有股票的TTM数据
        result_df = pd.concat(result, ignore_index=True)
        
        # 获取最新日期的TTM数据
        latest_ttm = result_df.loc[result_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 只保留TTM列和必要的标识列
        ttm_cols = ['stock_code', 'REPORT_DATE'] + [f'{col}_ttm' for col in ttm_columns if f'{col}_ttm' in latest_ttm.columns]
        
        return latest_ttm[ttm_cols]
    
    def _calculate_profitability_ratios(self, result_df):
        """计算盈利能力比率"""
        # 计算毛利润率 = (营业收入 - 营业成本) / 营业收入
        # 由于缺少营业成本数据，这里使用利润总额/营业收入作为替代
        if 'OPERATE_INCOME' in result_df.columns and 'TOTAL_PROFIT' in result_df.columns:
            result_df['gross_income_ratio'] = self.safe_divide(result_df['TOTAL_PROFIT'], result_df['OPERATE_INCOME'])
        
        # 计算净利润率 = 净利润 / 营业收入
        if 'NETPROFIT' in result_df.columns and 'OPERATE_INCOME' in result_df.columns:
            result_df['netprofit_margin'] = self.safe_divide(result_df['NETPROFIT'], result_df['OPERATE_INCOME'])
        
        # 计算净利润率TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'OPERATE_INCOME_ttm' in result_df.columns:
            result_df['netprofit_margin_ttm'] = self.safe_divide(result_df['NETPROFIT_ttm'], result_df['OPERATE_INCOME_ttm'])
        
        # 计算经营利润率 = 利润总额 / 营业收入
        if 'TOTAL_PROFIT' in result_df.columns and 'OPERATE_INCOME' in result_df.columns:
            result_df['operating_profit_margin'] = self.safe_divide(result_df['TOTAL_PROFIT'], result_df['OPERATE_INCOME'])
        
        # 计算经营利润率TTM
        if 'TOTAL_PROFIT_ttm' in result_df.columns and 'OPERATE_INCOME_ttm' in result_df.columns:
            result_df['operating_profit_margin_ttm'] = self.safe_divide(result_df['TOTAL_PROFIT_ttm'], result_df['OPERATE_INCOME_ttm'])
        
        # 计算利润率 = 净利润 / 营业收入 (与净利润率相同)
        if 'NETPROFIT' in result_df.columns and 'OPERATE_INCOME' in result_df.columns:
            result_df['profit_margin'] = self.safe_divide(result_df['NETPROFIT'], result_df['OPERATE_INCOME'])
        
        # 计算利润率TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'OPERATE_INCOME_ttm' in result_df.columns:
            result_df['profit_margin_ttm'] = self.safe_divide(result_df['NETPROFIT_ttm'], result_df['OPERATE_INCOME_ttm'])
        
        return result_df
    
    def _calculate_return_ratios(self, result_df):
        """计算回报率比率"""
        # 计算ROA = 净利润 / 总资产
        if 'NETPROFIT' in result_df.columns and 'TOTAL_ASSETS' in result_df.columns:
            result_df['roa'] = self.safe_divide(result_df['NETPROFIT'], result_df['TOTAL_ASSETS'])
        
        # 计算ROA TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'TOTAL_ASSETS_ttm' in result_df.columns:
            result_df['roa_ttm'] = self.safe_divide(result_df['NETPROFIT_ttm'], result_df['TOTAL_ASSETS_ttm'])
        
        # 计算平均ROA = 净利润 / 平均总资产
        # 假设平均总资产 = (当前总资产 + 上期总资产) / 2
        if 'NETPROFIT' in result_df.columns and 'TOTAL_ASSETS' in result_df.columns:
            result_df['roa_average'] = self.safe_divide(result_df['NETPROFIT'], result_df['TOTAL_ASSETS'] * 0.9)  # 假设平均资产为当前资产的90%
        
        # 计算平均ROA TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'TOTAL_ASSETS_ttm' in result_df.columns:
            result_df['roa_average_ttm'] = self.safe_divide(result_df['NETPROFIT_ttm'], result_df['TOTAL_ASSETS_ttm'] * 0.9)
        
        # 计算ROA变化 = 当前ROA - 上期ROA
        # 由于缺少历史数据，这里暂时设为0
        if 'roa' in result_df.columns:
            result_df['roa_change'] = 0
        
        # 计算ROA变化TTM
        if 'roa_ttm' in result_df.columns:
            result_df['roa_change_ttm'] = 0
        
        # 计算ROA差值 = ROA - 行业平均ROA
        # 由于缺少行业数据，这里暂时设为0
        if 'roa' in result_df.columns:
            result_df['roa_delta'] = 0
        
        # 计算ROA差值TTM
        if 'roa_ttm' in result_df.columns:
            result_df['roa_delta_ttm'] = 0
        
        # 计算ROE = 净利润 / 股东权益
        if 'NETPROFIT' in result_df.columns and 'TOTAL_EQUITY' in result_df.columns:
            result_df['roe'] = self.safe_divide(result_df['NETPROFIT'], result_df['TOTAL_EQUITY'])
        
        # 计算ROE TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'TOTAL_EQUITY_ttm' in result_df.columns:
            result_df['roe_ttm'] = self.safe_divide(result_df['NETPROFIT_ttm'], result_df['TOTAL_EQUITY_ttm'])
        
        # 计算平均ROE = 净利润 / 平均股东权益
        if 'NETPROFIT' in result_df.columns and 'TOTAL_EQUITY' in result_df.columns:
            result_df['roe_average'] = self.safe_divide(result_df['NETPROFIT'], result_df['TOTAL_EQUITY'] * 0.95)  # 假设平均权益为当前权益的95%
        
        # 计算平均ROE TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'TOTAL_EQUITY_ttm' in result_df.columns:
            result_df['roe_average_ttm'] = self.safe_divide(result_df['NETPROFIT_ttm'], result_df['TOTAL_EQUITY_ttm'] * 0.95)
        
        # 计算ROE变化 = 当前ROE - 上期ROE
        if 'roe' in result_df.columns:
            result_df['roe_change'] = 0
        
        # 计算ROE变化TTM
        if 'roe_ttm' in result_df.columns:
            result_df['roe_change_ttm'] = 0
        
        # 计算ROE差值 = ROE - 行业平均ROE
        if 'roe' in result_df.columns:
            result_df['roe_delta'] = 0
        
        # 计算ROE差值TTM
        if 'roe_ttm' in result_df.columns:
            result_df['roe_delta_ttm'] = 0
        
        # 计算ROIC = NOPAT / 投入资本
        # NOPAT = 净利润 + 所得税费用
        # 投入资本 = 总资产 - 无息流动负债
        
        # 计算NOPAT
        if 'NETPROFIT' in result_df.columns:
            result_df['NOPAT'] = result_df['NETPROFIT'] * 1.25  # 假设所得税费用为净利润的25%
        
        # 计算无息流动负债
        if 'TOTAL_CURRENT_LIAB' in result_df.columns:
            result_df['interest_free_current_liability'] = result_df['TOTAL_CURRENT_LIAB'] * 0.7  # 假设无息流动负债占总流动负债的70%
        
        # 计算投入资本
        if 'TOTAL_ASSETS' in result_df.columns and 'interest_free_current_liability' in result_df.columns:
            result_df['invested_capital'] = result_df['TOTAL_ASSETS'] - result_df['interest_free_current_liability']
        
        # 计算ROIC
        if 'NOPAT' in result_df.columns and 'invested_capital' in result_df.columns:
            result_df['roic'] = self.safe_divide(result_df['NOPAT'], result_df['invested_capital'])
        
        # 计算ROIC TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'TOTAL_ASSETS_ttm' in result_df.columns and 'TOTAL_CURRENT_LIAB_ttm' in result_df.columns:
            NOPAT_ttm = result_df['NETPROFIT_ttm'] * 1.25
            interest_free_current_liability_ttm = result_df['TOTAL_CURRENT_LIAB_ttm'] * 0.7
            invested_capital_ttm = result_df['TOTAL_ASSETS_ttm'] - interest_free_current_liability_ttm
            result_df['roic_ttm'] = self.safe_divide(NOPAT_ttm, invested_capital_ttm)
        
        # 计算平均ROIC = NOPAT / 平均投入资本
        if 'NOPAT' in result_df.columns and 'invested_capital' in result_df.columns:
            result_df['roic_average'] = self.safe_divide(result_df['NOPAT'], result_df['invested_capital'] * 0.95)
        
        # 计算平均ROIC TTM
        if 'roic_ttm' in result_df.columns:
            result_df['roic_average_ttm'] = result_df['roic_ttm'] * 1.05  # 假设平均投入资本为当前投入资本的95%
        
        # 计算ROIC变化 = 当前ROIC - 上期ROIC
        if 'roic' in result_df.columns:
            result_df['roic_change'] = 0
        
        # 计算ROIC变化TTM
        if 'roic_ttm' in result_df.columns:
            result_df['roic_change_ttm'] = 0
        
        # 计算ROIC差值 = ROIC - 行业平均ROIC
        if 'roic' in result_df.columns:
            result_df['roic_delta'] = 0
        
        # 计算ROIC差值TTM
        if 'roic_ttm' in result_df.columns:
            result_df['roic_delta_ttm'] = 0
        
        return result_df
    
    def _calculate_ebitda_ratios(self, result_df):
        """计算EBITDA比率"""
        # 计算EBITDA = 净利润 + 所得税 + 利息支出
        # 由于数据中没有DEPRECIATION和AMORTIZATION列，我们简化计算
        if 'NETPROFIT' in result_df.columns and 'FINANCE_EXPENSE' in result_df.columns:
            result_df['EBITDA'] = result_df['NETPROFIT'] * 1.25 + result_df['FINANCE_EXPENSE']
        
        # 计算EBITDA TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'FINANCE_EXPENSE_ttm' in result_df.columns:
            result_df['EBITDA_ttm'] = result_df['NETPROFIT_ttm'] * 1.25 + result_df['FINANCE_EXPENSE_ttm']
        
        # 计算EBITDA/资产
        if 'EBITDA' in result_df.columns and 'TOTAL_ASSETS' in result_df.columns:
            result_df['ebitda_to_assets'] = self.safe_divide(result_df['EBITDA'], result_df['TOTAL_ASSETS'])
        
        # 计算EBITDA/资产TTM
        if 'EBITDA_ttm' in result_df.columns and 'TOTAL_ASSETS_ttm' in result_df.columns:
            result_df['ebitda_to_assets_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['TOTAL_ASSETS_ttm'])
        
        # 计算EBITDA/平均资产
        if 'EBITDA' in result_df.columns and 'TOTAL_ASSETS' in result_df.columns:
            result_df['ebitda_to_average_assets'] = self.safe_divide(result_df['EBITDA'], result_df['TOTAL_ASSETS'] * 0.9)
        
        # 计算EBITDA/平均资产TTM
        if 'EBITDA_ttm' in result_df.columns and 'TOTAL_ASSETS_ttm' in result_df.columns:
            result_df['ebitda_to_average_assets_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['TOTAL_ASSETS_ttm'] * 0.9)
        
        # 计算EBITDA利润率 = EBITDA / 营业收入
        if 'EBITDA' in result_df.columns and 'OPERATE_INCOME' in result_df.columns:
            result_df['ebitda_margin'] = self.safe_divide(result_df['EBITDA'], result_df['OPERATE_INCOME'])
        
        # 计算EBITDA利润率TTM
        if 'EBITDA_ttm' in result_df.columns and 'OPERATE_INCOME_ttm' in result_df.columns:
            result_df['ebitda_margin_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['OPERATE_INCOME_ttm'])
        
        # 计算EBITDA/股东权益
        if 'EBITDA' in result_df.columns and 'TOTAL_EQUITY' in result_df.columns:
            result_df['ebitda_to_equity'] = self.safe_divide(result_df['EBITDA'], result_df['TOTAL_EQUITY'])
        
        # 计算EBITDA/股东权益TTM
        if 'EBITDA_ttm' in result_df.columns and 'TOTAL_EQUITY_ttm' in result_df.columns:
            result_df['ebitda_to_equity_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['TOTAL_EQUITY_ttm'])
        
        # 计算EBITDA/平均股东权益
        if 'EBITDA' in result_df.columns and 'TOTAL_EQUITY' in result_df.columns:
            result_df['ebitda_to_average_equity'] = self.safe_divide(result_df['EBITDA'], result_df['TOTAL_EQUITY'] * 0.95)
        
        # 计算EBITDA/平均股东权益TTM
        if 'EBITDA_ttm' in result_df.columns and 'TOTAL_EQUITY_ttm' in result_df.columns:
            result_df['ebitda_to_average_equity_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['TOTAL_EQUITY_ttm'] * 0.95)
        
        # 计算EBITDA/利息支出
        if 'EBITDA' in result_df.columns and 'FINANCE_EXPENSE' in result_df.columns:
            result_df['ebitda_to_interest'] = self.safe_divide(result_df['EBITDA'], result_df['FINANCE_EXPENSE'])
        
        # 计算EBITDA/利息支出TTM
        if 'EBITDA_ttm' in result_df.columns and 'FINANCE_EXPENSE_ttm' in result_df.columns:
            result_df['ebitda_to_interest_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['FINANCE_EXPENSE_ttm'])
        
        # 计算EBITDA/营业收入
        if 'EBITDA' in result_df.columns and 'OPERATE_INCOME' in result_df.columns:
            result_df['ebitda_to_sales'] = self.safe_divide(result_df['EBITDA'], result_df['OPERATE_INCOME'])
        
        # 计算EBITDA/营业收入TTM
        if 'EBITDA_ttm' in result_df.columns and 'OPERATE_INCOME_ttm' in result_df.columns:
            result_df['ebitda_to_sales_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['OPERATE_INCOME_ttm'])
        
        # 计算EBITDA/固定资产
        if 'EBITDA' in result_df.columns and 'FIXED_ASSET' in result_df.columns:
            result_df['ebitda_to_fixed_assets'] = self.safe_divide(result_df['EBITDA'], result_df['FIXED_ASSET'])
        
        # 计算EBITDA/固定资产TTM
        if 'EBITDA_ttm' in result_df.columns and 'FIXED_ASSET_ttm' in result_df.columns:
            result_df['ebitda_to_fixed_assets_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['FIXED_ASSET_ttm'])
        
        # 计算EBITDA/流动资产
        if 'EBITDA' in result_df.columns and 'TOTAL_CURRENT_ASSETS' in result_df.columns:
            result_df['ebitda_to_current_assets'] = self.safe_divide(result_df['EBITDA'], result_df['TOTAL_CURRENT_ASSETS'])
        
        # 计算EBITDA/流动资产TTM
        if 'EBITDA_ttm' in result_df.columns and 'TOTAL_CURRENT_ASSETS_ttm' in result_df.columns:
            result_df['ebitda_to_current_assets_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['TOTAL_CURRENT_ASSETS_ttm'])
        
        # 计算EBITDA/有形资产 = EBITDA / (总资产 - 无形资产)
        if 'EBITDA' in result_df.columns and 'TOTAL_ASSETS' in result_df.columns and 'INTANGIBLE_ASSET' in result_df.columns:
            result_df['tangible_assets'] = result_df['TOTAL_ASSETS'] - result_df['INTANGIBLE_ASSET']
            result_df['ebitda_to_tangible_assets'] = self.safe_divide(result_df['EBITDA'], result_df['tangible_assets'])
        
        # 计算EBITDA/有形资产TTM
        if 'EBITDA_ttm' in result_df.columns and 'TOTAL_ASSETS_ttm' in result_df.columns and 'INTANGIBLE_ASSET_ttm' in result_df.columns:
            result_df['tangible_assets_ttm'] = result_df['TOTAL_ASSETS_ttm'] - result_df['INTANGIBLE_ASSET_ttm']
            result_df['ebitda_to_tangible_assets_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['tangible_assets_ttm'])
        
        # 计算EBITDA/营运资本 = EBITDA / (流动资产 - 流动负债)
        if 'EBITDA' in result_df.columns and 'TOTAL_CURRENT_ASSETS' in result_df.columns and 'TOTAL_CURRENT_LIAB' in result_df.columns:
            result_df['working_capital'] = result_df['TOTAL_CURRENT_ASSETS'] - result_df['TOTAL_CURRENT_LIAB']
            result_df['ebitda_to_working_capital'] = self.safe_divide(result_df['EBITDA'], result_df['working_capital'])
        
        # 计算EBITDA/营运资本TTM
        if 'EBITDA_ttm' in result_df.columns and 'TOTAL_CURRENT_ASSETS_ttm' in result_df.columns and 'TOTAL_CURRENT_LIAB_ttm' in result_df.columns:
            result_df['working_capital_ttm'] = result_df['TOTAL_CURRENT_ASSETS_ttm'] - result_df['TOTAL_CURRENT_LIAB_ttm']
            result_df['ebitda_to_working_capital_ttm'] = self.safe_divide(result_df['EBITDA_ttm'], result_df['working_capital_ttm'])
        
        return result_df
    
    def _calculate_ebit_ratios(self, result_df):
        """计算EBIT比率"""
        # 计算EBIT = 净利润 + 所得税 + 利息支出
        if 'NETPROFIT' in result_df.columns and 'FINANCE_EXPENSE' in result_df.columns:
            result_df['EBIT'] = result_df['NETPROFIT'] * 1.25 + result_df['FINANCE_EXPENSE']
        
        # 计算EBIT TTM
        if 'NETPROFIT_ttm' in result_df.columns and 'FINANCE_EXPENSE_ttm' in result_df.columns:
            result_df['EBIT_ttm'] = result_df['NETPROFIT_ttm'] * 1.25 + result_df['FINANCE_EXPENSE_ttm']
        
        # 计算EBIT/资产
        if 'EBIT' in result_df.columns and 'TOTAL_ASSETS' in result_df.columns:
            result_df['ebit_to_assets'] = self.safe_divide(result_df['EBIT'], result_df['TOTAL_ASSETS'])
        
        # 计算EBIT/资产TTM
        if 'EBIT_ttm' in result_df.columns and 'TOTAL_ASSETS_ttm' in result_df.columns:
            result_df['ebit_to_assets_ttm'] = self.safe_divide(result_df['EBIT_ttm'], result_df['TOTAL_ASSETS_ttm'])
        
        # 计算EBIT/平均资产
        if 'EBIT' in result_df.columns and 'TOTAL_ASSETS' in result_df.columns:
            result_df['ebit_to_average_assets'] = self.safe_divide(result_df['EBIT'], result_df['TOTAL_ASSETS'] * 0.9)
        
        # 计算EBIT/平均资产TTM
        if 'EBIT_ttm' in result_df.columns and 'TOTAL_ASSETS_ttm' in result_df.columns:
            result_df['ebit_to_average_assets_ttm'] = self.safe_divide(result_df['EBIT_ttm'], result_df['TOTAL_ASSETS_ttm'] * 0.9)
        
        # 计算EBIT/利息支出
        if 'EBIT' in result_df.columns and 'FINANCE_EXPENSE' in result_df.columns:
            result_df['ebit_to_interest'] = self.safe_divide(result_df['EBIT'], result_df['FINANCE_EXPENSE'])
        
        # 计算EBIT/利息支出TTM
        if 'EBIT_ttm' in result_df.columns and 'FINANCE_EXPENSE_ttm' in result_df.columns:
            result_df['ebit_to_interest_ttm'] = self.safe_divide(result_df['EBIT_ttm'], result_df['FINANCE_EXPENSE_ttm'])
        
        # 计算EBIT/营业收入
        if 'EBIT' in result_df.columns and 'OPERATE_INCOME' in result_df.columns:
            result_df['ebit_to_sales'] = self.safe_divide(result_df['EBIT'], result_df['OPERATE_INCOME'])
        
        # 计算EBIT/营业收入TTM
        if 'EBIT_ttm' in result_df.columns and 'OPERATE_INCOME_ttm' in result_df.columns:
            result_df['ebit_to_sales_ttm'] = self.safe_divide(result_df['EBIT_ttm'], result_df['OPERATE_INCOME_ttm'])
        
        # 计算EBIT/股东权益
        if 'EBIT' in result_df.columns and 'TOTAL_EQUITY' in result_df.columns:
            result_df['ebit_to_equity'] = self.safe_divide(result_df['EBIT'], result_df['TOTAL_EQUITY'])
        
        # 计算EBIT/股东权益TTM
        if 'EBIT_ttm' in result_df.columns and 'TOTAL_EQUITY_ttm' in result_df.columns:
            result_df['ebit_to_equity_ttm'] = self.safe_divide(result_df['EBIT_ttm'], result_df['TOTAL_EQUITY_ttm'])
        
        # 计算EBIT/平均股东权益
        if 'EBIT' in result_df.columns and 'TOTAL_EQUITY' in result_df.columns:
            result_df['ebit_to_average_equity'] = self.safe_divide(result_df['EBIT'], result_df['TOTAL_EQUITY'] * 0.95)
        
        # 计算EBIT/平均股东权益TTM
        if 'EBIT_ttm' in result_df.columns and 'TOTAL_EQUITY_ttm' in result_df.columns:
            result_df['ebit_to_average_equity_ttm'] = self.safe_divide(result_df['EBIT_ttm'], result_df['TOTAL_EQUITY_ttm'] * 0.95)
        
        # 计算EBIT/利息支出 (与ebit_to_interest相同)
        if 'ebit_to_interest' in result_df.columns:
            result_df['ebit_to_interest_expense'] = result_df['ebit_to_interest']
        
        # 计算EBIT/利息支出TTM (与ebit_to_interest_ttm相同)
        if 'ebit_to_interest_ttm' in result_df.columns:
            result_df['ebit_to_interest_expense_ttm'] = result_df['ebit_to_interest_ttm']
        
        # 计算EBITDA/利息支出 (与ebitda_to_interest相同)
        if 'ebitda_to_interest' in result_df.columns:
            result_df['ebitda_to_interest_expense'] = result_df['ebitda_to_interest']
        
        # 计算EBITDA/利息支出TTM (与ebitda_to_interest_ttm相同)
        if 'ebitda_to_interest_ttm' in result_df.columns:
            result_df['ebitda_to_interest_expense_ttm'] = result_df['ebitda_to_interest_ttm']
        
        return result_df
    
    def _calculate_debt_ratios(self, result_df):
        """计算债务比率"""
        # 计算有形净值债务率 = 总负债 / (股东权益 - 无形资产)
        if 'TOTAL_LIABILITIES' in result_df.columns and 'TOTAL_EQUITY' in result_df.columns and 'INTANGIBLE_ASSET' in result_df.columns:
            result_df['tangible_equity'] = result_df['TOTAL_EQUITY'] - result_df['INTANGIBLE_ASSET']
            result_df['debt_to_tangible_equity_ratio'] = self.safe_divide(result_df['TOTAL_LIABILITIES'], result_df['tangible_equity'])
        
        return result_df