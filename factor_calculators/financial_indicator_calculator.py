#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
财务指标和行业因子计算器
基于财务指标数据和行业分类数据计算因子
"""

import pandas as pd
import numpy as np
from base_calculator import BaseFactorCalculator

class FinancialIndicatorCalculator(BaseFactorCalculator):
    """财务指标和行业因子计算器"""
    
    def __init__(self):
        """初始化财务指标和行业因子计算器"""
        super().__init__()
        self.factor_names = [
            # 财务指标因子
            'eps_diluted',               # 稀释每股收益
            'eps_weighted',              # 加权每股收益
            'eps_adjusted',              # 调整每股收益
            'eps_deducted',              # 扣非每股收益
            'bps_before_adj',            # 调整前每股净资产
            'bps_after_adj',             # 调整后每股净资产
            'ocfps',                     # 经营现金流每股
            'reserve_per_share',         # 每股公积金
            'retained_earnings_per_share', # 每股留存收益
            'roa',                       # 资产收益率
            'main_business_profit_margin', # 主营业务利润率
            'net_profit_margin_on_assets', # 净利润率
            'cost_expense_profit_margin', # 成本费用利润率
            'operating_profit_margin',    # 经营利润率
            'main_business_cost_ratio',  # 主营业务成本率
            'net_sales_margin',          # 销售净利率
            'capital_return_rate',        # 资本回报率
            'roe_return',                # 净资产收益率
            'asset_return_rate',         # 资产回报率
            'gross_profit_margin',       # 毛利率
            'three_expense_ratio',       # 三费占比
            'non_main_business_ratio',   # 非主营占比
            'main_business_profit_ratio', # 主营利润占比
            'dividend_payout_ratio',     # 股利支付率
            'investment_return_rate',    # 投资回报率
            'roe',                       # 净资产收益率
            'weighted_roe',              # 加权净资产收益率
            'net_profit_deducted',        # 扣非净利润
            'main_business_revenue_growth', # 主营收入增长率
            'net_profit_growth',         # 净利润增长率
            'net_asset_growth',          # 净资产增长率
            'total_asset_growth',        # 总资产增长率
            'accounts_receivable_turnover', # 应收账款周转率
            'accounts_receivable_turnover_days', # 应收账款周转天数
            'inventory_turnover_days',    # 存货周转天数
            'inventory_turnover',         # 存货周转率
            'fixed_asset_turnover',       # 固定资产周转率
            'total_asset_turnover',       # 总资产周转率
            'total_asset_turnover_days',  # 总资产周转天数
            'current_asset_turnover',     # 流动资产周转率
            'current_asset_turnover_days', # 流动资产周转天数
            'shareholder_equity_turnover', # 股东权益周转率
            'current_ratio',             # 流动比率
            'quick_ratio',               # 速动比率
            'cash_ratio',                # 现金比率
            'interest_coverage_ratio',   # 利息保障倍数
            'long_term_debt_to_working_capital', # 长期债务与营运资金比率
            'shareholder_equity_ratio',  # 股东权益比率
            'long_term_debt_ratio',      # 长期负债比率
            'equity_to_fixed_assets_ratio', # 权益对固定资产比率
            'debt_to_equity_ratio',      # 产权比率
            'long_term_assets_to_funds_ratio', # 长期资产与资金比率
            'capitalization_ratio',      # 资本化比率
            'fixed_asset_net_value_ratio', # 固定资产净值比率
            'capital_fixed_ratio',       # 资本固定比率
            'property_rights_ratio',     # 产权比率
            'liquidation_value_ratio',   # 清算价值比率
            'fixed_asset_ratio',         # 固定资产比率
            'debt_to_asset_ratio',       # 资产负债率
            'total_assets',              # 总资产
            'operating_cash_to_sales_ratio', # 经营现金流与销售收入比率
            'operating_cash_return_on_assets', # 经营现金流资产回报率
            'operating_cash_to_net_profit_ratio', # 经营现金流与净利润比率
            'operating_cash_to_debt_ratio', # 经营现金流与负债比率
            'cash_flow_ratio',          # 现金流比率
            
            # 行业因子
            'industry_code_cs',          # 申万行业代码
            'industry_code_sw2014',      # 申万2014行业代码
            'industry_code_sw2021',      # 申万2021行业代码
            'industry_name_cs',          # 申万行业名称
            'industry_name_sw2014',      # 申万2014行业名称
            'industry_name_sw2021',      # 申万2021行业名称
        ]
    
    def calculate(self, financial_data, price_data, index_data, financial_indicator_data=None, industry_data=None):
        """
        计算财务指标和行业因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            financial_indicator_data: 财务指标数据DataFrame (可选)
            industry_data: 行业分类数据DataFrame (可选)
            
        返回:
            财务指标和行业因子DataFrame
        """
        # 准备数据
        financial_df, price_df, index_df, financial_indicator_df, industry_df = self.prepare_data(
            financial_data, price_data, index_data, financial_indicator_data, industry_data
        )
        
        # 获取最新日期的价格数据
        latest_price = price_df.loc[price_df.groupby('stock_code')['date'].idxmax()]
        
        # 初始化结果DataFrame
        result = latest_price[['stock_code', 'date']].copy()
        
        # 计算财务指标因子
        if financial_indicator_df is not None:
            result = self._calculate_financial_indicator_factors(result, financial_indicator_df)
        
        # 计算行业因子
        if industry_df is not None:
            result = self._calculate_industry_factors(result, industry_df)
        
        # 选择需要的列
        columns_to_keep = ['stock_code', 'date'] + self.factor_names
        # 确保只选择实际存在的列
        existing_columns = [col for col in columns_to_keep if col in result.columns]
        result = result[existing_columns]
        
        return result
    
    def _calculate_financial_indicator_factors(self, result_df, financial_indicator_df):
        """计算财务指标因子"""
        # 获取最新日期的财务指标数据
        latest_financial_indicator = financial_indicator_df.loc[
            financial_indicator_df.groupby('symbol')['date'].idxmax()
        ]
        
        # 将symbol列重命名为stock_code以保持一致性
        latest_financial_indicator = latest_financial_indicator.rename(columns={'symbol': 'stock_code'})
        
        # 确保stock_code列的数据类型一致
        latest_financial_indicator['stock_code'] = latest_financial_indicator['stock_code'].astype(str)
        result_df['stock_code'] = result_df['stock_code'].astype(str)
        
        # 获取财务指标数据中实际存在的列
        financial_cols = ['stock_code'] + [col for col in self.factor_names[:67] if col in latest_financial_indicator.columns]
        
        # 合并财务指标数据
        result_df = pd.merge(
            result_df,
            latest_financial_indicator[financial_cols],
            on='stock_code',
            how='left'
        )
        
        return result_df
    
    def _calculate_industry_factors(self, result_df, industry_df):
        """计算行业因子"""
        # 获取最新日期的行业数据
        latest_industry = industry_df.loc[industry_df.groupby('instrument')['date'].idxmax()]
        
        # 将instrument列重命名为stock_code以保持一致性
        latest_industry = latest_industry.rename(columns={'instrument': 'stock_code'})
        
        # 确保stock_code列的数据类型一致
        latest_industry['stock_code'] = latest_industry['stock_code'].astype(str)
        result_df['stock_code'] = result_df['stock_code'].astype(str)
        
        # 创建行业编码的数值因子
        industry_mapping = {}
        
        # 为每个行业分类系统创建编码映射
        for col in ['cs', 'sw2014', 'sw2021']:
            if col in latest_industry.columns:
                unique_industries = latest_industry[col].dropna().unique()
                industry_mapping[col] = {industry: i for i, industry in enumerate(unique_industries)}
                
                # 添加行业编码因子
                latest_industry[f'industry_code_{col}'] = latest_industry[col].map(industry_mapping[col])
        
        # 合并行业数据
        industry_cols = ['stock_code', 'industry_name_cs', 'industry_name_sw2014', 'industry_name_sw2021',
                         'industry_code_cs', 'industry_code_sw2014', 'industry_code_sw2021']
        
        # 只选择存在的列
        existing_cols = [col for col in industry_cols if col in latest_industry.columns]
        
        result_df = pd.merge(
            result_df,
            latest_industry[existing_cols],
            on='stock_code',
            how='left'
        )
        
        return result_df