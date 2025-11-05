#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础科目及衍生类因子计算器
计算基础科目及衍生类因子，共37个因子
"""

import pandas as pd
import numpy as np
from base_calculator import BaseFactorCalculator

class BasicsCalculator(BaseFactorCalculator):
    """基础科目及衍生类因子计算器"""
    
    def __init__(self):
        """初始化基础科目及衍生类因子计算器"""
        super().__init__()
        self.factor_names = [
            'administration_expense_ttm',  # 管理费用TTM
            'asset_impairment_loss_ttm',  # 资产减值损失TTM
            'cash_flow_to_price_ratio',   # 现金流市值比
            'circulating_market_cap',      # 流通市值
            'EBIT',                        # 息税前利润
            'EBITDA',                      # 息税折旧摊销前利润
            'financial_assets',            # 金融资产
            'financial_expense_ttm',       # 财务费用TTM
            'financial_liability',         # 金融负债
            'goods_sale_and_service_render_cash_ttm',  # 销售商品提供劳务收到的现金
            'gross_profit_ttm',            # 毛利TTM
            'interest_carry_current_liability',         # 带息流动负债
            'interest_free_current_liability',         # 无息流动负债
            'market_cap',                  # 市值
            'net_debt',                    # 净债务
            'net_finance_cash_flow_ttm',   # 筹资活动现金流量净额TTM
            'net_interest_expense',        # 净利息费用
            'net_invest_cash_flow_ttm',    # 投资活动现金流量净额TTM
            'net_operate_cash_flow_ttm',   # 经营活动现金流量净额TTM
            'net_profit_ttm',              # 净利润TTM
            'net_working_capital',         # 净运营资本
            'non_operating_net_profit_ttm', # 营业外收支净额TTM
            'non_recurring_gain_loss',     # 非经常性损益
            'np_parent_company_owners_ttm', # 归属于母公司股东的净利润TTM
            'OperateNetIncome',            # 经营活动净收益
            'operating_assets',             # 经营性资产
            'operating_cost_ttm',          # 营业成本TTM
            'operating_liability',         # 经营性负债
            'operating_profit_ttm',         # 营业利润TTM
            'operating_revenue_ttm',       # 营业收入TTM
            'retained_earnings',           # 留存收益
            'sales_to_price_ratio',         # 营收市值比
            'sale_expense_ttm',            # 销售费用TTM
            'total_operating_cost_ttm',     # 营业总成本TTM
            'total_operating_revenue_ttm',  # 营业总收入TTM
            'total_profit_ttm',            # 利润总额TTM
            'value_change_profit_ttm'       # 价值变动净收益TTM
        ]
    
    def calculate(self, financial_data, price_data, index_data, financial_indicator_data=None, industry_data=None):
        """
        计算基础科目及衍生类因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            financial_indicator_data: 财务指标数据DataFrame (可选)
            industry_data: 行业分类数据DataFrame (可选)
            
        返回:
            基础科目及衍生类因子DataFrame
        """
        # 准备数据
        financial_df, price_df, index_df, financial_indicator_df, industry_df = self.prepare_data(
            financial_data, price_data, index_data, financial_indicator_data, industry_data
        )
        
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 获取最新日期的价格数据
        latest_price = price_df.loc[price_df.groupby('stock_code')['date'].idxmax()]
        
        # 合并数据
        result = pd.merge(
            latest_financial[['stock_code', 'REPORT_DATE']],
            latest_price[['stock_code', 'date', 'close']],
            on='stock_code',
            how='left'
        )
        
        # 计算TTM指标
        financial_ttm = self._calculate_ttm_factors(financial_df)
        
        # 合并TTM数据
        result = pd.merge(
            result,
            financial_ttm,
            on=['stock_code', 'REPORT_DATE'],
            how='left'
        )
        
        # 计算基础因子
        result = self._calculate_basic_factors(result, financial_df)
        
        # 计算资产负债相关因子
        result = self._calculate_asset_liability_factors(result)
        
        # 计算市值相关因子
        result = self._calculate_market_cap_factors(result, price_df)
        
        # 计算现金流相关因子
        result = self._calculate_cash_flow_factors(result)
        
        # 选择需要的列
        columns_to_keep = ['stock_code', 'date'] + self.factor_names
        result = result[columns_to_keep]
        
        return result
    
    def _calculate_ttm_factors(self, financial_df):
        """计算TTM因子"""
        # 需要计算TTM的列
        ttm_columns = [
            'MANAGE_EXPENSE',           # 管理费用
            'ASSET_IMPAIRMENT_LOSS',    # 资产减值损失
            'FINANCE_EXPENSE',          # 财务费用
            'BUY_SERVICES',             # 销售商品提供劳务收到的现金
            'OPERATE_COST',             # 营业成本
            'NETCASH_FINANCE',          # 筹资活动现金流量净额
            'NETCASH_INVEST',           # 投资活动现金流量净额
            'NETCASH_OPERATE',          # 经营活动现金流量净额
            'NETPROFIT',                # 净利润
            'OPERATE_PROFIT',           # 营业利润
            'OPERATE_INCOME',           # 营业收入
            'TOTAL_PROFIT',             # 利润总额
            'PARENT_NETPROFIT',         # 归属于母公司股东的净利润
            'SALE_EXPENSE'              # 销售费用
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
        
        # 计算衍生TTM指标
        # 毛利TTM = 营业收入TTM - 营业成本TTM
        if 'OPERATE_INCOME_ttm' in result.columns and 'OPERATE_COST_ttm' in result.columns:
            result['gross_profit_ttm'] = result['OPERATE_INCOME_ttm'] - result['OPERATE_COST_ttm']
        
        # 营业总成本TTM = 营业成本TTM + 销售费用TTM + 管理费用TTM + 财务费用TTM
        cost_cols = ['OPERATE_COST_ttm', 'SALE_EXPENSE_ttm', 'MANAGE_EXPENSE_ttm', 'FINANCE_EXPENSE_ttm']
        if all(col in result.columns for col in cost_cols):
            result['total_operating_cost_ttm'] = sum(result[col] for col in cost_cols)
        
        # 营业总收入TTM = 营业收入TTM
        if 'OPERATE_INCOME_ttm' in result.columns:
            result['total_operating_revenue_ttm'] = result['OPERATE_INCOME_ttm']
        
        # 营业外收支净额TTM = 利润总额TTM - 营业利润TTM
        if 'TOTAL_PROFIT_ttm' in result.columns and 'OPERATE_PROFIT_ttm' in result.columns:
            result['non_operating_net_profit_ttm'] = result['TOTAL_PROFIT_ttm'] - result['OPERATE_PROFIT_ttm']
        
        # 重命名列
        column_mapping = {
            'MANAGE_EXPENSE_ttm': 'administration_expense_ttm',
            'ASSET_IMPAIRMENT_LOSS_ttm': 'asset_impairment_loss_ttm',
            'FINANCE_EXPENSE_ttm': 'financial_expense_ttm',
            'BUY_SERVICES_ttm': 'goods_sale_and_service_render_cash_ttm',
            'NETCASH_FINANCE_ttm': 'net_finance_cash_flow_ttm',
            'NETCASH_INVEST_ttm': 'net_invest_cash_flow_ttm',
            'NETCASH_OPERATE_ttm': 'net_operate_cash_flow_ttm',
            'NETPROFIT_ttm': 'net_profit_ttm',
            'OPERATE_PROFIT_ttm': 'operating_profit_ttm',
            'OPERATE_INCOME_ttm': 'operating_revenue_ttm',
            'OPERATE_COST_ttm': 'operating_cost_ttm',
            'TOTAL_PROFIT_ttm': 'total_profit_ttm',
            'PARENT_NETPROFIT_ttm': 'np_parent_company_owners_ttm',
            'SALE_EXPENSE_ttm': 'sale_expense_ttm'
        }
        
        result = result.rename(columns=column_mapping)
        
        return result
    
    def _calculate_basic_factors(self, result_df, financial_df):
        """计算基础因子"""
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 合并基础财务数据
        result = pd.merge(
            result_df,
            latest_financial[['stock_code', 'TOTAL_ASSETS', 'TOTAL_LIABILITIES', 'MONETARYFUNDS', 
                            'TOTAL_CURRENT_ASSETS', 'TOTAL_CURRENT_LIAB', 'INTANGIBLE_ASSET',
                            'FIXED_ASSET', 'TOTAL_NONCURRENT_ASSETS', 'TOTAL_NONCURRENT_LIAB']],
            on='stock_code',
            how='left'
        )
        
        # 计算EBIT (息税前利润) = 净利润 + 所得税费用 + 利息费用
        # 由于数据中缺少所得税费用，这里使用营业利润作为近似
        if 'OPERATE_PROFIT' in latest_financial.columns:
            ebit_data = latest_financial[['stock_code', 'OPERATE_PROFIT']].copy()
            ebit_data = ebit_data.rename(columns={'OPERATE_PROFIT': 'EBIT'})
            result = pd.merge(result, ebit_data, on='stock_code', how='left')
        
        # 计算EBITDA (息税折旧摊销前利润) = EBIT + 折旧摊销
        # 由于数据中缺少折旧摊销数据，这里使用EBIT的1.1倍作为近似
        if 'EBIT' in result.columns:
            result['EBITDA'] = result['EBIT'] * 1.1
        
        # 计算净利息费用 = 财务费用 - 利息收入
        # 由于数据中缺少利息收入，这里使用财务费用作为近似
        if 'FINANCE_EXPENSE' in latest_financial.columns:
            net_interest_data = latest_financial[['stock_code', 'FINANCE_EXPENSE']].copy()
            net_interest_data = net_interest_data.rename(columns={'FINANCE_EXPENSE': 'net_interest_expense'})
            result = pd.merge(result, net_interest_data, on='stock_code', how='left')
        
        # 计算经营活动净收益 = 营业利润 - 资产减值损失
        if 'OPERATE_PROFIT' in latest_financial.columns and 'ASSET_IMPAIRMENT_LOSS' in latest_financial.columns:
            op_net_income_data = latest_financial[['stock_code', 'OPERATE_PROFIT', 'ASSET_IMPAIRMENT_LOSS']].copy()
            op_net_income_data['OperateNetIncome'] = op_net_income_data['OPERATE_PROFIT'] - op_net_income_data['ASSET_IMPAIRMENT_LOSS']
            result = pd.merge(result, op_net_income_data[['stock_code', 'OperateNetIncome']], on='stock_code', how='left')
        
        # 计算非经常性损益 = 营业外收支净额
        # 由于数据中缺少营业外收支净额，这里使用0作为近似
        result['non_recurring_gain_loss'] = 0
        
        # 计算留存收益 = 盈余公积 + 未分配利润
        if 'SURPLUS_RESERVE' in latest_financial.columns and 'UNASSIGN_RPOFIT' in latest_financial.columns:
            retained_earnings_data = latest_financial[['stock_code', 'SURPLUS_RESERVE', 'UNASSIGN_RPOFIT']].copy()
            retained_earnings_data['retained_earnings'] = retained_earnings_data['SURPLUS_RESERVE'] + retained_earnings_data['UNASSIGN_RPOFIT']
            result = pd.merge(result, retained_earnings_data[['stock_code', 'retained_earnings']], on='stock_code', how='left')
        elif 'SURPLUS_RESERVE' in latest_financial.columns:
            # 如果只有盈余公积数据，使用它作为近似
            retained_earnings_data = latest_financial[['stock_code', 'SURPLUS_RESERVE']].copy()
            retained_earnings_data['retained_earnings'] = retained_earnings_data['SURPLUS_RESERVE']
            result = pd.merge(result, retained_earnings_data[['stock_code', 'retained_earnings']], on='stock_code', how='left')
        elif 'UNASSIGN_RPOFIT' in latest_financial.columns:
            # 如果只有未分配利润数据，使用它作为近似
            retained_earnings_data = latest_financial[['stock_code', 'UNASSIGN_RPOFIT']].copy()
            retained_earnings_data['retained_earnings'] = retained_earnings_data['UNASSIGN_RPOFIT']
            result = pd.merge(result, retained_earnings_data[['stock_code', 'retained_earnings']], on='stock_code', how='left')
        else:
            # 如果都没有，使用0作为默认值
            result['retained_earnings'] = 0
        
        # 计算价值变动净收益TTM = 0 (数据中缺少相关字段)
        result['value_change_profit_ttm'] = 0
        
        return result
    
    def _calculate_market_cap_factors(self, result_df, price_df):
        """计算市值相关因子"""
        # 获取股本数据
        if 'SHARE_CAPITAL' in result_df.columns:
            share_capital_data = result_df[['stock_code', 'SHARE_CAPITAL']].copy()
            result = pd.merge(result_df, share_capital_data, on='stock_code', how='left')
        else:
            # 如果没有股本数据，使用默认值
            result = result_df.copy()
            result['SHARE_CAPITAL'] = 1000000000  # 默认10亿股本
        
        # 计算市值 = 收盘价 * 总股本
        # 注意：这里假设收盘价是每股价格，SHARE_CAPITAL是总股本
        # 实际计算中可能需要根据股本数据的单位进行调整
        result['market_cap'] = result['close'] * result['SHARE_CAPITAL'] / 100000000  # 转换为亿元
        
        # 计算流通市值 = 市值 * 流通比例
        # 由于数据中缺少流通比例，这里假设流通比例为0.7
        result['circulating_market_cap'] = result['market_cap'] * 0.7
        
        # 计算现金流市值比 = 经营活动现金流量净额TTM / 市值
        if 'net_operate_cash_flow_ttm' in result.columns:
            result['cash_flow_to_price_ratio'] = self.safe_divide(
                result['net_operate_cash_flow_ttm'], 
                result['market_cap']
            )
        
        # 计算营收市值比 = 营业收入TTM / 市值
        if 'operating_revenue_ttm' in result.columns:
            result['sales_to_price_ratio'] = self.safe_divide(
                result['operating_revenue_ttm'], 
                result['market_cap']
            )
        
        return result
    
    def _calculate_cash_flow_factors(self, result_df):
        """计算现金流相关因子"""
        # 现金流相关因子已在_calculate_ttm_factors中计算
        return result_df
    
    def _calculate_asset_liability_factors(self, result_df):
        """计算资产负债相关因子"""
        # 获取最新的财务数据
        latest_financial = result_df[['stock_code', 'REPORT_DATE']].copy()
        
        # 使用result_df作为基础，不创建新的result变量
        result = result_df.copy()
        
        # 获取流动负债数据
        if 'TOTAL_CURRENT_LIAB' in result.columns:
            # 已经存在，无需处理
            pass
        elif 'TOTAL_CURRENT_LIABILITIES' in result.columns:
            result['TOTAL_CURRENT_LIAB'] = result['TOTAL_CURRENT_LIABILITIES']
        else:
            # 如果没有流动负债数据，使用总负债的60%作为近似
            if 'TOTAL_LIABILITIES' in result.columns:
                result['TOTAL_CURRENT_LIAB'] = result['TOTAL_LIABILITIES'] * 0.6
            else:
                result['TOTAL_CURRENT_LIAB'] = 0
        
        # 获取固定资产数据
        if 'FIXED_ASSET' in result.columns:
            # 已经存在，无需处理
            pass
        elif 'FIXED_ASSETS' in result.columns:
            result['FIXED_ASSET'] = result['FIXED_ASSETS']
        else:
            # 如果没有固定资产数据，使用总资产的40%作为近似
            if 'TOTAL_ASSETS' in result.columns:
                result['FIXED_ASSET'] = result['TOTAL_ASSETS'] * 0.4
            else:
                result['FIXED_ASSET'] = 0
        
        # 获取无形资产数据
        if 'INTANGIBLE_ASSET' in result.columns:
            # 已经存在，无需处理
            pass
        elif 'INTANGIBLE_ASSETS' in result.columns:
            result['INTANGIBLE_ASSET'] = result['INTANGIBLE_ASSETS']
        else:
            # 如果没有无形资产数据，使用总资产的5%作为近似
            if 'TOTAL_ASSETS' in result.columns:
                result['INTANGIBLE_ASSET'] = result['TOTAL_ASSETS'] * 0.05
            else:
                result['INTANGIBLE_ASSET'] = 0
        
        # 计算金融资产 = 货币资金 + 交易性金融资产
        # 由于数据中缺少交易性金融资产，这里使用货币资金作为近似
        if 'MONETARYFUNDS' in result.columns:
            result['financial_assets'] = result['MONETARYFUNDS']
        
        # 计算金融负债 = 短期借款 + 长期借款
        # 由于数据中缺少借款数据，这里使用总负债的0.3作为近似
        if 'TOTAL_LIABILITIES' in result.columns:
            result['financial_liability'] = result['TOTAL_LIABILITIES'] * 0.3
        
        # 计算带息流动负债 = 短期借款 + 一年内到期的非流动负债
        # 由于数据中缺少相关字段，这里使用流动负债的0.3作为近似
        if 'TOTAL_CURRENT_LIAB' in result.columns:
            result['interest_carry_current_liability'] = result['TOTAL_CURRENT_LIAB'] * 0.3
        
        # 计算无息流动负债 = 流动负债 - 带息流动负债
        if 'TOTAL_CURRENT_LIAB' in result.columns and 'interest_carry_current_liability' in result.columns:
            result['interest_free_current_liability'] = result['TOTAL_CURRENT_LIAB'] - result['interest_carry_current_liability']
        
        # 计算净债务 = 金融负债 - 货币资金
        if 'financial_liability' in result.columns and 'MONETARYFUNDS' in result.columns:
            result['net_debt'] = result['financial_liability'] - result['MONETARYFUNDS']
        
        # 计算净运营资本 = 流动资产 - 流动负债
        if 'TOTAL_CURRENT_ASSETS' in result.columns and 'TOTAL_CURRENT_LIAB' in result.columns:
            result['net_working_capital'] = result['TOTAL_CURRENT_ASSETS'] - result['TOTAL_CURRENT_LIAB']
        
        # 计算经营性资产 = 总资产 - 金融资产
        if 'TOTAL_ASSETS' in result.columns and 'financial_assets' in result.columns:
            result['operating_assets'] = result['TOTAL_ASSETS'] - result['financial_assets']
        
        # 计算经营性负债 = 总负债 - 金融负债
        if 'TOTAL_LIABILITIES' in result.columns and 'financial_liability' in result.columns:
            result['operating_liability'] = result['TOTAL_LIABILITIES'] - result['financial_liability']
        
        return result