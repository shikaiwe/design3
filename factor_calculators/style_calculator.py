#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格因子计算器
计算风格因子，共28个因子
"""

import pandas as pd
import numpy as np
from base_calculator import BaseFactorCalculator

class StyleCalculator(BaseFactorCalculator):
    """风格因子计算器"""
    
    def __init__(self):
        """初始化风格因子计算器"""
        super().__init__()
        self.factor_names = [
            'beta',                         # Beta值
            'book_to_price_ratio',          # 账面市值比
            'current_ratio',                # 流动比率
            'debt_to_equity_ratio',         # 产权比率
            'dividend_yield',               # 股息率
            'earnings_yield',               # 盈利收益率
            'earnings_to_price_ratio',      # 盈利价格比
            'eps_growth',                   # 每股收益增长率
            'ev_to_ebitda',                 # EV/EBITDA
            'ev_to_sales',                  # EV/营业收入
            'gross_margin',                 # 毛利率
            'leverage_ratio',               # 杠杆比率
            'liquidity_ratio',              # 流动性比率
            'market_cap',                   # 市值
            'net_profit_margin',            # 净利率
            'operating_margin',             # 营业利润率
            'pb_ratio',                     # 市净率
            'pe_ratio',                     # 市盈率
            'price_to_sales_ratio',         # 市销率
            'quick_ratio',                  # 速动比率
            'return_on_assets',             # 资产收益率
            'return_on_equity',             # 净资产收益率
            'revenue_growth',               # 营业收入增长率
            'roic',                         # 投入资本回报率
            'total_debt_to_equity_ratio',   # 总债务权益比
            'total_debt_to_total_assets_ratio',  # 总债务资产比
            'volatility'                    # 波动率
        ]
    
    def calculate(self, financial_data, price_data, index_data, financial_indicator_data=None, industry_data=None):
        """
        计算风格因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            financial_indicator_data: 财务指标数据DataFrame (可选)
            industry_data: 行业分类数据DataFrame (可选)
            
        返回:
            风格因子DataFrame
        """
        # 准备数据
        financial_df, price_df, index_df, financial_indicator_df, industry_df = self.prepare_data(
            financial_data, 
            price_data, 
            index_data, 
            financial_indicator_data, 
            industry_data
        )
        
        # 映射中文列名为英文列名
        price_df = self._map_chinese_columns(price_df)
        
        # 获取最新日期的数据
        latest_date = price_df['date'].max()
        latest_prices = price_df[price_df['date'] == latest_date]
        
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 初始化结果DataFrame
        result = latest_prices[['stock_code', 'date']].copy()
        
        # 计算风格因子
        result = self._calculate_valuation_ratios(result, latest_prices, latest_financial)
        result = self._calculate_profitability_ratios(result, latest_financial)
        result = self._calculate_growth_ratios(result, latest_financial)
        result = self._calculate_liquidity_ratios(result, latest_financial)
        result = self._calculate_leverage_ratios(result, latest_financial)
        result = self._calculate_market_based_factors(result, latest_prices, price_df, index_df)
        
        return result
    
    def _map_chinese_columns(self, df):
        """将中文列名映射为英文列名"""
        # 检查是否已经存在英文列名，避免重复映射
        if 'close' in df.columns and 'date' in df.columns:
            # 如果已经有英文列名，可能已经映射过了，直接返回
            return df
            
        column_mapping = {
            '日期': 'date',
            '股票代码': 'stock_code_cn',  # 避免与现有的stock_code列冲突
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_pct',
            '涨跌额': 'change_amount',
            '换手率': 'turnover'
        }
        
        # 只重命名存在的列
        for chinese_col, english_col in column_mapping.items():
            if chinese_col in df.columns and english_col not in df.columns:
                df = df.rename(columns={chinese_col: english_col})
        
        # 确保date列是datetime类型
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                print(f"转换日期列时出错: {e}")
                # 如果转换失败，尝试使用其他方法
                pass
        
        return df
    
    def _calculate_valuation_ratios(self, result_df, latest_prices, latest_financial):
        """计算估值比率"""
        # 合并价格和财务数据
        merged = pd.merge(latest_prices, latest_financial, on='stock_code', how='left')
        
        # 计算市值 = 收盘价 * 总股本
        if 'close' in merged.columns and 'SHARE_CAPITAL' in merged.columns:
            merged['market_cap'] = merged['close'] * merged['SHARE_CAPITAL']
        
        # 计算市盈率 = 市值 / 净利润
        if 'market_cap' in merged.columns and 'NETPROFIT' in merged.columns:
            merged['pe_ratio'] = self.safe_divide(merged['market_cap'], merged['NETPROFIT'])
        
        # 计算市净率 = 市值 / 股东权益
        if 'market_cap' in merged.columns and 'TOTAL_EQUITY' in merged.columns:
            merged['pb_ratio'] = self.safe_divide(merged['market_cap'], merged['TOTAL_EQUITY'])
        
        # 计算市销率 = 市值 / 营业收入
        if 'market_cap' in merged.columns and 'OPERATE_INCOME' in merged.columns:
            merged['price_to_sales_ratio'] = self.safe_divide(merged['market_cap'], merged['OPERATE_INCOME'])
        
        # 计算账面市值比 = 股东权益 / 市值
        if 'TOTAL_EQUITY' in merged.columns and 'market_cap' in merged.columns:
            merged['book_to_price_ratio'] = self.safe_divide(merged['TOTAL_EQUITY'], merged['market_cap'])
        
        # 计算盈利收益率 = 净利润 / 市值
        if 'NETPROFIT' in merged.columns and 'market_cap' in merged.columns:
            merged['earnings_yield'] = self.safe_divide(merged['NETPROFIT'], merged['market_cap'])
        
        # 计算盈利价格比 = 净利润 / 市值 (与盈利收益率相同)
        if 'earnings_yield' in merged.columns:
            merged['earnings_to_price_ratio'] = merged['earnings_yield']
        
        # 计算EV = 市值 + 总负债 - 现金
        if 'market_cap' in merged.columns and 'TOTAL_LIABILITIES' in merged.columns and 'MONETARYFUNDS' in merged.columns:
            merged['ev'] = merged['market_cap'] + merged['TOTAL_LIABILITIES'] - merged['MONETARYFUNDS']
        
        # 计算EBITDA = 净利润 + 所得税 + 利息支出 + 折旧 + 摊销
        if 'NETPROFIT' in merged.columns and 'FINANCE_EXPENSE' in merged.columns and 'DEPRECIATION' in merged.columns and 'AMORTIZATION' in merged.columns:
            merged['EBITDA'] = merged['NETPROFIT'] * 1.25 + merged['FINANCE_EXPENSE'] + merged['DEPRECIATION'] + merged['AMORTIZATION']
        
        # 计算EV/EBITDA
        if 'ev' in merged.columns and 'EBITDA' in merged.columns:
            merged['ev_to_ebitda'] = self.safe_divide(merged['ev'], merged['EBITDA'])
        
        # 计算EV/营业收入
        if 'ev' in merged.columns and 'OPERATE_INCOME' in merged.columns:
            merged['ev_to_sales'] = self.safe_divide(merged['ev'], merged['OPERATE_INCOME'])
        
        # 计算股息率 = 股息 / 市值
        # 由于缺少股息数据，这里暂时设为0
        merged['dividend_yield'] = 0
        
        # 合并到结果
        for col in ['market_cap', 'pe_ratio', 'pb_ratio', 'price_to_sales_ratio', 
                   'book_to_price_ratio', 'earnings_yield', 'earnings_to_price_ratio',
                   'ev_to_ebitda', 'ev_to_sales', 'dividend_yield']:
            if col in merged.columns:
                result_df[col] = merged[col]
        
        return result_df
    
    def _calculate_profitability_ratios(self, result_df, latest_financial):
        """计算盈利能力比率"""
        # 计算毛利率 = (营业收入 - 营业成本) / 营业收入
        # 由于缺少营业成本数据，这里使用利润总额/营业收入作为替代
        if 'OPERATE_INCOME' in latest_financial.columns and 'TOTAL_PROFIT' in latest_financial.columns:
            result_df['gross_margin'] = self.safe_divide(latest_financial['TOTAL_PROFIT'], latest_financial['OPERATE_INCOME'])
        
        # 计算净利率 = 净利润 / 营业收入
        if 'NETPROFIT' in latest_financial.columns and 'OPERATE_INCOME' in latest_financial.columns:
            result_df['net_profit_margin'] = self.safe_divide(latest_financial['NETPROFIT'], latest_financial['OPERATE_INCOME'])
        
        # 计算营业利润率 = 利润总额 / 营业收入
        if 'TOTAL_PROFIT' in latest_financial.columns and 'OPERATE_INCOME' in latest_financial.columns:
            result_df['operating_margin'] = self.safe_divide(latest_financial['TOTAL_PROFIT'], latest_financial['OPERATE_INCOME'])
        
        # 计算资产收益率 = 净利润 / 总资产
        if 'NETPROFIT' in latest_financial.columns and 'TOTAL_ASSETS' in latest_financial.columns:
            result_df['return_on_assets'] = self.safe_divide(latest_financial['NETPROFIT'], latest_financial['TOTAL_ASSETS'])
        
        # 计算净资产收益率 = 净利润 / 股东权益
        if 'NETPROFIT' in latest_financial.columns and 'TOTAL_EQUITY' in latest_financial.columns:
            result_df['return_on_equity'] = self.safe_divide(latest_financial['NETPROFIT'], latest_financial['TOTAL_EQUITY'])
        
        # 计算投入资本回报率
        # NOPAT = 净利润 + 所得税费用
        # 投入资本 = 总资产 - 无息流动负债
        
        # 计算NOPAT
        if 'NETPROFIT' in latest_financial.columns:
            nopat = latest_financial['NETPROFIT'] * 1.25  # 假设所得税费用为净利润的25%
        else:
            nopat = None
        
        # 计算无息流动负债
        if 'TOTAL_CURRENT_LIABILITIES' in latest_financial.columns:
            interest_free_current_liability = latest_financial['TOTAL_CURRENT_LIABILITIES'] * 0.7  # 假设无息流动负债占总流动负债的70%
        else:
            interest_free_current_liability = None
        
        # 计算投入资本
        if 'TOTAL_ASSETS' in latest_financial.columns and interest_free_current_liability is not None:
            invested_capital = latest_financial['TOTAL_ASSETS'] - interest_free_current_liability
        else:
            invested_capital = None
        
        # 计算ROIC
        if nopat is not None and invested_capital is not None:
            result_df['roic'] = self.safe_divide(nopat, invested_capital)
        
        return result_df
    
    def _calculate_growth_ratios(self, result_df, latest_financial):
        """计算成长比率"""
        # 计算每股收益增长率
        # 由于缺少历史数据，这里暂时设为0
        result_df['eps_growth'] = 0
        
        # 计算营业收入增长率
        # 由于缺少历史数据，这里暂时设为0
        result_df['revenue_growth'] = 0
        
        return result_df
    
    def _calculate_liquidity_ratios(self, result_df, latest_financial):
        """计算流动性比率"""
        # 计算流动比率 = 流动资产 / 流动负债
        if 'TOTAL_CURRENT_ASSETS' in latest_financial.columns and 'TOTAL_CURRENT_LIABILITIES' in latest_financial.columns:
            result_df['current_ratio'] = self.safe_divide(latest_financial['TOTAL_CURRENT_ASSETS'], latest_financial['TOTAL_CURRENT_LIABILITIES'])
        
        # 计算速动比率 = (流动资产 - 存货) / 流动负债
        # 由于缺少存货数据，这里使用流动资产的80%作为速动资产
        if 'TOTAL_CURRENT_ASSETS' in latest_financial.columns and 'TOTAL_CURRENT_LIABILITIES' in latest_financial.columns:
            quick_assets = latest_financial['TOTAL_CURRENT_ASSETS'] * 0.8  # 假设存货占流动资产的20%
            result_df['quick_ratio'] = self.safe_divide(quick_assets, latest_financial['TOTAL_CURRENT_LIABILITIES'])
        
        # 计算流动性比率 = (流动资产 - 存货) / 流动负债 (与速动比率相同)
        if 'quick_ratio' in result_df.columns:
            result_df['liquidity_ratio'] = result_df['quick_ratio']
        
        return result_df
    
    def _calculate_leverage_ratios(self, result_df, latest_financial):
        """计算杠杆比率"""
        # 计算产权比率 = 总负债 / 股东权益
        if 'TOTAL_LIABILITIES' in latest_financial.columns and 'TOTAL_EQUITY' in latest_financial.columns:
            result_df['debt_to_equity_ratio'] = self.safe_divide(latest_financial['TOTAL_LIABILITIES'], latest_financial['TOTAL_EQUITY'])
        
        # 计算总债务权益比 = 总负债 / 股东权益 (与产权比率相同)
        if 'debt_to_equity_ratio' in result_df.columns:
            result_df['total_debt_to_equity_ratio'] = result_df['debt_to_equity_ratio']
        
        # 计算总债务资产比 = 总负债 / 总资产
        if 'TOTAL_LIABILITIES' in latest_financial.columns and 'TOTAL_ASSETS' in latest_financial.columns:
            result_df['total_debt_to_total_assets_ratio'] = self.safe_divide(latest_financial['TOTAL_LIABILITIES'], latest_financial['TOTAL_ASSETS'])
        
        # 计算杠杆比率 = 总负债 / 股东权益 (与产权比率相同)
        if 'debt_to_equity_ratio' in result_df.columns:
            result_df['leverage_ratio'] = result_df['debt_to_equity_ratio']
        
        return result_df
    
    def _calculate_market_based_factors(self, result_df, latest_prices, price_df, index_df):
        """计算基于市场的因子"""
        # 计算Beta值
        result_df = self._calculate_beta(result_df, price_df, index_df)
        
        # 计算波动率
        result_df = self._calculate_volatility(result_df, price_df)
        
        return result_df
    
    def _calculate_beta(self, result_df, price_df, index_df):
        """计算Beta值"""
        # 计算收益率，使用transform确保索引对齐
        price_df = price_df.sort_values(['stock_code', 'date'])
        price_df['return'] = price_df.groupby('stock_code')['close'].transform(
            lambda x: x.pct_change()
        )
        
        # 计算指数收益率
        index_df = index_df.sort_values('date')
        index_df['market_return'] = index_df['close'].pct_change()
        
        # 获取最近60个交易日的数据
        latest_date = price_df['date'].max()
        start_date = latest_date - pd.Timedelta(days=90)  # 大约60个交易日
        
        # 过滤数据
        recent_prices = price_df[price_df['date'] >= start_date]
        recent_index = index_df[index_df['date'] >= start_date]
        
        # 计算每个股票的Beta值
        beta_values = []
        
        for stock_code in result_df['stock_code']:
            stock_returns = recent_prices[recent_prices['stock_code'] == stock_code][['date', 'return']].dropna()
            
            if len(stock_returns) < 10:  # 数据不足
                beta_values.append(np.nan)
                continue
            
            # 获取对应日期的指数收益率
            # 使用merge而不是索引，确保日期匹配
            merged_data = pd.merge(
                stock_returns,
                recent_index[['date', 'market_return']],
                on='date',
                how='inner'
            )
            
            if len(merged_data) < 10:  # 匹配数据不足
                beta_values.append(np.nan)
                continue
            
            # 计算Beta
            stock_returns_values = merged_data['return'].values
            market_returns_values = merged_data['market_return'].values
            
            # 使用线性回归计算Beta
            covariance = np.cov(stock_returns_values, market_returns_values)[0, 1]
            market_variance = np.var(market_returns_values)
            
            if market_variance == 0:
                beta_values.append(np.nan)
            else:
                beta = covariance / market_variance
                beta_values.append(beta)
        
        # 添加Beta值到结果
        result_df['beta'] = beta_values
        
        return result_df
    
    def _calculate_volatility(self, result_df, price_df):
        """计算波动率"""
        # 计算收益率，使用transform确保索引对齐
        price_df = price_df.sort_values(['stock_code', 'date'])
        price_df['return'] = price_df.groupby('stock_code')['close'].transform(
            lambda x: x.pct_change()
        )
        
        # 获取最近60个交易日的数据
        latest_date = price_df['date'].max()
        start_date = latest_date - pd.Timedelta(days=90)  # 大约60个交易日
        
        # 过滤数据
        recent_prices = price_df[price_df['date'] >= start_date]
        
        # 计算每个股票的波动率
        volatility_values = []
        
        for stock_code in result_df['stock_code']:
            stock_returns = recent_prices[recent_prices['stock_code'] == stock_code]['return'].dropna()
            
            if len(stock_returns) < 10:  # 数据不足
                volatility_values.append(np.nan)
            else:
                # 计算年化波动率
                volatility = stock_returns.std() * np.sqrt(252)
                volatility_values.append(volatility)
        
        # 添加波动率到结果
        result_df['volatility'] = volatility_values
        
        return result_df