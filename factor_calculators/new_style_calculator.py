#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新风格因子计算器
计算新风格因子，共18个因子
"""

import pandas as pd
import numpy as np
from .base_calculator import BaseFactorCalculator

class NewStyleCalculator(BaseFactorCalculator):
    """新风格因子计算器"""
    
    def __init__(self):
        """初始化新风格因子计算器"""
        super().__init__()
        self.factor_names = [
            'btop',                         # 账面市值比
            'divyild',                      # 股息率
            'dolvol',                       # 成交额波动率
            'earnyild',                     # 盈利收益率
            'growth',                       # 成长性
            'illiquidity',                  # 非流动性
            'indmom',                       # 行业动量
            'lev',                          # 杠杆率
            'maxret',                       # 最大收益率
            'mom12m',                       # 12个月动量
            'mom1m',                        # 1个月动量
            'mom36m',                       # 36个月动量
            'mom6m',                        # 6个月动量
            'price_momentum',               # 价格动量
            'seasonality',                  # 季节性
            'size',                         # 规模
            'turnover',                     # 换手率
            'volatility'                    # 波动率
        ]
    
    def calculate(self, financial_data, price_data, index_data):
        """
        计算新风格因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            
        返回:
            新风格因子DataFrame
        """
        # 准备数据
        financial_df, price_df, index_df = self.prepare_data(financial_data, price_data, index_data)
        
        # 获取最新日期的数据
        latest_date = price_df['date'].max()
        latest_prices = price_df[price_df['date'] == latest_date]
        
        # 获取最新日期的财务数据
        latest_financial = financial_df.loc[financial_df.groupby('stock_code')['REPORT_DATE'].idxmax()]
        
        # 初始化结果DataFrame
        result = latest_prices[['stock_code', 'date']].copy()
        
        # 计算新风格因子
        result = self._calculate_value_factors(result, latest_prices, latest_financial)
        result = self._calculate_momentum_factors(result, price_df)
        result = self._calculate_liquidity_factors(result, price_df, latest_financial)
        result = self._calculate_other_factors(result, price_df)
        
        return result
    
    def _calculate_value_factors(self, result_df, latest_prices, latest_financial):
        """计算价值因子"""
        # 合并价格和财务数据
        merged = pd.merge(latest_prices, latest_financial, on='stock_code', how='left')
        
        # 计算市值 = 收盘价 * 总股本
        if 'close' in merged.columns and 'SHARE_CAPITAL' in merged.columns:
            merged['market_cap'] = merged['close'] * merged['SHARE_CAPITAL']
        
        # 计算账面市值比 = 股东权益 / 市值
        if 'TOTAL_EQUITY' in merged.columns and 'market_cap' in merged.columns:
            merged['btop'] = self.safe_divide(merged['TOTAL_EQUITY'], merged['market_cap'])
        
        # 计算股息率 = 股息 / 市值
        # 由于缺少股息数据，这里暂时设为0
        merged['divyild'] = 0
        
        # 计算盈利收益率 = 净利润 / 市值
        if 'NETPROFIT' in merged.columns and 'market_cap' in merged.columns:
            merged['earnyild'] = self.safe_divide(merged['NETPROFIT'], merged['market_cap'])
        
        # 计算杠杆率 = 总负债 / 股东权益
        if 'TOTAL_LIABILITIES' in merged.columns and 'TOTAL_EQUITY' in merged.columns:
            merged['lev'] = self.safe_divide(merged['TOTAL_LIABILITIES'], merged['TOTAL_EQUITY'])
        
        # 计算规模因子 = 市值的对数
        if 'market_cap' in merged.columns:
            merged['size'] = np.log(merged['market_cap'])
        
        # 合并到结果
        for col in ['btop', 'divyild', 'earnyild', 'lev', 'size']:
            if col in merged.columns:
                result_df[col] = merged[col]
        
        return result_df
    
    def _calculate_momentum_factors(self, result_df, price_df):
        """计算动量因子"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算收益率
        price_df['return'] = price_df.groupby('stock_code')['close'].pct_change()
        
        # 计算不同周期的动量
        momentum_periods = [1, 6, 12, 36]  # 月数
        trading_days_per_month = 21  # 每月交易天数
        
        for months in momentum_periods:
            days = months * trading_days_per_month
            momentum_col = f'mom{months}m'
            
            # 计算滚动收益率
            def calculate_momentum(group):
                if len(group) < days + 1:
                    return pd.Series([np.nan], [momentum_col])
                
                # 计算days天前的价格
                past_price = group['close'].iloc[-days-1]
                current_price = group['close'].iloc[-1]
                
                # 计算动量
                momentum = (current_price - past_price) / past_price
                
                return pd.Series([momentum], [momentum_col])
            
            momentum = price_df.groupby('stock_code').apply(calculate_momentum).reset_index()
            
            # 合并到结果
            result_df = pd.merge(result_df, momentum, on='stock_code', how='left')
        
        # 计算价格动量 (与mom12m相同)
        if 'mom12m' in result_df.columns:
            result_df['price_momentum'] = result_df['mom12m']
        
        # 计算最大收益率
        def calculate_max_return(group):
            if len(group) < 21:  # 至少一个月的数据
                return pd.Series([np.nan], ['maxret'])
            
            # 计算最近一个月的每日收益率
            recent_returns = group['return'].tail(21)
            
            # 最大收益率
            max_return = recent_returns.max()
            
            return pd.Series([max_return], ['maxret'])
        
        max_ret = price_df.groupby('stock_code').apply(calculate_max_return).reset_index()
        
        # 合并到结果
        result_df = pd.merge(result_df, max_ret, on='stock_code', how='left')
        
        # 计算行业动量
        # 由于缺少行业分类数据，这里暂时设为0
        result_df['indmom'] = 0
        
        # 计算季节性
        # 由于缺少足够长的历史数据，这里暂时设为0
        result_df['seasonality'] = 0
        
        return result_df
    
    def _calculate_liquidity_factors(self, result_df, price_df, latest_financial=None):
        """计算流动性因子"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算换手率
        # 首先检查是否有'SHARE_CAPITAL'列，如果没有，尝试从财务数据获取
        if 'volume' in price_df.columns:
            # 如果price_df中没有SHARE_CAPITAL列，先获取最新财务数据
            if 'SHARE_CAPITAL' not in price_df.columns:
                # 使用传入的latest_financial参数
                if latest_financial is not None and 'SHARE_CAPITAL' in latest_financial.columns:
                    # 合并股本数据到价格数据
                    price_df = pd.merge(
                        price_df, 
                        latest_financial[['stock_code', 'SHARE_CAPITAL']], 
                        on='stock_code', 
                        how='left'
                    )
            
            # 计算换手率
            if 'SHARE_CAPITAL' in price_df.columns:
                price_df['turnover'] = self.safe_divide(price_df['volume'], price_df['SHARE_CAPITAL'])
            else:
                # 如果没有股本数据，使用成交额/收盘价作为交易量的近似
                if 'amount' in price_df.columns and 'close' in price_df.columns:
                    price_df['turnover'] = self.safe_divide(price_df['amount'], price_df['close'] * 10000)  # 假设股价单位为元，成交额单位为万元
                else:
                    price_df['turnover'] = 0  # 如果没有足够数据，设为0
        else:
            price_df['turnover'] = 0  # 如果没有成交量数据，设为0
        
        # 计算最近一个月的平均换手率
        def calculate_avg_turnover(group):
            if len(group) < 21:  # 至少一个月的数据
                return pd.DataFrame({'turnover': [np.nan], 'stock_code': [group.name]})
            
            # 计算最近一个月的平均换手率
            recent_turnover = group['turnover'].tail(21)
            avg_turnover = recent_turnover.mean()
            
            return pd.DataFrame({'turnover': [avg_turnover], 'stock_code': [group.name]})
        
        turnover = price_df.groupby('stock_code').apply(calculate_avg_turnover).reset_index(drop=True)
        
        # 合并到结果
        result_df = pd.merge(result_df, turnover, on='stock_code', how='left')
        
        # 计算收益率（如果还没有计算）
        if 'return' not in price_df.columns:
            price_df['return'] = price_df.groupby('stock_code')['close'].pct_change()
        
        # 计算非流动性
        # 非流动性 = |收益率| / 成交额
        if 'amount' in price_df.columns:
            price_df['illiquidity_daily'] = self.safe_divide(abs(price_df['return']), price_df['amount'])
        else:
            price_df['illiquidity_daily'] = 0  # 如果没有成交额数据，设为0
        
        # 计算最近一个月的平均非流动性
        def calculate_avg_illiquidity(group):
            if len(group) < 21:  # 至少一个月的数据
                return pd.DataFrame({'illiquidity': [np.nan], 'stock_code': [group.name]})
            
            # 计算最近一个月的平均非流动性
            recent_illiquidity = group['illiquidity_daily'].tail(21)
            avg_illiquidity = recent_illiquidity.mean()
            
            return pd.DataFrame({'illiquidity': [avg_illiquidity], 'stock_code': [group.name]})
        
        illiquidity = price_df.groupby('stock_code').apply(calculate_avg_illiquidity).reset_index(drop=True)
        
        # 合并到结果
        result_df = pd.merge(result_df, illiquidity, on='stock_code', how='left')
        
        # 计算成交额波动率
        if 'amount' in price_df.columns:
            # 计算成交额的对数
            price_df['log_amount'] = np.log(price_df['amount'])
            
            # 计算成交额对数的日变化
            price_df['log_amount_change'] = price_df.groupby('stock_code')['log_amount'].diff()
            
            # 计算最近一个月的成交额波动率
            def calculate_amount_volatility(group):
                if len(group) < 21:  # 至少一个月的数据
                    return pd.DataFrame({'dolvol': [np.nan], 'stock_code': [group.name]})
                
                # 计算最近一个月的成交额变化
                recent_changes = group['log_amount_change'].tail(21).dropna()
                
                if len(recent_changes) < 5:
                    return pd.DataFrame({'dolvol': [np.nan], 'stock_code': [group.name]})
                
                # 计算波动率
                volatility = recent_changes.std()
                
                return pd.DataFrame({'dolvol': [volatility], 'stock_code': [group.name]})
            
            dolvol = price_df.groupby('stock_code').apply(calculate_amount_volatility).reset_index(drop=True)
            
            # 合并到结果
            result_df = pd.merge(result_df, dolvol, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_other_factors(self, result_df, price_df):
        """计算其他因子"""
        # 计算成长性
        # 由于缺少历史数据，这里暂时设为0
        result_df['growth'] = 0
        
        # 计算波动率
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算收益率
        price_df['return'] = price_df.groupby('stock_code')['close'].pct_change()
        
        # 计算最近一个月的波动率
        def calculate_volatility(group):
            if len(group) < 21:  # 至少一个月的数据
                return pd.Series([np.nan], ['volatility'])
            
            # 计算最近一个月的收益率
            recent_returns = group['return'].tail(21).dropna()
            
            if len(recent_returns) < 5:
                return pd.Series([np.nan], ['volatility'])
            
            # 计算波动率
            volatility = recent_returns.std()
            
            return pd.Series([volatility], ['volatility'])
        
        volatility = price_df.groupby('stock_code').apply(calculate_volatility).reset_index()
        
        # 合并到结果
        result_df = pd.merge(result_df, volatility, on='stock_code', how='left')
        
        return result_df