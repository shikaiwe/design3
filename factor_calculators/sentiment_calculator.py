#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情绪类因子计算器
计算情绪类因子，共30个因子
"""

import pandas as pd
import numpy as np
from base_calculator import BaseFactorCalculator

class SentimentCalculator(BaseFactorCalculator):
    """情绪类因子计算器"""
    
    def __init__(self):
        """初始化情绪类因子计算器"""
        super().__init__()
        self.factor_names = [
            'MA5', 'MA10', 'MA20', 'MA60', 'MA120',     # 移动平均线
            'STD5', 'STD20', 'STD60',                    # 标准差
            'ATR6', 'ATR14',                            # 平均真实波幅
            'BOLL_UPPER', 'BOLL_LOWER',                  # 布林带
            'RSI6', 'RSI12',                            # 相对强弱指数
            'MACD', 'MACD_SIGNAL', 'MACD_HIST',         # MACD指标
            'MFI',                                      # 资金流量指标
            'CCI5', 'CCI20', 'CCI60',                   # 商品通道指数
            'MOM',                                      # 动量指标
            'MTM', 'MTM6', 'MTMMA',                     # 变化率指标
            'CMO',                                      # 钱德动量摆动指标
            'DPO',                                      # 区间震荡线
            'EMV',                                      # 简易波动指标
            'VOLUME5', 'VOLUME20', 'VOLUME60',          # 成交量均值
            'INDEX_RETURN'                              # 指数收益率
        ]
    
    def calculate(self, financial_data, price_data, index_data, financial_indicator_data=None, industry_data=None):
        """
        计算情绪类因子
        
        参数:
            financial_data: 财务数据DataFrame
            price_data: 价格数据DataFrame
            index_data: 指数数据DataFrame
            financial_indicator_data: 财务指标数据DataFrame (可选)
            industry_data: 行业分类数据DataFrame (可选)
            
        返回:
            情绪类因子DataFrame
        """
        # 准备数据
        price_df = self._prepare_price_data(price_data)
        index_df = self._prepare_index_data(index_data)
        
        # 获取最新日期的数据
        latest_date = price_df['date'].max()
        latest_prices = price_df[price_df['date'] == latest_date]
        
        # 初始化结果DataFrame
        result = latest_prices[['stock_code', 'date']].copy()
        
        # 计算情绪类因子
        result = self._calculate_moving_averages(result, price_df)
        result = self._calculate_standard_deviation(result, price_df)
        result = self._calculate_atr(result, price_df)
        result = self._calculate_bollinger_bands(result, price_df)
        result = self._calculate_rsi(result, price_df)
        result = self._calculate_macd(result, price_df)
        result = self._calculate_mfi(result, price_df)
        result = self._calculate_cci(result, price_df)
        result = self._calculate_momentum(result, price_df)
        result = self._calculate_cmo(result, price_df)
        result = self._calculate_dpo(result, price_df)
        result = self._calculate_emv(result, price_df)
        result = self._calculate_volume_averages(result, price_df)
        result = self._calculate_index_returns(result, index_df)
        
        return result
    
    def _prepare_price_data(self, price_data):
        """准备价格数据，确保列名正确"""
        # 创建价格数据的副本
        price_df = price_data.copy()
        
        # 确保日期列是datetime类型
        if 'date' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['date'])
        
        return price_df
    
    def _prepare_index_data(self, index_data):
        """准备指数数据，确保列名正确"""
        # 创建指数数据的副本
        index_df = index_data.copy()
        
        # 确保日期列是datetime类型
        if 'date' in index_df.columns:
            index_df['date'] = pd.to_datetime(index_df['date'])
        
        return index_df
    
    def _calculate_moving_averages(self, result_df, price_df):
        """计算移动平均线"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算不同周期的移动平均线
        for period in [5, 10, 20, 60, 120]:
            ma_col = f'MA{period}'
            ma = price_df.groupby('stock_code')['close'].rolling(window=period).mean().reset_index()
            ma = ma.rename(columns={'close': ma_col})
            
            # 获取每只股票最新日期的MA值
            latest_ma = ma.groupby('stock_code').last().reset_index()
            latest_ma = latest_ma[['stock_code', ma_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_ma, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_standard_deviation(self, result_df, price_df):
        """计算标准差"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算不同周期的标准差
        for period in [5, 20, 60]:
            std_col = f'STD{period}'
            std = price_df.groupby('stock_code')['close'].rolling(window=period).std().reset_index()
            std = std.rename(columns={'close': std_col})
            
            # 获取每只股票最新日期的STD值
            latest_std = std.groupby('stock_code').last().reset_index()
            latest_std = latest_std[['stock_code', std_col]]
            
            # 合并到结果
            result_df = pd.merge(result_df, latest_std, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_atr(self, result_df, price_df):
        """计算平均真实波幅"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算真实波幅(TR)
        price_df['prev_close'] = price_df.groupby('stock_code')['close'].shift(1)
        price_df['tr1'] = price_df['high'] - price_df['low']
        price_df['tr2'] = abs(price_df['high'] - price_df['prev_close'])
        price_df['tr3'] = abs(price_df['low'] - price_df['prev_close'])
        price_df['tr'] = price_df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算ATR
        for period in [6, 14]:
            atr_col = f'ATR{period}'
            # 使用transform确保索引对齐
            price_df[atr_col] = price_df.groupby('stock_code')['tr'].transform(lambda x: x.rolling(window=period).mean())
        
        # 获取最新日期的ATR值
        latest_atr = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'ATR6', 'ATR14']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_atr, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_bollinger_bands(self, result_df, price_df):
        """计算布林带"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算20日移动平均线和标准差
        price_df['MA20'] = price_df.groupby('stock_code')['close'].transform(lambda x: x.rolling(window=20).mean())
        price_df['STD20'] = price_df.groupby('stock_code')['close'].transform(lambda x: x.rolling(window=20).std())
        
        # 计算布林带上下轨
        price_df['BOLL_UPPER'] = price_df['MA20'] + 2 * price_df['STD20']
        price_df['BOLL_LOWER'] = price_df['MA20'] - 2 * price_df['STD20']
        
        # 获取最新日期的布林带值
        latest_boll = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'BOLL_UPPER', 'BOLL_LOWER']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_boll, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_rsi(self, result_df, price_df):
        """计算相对强弱指数"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算价格变化
        price_df['price_change'] = price_df.groupby('stock_code')['close'].transform(lambda x: x.diff())
        
        # 计算RSI
        for period in [6, 12]:
            rsi_col = f'RSI{period}'
            
            # 计算涨跌幅
            price_df['gain'] = price_df['price_change'].apply(lambda x: x if x > 0 else 0)
            price_df['loss'] = price_df['price_change'].apply(lambda x: -x if x < 0 else 0)
            
            # 计算平均涨跌幅
            price_df['avg_gain'] = price_df.groupby('stock_code')['gain'].transform(lambda x: x.rolling(window=period).mean())
            price_df['avg_loss'] = price_df.groupby('stock_code')['loss'].transform(lambda x: x.rolling(window=period).mean())
            
            # 计算RSI
            price_df[rsi_col] = 100 - (100 / (1 + price_df['avg_gain'] / price_df['avg_loss']))
        
        # 获取最新日期的RSI值
        latest_rsi = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'RSI6', 'RSI12']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_rsi, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_macd(self, result_df, price_df):
        """计算MACD指标"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算MACD
        def calculate_macd(group):
            # 计算EMA
            ema12 = group['close'].ewm(span=12).mean()
            ema26 = group['close'].ewm(span=26).mean()
            
            # 计算MACD线
            macd_line = ema12 - ema26
            
            # 计算信号线
            signal_line = macd_line.ewm(span=9).mean()
            
            # 计算MACD柱状图
            macd_hist = macd_line - signal_line
            
            return pd.DataFrame({
                'MACD': macd_line,
                'MACD_SIGNAL': signal_line,
                'MACD_HIST': macd_hist
            }, index=group.index)
        
        # 应用MACD计算并保持索引
        macd_data = price_df.groupby('stock_code').apply(calculate_macd).reset_index(level=0, drop=True)
        
        # 合并MACD数据到原始DataFrame
        price_df = pd.concat([price_df, macd_data], axis=1)
        
        # 获取最新日期的MACD值
        latest_macd = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'MACD', 'MACD_SIGNAL', 'MACD_HIST']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_macd, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_mfi(self, result_df, price_df):
        """计算资金流量指标"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算典型价格
        price_df['typical_price'] = (price_df['high'] + price_df['low'] + price_df['close']) / 3
        
        # 计算资金流量
        price_df['money_flow'] = price_df['typical_price'] * price_df['volume']
        
        # 计算MFI
        def calculate_mfi_group(group):
            # 计算正负资金流量
            group['positive_mf'] = 0
            group['negative_mf'] = 0
            
            # 比较前一天的典型价格
            group['prev_typical_price'] = group['typical_price'].shift(1)
            
            # 标记正负资金流量
            positive_mask = group['typical_price'] > group['prev_typical_price']
            negative_mask = group['typical_price'] < group['prev_typical_price']
            
            group.loc[positive_mask, 'positive_mf'] = group.loc[positive_mask, 'money_flow']
            group.loc[negative_mask, 'negative_mf'] = group.loc[negative_mask, 'money_flow']
            
            # 计算14日正负资金流量总和
            positive_mf_sum = group['positive_mf'].rolling(window=14).sum()
            negative_mf_sum = group['negative_mf'].rolling(window=14).sum()
            
            # 计算MFI
            mfi = 100 - (100 / (1 + positive_mf_sum / negative_mf_sum))
            
            return mfi
        
        price_df['MFI'] = price_df.groupby('stock_code').apply(calculate_mfi_group).reset_index(level=0, drop=True)
        
        # 获取最新日期的MFI值
        latest_mfi = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'MFI']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_mfi, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_cci(self, result_df, price_df):
        """计算商品通道指数"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算CCI
        for period in [5, 20, 60]:
            cci_col = f'CCI{period}'
            
            # 计算典型价格
            price_df['typical_price'] = (price_df['high'] + price_df['low'] + price_df['close']) / 3
            
            # 计算典型价格的移动平均线
            price_df['tp_ma'] = price_df.groupby('stock_code')['typical_price'].transform(lambda x: x.rolling(window=period).mean())
            
            # 计算平均绝对偏差
            price_df['mad'] = price_df.groupby('stock_code')['typical_price'].transform(lambda x: x.rolling(window=period).apply(lambda y: np.abs(y - y.mean()).mean()))
            
            # 计算CCI
            price_df[cci_col] = (price_df['typical_price'] - price_df['tp_ma']) / (0.015 * price_df['mad'])
        
        # 获取最新日期的CCI值
        latest_cci = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'CCI5', 'CCI20', 'CCI60']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_cci, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_momentum(self, result_df, price_df):
        """计算动量指标"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算动量
        price_df['MOM'] = price_df.groupby('stock_code')['close'].transform(lambda x: x.diff())
        
        # 获取最新日期的MOM值
        latest_mom = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'MOM']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_mom, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_cmo(self, result_df, price_df):
        """计算钱德动量摆动指标"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算价格变化
        price_df['price_change'] = price_df.groupby('stock_code')['close'].transform(lambda x: x.diff())
        
        # 计算涨跌幅
        price_df['gain'] = price_df['price_change'].apply(lambda x: x if x > 0 else 0)
        price_df['loss'] = price_df['price_change'].apply(lambda x: -x if x < 0 else 0)
        
        # 计算累计涨跌幅
        price_df['cum_gain'] = price_df.groupby('stock_code')['gain'].transform(lambda x: x.rolling(window=14).sum())
        price_df['cum_loss'] = price_df.groupby('stock_code')['loss'].transform(lambda x: x.rolling(window=14).sum())
        
        # 计算CMO
        price_df['CMO'] = 100 * (price_df['cum_gain'] - price_df['cum_loss']) / (price_df['cum_gain'] + price_df['cum_loss'])
        
        # 获取最新日期的CMO值
        latest_cmo = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'CMO']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_cmo, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_dpo(self, result_df, price_df):
        """计算区间震荡线"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算DPO
        period = 20
        displacement = period // 2 + 1
        
        # 计算移动平均线
        price_df['ma'] = price_df.groupby('stock_code')['close'].transform(lambda x: x.rolling(window=period).mean())
        
        # 计算DPO
        price_df['DPO'] = price_df['close'] - price_df['ma'].shift(displacement)
        
        # 获取最新日期的DPO值
        latest_dpo = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'DPO']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_dpo, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_emv(self, result_df, price_df):
        """计算简易波动指标"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算距离
        price_df['distance'] = (price_df['high'] + price_df['low']) / 2 - (price_df.groupby('stock_code')['high'].shift(1) + price_df.groupby('stock_code')['low'].shift(1)) / 2
        
        # 计算移动值
        price_df['move'] = price_df['distance'] / price_df['volume']
        
        # 计算EMV
        price_df['EMV'] = price_df.groupby('stock_code')['move'].transform(lambda x: x.rolling(window=14).sum())
        
        # 获取最新日期的EMV值
        latest_emv = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'EMV']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_emv, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_volume_averages(self, result_df, price_df):
        """计算成交量均值"""
        # 按股票代码和日期排序
        price_df = price_df.sort_values(['stock_code', 'date'])
        
        # 计算不同周期的成交量均值
        for period in [5, 20, 60]:
            vol_col = f'VOLUME{period}'
            
            # 计算每个股票的成交量均值
            price_df[vol_col] = price_df.groupby('stock_code')['volume'].transform(lambda x: x.rolling(window=period).mean())
        
        # 获取最新日期的成交量均值
        latest_vol = price_df[price_df['date'] == price_df['date'].max()][['stock_code', 'VOLUME5', 'VOLUME20', 'VOLUME60']]
        
        # 合并到结果
        result_df = pd.merge(result_df, latest_vol, on='stock_code', how='left')
        
        return result_df
    
    def _calculate_index_returns(self, result_df, index_df):
        """计算指数收益率"""
        # 按日期排序
        index_df = index_df.sort_values('date')
        
        # 计算指数收益率
        index_df['INDEX_RETURN'] = index_df['close'].pct_change()
        
        # 获取最新日期的指数收益率
        latest_return = index_df.tail(1)[['INDEX_RETURN']]
        
        # 将指数收益率应用到所有股票
        if len(latest_return) > 0 and not pd.isna(latest_return['INDEX_RETURN'].values[0]):
            result_df['INDEX_RETURN'] = latest_return['INDEX_RETURN'].values[0]
        else:
            result_df['INDEX_RETURN'] = np.nan
        
        return result_df