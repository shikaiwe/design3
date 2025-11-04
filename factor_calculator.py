import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class FactorCalculator:
    def __init__(self, price_data_path):
        """
        初始化因子计算器
        
        参数:
        price_data_path: 价格数据文件路径
        """
        self.price_data_path = price_data_path
        self.price_data = None
        self.factor_data = None
        
    def load_price_data(self):
        """加载价格数据"""
        try:
            self.price_data = pd.read_csv(self.price_data_path)
            # 确保日期格式正确
            self.price_data['日期'] = pd.to_datetime(self.price_data['日期'])
            # 按股票代码和日期排序
            self.price_data = self.price_data.sort_values(['stock_code', '日期'])
            print(f"成功加载价格数据，共{len(self.price_data)}行")
            return True
        except Exception as e:
            print(f"加载价格数据失败: {e}")
            return False
    
    def calculate_momentum_factors(self):
        """计算动量类因子"""
        print("计算动量类因子...")
        
        # 按股票代码分组计算
        for stock_code, group in self.price_data.groupby('stock_code'):
            group = group.sort_values('日期').reset_index(drop=True)
            
            # 计算ROC (Rate of Change) - 变动速率
            for period in [6, 12, 20, 60, 120]:
                if len(group) > period:
                    group[f'ROC{period}'] = (group['收盘'] / group['收盘'].shift(period) - 1) * 100
            
            # 计算BIAS (乖离率)
            for period in [5, 10, 20, 60]:
                if len(group) > period:
                    ma = group['收盘'].rolling(window=period).mean()
                    group[f'BIAS{period}'] = (group['收盘'] - ma) / ma * 100
            
            # 计算CCI (顺势指标)
            for period in [10, 15, 20, 88]:
                if len(group) > period:
                    tp = (group['最高'] + group['最低'] + group['收盘']) / 3
                    ma_tp = tp.rolling(window=period).mean()
                    md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
                    group[f'CCI{period}'] = (tp - ma_tp) / (0.015 * md)
            
            # 计算CR指标
            if len(group) > 20:
                group['CR20'] = self.calculate_cr(group, 20)
            
            # 计算TRIX
            for period in [5, 10]:
                if len(group) > period:
                    group[f'TRIX{period}'] = self.calculate_trix(group['收盘'], period)
            
            # 计算价格相对位置
            if len(group) > 252:  # 一年约252个交易日
                group['fifty_two_week_close_rank'] = group['收盘'].rolling(window=252).rank(pct=True)
            
            # 计算PLRC (价格线性回归系数)
            for period in [6, 12, 24]:
                if len(group) > period:
                    group[f'PLRC{period}'] = self.calculate_plrc(group['收盘'], period)
            
            # 计算价格相对均值
            if len(group) > 21:  # 一个月约21个交易日
                group['Price1M'] = group['收盘'] / group['收盘'].rolling(window=21).mean() - 1
            
            if len(group) > 63:  # 三个月约63个交易日
                group['Price3M'] = group['收盘'] / group['收盘'].rolling(window=63).mean() - 1
            
            if len(group) > 252:  # 一年约252个交易日
                group['Price1Y'] = group['收盘'] / group['收盘'].rolling(window=252).mean() - 1
            
            # 计算排名因子
            if len(group) > 21:
                returns_1m = group['收盘'].pct_change(periods=21)
                group['Rank1M'] = 1 - returns_1m.rolling(window=21).rank(pct=True)
            
            # 计算Aroon指标
            if len(group) > 25:
                group['arron_up_25'], group['arron_down_25'] = self.calculate_aroon(group, 25)
            
            # 计算多空力道
            if len(group) > 13:
                ema13 = group['收盘'].ewm(span=13).mean()
                group['bull_power'] = group['最高'] - ema13
                group['bear_power'] = group['最低'] - ema13
            
            # 计算BBI动量
            if len(group) > 20:
                ma3 = group['收盘'].rolling(window=3).mean()
                ma6 = group['收盘'].rolling(window=6).mean()
                ma12 = group['收盘'].rolling(window=12).mean()
                ma24 = group['收盘'].rolling(window=24).mean()
                group['BBIC'] = (ma3 + ma6 + ma12 + ma24) / 4
            
            # 计算MASS指标
            if len(group) > 25:
                group['MASS'] = self.calculate_mass(group, 9, 25)
            
            # 计算单日VPT
            if len(group) > 1:
                group['single_day_VPT'] = (group['收盘'] - group['收盘'].shift(1)) / group['收盘'].shift(1) * group['成交量']
                group['single_day_VPT_6'] = group['single_day_VPT'].rolling(window=6).mean()
                group['single_day_VPT_12'] = group['single_day_VPT'].rolling(window=12).mean()
            
            # 计算Volume1M
            if len(group) > 21:
                vol_mean_1m = group['成交量'].rolling(window=21).mean()
                returns_mean_20 = group['收盘'].pct_change().rolling(window=20).mean()
                group['Volume1M'] = (group['成交量'] / vol_mean_1m) * returns_mean_20
            
            # 更新到主数据框
            for col in group.columns:
                if col not in self.price_data.columns:
                    if col not in ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']:
                        self.price_data.loc[self.price_data['stock_code'] == stock_code, col] = group[col].values
        
        print("动量类因子计算完成")
        
        # 保存动量类因子
        self.save_momentum_factors()
        
        return self.price_data
    
    def save_momentum_factors(self):
        """保存动量类因子到单独的文件夹"""
        if self.price_data is None:
            print("没有可保存的因子数据")
            return False
        
        try:
            # 创建输出目录
            base_dir = "c:\\Users\\Administrator\\Desktop\\design3\\data\\factories"
            
            # 识别动量类因子列
            momentum_cols = []
            base_cols = ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            
            # 分类动量因子列
            for col in self.price_data.columns:
                if col not in base_cols:
                    # 动量类因子
                    if col.startswith('ROC') or col.startswith('BIAS') or col.startswith('CCI') or col == 'CR20' or \
                       col.startswith('TRIX') or col == 'fifty_two_week_close_rank' or col.startswith('PLRC') or \
                       col.startswith('Price') or col.startswith('Rank') or col.startswith('arron') or \
                       col in ['bull_power', 'bear_power', 'BBIC', 'MASS', 'single_day_VPT', 'single_day_VPT_6', 
                               'single_day_VPT_12', 'Volume1M']:
                        momentum_cols.append(col)
            
            # 保存动量类因子
            if momentum_cols:
                momentum_data = self.price_data[base_cols + momentum_cols]
                momentum_path = os.path.join(base_dir, "momentum", "momentum_factors.csv")
                momentum_data.to_csv(momentum_path, index=False)
                print(f"动量类因子已保存到: {momentum_path}")
                return True
            else:
                print("没有找到动量类因子")
                return False
                
        except Exception as e:
            print(f"保存动量类因子失败: {e}")
            return False

    def calculate_emotion_factors(self):
        """计算情绪类因子"""
        print("计算情绪类因子...")
        
        # 按股票代码分组计算
        for stock_code, group in self.price_data.groupby('stock_code'):
            group = group.sort_values('日期').reset_index(drop=True)
            
            # 计算AR指标
            if len(group) > 26:
                group['AR'] = self.calculate_ar(group, 26)
            
            # 计算BR指标
            if len(group) > 26:
                group['BR'] = self.calculate_br(group, 26)
            
            # 计算ARBR
            if 'AR' in group.columns and 'BR' in group.columns:
                group['ARBR'] = group['AR'] + group['BR']
            
            # 计算ATR (Average True Range)
            for period in [6, 14]:
                if len(group) > period:
                    group[f'ATR{period}'] = self.calculate_atr(group, period)
            
            # 计算换手率相关指标
            for period in [5, 10, 20, 60, 120, 240]:
                if len(group) > period:
                    group[f'VOL{period}'] = group['换手率'].rolling(window=period).mean()
            
            # 计算换手率相对波动率
            if len(group) > 120:
                vol_mean_120 = group['换手率'].rolling(window=120).mean()
                vol_std_120 = group['换手率'].rolling(window=120).std()
                group['turnover_volatility'] = vol_std_120 / vol_mean_120
            
            # 计算DAVOL (换手率相对比率)
            for period in [5, 10, 20]:
                if len(group) > 120:
                    vol_mean_period = group['换手率'].rolling(window=period).mean()
                    vol_mean_120 = group['换手率'].rolling(window=120).mean()
                    group[f'DAVOL{period}'] = vol_mean_period / vol_mean_120
            
            # 计算成交金额相关指标
            for period in [6, 20]:
                if len(group) > period:
                    group[f'TVMA{period}'] = group['成交额'].rolling(window=period).mean()
                    group[f'TVSTD{period}'] = group['成交额'].rolling(window=period).std()
            
            # 计算成交量移动平均
            for period in [5, 10, 12, 26]:
                if len(group) > period:
                    group[f'VEMA{period}'] = group['成交量'].ewm(span=period).mean()
            
            # 计算VMACD指标
            if len(group) > 26:
                vema12 = group['成交量'].ewm(span=12).mean()
                vema26 = group['成交量'].ewm(span=26).mean()
                group['VDIFF'] = vema12 - vema26
                group['VDEA'] = group['VDIFF'].ewm(span=9).mean()
                group['VMACD'] = 2 * (group['VDIFF'] - group['VDEA'])
            
            # 计算VOSC (成交量震荡)
            if len(group) > 10:
                vema10 = group['成交量'].ewm(span=10).mean()
                vema20 = group['成交量'].ewm(span=20).mean()
                group['VOSC'] = (vema10 - vema20) / vema20 * 100
            
            # 计算VR (成交量比率)
            if len(group) > 26:
                group['VR'] = self.calculate_vr(group, 26)
            
            # 计算VROC (成交量变动速率)
            for period in [6, 12]:
                if len(group) > period:
                    group[f'VROC{period}'] = (group['成交量'] / group['成交量'].shift(period) - 1) * 100
            
            # 计算成交量标准差
            for period in [10, 20]:
                if len(group) > period:
                    group[f'VSTD{period}'] = group['成交量'].rolling(window=period).std()
            
            # 计算WVAD (威廉变异离散量)
            if len(group) > 6:
                group['WVAD'] = self.calculate_wvad(group)
                group['MAWVAD'] = group['WVAD'].rolling(window=6).mean()
            
            # 计算PSY (心理线指标)
            if len(group) > 12:
                group['PSY'] = self.calculate_psy(group, 12)
            
            # 计算资金流量指标
            if len(group) > 20:
                group['money_flow_20'] = self.calculate_money_flow(group, 20)
            
            # 更新到主数据框
            for col in group.columns:
                if col not in self.price_data.columns:
                    if col not in ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']:
                        self.price_data.loc[self.price_data['stock_code'] == stock_code, col] = group[col].values
        
        print("情绪类因子计算完成")
        self.save_emotion_factors()
        return self.price_data
    
    def save_emotion_factors(self):
        """保存情绪类因子到单独的文件夹"""
        if self.price_data is None:
            print("没有可保存的因子数据")
            return False
        
        try:
            # 创建输出目录
            base_dir = "c:\\Users\\Administrator\\Desktop\\design3\\data\\factories"
            
            # 识别情绪类因子列
            emotion_cols = []
            base_cols = ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            
            # 分类情绪因子列
            for col in self.price_data.columns:
                if col not in base_cols:
                    # 情绪类因子
                    if col.startswith('PSY') or col.startswith('VR') or col.startswith('WVAD') or \
                       col.startswith('MFI') or col.startswith('Money') or col.startswith('BR') or \
                       col.startswith('AR') or col.startswith('ATR'):
                        emotion_cols.append(col)
            
            # 保存情绪类因子
            if emotion_cols:
                emotion_data = self.price_data[base_cols + emotion_cols]
                emotion_path = os.path.join(base_dir, "emotion", "emotion_factors.csv")
                emotion_data.to_csv(emotion_path, index=False)
                print(f"情绪类因子已保存到: {emotion_path}")
                return True
            else:
                print("没有找到情绪类因子")
                return False
                
        except Exception as e:
            print(f"保存情绪类因子失败: {e}")
            return False

    def calculate_risk_factors(self):
        """计算风险类因子"""
        print("计算风险类因子...")
        
        # 按股票代码分组计算
        for stock_code, group in self.price_data.groupby('stock_code'):
            group = group.sort_values('日期').reset_index(drop=True)
            
            # 计算日收益率
            group['daily_return'] = group['收盘'].pct_change()
            
            # 计算方差
            for period in [20, 60, 120]:
                if len(group) > period:
                    group[f'Variance{period}'] = group['daily_return'].rolling(window=period).var()
            
            # 计算偏度
            for period in [20, 60, 120]:
                if len(group) > period:
                    group[f'Skewness{period}'] = group['daily_return'].rolling(window=period).skew()
            
            # 计算峰度
            for period in [20, 60, 120]:
                if len(group) > period:
                    group[f'Kurtosis{period}'] = group['daily_return'].rolling(window=period).kurt()
            
            # 计算夏普比率 (假设无风险利率为0)
            for period in [20, 60, 120]:
                if len(group) > period:
                    mean_return = group['daily_return'].rolling(window=period).mean()
                    std_return = group['daily_return'].rolling(window=period).std()
                    group[f'sharpe_ratio_{period}'] = mean_return / std_return
            
            # 更新到主数据框
            for col in group.columns:
                if col not in self.price_data.columns:
                    if col not in ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']:
                        self.price_data.loc[self.price_data['stock_code'] == stock_code, col] = group[col].values
        
        print("风险类因子计算完成")
        self.save_risk_factors()
        return self.price_data
    
    def save_risk_factors(self):
        """保存风险类因子到单独的文件夹"""
        if self.price_data is None:
            print("没有可保存的因子数据")
            return False
        
        try:
            # 创建输出目录
            base_dir = "c:\\Users\\Administrator\\Desktop\\design3\\data\\factories"
            
            # 识别风险类因子列
            risk_cols = []
            base_cols = ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            
            # 分类风险因子列
            for col in self.price_data.columns:
                if col not in base_cols:
                    # 风险类因子
                    if col.startswith('Variance') or col.startswith('Skewness') or col.startswith('Kurtosis') or \
                       col.startswith('Sharpe') or col.startswith('MaxDrawdown') or col.startswith('Volatility') or \
                       col.startswith('Downside') or col.startswith('Beta'):
                        risk_cols.append(col)
            
            # 保存风险类因子
            if risk_cols:
                risk_data = self.price_data[base_cols + risk_cols]
                risk_path = os.path.join(base_dir, "risk", "risk_factors.csv")
                risk_data.to_csv(risk_path, index=False)
                print(f"风险类因子已保存到: {risk_path}")
                return True
            else:
                print("没有找到风险类因子")
                return False
                
        except Exception as e:
            print(f"保存风险类因子失败: {e}")
            return False

    def calculate_technical_factors(self):
        """计算技术指标因子"""
        print("计算技术指标因子...")
        
        # 按股票代码分组计算
        for stock_code, group in self.price_data.groupby('stock_code'):
            group = group.sort_values('日期').reset_index(drop=True)
            
            # 计算移动平均线
            for period in [5, 10, 20, 60, 120]:
                if len(group) > period:
                    group[f'MAC{period}'] = group['收盘'].rolling(window=period).mean()
            
            # 计算指数移动平均线
            for period in [5, 10, 12, 20, 26, 120]:
                if len(group) > period:
                    group[f'EMAC{period}'] = group['收盘'].ewm(span=period).mean()
            
            # 计算布林线
            if len(group) > 20:
                ma20 = group['收盘'].rolling(window=20).mean()
                std20 = group['收盘'].rolling(window=20).std()
                group['boll_up'] = ma20 + 2 * std20
                group['boll_down'] = ma20 - 2 * std20
            
            # 计算MACD
            if len(group) > 26:
                ema12 = group['收盘'].ewm(span=12).mean()
                ema26 = group['收盘'].ewm(span=26).mean()
                dif = ema12 - ema26
                dea = dif.ewm(span=9).mean()
                group['MACDC'] = 2 * (dif - dea)
            
            # 计算MFI (资金流量指标)
            if len(group) > 14:
                group['MFI14'] = self.calculate_mfi(group, 14)
            
            # 计算不复权价格因子
            group['price_no_fq'] = group['收盘']
            
            # 更新到主数据框
            for col in group.columns:
                if col not in self.price_data.columns:
                    if col not in ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']:
                        self.price_data.loc[self.price_data['stock_code'] == stock_code, col] = group[col].values
        
        print("技术指标因子计算完成")
        self.save_technical_factors()
        return self.price_data
    
    # 辅助计算函数
    def save_technical_factors(self):
        """保存技术类因子到单独的文件夹"""
        if self.price_data is None:
            print("没有可保存的因子数据")
            return False
        
        try:
            # 创建输出目录
            base_dir = "c:\\Users\\Administrator\\Desktop\\design3\\data\\factories"
            
            # 识别技术类因子列
            technical_cols = []
            base_cols = ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            
            # 分类技术因子列
            for col in self.price_data.columns:
                if col not in base_cols:
                    # 技术类因子
                    if col.startswith('MA') or col.startswith('EMA') or col.startswith('BOLL') or \
                       col.startswith('MACD') or col.startswith('RSI') or col.startswith('KDJ') or \
                       col.startswith('WR') or col.startswith('SAR') or col.startswith('DMI') or \
                       col.startswith('CCI') or col.startswith('OBV') or col.startswith('MTM'):
                        technical_cols.append(col)
            
            # 保存技术类因子
            if technical_cols:
                technical_data = self.price_data[base_cols + technical_cols]
                technical_path = os.path.join(base_dir, "technical", "technical_factors.csv")
                technical_data.to_csv(technical_path, index=False)
                print(f"技术类因子已保存到: {technical_path}")
                return True
            else:
                print("没有找到技术类因子")
                return False
                
        except Exception as e:
            print(f"保存技术类因子失败: {e}")
            return False

    def calculate_cr(self, group, period):
        """计算CR指标"""
        if len(group) < period + 1:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        cr_values = []
        for i in range(period, len(group)):
            # 计算今日的CR值
            hm = group.loc[i-period+1:i, '最高'].max()
            lm = group.loc[i-period+1:i, '最低'].min()
            hly = group.loc[i-period+1:i, '最高'].sum()
            lly = group.loc[i-period+1:i, '最低'].sum()
            
            if hly - lly != 0:
                cr = (hm - lm) / (hly - lly) * 100
            else:
                cr = 0
            
            cr_values.append(cr)
        
        # 前面period-1个值为NaN
        result_length = len(group)
        values = [np.nan] * (period - 1) + cr_values
        # 确保返回值长度与输入数据长度一致
        if len(values) < result_length:
            values = values + [np.nan] * (result_length - len(values))
        elif len(values) > result_length:
            values = values[:result_length]
            
        return pd.Series(values, index=group.index)
    
    def calculate_trix(self, close_prices, period):
        """计算TRIX指标"""
        if len(close_prices) < period * 3:
            return np.nan
        
        # 第一次指数平滑
        ema1 = close_prices.ewm(span=period).mean()
        # 第二次指数平滑
        ema2 = ema1.ewm(span=period).mean()
        # 第三次指数平滑
        ema3 = ema2.ewm(span=period).mean()
        
        # 计算TRIX
        trix = ema3.pct_change() * 100
        
        return trix
    
    def calculate_plrc(self, close_prices, period):
        """计算价格线性回归系数"""
        if len(close_prices) < period:
            return np.nan
        
        plrc_values = []
        for i in range(period, len(close_prices) + 1):
            y = close_prices.iloc[i-period:i].values
            x = np.arange(period)
            
            # 计算线性回归系数
            if len(x) > 1 and len(y) > 1:
                slope, _ = np.polyfit(x, y, 1)
                plrc_values.append(slope)
            else:
                plrc_values.append(np.nan)
        
        # 前面period-1个值为NaN
        return [np.nan] * (period - 1) + plrc_values
    
    def calculate_aroon(self, group, period):
        """计算Aroon指标"""
        if len(group) < period:
            return np.nan, np.nan
        
        up_values = []
        down_values = []
        
        for i in range(period, len(group) + 1):
            # 计算Aroon Up
            high_period = group.loc[i-period:i-1, '最高']
            high_max_idx = high_period.idxmax()
            days_since_high = (i-1) - high_max_idx
            aroon_up = ((period - days_since_high) / period) * 100
            
            # 计算Aroon Down
            low_period = group.loc[i-period:i-1, '最低']
            low_min_idx = low_period.idxmin()
            days_since_low = (i-1) - low_min_idx
            aroon_down = ((period - days_since_low) / period) * 100
            
            up_values.append(aroon_up)
            down_values.append(aroon_down)
        
        # 前面period-1个值为NaN
        return [np.nan] * (period - 1) + up_values, [np.nan] * (period - 1) + down_values
    
    def calculate_mass(self, group, ema_period, sum_period):
        """计算MASS指标"""
        if len(group) < sum_period:
            return np.nan
        
        # 计算高低价差
        high_low_diff = group['最高'] - group['最低']
        
        # 计算指数移动平均
        ema = high_low_diff.ewm(span=ema_period).mean()
        
        # 计算比率
        ratio = ema / ema.shift(1)
        
        # 计算移动总和
        mass = ratio.rolling(window=sum_period).sum()
        
        return mass
    
    def calculate_ar(self, group, period):
        """计算AR指标"""
        if len(group) < period + 1:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        ar_values = []
        for i in range(period, len(group)):
            # 计算AR值
            ho = group.loc[i-period+1:i, '开盘'].sum()
            lo = group.loc[i-period+1:i, '最低'].sum()
            hc = group.loc[i-period+1:i, '最高'].sum()
            lc = group.loc[i-period+1:i, '最低'].sum()
            
            if ho - lo != 0:
                ar = (hc - lo) / (ho - lo) * 100
            else:
                ar = 0
            
            ar_values.append(ar)
        
        # 前面period-1个值为NaN
        result_length = len(group)
        values = [np.nan] * (period - 1) + ar_values
        # 确保返回值长度与输入数据长度一致
        if len(values) < result_length:
            values = values + [np.nan] * (result_length - len(values))
        elif len(values) > result_length:
            values = values[:result_length]
            
        return pd.Series(values, index=group.index)
    
    def calculate_br(self, group, period):
        """计算BR指标"""
        if len(group) < period + 1:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        br_values = []
        for i in range(period, len(group)):
            # 计算BR值
            hc = group.loc[i-period+1:i, '最高'].sum()
            lc = group.loc[i-period+1:i, '最低'].sum()
            hyc = group.loc[i-period:i-1, '最高'].sum()
            lyc = group.loc[i-period:i-1, '最低'].sum()
            
            if hyc - lyc != 0:
                br = (hc - lyc) / (hyc - lyc) * 100
            else:
                br = 0
            
            br_values.append(br)
        
        # 前面period-1个值为NaN
        result_length = len(group)
        values = [np.nan] * (period - 1) + br_values
        # 确保返回值长度与输入数据长度一致
        if len(values) < result_length:
            values = values + [np.nan] * (result_length - len(values))
        elif len(values) > result_length:
            values = values[:result_length]
            
        return pd.Series(values, index=group.index)
    
    def calculate_atr(self, group, period):
        """计算ATR指标"""
        if len(group) < period + 1:
            return np.nan
        
        # 计算真实范围
        tr1 = group['最高'] - group['最低']
        tr2 = abs(group['最高'] - group['收盘'].shift(1))
        tr3 = abs(group['最低'] - group['收盘'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_vr(self, group, period):
        """计算VR指标"""
        if len(group) < period + 1:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        vr_values = []
        for i in range(period, len(group)):
            # 计算VR值
            av = group.loc[i-period+1:i, '成交量'][group.loc[i-period+1:i, '收盘'] > group.loc[i-period+1:i, '开盘']].sum()
            bv = group.loc[i-period+1:i, '成交量'][group.loc[i-period+1:i, '收盘'] < group.loc[i-period+1:i, '开盘']].sum()
            cv = group.loc[i-period+1:i, '成交量'][group.loc[i-period+1:i, '收盘'] == group.loc[i-period+1:i, '开盘']].sum()
            
            if bv + cv/2 != 0:
                vr = (av + cv/2) / (bv + cv/2) * 100
            else:
                vr = 0
            
            vr_values.append(vr)
        
        # 前面period-1个值为NaN
        result_length = len(group)
        values = [np.nan] * (period - 1) + vr_values
        # 确保返回值长度与输入数据长度一致
        if len(values) < result_length:
            values = values + [np.nan] * (result_length - len(values))
        elif len(values) > result_length:
            values = values[:result_length]
            
        return pd.Series(values, index=group.index)
    
    def calculate_wvad(self, group):
        """计算WVAD指标"""
        if len(group) < 2:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        wvad_values = []
        
        for i in range(1, len(group)):
            # 计算WVAD值 - 修复除零错误
            high_low_diff = group.loc[i, '最高'] - group.loc[i, '最低']
            if high_low_diff != 0:
                price_change = (group.loc[i, '收盘'] - group.loc[i, '开盘']) / high_low_diff
            else:
                price_change = 0
            
            wvad = price_change * group.loc[i, '成交量']
            wvad_values.append(wvad)
        
        # 第一个值为NaN
        result_length = len(group)
        values = [np.nan] + wvad_values
        # 确保返回值长度与输入数据长度一致
        if len(values) < result_length:
            values = values + [np.nan] * (result_length - len(values))
        elif len(values) > result_length:
            values = values[:result_length]
            
        return pd.Series(values, index=group.index)
    
    def calculate_psy(self, group, period):
        """计算PSY指标"""
        if len(group) < period + 1:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        psy_values = []
        for i in range(period, len(group)):
            # 计算PSY值 - 修复Series比较问题
            current_prices = group.loc[i-period+1:i, '收盘'].values
            previous_prices = group.loc[i-period:i-1, '收盘'].values
            up_days = (current_prices > previous_prices).sum()
            psy = up_days / period * 100
            psy_values.append(psy)
        
        # 前面period-1个值为NaN
        result_length = len(group)
        values = [np.nan] * (period - 1) + psy_values
        # 确保返回值长度与输入数据长度一致
        if len(values) < result_length:
            values = values + [np.nan] * (result_length - len(values))
        elif len(values) > result_length:
            values = values[:result_length]
            
        return pd.Series(values, index=group.index)
    
    def calculate_money_flow(self, group, period):
        """计算资金流量指标"""
        if len(group) < period + 1:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        mf_values = []
        for i in range(period, len(group)):
            # 计算资金流量
            typical_price = (group.loc[i-period+1:i, '最高'] + group.loc[i-period+1:i, '最低'] + group.loc[i-period+1:i, '收盘']) / 3
            money_flow = typical_price * group.loc[i-period+1:i, '成交量']
            mf_values.append(money_flow.sum())
        
        # 前面period-1个值为NaN
        result_length = len(group)
        values = [np.nan] * (period - 1) + mf_values
        # 确保返回值长度与输入数据长度一致
        if len(values) < result_length:
            values = values + [np.nan] * (result_length - len(values))
        elif len(values) > result_length:
            values = values[:result_length]
            
        return pd.Series(values, index=group.index)
    
    def calculate_mfi(self, group, period):
        """计算MFI指标"""
        if len(group) < period + 1:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        mfi_values = []
        for i in range(period, len(group)):
            # 计算典型价格
            typical_price = (group.loc[i-period+1:i, '最高'] + group.loc[i-period+1:i, '最低'] + group.loc[i-period+1:i, '收盘']) / 3
            
            # 计算原始资金流量
            raw_money_flow = typical_price * group.loc[i-period+1:i, '成交量']
            
            # 区分正负资金流量 - 修复Series比较问题
            tp_values = typical_price.values
            tp_shifted = np.roll(tp_values, 1)
            tp_shifted[0] = tp_values[0]  # 第一个值无法比较，使用自身值
            
            positive_mask = tp_values > tp_shifted
            negative_mask = tp_values < tp_shifted
            
            positive_flow = raw_money_flow[positive_mask].sum()
            negative_flow = raw_money_flow[negative_mask].sum()
            
            # 计算MFI
            if negative_flow != 0:
                mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            else:
                mfi = 100
            
            mfi_values.append(mfi)
        
        # 前面period-1个值为NaN
        result_length = len(group)
        values = [np.nan] * (period - 1) + mfi_values
        # 确保返回值长度与输入数据长度一致
        if len(values) < result_length:
            values = values + [np.nan] * (result_length - len(values))
        elif len(values) > result_length:
            values = values[:result_length]
            
        return pd.Series(values, index=group.index)
    
    def calculate_all_factors(self):
        """计算所有因子"""
        if not self.load_price_data():
            return None
        
        # 计算各类因子
        self.price_data = self.calculate_momentum_factors()
        self.price_data = self.calculate_emotion_factors()
        self.price_data = self.calculate_risk_factors()
        self.price_data = self.calculate_technical_factors()
        
        return self.price_data
    
    def save_factor_data(self, output_path):
        """保存因子数据"""
        if self.price_data is not None:
            try:
                self.price_data.to_csv(output_path, index=False)
                print(f"因子数据已保存到: {output_path}")
                return True
            except Exception as e:
                print(f"保存因子数据失败: {e}")
                return False
        else:
            print("没有可保存的因子数据")
            return False
    
    def save_factors_by_type(self):
        """将各类因子单独保存到不同文件夹"""
        if self.price_data is None:
            print("没有可保存的因子数据")
            return False
        
        try:
            # 创建输出目录
            base_dir = "c:\\Users\\Administrator\\Desktop\\design3\\data\\factories"
            
            # 识别各类因子列
            momentum_cols = []
            emotion_cols = []
            risk_cols = []
            technical_cols = []
            
            # 基础列
            base_cols = ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            
            # 分类因子列
            for col in self.price_data.columns:
                if col not in base_cols:
                    # 动量类因子
                    if col.startswith('ROC') or col.startswith('BIAS') or col.startswith('CCI') or col == 'CR20' or \
                       col.startswith('TRIX') or col == 'fifty_two_week_close_rank' or col.startswith('PLRC') or \
                       col.startswith('Price') or col.startswith('Rank') or col.startswith('arron') or \
                       col in ['bull_power', 'bear_power', 'BBIC', 'MASS', 'single_day_VPT', 'single_day_VPT_6', 
                               'single_day_VPT_12', 'Volume1M']:
                        momentum_cols.append(col)
                    
                    # 情绪类因子
                    elif col in ['AR', 'BR', 'ARBR'] or col.startswith('ATR') or col.startswith('VOL') or \
                         col == 'turnover_volatility' or col.startswith('DAVOL') or col.startswith('TVMA') or \
                         col.startswith('TVSTD') or col.startswith('VEMA') or col in ['VDIFF', 'VDEA', 'VMACD', 
                                                                                      'VOSC', 'VR'] or col.startswith('VROC') or \
                         col.startswith('VSTD') or col in ['WVAD', 'MAWVAD', 'PSY', 'money_flow_20']:
                        emotion_cols.append(col)
                    
                    # 风险类因子
                    elif col.startswith('Variance') or col.startswith('Skewness') or col.startswith('Kurtosis') or \
                         col.startswith('sharpe_ratio'):
                        risk_cols.append(col)
                    
                    # 技术指标类因子
                    elif col.startswith('MAC') or col.startswith('EMAC') or col in ['boll_up', 'boll_down', 'MACDC', 'MFI14', 'price_no_fq']:
                        technical_cols.append(col)
            
            # 保存动量类因子
            if momentum_cols:
                momentum_data = self.price_data[base_cols + momentum_cols]
                momentum_path = os.path.join(base_dir, "momentum", "momentum_factors.csv")
                momentum_data.to_csv(momentum_path, index=False)
                print(f"动量类因子已保存到: {momentum_path}")
            
            # 保存情绪类因子
            if emotion_cols:
                emotion_data = self.price_data[base_cols + emotion_cols]
                emotion_path = os.path.join(base_dir, "emotion", "emotion_factors.csv")
                emotion_data.to_csv(emotion_path, index=False)
                print(f"情绪类因子已保存到: {emotion_path}")
            
            # 保存风险类因子
            if risk_cols:
                risk_data = self.price_data[base_cols + risk_cols]
                risk_path = os.path.join(base_dir, "risk", "risk_factors.csv")
                risk_data.to_csv(risk_path, index=False)
                print(f"风险类因子已保存到: {risk_path}")
            
            # 保存技术指标类因子
            if technical_cols:
                technical_data = self.price_data[base_cols + technical_cols]
                technical_path = os.path.join(base_dir, "technical", "technical_factors.csv")
                technical_data.to_csv(technical_path, index=False)
                print(f"技术指标类因子已保存到: {technical_path}")
            
            return True
            
        except Exception as e:
            print(f"保存分类因子数据失败: {e}")
            return False

def main():
    # 设置文件路径
    price_data_path = "c:\\Users\\Administrator\\Desktop\\design3\\data\\daliy_prices_merged\\all_price_data.csv"
    output_path = "c:\\Users\\Administrator\\Desktop\\design3\\data\\calculated_factors.csv"
    
    # 创建因子计算器
    calculator = FactorCalculator(price_data_path)
    
    # 计算所有因子
    factor_data = calculator.calculate_all_factors()
    
    if factor_data is not None:
        # 保存因子数据
        calculator.save_factor_data(output_path)
        
        # 将各类因子单独保存到不同文件夹
        calculator.save_factors_by_type()
        
        # 打印因子数据信息
        print(f"计算完成，共计算了{len(factor_data.columns)}个字段")
        print("因子数据前5行:")
        print(factor_data.head())
    else:
        print("因子计算失败")

if __name__ == "__main__":
    main()