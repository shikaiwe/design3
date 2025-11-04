import pandas as pd
import numpy as np
from scipy import stats
import os
from datetime import datetime

def read_epu_data():
    """
    读取中国EPU指数Excel文件，处理多个sheet并筛选2019-2024年的数据
    """
    file_path = r"c:\Users\Administrator\Desktop\design3\data\EPU\China_Mainland_Paper_EPU.xlsx"
    
    # 读取Excel文件的所有sheet
    try:
        # 获取所有sheet名称
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"Excel文件包含的sheet: {sheet_names}")
        
        # 读取第一个sheet（通常包含主要数据）
        df = pd.read_excel(file_path, sheet_name=sheet_names[0])
        print("EPU数据读取成功!")
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 重命名列以便更好地处理
        df.columns = ['year', 'month', 'EPU', 'col3', 'col4', 'source']
        
        # 删除不需要的列
        df = df[['year', 'month', 'EPU']].copy()
        
        # 转换数据类型
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df['EPU'] = pd.to_numeric(df['EPU'], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 创建日期列
        df['date'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-' + df['month'].astype(int).astype(str) + '-01')
        
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 筛选2000-2024年的数据（保留更早的历史数据用于计算因子）
        start_date = pd.to_datetime('2000-01-01')
        end_date = pd.to_datetime('2024-12-31')
        df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        
        print(f"\n筛选前数据范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"筛选后数据范围: {df_filtered['date'].min()} 到 {df_filtered['date'].max()}")
        print(f"筛选后数据形状: {df_filtered.shape}")
        print("\n处理后的前5行数据:")
        print(df_filtered.head())
        
        return df_filtered
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None

def analyze_epu_data(df):
    """
    分析EPU数据结构和时间序列特征
    """
    if df is None or df.empty:
        print("没有数据可供分析")
        return None
    
    print("\n===== EPU数据分析 =====")
    
    # 基本统计信息
    print("\n基本统计信息:")
    print(df['EPU'].describe())
    
    # 按年份统计
    print("\n按年份统计:")
    yearly_stats = df.groupby('year')['EPU'].agg(['mean', 'std', 'min', 'max'])
    print(yearly_stats)
    
    # 按月份统计（所有年份的平均）
    print("\n按月份统计（所有年份平均）:")
    monthly_stats = df.groupby('month')['EPU'].mean()
    print(monthly_stats)
    
    # 计算年度变化
    print("\n年度EPU变化:")
    yearly_change = yearly_stats['mean'].pct_change() * 100
    print(yearly_change)
    
    # 识别高EPU和低EPU时期
    print("\n高EPU时期（EPU > 75%分位数）:")
    high_epu_threshold = df['EPU'].quantile(0.75)
    high_epu_periods = df[df['EPU'] > high_epu_threshold]
    print(f"高EPU阈值: {high_epu_threshold:.2f}")
    print(high_epu_periods[['year', 'month', 'EPU']].head(10))
    
    print("\n低EPU时期（EPU < 25%分位数）:")
    low_epu_threshold = df['EPU'].quantile(0.25)
    low_epu_periods = df[df['EPU'] < low_epu_threshold]
    print(f"低EPU阈值: {low_epu_threshold:.2f}")
    print(low_epu_periods[['year', 'month', 'EPU']].head(10))
    
    return df

def calculate_epu_factors(df):
    """
    计算EPU相关因子
    """
    if df is None or df.empty:
        print("没有数据可供计算因子")
        return None
    
    print("\n===== 计算EPU相关因子 =====")
    
    # 创建一个新的DataFrame来存储因子
    factors_df = df.copy()
    
    # 1. EPU移动平均因子（3个月、6个月、12个月）
    # 使用最小周期数计算，避免空值
    factors_df['EPU_MA3'] = factors_df['EPU'].rolling(window=3, min_periods=1).mean()
    factors_df['EPU_MA6'] = factors_df['EPU'].rolling(window=6, min_periods=1).mean()
    factors_df['EPU_MA12'] = factors_df['EPU'].rolling(window=12, min_periods=1).mean()
    
    # 2. EPU变化率因子（月度变化、季度变化、年度变化）
    # 对于没有足够历史数据的点，使用可用数据计算
    factors_df['EPU_MOM'] = factors_df['EPU'].pct_change(periods=1, fill_method=None) * 100
    factors_df['EPU_QOQ'] = factors_df['EPU'].pct_change(periods=3, fill_method=None) * 100
    factors_df['EPU_YOY'] = factors_df['EPU'].pct_change(periods=12, fill_method=None) * 100
    
    # 3. EPU波动率因子（3个月、6个月、12个月滚动标准差）
    # 使用最小周期数计算，避免空值
    factors_df['EPU_VOL3'] = factors_df['EPU'].rolling(window=3, min_periods=1).std()
    factors_df['EPU_VOL6'] = factors_df['EPU'].rolling(window=6, min_periods=1).std()
    factors_df['EPU_VOL12'] = factors_df['EPU'].rolling(window=12, min_periods=1).std()
    
    # 4. EPU相对位置因子（相对于过去12个月的位置）
    # 对于数据点不足的情况，使用可用数据计算排名
    def rolling_rank(series):
        """自定义滚动排名函数，处理数据点不足的情况"""
        return series.rank(pct=True).iloc[-1]
    
    factors_df['EPU_RANK12'] = factors_df['EPU'].rolling(window=12, min_periods=1).apply(rolling_rank, raw=False)
    
    # 5. EPU极端值因子（是否高于或低于历史分位数）
    high_threshold = factors_df['EPU'].quantile(0.8)
    low_threshold = factors_df['EPU'].quantile(0.2)
    factors_df['EPU_HIGH_FLAG'] = (factors_df['EPU'] > high_threshold).astype(int)
    factors_df['EPU_LOW_FLAG'] = (factors_df['EPU'] < low_threshold).astype(int)
    
    # 6. EPU趋势因子（线性回归斜率）
    def calculate_trend(series, window=12):
        """计算时间序列的线性趋势，处理数据点不足的情况"""
        actual_window = min(window, len(series))
        if actual_window < 2:
            return 0
        x = np.arange(actual_window)
        y = series.values[-actual_window:]
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    factors_df['EPU_TREND12'] = factors_df['EPU'].rolling(window=12, min_periods=1).apply(calculate_trend, raw=False)
    
    # 7. EPU周期性因子（与平均月度偏差）
    monthly_avg = factors_df.groupby('month')['EPU'].mean()
    factors_df['EPU_MONTHLY_AVG'] = factors_df['month'].map(monthly_avg)
    factors_df['EPU_SEASONAL'] = factors_df['EPU'] - factors_df['EPU_MONTHLY_AVG']
    
    # 8. EPU加速度因子（变化率的变化）
    factors_df['EPU_ACCELERATION'] = factors_df['EPU_MOM'].diff()
    
    # 9. EPU离散度因子（与移动平均的绝对偏差）
    factors_df['EPU_DEVIATION_MA12'] = abs(factors_df['EPU'] - factors_df['EPU_MA12'])
    
    # 10. EPU复合因子（结合多个维度的综合指标）
    # 标准化各个因子
    def normalize(series):
        """标准化序列，处理空值"""
        if series.std() == 0:
            return series - series.mean()
        return (series - series.mean()) / series.std()
    
    # 填充空值后进行标准化
    mom_filled = factors_df['EPU_MOM'].fillna(0)
    vol_filled = factors_df['EPU_VOL6'].fillna(0)
    rank_filled = factors_df['EPU_RANK12'].fillna(0.5)
    trend_filled = factors_df['EPU_TREND12'].fillna(0)
    
    factors_df['EPU_COMPOSITE'] = (
        normalize(mom_filled) * 0.3 +
        normalize(vol_filled) * 0.2 +
        normalize(rank_filled) * 0.2 +
        normalize(trend_filled) * 0.3
    )
    
    # 填充剩余的空值
    factors_df = factors_df.fillna(0)
    
    print("已计算的EPU相关因子:")
    print(f"- 移动平均因子: EPU_MA3, EPU_MA6, EPU_MA12")
    print(f"- 变化率因子: EPU_MOM, EPU_QOQ, EPU_YOY")
    print(f"- 波动率因子: EPU_VOL3, EPU_VOL6, EPU_VOL12")
    print(f"- 相对位置因子: EPU_RANK12")
    print(f"- 极端值因子: EPU_HIGH_FLAG, EPU_LOW_FLAG")
    print(f"- 趋势因子: EPU_TREND12")
    print(f"- 周期性因子: EPU_SEASONAL")
    print(f"- 加速度因子: EPU_ACCELERATION")
    print(f"- 离散度因子: EPU_DEVIATION_MA12")
    print(f"- 复合因子: EPU_COMPOSITE")
    
    print(f"\n因子数据形状: {factors_df.shape}")
    print("\n因子数据前5行:")
    print(factors_df.head())
    
    # 只返回2019-2024年的数据
    result_df = factors_df[(factors_df['year'] >= 2019) & (factors_df['year'] <= 2024)].copy()
    print(f"\n筛选后返回2019-2024年数据，形状: {result_df.shape}")
    
    return result_df

def save_results_to_csv(factors_df):
    """
    将计算结果保存为CSV文件
    """
    if factors_df is None or factors_df.empty:
        print("没有数据可保存")
        return False
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "EPU")
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成固定文件名
    output_file = os.path.join(output_dir, "EPU_factors.csv")
    
    try:
        # 保存为CSV文件
        factors_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
        print(f"保存了 {len(factors_df)} 行数据和 {len(factors_df.columns)} 列因子")
        
        # 保存因子描述文件
        description_file = os.path.join(output_dir, "EPU_factors_description.txt")
        with open(description_file, 'w', encoding='utf-8') as f:
            f.write("中国EPU指数相关因子说明\n")
            f.write("=" * 50 + "\n\n")
            f.write("基础数据:\n")
            f.write("- year: 年份\n")
            f.write("- month: 月份\n")
            f.write("- EPU: 经济政策不确定性指数\n")
            f.write("- date: 日期\n\n")
            
            f.write("计算因子:\n")
            f.write("1. 移动平均因子:\n")
            f.write("   - EPU_MA3: 3个月移动平均\n")
            f.write("   - EPU_MA6: 6个月移动平均\n")
            f.write("   - EPU_MA12: 12个月移动平均\n\n")
            
            f.write("2. 变化率因子:\n")
            f.write("   - EPU_MOM: 月度变化率(%)\n")
            f.write("   - EPU_QOQ: 季度变化率(%)\n")
            f.write("   - EPU_YOY: 年度变化率(%)\n\n")
            
            f.write("3. 波动率因子:\n")
            f.write("   - EPU_VOL3: 3个月滚动标准差\n")
            f.write("   - EPU_VOL6: 6个月滚动标准差\n")
            f.write("   - EPU_VOL12: 12个月滚动标准差\n\n")
            
            f.write("4. 相对位置因子:\n")
            f.write("   - EPU_RANK12: 过去12个月内的相对位置(0-1)\n\n")
            
            f.write("5. 极端值因子:\n")
            f.write("   - EPU_HIGH_FLAG: 高EPU时期标记(1表示高于80%分位数)\n")
            f.write("   - EPU_LOW_FLAG: 低EPU时期标记(1表示低于20%分位数)\n\n")
            
            f.write("6. 趋势因子:\n")
            f.write("   - EPU_TREND12: 12个月线性趋势斜率\n\n")
            
            f.write("7. 周期性因子:\n")
            f.write("   - EPU_MONTHLY_AVG: 历史同月平均值\n")
            f.write("   - EPU_SEASONAL: 与历史同月平均值的偏差\n\n")
            
            f.write("8. 加速度因子:\n")
            f.write("   - EPU_ACCELERATION: 月度变化率的变化\n\n")
            
            f.write("9. 离散度因子:\n")
            f.write("   - EPU_DEVIATION_MA12: 与12个月移动平均的绝对偏差\n\n")
            
            f.write("10. 复合因子:\n")
            f.write("    - EPU_COMPOSITE: 结合多个维度的综合指标(标准化后加权平均)\n")
        
        print(f"因子说明已保存到: {description_file}")
        return True
        
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False

if __name__ == "__main__":
    epu_data = read_epu_data()
    analyzed_data = analyze_epu_data(epu_data)
    epu_factors = calculate_epu_factors(analyzed_data)
    save_results_to_csv(epu_factors)