import pandas as pd
import os

# 读取我们计算的因子
momentum_df = pd.read_csv('c:\\Users\\Administrator\\Desktop\\design3\\data\\factories\\momentum\\momentum_factors.csv')
emotion_df = pd.read_csv('c:\\Users\\Administrator\\Desktop\\design3\\data\\factories\\emotion\\emotion_factors.csv')
risk_df = pd.read_csv('c:\\Users\\Administrator\\Desktop\\design3\\data\\factories\\risk\\risk_factors.csv')
technical_df = pd.read_csv('c:\\Users\\Administrator\\Desktop\\design3\\data\\factories\\technical\\technical_factors.csv')

# 基础数据列（非因子）
base_columns = ['stock_code', '日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

# 提取我们计算的因子列名（排除基础数据列）
our_momentum_factors = [col for col in momentum_df.columns if col not in base_columns]
our_emotion_factors = [col for col in emotion_df.columns if col not in base_columns]
our_risk_factors = [col for col in risk_df.columns if col not in base_columns]
our_technical_factors = [col for col in technical_df.columns if col not in base_columns]

# factors_analysis.md中定义的因子
doc_momentum_factors = [
    'arron_down_25', 'arron_up_25', 'BBIC', 'bear_power', 'BIAS10', 'BIAS20', 
    'BIAS5', 'BIAS60', 'bull_power', 'CCI10', 'CCI15', 'CCI20', 'CCI88', 
    'CR20', 'fifty_two_week_close_rank', 'MASS', 'PLRC12', 'PLRC24', 
    'PLRC6', 'Price1M', 'Price1Y', 'Price3M', 'Rank1M', 'ROC12', 'ROC120', 
    'ROC20', 'ROC6', 'ROC60', 'single_day_VPT', 'single_day_VPT_12', 
    'single_day_VPT_6', 'TRIX10', 'TRIX5', 'Volume1M'
]

doc_emotion_factors = [
    'AR', 'ARBR', 'ATR14', 'ATR6', 'BR', 'DAVOL10', 'DAVOL20', 'DAVOL5', 
    'MAWVAD', 'money_flow_20', 'PSY', 'turnover_volatility', 'TVMA20', 
    'TVMA6', 'TVSTD20', 'TVSTD6', 'VDEA', 'VDIFF', 'VEMA10', 'VEMA12', 
    'VEMA26', 'VEMA5', 'VMACD', 'VOL10', 'VOL120', 'VOL20', 'VOL240', 
    'VOL5', 'VOL60', 'VOSC', 'VR', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20', 'WVAD'
]

doc_risk_factors = [
    'Kurtosis120', 'Kurtosis20', 'Kurtosis60', 'sharpe_ratio_120', 
    'sharpe_ratio_20', 'sharpe_ratio_60', 'Skewness120', 'Skewness20', 
    'Skewness60', 'Variance120', 'Variance20', 'Variance60'
]

doc_technical_factors = [
    'boll_down', 'boll_up', 'EMA5', 'EMAC10', 'EMAC12', 'EMAC120', 
    'EMAC20', 'EMAC26', 'MAC10', 'MAC120', 'MAC20', 'MAC5', 'MAC60', 
    'MACDC', 'MFI14', 'price_no_fq'
]

# 比较函数
def compare_factors(our_factors, doc_factors, category_name):
    print(f"\n=== {category_name}因子比较 ===")
    print(f"我们计算的因子数量: {len(our_factors)}")
    print(f"文档中定义的因子数量: {len(doc_factors)}")
    
    # 找出我们计算但文档中没有的因子
    extra_factors = [f for f in our_factors if f not in doc_factors]
    print(f"\n我们计算但文档中没有的因子 ({len(extra_factors)}个):")
    for f in extra_factors:
        print(f"  - {f}")
    
    # 找出文档中有但我们没有计算的因子
    missing_factors = [f for f in doc_factors if f not in our_factors]
    print(f"\n文档中有但我们没有计算的因子 ({len(missing_factors)}个):")
    for f in missing_factors:
        print(f"  - {f}")
    
    # 找出共同的因子
    common_factors = [f for f in our_factors if f in doc_factors]
    print(f"\n共同的因子 ({len(common_factors)}个):")
    for f in common_factors:
        print(f"  - {f}")

# 执行比较
compare_factors(our_momentum_factors, doc_momentum_factors, "动量类")
compare_factors(our_emotion_factors, doc_emotion_factors, "情绪类")
compare_factors(our_risk_factors, doc_risk_factors, "风险类")
compare_factors(our_technical_factors, doc_technical_factors, "技术指标类")

# 汇总
print("\n=== 汇总 ===")
print(f"我们总共计算的因子数量: {len(our_momentum_factors) + len(our_emotion_factors) + len(our_risk_factors) + len(our_technical_factors)}")
print(f"文档中总共定义的因子数量: {len(doc_momentum_factors) + len(doc_emotion_factors) + len(doc_risk_factors) + len(doc_technical_factors)}")