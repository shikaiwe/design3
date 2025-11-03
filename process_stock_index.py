import pandas as pd

# 读取CSV文件
df = pd.read_csv('c:\\Users\\Administrator\\Desktop\\design3\\data\\components\\stock_index_cut.csv')

# 将品种代码转换为字符串，确保保留全部六位数字
df['品种代码'] = df['品种代码'].astype(str)

# 去重
df_unique = df.drop_duplicates(subset=['品种代码'], keep='first')

# 按品种代码的数值从小到大排序
df_sorted = df_unique.copy()
df_sorted['品种代码_数值'] = pd.to_numeric(df_sorted['品种代码'])
df_sorted = df_sorted.sort_values(by='品种代码_数值')
df_sorted = df_sorted.drop('品种代码_数值', axis=1)

# 确保所有品种代码都是六位数，不足六位在前面补零
df_sorted['品种代码'] = df_sorted['品种代码'].apply(lambda x: x.zfill(6))

# 保存处理后的文件
output_path = 'c:\\Users\\Administrator\\Desktop\\design3\\data\\components\\stock_index_sorted.csv'
df_sorted.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"处理完成，结果已保存到: {output_path}")
print(f"原始数据行数: {len(df)}")
print(f"去重后数据行数: {len(df_unique)}")
print(f"排序后数据行数: {len(df_sorted)}")

# 显示前10行和后10行数据
print("\n前10行数据:")
print(df_sorted.head(10))
print("\n后10行数据:")
print(df_sorted.tail(10))