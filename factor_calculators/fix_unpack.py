#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量修复所有因子计算器中的prepare_data解包问题
"""

import os
import re

# 需要更新的计算器列表
calculators = [
    'basics_calculator.py',
    'growth_calculator.py',
    'momentum_calculator.py',
    'per_share_calculator.py',
    'quality_calculator.py',
    'risk_calculator.py',
    'sentiment_calculator.py',
    'style_calculator.py',
    'new_style_calculator.py',
    'technical_calculator.py'
]

# 计算器目录
calculator_dir = r"c:\Users\Administrator\Desktop\design3\factor_calculators"

# 更新每个计算器
for calculator in calculators:
    file_path = os.path.join(calculator_dir, calculator)
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新prepare_data解包
    # 查找prepare_data调用
    unpack_pattern = r'financial_df, price_df, index_df = self\.prepare_data\('
    new_unpack = 'financial_df, price_df, index_df, financial_indicator_df, industry_df = self.prepare_data('
    
    if re.search(unpack_pattern, content):
        content = re.sub(unpack_pattern, new_unpack, content)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已更新 {calculator}")
    else:
        print(f"未在 {calculator} 中找到prepare_data解包，跳过更新")

print("所有计算器更新完成")