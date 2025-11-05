#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量更新所有因子计算器以支持新的数据源
"""

import os
import re

# 需要更新的计算器列表
calculators = [
    'momentum_calculator.py',
    'per_share_calculator.py',
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
    
    # 更新calculate方法签名
    # 查找calculate方法定义
    calculate_pattern = r'def calculate\(self, financial_data, price_data, index_data\):'
    new_signature = 'def calculate(self, financial_data, price_data, index_data, financial_indicator_data=None, industry_data=None):'
    
    if re.search(calculate_pattern, content):
        content = re.sub(calculate_pattern, new_signature, content)
        
        # 更新方法文档字符串
        doc_pattern = r'(def calculate\(self, financial_data, price_data, index_data, financial_indicator_data=None, industry_data=None\):\s*"""[\s\S]*?参数:\s*financial_data: 财务数据DataFrame\s*price_data: 价格数据DataFrame\s*index_data: 指数数据DataFrame)'
        new_doc = r'\1\n            financial_indicator_data: 财务指标数据DataFrame (可选)\n            industry_data: 行业分类数据DataFrame (可选)'
        
        if re.search(doc_pattern, content):
            content = re.sub(doc_pattern, new_doc, content)
        else:
            # 如果找不到匹配的文档字符串模式，尝试简单添加
            param_pattern = r'(index_data: 指数数据DataFrame)'
            new_param = r'\1\n            financial_indicator_data: 财务指标数据DataFrame (可选)\n            industry_data: 行业分类数据DataFrame (可选)'
            
            if re.search(param_pattern, content):
                content = re.sub(param_pattern, new_param, content)
        
        # 更新prepare_data调用
        prepare_pattern = r'financial_df, price_df, index_df = self\.prepare_data\(financial_data, price_data, index_data\)'
        new_prepare = 'financial_df, price_df, index_df = self.prepare_data(\n            financial_data, \n            price_data, \n            index_data, \n            financial_indicator_data, \n            industry_data\n        )'
        
        if re.search(prepare_pattern, content):
            content = re.sub(prepare_pattern, new_prepare, content)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已更新 {calculator}")
    else:
        print(f"未在 {calculator} 中找到calculate方法，跳过更新")

print("所有计算器更新完成")