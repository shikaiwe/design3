#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务指标数据合并脚本
将所有股票的财务指标数据合并成一个统一的数据集
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_financial_indicator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# 输入目录（财务指标数据目录）
INPUT_DIR = os.path.join(PARENT_DIR, "data", "financial_indicator_data")
# 输出目录
OUTPUT_DIR = os.path.join(PARENT_DIR, "data", "merged_data")
# 输出文件名
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "financial_indicator_merged.csv")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_financial_indicator_files() -> List[Tuple[str, pd.DataFrame]]:
    """
    加载所有财务指标数据文件
    
    Returns:
        List[Tuple[str, pd.DataFrame]]: 股票代码和数据框的列表
    """
    files_data = []
    
    try:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv') and f.startswith('financial_indicator_')]
        logger.info(f"找到 {len(files)} 个财务指标数据文件")
        
        for filename in files:
            # 提取股票代码
            symbol = filename.replace('financial_indicator_', '').replace('.csv', '')
            filepath = os.path.join(INPUT_DIR, filename)
            
            try:
                # 读取CSV文件
                df = pd.read_csv(filepath)
                
                if not df.empty:
                    # 添加股票代码列
                    df['股票代码'] = symbol
                    files_data.append((symbol, df))
                    logger.info(f"成功加载股票 {symbol} 的财务指标数据，共 {len(df)} 条记录")
                else:
                    logger.warning(f"股票 {symbol} 的财务指标数据为空")
                    
            except Exception as e:
                logger.error(f"加载股票 {symbol} 的财务指标数据失败: {e}")
        
        logger.info(f"成功加载 {len(files_data)} 个股票的财务指标数据")
        return files_data
        
    except Exception as e:
        logger.error(f"加载财务指标数据文件失败: {e}")
        return []


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化列名
    
    Args:
        df (pd.DataFrame): 原始数据框
        
    Returns:
        pd.DataFrame: 标准化后的数据框
    """
    # 创建列名映射字典
    column_mapping = {}
    
    for col in df.columns:
        # 如果列名包含括号，保留括号前的部分
        if '(' in col and ')' in col:
            base_col = col.split('(')[0].strip()
            column_mapping[col] = base_col
        else:
            column_mapping[col] = col
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 标准化常见列名
    standard_mapping = {
        '日期': 'date',
        '股票代码': 'symbol',
        '摊薄每股收益': 'eps_diluted',
        '加权每股收益': 'eps_weighted',
        '每股收益_调整后': 'eps_adjusted',
        '扣除非经常性损益后的每股收益': 'eps_deducted',
        '每股净资产_调整前': 'bps_before_adj',
        '每股净资产_调整后': 'bps_after_adj',
        '每股经营性现金流': 'ocfps',
        '每股资本公积金': 'reserve_per_share',
        '每股未分配利润': 'retained_earnings_per_share',
        '调整后的每股净资产': 'adjusted_bps',
        '总资产利润率': 'roa',
        '主营业务利润率': 'main_business_profit_margin',
        '总资产净利润率': 'net_profit_margin_on_assets',
        '成本费用利润率': 'cost_expense_profit_margin',
        '营业利润率': 'operating_profit_margin',
        '主营业务成本率': 'main_business_cost_ratio',
        '销售净利率': 'net_sales_margin',
        '股本报酬率': 'capital_return_rate',
        '净资产报酬率': 'roe_return',
        '资产报酬率': 'asset_return_rate',
        '销售毛利率': 'gross_profit_margin',
        '三项费用比重': 'three_expense_ratio',
        '非主营比重': 'non_main_business_ratio',
        '主营利润比重': 'main_business_profit_ratio',
        '股息发放率': 'dividend_payout_ratio',
        '投资收益率': 'investment_return_rate',
        '主营业务利润': 'main_business_profit',
        '净资产收益率': 'roe',
        '加权净资产收益率': 'weighted_roe',
        '扣除非经常性损益后的净利润': 'net_profit_deducted',
        '主营业务收入增长率': 'main_business_revenue_growth',
        '净利润增长率': 'net_profit_growth',
        '净资产增长率': 'net_asset_growth',
        '总资产增长率': 'total_asset_growth',
        '应收账款周转率': 'accounts_receivable_turnover',
        '应收账款周转天数': 'accounts_receivable_turnover_days',
        '存货周转天数': 'inventory_turnover_days',
        '存货周转率': 'inventory_turnover',
        '固定资产周转率': 'fixed_asset_turnover',
        '总资产周转率': 'total_asset_turnover',
        '总资产周转天数': 'total_asset_turnover_days',
        '流动资产周转率': 'current_asset_turnover',
        '流动资产周转天数': 'current_asset_turnover_days',
        '股东权益周转率': 'shareholder_equity_turnover',
        '流动比率': 'current_ratio',
        '速动比率': 'quick_ratio',
        '现金比率': 'cash_ratio',
        '利息支付倍数': 'interest_coverage_ratio',
        '长期债务与营运资金比率': 'long_term_debt_to_working_capital',
        '股东权益比率': 'shareholder_equity_ratio',
        '长期负债比率': 'long_term_debt_ratio',
        '股东权益与固定资产比率': 'equity_to_fixed_assets_ratio',
        '负债与所有者权益比率': 'debt_to_equity_ratio',
        '长期资产与长期资金比率': 'long_term_assets_to_funds_ratio',
        '资本化比率': 'capitalization_ratio',
        '固定资产净值率': 'fixed_asset_net_value_ratio',
        '资本固定化比率': 'capital_fixed_ratio',
        '产权比率': 'property_rights_ratio',
        '清算价值比率': 'liquidation_value_ratio',
        '固定资产比重': 'fixed_asset_ratio',
        '资产负债率': 'debt_to_asset_ratio',
        '总资产': 'total_assets',
        '经营现金净流量对销售收入比率': 'operating_cash_to_sales_ratio',
        '资产的经营现金流量回报率': 'operating_cash_return_on_assets',
        '经营现金净流量与净利润的比率': 'operating_cash_to_net_profit_ratio',
        '经营现金净流量对负债比率': 'operating_cash_to_debt_ratio',
        '现金流量比率': 'cash_flow_ratio',
        '短期股票投资': 'short_term_stock_investment',
        '短期债券投资': 'short_term_bond_investment',
        '短期其它经营性投资': 'short_term_other_investment',
        '长期股票投资': 'long_term_stock_investment',
        '长期债券投资': 'long_term_bond_investment',
        '长期其它经营性投资': 'long_term_other_investment',
        '1年以内应收帐款': 'accounts_receivable_1yr',
        '1-2年以内应收帐款': 'accounts_receivable_1_2yr',
        '2-3年以内应收帐款': 'accounts_receivable_2_3yr',
        '3年以内应收帐款': 'accounts_receivable_3yr',
        '1年以内预付货款': 'prepayment_1yr',
        '1-2年以内预付货款': 'prepayment_1_2yr',
        '2-3年以内预付货款': 'prepayment_2_3yr',
        '3年以内预付货款': 'prepayment_3yr',
        '1年以内其它应收款': 'other_receivables_1yr',
        '1-2年以内其它应收款': 'other_receivables_1_2yr',
        '2-3年以内其它应收款': 'other_receivables_2_3yr',
        '3年以内其它应收款': 'other_receivables_3yr'
    }
    
    # 应用标准列名映射
    df = df.rename(columns=standard_mapping)
    
    return df


def merge_financial_indicator_data(files_data: List[Tuple[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
    """
    合并所有财务指标数据
    
    Args:
        files_data (List[Tuple[str, pd.DataFrame]]): 股票代码和数据框的列表
        
    Returns:
        Optional[pd.DataFrame]: 合并后的数据框，如果合并失败返回None
    """
    if not files_data:
        logger.error("没有可合并的财务指标数据")
        return None
    
    try:
        # 标准化所有数据框的列名
        standardized_data = []
        for symbol, df in files_data:
            df_std = standardize_columns(df)
            standardized_data.append((symbol, df_std))
        
        # 提取所有数据框
        dfs = [df for _, df in standardized_data]
        
        # 合并所有数据框
        logger.info("开始合并财务指标数据...")
        merged_df = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"合并完成，共 {len(merged_df)} 条记录，{len(merged_df.columns)} 列")
        
        # 检查关键列是否存在
        required_columns = ['date', 'symbol']
        missing_columns = [col for col in required_columns if col not in merged_df.columns]
        
        if missing_columns:
            logger.warning(f"缺少关键列: {missing_columns}")
        
        # 转换日期列为datetime类型
        if 'date' in merged_df.columns:
            try:
                merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
                # 移除日期为NaT的行
                merged_df = merged_df.dropna(subset=['date'])
                # 只保留2019-01-01至2024-12-31的数据
                start_date = pd.to_datetime('2019-01-01')
                end_date = pd.to_datetime('2024-12-31')
                merged_df = merged_df[(merged_df['date'] >= start_date) & (merged_df['date'] <= end_date)]
                logger.info(f"日期过滤完成，有效记录数: {len(merged_df)}")
            except Exception as e:
                logger.error(f"日期转换失败: {e}")
        
        # 处理空值
        logger.info("开始处理空值...")
        
        # 首先检查并删除空值比例为100%的列
        null_ratio = merged_df.isnull().sum() / len(merged_df)
        full_null_columns = null_ratio[null_ratio == 1.0].index.tolist()
        
        if full_null_columns:
            logger.info(f"发现 {len(full_null_columns)} 个全空列，将直接删除: {full_null_columns}")
            merged_df = merged_df.drop(columns=full_null_columns)
        
        # 获取数值型列
        numeric_columns = merged_df.select_dtypes(include=['number']).columns.tolist()
        # 排除关键列
        numeric_columns = [col for col in numeric_columns if col not in ['date', 'symbol']]
        
        # 对数值型列的空值进行填充
        if numeric_columns:
            # 按股票代码分组，使用前向填充和后向填充
            for col in numeric_columns:
                # 按股票代码分组，然后按日期排序，进行前向填充
                merged_df[col] = merged_df.groupby('symbol')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                
                # 如果仍有空值，使用该列的中位数填充
                if merged_df[col].isnull().sum() > 0:
                    median_value = merged_df[col].median()
                    merged_df[col] = merged_df[col].fillna(median_value)
                    logger.info(f"列 {col} 使用中位数 {median_value} 填充剩余空值")
        
        # 获取非数值型列（排除关键列）
        non_numeric_columns = merged_df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        non_numeric_columns = [col for col in non_numeric_columns if col not in ['date', 'symbol']]
        
        # 对非数值型列的空值进行填充
        if non_numeric_columns:
            for col in non_numeric_columns:
                # 按股票代码分组，使用前向填充和后向填充
                merged_df[col] = merged_df.groupby('symbol')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                
                # 如果仍有空值，使用该列的众数填充
                if merged_df[col].isnull().sum() > 0:
                    mode_values = merged_df[col].mode()
                    if not mode_values.empty:
                        mode_value = mode_values[0]
                        merged_df[col] = merged_df[col].fillna(mode_value)
                        logger.info(f"列 {col} 使用众数 {mode_value} 填充剩余空值")
        
        # 统计处理后的空值情况
        null_counts = merged_df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"处理后仍有空值，列名及空值数量: {null_counts[null_counts > 0].to_dict()}")
        else:
            logger.info("所有空值已处理完成")
        
        # 按股票代码和日期排序
        if 'symbol' in merged_df.columns and 'date' in merged_df.columns:
            merged_df = merged_df.sort_values(['symbol', 'date'])
            merged_df = merged_df.reset_index(drop=True)
        
        return merged_df
        
    except Exception as e:
        logger.error(f"合并财务指标数据失败: {e}")
        return None


def save_merged_data(df: pd.DataFrame, output_file: str) -> bool:
    """
    保存合并后的数据
    
    Args:
        df (pd.DataFrame): 合并后的数据框
        output_file (str): 输出文件路径
        
    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存到CSV文件
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"成功保存合并后的财务指标数据到 {output_file}")
        
        # 保存数据摘要
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"财务指标数据合并摘要\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总记录数: {len(df)}\n")
            f.write(f"总列数: {len(df.columns)}\n")
            f.write(f"股票数量: {df['symbol'].nunique() if 'symbol' in df.columns else '未知'}\n")
            
            if 'date' in df.columns:
                min_date = df['date'].min().strftime('%Y-%m-%d')
                max_date = df['date'].max().strftime('%Y-%m-%d')
                f.write(f"数据日期范围: {min_date} 至 {max_date}\n")
            
            f.write("\n列名列表:\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i}. {col}\n")
        
        logger.info(f"数据摘要已保存到 {summary_file}")
        return True
        
    except Exception as e:
        logger.error(f"保存合并后的财务指标数据失败: {e}")
        return False


def validate_merged_data(df: pd.DataFrame) -> Dict:
    """
    验证合并后的数据质量
    
    Args:
        df (pd.DataFrame): 合并后的数据框
        
    Returns:
        Dict: 数据质量报告
    """
    report = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "unique_stocks": 0,
        "date_range": {"earliest": None, "latest": None},
        "data_types": {},
        "sample_data": {}
    }
    
    try:
        # 股票数量
        if 'symbol' in df.columns:
            report["unique_stocks"] = df['symbol'].nunique()
        
        # 日期范围
        if 'date' in df.columns:
            min_date = df['date'].min()
            max_date = df['date'].max()
            report["date_range"]["earliest"] = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else None
            report["date_range"]["latest"] = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else None
        
        # 数据类型
        for col in df.columns:
            report["data_types"][col] = str(df[col].dtype)
        
        # 示例数据
        if not df.empty:
            sample_size = min(5, len(df))
            report["sample_data"] = df.head(sample_size).to_dict('records')
        
    except Exception as e:
        logger.error(f"数据验证失败: {e}")
    
    return report


def main():
    """
    主函数
    """
    logger.info("开始合并财务指标数据")
    start_time = time.time()
    
    # 加载所有财务指标数据文件
    files_data = load_financial_indicator_files()
    
    if not files_data:
        logger.error("没有找到可合并的财务指标数据文件，程序退出")
        return
    
    # 合并数据
    merged_df = merge_financial_indicator_data(files_data)
    
    if merged_df is None:
        logger.error("合并财务指标数据失败，程序退出")
        return
    
    # 保存合并后的数据
    save_success = save_merged_data(merged_df, OUTPUT_FILE)
    
    if not save_success:
        logger.error("保存合并后的财务指标数据失败，程序退出")
        return
    
    # 验证数据质量
    logger.info("开始验证合并后的数据质量")
    quality_report = validate_merged_data(merged_df)
    
    # 输出统计信息
    elapsed_time = time.time() - start_time
    logger.info(f"财务指标数据合并完成")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"总记录数: {quality_report['total_records']}")
    logger.info(f"总列数: {quality_report['total_columns']}")
    logger.info(f"股票数量: {quality_report['unique_stocks']}")
    
    if quality_report["date_range"]["earliest"] and quality_report["date_range"]["latest"]:
        logger.info(f"数据日期范围: {quality_report['date_range']['earliest']} 至 {quality_report['date_range']['latest']}")
    
    # 保存质量报告
    report_file = OUTPUT_FILE.replace('.csv', '_quality_report.json')
    try:
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"数据质量报告已保存到 {report_file}")
    except Exception as e:
        logger.error(f"保存数据质量报告失败: {e}")


if __name__ == "__main__":
    main()