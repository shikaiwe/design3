#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务指标数据收集器
从新浪财经获取股票的财务指标数据
"""

import os
import time
import json
import logging
import pandas as pd
import akshare as ak
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_indicator_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# 输出目录
OUTPUT_DIR = os.path.join(PARENT_DIR, "data", "financial_indicator_data")
# 断点文件
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "financial_indicator_checkpoint.json")
# 股票列表文件
STOCK_LIST_FILE = os.path.join(PARENT_DIR, "data", "components", "stock_index_cut.csv") 
# 批处理大小
BATCH_SIZE = 10
# 请求间隔（秒）
REQUEST_INTERVAL = 2
# 最大重试次数
MAX_RETRIES = 3

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_checkpoint() -> Dict:
    """
    加载断点信息
    
    Returns:
        Dict: 断点信息
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载断点文件失败: {e}")
    
    return {
        "processed_stocks": [],
        "current_batch": 1,
        "last_processed_index": -1
    }


def save_checkpoint(checkpoint: Dict) -> None:
    """
    保存断点信息
    
    Args:
        checkpoint (Dict): 断点信息
    """
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存断点文件失败: {e}")


def load_stock_list() -> List[str]:
    """
    加载股票列表
    
    Returns:
        List[str]: 股票代码列表
    """
    try:
        # 使用pandas读取CSV文件，只获取第一列（品种代码）
        df = pd.read_csv(STOCK_LIST_FILE, encoding='utf-8-sig')  # 使用utf-8-sig处理BOM
        stock_list = df.iloc[:, 0].astype(str).tolist()  # 获取第一列作为股票代码，并转为字符串
        # 确保股票代码格式正确（6位数字，不足补0）
        stock_list = [code.zfill(6) if code.isdigit() else code for code in stock_list]
        logger.info(f"加载股票列表成功，共 {len(stock_list)} 只股票")
        return stock_list
    except Exception as e:
        logger.error(f"加载股票列表失败: {e}")
        return []


def get_financial_indicator_data(symbol: str, start_year: str = "2019") -> Optional[pd.DataFrame]:
    """
    获取财务指标数据
    
    Args:
        symbol (str): 股票代码
        start_year (str): 开始年份，默认为2019
        
    Returns:
        Optional[pd.DataFrame]: 财务指标数据，如果获取失败返回None
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"获取股票 {symbol} 的财务指标数据 (尝试 {attempt + 1}/{MAX_RETRIES})")
            
            # 调用akshare接口获取财务指标数据
            df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
            
            if df is not None and not df.empty:
                logger.info(f"成功获取股票 {symbol} 的财务指标数据，共 {len(df)} 条记录")
                return df
            else:
                logger.warning(f"股票 {symbol} 的财务指标数据为空")
                return None
                
        except Exception as e:
            logger.warning(f"获取股票 {symbol} 的财务指标数据失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_INTERVAL * (attempt + 1))  # 递增等待时间
    
    logger.error(f"获取股票 {symbol} 的财务指标数据失败，已达到最大重试次数")
    return None


def save_financial_indicator_data(df: pd.DataFrame, symbol: str) -> bool:
    """
    保存财务指标数据到文件
    
    Args:
        df (pd.DataFrame): 财务指标数据
        symbol (str): 股票代码
        
    Returns:
        bool: 保存是否成功
    """
    try:
        # 创建文件名
        filename = f"financial_indicator_{symbol}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # 保存到CSV文件
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"成功保存股票 {symbol} 的财务指标数据到 {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"保存股票 {symbol} 的财务指标数据失败: {e}")
        return False


def process_stock(symbol: str, start_year: str = "2019") -> bool:
    """
    处理单只股票的财务指标数据
    
    Args:
        symbol (str): 股票代码
        start_year (str): 开始年份
        
    Returns:
        bool: 处理是否成功
    """
    try:
        # 获取财务指标数据
        df = get_financial_indicator_data(symbol, start_year)
        
        # 保存数据（即使为空也保存，确保文件存在）
        if df is not None:
            save_success = save_financial_indicator_data(df, symbol)
            return save_success
        else:
            # 创建空的DataFrame并保存
            empty_df = pd.DataFrame()
            save_success = save_financial_indicator_data(empty_df, symbol)
            return save_success
            
    except Exception as e:
        logger.error(f"处理股票 {symbol} 时发生异常: {e}")
        return False


def process_batch(stock_list: List[str], start_index: int, end_index: int, start_year: str = "2019") -> Tuple[int, int]:
    """
    处理一批股票
    
    Args:
        stock_list (List[str]): 股票列表
        start_index (int): 开始索引
        end_index (int): 结束索引
        start_year (str): 开始年份
        
    Returns:
        Tuple[int, int]: (成功数量, 失败数量)
    """
    success_count = 0
    failure_count = 0
    
    for i in range(start_index, min(end_index, len(stock_list))):
        symbol = stock_list[i]
        logger.info(f"处理股票 {symbol} ({i+1}/{len(stock_list)})")
        
        if process_stock(symbol, start_year):
            success_count += 1
        else:
            failure_count += 1
        
        # 请求间隔
        time.sleep(REQUEST_INTERVAL)
    
    return success_count, failure_count


def validate_data_quality() -> Dict:
    """
    验证数据质量
    
    Returns:
        Dict: 数据质量报告
    """
    report = {
        "total_files": 0,
        "empty_files": 0,
        "valid_files": 0,
        "date_range": {"earliest": None, "latest": None},
        "columns": set()
    }
    
    try:
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
        report["total_files"] = len(files)
        
        for filename in files:
            filepath = os.path.join(OUTPUT_DIR, filename)
            try:
                df = pd.read_csv(filepath)
                
                if df.empty:
                    report["empty_files"] += 1
                else:
                    report["valid_files"] += 1
                    
                    # 收集列名
                    report["columns"].update(df.columns.tolist())
                    
                    # 检查日期列
                    if '日期' in df.columns and not df['日期'].empty:
                        dates = pd.to_datetime(df['日期'], errors='coerce')
                        valid_dates = dates.dropna()
                        
                        if not valid_dates.empty:
                            earliest = valid_dates.min().strftime('%Y-%m-%d')
                            latest = valid_dates.max().strftime('%Y-%m-%d')
                            
                            if report["date_range"]["earliest"] is None or earliest < report["date_range"]["earliest"]:
                                report["date_range"]["earliest"] = earliest
                            
                            if report["date_range"]["latest"] is None or latest > report["date_range"]["latest"]:
                                report["date_range"]["latest"] = latest
                                
            except Exception as e:
                logger.warning(f"验证文件 {filename} 时出错: {e}")
                report["empty_files"] += 1
        
        # 转换set为list以便JSON序列化
        report["columns"] = sorted(list(report["columns"]))
        
    except Exception as e:
        logger.error(f"数据质量验证失败: {e}")
    
    return report


def main():
    """
    主函数
    """
    logger.info("开始财务指标数据采集")
    start_time = time.time()
    
    # 加载股票列表
    stock_list = load_stock_list()
    if not stock_list:
        logger.error("股票列表为空，程序退出")
        return
    
    # 加载断点
    checkpoint = load_checkpoint()
    processed_stocks = set(checkpoint["processed_stocks"])
    current_batch = checkpoint["current_batch"]
    last_processed_index = checkpoint["last_processed_index"]
    
    logger.info(f"已处理股票数量: {len(processed_stocks)}")
    logger.info(f"当前批次: {current_batch}")
    logger.info(f"上次处理索引: {last_processed_index}")
    
    # 计算总批次数
    total_batches = (len(stock_list) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"总批次数: {total_batches}")
    
    # 处理股票
    total_success = 0
    total_failure = 0
    
    for batch_idx in range(current_batch - 1, total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(stock_list))
        
        # 如果是第一批次且不是从头开始，调整起始索引
        if batch_idx == current_batch - 1 and last_processed_index >= 0:
            start_idx = last_processed_index + 1
        
        logger.info(f"处理批次 {batch_idx + 1}/{total_batches} (股票 {start_idx+1}-{end_idx})")
        
        success_count, failure_count = process_batch(stock_list, start_idx, end_idx)
        total_success += success_count
        total_failure += failure_count
        
        # 更新已处理股票列表
        for i in range(start_idx, end_idx):
            processed_stocks.add(stock_list[i])
        
        # 更新断点
        checkpoint = {
            "processed_stocks": list(processed_stocks),
            "current_batch": batch_idx + 1,
            "last_processed_index": end_idx - 1
        }
        save_checkpoint(checkpoint)
        
        logger.info(f"批次 {batch_idx + 1} 完成，成功: {success_count}, 失败: {failure_count}")
    
    # 数据质量验证
    logger.info("开始数据质量验证")
    quality_report = validate_data_quality()
    
    # 输出统计信息
    elapsed_time = time.time() - start_time
    logger.info(f"财务指标数据采集完成")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"总股票数: {len(stock_list)}")
    logger.info(f"成功处理: {total_success}")
    logger.info(f"处理失败: {total_failure}")
    logger.info(f"总文件数: {quality_report['total_files']}")
    logger.info(f"有效文件数: {quality_report['valid_files']}")
    logger.info(f"空文件数: {quality_report['empty_files']}")
    
    if quality_report["date_range"]["earliest"] and quality_report["date_range"]["latest"]:
        logger.info(f"数据日期范围: {quality_report['date_range']['earliest']} 至 {quality_report['date_range']['latest']}")
    
    # 保存质量报告
    report_file = os.path.join(OUTPUT_DIR, "quality_report.json")
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        logger.info(f"数据质量报告已保存到 {report_file}")
    except Exception as e:
        logger.error(f"保存数据质量报告失败: {e}")


if __name__ == "__main__":
    main()