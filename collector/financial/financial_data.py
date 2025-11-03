import akshare as ak
import pandas as pd
import os
import json
import time
from datetime import datetime
import traceback

# 获取脚本所在目录的父目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# 配置参数
CONFIG = {
    "stock_list_file": os.path.join(PARENT_DIR, "data", "components", "stock_index_cut.csv"),
    "output_dir": os.path.join(PARENT_DIR, "data", "financial_data"),
    "batch_size": 10,  # 每批处理的股票数量
    "retry_times": 3,  # 请求失败重试次数
    "retry_interval": 5,  # 重试间隔(秒)
    "request_interval": 1,  # 请求间隔(秒)
    "checkpoint_file": "checkpoint.json"  # 断点续传文件
}

# 需要获取的财务数据类型
FINANCIAL_REPORTS = {
    "balance_sheet": {
        "function": ak.stock_balance_sheet_by_report_em,
        "filename": "balance_sheet_{symbol}.csv"
    },
    "profit_sheet": {
        "function": ak.stock_profit_sheet_by_report_em,
        "filename": "profit_sheet_{symbol}.csv"
    },
    "cash_flow": {
        "function": ak.stock_cash_flow_sheet_by_report_em,
        "filename": "cash_flow_{symbol}.csv"
    }
}

# 需要保留的核心字段
CORE_COLUMNS = {
    "balance_sheet": [
        "TOTAL_ASSETS", "TOTAL_CURRENT_ASSETS", "TOTAL_NONCURRENT_ASSETS",
        "TOTAL_LIABILITIES", "TOTAL_CURRENT_LIAB", "TOTAL_NONCURRENT_LIAB",
        "TOTAL_EQUITY", "TOTAL_PARENT_EQUITY", "MONETARYFUNDS", "ACCOUNTS_RECE",
        "INVENTORY", "FIXED_ASSET", "INTANGIBLE_ASSET", "GOODWILL",
        "SHARE_CAPITAL", "CAPITAL_RESERVE", "SURPLUS_RESERVE", "UNASSIGN_RPOFIT"
    ],
    "profit_sheet": [
        "TOTAL_OPERATE_INCOME", "OPERATE_INCOME", "OPERATE_COST",
        "SALE_EXPENSE", "MANAGE_EXPENSE", "FINANCE_EXPENSE", "RESEARCH_EXPENSE",
        "ASSET_IMPAIRMENT_LOSS", "OPERATE_PROFIT", "TOTAL_PROFIT",
        "NETPROFIT", "PARENT_NETPROFIT"
    ],
    "cash_flow": [
        "NETCASH_OPERATE", "NETCASH_INVEST", "NETCASH_FINANCE",
        "SALES_SERVICES", "BUY_SERVICES"
    ]
}

def load_checkpoint():
    """加载断点信息"""
    if os.path.exists(CONFIG["checkpoint_file"]):
        try:
            with open(CONFIG["checkpoint_file"], "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载断点文件失败: {e}")
    return {"completed_stocks": [], "current_batch": 0, "last_processed_index": -1}

def save_checkpoint(checkpoint):
    """保存断点信息"""
    try:
        with open(CONFIG["checkpoint_file"], "w") as f:
            json.dump(checkpoint, f, indent=4)
        print("断点信息已保存")
    except Exception as e:
        print(f"保存断点文件失败: {e}")

def load_stock_list():
    """加载股票列表并格式化股票代码"""
    try:
        df = pd.read_csv(CONFIG["stock_list_file"])
        # 确保股票代码是6位数字字符串
        df['品种代码'] = df['品种代码'].astype(str).str.zfill(6)
        # 直接使用已处理的股票代码，不再进行格式化转换
        stock_codes = df['品种代码'].tolist()
        print(f"成功加载股票列表，共 {len(stock_codes)} 只股票")
        return stock_codes
    except Exception as e:
        print(f"加载股票列表失败: {e}")
        raise

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"输出目录: {os.path.abspath(CONFIG['output_dir'])}")

def format_symbol(symbol):
    """格式化股票代码为akshare需要的格式"""
    # 确保股票代码是6位数字字符串
    symbol = str(symbol).zfill(6)
    # 根据股票代码添加交易所前缀
    if symbol.startswith("6"):
        return f"SH{symbol}"
    elif symbol.startswith(("0", "3")):
        return f"SZ{symbol}"
    else:
        return symbol

def fetch_financial_data(symbol, report_type, retry_times=CONFIG["retry_times"]):
    """获取单只股票的财务数据"""
    func = FINANCIAL_REPORTS[report_type]["function"]
    
    # 格式化股票代码为akshare需要的格式
    formatted_symbol = format_symbol(symbol)
    
    for attempt in range(retry_times):
        try:
            print(f"正在获取 {symbol} ({report_type}) 数据，尝试次数: {attempt + 1}")
            df = func(symbol=formatted_symbol)
            
            if df is None or df.empty:
                print(f"{symbol} ({report_type}) 数据为空")
                return None
                
            # 只保留核心字段
            core_cols = [col for col in CORE_COLUMNS[report_type] if col in df.columns]
            if core_cols:
                df = df[["REPORT_DATE", "REPORT_TYPE"] + core_cols]
            
            # 过滤日期，只保留2019-01-01到2024-12-31之间的数据
            if "REPORT_DATE" in df.columns:
                df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"])
                start_date = pd.to_datetime("2019-01-01")
                end_date = pd.to_datetime("2024-12-31")
                df = df[(df["REPORT_DATE"] >= start_date) & (df["REPORT_DATE"] <= end_date)]
                if df.empty:
                    print(f"{symbol} ({report_type}) 在指定日期范围内无数据")
                    return None
            
            return df
            
        except Exception as e:
            print(f"获取 {symbol} ({report_type}) 数据失败 (尝试 {attempt + 1}/{retry_times}): {e}")
            if attempt < retry_times - 1:
                time.sleep(CONFIG["retry_interval"])
            else:
                print(f"获取 {symbol} ({report_type}) 数据最终失败，跳过此股票")
                return None

def save_financial_data(df, symbol, report_type):
    """保存财务数据到文件"""
    if df is None or df.empty:
        return False
    
    # 使用原始股票代码作为文件名
    filename = FINANCIAL_REPORTS[report_type]["filename"].format(symbol=symbol)
    filepath = os.path.join(CONFIG["output_dir"], filename)
    
    try:
        df.to_csv(filepath, index=False)
        print(f"已保存 {symbol} ({report_type}) 数据到 {filepath}")
        return True
    except Exception as e:
        print(f"保存 {symbol} ({report_type}) 数据失败: {e}")
        return False

def process_stock(symbol):
    """处理单只股票的所有财务数据"""
    success = True
    for report_type in FINANCIAL_REPORTS.keys():
        df = fetch_financial_data(symbol, report_type)
        if not save_financial_data(df, symbol, report_type):
            success = False
    
    return success

def process_batch(stock_batch, checkpoint):
    """处理一批股票"""
    batch_success = True
    completed_stocks = checkpoint["completed_stocks"]
    
    for symbol in stock_batch:
        if symbol in completed_stocks:
            print(f"跳过已处理的股票: {symbol}")
            continue
            
        print(f"开始处理股票: {symbol}")
        success = process_stock(symbol)
        
        if success:
            completed_stocks.append(symbol)
            checkpoint["completed_stocks"] = completed_stocks
            save_checkpoint(checkpoint)
            print(f"成功处理股票: {symbol}")
        else:
            print(f"处理股票 {symbol} 失败，停止运行")
            batch_success = False
            break
            
        # 请求间隔
        time.sleep(CONFIG["request_interval"])
    
    return batch_success

def main():
    """主函数"""
    print("开始财务数据采集")
    
    # 确保输出目录存在
    ensure_output_dir()
    
    # 加载股票列表
    stock_list = load_stock_list()
    
    # 加载断点信息
    checkpoint = load_checkpoint()
    if checkpoint is None:
        checkpoint = {"completed_stocks": [], "current_batch": 0, "last_processed_index": -1}
    
    # 计算起始批次
    start_index = checkpoint["last_processed_index"] + 1
    total_batches = (len(stock_list) + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
    
    print(f"总共 {len(stock_list)} 只股票，分为 {total_batches} 批处理")
    print(f"从第 {start_index // CONFIG['batch_size'] + 1} 批开始处理")
    
    # 分批处理股票
    for i in range(start_index, len(stock_list), CONFIG["batch_size"]):
        batch_num = i // CONFIG["batch_size"] + 1
        stock_batch = stock_list[i:i + CONFIG["batch_size"]]
        
        print(f"开始处理第 {batch_num}/{total_batches} 批，股票数量: {len(stock_batch)}")
        
        # 更新断点信息
        checkpoint["current_batch"] = batch_num
        checkpoint["last_processed_index"] = i + len(stock_batch) - 1
        save_checkpoint(checkpoint)
        
        # 处理当前批次
        batch_success = process_batch(stock_batch, checkpoint)
        
        if not batch_success:
            print(f"第 {batch_num} 批处理失败，程序终止")
            return
        
        print(f"第 {batch_num} 批处理完成")
    
    print("所有股票数据处理完成")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"程序运行出错: {e}")
        print(traceback.format_exc())
