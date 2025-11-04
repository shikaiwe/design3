import os
import pandas as pd
import glob
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_data_merge_all.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 获取脚本所在目录的父目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# 配置参数
CONFIG = {
    "input_dir": "data\\financial_data",
    "output_dir": "data\\financial_data_merged",
    "merged_file": "all_financial_data.csv"
}

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger.info(f"输出目录已准备: {CONFIG['output_dir']}")

def extract_stock_code(filename):
    """从文件名中提取股票代码"""
    # 文件名格式: balance_sheet_600519.csv
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[-1].split('.')[0]
    return None

def load_financial_data():
    """加载所有财务数据文件"""
    logger.info("开始加载财务数据文件...")
    
    # 添加调试信息
    current_dir = os.getcwd()
    input_dir = os.path.abspath(CONFIG["input_dir"])
    logger.info(f"当前工作目录: {current_dir}")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输入目录是否存在: {os.path.exists(input_dir)}")
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(CONFIG["input_dir"], "*.csv"))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    # 如果没有找到文件，尝试列出目录内容
    if len(csv_files) == 0:
        logger.info("尝试列出目录内容:")
        try:
            all_files = os.listdir(input_dir)
            csv_files_in_dir = [f for f in all_files if f.endswith('.csv')]
            logger.info(f"目录中所有文件数量: {len(all_files)}")
            logger.info(f"目录中CSV文件数量: {len(csv_files_in_dir)}")
            if csv_files_in_dir:
                logger.info(f"前5个CSV文件: {csv_files_in_dir[:5]}")
        except Exception as e:
            logger.error(f"列出目录内容失败: {e}")
    
    # 按股票代码组织数据
    stock_data = {}
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        stock_code = extract_stock_code(filename)
        
        if not stock_code:
            logger.warning(f"无法从文件名提取股票代码: {filename}")
            continue
            
        # 确定报表类型
        if "balance_sheet" in filename:
            report_type = "balance_sheet"
        elif "profit_sheet" in filename:
            report_type = "profit_sheet"
        elif "cash_flow" in filename:
            report_type = "cash_flow"
        else:
            logger.warning(f"未知的报表类型: {filename}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"已加载 {filename}, 形状: {df.shape}")
            
            # 添加报表类型列
            df['report_type'] = report_type
            
            # 添加股票代码列
            df['stock_code'] = stock_code
            
            # 按股票代码组织
            if stock_code not in stock_data:
                stock_data[stock_code] = {}
                
            stock_data[stock_code][report_type] = df
            
        except Exception as e:
            logger.error(f"加载文件 {filename} 失败: {e}")
    
    logger.info(f"成功加载 {len(stock_data)} 只股票的数据")
    return stock_data

def merge_stock_reports(stock_data):
    """为每只股票合并三种报表数据"""
    logger.info("开始合并每只股票的报表数据...")
    
    merged_stocks = []
    incomplete_stocks = []
    
    for stock_code, reports in stock_data.items():
        logger.info(f"正在合并股票 {stock_code} 的数据...")
        
        # 获取三种报表数据
        balance_sheet = reports.get("balance_sheet")
        profit_sheet = reports.get("profit_sheet")
        cash_flow = reports.get("cash_flow")
        
        # 检查是否有报表数据缺失
        missing_reports = []
        if balance_sheet is None:
            missing_reports.append("资产负债表")
        if profit_sheet is None:
            missing_reports.append("利润表")
        if cash_flow is None:
            missing_reports.append("现金流量表")
            
        if missing_reports:
            logger.warning(f"股票 {stock_code} 缺少报表: {', '.join(missing_reports)}")
            incomplete_stocks.append((stock_code, missing_reports))
            
            # 即使数据不完整，也尝试合并已有的报表
            if not any([balance_sheet, profit_sheet, cash_flow]):
                logger.warning(f"股票 {stock_code} 没有任何报表数据，跳过")
                continue
        
        try:
            # 以资产负债表为基础，如果没有则以利润表为基础，再没有则以现金流量表为基础
            if balance_sheet is not None:
                merged = balance_sheet.copy()
            elif profit_sheet is not None:
                merged = profit_sheet.copy()
            elif cash_flow is not None:
                merged = cash_flow.copy()
            else:
                continue
                
            # 确保有必要的列
            if 'REPORT_DATE' not in merged.columns:
                logger.error(f"股票 {stock_code} 数据缺少REPORT_DATE列，跳过")
                continue
                
            # 添加其他报表的列
            if profit_sheet is not None:
                profit_cols = [col for col in profit_sheet.columns if col not in ['REPORT_DATE', 'REPORT_TYPE', 'report_type', 'stock_code']]
                for col in profit_cols:
                    merged[col] = None
                    
            if cash_flow is not None:
                cash_flow_cols = [col for col in cash_flow.columns if col not in ['REPORT_DATE', 'REPORT_TYPE', 'report_type', 'stock_code']]
                for col in cash_flow_cols:
                    merged[col] = None
                    
            # 为每个报告日期填充数据
            for i, row in merged.iterrows():
                report_date = row['REPORT_DATE']
                
                # 查找相同日期的利润表数据
                if profit_sheet is not None:
                    profit_match = profit_sheet[profit_sheet['REPORT_DATE'] == report_date]
                    if not profit_match.empty:
                        for col in profit_cols:
                            merged.at[i, col] = profit_match[col].iloc[0]
                
                # 查找相同日期的现金流量表数据
                if cash_flow is not None:
                    cash_flow_match = cash_flow[cash_flow['REPORT_DATE'] == report_date]
                    if not cash_flow_match.empty:
                        for col in cash_flow_cols:
                            merged.at[i, col] = cash_flow_match[col].iloc[0]
            
            # 添加报表类型信息
            merged['has_balance_sheet'] = balance_sheet is not None
            merged['has_profit_sheet'] = profit_sheet is not None
            merged['has_cash_flow'] = cash_flow is not None
            
            merged_stocks.append(merged)
            logger.info(f"成功合并股票 {stock_code} 的数据，形状: {merged.shape}")
            
        except Exception as e:
            logger.error(f"合并股票 {stock_code} 数据失败: {e}")
    
    if not merged_stocks:
        logger.error("没有成功合并任何股票数据")
        return None, incomplete_stocks
        
    # 合并所有股票数据
    logger.info("合并所有股票数据...")
    all_data = pd.concat(merged_stocks, ignore_index=True)
    
    # 将stock_code列移到第一位
    if 'stock_code' in all_data.columns:
        cols = all_data.columns.tolist()
        cols.insert(0, cols.pop(cols.index('stock_code')))
        all_data = all_data[cols]
    
    # 按股票代码从小到大，然后按日期从早到晚排序
    logger.info("对数据进行排序...")
    if 'stock_code' in all_data.columns and 'REPORT_DATE' in all_data.columns:
        # 确保REPORT_DATE是日期类型
        all_data['REPORT_DATE'] = pd.to_datetime(all_data['REPORT_DATE'])
        # 先按股票代码排序，再按日期排序
        all_data = all_data.sort_values(by=['stock_code', 'REPORT_DATE'], ascending=[True, True])
        logger.info("数据排序完成: 股票代码从小到大，日期从早到晚")
    
    logger.info(f"合并完成，总数据形状: {all_data.shape}")
    
    # 打印不完整股票信息
    if incomplete_stocks:
        logger.warning(f"共有 {len(incomplete_stocks)} 只股票数据不完整:")
        for stock_code, missing_reports in incomplete_stocks:
            logger.warning(f"  股票 {stock_code}: 缺少 {', '.join(missing_reports)}")
    
    return all_data, incomplete_stocks

def save_merged_data(df, incomplete_stocks):
    """保存合并后的数据"""
    if df is None or df.empty:
        logger.error("没有数据可保存")
        return False
        
    output_path = os.path.join(CONFIG["output_dir"], CONFIG["merged_file"])
    
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"合并数据已保存到: {output_path}")
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"列名: {df.columns.tolist()}")
        
        # 保存不完整股票列表
        if incomplete_stocks:
            incomplete_path = os.path.join(CONFIG["output_dir"], "incomplete_stocks.csv")
            incomplete_df = pd.DataFrame(incomplete_stocks, columns=['stock_code', 'missing_reports'])
            incomplete_df.to_csv(incomplete_path, index=False)
            logger.info(f"不完整股票列表已保存到: {incomplete_path}")
        
        return True
    except Exception as e:
        logger.error(f"保存合并数据失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始合并财务数据...")
    start_time = datetime.now()
    
    # 确保输出目录存在
    ensure_output_dir()
    
    # 加载所有财务数据
    stock_data = load_financial_data()
    
    # 合并每只股票的报表数据
    merged_data, incomplete_stocks = merge_stock_reports(stock_data)
    
    # 保存合并后的数据
    if merged_data is not None:
        success = save_merged_data(merged_data, incomplete_stocks)
        if success:
            logger.info("财务数据合并完成")
        else:
            logger.error("财务数据合并失败")
    else:
        logger.error("没有数据可合并")
    
    end_time = datetime.now()
    logger.info(f"总耗时: {end_time - start_time}")

if __name__ == "__main__":
    main()