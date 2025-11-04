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
        logging.FileHandler('price_data_merge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 获取脚本所在目录的父目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # 需要上两级目录

# 配置参数
CONFIG = {
    "input_dir": os.path.join(PARENT_DIR, "data", "daliy_prices"),
    "output_dir": os.path.join(PARENT_DIR, "data", "daliy_prices_merged"),
    "merged_file": "all_price_data.csv"
}

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger.info(f"输出目录已准备: {CONFIG['output_dir']}")

def extract_stock_code(filename):
    """从文件名中提取股票代码"""
    # 文件名格式: price_000001.csv
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[-1].split('.')[0]
    return None

def load_price_data():
    """加载所有价格数据文件"""
    logger.info("开始加载价格数据文件...")
    
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
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"已加载 {filename}, 形状: {df.shape}")
            
            # 添加股票代码列
            df['stock_code'] = stock_code
            
            # 按股票代码组织
            stock_data[stock_code] = df
            
        except Exception as e:
            logger.error(f"加载文件 {filename} 失败: {e}")
    
    logger.info(f"成功加载 {len(stock_data)} 只股票的数据")
    return stock_data

def merge_price_data(stock_data):
    """合并所有股票的价格数据"""
    logger.info("开始合并所有股票的价格数据...")
    
    if not stock_data:
        logger.error("没有股票数据可合并")
        return None
    
    # 合并所有股票数据
    logger.info("合并所有股票数据...")
    all_data = pd.concat(stock_data.values(), ignore_index=True)
    
    # 确保日期列是日期类型
    if '日期' in all_data.columns:
        all_data['日期'] = pd.to_datetime(all_data['日期'])
    elif 'date' in all_data.columns:
        all_data['date'] = pd.to_datetime(all_data['date'])
    
    # 将stock_code列移到第一位
    if 'stock_code' in all_data.columns:
        cols = all_data.columns.tolist()
        cols.insert(0, cols.pop(cols.index('stock_code')))
        all_data = all_data[cols]
    
    # 按股票代码从小到大，然后按日期从早到晚排序
    logger.info("对数据进行排序...")
    if 'stock_code' in all_data.columns:
        date_col = '日期' if '日期' in all_data.columns else 'date'
        if date_col in all_data.columns:
            # 先按股票代码排序，再按日期排序
            all_data = all_data.sort_values(by=['stock_code', date_col], ascending=[True, True])
            logger.info("数据排序完成: 股票代码从小到大，日期从早到晚")
    
    logger.info(f"合并完成，总数据形状: {all_data.shape}")
    logger.info(f"列名: {all_data.columns.tolist()}")
    
    return all_data

def save_merged_data(df):
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
        
        # 保存股票列表
        if 'stock_code' in df.columns:
            stock_list_path = os.path.join(CONFIG["output_dir"], "stock_list.csv")
            stock_list = df['stock_code'].unique()
            stock_df = pd.DataFrame({'stock_code': stock_list})
            stock_df.to_csv(stock_list_path, index=False)
            logger.info(f"股票列表已保存到: {stock_list_path}")
        
        return True
    except Exception as e:
        logger.error(f"保存合并数据失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始合并价格数据...")
    start_time = datetime.now()
    
    # 确保输出目录存在
    ensure_output_dir()
    
    # 加载所有价格数据
    stock_data = load_price_data()
    
    # 合并所有股票的价格数据
    merged_data = merge_price_data(stock_data)
    
    # 保存合并后的数据
    if merged_data is not None:
        success = save_merged_data(merged_data)
        if success:
            logger.info("价格数据合并完成")
        else:
            logger.error("价格数据合并失败")
    else:
        logger.error("没有数据可合并")
    
    end_time = datetime.now()
    logger.info(f"总耗时: {end_time - start_time}")

if __name__ == "__main__":
    main()