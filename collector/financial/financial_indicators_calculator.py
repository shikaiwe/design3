import pandas as pd
import numpy as np
import os
from datetime import datetime

class FinancialIndicatorCalculator:
    """
    财务指标计算器
    基于合并后的财务数据计算各种财务指标
    """
    
    def __init__(self, data_file_path):
        """
        初始化财务指标计算器
        
        参数:
        data_file_path: 合并后的财务数据CSV文件路径
        """
        self.data_file_path = data_file_path
        self.data = None
        self.indicators = None
        
    def load_data(self):
        """加载财务数据"""
        try:
            self.data = pd.read_csv(self.data_file_path)
            print(f"成功加载数据，共{len(self.data)}行记录")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def preprocess_data(self):
        """预处理数据，处理缺失值和异常值"""
        if self.data is None:
            print("数据未加载，请先调用load_data()")
            return False
            
        # 将缺失值替换为0
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns] = self.data[numeric_columns].fillna(0)
        
        # 确保日期列格式正确
        self.data['REPORT_DATE'] = pd.to_datetime(self.data['REPORT_DATE'])
        
        # 按股票代码和报告日期排序
        self.data = self.data.sort_values(['stock_code', 'REPORT_DATE'])
        
        print("数据预处理完成")
        return True
    
    def calculate_profitability_indicators(self):
        """
        计算盈利能力指标
        包括：ROE(净资产收益率)、ROA(总资产收益率)、毛利率、净利率等
        """
        if self.data is None:
            print("数据未加载，请先调用load_data()")
            return False
            
        # 计算ROE (净资产收益率) = 净利润 / 股东权益
        # 使用PARENT_NETPROFIT(归母净利润)和TOTAL_PARENT_EQUITY(归母股东权益)
        self.data['ROE'] = self.data.apply(
            lambda row: row['PARENT_NETPROFIT'] / row['TOTAL_PARENT_EQUITY'] 
            if row['TOTAL_PARENT_EQUITY'] != 0 else np.nan, axis=1)
        
        # 计算ROA (总资产收益率) = 净利润 / 总资产
        self.data['ROA'] = self.data.apply(
            lambda row: row['NETPROFIT'] / row['TOTAL_ASSETS'] 
            if row['TOTAL_ASSETS'] != 0 else np.nan, axis=1)
        
        # 计算毛利率 = (营业收入 - 营业成本) / 营业收入
        # 使用OPERATE_INCOME(营业收入)和OPERATE_COST(营业成本)
        self.data['GROSS_PROFIT_MARGIN'] = self.data.apply(
            lambda row: (row['OPERATE_INCOME'] - row['OPERATE_COST']) / row['OPERATE_INCOME'] 
            if row['OPERATE_INCOME'] != 0 else np.nan, axis=1)
        
        # 计算净利率 = 净利润 / 营业收入
        self.data['NET_PROFIT_MARGIN'] = self.data.apply(
            lambda row: row['NETPROFIT'] / row['OPERATE_INCOME'] 
            if row['OPERATE_INCOME'] != 0 else np.nan, axis=1)
        
        # 计算销售费用率 = 销售费用 / 营业收入
        self.data['SALE_EXPENSE_RATIO'] = self.data.apply(
            lambda row: row['SALE_EXPENSE'] / row['OPERATE_INCOME'] 
            if row['OPERATE_INCOME'] != 0 else np.nan, axis=1)
        
        # 计算管理费用率 = 管理费用 / 营业收入
        self.data['MANAGE_EXPENSE_RATIO'] = self.data.apply(
            lambda row: row['MANAGE_EXPENSE'] / row['OPERATE_INCOME'] 
            if row['OPERATE_INCOME'] != 0 else np.nan, axis=1)
        
        print("盈利能力指标计算完成")
        return True
    
    def calculate_solvency_indicators(self):
        """
        计算偿债能力指标
        包括：资产负债率、流动比率、速动比率等
        """
        if self.data is None:
            print("数据未加载，请先调用load_data()")
            return False
            
        # 计算资产负债率 = 总负债 / 总资产
        self.data['DEBT_TO_ASSET_RATIO'] = self.data.apply(
            lambda row: row['TOTAL_LIABILITIES'] / row['TOTAL_ASSETS'] 
            if row['TOTAL_ASSETS'] != 0 else np.nan, axis=1)
        
        # 计算流动比率 = 流动资产 / 流动负债
        self.data['CURRENT_RATIO'] = self.data.apply(
            lambda row: row['TOTAL_CURRENT_ASSETS'] / row['TOTAL_CURRENT_LIAB'] 
            if row['TOTAL_CURRENT_LIAB'] != 0 else np.nan, axis=1)
        
        # 计算速动比率 = (流动资产 - 存货) / 流动负债
        self.data['QUICK_RATIO'] = self.data.apply(
            lambda row: (row['TOTAL_CURRENT_ASSETS'] - row['INVENTORY']) / row['TOTAL_CURRENT_LIAB'] 
            if row['TOTAL_CURRENT_LIAB'] != 0 else np.nan, axis=1)
        
        # 计算权益乘数 = 总资产 / 股东权益
        self.data['EQUITY_MULTIPLIER'] = self.data.apply(
            lambda row: row['TOTAL_ASSETS'] / row['TOTAL_EQUITY'] 
            if row['TOTAL_EQUITY'] != 0 else np.nan, axis=1)
        
        print("偿债能力指标计算完成")
        return True
    
    def calculate_operational_indicators(self):
        """
        计算运营能力指标
        包括：存货周转率、应收账款周转率、总资产周转率等
        """
        if self.data is None:
            print("数据未加载，请先调用load_data()")
            return False
            
        # 计算存货周转率 = 营业成本 / 平均存货
        # 由于缺少历史数据，这里使用当前存货代替平均存货
        self.data['INVENTORY_TURNOVER'] = self.data.apply(
            lambda row: row['OPERATE_COST'] / row['INVENTORY'] 
            if row['INVENTORY'] != 0 else np.nan, axis=1)
        
        # 计算应收账款周转率 = 营业收入 / 平均应收账款
        # 使用ACCOUNTS_RECE(应收账款)
        self.data['RECEIVABLE_TURNOVER'] = self.data.apply(
            lambda row: row['OPERATE_INCOME'] / row['ACCOUNTS_RECE'] 
            if row['ACCOUNTS_RECE'] != 0 else np.nan, axis=1)
        
        # 计算总资产周转率 = 营业收入 / 平均总资产
        # 使用当前总资产代替平均总资产
        self.data['ASSET_TURNOVER'] = self.data.apply(
            lambda row: row['OPERATE_INCOME'] / row['TOTAL_ASSETS'] 
            if row['TOTAL_ASSETS'] != 0 else np.nan, axis=1)
        
        print("运营能力指标计算完成")
        return True
    
    def calculate_growth_indicators(self):
        """
        计算成长能力指标
        包括：营收增长率、净利润增长率、总资产增长率等
        """
        if self.data is None:
            print("数据未加载，请先调用load_data()")
            return False
            
        # 按股票代码分组，计算同比增长率
        grouped = self.data.groupby('stock_code')
        
        # 计算营收增长率 = (本期营收 - 上期营收) / 上期营收
        self.data['REVENUE_GROWTH'] = grouped['OPERATE_INCOME'].pct_change() * 100
        
        # 计算净利润增长率 = (本期净利润 - 上期净利润) / 上期净利润
        self.data['NETPROFIT_GROWTH'] = grouped['NETPROFIT'].pct_change() * 100
        
        # 计算总资产增长率 = (本期总资产 - 上期总资产) / 上期总资产
        self.data['ASSET_GROWTH'] = grouped['TOTAL_ASSETS'].pct_change() * 100
        
        # 计算净资产增长率 = (本期净资产 - 上期净资产) / 上期净资产
        self.data['EQUITY_GROWTH'] = grouped['TOTAL_EQUITY'].pct_change() * 100
        
        print("成长能力指标计算完成")
        return True
    
    def calculate_all_indicators(self):
        """计算所有财务指标"""
        if not self.load_data():
            return False
            
        if not self.preprocess_data():
            return False
            
        # 计算各类财务指标
        self.calculate_profitability_indicators()
        self.calculate_solvency_indicators()
        self.calculate_operational_indicators()
        self.calculate_growth_indicators()
        
        # 选择需要保留的列
        indicator_columns = [
            'stock_code', 'REPORT_DATE', 'REPORT_TYPE',
            # 盈利能力指标
            'ROE', 'ROA', 'GROSS_PROFIT_MARGIN', 'NET_PROFIT_MARGIN', 
            'SALE_EXPENSE_RATIO', 'MANAGE_EXPENSE_RATIO',
            # 偿债能力指标
            'DEBT_TO_ASSET_RATIO', 'CURRENT_RATIO', 'QUICK_RATIO', 'EQUITY_MULTIPLIER',
            # 运营能力指标
            'INVENTORY_TURNOVER', 'RECEIVABLE_TURNOVER', 'ASSET_TURNOVER',
            # 成长能力指标
            'REVENUE_GROWTH', 'NETPROFIT_GROWTH', 'ASSET_GROWTH', 'EQUITY_GROWTH'
        ]
        
        self.indicators = self.data[indicator_columns].copy()
        
        print("所有财务指标计算完成")
        return True
    
    def save_indicators(self, output_path):
        """保存计算结果到CSV文件"""
        if self.indicators is None:
            print("指标未计算，请先调用calculate_all_indicators()")
            return False
            
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存数据
            self.indicators.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"财务指标已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def get_stock_indicators(self, stock_code):
        """获取指定股票的财务指标"""
        if self.indicators is None:
            print("指标未计算，请先调用calculate_all_indicators()")
            return None
            
        return self.indicators[self.indicators['stock_code'] == stock_code]
    
    def get_latest_indicators(self, stock_code):
        """获取指定股票的最新财务指标"""
        stock_data = self.get_stock_indicators(stock_code)
        if stock_data is None or len(stock_data) == 0:
            return None
            
        # 按日期排序，获取最新数据
        stock_data = stock_data.sort_values('REPORT_DATE')
        return stock_data.iloc[-1:]

def main():
    """主函数"""
    # 数据文件路径
    data_file = "c:\\Users\\Administrator\\Desktop\\design3\\data\\financial_data_merged\\all_financial_data.csv"
    
    # 输出文件路径
    output_file = "c:\\Users\\Administrator\\Desktop\\design3\\data\\financial_indicators\\financial_indicators.csv"
    
    # 创建财务指标计算器
    calculator = FinancialIndicatorCalculator(data_file)
    
    # 计算所有财务指标
    if calculator.calculate_all_indicators():
        # 保存结果
        calculator.save_indicators(output_file)
        
        # 示例：获取特定股票的最新指标
        stock_code = "000001"
        latest_indicators = calculator.get_latest_indicators(stock_code)
        if latest_indicators is not None:
            print(f"\n股票 {stock_code} 的最新财务指标:")
            print(latest_indicators.to_string())

if __name__ == "__main__":
    main()