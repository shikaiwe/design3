# 因子数据分析报告

## 概述

本报告基于 `c:\Users\Administrator\Desktop\design3\data\factors\factors_data.csv` 文件，对金融量化分析中使用的各类因子进行了系统性的分类和整理。

**数据统计：**
- 总因子数量：276个
- 分类数量：9个大类
- 数据来源：factors_data.csv

## 因子分类详情

### 1. 基础科目及衍生类因子 (basics) - 37个因子

**类别介绍：** 基础科目及衍生类因子，主要涵盖财务报表基础科目和衍生计算指标

| 因子名称 | 因子说明 |
|---------|---------|
| administration_expense_ttm | 管理费用TTM |
| asset_impairment_loss_ttm | 资产减值损失TTM |
| cash_flow_to_price_ratio | 现金流市值比 |
| circulating_market_cap | 流通市值 |
| EBIT | 息税前利润 |
| EBITDA | 息税折旧摊销前利润 |
| financial_assets | 金融资产 |
| financial_expense_ttm | 财务费用TTM |
| financial_liability | 金融负债 |
| goods_sale_and_service_render_cash_ttm | 销售商品提供劳务收到的现金 |
| gross_profit_ttm | 毛利TTM |
| interest_carry_current_liability | 带息流动负债 |
| interest_free_current_liability | 无息流动负债 |
| market_cap | 市值 |
| net_debt | 净债务 |
| net_finance_cash_flow_ttm | 筹资活动现金流量净额TTM |
| net_interest_expense | 净利息费用 |
| net_invest_cash_flow_ttm | 投资活动现金流量净额TTM |
| net_operate_cash_flow_ttm | 经营活动现金流量净额TTM |
| net_profit_ttm | 净利润TTM |
| net_working_capital | 净运营资本 |
| non_operating_net_profit_ttm | 营业外收支净额TTM |
| non_recurring_gain_loss | 非经常性损益 |
| np_parent_company_owners_ttm | 归属于母公司股东的净利润TTM |
| OperateNetIncome | 经营活动净收益 |
| operating_assets | 经营性资产 |
| operating_cost_ttm | 营业成本TTM |
| operating_liability | 经营性负债 |
| operating_profit_ttm | 营业利润TTM |
| operating_revenue_ttm | 营业收入TTM |
| retained_earnings | 留存收益 |
| sales_to_price_ratio | 营收市值比 |
| sale_expense_ttm | 销售费用TTM |
| total_operating_cost_ttm | 营业总成本TTM |
| total_operating_revenue_ttm | 营业总收入TTM |
| total_profit_ttm | 利润总额TTM |
| value_change_profit_ttm | 价值变动净收益TTM |

### 2. 情绪类因子 (emotion) - 35个因子

**类别介绍：** 情绪类因子，主要反映市场情绪和交易活跃度

| 因子名称 | 因子说明 |
|---------|---------|
| AR | 人气指标 |
| ARBR | ARBR |
| ATR14 | 14日均幅指标 |
| ATR6 | 6日均幅指标 |
| BR | 意愿指标 |
| DAVOL10 | 10日平均换手率与120日平均换手率之比 |
| DAVOL20 | 20日平均换手率与120日平均换手率之比 |
| DAVOL5 | 5日平均换手率与120日平均换手率 |
| MAWVAD | 因子WVAD的6日均值 |
| money_flow_20 | 20日资金流量 |
| PSY | 心理线指标 |
| turnover_volatility | 换手率相对波动率 |
| TVMA20 | 20日成交金额的移动平均值 |
| TVMA6 | 6日成交金额的移动平均值 |
| TVSTD20 | 20日成交金额的标准差 |
| TVSTD6 | 6日成交金额的标准差 |
| VDEA | 计算VMACD因子的中间变量 |
| VDIFF | 计算VMACD因子的中间变量 |
| VEMA10 | 成交量的10日指数移动平均 |
| VEMA12 | 12日成交量的移动平均值 |
| VEMA26 | 成交量的26日指数移动平均 |
| VEMA5 | 成交量的5日指数移动平均 |
| VMACD | 成交量指数平滑异同移动平均线 |
| VOL10 | 10日平均换手率 |
| VOL120 | 120日平均换手率 |
| VOL20 | 20日平均换手率 |
| VOL240 | 240日平均换手率 |
| VOL5 | 5日平均换手率 |
| VOL60 | 60日平均换手率 |
| VOSC | 成交量震荡 |
| VR | 成交量比率（Volume Ratio） |
| VROC12 | 12日量变动速率指标 |
| VROC6 | 6日量变动速率指标 |
| VSTD10 | 10日成交量标准差 |
| VSTD20 | 20日成交量标准差 |
| WVAD | 威廉变异离散量 |

### 3. 成长类因子 (growth) - 9个因子

**类别介绍：** 成长类因子，主要衡量企业成长性和发展潜力

| 因子名称 | 因子说明 |
|---------|---------|
| financing_cash_growth_rate | 筹资活动产生的现金流量净额增长率 |
| net_asset_growth_rate | 净资产增长率 |
| net_operate_cashflow_growth_rate | 经营活动产生的现金流量净额增长率 |
| net_profit_growth_rate | 净利润增长率 |
| np_parent_company_owners_growth_rate | 归属母公司股东的净利润增长率 |
| operating_revenue_growth_rate | 营业收入增长率 |
| PEG | PEG |
| total_asset_growth_rate | 总资产增长率 |
| total_profit_growth_rate | 利润总额增长率 |

### 4. 动量类因子 (momentum) - 32个因子

**类别介绍：** 动量类因子，主要反映价格趋势和动量效应

| 因子名称 | 因子说明 |
|---------|---------|
| arron_down_25 | Aroon指标下轨 |
| arron_up_25 | Aroon指标上轨 |
| BBIC | BBI 动量 |
| bear_power | 空头力道 |
| BIAS10 | 10日乖离率 |
| BIAS20 | 20日乖离率 |
| BIAS5 | 5日乖离率 |
| BIAS60 | 60日乖离率 |
| bull_power | 多头力道 |
| CCI10 | 10日顺势指标 |
| CCI15 | 15日顺势指标 |
| CCI20 | 20日顺势指标 |
| CCI88 | 88日顺势指标 |
| CR20 | CR指标 |
| fifty_two_week_close_rank | 当前价格处于过去1年股价的位置 |
| MASS | 梅斯线 |
| PLRC12 | 12日收盘价格与日期线性回归系数 |
| PLRC24 | 24日收盘价格与日期线性回归系数 |
| PLRC6 | 6日收盘价格与日期线性回归系数 |
| Price1M | 当前股价除以过去一个月股价均值再减1 |
| Price1Y | 当前股价除以过去一年股价均值再减1 |
| Price3M | 当前股价除以过去三个月股价均值再减1 |
| Rank1M | 1减去 过去一个月收益率排名与股票总数的比值 |
| ROC12 | 12日变动速率（Price Rate of Change） |
| ROC120 | 120日变动速率（Price Rate of Change） |
| ROC20 | 20日变动速率（Price Rate of Change） |
| ROC6 | 6日变动速率（Price Rate of Change） |
| ROC60 | 60日变动速率（Price Rate of Change） |
| single_day_VPT | 单日价量趋势 |
| single_day_VPT_12 | 单日价量趋势12均值 |
| single_day_VPT_6 | 单日价量趋势6日均值 |
| TRIX10 | 10日终极指标TRIX |
| TRIX5 | 5日终极指标TRIX |
| Volume1M | 当前交易量相比过去1个月日均交易量 与过去过去20日日均收益率乘积 |

### 5. 每股指标因子 (pershare) - 15个因子

**类别介绍：** 每股指标因子，主要基于每股数据进行计算

| 因子名称 | 因子说明 |
|---------|---------|
| capital_reserve_fund_per_share | 每股资本公积金 |
| cashflow_per_share_ttm | 每股现金流量净额，根据当时日期来获取最近变更日的总股本 |
| cash_and_equivalents_per_share | 每股现金及现金等价物余额 |
| eps_ttm | 每股收益TTM |
| net_asset_per_share | 每股净资产 |
| net_operate_cash_flow_per_share | 每股经营活动产生的现金流量净额 |
| operating_profit_per_share | 每股营业利润 |
| operating_profit_per_share_ttm | 每股营业利润TTM |
| operating_revenue_per_share | 每股营业收入 |
| operating_revenue_per_share_ttm | 每股营业收入TTM |
| retained_earnings_per_share | 每股留存收益 |
| retained_profit_per_share | 每股未分配利润 |
| surplus_reserve_fund_per_share | 每股盈余公积金 |
| total_operating_revenue_per_share | 每股营业总收入 |
| total_operating_revenue_per_share_ttm | 每股营业总收入TTM |

### 6. 质量类因子 (quality) - 68个因子

**类别介绍：** 质量类因子，主要衡量企业经营质量和财务健康状况

| 因子名称 | 因子说明 |
|---------|---------|
| ACCA | 现金流资产比和资产回报率之差 |
| accounts_payable_turnover_days | 应付账款周转天数 |
| accounts_payable_turnover_rate | 应付账款周转率 |
| account_receivable_turnover_days | 应收账款周转天数 |
| account_receivable_turnover_rate | 应收账款周转率 |
| adjusted_profit_to_total_profit | 扣除非经常损益后的净利润/净利润 |
| admin_expense_rate | 管理费用与营业总收入之比 |
| asset_turnover_ttm | 经营资产周转率TTM |
| cash_rate_of_sales | 经营活动产生的现金流量净额与营业收入之比 |
| cash_to_current_liability | 现金比率 |
| cfo_to_ev | 经营活动产生的现金流量净额与企业价值之比TTM |
| current_asset_turnover_rate | 流动资产周转率TTM |
| current_ratio | 流动比率(单季度) |
| debt_to_asset_ratio | 债务总资产比 |
| debt_to_equity_ratio | 产权比率 |
| debt_to_tangible_equity_ratio | 有形净值债务率 |
| DEGM | 毛利率增长 |
| DEGM_8y | 长期毛利率增长 |
| DSRI | 应收账款指数 |
| equity_to_asset_ratio | 股东权益比率 |
| equity_to_fixed_asset_ratio | 股东权益与固定资产比率 |
| equity_turnover_rate | 股东权益周转率 |
| financial_expense_rate | 财务费用与营业总收入之比 |
| fixed_assets_turnover_rate | 固定资产周转率 |
| fixed_asset_ratio | 固定资产比率 |
| GMI | 毛利率指数 |
| goods_service_cash_to_operating_revenue_ttm | 销售商品提供劳务收到的现金与营业收入之比 |
| gross_income_ratio | 销售毛利率 |
| intangible_asset_ratio | 无形资产比率 |
| inventory_turnover_days | 存货周转天数 |
| inventory_turnover_rate | 存货周转率 |
| invest_income_associates_to_total_profit | 对联营和合营公司投资收益/利润总额 |
| long_debt_to_asset_ratio | 长期借款与资产总计之比 |
| long_debt_to_working_capital_ratio | 长期负债与营运资金比率 |
| long_term_debt_to_asset_ratio | 长期负债与资产总计之比 |
| LVGI | 财务杠杆指数 |
| margin_stability | 盈利能力稳定性 |
| maximum_margin | 最大盈利水平 |
| MLEV | 市场杠杆 |
| net_non_operating_income_to_total_profit | 营业外收支利润净额/利润总额 |
| net_operate_cash_flow_to_asset | 总资产现金回收率 |
| net_operate_cash_flow_to_net_debt | 经营活动产生现金流量净额/净债务 |
| net_operate_cash_flow_to_operate_income | 经营活动产生的现金流量净额与经营活动净收益之比 |
| net_operate_cash_flow_to_total_current_liability | 现金流动负债比 |
| net_operate_cash_flow_to_total_liability | 经营活动产生的现金流量净额/负债合计 |
| net_operating_cash_flow_coverage | 净利润现金含量 |
| net_profit_ratio | 销售净利率 |
| net_profit_to_total_operate_revenue_ttm | 净利润与营业总收入之比 |
| non_current_asset_ratio | 非流动资产比率 |
| OperatingCycle | 营业周期 |
| operating_cost_to_operating_revenue_ratio | 销售成本率 |
| operating_profit_growth_rate | 营业利润增长率 |
| operating_profit_ratio | 营业利润率 |
| operating_profit_to_operating_revenue | 营业利润与营业总收入之比 |
| operating_profit_to_total_profit | 经营活动净收益/利润总额 |
| operating_tax_to_operating_revenue_ratio_ttm | 销售税金率 |
| profit_margin_ttm | 销售利润率TTM |
| quick_ratio | 速动比率 |
| rnoa_ttm | 经营资产回报率TTM |
| ROAEBITTTM | 总资产报酬率 |
| roa_ttm | 资产回报率TTM |
| roa_ttm_8y | 长期资产回报率TTM |
| roe_ttm | 权益回报率TTM |
| roe_ttm_8y | 长期权益回报率TTM |
| roic_ttm | 投资资本回报率TTM |
| sale_expense_to_operating_revenue | 营业费用与营业总收入之比 |
| SGAI | 销售管理费用指数 |
| SGI | 营业收入指数 |
| super_quick_ratio | 超速动比率 |
| total_asset_turnover_rate | 总资产周转率 |
| total_profit_to_cost_ratio | 成本费用利润率 |

### 7. 风险类因子 (risk) - 12个因子

**类别介绍：** 风险类因子，主要衡量投资风险和收益分布特征

| 因子名称 | 因子说明 |
|---------|---------|
| Kurtosis120 | 个股收益的120日峰度 |
| Kurtosis20 | 个股收益的20日峰度 |
| Kurtosis60 | 个股收益的60日峰度 |
| sharpe_ratio_120 | 120日夏普比率 |
| sharpe_ratio_20 | 20日夏普比率 |
| sharpe_ratio_60 | 60日夏普比率 |
| Skewness120 | 个股收益的120日偏度 |
| Skewness20 | 个股收益的20日偏度 |
| Skewness60 | 个股收益的60日偏度 |
| Variance120 | 120日收益方差 |
| Variance20 | 20日收益方差 |
| Variance60 | 60日收益方差 |

### 8. 风格因子 (style) - 28个因子

**类别介绍：** 风险因子 - 风格因子，主要反映投资风格和市场特征

| 因子名称 | 因子说明 |
|---------|---------|
| average_share_turnover_annual | 年度平均月换手率 |
| average_share_turnover_quarterly | 季度平均平均月换手率 |
| beta | BETA |
| book_leverage | 账面杠杆 |
| book_to_price_ratio | 市净率因子 |
| cash_earnings_to_price_ratio | 现金流量市值比 |
| cube_of_size | 市值立方因子 |
| cumulative_range | 收益离差 |
| daily_standard_deviation | 日收益率标准差 |
| debt_to_assets | 资产负债率 |
| earnings_growth | 5年盈利增长率 |
| earnings_to_price_ratio | 利润市值比 |
| earnings_yield | 盈利预期因子 |
| growth | 成长因子 |
| historical_sigma | 残差历史波动率 |
| leverage | 杠杆因子 |
| liquidity | 流动性因子 |
| long_term_predicted_earnings_growth | 预期长期盈利增长率 |
| market_leverage | 市场杠杆 |
| momentum | 动量因子 |
| natural_log_of_market_cap | 对数总市值 |
| non_linear_size | 非线性市值因子 |
| predicted_earnings_to_price_ratio | 预期市盈率 |
| raw_beta | RAW BETA |
| relative_strength | 相对强弱 |
| residual_volatility | 残差波动因子 |
| sales_growth | 5年营业收入增长率 |
| share_turnover_monthly | 月换手率 |
| short_term_predicted_earnings_growth | 预期短期盈利增长率 |
| size | 市值因子 |

### 9. 新风格因子 (style_pro) - 18个因子

**类别介绍：** 风险因子 - 新风格因子，新一代风格因子体系

| 因子名称 | 因子说明 |
|---------|---------|
| btop | 市净率因子 |
| divyild | 分红因子 |
| earnqlty | 盈利质量因子 |
| earnvar | 盈利变动率因子 |
| earnyild | 收益因子 |
| financial_leverage | 财务杠杆因子 |
| invsqlty | 投资能力因子 |
| liquidty | 流动性因子 |
| long_growth | 长期成长因子 |
| ltrevrsl | 长期反转因子 |
| market_beta | 市场波动率因子 |
| market_size | 市值规模因子 |
| midcap | 中等市值因子 |
| profit | 盈利能力因子 |
| relative_momentum | 相对动量因子 |
| resvol | 残余波动率因子 |

### 10. 技术指标因子 (technical) - 17个因子

**类别介绍：** 技术指标因子，主要基于技术分析指标

| 因子名称 | 因子说明 |
|---------|---------|
| boll_down | 下轨线（布林线）指标 |
| boll_up | 上轨线（布林线）指标 |
| EMA5 | 5日指数移动均线 |
| EMAC10 | 10日指数移动均线 |
| EMAC12 | 12日指数移动均线 |
| EMAC120 | 120日指数移动均线 |
| EMAC20 | 20日指数移动均线 |
| EMAC26 | 26日指数移动均线 |
| MAC10 | 10日移动均线 |
| MAC120 | 120日移动均线 |
| MAC20 | 20日移动均线 |
| MAC5 | 5日移动均线 |
| MAC60 | 60日移动均线 |
| MACDC | 平滑异同移动平均线 |
| MFI14 | 资金流量指标 |
| price_no_fq | 不复权价格因子 |

## 因子分布统计

| 类别 | 因子数量 | 占比 |
|------|---------|------|
| 基础科目及衍生类因子 | 37 | 13.41% |
| 情绪类因子 | 35 | 12.68% |
| 成长类因子 | 9 | 3.26% |
| 动量类因子 | 32 | 11.59% |
| 每股指标因子 | 15 | 5.43% |
| 质量类因子 | 68 | 24.64% |
| 风险类因子 | 12 | 4.35% |
| 风格因子 | 28 | 10.14% |
| 新风格因子 | 18 | 6.52% |
| 技术指标因子 | 17 | 6.16% |
| **总计** | **276** | **100%** |

## 数据需求分析

基于以上因子分类，需要收集的数据条目主要包括：

### 基础数据需求
1. **财务报表数据**：利润表、资产负债表、现金流量表相关科目
2. **市场交易数据**：价格、成交量、换手率等
3. **技术指标数据**：移动平均线、布林线、MACD等
4. **风险指标数据**：收益率、方差、偏度、峰度等

### 数据频率需求
- 日度数据：技术指标、情绪类因子
- 月度数据：财务指标、风格因子
- 季度数据：财务报表相关因子
- 年度数据：长期趋势指标

### 数据来源建议
1. 财务数据：Wind、聚宽、Tushare等金融数据平台
2. 市场数据：交易所公开数据、第三方数据提供商
3. 技术指标：可通过基础数据计算得出

---

**生成时间：** 2025年
**数据来源：** `c:\Users\Administrator\Desktop\design3\data\factors\factors_data.csv`
**文件版本：** 1.0