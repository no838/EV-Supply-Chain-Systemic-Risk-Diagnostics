#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from datetime import datetime

# ========== Gini函数 ==========
def gini(array):
    array = np.sort(np.abs(array))
    n = len(array)
    cumvals = np.cumsum(array)
    return (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n if cumvals[-1] != 0 else 0

# ========== 路径 ==========
INPUT_PATH = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/outputs_dwrc/primaryValue/network_role_dwrc_shock_all_params.csv"
OUTPUT_DIR = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/历年逐月/maps_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 数据加载 ==========
df = pd.read_csv(INPUT_PATH)
df = df[~df["country"].str.contains("_X")]
df["date"] = pd.to_datetime(df["ym"], format="%Y-%m")
# ========== Z-score 标准化 ==========
df["DWRC_z"] = df.groupby("date")["DWRC"].transform(lambda x: (x - x.mean()) / x.std())

# ========== 多阈值极端比例统计 ==========
thresholds = [1.0, 1.5, 1.96, 2.5]
summary = []
for thresh in thresholds:
    extreme_count = df.groupby("date").apply(lambda x: (np.abs(x["DWRC_z"]) > thresh).sum())
    total_count = df.groupby("date")["country"].count()
    ratio = (extreme_count / total_count).rename(f"extreme_ratio_{thresh}")
    summary.append(ratio)
extreme_df = pd.concat(summary, axis=1).reset_index()

# ========== Gini + Kurtosis ==========
gini_series = df.groupby("date")["DWRC_z"].apply(gini).rename("gini")
kurt_series = df.groupby("date")["DWRC_z"].apply(lambda x: kurtosis(x, fisher=False)).rename("kurtosis")
gk_df = pd.concat([gini_series, kurt_series], axis=1).reset_index()

# ========== 系统性风险图 + 事件标注 ==========
plt.figure(figsize=(12, 6))
for thresh in thresholds:
    plt.plot(extreme_df["date"], extreme_df[f"extreme_ratio_{thresh}"], label=f"Z > {thresh}")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Extreme Node Ratio")
plt.title("Systemic Risk: Proportion of Extreme DWRC Nodes (Z-score)")

# 🎯 关键事件标注
events = {
    "2016-01": "US Election", 
    "2018-07": "Trade War Start",
    "2020-03": "COVID Shock",
    "2022-02": "Russia–Ukraine"
}
for edate, label in events.items():
    d = pd.to_datetime(edate)
    plt.axvline(d, color="gray", linestyle="--", alpha=0.5)
    plt.text(d, plt.ylim()[1]*0.95, label, rotation=90, verticalalignment='top', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "extreme_ratio_by_thresholds_annotated.png"))
plt.close()

# ========== Gini + Kurtosis ==========
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axs[0].plot(gk_df["date"], gk_df["gini"], color="orange")
axs[0].set_title("Gini Coefficient of DWRC_z")
axs[0].set_ylabel("Gini")

axs[1].plot(gk_df["date"], gk_df["kurtosis"], color="purple")
axs[1].set_title("Kurtosis of DWRC_z")
axs[1].set_ylabel("Kurtosis")

plt.suptitle("Inequality and Extremity in DWRC_z over Time", fontsize=14)
plt.xlabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gini_kurtosis_dwrc_z.png"))
plt.close()

# ========== Segment 热力图 ==========
# ========== Segment 热力图（年度刻度 & 指定顺序）==========
heat_df = df.copy()
heat_df["abs_DWRC_z"] = np.abs(heat_df["DWRC_z"])

# 上下游顺序：从原材料到下游整车/后端
segment_order = [
    "Raw_Ore",
    "Rare_Earth",
    "Li_Material",
    "Adv_Material",
    "Battery_Module",
    "Power_Electronics",
    "Electric_Motor",
    "Machinery",
    "EV_Final"
]

# 计算每月各部门平均 |DWRC_z|
seg_month_avg = (
    heat_df
    .groupby(["date", "segment"])["abs_DWRC_z"]
    .mean()
    .unstack()                    # 行：date 列：segment
    .reindex(columns=segment_order)  # 强制指定列顺序
)

plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    seg_month_avg.T,
    cmap="YlOrRd",
    cbar_kws={"label": "|DWRC_z|"},
    linewidths=0.5,
    linecolor="white"
)

# 只显示每年1月的刻度
dates = seg_month_avg.index
# 找到所有 1 月 的行号
jan_idx = [i for i, d in enumerate(dates) if d.month == 1]
# 刻度位置（格子中间）
xticks = [i + 0.5 for i in jan_idx]
# 刻度标签
xticklabels = [dates[i].strftime("%Y") for i in jan_idx]

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=0, ha="center", fontsize=10)

# 美化
ax.set_title("Segment-Level Average |DWRC_z| Over Time", fontsize=14, weight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Segment (Upstream → Downstream)", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "segment_heatmap_dwrc_z.png"), dpi=300)
plt.close()
# ========== 国家风险排名 ==========
risk_max = df.groupby("country")["DWRC_z"].apply(lambda x: np.max(np.abs(x)))
top_risk = risk_max.sort_values(ascending=False).head(20)
top_risk.to_csv(os.path.join(OUTPUT_DIR, "top20_country_risk_peak.csv"))

# ========== 阈值漂移趋势分析 ==========
plt.figure(figsize=(12, 5))
for thresh in thresholds:
    plt.plot(extreme_df["date"], extreme_df[f"extreme_ratio_{thresh}"], label=f"Z > {thresh}")
plt.xlabel("Date")
plt.ylabel("Proportion")
plt.title("Early-Warning Signal: Extreme Ratio Drift (DWRC_z)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "extreme_ratio_drift.png"))
plt.close()

print("✅ 所有图表和分析已完成并保存在：", OUTPUT_DIR)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import matplotlib.dates as mdates
import seaborn as sns
import os
import matplotlib.font_manager as fm
from datetime import datetime

# ======= 配置路径和字体 =======
SAVE_DIR = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/历年逐月/maps_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# 设置中文字体（如果需要）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于中文显示
# plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ======= 事件库定义 =======
EVENTS = [
    # Policy / Finance
    {"month":"2010-08","desc":"China rare-earth export restrictions","category":"policy"},
    {"month":"2013-01","desc":"EU ETS carbon price crash & reform debates","category":"policy"},
    {"month":"2014-01","desc":"Indonesia bans exports of unprocessed nickel ore (first ban)","category":"policy"},
    {"month":"2014-11","desc":"OPEC no-cut decision—oil price collapse starts","category":"policy"},
    {"month":"2015-08","desc":"China RMB '8·11' FX regime adjustment","category":"policy"},
    {"month":"2016-06","desc":"UK 'Brexit' referendum","category":"policy"},
    {"month":"2017-03","desc":"US withdrawal from TPP","category":"policy"},
    {"month":"2018-03","desc":"US Section 232 steel/aluminum tariffs","category":"policy"},
    {"month":"2018-07","desc":"US–China tariffs (List 1)","category":"policy"},
    {"month":"2018-09","desc":"US–China tariffs (List 3 escalation)","category":"policy"},
    {"month":"2019-05","desc":"US threatens tariffs on Mexico (migration)","category":"policy"},
    {"month":"2020-01","desc":"Indonesia reinstates nickel ore export ban","category":"policy"},
    {"month":"2022-06","desc":"UFLPA takes effect (solar polysilicon)","category":"policy"},
    {"month":"2022-08","desc":"US Inflation Reduction Act (IRA) enacted—EV credits","category":"policy"},
    {"month":"2023-04","desc":"US Treasury mineral & battery component guidance","category":"policy"},
    {"month":"2023-06","desc":"Indonesia bauxite export ban takes effect","category":"policy"},
    {"month":"2023-08","desc":"EU Battery Regulation enters into force","category":"policy"},
    {"month":"2023-08","desc":"China export controls on gallium & germanium","category":"policy"},
    {"month":"2023-09","desc":"EU anti-subsidy probe into Chinese BEV imports","category":"policy"},
    {"month":"2023-10","desc":"EU CBAM transitional phase starts","category":"policy"},
    {"month":"2023-12","desc":"China export controls on graphite products","category":"policy"},
    # Logistics / Transport
    {"month":"2015-08","desc":"Tianjin port explosions","category":"logistics"},
    {"month":"2020-03","desc":"COVID-19 mobility collapse","category":"logistics"},
    {"month":"2021-03","desc":"Suez Canal blockage","category":"logistics"},
    {"month":"2021-10","desc":"LA/LB port congestion peaks","category":"logistics"},
    {"month":"2022-04","desc":"Shanghai city-wide lockdown","category":"logistics"},
    {"month":"2023-07","desc":"Canada BC ports strike","category":"logistics"},
    {"month":"2023-11","desc":"Panama Canal drought restrictions","category":"logistics"},
    {"month":"2023-11","desc":"Red Sea/Houthi risk","category":"logistics"},
    # Natural disasters
    {"month":"2011-03","desc":"Tōhoku earthquake + Thai floods","category":"disaster"},
    {"month":"2016-04","desc":"Kumamoto earthquake","category":"disaster"},
    {"month":"2017-08","desc":"Hurricane Harvey","category":"disaster"},
    {"month":"2021-02","desc":"Texas winter storm","category":"disaster"},
    {"month":"2022-08","desc":"Sichuan drought & power rationing","category":"disaster"}
]

# 事件类别颜色映射
EVENT_COLORS = {
    "policy": "#D62728",  # 红色 - 政策事件
    "logistics": "#1F77B4",  # 蓝色 - 物流事件
    "disaster": "#2CA02C"   # 绿色 - 灾害事件
}

# ======= 1. 创建模拟数据（如果实际数据不可用） =======
# 注意：在实际应用中应使用真实数据，此处仅为演示
# def generate_synthetic_sri_data():
#     # 创建2010-2024年的时间范围（按月）
#     date_range = pd.date_range(start='2010-01-01', end='2024-12-31', freq='M')
    
#     # 创建随机但有一定趋势的SRI数据
#     np.random.seed(42)
#     base = np.zeros(len(date_range))
    
#     # 添加趋势和季节性
#     for i in range(len(date_range)):
#         # 长期趋势 - 先升后降
#         trend = 0.7 * (1 - abs(i/len(date_range) - 0.4))
        
#         # 季节性波动
#         season = 0.3 * np.sin(2 * np.pi * i/12)
        
#         # 随机噪声
#         noise = 0.2 * np.random.randn()
        
#         # 重大事件影响（位置根据实际事件调整）
#         event_impact = 0
#         if date_range[i].year == 2018 and date_range[i].month == 3:
#             event_impact = 0.8  # 中美贸易战
#         if date_range[i].year == 2020 and date_range[i].month == 3:
#             event_impact = 1.0  # 疫情冲击
#         if date_range[i].year == 2022 and date_range[i].month == 8:
#             event_impact = 0.7  # IRA法案
        
#         base[i] = trend + season + noise + event_impact
    
#     # 标准化
#     base = (base - base.mean()) / base.std()
    
#     # 确保在-0.75到1.25之间
#     min_val, max_val = base.min(), base.max()
#     base = -0.75 + 2.0 * (base - min_val) / (max_val - min_val)
    
#     return pd.DataFrame({
#         'date': date_range,
#         'SRI': base
#     }).set_index('date')

# # 生成模拟SRI数据
# sri_df = generate_synthetic_sri_data()

# ======= 2. 分析事件对SRI的影响 =======
# ======= 2. 分析事件对SRI的影响 =======
def analyze_events_impact(events, sri_df):
    event_records = []
    sri_months = sri_df.index.strftime('%Y-%m')  # 获取所有月份
    
    for event in events:
        event_month = event["month"]
        
        if event_month in sri_months:
            # 获取该月最后一天
            event_date_in_df = sri_df.index[sri_df.index.strftime('%Y-%m') == event_month][0]
            
            # 事件发生时的SRI值
            event_sri = sri_df.loc[event_date_in_df, "SRI"]
            
            # 事件前3个月的平均值
            prev_index = sri_df.index.get_loc(event_date_in_df)
            pre_dates = sri_df.index[max(0, prev_index-3):prev_index]  # 确保不越界
            pre_avg = sri_df.loc[pre_dates, "SRI"].mean()
            
            # 事件影响程度
            impact = event_sri - pre_avg
            
            # 事件后波动性计算
            post_dates = sri_df.index[prev_index+1:min(len(sri_df), prev_index+7)]  # 后6个月
            post_std = sri_df.loc[post_dates, "SRI"].std() if len(post_dates) > 1 else np.nan
                
            event_records.append({
                "event_date": event_month,
                "event_date_in_df": event_date_in_df,  # 记录实际匹配的日期
                "description": event["desc"],
                "category": event["category"],
                "sri_at_event": event_sri,
                "pre_event_avg": pre_avg,
                "impact": impact,
                "post_volatility": post_std,
                "impact_level": "high" if abs(impact) > 0.5 else "medium" if abs(impact) > 0.3 else "low"
            })
    
    return pd.DataFrame(event_records)

# 创建事件影响分析表
events_impact_df = analyze_events_impact(EVENTS, sri_df)

# ======= 3. 绘制专业SRI图表（带事件标记） =======
def create_sri_chart_with_events(sri_df, events_impact_df, events=None):   
    """
    创建专业SRI图表（带事件标记），使用events_impact_df中的事件数据
    """
    # 设置图表大小和样式
    plt.figure(figsize=(16, 8), facecolor='#f0f0f0')
    ax = plt.gca()
    ax.set_facecolor('#ffffff')
    
    # 绘制SRI折线图
    ax.plot(sri_df.index, sri_df['SRI'], 
            color='#D62728',  # 红色折线
            linewidth=3, 
            alpha=0.9,
            label='Systemic Risk Index')
    
    # 设置坐标轴范围
    ax.set_ylim(-0.75, 1.25)
    ax.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2024-12-31'))
    
    # 设置网格线
    ax.grid(True, linestyle='-', linewidth=0.5, color='#ffffff', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='#e0e0e0', alpha=0.5)
    
    # 设置X轴（年份刻度）
    years = mdates.YearLocator()   # 每年一个刻度
    years_fmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # 设置Y轴
    ax.set_yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.set_yticklabels(['-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75', '1.00', '1.25'])
    
    # 设置标题和标签
    plt.title('Systemic Risk Index (SRI) — Composite of Gini, Kurtosis, and Extreme Ratio', 
              fontsize=18, pad=20, fontweight='bold')
    plt.ylabel('Standardized Composite SRI', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=14, fontweight='bold')
    
    # 添加事件标记
    max_sri = sri_df['SRI'].max()
    min_sri = sri_df['SRI'].min()
    
    # 使用events_impact_df中的事件数据
    for idx, row in events_impact_df.iterrows():
        event_date = row['event_date_in_df']  # 确保使用匹配的日期
        event_sri = row['sri_at_event']
        category = row['category']
        description = row['description']
        
        # 设置事件线条的颜色和位置
        color = EVENT_COLORS.get(category, "#000000")
        
        # 添加垂直线
        plt.axvline(x=event_date, color=color, linestyle='--', alpha=0.8, linewidth=1.2)
        
        # 处理描述文本
        desc = description.split("—")[0]  # 使用主要描述部分
        if len(desc) > 25:
            desc = desc[:25] + "..."  # 截断长文本
        
        # 计算标签位置（避免重叠）
        angle = 70
        va = 'center'
        ha = 'center'
        text_y = event_sri + 0.1 if event_sri < 0.6 else event_sri - 0.1
        text_offset = 0
        rotation = 0
        
        # 特殊处理密集区域
        if event_date.year == 2018:
            if "tariffs" in description:
                text_y = max_sri - 0.1
                va = 'top'
                rotation = 30
            elif "Mexico" in description:
                text_y = min_sri + 0.1
                va = 'bottom'
                rotation = -30
        elif event_date.year == 2020 and event_date.month == 3:  # 特殊处理COVID事件
            text_y = max_sri - 0.1
            va = 'top'
            rotation = 30
        
        # 添加文本标签
        plt.text(event_date, text_y + text_offset, desc, 
                 fontsize=9, rotation=rotation, va=va, ha=ha,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.2'))
        
        # 在SRI线上添加标记点
        plt.plot(event_date, event_sri, 'o', markersize=8, 
                 markeredgecolor=color, markerfacecolor='white', markeredgewidth=1.5)
    
    # 添加图例说明
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=EVENT_COLORS["policy"], lw=2, label='Policy Events'),
        Line2D([0], [0], color=EVENT_COLORS["logistics"], lw=2, label='Logistics Events'),
        Line2D([0], [0], color=EVENT_COLORS["disaster"], lw=2, label='Disaster Events')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # 添加图表来源说明
    plt.figtext(0.95, 0.01, 'Source: GVC Systemic Risk Model', 
                horizontalalignment='right', fontsize=10)
    
    # 优化布局
    plt.tight_layout()
    
    return plt

# ======= 4. 创建图表 =======
chart = create_sri_chart_with_events(sri_df, events_impact_df, EVENTS)  # 传递第三个参数

# ======= 5. 保存图表 =======
chart.savefig(os.path.join(SAVE_DIR, "sri_with_events.png"), dpi=300)
plt.close()

# ======= 6. 创建事件影响分析表格 =======
def create_events_impact_table(df):
    """创建格式精美的事件影响分析表格"""
    # 创建表格HTML
    table_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Systemic Risk Event Impact Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .high-impact { background-color: #ffcccc; font-weight: bold; }
        .medium-impact { background-color: #ffe0b3; }
        .low-impact { background-color: #e6f7ff; }
    </style>
    </head>
    <body>
        <h2>Systemic Risk Index (SRI) Event Impact Analysis</h2>
        <p>The impact is calculated as: (SRI at event date) - (3-month average prior to event)</p>
        <table>
            <tr>
                <th>Date</th>
                <th>Description</th>
                <th>Category</th>
                <th>SRI at Event</th>
                <th>Pre-Event Average</th>
                <th>Impact (Δ)</th>
                <th>Impact Level</th>
            </tr>
    """
    
    # 添加表格行
    for _, row in df.iterrows():
        # 设置影响等级样式
        impact_class = ""
        if row['impact'] > 0.5 or row['impact'] < -0.5:
            impact_class = "high-impact"
        elif row['impact'] > 0.3 or row['impact'] < -0.3:
            impact_class = "medium-impact"
        else:
            impact_class = "low-impact"
        
        table_html += f"""
        <tr class="{impact_class}">
            <td>{row['event_date']}</td>
            <td>{row['description']}</td>
            <td><span style="color: {EVENT_COLORS.get(row['category'], '#000000')};">{row['category'].title()}</span></td>
            <td>{row['sri_at_event']:.3f}</td>
            <td>{row['pre_event_avg']:.3f}</td>
            <td><b>{row['impact']:.3f}</b></td>
            <td>{row['impact_level'].title()}</td>
        </tr>
        """
    
    # 结束表格
    table_html += """
        </table>
        <p><i>Generated on {}</i></p>
    </body>
    </html>
    """.format(datetime.now().strftime("%Y-%m-%d"))
     
    return table_html

# 创建并保存事件影响分析表格（HTML）
events_table_html = create_events_impact_table(events_impact_df)
with open(os.path.join(SAVE_DIR, "events_impact_analysis.html"), "w") as f:
    f.write(events_table_html)

# ======= 7. 事件影响统计 =======
def create_event_impact_summary(df):
    """创建事件影响统计分析图"""
    # 按类别分组
    cat_impacts = df.groupby('category')['impact'].apply(list).reset_index()
    
    # 创建箱线图
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [cat_impacts.loc[cat_impacts['category'] == 'policy', 'impact'].iloc[0],
         cat_impacts.loc[cat_impacts['category'] == 'logistics', 'impact'].iloc[0],
         cat_impacts.loc[cat_impacts['category'] == 'disaster', 'impact'].iloc[0]], 
        showmeans=True,
        patch_artist=True,
        boxprops=dict(facecolor='#f8f8f8', color='black'),
        medianprops=dict(color='red', linewidth=2),
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='green')
    )
    
    # 设置标签和标题
    plt.xticks([1, 2, 3], ['Policy', 'Logistics', 'Disaster'])
    plt.ylabel('Impact Magnitude')
    plt.title('Event Impact Distribution by Category')
    plt.grid(axis='y', alpha=0.7)
    
    # 添加注释
    plt.text(0.5, 0.95, 'Positive Impacts', transform=plt.gca().transAxes, color='red')
    plt.text(0.5, 0.05, 'Negative Impacts', transform=plt.gca().transAxes, color='blue')
    
    return plt

# 创建并保存事件影响统计分析图
impact_summary = create_event_impact_summary(events_impact_df)
impact_summary.savefig(os.path.join(SAVE_DIR, "event_impact_summary.png"), dpi=300)
plt.close()

# ======= 8. 保存所有数据 =======
sri_df.to_csv(os.path.join(SAVE_DIR, "sri_index.csv"))
events_impact_df.to_csv(os.path.join(SAVE_DIR, "events_impact_analysis.csv"), index=False)

print("✅ 所有图表与数据已保存至:", SAVE_DIR)
print("1. SRI图表 (带事件标记): sri_with_events.png")
print("2. 事件影响分析: events_impact_analysis.html")
print("3. 事件影响统计分析: event_impact_summary.png")
print("4. SRI数据: sri_index.csv")
print("5. 事件影响分析数据: events_impact_analysis.csv")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
import os
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict

# ======= 配置路径 =======
SAVE_DIR = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/历年逐月/maps_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======= 加载真实数据 =======
INPUT_PATH = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/outputs_dwrc/primaryValue/network_role_dwrc_shock_all_params.csv"
df = pd.read_csv(INPUT_PATH)

# 预处理数据
df = df[~df["country"].str.contains("_X")]  # 移除特殊国家
df["date"] = pd.to_datetime(df["ym"], format="%Y-%m")  # 创建日期列
df["DWRC_z"] = df.groupby("date")["DWRC"].transform(lambda x: (x - x.mean()) / x.std())  # 标准化DWRC

# ======= 计算系统性风险指标 (SRI) =======
def calculate_sri(df):
    """基于真实数据计算系统性风险指标 (SRI)"""
    # 计算每个时间点的基尼系数
    gini = df.groupby("date")["DWRC_z"].apply(lambda x: gini_coefficient(np.abs(x)))
    
    # 计算每个时间点的峰度
    kurt = df.groupby("date")["DWRC_z"].apply(kurtosis, fisher=False)
    
    # 计算极端比例 (超过不同阈值的比例)
    thresholds = [1.0, 1.5, 1.96, 2.5]
    extreme_ratios = []
    for thresh in thresholds:
        ratio = df.groupby("date")["DWRC_z"].apply(lambda x: (np.abs(x) > thresh).mean())
        extreme_ratios.append(ratio)
    
    # 组合指标
    sri_df = pd.DataFrame({
        "gini": gini,
        "kurtosis": kurt,
        "extreme_ratio_avg": pd.concat(extreme_ratios, axis=1).mean(axis=1)
    })
    
    # 标准化每个指标
    sri_df["gini_z"] = (sri_df["gini"] - sri_df["gini"].mean()) / sri_df["gini"].std()
    sri_df["kurtosis_z"] = (sri_df["kurtosis"] - sri_df["kurtosis"].mean()) / sri_df["kurtosis"].std()
    sri_df["extreme_ratio_z"] = (sri_df["extreme_ratio_avg"] - sri_df["extreme_ratio_avg"].mean()) / sri_df["extreme_ratio_avg"].std()
    
    # 计算综合SRI
    sri_df["SRI"] = (sri_df["gini_z"] + sri_df["kurtosis_z"] + sri_df["extreme_ratio_z"]) / 3
    
    return sri_df

# 基尼系数计算函数
def gini_coefficient(array):
    """计算基尼系数"""
    array = np.sort(np.abs(array))
    n = len(array)
    cumvals = np.cumsum(array)
    return (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n if cumvals[-1] != 0 else 0

# 计算SRI
sri_df = calculate_sri(df)

# ======= 事件库定义 =======
EVENTS = [
    # Policy / Finance
    {"month":"2010-08","desc":"China rare-earth export restrictions","category":"policy"},
    {"month":"2013-01","desc":"EU ETS carbon price crash & reform debates","category":"policy"},
    {"month":"2014-01","desc":"Indonesia bans exports of unprocessed nickel ore (first ban)","category":"policy"},
    {"month":"2014-11","desc":"OPEC no-cut decision—oil price collapse starts","category":"policy"},
    {"month":"2015-08","desc":"China RMB '8·11' FX regime adjustment","category":"policy"},
    {"month":"2016-06","desc":"UK 'Brexit' referendum","category":"policy"},
    {"month":"2017-03","desc":"US withdrawal from TPP","category":"policy"},
    {"month":"2018-03","desc":"US Section 232 steel/aluminum tariffs","category":"policy"},
    {"month":"2018-07","desc":"US–China tariffs (List 1)","category":"policy"},
    {"month":"2018-09","desc":"US–China tariffs (List 3 escalation)","category":"policy"},
    {"month":"2019-05","desc":"US threatens tariffs on Mexico (migration)","category":"policy"},
    {"month":"2020-01","desc":"Indonesia reinstates nickel ore export ban","category":"policy"},
    {"month":"2022-06","desc":"UFLPA takes effect (solar polysilicon)","category":"policy"},
    {"month":"2022-08","desc":"US Inflation Reduction Act (IRA) enacted—EV credits","category":"policy"},
    {"month":"2023-04","desc":"US Treasury mineral & battery component guidance","category":"policy"},
    {"month":"2023-06","desc":"Indonesia bauxite export ban takes effect","category":"policy"},
    {"month":"2023-08","desc":"EU Battery Regulation enters into force","category":"policy"},
    {"month":"2023-08","desc":"China export controls on gallium & germanium","category":"policy"},
    {"month":"2023-09","desc":"EU anti-subsidy probe into Chinese BEV imports","category":"policy"},
    {"month":"2023-10","desc":"EU CBAM transitional phase starts","category":"policy"},
    {"month":"2023-12","desc":"China export controls on graphite products","category":"policy"},
    # Logistics / Transport
    {"month":"2015-08","desc":"Tianjin port explosions","category":"logistics"},
    {"month":"2020-03","desc":"COVID-19 mobility collapse","category":"logistics"},
    {"month":"2021-03","desc":"Suez Canal blockage","category":"logistics"},
    {"month":"2021-10","desc":"LA/LB port congestion peaks","category":"logistics"},
    {"month":"2022-04","desc":"Shanghai city-wide lockdown","category":"logistics"},
    {"month":"2023-07","desc":"Canada BC ports strike","category":"logistics"},
    {"month":"2023-11","desc":"Panama Canal drought restrictions","category":"logistics"},
    {"month":"2023-11","desc":"Red Sea/Houthi risk","category":"logistics"},
    # Natural disasters
    {"month":"2011-03","desc":"Tōhoku earthquake + Thai floods","category":"disaster"},
    {"month":"2016-04","desc":"Kumamoto earthquake","category":"disaster"},
    {"month":"2017-08","desc":"Hurricane Harvey","category":"disaster"},
    {"month":"2021-02","desc":"Texas winter storm","category":"disaster"},
    {"month":"2022-08","desc":"Sichuan drought & power rationing","category":"disaster"}
]

# 事件类别颜色映射
EVENT_COLORS = {
    "policy": "#D62728",  # 红色 - 政策事件
    "logistics": "#1F77B4",  # 蓝色 - 物流事件
    "disaster": "#2CA02C"   # 绿色 - 灾害事件
}

# ======= 分析事件对SRI的影响 =======
def analyze_events_impact(events, sri_df):
    """分析事件对SRI的影响"""
    event_records = []
    sri_df["month_str"] = sri_df.index.strftime("%Y-%m")
    
    for event in events:
        event_month = event["month"]
        
        if event_month in sri_df["month_str"].values:
            event_date = sri_df[sri_df["month_str"] == event_month].index[0]
            event_sri = sri_df.loc[event_date, "SRI"]
            
            # 找到事件发生前的3个月
            prev_dates = sri_df.index[sri_df.index < event_date][-3:]
            if len(prev_dates) > 0:
                pre_avg = sri_df.loc[prev_dates, "SRI"].mean()
            else:
                pre_avg = np.nan
            
            impact = event_sri - pre_avg if not np.isnan(pre_avg) else np.nan
            
            # 找到事件发生后的6个月
            post_dates = sri_df.index[sri_df.index > event_date][:6]
            if len(post_dates) > 1:
                post_std = sri_df.loc[post_dates, "SRI"].std()
            else:
                post_std = np.nan
                
            event_records.append({
                "event_date": event_month,
                "event_date_in_df": event_date,
                "description": event["desc"],
                "category": event["category"],
                "sri_at_event": event_sri,
                "pre_event_avg": pre_avg,
                "impact": impact,
                "post_volatility": post_std,
                "impact_level": "high" if abs(impact) > 0.5 else "medium" if abs(impact) > 0.3 else "low"
            })
    
    return pd.DataFrame(event_records)

# 创建事件影响分析表
events_impact_df = analyze_events_impact(EVENTS, sri_df)

# ======= 绘制专业SRI图表 =======
def create_sri_chart_with_events(sri_df, events_impact_df):   
    plt.figure(figsize=(16, 8), facecolor='#f0f0f0')
    ax = plt.gca()
    ax.set_facecolor('#ffffff')
    
    # 绘制SRI时间序列
    ax.plot(sri_df.index, sri_df['SRI'], 
            color='#D62728',
            linewidth=3, 
            alpha=0.9,
            label='Systemic Risk Index (SRI)')
    
    # 设置Y轴范围
    sri_min, sri_max = sri_df['SRI'].min(), sri_df['SRI'].max()
    buffer = (sri_max - sri_min) * 0.1
    ax.set_ylim(sri_min - buffer, sri_max + buffer)
    
    # 设置X轴范围
    ax.set_xlim(sri_df.index.min(), sri_df.index.max())
    
    # 网格线设置
    ax.grid(True, linestyle='-', linewidth=0.5, color='#ffffff', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='#e0e0e0', alpha=0.5)
    
    # 设置X轴格式
    years = mdates.YearLocator()
    years_fmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # 标题和标签
    plt.title('Systemic Risk Index (SRI) — Based on Real DWRC Data', 
              fontsize=18, pad=20, fontweight='bold')
    plt.ylabel('SRI Value', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=14, fontweight='bold')
    
    # 添加事件标记
    for idx, row in events_impact_df.iterrows():
        event_date = row['event_date_in_df']
        event_sri = row['sri_at_event']
        category = row['category']
        description = row['description']
        
        color = EVENT_COLORS.get(category, "#000000")
        plt.axvline(x=event_date, color=color, linestyle='--', alpha=0.8, linewidth=1.2)
        
        # 简化描述文本
        desc = description.split("—")[0]
        if len(desc) > 25:
            desc = desc[:25] + "..."
        
        # 计算文本位置
        text_y = event_sri + (sri_max - sri_min) * 0.05
        rotation = 0
        
        # 特殊处理某些事件的位置
        if "COVID" in description:
            text_y = sri_max - (sri_max - sri_min) * 0.1
            rotation = 45
        elif "Shanghai" in description:
            text_y = sri_min + (sri_max - sri_min) * 0.1
            rotation = -45
        
        plt.text(event_date, text_y, desc, 
                 fontsize=9, rotation=rotation, va='center', ha='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.2'))
        plt.plot(event_date, event_sri, 'o', markersize=8, 
                 markeredgecolor=color, markerfacecolor='white', markeredgewidth=1.5)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=EVENT_COLORS["policy"], lw=2, label='Policy Events'),
        Line2D([0], [0], color=EVENT_COLORS["logistics"], lw=2, label='Logistics Events'),
        Line2D([0], [0], color=EVENT_COLORS["disaster"], lw=2, label='Disaster Events')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # 添加数据来源说明
    plt.figtext(0.95, 0.01, 'Source: Real DWRC Data Analysis', 
                horizontalalignment='right', fontsize=10)
    
    plt.tight_layout()
    return plt

# 创建图表
chart = create_sri_chart_with_events(sri_df, events_impact_df)

# 保存图表
chart.savefig(os.path.join(SAVE_DIR, "sri_with_events_real.png"), dpi=300)
plt.close()

# ======= 事件相关国家与部门分析 =======
def plot_event_entities(events_impact_df, df, events):
    """分析事件相关国家和部门"""
    # 筛选出事件月份的数据
    event_months = [e["month"] for e in events]
    event_data = df[df["ym"].isin(event_months)].copy()
    event_data["abs_DWRC_z"] = np.abs(event_data["DWRC_z"])
    
    # 创建图表
    plt.figure(figsize=(15, 18))
    
    # 国家风险峰值热力图
    plt.subplot(3, 1, 1)
    country_risk = event_data.groupby(["country", "ym"])["abs_DWRC_z"].max().unstack().fillna(0)
    
    # 选择风险最高的20个国家
    country_risk_total = country_risk.sum(axis=1)
    top_countries = country_risk_total.sort_values(ascending=False).head(20).index
    country_risk = country_risk.loc[top_countries]
    
    sns.heatmap(country_risk, cmap="Reds", cbar_kws={"label": "Max |DWRC_z|"})
    plt.title("Top 20 Countries with Highest Risk Peaks during Events")
    plt.ylabel("Country")
    plt.xlabel("Event Month")
    
    # 部门平均冲击热力图
    plt.subplot(3, 1, 2)
    segment_impact = event_data.groupby(["segment", "ym"])["abs_DWRC_z"].mean().unstack().fillna(0)
    sns.heatmap(segment_impact, cmap="Oranges", cbar_kws={"label": "Mean |DWRC_z|"})
    plt.title("Sector-Level Average Risk Impact during Events")
    plt.ylabel("Sector")
    plt.xlabel("Event Month")
    
    # 国家-部门组合热力图
    plt.subplot(3, 1, 3)
    country_segment_risk = event_data.groupby(["country", "segment"])["abs_DWRC_z"].max().reset_index()
    pivot_data = country_segment_risk.pivot(
        index="country", columns="segment", values="abs_DWRC_z"
    ).fillna(0)
    
    # 选择风险最高的20个国家
    country_mean_risk = pivot_data.mean(axis=1)
    top_countries = country_mean_risk.sort_values(ascending=False).head(20).index
    pivot_data = pivot_data.loc[top_countries]
    
    sns.heatmap(pivot_data, cmap="Reds", cbar_kws={"label": "Max |DWRC_z|"})
    plt.title("Country-Sector Risk Exposure during Critical Events")
    plt.ylabel("Country")
    plt.xlabel("Sector")
    
    plt.tight_layout(h_pad=2)
    return plt

# 执行事件实体分析
entity_chart = plot_event_entities(events_impact_df, df, EVENTS)
entity_chart.savefig(os.path.join(SAVE_DIR, "event_entities_analysis_real.png"), dpi=300)
plt.close()

# ======= 单事件深度分析 =======
def plot_single_event_analysis(event_month, df):
    """对单个事件进行深度分析"""
    event_data = df[df["ym"] == event_month].copy()
    event_data["abs_DWRC_z"] = np.abs(event_data["DWRC_z"])
    
    plt.figure(figsize=(18, 15))
    plt.suptitle(f"Deep Dive: Supply Chain Risk Distribution during {event_month} Event", 
                 fontsize=16, y=0.95)
    
    # 国家风险分布
    plt.subplot(2, 2, 1)
    country_risk = event_data.groupby("country")["abs_DWRC_z"].max()
    top_countries = country_risk.sort_values(ascending=False).head(20)
    sns.barplot(x=top_countries.values, y=top_countries.index, palette="Reds")
    plt.title(f"Top 20 Most Impacted Countries ({event_month})")
    plt.xlabel("Max |DWRC_z|")
    plt.ylabel("Country")
    
    # 部门风险分布
    plt.subplot(2, 2, 2)
    sector_risk = event_data.groupby("segment")["abs_DWRC_z"].mean().sort_values(ascending=False)
    sns.barplot(x=sector_risk.values, y=sector_risk.index, palette="Oranges")
    plt.title(f"Sector-Level Risk Impact ({event_month})")
    plt.xlabel("Mean |DWRC_z|")
    plt.ylabel("Sector")
    
    # 国家-部门组合风险
    plt.subplot(2, 1, 2)
    pivot_df = event_data.pivot_table(
        index="country", 
        columns="segment", 
        values="abs_DWRC_z", 
        aggfunc=np.mean
    ).fillna(0)
    
    # 选择风险高于平均值的国家
    country_mean_risk = pivot_df.mean(axis=1)
    pivot_df = pivot_df[country_mean_risk > country_mean_risk.mean()]
    sns.heatmap(pivot_df, cmap="Reds", cbar_kws={"label": "Mean |DWRC_z|"})
    plt.title(f"Country-Sector Risk Matrix ({event_month})")
    plt.ylabel("Country")
    plt.xlabel("Sector")
    
    plt.tight_layout(pad=3.0)
    return plt

# 执行单事件分析（以2020-03疫情冲击为例）
covid_chart = plot_single_event_analysis("2020-03", df)
covid_chart.savefig(os.path.join(SAVE_DIR, "covid_event_deepdive_real.png"), dpi=300)
plt.close()

# ======= 预警分析 =======
def analyze_early_warnings(sri_df, events_impact_df):
    """分析系统性风险预警能力"""
    results = []
    
    for idx, row in events_impact_df.iterrows():
        event_date = row['event_date_in_df']
        event_month = event_date.strftime("%Y-%m")
        
        # 获取事件发生前3个月的SRI数据
        warning_period = 3
        warning_dates = sri_df.index[sri_df.index < event_date][-warning_period:]
        
        if len(warning_dates) > 0:
            # 计算预警信号强度
            warning_sri = sri_df.loc[warning_dates, "SRI"]
            warning_strength = warning_sri.mean()
            warning_trend = warning_sri.pct_change().mean() * 100  # 百分比变化
            
            # 计算预警精度指标
            actual_impact = row['impact']
            warning_accuracy = min(1.0, max(0.0, (warning_strength - sri_df['SRI'].mean()) / sri_df['SRI'].std()))
            
            results.append({
                "event": row['description'],
                "date": event_month,
                "category": row['category'],
                "warning_strength": warning_strength,
                "warning_trend": warning_trend,
                "actual_impact": actual_impact,
                "warning_accuracy": warning_accuracy,
                "warning_level": "high" if warning_strength > 0.5 else "medium" if warning_strength > 0.3 else "low"
            })
    
    return pd.DataFrame(results)

# 执行预警分析
warning_df = analyze_early_warnings(sri_df, events_impact_df)

# 绘制预警分析图表
def plot_warning_analysis(warning_df):
    """绘制预警分析结果"""
    plt.figure(figsize=(14, 10))
    
    # 预警强度与实际影响
    plt.subplot(2, 1, 1)
    sns.scatterplot(data=warning_df, x="warning_strength", y="actual_impact", 
                   hue="category", palette=EVENT_COLORS, s=100)
    
    # 添加趋势线
    sns.regplot(data=warning_df, x="warning_strength", y="actual_impact", 
                scatter=False, color="gray")
    
    plt.title("Warning Strength vs Actual Impact", fontsize=16)
    plt.xlabel("Warning Strength (Average SRI in Preceding 3 Months)")
    plt.ylabel("Actual Impact (SRI Change)")
    plt.grid(True, alpha=0.3)
    
    # 预警准确率
    plt.subplot(2, 1, 2)
    sns.barplot(data=warning_df, x="event", y="warning_accuracy", hue="category", 
               palette=EVENT_COLORS, dodge=False)
    plt.title("Warning Accuracy by Event", fontsize=16)
    plt.xlabel("Event")
    plt.ylabel("Warning Accuracy")
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

# 绘制并保存预警分析图表
warning_chart = plot_warning_analysis(warning_df)
warning_chart.savefig(os.path.join(SAVE_DIR, "warning_analysis_real.png"), dpi=300, bbox_inches='tight')
plt.close()

# ======= 保存所有输出 =======
sri_df.to_csv(os.path.join(SAVE_DIR, "sri_index_real.csv"))
events_impact_df.to_csv(os.path.join(SAVE_DIR, "events_impact_analysis_real.csv"), index=False)
warning_df.to_csv(os.path.join(SAVE_DIR, "warning_analysis_results.csv"), index=False)

print("✅ 基于真实数据的分析已完成，输出保存在:", SAVE_DIR)
print("1. SRI主图表: sri_with_events_real.png")
print("2. 事件实体分析: event_entities_analysis_real.png")
print("3. 单事件深度分析: covid_event_deepdive_real.png")
print("4. 预警分析: warning_analysis_real.png")
print("5. SRI数据: sri_index_real.csv")
print("6. 事件影响分析: events_impact_analysis_real.csv")
print("7. 预警分析结果: warning_analysis_results.csv")





#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates
from datetime import datetime
from scipy.stats import ttest_ind, mannwhitneyu, norm

# ======= 配置路径 =======
SAVE_DIR = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/历年逐月/maps_output"
os.makedirs(SAVE_DIR, exist_ok=True)

INPUT_PATH = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/outputs_dwrc/primaryValue/network_role_dwrc_shock_all_params.csv"
df = pd.read_csv(INPUT_PATH)

# 数据预处理
df = df[~df["country"].str.contains("_X")]
df["date"] = pd.to_datetime(df["ym"], format="%Y-%m")
df["DWRC_z"] = df.groupby("date")["DWRC"].transform(lambda x: (x - x.mean()) / x.std())

# SRI指标计算
def gini_coefficient(array):
    array = np.sort(np.abs(array))
    n = len(array)
    cumvals = np.cumsum(array)
    return (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n if cumvals[-1] != 0 else 0

from scipy.stats import kurtosis as sp_kurtosis

def calculate_sri(df):
    gini = df.groupby("date")["DWRC_z"].apply(lambda x: gini_coefficient(np.abs(x)))
    # 用 scipy 的 kurtosis
    kurt = df.groupby("date")["DWRC_z"].apply(lambda x: sp_kurtosis(x, fisher=False, nan_policy='omit'))
    thresholds = [1.0, 1.5, 1.96, 2.5]
    extreme_ratios = []
    for thresh in thresholds:
        ratio = df.groupby("date")["DWRC_z"].apply(lambda x: (np.abs(x) > thresh).mean())
        extreme_ratios.append(ratio)
    sri_df = pd.DataFrame({
        "gini": gini,
        "kurtosis": kurt,
        "extreme_ratio_avg": pd.concat(extreme_ratios, axis=1).mean(axis=1)
    })
    sri_df["gini_z"] = (sri_df["gini"] - sri_df["gini"].mean()) / sri_df["gini"].std()
    sri_df["kurtosis_z"] = (sri_df["kurtosis"] - sri_df["kurtosis"].mean()) / sri_df["kurtosis"].std()
    sri_df["extreme_ratio_z"] = (sri_df["extreme_ratio_avg"] - sri_df["extreme_ratio_avg"].mean()) / sri_df["extreme_ratio_avg"].std()
    sri_df["SRI"] = (sri_df["gini_z"] + sri_df["kurtosis_z"] + sri_df["extreme_ratio_z"]) / 3
    return sri_df

sri_df = calculate_sri(df)

# ======= 事件库定义 =======
EVENTS = [
    {"month":"2010-08","desc":"China rare-earth export restrictions","category":"policy"},
    {"month":"2013-01","desc":"EU ETS carbon price crash & reform debates","category":"policy"},
    {"month":"2014-01","desc":"Indonesia bans exports of unprocessed nickel ore (first ban)","category":"policy"},
    {"month":"2014-11","desc":"OPEC no-cut decision—oil price collapse starts","category":"policy"},
    {"month":"2015-08","desc":"China RMB '8·11' FX regime adjustment","category":"policy"},
    {"month":"2016-06","desc":"UK 'Brexit' referendum","category":"policy"},
    {"month":"2017-03","desc":"US withdrawal from TPP","category":"policy"},
    {"month":"2018-03","desc":"US Section 232 steel/aluminum tariffs","category":"policy"},
    {"month":"2018-07","desc":"US–China tariffs (List 1)","category":"policy"},
    {"month":"2018-09","desc":"US–China tariffs (List 3 escalation)","category":"policy"},
    {"month":"2019-05","desc":"US threatens tariffs on Mexico (migration)","category":"policy"},
    {"month":"2020-01","desc":"Indonesia reinstates nickel ore export ban","category":"policy"},
    {"month":"2022-06","desc":"UFLPA takes effect (solar polysilicon)","category":"policy"},
    {"month":"2022-08","desc":"US Inflation Reduction Act (IRA) enacted—EV credits","category":"policy"},
    {"month":"2023-04","desc":"US Treasury mineral & battery component guidance","category":"policy"},
    {"month":"2023-06","desc":"Indonesia bauxite export ban takes effect","category":"policy"},
    {"month":"2023-08","desc":"EU Battery Regulation enters into force","category":"policy"},
    {"month":"2023-08","desc":"China export controls on gallium & germanium","category":"policy"},
    {"month":"2023-09","desc":"EU anti-subsidy probe into Chinese BEV imports","category":"policy"},
    {"month":"2023-10","desc":"EU CBAM transitional phase starts","category":"policy"},
    {"month":"2023-12","desc":"China export controls on graphite products","category":"policy"},
    {"month":"2015-08","desc":"Tianjin port explosions","category":"logistics"},
    {"month":"2020-03","desc":"COVID-19 mobility collapse","category":"logistics"},
    {"month":"2021-03","desc":"Suez Canal blockage","category":"logistics"},
    {"month":"2021-10","desc":"LA/LB port congestion peaks","category":"logistics"},
    {"month":"2022-04","desc":"Shanghai city-wide lockdown","category":"logistics"},
    {"month":"2023-07","desc":"Canada BC ports strike","category":"logistics"},
    {"month":"2023-11","desc":"Panama Canal drought restrictions","category":"logistics"},
    {"month":"2023-11","desc":"Red Sea/Houthi risk","category":"logistics"},
    {"month":"2011-03","desc":"Tōhoku earthquake + Thai floods","category":"disaster"},
    {"month":"2016-04","desc":"Kumamoto earthquake","category":"disaster"},
    {"month":"2017-08","desc":"Hurricane Harvey","category":"disaster"},
    {"month":"2021-02","desc":"Texas winter storm","category":"disaster"},
    {"month":"2022-08","desc":"Sichuan drought & power rationing","category":"disaster"}
]
EVENT_COLORS = {"policy": "#D62728", "logistics": "#1F77B4", "disaster": "#2CA02C"}

# ======= 事件影响分析表 =======
def analyze_events_impact(events, sri_df):
    event_records = []
    sri_df["month_str"] = sri_df.index.strftime("%Y-%m")
    for event in events:
        event_month = event["month"]
        if event_month in sri_df["month_str"].values:
            event_date = sri_df[sri_df["month_str"] == event_month].index[0]
            event_sri = sri_df.loc[event_date, "SRI"]
            prev_dates = sri_df.index[sri_df.index < event_date][-3:]
            pre_avg = sri_df.loc[prev_dates, "SRI"].mean() if len(prev_dates) > 0 else np.nan
            impact = event_sri - pre_avg if not np.isnan(pre_avg) else np.nan
            post_dates = sri_df.index[sri_df.index > event_date][:6]
            post_std = sri_df.loc[post_dates, "SRI"].std() if len(post_dates) > 1 else np.nan
            event_records.append({
                "event_date": event_month,
                "event_date_in_df": event_date,
                "description": event["desc"],
                "category": event["category"],
                "sri_at_event": event_sri,
                "pre_event_avg": pre_avg,
                "impact": impact,
                "post_volatility": post_std,
                "impact_level": "high" if abs(impact) > 0.5 else "medium" if abs(impact) > 0.3 else "low"
            })
    return pd.DataFrame(event_records)

events_impact_df = analyze_events_impact(EVENTS, sri_df)

# ======= 多窗口显著性分析 =======
def analyze_event_significance_multi(sri_df, events_impact_df, windows=[1, 3, 6]):
    """
    针对每个窗口，分别输出每个事件的前后SRI均值、t检验p值、U检验p值、显著性
    """
    sri_df = sri_df.copy()
    sri_df['month_str'] = sri_df.index.strftime("%Y-%m")
    records = []
    for win in windows:
        for idx, row in events_impact_df.iterrows():
            event_date = row['event_date_in_df']
            event_month = event_date.strftime("%Y-%m")
            prev_idx = sri_df.index[sri_df.index < event_date][-win:]
            post_idx = sri_df.index[sri_df.index > event_date][:win]
            if (len(prev_idx) == win) and (len(post_idx) == win):
                pre_sri = sri_df.loc[prev_idx, "SRI"].values
                post_sri = sri_df.loc[post_idx, "SRI"].values
                t_stat, t_pval = ttest_ind(post_sri, pre_sri, equal_var=False)
                u_stat, u_pval = mannwhitneyu(post_sri, pre_sri, alternative="two-sided")
                # 计算效应量
                diff = np.mean(post_sri) - np.mean(pre_sri)
                records.append({
                    "event": row["description"],
                    "date": event_month,
                    "category": row["category"],
                    "window": win,
                    "pre_mean": np.mean(pre_sri),
                    "post_mean": np.mean(post_sri),
                    "diff": diff,
                    "t_pval": t_pval,
                    "u_pval": u_pval,
                    "significant_t": t_pval < 0.05,
                    "significant_u": u_pval < 0.05
                })
    return pd.DataFrame(records)

event_sig_df = analyze_event_significance_multi(sri_df, events_impact_df, windows=[1, 3, 6])

# ======= 绘制多窗口事件前后SRI分布箱线图 =======
def plot_event_prepost_sri_multi(sri_df, events_impact_df, windows=[1,3,6]):
    n_win = len(windows)
    fig, axes = plt.subplots(n_win, 1, figsize=(22, 6 * n_win), sharex=True)
    if n_win == 1: axes = [axes]
    for w, win in enumerate(windows):
        event_labels = []
        for i, row in events_impact_df.iterrows():
            event_date = row['event_date_in_df']
            event_month = event_date.strftime("%Y-%m")
            prev_idx = sri_df.index[sri_df.index < event_date][-win:]
            post_idx = sri_df.index[sri_df.index > event_date][:win]
            if (len(prev_idx) == win) and (len(post_idx) == win):
                pre_sri = sri_df.loc[prev_idx, "SRI"].values
                post_sri = sri_df.loc[post_idx, "SRI"].values
                axes[w].boxplot([pre_sri, post_sri], positions=[i*3, i*3+1], widths=0.7,
                                patch_artist=True,
                                boxprops=dict(facecolor="#7fc97f" if np.mean(post_sri) > np.mean(pre_sri) else "#fdc086"))
                event_labels.append(row["description"][:32] + ("" if len(row["description"])<32 else "..."))
        ticks = [i*3+0.5 for i in range(len(event_labels))]
        axes[w].set_xticks(ticks)
        axes[w].set_xticklabels(event_labels, rotation=90, fontsize=12)
        axes[w].set_ylabel("SRI Value", fontsize=15)
        axes[w].set_title(f"SRI Before & After Events (Window={win} month{'s' if win>1 else ''})", fontsize=16)
    plt.xlabel("Event", fontsize=15)
    plt.tight_layout()
    return plt

# 保存显著性检验表
event_sig_df.to_csv(os.path.join(SAVE_DIR, "event_sri_significance_multiwindow.csv"), index=False)

# 绘制并保存多窗口箱线图
prepost_chart_multi = plot_event_prepost_sri_multi(sri_df, events_impact_df, windows=[1,3,6])
prepost_chart_multi.savefig(os.path.join(SAVE_DIR, "event_prepost_sri_boxplot_multi.png"), dpi=300, bbox_inches='tight')
plt.close()

print("8. 事件前后SRI多窗口显著性检验: event_sri_significance_multiwindow.csv")
print("9. 事件前后SRI多窗口分布图: event_prepost_sri_boxplot_multi.png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/历年逐月/maps_output/event_sri_significance_multiwindow.csv"
df = pd.read_csv(csv_path)

# 热力图
plot_df = df[["event", "window", "pre_mean", "post_mean", "diff", "t_pval", "significant_t"]].copy()
heatmap_df = plot_df.pivot(index="event", columns="window", values="diff")
sigmask = plot_df.pivot(index="event", columns="window", values="significant_t")
pvals = plot_df.pivot(index="event", columns="window", values="t_pval")
plt.figure(figsize=(14, max(7,0.32*len(heatmap_df))))
ax = sns.heatmap(heatmap_df, annot=pvals.round(3), fmt='', cmap="RdBu_r", center=0,
                 mask=~sigmask, cbar_kws={"label":"Post-Pre SRI Difference (Significant Only)"})
plt.title("Event-wise SRI Change (Significant Events, t-test, annotated by p-value)", fontsize=16)
plt.xlabel("Window (months)", fontsize=13)
plt.ylabel("Event", fontsize=13)
plt.tight_layout()
plt.savefig(csv_path.replace(".csv", "_sig_heatmap.png"), dpi=300)
plt.close()

# 柱状图
sig_count = df.groupby("window")["significant_t"].sum()
total_count = df.groupby("window")["significant_t"].count()
plt.figure(figsize=(6,4))
plt.bar(sig_count.index.astype(str), sig_count.values, label="Significant (p<0.05)", color="#2ca02c")
plt.bar(sig_count.index.astype(str), total_count.values-sig_count.values, bottom=sig_count.values, alpha=0.3, color="grey", label="Not Significant")
plt.ylabel("Number of Events")
plt.xlabel("Window (months)")
plt.title("Count of Events with Significant SRI Change (t-test)")
plt.legend()
plt.tight_layout()
plt.savefig(csv_path.replace(".csv", "_sig_count.png"), dpi=300)
plt.close()

# 条形图
sig_events = df[df['significant_t']].copy()
sig_events['label'] = sig_events['event'] + " (win=" + sig_events['window'].astype(str) + ")"
sig_events = sig_events.sort_values("diff")
plt.figure(figsize=(10, max(4,0.3*len(sig_events))))
colors = ['#d62728' if v<0 else '#2ca02c' for v in sig_events['diff']]
plt.barh(sig_events['label'], sig_events['diff'], color=colors)
plt.axvline(0, ls='--', color='gray')
plt.xlabel("Post-Pre SRI Difference")
plt.title("Significant SRI Change Before vs After Events (t-test)")
plt.tight_layout()
plt.savefig(csv_path.replace(".csv", "_sig_diff_bar.png"), dpi=300)
plt.close()

print("显著性热力图、统计柱状图、差值条形图已全部输出。")
