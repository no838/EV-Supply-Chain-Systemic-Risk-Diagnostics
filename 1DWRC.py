#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CSV_DIR = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/comtrade_csv"
OUT_BASE_DIR = "/Users/raism/Desktop/Raism/论文写/2025/美国关税/EV_SupplyChain_Strategy/src/outputs_dwrc"

# 基础阈值配置（最低保障）
BASE_THRESHOLDS = { 
    "primaryValue": 100,   # 贸易额不低于100美元
    "netWgt": 50           # 重量不低于50kg
}

# 参数组合遍历
DELTAS = [0.6, 0.7, 0.8, 0.9]  # 折扣系数 δ 范围
MAX_STEPS_LIST = [3, 4, 5]      # Shock 扩散最大步长范围
VALUE_FIELDS = ["primaryValue", "netWgt"]  # 两个值字段
# =================================================

# ---------- 时间阶段 ----------
PHASES = [
    ("2010-01", "2015-12", "PreTariff"),
    ("2016-01", "2019-12", "TradeWar"),
    ("2020-01", "2024-12", "PostCOVID_War"),
]

# ---------- 商品编码 → 环节 ----------
CMD_SEGMENT_MAP = {
    "260300":"Raw_Ore","260200":"Raw_Ore","260400":"Raw_Ore","261400":"Raw_Ore","250410":"Adv_Material",
    "282530":"Adv_Material","280461":"Adv_Material","284690":"Rare_Earth","280530":"Rare_Earth",
    "282520":"Li_Material","283691":"Li_Material","282560":"Adv_Material","282550":"Adv_Material",
    "38019010":"Adv_Material","382490":"Adv_Material","390469":"Adv_Material","390230":"Adv_Material",
    "85044012":"Power_Electronics","85044011":"Power_Electronics","850440":"Power_Electronics",
    "850152":"Electric_Motor","850131":"Electric_Motor","853710":"Power_Electronics","854231":"Power_Electronics",
    "850431":"Power_Electronics","850720":"Battery_Module","854890":"Power_Electronics",
    "847989":"Machinery","850120":"Electric_Motor","841960":"Machinery",
    "870380":"EV_Final","87038010":"EV_Final","87039010":"EV_Final"
}

# ---------- 辅助函数 ----------
def get_segment(cmd):
    """获取环节类型"""
    cmd_str = str(cmd).strip()
    if cmd_str in CMD_SEGMENT_MAP:
        return CMD_SEGMENT_MAP[cmd_str]
    if cmd_str.startswith(("2603", "2602", "2604", "2614")):
        return "Raw_Ore"
    return "Other"

def period_to_ym(period):
    """将时期转换为年月格式"""
    s = str(period).zfill(6)
    return f"{s[:4]}-{s[4:6]}" if len(s) >= 6 else "0000-00"

def calculate_dynamic_threshold(df, value_field):
    """计算动态阈值 - 使用min方案"""
    # 计算绝对阈值（25%分位数）和相对阈值（中位数的1%）
    abs_threshold = df[value_field].quantile(0.25)  # 25%分位数
    rel_threshold = df[value_field].median() * 0.01  # 中位数的1%
    
    # 取两者中较小值（min方案）
    dynamic_threshold = min(abs_threshold, rel_threshold)
    
    # 确保不低于基础阈值
    final_threshold = max(dynamic_threshold, BASE_THRESHOLDS[value_field])
    
    return final_threshold, abs_threshold, rel_threshold

def load_trade_data(csv_dir, value_field):
    """加载贸易数据（应用统一的min阈值策略）"""
    files = sorted(Path(csv_dir).glob("*.csv"))
    all_dfs = []
    
    print(f"加载{value_field}贸易数据并计算阈值...")
    
    for fp in tqdm(files, desc=f"处理CSV文件"):
        try:
            # 只读取必要列
            NEED_COLS = ["reporterISO", "partnerISO", "period", "flowDesc", value_field, "cmdCode"]
            df_pd = pd.read_csv(fp, usecols=NEED_COLS, dtype=str, low_memory=False)
            
            # 处理列名
            df_pd.columns = [c.strip() for c in df_pd.columns]
            
            # 添加环节和年月列
            df_pd["segment"] = df_pd["cmdCode"].apply(get_segment)
            df_pd["ym"] = df_pd["period"].apply(period_to_ym)
            
            # 转换值字段为数字
            df_pd[value_field] = pd.to_numeric(df_pd[value_field], errors="coerce")
            df_pd = df_pd.dropna(subset=[value_field])
            
            # 确保数据点足够
            if len(df_pd) < 5:
                threshold = BASE_THRESHOLDS[value_field]
                print(f"  文件 {fp.name}: 数据量不足，使用基础阈值={threshold:.2f}")
            else:
                # 计算动态阈值
                threshold, abs_threshold, rel_threshold = calculate_dynamic_threshold(df_pd, value_field)
                print(f"  文件 {fp.name}: 25%分位数={abs_threshold:.2f}, "
                      f"中位数1%={rel_threshold:.2f}, 采用阈值={threshold:.2f}")
            
            # 应用阈值过滤
            df_filtered = df_pd[df_pd[value_field] >= threshold]
            
            if df_filtered.empty:
                continue
                
            # 处理进出口流向
            export_df = df_filtered[df_filtered["flowDesc"] == "Export"].copy()
            import_df = df_filtered[df_filtered["flowDesc"] == "Import"].copy()
            
            # 重命名列
            export_df = export_df.rename(columns={
                "reporterISO": "from_country",
                "partnerISO": "to_country"
            }).drop(columns=["flowDesc"])
            
            import_df = import_df.rename(columns={
                "reporterISO": "to_country",
                "partnerISO": "from_country"
            }).drop(columns=["flowDesc"])
            
            # 添加到结果列表
            all_dfs.append(export_df)
            all_dfs.append(import_df)
            
        except Exception as e:
            print(f"处理{os.path.basename(fp)}出错: {e}")
    
    # 合并所有数据
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

def build_monthly_network(df, ym, segment, value_field):
    """构建月度网络图"""
    # 过滤指定月份和环节
    d = df[(df["ym"] == ym) & (df["segment"] == segment)]
    if d.empty:
        return nx.DiGraph()
    
    # 计算总贸易值用于归一化
    total_value = d[value_field].sum()
    if total_value <= 0:
        return nx.DiGraph()
    
    # 创建有向图
    G = nx.DiGraph()
    # 使用groupby优化边添加
    grouped = d.groupby(["from_country", "to_country"])[value_field].sum().reset_index()
    
    for _, row in grouped.iterrows():
        from_country = row["from_country"]
        to_country = row["to_country"]
        weight = row[value_field]
        G.add_edge(from_country, to_country, weight=weight, w_norm=weight/total_value)
    
    return G

def discounted_weighted_reach(G, delta, max_steps):
    """优化折扣加权可达性计算"""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {}, {}
    
    nodes = list(G.nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    # 构建稀疏权重矩阵
    adj_matrix = sp.lil_matrix((n, n), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        adj_matrix[i, j] = data["w_norm"] * delta
    
    # 计算可达性矩阵
    reach_matrix = sp.lil_matrix((n, n), dtype=np.float32)
    power_matrix = adj_matrix.copy()
    
    for step in range(1, max_steps + 1):
        reach_matrix += power_matrix
        if step < max_steps:
            power_matrix = power_matrix @ adj_matrix
    
    # 计算加权可达性
    wr = {}
    wc = {}
    for node in nodes:
        i = node_index[node]
        wr[node] = reach_matrix[i, :].sum()
        wc[node] = reach_matrix[:, i].sum()
    
    return wr, wc

def shock_flow_amplification(G, delta, max_steps):
    """优化冲击流模拟"""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {}
    
    nodes = list(G.nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    # 构建转移概率矩阵
    P = sp.lil_matrix((n, n), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        P[i, j] = data["w_norm"]
    
    # 行归一化（处理出度为0的节点）
    row_sums = P.sum(axis=1)
    for i in range(n):
        if row_sums[i] == 0:
            P[i, :] = 0  # 防止除零
        else:
            P[i, :] /= row_sums[i]
    
    # 模拟冲击传播
    shock_matrix = sp.eye(n, format="lil", dtype=np.float32)
    total_shock = sp.lil_matrix((n, n), dtype=np.float32)
    
    for step in range(1, max_steps + 1):
        # 应用折扣因子
        shock_matrix = shock_matrix * (delta ** step)
        # 累加冲击
        total_shock += shock_matrix
        # 传播冲击
        shock_matrix = shock_matrix @ P
    
    # 计算平均冲击
    amp = {}
    for j, node in enumerate(nodes):
        amp[node] = total_shock[:, j].sum() / n
    
    return amp

def process_month_segment(args):
    """处理单个月份和环节的任务"""
    ym, seg, df, value_field, delta, max_steps = args
    try:
        # 构建网络图
        G = build_monthly_network(df, ym, seg, value_field)
        if G.number_of_nodes() < 2:  # 至少需要2个节点的网络
            return []
        
        # 计算节点强度
        out_strength = {node: data for node, data in G.out_degree(weight="weight")}
        in_strength = {node: data for node, data in G.in_degree(weight="weight")}
        
        # 计算折扣加权可达性
        wr, wc = discounted_weighted_reach(G, delta, max_steps)
        
        # 计算冲击放大效应
        amp = shock_flow_amplification(G, delta, max_steps)
        
        # 收集结果
        records = []
        all_nodes = set(G.nodes) | set(wr.keys()) | set(wc.keys()) | set(amp.keys())
        
        for node in all_nodes:
            records.append({
                "ym": ym,
                "segment": seg,
                "country": node,
                "out_strength": out_strength.get(node, 0.0),
                "in_strength": in_strength.get(node, 0.0),
                "DWRC": wr.get(node, 0.0) - wc.get(node, 0.0),
                "WReach": wr.get(node, 0.0),
                "WCover": wc.get(node, 0.0),
                "ShockAmp": amp.get(node, 0.0),
                "delta": delta,
                "max_steps": max_steps,
                "value_field": value_field
            })
        
        return records
    except Exception as e:
        print(f"处理{ym}-{seg}出错 (delta={delta}, steps={max_steps}): {str(e)}")
        return []

def run_analysis_for_value_field(value_field, out_base_dir):
    """为单个值字段运行分析"""
    os.makedirs(out_base_dir, exist_ok=True)
    start_time = time.time()
    
    print(f"加载{value_field}贸易数据...")
    df = load_trade_data(CSV_DIR, value_field)
    
    if df is None or df.empty:
        print(f"{value_field}没有数据，跳过分析")
        return
    
    print(f"数据加载完成 ({len(df)}条记录), 耗时: {time.time()-start_time:.2f}秒")
    
    # 获取唯一的年份-月份和环节
    segs = df["segment"].unique()
    yms = sorted(df["ym"].unique())
    
    # 准备所有参数组合的任务
    all_tasks = []
    for delta, max_steps in itertools.product(DELTAS, MAX_STEPS_LIST):
        for ym in yms:
            for seg in segs:
                all_tasks.append((ym, seg, df, value_field, delta, max_steps))
    
    print(f"开始处理{len(all_tasks)}个任务，使用{min(12, cpu_count())}个进程...")
    
    # 使用多进程并行处理
    all_records = []
    with Pool(processes=min(12, cpu_count())) as pool:
        # 分批处理任务避免内存溢出
        chunksize = max(1, len(all_tasks) // (100 * min(12, cpu_count())))
        results = list(tqdm(pool.imap(process_month_segment, all_tasks, chunksize=chunksize), total=len(all_tasks)))
        
        for res in results:
            if res:  # 跳过空结果
                all_records.extend(res)
    
    # 保存结果
    if all_records:
        df_out = pd.DataFrame(all_records)
        output_file = f"{out_base_dir}/network_role_dwrc_shock_all_params.csv"
        df_out.to_csv(output_file, index=False)
        print(f"结果已保存至: {output_file}")
        
        # 按阶段聚合结果  
        df_out["phase"] = "Other"
        for start, end, tag in PHASES:
            mask = (df_out["ym"] >= start) & (df_out["ym"] <= end)
            df_out.loc[mask, "phase"] = tag
        
        # 按阶段、环节和国家聚合
        role_stats = df_out.groupby(["phase", "segment", "country", "delta", "max_steps"]).agg({
            "out_strength": "median",
            "in_strength": "median",
            "DWRC": "median",
            "WReach": "median",
            "WCover": "median",
            "ShockAmp": "median",
        }).reset_index()
        
        stats_file = f"{out_base_dir}/ .csv"
        role_stats.to_csv(stats_file, index=False)
        print(f"阶段聚合结果已保存至: {stats_file}")
        
        # 参数敏感性分析
        param_stats = df_out.groupby(["delta", "max_steps"]).agg({
            "DWRC": ["mean", "std"],
            "ShockAmp": ["mean", "std"],
        }).reset_index()
        param_stats.columns = ["delta", "max_steps", "DWRC_mean", "DWRC_std", "ShockAmp_mean", "ShockAmp_std"]
        sensitivity_file = f"{out_base_dir}/parameter_sensitivity.csv"
        param_stats.to_csv(sensitivity_file, index=False)
        print(f"参数敏感性分析已保存至: {sensitivity_file}")
    else:
        print(f"没有生成结果记录")
    
    print(f"{value_field}分析完成，总耗时: {time.time()-start_time:.2f}秒")

def main():
    """主函数，为每个值字段运行分析"""
    for value_field in VALUE_FIELDS:
        print(f"\n{'='*60}")
        print(f"开始处理值字段: {value_field}")
        print(f"{'='*60}")
        out_dir = os.path.join(OUT_BASE_DIR, value_field)
        run_analysis_for_value_field(value_field, out_dir)

if __name__ == "__main__":
    main()