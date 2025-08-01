#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Causal Network Analysis - Phase Split + Export/Import Union + Robustness + Pub-ready Output
2024-06-23
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, Memory, parallel_backend
import traceback
import shutil
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

from statsmodels.tsa.stattools import grangercausalitytests
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# =============== 0. 全局参数 ===============
SELECTED_FIELD = "netWgt"  # "tradeValue" or "netWgt"
INPUT_FILES = {
    "tradeValue": "country2country_segment_monthly_event_value.csv",
    "netWgt": "country2country_segment_monthly_event_netwgt.csv",
}
OUT_BASE = "results_causal_networks_phases_union"
os.makedirs(OUT_BASE, exist_ok=True)

USE_CACHE = True
CACHEDIR = './cache_granger'
if not USE_CACHE and os.path.exists(CACHEDIR):
    shutil.rmtree(CACHEDIR)
os.makedirs(CACHEDIR, exist_ok=True)
memory = Memory(CACHEDIR, verbose=0) if USE_CACHE else Memory(location=None, verbose=0)

SEGMENT_AGGREGATE_LIST = [False, True]  # 保持参数兼容
LAGS = [12]
PTHRS = [0.01]
CORR_THR_LIST = [0.4]
TOP_K_LIST = [30]
MIN_MONTHS_LIST = [48]
MAX_PCMCI_VARS_LIST = [30]
TOP_EDGES_FOR_MAINCHAIN = 80
N_CORES = min(10, max(1, os.cpu_count() - 2))

PHASES = [
    ("2010-01", "2015-12", "PreTariff"),
    ("2016-01", "2019-12", "TradeWar"),
    ("2020-01", "2024-12", "PostCOVID_War"),
]
def ym_to_phase(ym):
    for start, end, label in PHASES:
        if start <= ym <= end:
            return label
    return None

# ============ 1. 数据预处理（核心改动） ============

def make_country_segment_inout_table(df, value_field):
    # 出口（出口国-环节）：对所有目的国加总
    df_export = df.groupby(["ym", "from_country", "segment"], as_index=False)[value_field].sum()
    df_export["flow_type"] = "Export"
    df_export = df_export.rename(columns={"from_country": "country"})
    # 进口（进口国-环节）：对所有来源国加总
    df_import = df.groupby(["ym", "to_country", "segment"], as_index=False)[value_field].sum()
    df_import["flow_type"] = "Import"
    df_import = df_import.rename(columns={"to_country": "country"})
    # 合并
    df_all = pd.concat([df_export, df_import], axis=0, ignore_index=True)
    return df_all[["ym", "country", "segment", "flow_type", value_field]]

def build_varnames(df_all):
    countries = sorted(df_all["country"].unique())
    segments = sorted(df_all["segment"].unique())
    flow_types = ["Export", "Import"]
    var_names = [f"{c}|{s}|{ft}" for c in countries for s in segments for ft in flow_types]
    return var_names

def fast_data_matrix(df_all, var_names, value_col):
    ym_map = {y: i for i, y in enumerate(sorted(df_all["ym"].unique()))}
    var_map = {v: i for i, v in enumerate(var_names)}
    n_time = len(ym_map)
    n_vars = len(var_names)
    data_mat = np.zeros((n_time, n_vars), dtype=np.float32)
    for _, row in df_all.iterrows():
        t = ym_map.get(row["ym"], -1)
        var_name = f"{row['country']}|{row['segment']}|{row['flow_type']}"
        v = var_map.get(var_name, -1)
        if t != -1 and v != -1:
            val = np.log1p(row[value_col])
            data_mat[t, v] = val if val > 0 else 0
    return data_mat

# ============ 2. 其它函数（原流程） ============

@memory.cache
def batch_granger_test(data, pairs, maxlag, pthr):
    results = []
    for i, j in pairs:
        try:
            test_result = grangercausalitytests(
                data[:, [i, j]],
                maxlag=maxlag,
                verbose=False
            )
            min_p = min(test_result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1))
            if min_p < pthr:
                for lag in range(1, maxlag+1):
                    if test_result[lag][0]['ssr_ftest'][1] == min_p:
                        best_lag = lag
                        break
                results.append((i, j, min_p, best_lag))
        except Exception:
            continue
    return results

def select_relevant_variables(main_idx, abs_corr_mat, corr_thr, pair_df, top_k):
    correlated = set(np.where(abs_corr_mat[main_idx] > corr_thr)[0])
    top_targets = set()
    if not pair_df.empty:
        targets = pair_df.sort_values('min_pval').head(top_k)['target_idx'].unique()
        top_targets = set(targets)
    return correlated | top_targets

def run_pcmci_subnet(main_var, var_map, idx_to_var, data_mat, abs_corr_mat,
                     pairwise_df, lag, pthr, corr_thr, top_k, max_pcmci_vars,
                     out_dir):
    pcmci_links = []
    err_log = os.path.join(out_dir, "error.log")
    try:
        if main_var not in var_map:
            return []
        main_idx = var_map[main_var]
        pair_df = pairwise_df[pairwise_df['source'] == main_var]
        var_idxs = select_relevant_variables(main_idx, abs_corr_mat, corr_thr, pair_df, top_k)
        var_idxs.add(main_idx)
        if len(var_idxs) > max_pcmci_vars:
            corr_scores = abs_corr_mat[main_idx, list(var_idxs)]
            top_idx = np.argsort(corr_scores)[-max_pcmci_vars:]
            var_idxs = set(np.array(list(var_idxs))[top_idx])
        var_names_sub = [idx_to_var[idx] for idx in var_idxs]
        data_sub = data_mat[:, list(var_idxs)]
        pcmci = PCMCI(
            dataframe=pp.DataFrame(data_sub, var_names=var_names_sub),
            cond_ind_test=ParCorr(significance="analytic", verbosity=0)
        )
        results = pcmci.run_pcmci(tau_max=lag, pc_alpha=pthr)
        V, P = results["val_matrix"], results["p_matrix"]
        n_vars = len(var_names_sub)
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j: continue
                best_tau = None
                best_val = None
                for tau in range(1, lag+1):
                    coef, pval = V[i, j, tau], P[i, j, tau]
                    if pval < pthr and (best_val is None or abs(coef) > abs(best_val)):
                        best_tau = tau
                        best_val = coef
                if best_tau is not None:
                    pcmci_links.append([
                        var_names_sub[j], var_names_sub[i], best_tau, best_val, pval
                    ])
    except Exception as e:
        with open(err_log, "a") as f:
            f.write(f"PCMCI error for {main_var} : {str(e)}\n")
            f.write(traceback.format_exc() + "\n")
    return pcmci_links

def fast_visualization(df, out_png, title=""):
    if df.empty:
        return
    plt.figure(figsize=(12, 8))
    top_df = df.nlargest(min(50, len(df)), "coef")
    G = nx.from_pandas_edgelist(top_df, 'source', 'target',
                                edge_attr=['coef', 'lag'],
                                create_using=nx.DiGraph())
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, width=[0.5 + abs(d['coef'])*3 for u, v, d in G.edges(data=True)],
                           arrowstyle='->', arrowsize=12)
    deg = dict(G.degree())
    important_nodes = [n for n, d in deg.items() if d > 1]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in important_nodes}, font_size=7)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close()

def save_progress(progress_file, stage, value):
    pd.DataFrame({"stage": [stage], "value": [value]}).to_csv(progress_file, index=False)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        df = pd.read_csv(progress_file)
        if not df.empty:
            return df.iloc[-1]["stage"], df.iloc[-1]["value"]
    return None, None

def split_label(label):
    parts = label.split("|")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "ALL", "ALL", label

def summary_mainchain(df):
    if df.empty:
        return pd.DataFrame()
    top_df = df.sort_values(by="coef", ascending=False).head(TOP_EDGES_FOR_MAINCHAIN) if "coef" in df.columns else df
    cross_country = 0
    cross_segment = 0
    cross_io = 0
    for _, row in top_df.iterrows():
        src = split_label(row["source"])
        tgt = split_label(row["target"])
        if src[0] != tgt[0]:
            cross_country += 1
        if src[1] != tgt[1]:
            cross_segment += 1
        if src[2] != tgt[2]:
            cross_io += 1
    res = pd.DataFrame([{
        "n_edges": len(top_df),
        "n_nodes": len(set(top_df["source"]).union(set(top_df["target"]))),
        "cross_country_ratio": cross_country / len(top_df) if len(top_df) > 0 else 0,
        "cross_segment_ratio": cross_segment / len(top_df) if len(top_df) > 0 else 0,
        "cross_inout_ratio": cross_io / len(top_df) if len(top_df) > 0 else 0,
    }])
    return res

def compare_mainchain_edges(edge_dict):
    keys = list(edge_dict.keys())
    n = len(keys)
    matrix = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            s1, s2 = edge_dict[ki], edge_dict[kj]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            matrix[i, j] = inter / union if union > 0 else 1
    return keys, matrix

# ============ 3. 主流程函数（分阶段+联合建模） ============
def run_pipeline_by_phase(phase_df, phase_name, value_field, lag, pthr, corr_thr, top_k, min_months, max_pcmci_vars):
    param_str = f"{phase_name}_{value_field}_union_lag{lag}_p{str(pthr).replace('.','')}_corr{int(100*corr_thr)}_topk{top_k}_min{min_months}_maxpcmci{max_pcmci_vars}"
    out_dir = os.path.join(OUT_BASE, param_str)
    os.makedirs(out_dir, exist_ok=True)
    err_log = os.path.join(out_dir, "error.log")
    progress_file = os.path.join(out_dir, ".progress.csv")
    print(f"\n==== {param_str} ====")
    value_col = value_field if value_field in phase_df.columns else phase_df.columns[-1]

    # **1. 聚合为 国家-环节-进出口联合变量表**
    df_all = make_country_segment_inout_table(phase_df, value_col)
    # **2. 变量活跃度筛选**
    act_mask = df_all.groupby(["country", "segment", "flow_type"])[value_col].transform(lambda x: (x > 0).sum() >= min_months)
    df_all = df_all[act_mask]
    var_names = build_varnames(df_all)
    if len(var_names) < 5:
        print("跳过本阶段-变量数过少:", len(var_names))
        return param_str, pd.DataFrame(), set()
    data_mat = fast_data_matrix(df_all, var_names, value_col)
    print(f"数据矩阵: {data_mat.shape[0]}月 x {data_mat.shape[1]}变量")
    # ========== 2. Pairwise Granger ==========
    stage, last_val = load_progress(progress_file)
    if stage == "pairwise_done":
        pairwise_df = pd.read_csv(os.path.join(out_dir, "pairwise_significant.csv"))
    else:
        print("批量Pairwise Granger检验...")
        n_vars = len(var_names)
        total_pairs = n_vars * (n_vars - 1)
        all_pairs = [(i, j) for i in range(n_vars) for j in range(n_vars) if i != j]
        print("计算相关性矩阵...")
        dense_data = data_mat if isinstance(data_mat, np.ndarray) else data_mat.toarray()
        abs_corr_mat = np.abs(np.corrcoef(dense_data.T))
        filtered_pairs = [(i, j) for i, j in all_pairs if abs_corr_mat[i, j] > 0.15]
        print(f"过滤后保留 {len(filtered_pairs)}/{total_pairs} 变量对")
        batch_size = 500
        batches = [filtered_pairs[i:i+batch_size] for i in range(0, len(filtered_pairs), batch_size)]
        granger_results = []
        with parallel_backend('loky', n_jobs=N_CORES):
            results = Parallel(n_jobs=N_CORES)(
                delayed(batch_granger_test)(dense_data, batch, lag, pthr) for batch in tqdm(batches, desc="Granger-batch")
            )
            for res in results:
                granger_results.extend(res)
        pairwise_data = []
        for i, j, min_p, best_lag in granger_results:
            pairwise_data.append({
                "source": var_names[j],
                "target": var_names[i],
                "min_pval": min_p,
                "lag": best_lag,
                "source_idx": j,
                "target_idx": i,
            })
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df.to_csv(os.path.join(out_dir, "pairwise_significant.csv"), index=False)
        save_progress(progress_file, "pairwise_done", len(pairwise_df))
    # ========== 3. mini-PCMCI ==========
    if stage == "pcmci_done":
        try:
            pcmci_df = pd.read_csv(os.path.join(out_dir, "pcmci_direct_links.csv"))
        except Exception as e:
            with open(err_log, "a") as f:
                f.write(f"LOAD ERROR PCMCI: {str(e)}\n")
            return param_str, pd.DataFrame(), set()
    else:
        print("mini-PCMCI子网分析...")
        dense_data = data_mat if isinstance(data_mat, np.ndarray) else data_mat.toarray()
        abs_corr_mat = np.abs(np.corrcoef(dense_data.T))
        sig_vars = pairwise_df["source"].unique().tolist()
        var_freq = pd.Series(pairwise_df["source"].tolist() + pairwise_df["target"].tolist()).value_counts()
        sig_vars_sorted = var_freq.index.tolist()
        pcmci_links = []
        for main_var in tqdm(sig_vars_sorted[:min(40, len(sig_vars_sorted))], desc="mini-PCMCI"):
            links = run_pcmci_subnet(
                main_var, {v: i for i, v in enumerate(var_names)}, {i: v for i, v in enumerate(var_names)},
                dense_data, abs_corr_mat,
                pairwise_df[pairwise_df["source"] == main_var], lag, pthr, corr_thr, top_k, max_pcmci_vars, out_dir
            )
            pcmci_links.extend(links)
            if len(pcmci_links) % 100 == 0:
                pd.DataFrame(pcmci_links, columns=["source", "target", "lag", "coef", "pval"]).to_csv(
                    os.path.join(out_dir, "pcmci_direct_links.partial.csv"), index=False)
        if pcmci_links:
            pcmci_df = pd.DataFrame(pcmci_links, columns=["source", "target", "lag", "coef", "pval"])
            pcmci_df = pcmci_df.drop_duplicates(subset=["source", "target", "lag"])
            pcmci_df.to_csv(os.path.join(out_dir, "pcmci_direct_links.csv"), index=False)
            save_progress(progress_file, "pcmci_done", len(pcmci_df))
        else:
            pcmci_df = pd.DataFrame()
    # ====== 4. 结果输出 ======
    if not pcmci_df.empty:
        fast_visualization(pcmci_df, os.path.join(out_dir, "fast_mainchain.png"),
                          f"Mainchain-PCMCI ({param_str})")
        summary = summary_mainchain(pcmci_df)
        if not summary.empty:
            summary["param_str"] = param_str
            summary.to_csv(os.path.join(out_dir, "mainchain_summary.csv"), index=False)
        edge_set = set([(row["source"], row["target"]) for _, row in pcmci_df.iterrows()])
        sig_df = pcmci_df[pcmci_df['pval'] < 0.05].copy()
        if not sig_df.empty:
            cols = ["source", "target", "coef", "lag", "pval"]
            sig_df[cols].to_csv(os.path.join(out_dir, "pcmci_direct_links_sig005.csv"), index=False)
        return param_str, summary, edge_set
    else:
        return param_str, pd.DataFrame(), set()

# ============ 4. 外层控制（全阶段遍历） ============
def main():
    orig_df = pd.read_csv(INPUT_FILES[SELECTED_FIELD])
    if "stage_tag" not in orig_df.columns:
        orig_df["stage_tag"] = orig_df["ym"].astype(str).apply(ym_to_phase)
    orig_df = orig_df[~orig_df["stage_tag"].isna()].copy()
    print(orig_df["stage_tag"].value_counts())

    param_list = []
    for (start, end, phase_name) in PHASES:
        phase_df = orig_df[(orig_df["ym"] >= start) & (orig_df["ym"] <= end)].copy()
        if len(phase_df) < 36:
            print(f"阶段{phase_name}数据太少，跳过。")
            continue
        for lag in LAGS:
            for pthr in PTHRS:
                for corr_thr in CORR_THR_LIST:
                    for top_k in TOP_K_LIST:
                        for min_months in MIN_MONTHS_LIST:
                            for max_pcmci_vars in MAX_PCMCI_VARS_LIST:
                                param_list.append((phase_df, phase_name, SELECTED_FIELD, lag, pthr, corr_thr, top_k, min_months, max_pcmci_vars))

    print(f"总共 {len(param_list)} 组阶段参数将被分析")
    results = Parallel(n_jobs=N_CORES)(delayed(run_pipeline_by_phase)(*args) for args in param_list)
    summary_all = []
    edge_dict = {}
    for param_str, summary, edge_set in results:
        if summary is not None and not summary.empty:
            summary["param_str"] = param_str
            summary_all.append(summary)
        edge_dict[param_str] = edge_set
    if summary_all:
        all_sum = pd.concat(summary_all, ignore_index=True)
        all_sum.to_csv(os.path.join(OUT_BASE, f"mainchain_summary_all_{SELECTED_FIELD}_phases.csv"), index=False)
        all_sum.sort_values(by=["cross_country_ratio", "cross_segment_ratio", "cross_inout_ratio"], ascending=False).to_csv(
            os.path.join(OUT_BASE, f"mainchain_summary_all_sorted_{SELECTED_FIELD}_phases.csv"), index=False)
    if len(edge_dict) > 1:
        keys, matrix = compare_mainchain_edges(edge_dict)
        plt.figure(figsize=(max(8, 0.18*len(keys)), max(7, 0.18*len(keys))))
        sns.heatmap(matrix, xticklabels=keys, yticklabels=keys, cmap="Blues", annot=False)
        plt.title(f"Mainchain Structure Overlap (Jaccard Index)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_BASE, f"mainchain_overlap_heatmap_{SELECTED_FIELD}_phases.png"), dpi=600)
        plt.close()
    print("ALL PHASE ANALYSIS FINISHED. See:", OUT_BASE)

if __name__ == "__main__":
    main()