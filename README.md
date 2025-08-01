
# EV Supply Chain Systemic Risk Diagnostics

This repository provides core diagnostic tools developed for the paper:

> **"Directional Role Reconfiguration and Dual-Weighted Risk Signals in Global EV Supply Chains (2010–2024)"**

It contains a three-part analytical pipeline that detects, interprets, and quantifies systemic risk and directional influence shifts in monthly trade networks across nine electric vehicle (EV) segments.

---

## 📁 Repository Structure

| File         | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `1DWRC.py`   | Computes **Directional Weighted Role Contrast (DWRC)** for each country–segment–month node, capturing the difference between discounted multi-hop influence and exposure. |
| `2PCMCI.py`  | Implements **causal discovery** via the PCMCI algorithm (TIGRAMITE), producing lag-resolved, statistically validated directed links among nodes. |
| `3SRI.py`    | Calculates the **Systemic Risk Index (SRI)** using z-score-standardized tail indicators (kurtosis, Gini, extreme-node share) from value- and weight-weighted trade networks. |

---

## 🔧 Requirements

Install required packages (Python ≥ 3.8):

```bash
pip install pandas numpy networkx matplotlib seaborn tigramite
```

---

## 📊 Methodological Highlights

- **DWRC**: Identifies role shifts (e.g., propagator → absorber) using path-weighted reach and cover metrics.
- **PCMCI**: Extracts lagged directional dependencies, supporting the construction of **causal edge-phase maps**.
- **Dual-weighted SRI**: Separately captures:
  - *Value-weighted risk (VW)* — price/sentiment-driven stress
  - *Weight-weighted risk (WW)* — physical logistics bottlenecks

---

## 🧪 Usage Example

```bash
# Step 1: Compute DWRC
python 1DWRC.py

# Step 2: Run causal discovery with PCMCI
python 2PCMCI.py

# Step 3: Generate SRI time series
python 3SRI.py
```

Data paths should be configured in each script (`DATA_DIR`, `OUTPUT_DIR`) according to your local file structure.

---


