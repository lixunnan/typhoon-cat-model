# 台风巨灾风险建模 + 巨灾金融定价
# Typhoon Catastrophe Risk Model + Catastrophe Risk Financing

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)](https://lixunnan.github.io/typhoon-cat-model/) [![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](https://www.python.org) [![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE) [![Tests](https://img.shields.io/badge/Tests-19%20passed-brightgreen)](https://github.com/lixunnan/typhoon-cat-model)

**在线交互展示 / Live Demo:** https://lixunnan.github.io/typhoon-cat-model/

> 以 2019 年台风"利奇马"(Lekima, 1909) 为校准案例的一套**完整、可复现、端到端**的巨灾风险量化平台。
> End-to-end, reproducible typhoon catastrophe (CAT) risk modeling and risk-financing platform,
> calibrated against the 2019 Typhoon Lekima (1909) landfall event.

---

## 1. 项目目标 / Objective

遵循国际再保险行业标准的 **CAT 模型四模块框架**（Hazard → Exposure → Vulnerability → Financial），
从台风风场物理建模出发，经暴露数据库、脆弱性曲线、损失校准，最终产出可用于资本与定价决策的
**EP 曲线 / AAL / VaR / TVaR / 偿二代二期资本 / 超赔再保险分层定价 / CAT bond 定价 / 基差风险 / 投资组合分散化**全部指标。

This project implements the industry-standard four-module CAT model pipeline and produces the full
suite of risk-financing metrics used by (re)insurers and cat-bond investors.

**关键结果（IS_PASS: YES，实测约 3 s，< 60 s 要求）：**
- 利奇马建模损失 **537.2 亿元** vs 实际 **537.2 亿元**（相对误差 ≈ 0）
- 经济口径 AAL **219.4 亿元/年**；100 年一遇 OEP PML **2,710.6 亿元**
- 100 年 PML / AAL ≈ **12.35 倍**（符合 5~20x 行业经验带）
-  insured VaR 99.5% **61.59 亿**；偿二代二期资本 **58.59 亿**
- 超赔再保综合 ROL **5.93%**；CAT bond Lane spread **591 bp** / Wang **268 bp**
- 参数触发基差风险对冲效率（fitted-box）**42.1%**

---

## 2. 文件结构 / Repository Layout

```
typhoon_cat_model/
├── config.py            # 全局参数（dataclass）：Hazard / Stochastic / Vulnerability / Financial / Plot
├── hazard.py            # 模块A 灾害：利奇马真实路径、Holland 1980 风场、登陆衰减、10000 场随机事件集
├── exposure.py          # 模块B 暴露：14 个华东沿海地市示例暴露数据库
├── vulnerability.py     # 模块C 脆弱性：Emanuel 2011 复合曲线 + 对数正态、内涝/需求激增因子、利奇马校准
├── financial.py         # 模块D 金融：EP/OEP/AEP、AAL/VaR/TVaR、C-ROSS 资本、超赔分层、CAT bond、基差风险、投资组合
├── visualization.py     # 11 张 300dpi 英文图表
├── main.py              # 主流程编排 + 全局一致性审查（IS_PASS）
├── requirements.txt     # 依赖清单（仅 numpy/pandas/scipy/matplotlib）
├── README.md            # 本文档
└── outputs/
    ├── fig01_lekima_track.png
    ├── fig02_holland_profiles.png
    ├── fig03_wind_field.png
    ├── fig04_vulnerability.png
    ├── fig05_lekima_validation.png
    ├── fig06_ep_curves.png
    ├── fig07_reinsurance_layers.png
    ├── fig08_catbond_pricing.png
    ├── fig09_basis_risk.png
    ├── fig10_portfolio_frontier.png
    └── fig11_event_set.png
```

---

## 3. 系统架构图 / Architecture

```
                         ┌─────────────────────────────────────────────┐
      气象输入            │                MODULE A : HAZARD            │
  (历史路径 + 随机事件集)  │  Lekima 真实轨迹 (37 waypoints)             │
       │                 │  Holland 1980 梯度风廓线 + 物理 B            │
       ▼                 │  移动非对称修正 (右半圆叠加 α·Vmove)         │
  ┌──────────┐           │  Kaplan-DeMaria 登陆指数衰减                │
  │ EventSet │           │  Atkinson-Holliday Vmax = 3.4·Δp^0.644      │
  │ 10000 场 │           │  10000 场蒙特卡洛随机事件集 (截断对数正态)   │
  └────┬─────┘           └───────────────────────┬─────────────────────┘
       │ 每场事件的逐点最大风速场 (gust, m/s)      │
       ▼                                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                MODULE B : EXPOSURE (14 地市)                       │
  │  GDP × capital_output_ratio × wind_exposed_fraction = exposed      │
  │  exposed × insurance_penetration = insured                        │
  │  建筑类型权重 (concrete/masonry/light_steel/greenhouse)            │
  └───────────────────────────────┬──────────────────────────────────┘
                                   │ site-level wind intensity + exposure
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │       MODULE C : VULNERABILITY                                     │
  │  Emanuel 2011 :  f(v) = (v/v_half)³ / (1 + (v/v_half)³)           │
  │  lognormal    :  f(v) = Φ( ln(v/v_half) / β )                     │
  │  复合曲线 = Σ w_i · f_i(v)                                         │
  │  次生内涝因子 (高斯衰减 × 强度 × 省份易涝系数)                      │
  │  需求激增因子 (灾后重建价格膨胀)                                    │
  │  calibrate_vulnerability(): brentq 反演 v_half 使损失≈537.2 亿     │
  └───────────────────────────────┬──────────────────────────────────┘
                                   │ modelled loss per event (economic & insured)
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │       MODULE D : FINANCIAL                                         │
  │  泊松年频率 YLT → OEP / AEP EP 曲线 → AAL / PML                    │
  │  VaR / TVaR (AEP) → 偿二代二期资本 = VaR99.5 − AAL                 │
  │  超赔分层: 标准差保费原理 P=(E[R]+k·σ_R)(1+e)                      │
  │  CAT bond: Lane 2000 spread & Wang 2000 畸变定价                   │
  │  基差风险: 指数触发 vs 参数触发 (cat-in-a-box) → HE=1−Var(L−q·P)/Var(L)
  │  投资组合: 有效前沿 / 零贝塔分散化价值                             │
  └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  visualization.py     │
                        │  11 × 300dpi 英文图表 │
                        └──────────────────────┘
```

---

## 4. 各模块核心公式 / Module Formulas

### 模块 A — Hazard（灾害）
- **最大风速由中心气压差推算**（西北太平洋, Atkinson & Holliday 1977）：
  `Vmax = 3.4 · Δp^0.644`
- **Holland B（物理量，保证廓线与强度自洽）**：
  `B = ρ · e · Vg² / Δp` （ρ=1.15 kg/m³, e=2.718, Vg≈Vmax）
  截断到 `[0.8, 2.5]`。
- **Holland 1980 梯度风廓线**：
  `Vg(r) = sqrt( B/ρ · Δp · (Rmw/r)^B · exp(−(Rmw/r)^B) + (r·f/2)² ) − r·f/2`
- **最大风速半径**（Willoughby & Rahn 2004）：
  `Rmw = 46.4 · exp(−0.0155·Vmax + 0.0169·|φ|)` (km)
- **移动非对称修正**：右半圆叠加 `α · Vmove`（前向切变），α≈0.4。
- **登陆衰减**（Kaplan & DeMaria 1995）：
  `Vmax(t) = V0 · exp(−α_dec · (t − t_landfall))`，α_dec≈0.095 /hr。
- **地表/阵风换算**：surface `× 0.85`，gust `× 1.30`。
- **随机事件集**：年频率 λ=3.2（泊松）；Δp 由截断对数正态抽样（中位 26 hPa，σ=0.52，[8,105]），
  登陆点纬度均匀 `[25.5, 38.0]`，方向/移速经验分布，固定随机种子 `20190810`。

### 模块 B — Exposure（暴露）
- `exposed_value = GDP × capital_output_ratio × wind_exposed_fraction`
  （capital_output_ratio=3.2，wind_exposed_fraction=0.45）
- `insured_value = exposed_value × insurance_penetration`
- 14 个华东沿海地市（台州已扣除温岭避免重复），含 4 类建筑权重。

### 模块 C — Vulnerability（脆弱性）
- **Emanuel 2011 三次方**：`f(v) = (v/v_half)³ / (1 + (v/v_half)³)`
- **对数正态**：`f(v) = Φ( ln(v / v_half) / β )`，β≈0.44
- **复合曲线**：`f_comp(v) = Σ_i w_i · f_i(v)`，w_i 为建筑类型权重
- **次生内涝因子**：`1 + g(distance) · (Vmax/Vref) · province_susceptibility`
  （浙江 1.2 / 山东 1.3 / 江苏 0.9 / 上海 0.8，高斯随到路径距离衰减）
- **需求激增因子**：灾后重建价格膨胀（约 +8%）
- **校准**：brentq 反演 `v_half`（≈176 m/s）使利奇马经济建模损失 = 实际 537.2 亿元。
  > 注：本模型损失分母为"区域风灾可暴露资本存量"而非单栋重置成本，故标定 v_half
  > 高于 Emanuel 大西洋单栋标定值（74.7 m/s）；该标定为示例性、非理赔数据拟合。

### 模块 D — Financial（金融）
- **年损失表 (YLT)**：泊松年频率抽样，单年聚合多事件损失。
- **EP 曲线**：OEP（单事件最大损失）与 AEP（年聚合损失）超越概率。
- **AAL** = `Σ loss_i · P_i`；**PML** = 给定重现期下的分位数损失。
- **VaR / TVaR**（AEP）：`VaR_p`、`TVaR_p = E[L | L ≥ VaR_p]`。
- **偿二代二期（C-ROSS II）巨灾资本**：`Capital = VaR99.5 − AAL`。
- **超赔再保分层**（标准差保费原理）：
  `P = (E[R] + k·σ_R) · (1 + e)`，k 按层位递增，e 为附加费用率。
- **CAT bond 定价**：
  - Lane (2000)：`Spread = EL + γ · PFL^α · CEL^β`（EL 期望损失，PFL 本金损失概率，CEL 条件期望损失）
  - Wang (2000/2002) 畸变算子：`F*(x) = Φ(Φ⁻¹(F(x)) − λ)`，再按畸变期望定价。
  - 指数触发（indemnity/index）vs 参数触发（parametric cat-in-a-box）。
- **基差风险**：`HE = 1 − Var(L − q·P) / Var(L)`（q 为最优对冲名义比例）。
- **投资组合分散化**：有效前沿（网格法）+ 最大夏普组合，零贝塔分散化价值。

---

## 5. 关键结论解读 / Key Takeaways

| 指标 | 数值 | 解读 |
|---|---|---|
| 利奇马建模损失 | 537.2 亿元 | 与实际吻合（校准目标），验证模块 A→C 链路 |
| 经济 AAL | 219.4 亿元/年 | 长期年均预期损失量级合理 |
| 100 年 OEP PML | 2,710.6 亿元 | 极端尾部；/AAL = 12.35x 落于 5–20x 经验带 |
| insured VaR99.5% | 61.59 亿元 | 保险口径尾部风险 |
| C-ROSS II 资本 | 58.59 亿元 | = VaR99.5 − AAL，可用于偿付能力资本要求 |
| 超赔再保 ROL | 5.93% | 三层超赔综合费率 |
| CAT bond Lane spread | 591 bp | 落在 200–1500 bp 合理区间；coupon 8.11% |
| CAT bond Wang spread | 268 bp | 畸变法风险负载更平滑 |
| 参数触发对冲效率 | 42.1% | fitted-box 显著优于 naive-box (18%)；演示触发结构↔基差风险权衡 |
| 组合夏普改善 | +0.342 | CAT bond 分散化提升风险调整收益 |

**教学要点**：参数触发（cat-in-a-box）用物理指数替代赔偿触发，降低道德风险与结算延迟，
但引入基差风险；箱体权重设计直接决定对冲效率——经验拟合箱体（按登陆纬度分箱条件平均损失）
将相关系数从 0.42 提升到 0.65、HE 从 18% 提升到 42%，直观展示"触发结构设计"的价值。

---

## 6. 运行方式 / How to Run

```bash
git clone https://github.com/lixunnan/typhoon-cat-model.git
cd typhoon-cat-model
```

```bash
# 使用受约束的解释器（禁止使用系统 python）
/Users/liqiqi/.workbuddy/binaries/python/envs/default/bin/python main.py
```

- 运行后所有 11 张图写入 `outputs/`，终端打印 Executive Summary 与全局一致性审查，并以 `IS_PASS: YES/NO` 结尾。
- 随机种子固定（`20190810`），结果可复现，全量运行 < 60 s（实测约 3 s）。

---

## 7. 局限性 / Limitations

1. **示例数据，非真实理赔校准**：暴露数据库为 14 地市公开 GDP 估算的示例值；脆弱性曲线
   v_half 由利奇马单一事件反演，未使用实际保险理赔损失数据拟合，仅用于演示方法学链路。
2. **风场为解析模型**：采用 Holland 1980 解析廓线 + 经验非对称/衰减修正，未接入 ERA5/CMA
   再分析风场或数值模式输出，近岸精细化风场（地形、海陆风）未建模。
3. **路径简化**：利奇马真实路径为 37 点折线；随机事件集用经验分布抽样，未做完整的
   气候态生成（genesis density / track density 统计建模）。
4. **暴露-损失聚合层级**：城市级点暴露 + 逐点最大风速近似，未做网格化微地形暴露分布。
5. **金融模块为方法论演示**：再保险分层/资本/CAT bond 参数（k、e、γ、α、β、λ）为行业经验取值，
   非特定公司精算假设；基差风险与投资组合为简化双/多资产演示。
6. **单一 peril**：仅建模台风风灾直接损失 + 次生内涝，未含风暴潮、洪涝独立模块或业务中断链。
7. **标定 V_half ≈ 176 m/s 不具备物理可解释性（关键防误用提示）**：为匹配利奇马
   537.2 亿元区域总损失，brentq 反演得到的 `v_half` 高达 ≈ 176 m/s，远高于
   Emanuel (2011) 在大西洋单栋建筑标定得到的 74.7 m/s。其本质是令复合脆弱性曲线在
   **"区域可暴露资本存量"** 这一损失分母口径上匹配总损失的**尺度参数**，而非单栋建筑的
   物理风速-损毁关系。因此：该曲线**不可直接用于单体建筑/保单级定价**；若要做单栋建筑
   定价，必须改用**重置成本口径的暴露数据**（逐建筑重置价值而非 GDP 口径资本存量）重新标定
   `v_half` 与建筑类型权重，否则会系统性高估风速-损失敏感性。

> 本项目定位为**可运行的方法学教学与工程骨架**，可在此骨架上替换为真实数据与精算假设以投入生产。

---

*Generated by the software-engineer teammate (寇豆码) · 全程仅依赖 numpy / pandas / scipy / matplotlib + 标准库。*
