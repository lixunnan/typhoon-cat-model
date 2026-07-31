"""主流程编排：台风巨灾风险模型 + 巨灾金融定价（利奇马 2019 校准案例）。

运行方式::

    python main.py

流程：
    1. Hazard   —— 载入利奇马路径，构建 Holland 风场，生成 10,000 场随机事件
    2. Exposure —— 载入华东沿海 14 市暴露数据库
    3. Vulnerability —— 校准脆弱性曲线至利奇马实际损失 537.2 亿元
    4. Financial —— EP 曲线 / VaR / TVaR / 偿二代资本 / 再保分层 /
                    CAT bond 定价 / 基差风险 / 组合分散化
    5. Visualization —— 输出 11 张 300 dpi 英文标注图表至 ``outputs/``
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 仅精确屏蔽确实需要抑制的告警，其余（FutureWarning/DeprecationWarning 等）保持可见，
# 以避免依赖升级后掩盖真实的兼容性问题。
# Only suppress the specific categories we genuinely need to hide; keep all other
# warnings (FutureWarning / DeprecationWarning / etc.) visible.
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"matplotlib\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r".*divide by zero encountered.*")

from config import FINANCIAL, HAZARD, OUTPUT_DIR, PLOT, STOCHASTIC, VULNERABILITY
import financial as fin
import hazard as hz
import visualization as viz
from exposure import load_exposure
from vulnerability import (
    calibrate_vulnerability,
    city_losses,
    lekima_province_comparison,
)

# --------------------------------------------------------------------------- #
# 终端报告工具
# --------------------------------------------------------------------------- #

_W = 92


def _rule(char: str = "=") -> str:
    """生成分隔线。

    Args:
        char: 填充字符。

    Returns:
        str: 长度为 ``_W`` 的分隔线。
    """
    return char * _W


def _header(title: str) -> None:
    """打印章节标题。

    Args:
        title: 标题文本。
    """
    print()
    print(_rule("="))
    print(f"  {title}")
    print(_rule("="))


def _sub(title: str) -> None:
    """打印小节标题。

    Args:
        title: 标题文本。
    """
    print()
    print(f"-- {title} " + "-" * max(0, _W - len(title) - 4))


def _kv(key: str, value: str, unit: str = "") -> None:
    """打印对齐的键值行。

    Args:
        key: 指标名。
        value: 指标值（已格式化）。
        unit: 单位。
    """
    print(f"  {key:<46s} {value:>22s} {unit}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run() -> Dict[str, object]:
    """执行完整的巨灾建模与定价流程。

    Returns:
        Dict[str, object]: 关键中间结果与最终指标，便于外部调用或测试。
    """
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figures: List[str] = []

    print(_rule("#"))
    print("#  TYPHOON CATASTROPHE RISK MODEL & CAT RISK FINANCING")
    print("#  Calibration case: Typhoon Lekima (1909), August 2019, East China")
    print("#  Currency unit throughout: CNY 100 million (亿元)")
    print(_rule("#"))

    # ----------------------------------------------------------------- #
    # Module A/B —— Hazard & Exposure
    # ----------------------------------------------------------------- #
    _header("MODULE A/B  |  HAZARD & EXPOSURE")

    track = hz.load_lekima_track()
    exposure = load_exposure()
    i_lf1, i_lf2 = hz.lekima_landfall_indices()

    _sub("Lekima best track")
    _kv("Track points (6-hourly)", f"{track.n_points}")
    _kv("Peak intensity", f"{track.pc.min():.0f}", "hPa / "
        f"{track.vmax.max():.0f} m/s")
    _kv("Landfall 1 (Wenling, Zhejiang)",
        f"{track.lon[i_lf1]:.2f}E {track.lat[i_lf1]:.2f}N",
        f"| {track.pc[i_lf1]:.0f} hPa, {track.vmax[i_lf1]:.0f} m/s")
    _kv("Landfall 2 (Huangdao, Qingdao)",
        f"{track.lon[i_lf2]:.2f}E {track.lat[i_lf2]:.2f}N",
        f"| {track.pc[i_lf2]:.0f} hPa, {track.vmax[i_lf2]:.0f} m/s")

    _sub("Exposure portfolio (illustrative, non-official data)")
    print(f"  {'City':<14s}{'Province':<11s}{'GDP':>10s}{'ExposedVal':>13s}"
          f"{'Penetr.':>10s}{'InsuredVal':>13s}")
    print(f"  {'':<14s}{'':<11s}{'(100mn)':>10s}{'(100mn)':>13s}"
          f"{'(%)':>10s}{'(100mn)':>13s}")
    for _, r in exposure.table.iterrows():
        print(f"  {r['city']:<14s}{r['province']:<11s}{r['gdp']:>10,.0f}"
              f"{r['exposed_value']:>13,.0f}{r['penetration'] * 100:>10.2f}"
              f"{r['insured_value']:>13,.1f}")
    print(f"  {'TOTAL':<25s}{exposure.table['gdp'].sum():>10,.0f}"
          f"{exposure.exposed_value.sum():>13,.0f}"
          f"{'':>10s}{exposure.table['insured_value'].sum():>13,.1f}")

    lekima_gust = hz.max_wind_field_over_track(
        track, exposure.lon, exposure.lat, HAZARD, return_gust=True)
    lekima_dist, lekima_dp = hz.min_distance_to_track(
        track, exposure.lon, exposure.lat)

    _sub("Lekima modelled peak 3-second gust by city (m/s)")
    for name, g, d in zip(exposure.names, lekima_gust, lekima_dist):
        print(f"  {name:<14s} gust = {g:6.1f} m/s   "
              f"min distance to track = {d:6.1f} km")

    # ----------------------------------------------------------------- #
    # Module C —— Vulnerability calibration
    # ----------------------------------------------------------------- #
    _header("MODULE C  |  VULNERABILITY CALIBRATION TO LEKIMA ACTUAL LOSS")

    calib_e = calibrate_vulnerability(
        lekima_gust, lekima_dist, lekima_dp, exposure,
        VULNERABILITY, curve="emanuel")
    calib_l = calibrate_vulnerability(
        lekima_gust, lekima_dist, lekima_dp, exposure,
        VULNERABILITY, curve="lognormal")

    print(f"  Target (reported direct economic loss of Lekima, mainland China):"
          f" {VULNERABILITY.lekima_actual_loss:,.1f} (CNY 100 mn)")
    print()
    print(f"  {'Curve':<14s}{'Parameter':<16s}{'Before':>12s}{'After':>12s}"
          f"{'LossBefore':>14s}{'LossAfter':>13s}{'ErrAfter':>11s}")
    for c in (calib_e, calib_l):
        print(f"  {c.curve:<14s}{c.param_name:<16s}{c.param_before:>12.3f}"
              f"{c.param_after:>12.3f}{c.loss_before:>14,.1f}"
              f"{c.loss_after:>13,.1f}{c.rel_error_after * 100:>10.3f}%")
    print()
    _kv("Calibrated V_half (Emanuel, base)", f"{calib_e.param_after:.2f}", "m/s")
    _kv("Calibration convergence", "YES" if calib_e.converged else "NO")
    print("  Note: the calibrated V_half exceeds Emanuel's Atlantic value (74.7 m/s)")
    print("        because the loss denominator here is the whole regional")
    print("        wind-exposed capital stock, not per-structure replacement cost.")

    econ_city, ins_city = city_losses(
        lekima_gust, lekima_dist, lekima_dp, exposure, calib_e.param_after,
        VULNERABILITY, curve="emanuel")
    prov_cmp = lekima_province_comparison(econ_city, exposure)

    _sub("Lekima hindcast by province (CNY 100 mn)")
    print(f"  {'Province':<12s}{'Modelled':>13s}{'Reported':>13s}"
          f"{'Diff':>12s}{'RelError':>12s}")
    for _, r in prov_cmp.iterrows():
        print(f"  {r['province']:<12s}{r['modelled']:>13,.1f}"
              f"{r['actual']:>13,.1f}{r['diff']:>12,.1f}"
              f"{r['rel_error'] * 100:>11.1f}%")
    print(f"  {'TOTAL':<12s}{econ_city.sum():>13,.1f}"
          f"{VULNERABILITY.lekima_actual_loss:>13,.1f}"
          f"{econ_city.sum() - VULNERABILITY.lekima_actual_loss:>12,.1f}"
          f"{calib_e.rel_error_after * 100:>11.2f}%")
    _kv("Lekima modelled INSURED loss", f"{ins_city.sum():,.2f}", "(CNY 100 mn)")

    # ----------------------------------------------------------------- #
    # Stochastic event set
    # ----------------------------------------------------------------- #
    _header("MODULE A (cont.)  |  STOCHASTIC EVENT SET")

    t0 = time.time()
    events = hz.generate_event_set(STOCHASTIC, HAZARD)
    gust_ev, dist_ev, dp_ev = hz.event_set_max_gust(
        events, exposure.lon, exposure.lat, HAZARD, STOCHASTIC.batch_size)
    econ_ev_city, ins_ev_city = city_losses(
        gust_ev, dist_ev, dp_ev, exposure, calib_e.param_after,
        VULNERABILITY, curve="emanuel")
    econ_ev = econ_ev_city.sum(axis=1)
    ins_ev = ins_ev_city.sum(axis=1)
    t_events = time.time() - t0

    _kv("Synthetic events", f"{events.n_events:,}")
    _kv("Annual frequency lambda (Poisson)",
        f"{events.freq_lambda:.2f}", "events/yr")
    _kv("Wind + loss computation time", f"{t_events:.2f}", "s")
    _kv("Mean event economic loss", f"{econ_ev.mean():,.2f}", "(CNY 100 mn)")
    _kv("Max event economic loss", f"{econ_ev.max():,.1f}", "(CNY 100 mn)")
    _kv("P(landfall dp >= 80 hPa)",
        f"{np.mean(events.landfall_dp >= 80.0) * 100:.2f}", "%")
    _kv("Implied RP of a Lekima-strength landfall",
        f"{1.0 / max(np.mean(events.landfall_dp >= 80.0) * events.freq_lambda, 1e-9):,.1f}",
        "yr")

    # ----------------------------------------------------------------- #
    # Module D —— Financial
    # ----------------------------------------------------------------- #
    _header("MODULE D  |  LOSS DISTRIBUTION AND RISK METRICS")

    ylt_econ = fin.build_year_loss_table(
        econ_ev, events.freq_lambda, STOCHASTIC.n_simulation_years,
        STOCHASTIC.random_seed)
    ylt_ins = fin.build_year_loss_table(
        ins_ev, events.freq_lambda, STOCHASTIC.n_simulation_years,
        STOCHASTIC.random_seed)
    m_econ = fin.compute_risk_metrics(ylt_econ, FINANCIAL)
    m_ins = fin.compute_risk_metrics(ylt_ins, FINANCIAL)

    _sub(f"Simulated {ylt_econ.n_years:,} years  |  all figures in CNY 100 mn")
    print(f"  {'Metric':<34s}{'Economic basis':>22s}{'Insured basis':>22s}")
    print(f"  {'-' * 78}")
    print(f"  {'AAL (average annual loss)':<34s}{m_econ.aal:>22,.2f}"
          f"{m_ins.aal:>22,.2f}")
    print(f"  {'Annual loss std. dev.':<34s}{m_econ.std:>22,.2f}"
          f"{m_ins.std:>22,.2f}")
    print(f"  {'Loss-free years':<34s}"
          f"{m_econ.loss_free_prob * 100:>21.2f}%"
          f"{m_ins.loss_free_prob * 100:>21.2f}%")
    print()
    print(f"  {'Return period':<18s}{'OEP (econ)':>16s}{'AEP (econ)':>16s}"
          f"{'OEP (insured)':>17s}{'AEP (insured)':>17s}")
    print(f"  {'-' * 84}")
    for rp in FINANCIAL.pml_return_periods:
        print(f"  {str(int(rp)) + '-year':<18s}"
              f"{m_econ.oep_pml[rp]:>16,.1f}{m_econ.aep_pml[rp]:>16,.1f}"
              f"{m_ins.oep_pml[rp]:>17,.2f}{m_ins.aep_pml[rp]:>17,.2f}")
    print()
    for lv in FINANCIAL.var_levels:
        print(f"  {'VaR ' + format(lv * 100, '.1f') + '% (AEP)':<34s}"
              f"{m_econ.var[lv]:>22,.1f}{m_ins.var[lv]:>22,.2f}")
    print(f"  {'TVaR 99.0% (AEP)':<34s}{m_econ.tvar:>22,.1f}"
          f"{m_ins.tvar:>22,.2f}")
    print(f"  {'C-ROSS II cat capital (VaR99.5-AAL)':<34s}"
          f"{m_econ.c_ross_capital:>22,.1f}{m_ins.c_ross_capital:>22,.2f}")
    print()
    _kv("PML(100yr, econ) / AAL(econ) ratio",
        f"{m_econ.oep_pml[100.0] / m_econ.aal:.2f}", "x")
    _kv("Lekima modelled loss vs OEP curve return period",
        f"{_loss_return_period(ylt_econ.occurrence, float(econ_city.sum())):,.1f}",
        "yr")

    # -------------------------- Reinsurance --------------------------- #
    _header("MODULE D  |  EXCESS-OF-LOSS REINSURANCE PRICING (INSURED BASIS)")

    layers = fin.build_reinsurance_program(ylt_ins, FINANCIAL)
    print(f"  {'Layer':<9s}{'Structure':>22s}{'AttRP':>8s}{'ExhRP':>8s}"
          f"{'EL':>10s}{'ELrate':>9s}{'ROL':>9s}{'Mult':>8s}{'P(att)':>9s}")
    print(f"  {'-' * 88}")
    for ly in layers:
        print(f"  {ly.name:<9s}"
              f"{f'{ly.limit:,.0f} xs {ly.attachment:,.0f}':>22s}"
              f"{ly.attach_rp:>8.0f}{ly.exhaust_rp:>8.0f}"
              f"{ly.expected_loss:>10,.3f}{ly.el_rate * 100:>8.2f}%"
              f"{ly.rate_on_line * 100:>8.2f}%{ly.multiple:>7.2f}x"
              f"{ly.prob_attach * 100:>8.2f}%")
    total_prem = sum(ly.loaded_premium for ly in layers)
    total_limit = sum(ly.limit for ly in layers)
    total_el = sum(ly.expected_loss for ly in layers)
    print(f"  {'-' * 88}")
    print(f"  {'PROGRAMME':<9s}{f'{total_limit:,.0f} total limit':>22s}"
          f"{'':>16s}{total_el:>10,.3f}"
          f"{total_el / total_limit * 100:>8.2f}%"
          f"{total_prem / total_limit * 100:>8.2f}%"
          f"{(total_prem / total_limit) / (total_el / total_limit):>7.2f}x")
    print()
    _kv("Total programme premium", f"{total_prem:,.3f}", "(CNY 100 mn)")
    _kv("Residual capital need after reinsurance",
        f"{max(m_ins.c_ross_capital - total_limit, 0.0):,.3f}", "(CNY 100 mn)")

    # ---------------------------- CAT bond ---------------------------- #
    _header("MODULE D  |  CATASTROPHE BOND PRICING")

    a_idx = fin._nice_round(
        float(fin.pml_at_return_periods(
            ylt_ins.occurrence, [FINANCIAL.catbond_attach_rp])[0]))
    e_idx = fin._nice_round(
        float(fin.pml_at_return_periods(
            ylt_ins.occurrence, [FINANCIAL.catbond_exhaust_rp])[0]))
    if e_idx <= a_idx:
        e_idx = a_idx * 1.5
    bond_index = fin.price_cat_bond(
        ylt_ins.occurrence, a_idx, e_idx, "industry loss index", FINANCIAL)

    # 参数触发的三种设计：
    #   v1 单一条件   —— 仅按登陆中心气压阶梯
    #   v2 多重条件   —— 气压阶梯 x 手工设定的 cat-in-a-box 位置权重
    #   v3 优化多重条件 —— 气压阶梯 x 由模型损失经验拟合的位置权重
    box_design = fin.design_location_box(
        events.landfall_lat, ins_ev, n_bins=10,
        lat_range=(STOCHASTIC.landfall_lat_min, STOCHASTIC.landfall_lat_max))
    payout_p1 = fin.parametric_payout_ratio(events.landfall_pc, FINANCIAL)
    payout_p2 = fin.parametric_payout_ratio(
        events.landfall_pc, FINANCIAL, events.landfall_lat)
    payout_p3 = fin.parametric_payout_ratio(
        events.landfall_pc, FINANCIAL, events.landfall_lat, box_design)

    def _annual_param_index(payout_ratio: np.ndarray) -> np.ndarray:
        """将事件级参数赔付比例折算为年度最大等价指数。

        Args:
            payout_ratio: 事件级赔付比例 ``(N,)``。

        Returns:
            np.ndarray: 年度最大等价指数 ``(Y,)`` (亿元)。
        """
        idx_ev = payout_ratio * e_idx
        annual = np.zeros(ylt_ins.n_years, dtype=float)
        np.maximum.at(annual, ylt_ins.event_year, idx_ev[ylt_ins.event_index])
        return annual

    bond_param = fin.price_cat_bond(
        _annual_param_index(payout_p3), a_idx, e_idx,
        "parametric cat-in-a-box (fitted)", FINANCIAL)
    bond_param_simple = fin.price_cat_bond(
        _annual_param_index(payout_p1), a_idx, e_idx,
        "parametric (Pc only)", FINANCIAL)

    _sub("Fitted cat-in-a-box location weights (from modelled losses)")
    print(f"  {'Latitude band':<20s}{'Events':>9s}{'Mean loss':>13s}{'Weight':>10s}")
    for b in range(box_design.weights.size):
        print(f"  {f'{box_design.edges[b]:.2f} - {box_design.edges[b + 1]:.2f} N':<20s}"
              f"{box_design.counts[b]:>9,.0f}{box_design.mean_loss[b]:>13,.3f}"
              f"{box_design.weights[b]:>10.3f}")
    print()

    print(f"  Trigger index attachment / exhaustion : "
          f"{a_idx:,.1f} / {e_idx:,.1f} (CNY 100 mn)")
    print(f"  Bond principal (tranche size)         : "
          f"{bond_index.principal:,.1f} (CNY 100 mn)")
    print(f"  Risk-free rate                        : "
          f"{FINANCIAL.risk_free_rate * 100:.2f}%")
    print()
    print(f"  {'Quantity':<38s}{'Index trigger':>20s}{'Parametric box':>22s}")
    print(f"  {'-' * 80}")
    rows = [
        ("Expected loss EL (bp)", bond_index.expected_loss * 1e4,
         bond_param.expected_loss * 1e4, "{:>20,.1f}", "{:>22,.1f}"),
        ("Prob. of first loss PFL (%)", bond_index.prob_first_loss * 100,
         bond_param.prob_first_loss * 100, "{:>20,.3f}", "{:>22,.3f}"),
        ("Prob. of total loss (%)", bond_index.prob_total_loss * 100,
         bond_param.prob_total_loss * 100, "{:>20,.3f}", "{:>22,.3f}"),
        ("Cond. expected loss CEL (%)", bond_index.cond_expected_loss * 100,
         bond_param.cond_expected_loss * 100, "{:>20,.2f}", "{:>22,.2f}"),
        ("Spread - Lane (2000) (bp)", bond_index.spread_lane * 1e4,
         bond_param.spread_lane * 1e4, "{:>20,.1f}", "{:>22,.1f}"),
        ("Spread - Wang transform (bp)", bond_index.spread_wang * 1e4,
         bond_param.spread_wang * 1e4, "{:>20,.1f}", "{:>22,.1f}"),
        ("Coupon = rf + spread, Lane (%)", bond_index.coupon_lane * 100,
         bond_param.coupon_lane * 100, "{:>20,.2f}", "{:>22,.2f}"),
        ("Coupon = rf + spread, Wang (%)", bond_index.coupon_wang * 100,
         bond_param.coupon_wang * 100, "{:>20,.2f}", "{:>22,.2f}"),
        ("Investor exp. return, Lane (%)",
         bond_index.investor_expected_return_lane * 100,
         bond_param.investor_expected_return_lane * 100, "{:>20,.2f}", "{:>22,.2f}"),
        ("Multiple = spread/EL, Lane (x)", bond_index.multiple_lane,
         bond_param.multiple_lane, "{:>20,.2f}", "{:>22,.2f}"),
        ("Multiple = spread/EL, Wang (x)", bond_index.multiple_wang,
         bond_param.multiple_wang, "{:>20,.2f}", "{:>22,.2f}"),
        ("Wang lambda used", bond_index.wang_lambda_used,
         bond_param.wang_lambda_used, "{:>20,.3f}", "{:>22,.3f}"),
        ("Wang lambda implied by Lane spread", bond_index.wang_lambda_implied,
         bond_param.wang_lambda_implied, "{:>20,.3f}", "{:>22,.3f}"),
    ]
    for label, v1, v2, f1, f2 in rows:
        print(f"  {label:<38s}" + f1.format(v1) + f2.format(v2))

    sens = fin.catbond_attachment_sensitivity(
        ylt_ins, [20, 30, 50, 75, 100, 150, 200, 300, 500], 5.0, FINANCIAL)
    _sub("Attachment sensitivity (index trigger)")
    print(f"  {'AttRP(yr)':>10s}{'Attach':>11s}{'Exhaust':>11s}{'EL(bp)':>10s}"
          f"{'PFL(%)':>9s}{'CEL(%)':>9s}{'Lane(bp)':>11s}{'Wang(bp)':>11s}"
          f"{'LaneMult':>10s}")
    for _, r in sens.iterrows():
        mult = r["spread_lane"] / r["el"] if r["el"] > 0 else float("nan")
        print(f"  {r['attach_rp']:>10,.0f}{r['attachment']:>11,.1f}"
              f"{r['exhaustion']:>11,.1f}{r['el'] * 1e4:>10,.1f}"
              f"{r['pfl'] * 100:>9,.3f}{r['cel'] * 100:>9,.2f}"
              f"{r['spread_lane'] * 1e4:>11,.1f}{r['spread_wang'] * 1e4:>11,.1f}"
              f"{mult:>9,.2f}x")

    # -------------------------- Basis risk ---------------------------- #
    _header("MODULE D  |  BASIS RISK OF THE PARAMETRIC TRIGGER")

    basis_simple = fin.analyse_basis_risk(ins_ev, payout_p1)
    basis_naive = fin.analyse_basis_risk(ins_ev, payout_p2)
    basis = fin.analyse_basis_risk(ins_ev, payout_p3)

    print(f"  {'Metric':<38s}{'Pc only':>14s}{'Pc x naive box':>17s}"
          f"{'Pc x fitted box':>18s}")
    print(f"  {'-' * 87}")
    triples = [
        ("Pearson correlation", "corr", "{:>14.4f}", "{:>17.4f}", "{:>18.4f}"),
        ("Spearman rank correlation", "rank", "{:>14.4f}", "{:>17.4f}",
         "{:>18.4f}"),
        ("Hedge effectiveness (%)", "he", "{:>14.2f}", "{:>17.2f}", "{:>18.2f}"),
        ("Optimal notional q* (CNY 100 mn)", "q", "{:>14,.2f}", "{:>17,.2f}",
         "{:>18,.2f}"),
        ("P(under-payment, L > P) (%)", "short", "{:>14.2f}", "{:>17.2f}",
         "{:>18.2f}"),
        ("P(over-payment, P > L) (%)", "wind", "{:>14.2f}", "{:>17.2f}",
         "{:>18.2f}"),
    ]
    getters = {
        "corr": lambda b: b.correlation,
        "rank": lambda b: b.rank_correlation,
        "he": lambda b: b.hedge_effectiveness * 100,
        "q": lambda b: b.optimal_notional,
        "short": lambda b: b.prob_shortfall * 100,
        "wind": lambda b: b.prob_windfall * 100,
    }
    for label, key, f1, f2, f3 in triples:
        g = getters[key]
        print(f"  {label:<38s}" + f1.format(g(basis_simple))
              + f2.format(g(basis_naive)) + f3.format(g(basis)))
    print()
    _kv("Hedge effectiveness gain: fitted vs Pc-only",
        f"{(basis.hedge_effectiveness - basis_simple.hedge_effectiveness) * 100:+.2f}",
        "pp")
    _kv("Hedge effectiveness gain: fitted vs naive box",
        f"{(basis.hedge_effectiveness - basis_naive.hedge_effectiveness) * 100:+.2f}",
        "pp")
    print("  Interpretation: a pressure-only trigger fires on intensity regardless")
    print("  of where the storm lands, so it pays when little exposure is at risk")
    print("  and fails to pay for weaker-but-well-aimed storms (HE = corr^2 at the")
    print("  optimal notional). A hand-set location box can even HURT if the boxes")
    print("  are misaligned with the exposure profile - here the naive box over-")
    print("  weights southern Zhejiang, while modelled losses peak for landfalls")
    print("  near 33-34N, which sweep the Shanghai-Suzhou-Nantong exposure core.")
    print("  Fitting the box to modelled losses is what actually cuts basis risk;")
    print("  the cost is a more complex, less transparent, model-dependent trigger.")

    # -------------------------- Portfolio ----------------------------- #
    _header("MODULE D  |  PORTFOLIO DIVERSIFICATION VALUE OF CAT BONDS")

    cb_returns = bond_index.coupon_lane - bond_index.loss_ratios
    cb_mu = float(cb_returns.mean())
    cb_sigma = float(cb_returns.std(ddof=1))
    port = fin.analyse_portfolio(cb_mu, cb_sigma, FINANCIAL)

    print(f"  {'Asset':<12s}{'E[r] (%)':>12s}{'Vol (%)':>11s}{'Sharpe':>10s}")
    for i, name in enumerate(port.asset_names):
        sh = (port.mu[i] - FINANCIAL.risk_free_rate) / port.sigma[i]
        print(f"  {name:<12s}{port.mu[i] * 100:>12.2f}"
              f"{port.sigma[i] * 100:>11.2f}{sh:>10.3f}")
    print()
    vo, ro, so, wo = port.best_without
    vw, rw, sw, ww = port.best_with
    print(f"  {'Max-Sharpe portfolio':<28s}{'Equity':>10s}{'Bond':>9s}"
          f"{'CATbond':>10s}{'E[r]%':>9s}{'Vol%':>8s}{'Sharpe':>9s}")
    print(f"  {'without CAT bond':<28s}{wo[0] * 100:>9.0f}%{wo[1] * 100:>8.0f}%"
          f"{'-':>10s}{ro * 100:>9.2f}{vo * 100:>8.2f}{so:>9.3f}")
    print(f"  {'with CAT bond':<28s}{ww[0] * 100:>9.0f}%{ww[1] * 100:>8.0f}%"
          f"{ww[2] * 100:>9.0f}%{rw * 100:>9.2f}{vw * 100:>8.2f}{sw:>9.3f}")
    _kv("Sharpe ratio improvement", f"{port.sharpe_improvement:+.3f}")
    _kv("CAT bond correlation with equity",
        f"{FINANCIAL.corr_equity_cat:.2f}")

    # ----------------------------------------------------------------- #
    # Visualisation
    # ----------------------------------------------------------------- #
    _header("VISUALISATION  |  WRITING FIGURES TO outputs/")

    figures.append(viz.plot_track(track, exposure))

    profiles: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    r_km = np.linspace(2.0, 400.0, 400)
    for dp in (20.0, 40.0, 60.0, 80.0, 100.0):
        vmax_s = float(hz.vmax_from_delta_p(np.array([dp]))[0])
        vmax_g = vmax_s / HAZARD.surface_reduction_factor
        rmw = float(hz.radius_max_wind_km(np.array([vmax_s]),
                                          np.array([28.4]), HAZARD)[0])
        b = float(hz.holland_b_physical(np.array([vmax_g]),
                                        np.array([dp]), HAZARD)[0])
        profiles[dp] = (r_km, hz.holland_gradient_wind(
            r_km, rmw, dp, b, 28.4))

    v_axis = np.linspace(15.0, 70.0, 200)
    rmw_axis = hz.radius_max_wind_km(v_axis, np.full_like(v_axis, 28.4), HAZARD)
    dp_axis = np.power(v_axis / 3.4, 1.0 / 0.644)
    b_phys = hz.holland_b_physical(
        v_axis / HAZARD.surface_reduction_factor, dp_axis, HAZARD)
    b_vick = hz.holland_b_vickery(v_axis, rmw_axis,
                                  np.full_like(v_axis, 28.4), HAZARD)
    figures.append(viz.plot_holland_profiles(
        profiles, (v_axis, rmw_axis), (v_axis, b_phys, b_vick)))

    g_lon, g_lat, g_gust = hz.wind_field_grid(track, (117.0, 126.0),
                                              (25.0, 39.0), 150, HAZARD)
    figures.append(viz.plot_wind_field(g_lon, g_lat, g_gust, track,
                                       exposure, lekima_gust))
    figures.append(viz.plot_vulnerability(calib_e, calib_l, exposure))
    figures.append(viz.plot_lekima_validation(exposure, econ_city,
                                              prov_cmp, calib_e))
    figures.append(viz.plot_event_set_overview(
        events.landfall_lat, events.landfall_dp, econ_ev,
        float(econ_city.sum()), events.freq_lambda))
    figures.append(viz.plot_ep_and_distribution(ylt_ins, m_ins))
    figures.append(viz.plot_reinsurance_layers(layers, m_ins))
    figures.append(viz.plot_catbond_pricing(sens, bond_index, bond_param,
                                            FINANCIAL.risk_free_rate))
    figures.append(viz.plot_basis_risk(basis, basis_simple, basis_naive))
    figures.append(viz.plot_portfolio(port, FINANCIAL.risk_free_rate))

    for f in figures:
        size_kb = os.path.getsize(f) / 1024.0
        print(f"  [OK] {os.path.basename(f):<34s} {size_kb:>8.1f} KB")

    # ----------------------------------------------------------------- #
    # Executive summary
    # ----------------------------------------------------------------- #
    _header("EXECUTIVE SUMMARY  |  KEY NUMBERS")

    _kv("Lekima calibrated modelled loss",
        f"{econ_city.sum():,.1f}", f"vs actual {VULNERABILITY.lekima_actual_loss:,.1f}"
        f" (CNY 100 mn)")
    _kv("Calibrated Emanuel V_half", f"{calib_e.param_after:.2f}", "m/s")
    _kv("AAL - economic basis", f"{m_econ.aal:,.1f}", "(CNY 100 mn / yr)")
    _kv("AAL - insured basis", f"{m_ins.aal:,.2f}", "(CNY 100 mn / yr)")
    _kv("100-yr OEP PML - economic", f"{m_econ.oep_pml[100.0]:,.1f}",
        "(CNY 100 mn)")
    _kv("250-yr OEP PML - economic", f"{m_econ.oep_pml[250.0]:,.1f}",
        "(CNY 100 mn)")
    _kv("100-yr OEP PML - insured", f"{m_ins.oep_pml[100.0]:,.2f}",
        "(CNY 100 mn)")
    _kv("VaR 99.5% (insured, AEP)", f"{m_ins.var[0.995]:,.2f}",
        "(CNY 100 mn)")
    _kv("TVaR 99% (insured, AEP)", f"{m_ins.tvar:,.2f}", "(CNY 100 mn)")
    _kv("C-ROSS II cat capital (insured)", f"{m_ins.c_ross_capital:,.2f}",
        "(CNY 100 mn)")
    _kv("Reinsurance programme ROL (blended)",
        f"{total_prem / total_limit * 100:.2f}", "%")
    _kv("CAT bond spread - Lane", f"{bond_index.spread_lane * 1e4:,.0f}", "bp")
    _kv("CAT bond spread - Wang", f"{bond_index.spread_wang * 1e4:,.0f}", "bp")
    _kv("CAT bond coupon (Lane)", f"{bond_index.coupon_lane * 100:.2f}", "%")
    _kv("Parametric hedge effectiveness",
        f"{basis.hedge_effectiveness * 100:.1f}", "%")
    _kv("Sharpe improvement from CAT bond",
        f"{port.sharpe_improvement:+.3f}")

    elapsed = time.time() - t_start
    print()
    _kv("Total runtime", f"{elapsed:.2f}", "s")
    print(_rule("#"))

    return {
        "track": track, "exposure": exposure, "events": events,
        "calib_emanuel": calib_e, "calib_lognormal": calib_l,
        "lekima_econ_city": econ_city, "lekima_gust": lekima_gust,
        "econ_events": econ_ev, "ins_events": ins_ev,
        "ylt_econ": ylt_econ, "ylt_ins": ylt_ins,
        "metrics_econ": m_econ, "metrics_ins": m_ins,
        "layers": layers, "bond_index": bond_index, "bond_param": bond_param,
        "sensitivity": sens, "basis": basis, "portfolio": port,
        "figures": figures, "runtime_s": elapsed,
    }


def _loss_return_period(occurrence: np.ndarray, loss: float) -> float:
    """给定损失金额在 OEP 曲线上的重现期。

    Args:
        occurrence: 年度最大事件损失序列 ``(Y,)``。
        loss: 目标损失金额 (亿元)。

    Returns:
        float: 重现期 (年)；若从未被超越则返回模拟年数。
    """
    p = float(np.mean(np.asarray(occurrence, dtype=float) >= loss))
    return 1.0 / p if p > 0 else float(occurrence.size)


def consistency_check(res: Dict[str, object]) -> bool:
    """全局数值合理性与一致性审查。

    Args:
        res: ``run()`` 返回的结果字典。

    Returns:
        bool: 全部检查通过返回 True。
    """
    _header("GLOBAL CONSISTENCY REVIEW")
    checks: List[Tuple[str, bool, str]] = []

    calib = res["calib_emanuel"]
    m_econ = res["metrics_econ"]
    m_ins = res["metrics_ins"]
    layers = res["layers"]
    bond = res["bond_index"]
    basis = res["basis"]
    lek = float(np.sum(res["lekima_econ_city"]))

    checks.append(("Lekima modelled loss within 450-620 (CNY 100 mn)",
                   450.0 <= lek <= 620.0, f"{lek:,.1f}"))
    checks.append(("Economic AAL within 10-1000 (CNY 100 mn)",
                   10.0 <= m_econ.aal <= 1000.0, f"{m_econ.aal:,.1f}"))
    ratio = m_econ.oep_pml[100.0] / m_econ.aal
    checks.append(("100-yr PML / AAL within 3-25x", 3.0 <= ratio <= 25.0,
                   f"{ratio:.2f}x"))
    checks.append(("OEP PML monotonically increasing in RP",
                   all(m_econ.oep_pml[a] <= m_econ.oep_pml[b]
                       for a, b in zip(FINANCIAL.pml_return_periods[:-1],
                                       FINANCIAL.pml_return_periods[1:])),
                   "monotone"))
    checks.append(("AEP >= OEP at every return period",
                   all(m_econ.aep_pml[rp] >= m_econ.oep_pml[rp] - 1e-6
                       for rp in FINANCIAL.pml_return_periods), "ok"))
    checks.append(("TVaR99 >= VaR99", m_ins.tvar >= m_ins.var[0.99],
                   f"{m_ins.tvar:,.2f} >= {m_ins.var[0.99]:,.2f}"))
    checks.append(("C-ROSS capital > 0", m_ins.c_ross_capital > 0,
                   f"{m_ins.c_ross_capital:,.2f}"))
    checks.append(("All layer multiples within 1.0-6.0x",
                   all(1.0 <= ly.multiple <= 6.0 for ly in layers),
                   ", ".join(f"{ly.multiple:.2f}x" for ly in layers)))
    checks.append(("All layer ROL > EL rate",
                   all(ly.rate_on_line > ly.el_rate for ly in layers), "ok"))
    checks.append(("CAT bond spread within 200-1500 bp",
                   200.0 <= bond.spread_lane * 1e4 <= 1500.0,
                   f"{bond.spread_lane * 1e4:,.0f} bp"))
    checks.append(("CAT bond spread > EL (positive risk premium)",
                   bond.spread_lane > bond.expected_loss,
                   f"{bond.spread_lane * 1e4:,.0f} > "
                   f"{bond.expected_loss * 1e4:,.0f} bp"))
    checks.append(("Wang spread positive and finite",
                   np.isfinite(bond.spread_wang) and bond.spread_wang > 0,
                   f"{bond.spread_wang * 1e4:,.0f} bp"))
    checks.append(("Hedge effectiveness within 0-1",
                   0.0 <= basis.hedge_effectiveness <= 1.0,
                   f"{basis.hedge_effectiveness * 100:.1f}%"))
    checks.append(("Basis-risk correlation positive",
                   basis.correlation > 0, f"{basis.correlation:.3f}"))
    checks.append(("Sharpe improvement non-negative",
                   res["portfolio"].sharpe_improvement >= -1e-9,
                   f"{res['portfolio'].sharpe_improvement:+.3f}"))
    checks.append(("Calibration relative error < 1%",
                   abs(calib.rel_error_after) < 0.01,
                   f"{calib.rel_error_after * 100:+.3f}%"))
    checks.append(("All figures generated and > 20 KB",
                   all(os.path.getsize(f) > 20 * 1024 for f in res["figures"]),
                   f"{len(res['figures'])} figures"))
    checks.append(("Runtime under 60 s", res["runtime_s"] < 60.0,
                   f"{res['runtime_s']:.2f} s"))

    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52s} {detail}")

    all_ok = all(c[1] for c in checks)
    print()
    print(f"  IS_PASS: {'YES' if all_ok else 'NO'}")
    print(_rule("#"))
    return all_ok


def main() -> int:
    """程序入口。

    Returns:
        int: 退出码，0 表示成功且全部一致性检查通过。
    """
    res = run()
    ok = consistency_check(res)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
