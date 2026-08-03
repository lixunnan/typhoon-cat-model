"""绘图模块：全部图表使用英文标签，避免中文字体缺失导致的方块字。

统一采用 ``config.PlotConfig`` 中的专业色板与 300 dpi 输出。
所有函数均返回保存后的文件绝对路径。
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from config import COASTLINE, OUTPUT_DIR, PLOT, VULNERABILITY, PlotConfig
from exposure import BUILDING_TYPES, BUILDING_TYPE_LABELS, ExposureDatabase
from financial import (
    BasisRiskResult,
    CatBondResult,
    LayerResult,
    PortfolioResult,
    RiskMetrics,
    YearLossTable,
    ep_curve,
)
from hazard import Track, lekima_landfall_indices
from vulnerability import CalibrationResult, composite_emanuel_mdr, emanuel_mdr

if TYPE_CHECKING:  # 仅类型检查期导入，运行期避免 climate <-> visualization 耦合
    from climate import ClimateAnalysis

# 仅精确屏蔽 matplotlib 字体缺失类 UserWarning（英文标签下仍可能偶发），
# 以及 numpy 数值计算中可能触发的 RuntimeWarning（除零/无效值）。
# 其余告警（FutureWarning / DeprecationWarning 等）保持可见，以便依赖升级时暴露兼容性问题。
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"matplotlib\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r".*divide by zero encountered.*")

_COAST_LAT = np.array([p[0] for p in COASTLINE], dtype=float)
_COAST_LON = np.array([p[1] for p in COASTLINE], dtype=float)


def _init_style(cfg: PlotConfig = PLOT) -> None:
    """初始化 matplotlib 全局样式（英文字体，专业配色）。

    Args:
        cfg: 绘图配置。
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": cfg.bg,
        "axes.facecolor": cfg.bg,
        "axes.edgecolor": "#95A5A6",
        "axes.labelcolor": cfg.text_color,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": cfg.text_color,
        "ytick.color": cfg.text_color,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "grid.color": cfg.grid_color,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.facecolor": cfg.bg,
    })


def _ensure_output_dir(path: str = OUTPUT_DIR) -> str:
    """确保输出目录存在。

    Args:
        path: 目录路径。

    Returns:
        str: 目录绝对路径。
    """
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig: plt.Figure, filename: str, cfg: PlotConfig = PLOT) -> str:
    """保存并关闭图形。

    Args:
        fig: matplotlib Figure 对象。
        filename: 文件名（含扩展名）。
        cfg: 绘图配置。

    Returns:
        str: 保存后的文件绝对路径。
    """
    out = os.path.join(_ensure_output_dir(), filename)
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return out


def _draw_coastline(ax: plt.Axes, color: str = "#5D6D7E", lw: float = 1.4) -> None:
    """在坐标轴上绘制简化的中国东部海岸线并填充陆地。

    Args:
        ax: 目标坐标轴。
        color: 岸线颜色。
        lw: 线宽。
    """
    ax.plot(_COAST_LON, _COAST_LAT, color=color, lw=lw, zorder=1)
    ax.fill_betweenx(
        _COAST_LAT, np.full_like(_COAST_LON, 112.0), _COAST_LON,
        color="#EAEDED", zorder=0,
    )


# --------------------------------------------------------------------------- #
# Figure 1 — Lekima track
# --------------------------------------------------------------------------- #


def plot_track(
    track: Track, exposure: ExposureDatabase, cfg: PlotConfig = PLOT
) -> str:
    """绘制利奇马路径图（强度着色 + 两次登陆标注 + 暴露城市）。

    Args:
        track: 利奇马路径。
        exposure: 暴露数据库。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, ax = plt.subplots(figsize=(9.2, 8.6))
    _draw_coastline(ax)

    ax.plot(track.lon, track.lat, color="#34495E", lw=1.1, ls="--",
            alpha=0.8, zorder=2)
    sizes = 10.0 + (track.vmax ** 1.75) / 8.0
    sc = ax.scatter(track.lon, track.lat, c=track.vmax, s=sizes,
                    cmap="turbo", vmin=10, vmax=65, edgecolor="white",
                    linewidth=0.5, zorder=3)

    i1, i2 = lekima_landfall_indices()
    for idx, label, dx, dy in (
        (i1, "Landfall 1: Wenling, Zhejiang\n2019-08-10 01:45 BJT\n930 hPa / 52 m/s",
         -6.4, -1.4),
        (i2, "Landfall 2: Huangdao, Qingdao\n2019-08-11 20:50 BJT\n992 hPa / 23 m/s",
         -7.4, 1.0),
    ):
        ax.plot(track.lon[idx], track.lat[idx], marker="*", ms=22,
                color=cfg.palette[4], markeredgecolor="white",
                markeredgewidth=1.0, zorder=5)
        ax.annotate(
            label, xy=(track.lon[idx], track.lat[idx]),
            xytext=(track.lon[idx] + dx, track.lat[idx] + dy),
            fontsize=8.5, color=cfg.text_color,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDFEFE",
                      ec=cfg.palette[4], lw=0.9, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=cfg.palette[4], lw=1.1),
            zorder=6,
        )

    ax.scatter(exposure.lon, exposure.lat, marker="s", s=42,
               facecolor=cfg.palette[0], edgecolor="white", linewidth=0.7,
               zorder=4, label="Exposure city")
    for _, row in exposure.table.iterrows():
        ax.annotate(row["city"], (row["lon"], row["lat"]),
                    xytext=(4.5, 3.5), textcoords="offset points",
                    fontsize=7.2, color=cfg.palette[0])

    ax.annotate("Genesis 2019-08-04\nNW Pacific",
                xy=(track.lon[0], track.lat[0]), xytext=(132.0, 11.6),
                fontsize=8.5, color=cfg.text_color,
                arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.0))

    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.82)
    cb.set_label("Max sustained wind (m/s, 1-min at 10 m)", fontsize=9)

    ax.set_xlim(113.0, 140.0)
    ax.set_ylim(10.0, 41.0)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Fig.1  Typhoon Lekima (1909) Best Track and Exposure Portfolio",
                 pad=12)
    ax.grid(True, ls=":", alpha=0.55)
    ax.legend(loc="upper right")
    return _save(fig, "fig01_lekima_track.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 2 — Holland radial profiles
# --------------------------------------------------------------------------- #


def plot_holland_profiles(
    profiles: Dict[float, Tuple[np.ndarray, np.ndarray]],
    rmw_curve: Tuple[np.ndarray, np.ndarray],
    b_curve: Tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: PlotConfig = PLOT,
) -> str:
    """绘制 Holland 风廓线剖面与参数关系。

    Args:
        profiles: ``{delta_p: (r_km, v_ms)}`` 的风廓线字典。
        rmw_curve: ``(vmax_ms, rmw_km)`` 最大风速半径经验关系。
        b_curve: ``(vmax_ms, B_physical, B_vickery)`` 形状参数对比。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))

    ax = axes[0]
    for i, (dp, (r, v)) in enumerate(sorted(profiles.items())):
        ax.plot(r, v, lw=2.0, color=cfg.palette[i % len(cfg.palette)],
                label=f"$\\Delta p$ = {dp:.0f} hPa")
    ax.axhline(32.7, color="#95A5A6", ls=":", lw=1.0)
    ax.text(305, 33.6, "Typhoon threshold (32.7 m/s)", fontsize=7.6,
            color="#7F8C8D")
    ax.set_xlabel("Radius from centre (km)")
    ax.set_ylabel("Gradient wind speed (m/s)")
    ax.set_title("(a) Holland (1980) radial wind profiles")
    ax.set_xlim(0, 400)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()

    ax = axes[1]
    ax.plot(rmw_curve[0], rmw_curve[1], lw=2.2, color=cfg.palette[1])
    ax.fill_between(rmw_curve[0], rmw_curve[1], alpha=0.15, color=cfg.palette[1])
    ax.set_xlabel("Max sustained wind (m/s)")
    ax.set_ylabel("Radius of max wind $R_{mw}$ (km)")
    ax.set_title("(b) Willoughby & Rahn (2004) $R_{mw}$ relation")
    ax.grid(True, ls=":", alpha=0.6)

    ax = axes[2]
    ax.plot(b_curve[0], b_curve[1], lw=2.2, color=cfg.palette[0],
            label="Physical  $B=\\rho e V_g^2/\\Delta p$")
    ax.plot(b_curve[0], b_curve[2], lw=2.2, ls="--", color=cfg.palette[4],
            label="Empirical (Vickery-type)")
    ax.set_xlabel("Max sustained wind (m/s)")
    ax.set_ylabel("Holland shape parameter $B$")
    ax.set_title("(c) Shape parameter $B$: physical vs empirical")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()

    fig.suptitle("Fig.2  Parametric Wind Field Model Components",
                 fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, "fig02_holland_profiles.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 3 — 2-D wind field
# --------------------------------------------------------------------------- #


def plot_wind_field(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    gust: np.ndarray,
    track: Track,
    exposure: ExposureDatabase,
    city_gust: np.ndarray,
    cfg: PlotConfig = PLOT,
) -> str:
    """绘制利奇马过程最大阵风的二维空间分布。

    Args:
        grid_lon: 经度网格 ``(n, n)``。
        grid_lat: 纬度网格 ``(n, n)``。
        gust: 过程最大阵风场 ``(n, n)`` (m/s)。
        track: 利奇马路径。
        exposure: 暴露数据库。
        city_gust: 各城市过程最大阵风 ``(C,)`` (m/s)。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, ax = plt.subplots(figsize=(8.8, 8.8))

    levels = np.arange(0, 82, 4)
    cs = ax.contourf(grid_lon, grid_lat, gust, levels=levels,
                     cmap="Spectral_r", extend="max", alpha=0.92)
    ax.contour(grid_lon, grid_lat, gust, levels=[17.2, 32.7, 51.0],
               colors=["#2C3E50", "#2C3E50", "#2C3E50"],
               linewidths=[0.7, 1.0, 1.4], linestyles=["--", "-", "-"])

    ax.plot(_COAST_LON, _COAST_LAT, color="#1C2833", lw=1.6, zorder=3)
    ax.plot(track.lon, track.lat, color="black", lw=1.8, ls="--",
            zorder=4, label="Lekima track")

    sc = ax.scatter(exposure.lon, exposure.lat, c=city_gust, s=150,
                    cmap="Spectral_r", vmin=0, vmax=80, marker="o",
                    edgecolor="black", linewidth=1.1, zorder=5)
    for (_, row), g in zip(exposure.table.iterrows(), city_gust):
        ax.annotate(f"{row['city']}\n{g:.0f}", (row["lon"], row["lat"]),
                    xytext=(6, -10), textcoords="offset points",
                    fontsize=7.0, fontweight="bold", color="#111111",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.72), zorder=6)

    cb = fig.colorbar(cs, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Peak 3-second gust over the event (m/s)", fontsize=9)

    ax.set_xlim(grid_lon.min(), grid_lon.max())
    ax.set_ylim(grid_lat.min(), grid_lat.max())
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Fig.3  Lekima Peak Gust Footprint over East China", pad=12)
    ax.legend(loc="lower left")
    _ = sc
    return _save(fig, "fig03_wind_field.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 4 — Vulnerability curves
# --------------------------------------------------------------------------- #


def plot_vulnerability(
    calib_emanuel: CalibrationResult,
    calib_lognormal: CalibrationResult,
    exposure: ExposureDatabase,
    cfg: PlotConfig = PLOT,
) -> str:
    """绘制脆弱性曲线族（分建筑类型 + 校准前后对比）。

    Args:
        calib_emanuel: Emanuel 曲线校准结果。
        calib_lognormal: 对数正态曲线校准结果。
        exposure: 暴露数据库。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    vcfg = VULNERABILITY
    v = np.linspace(0, 110, 400)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))

    ax = axes[0]
    for i, btype in enumerate(BUILDING_TYPES):
        vh = calib_emanuel.param_after * vcfg.building_v_half_factor[btype]
        ax.plot(v, emanuel_mdr(v, vcfg.v_thresh_ms, vh) * 100.0, lw=2.1,
                color=cfg.palette[i], label=BUILDING_TYPE_LABELS[btype])
    ax.axvline(vcfg.v_thresh_ms, color="#95A5A6", ls=":", lw=1.0)
    ax.text(vcfg.v_thresh_ms + 1.2, 0.5, "$V_{thresh}$", fontsize=8,
            color="#7F8C8D")
    ax.set_xlabel("Peak 3-second gust (m/s)")
    ax.set_ylabel("Mean damage ratio (%)")
    ax.set_title("(a) Emanuel (2011) curves by building class\n(calibrated)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=8)

    ax = axes[1]
    from vulnerability import lognormal_mdr  # 局部导入避免顶层循环依赖风险
    for i, btype in enumerate(BUILDING_TYPES):
        mu, sigma, mdr_max = vcfg.lognormal_params[btype]
        ax.plot(v, lognormal_mdr(v, mu * calib_lognormal.param_after,
                                 sigma, mdr_max) * 100.0,
                lw=2.1, color=cfg.palette[i], label=BUILDING_TYPE_LABELS[btype])
    ax.set_xlabel("Peak 3-second gust (m/s)")
    ax.set_ylabel("Mean damage ratio (%)")
    ax.set_title("(b) Lognormal CDF curves by building class\n(calibrated)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=8)

    ax = axes[2]
    shares = exposure.building_shares
    w = (exposure.exposed_value / exposure.exposed_value.sum()).reshape(-1, 1)
    avg_share = (shares * w).sum(axis=0, keepdims=True)
    mdr_pre = composite_emanuel_mdr(
        np.tile(v.reshape(-1, 1), (1, 1)), avg_share, calib_emanuel.param_before
    ).ravel()
    mdr_post = composite_emanuel_mdr(
        np.tile(v.reshape(-1, 1), (1, 1)), avg_share, calib_emanuel.param_after
    ).ravel()
    ax.plot(v, mdr_pre * 100.0, lw=2.4, ls="--", color=cfg.palette[4],
            label=f"Before calibration  $V_{{half}}$={calib_emanuel.param_before:.1f} m/s")
    ax.plot(v, mdr_post * 100.0, lw=2.6, color=cfg.palette[0],
            label=f"After calibration  $V_{{half}}$={calib_emanuel.param_after:.1f} m/s")
    ax.fill_between(v, mdr_post * 100.0, mdr_pre * 100.0, color=cfg.palette[4],
                    alpha=0.10)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e2)
    ax.set_xlabel("Peak 3-second gust (m/s)")
    ax.set_ylabel("Portfolio-weighted MDR (%, log scale)")
    ax.set_title("(c) Calibration to Lekima actual loss\n(CNY 53.72 bn)")
    ax.grid(True, ls=":", alpha=0.6, which="both")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Fig.4  Vulnerability Functions and Loss Calibration",
                 fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, "fig04_vulnerability.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 5 — Lekima modelled vs actual
# --------------------------------------------------------------------------- #


def plot_lekima_validation(
    exposure: ExposureDatabase,
    city_loss: np.ndarray,
    province_cmp: pd.DataFrame,
    calib: CalibrationResult,
    cfg: PlotConfig = PLOT,
) -> str:
    """绘制利奇马模拟损失与实际损失的对比。

    Args:
        exposure: 暴露数据库。
        city_loss: 分城市模拟损失 ``(C,)`` (亿元)。
        province_cmp: 分省对比表。
        calib: 校准结果。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    order = np.argsort(city_loss)[::-1]
    names = [exposure.names[i] for i in order]
    vals = city_loss[order]
    prov = exposure.table["province"].to_numpy()[order]
    pcolor = {"Zhejiang": cfg.palette[0], "Jiangsu": cfg.palette[1],
              "Shandong": cfg.palette[3], "Shanghai": cfg.palette[5]}
    colors = [pcolor.get(p, cfg.palette[6]) for p in prov]

    ax = axes[0]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8)
    for b, vv in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.015,
                f"{vv:.1f}", ha="center", va="bottom", fontsize=7.6,
                color=cfg.text_color)
    ax.set_ylabel("Modelled direct economic loss (CNY 100 mn)")
    ax.set_title("(a) City-level modelled loss for Lekima")
    # 旋转更大角度并缩小字号，避免 "Taizhou-ZJ" 与相邻 "Wenling" 等标签重叠。
    # Steeper rotation + smaller font avoids x-label overlap (e.g. "Taizhou-ZJ" vs "Wenling").
    ax.tick_params(axis="x", rotation=65, labelsize=7.2)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in pcolor.values()]
    ax.legend(handles, list(pcolor.keys()), title="Province", fontsize=8,
              title_fontsize=8.5)

    ax = axes[1]
    x = np.arange(len(province_cmp))
    w = 0.38
    ax.bar(x - w / 2, province_cmp["modelled"], w, label="Modelled",
           color=cfg.palette[0], edgecolor="white")
    ax.bar(x + w / 2, province_cmp["actual"], w, label="Reported actual",
           color=cfg.palette[3], edgecolor="white")
    for i, (m, a) in enumerate(zip(province_cmp["modelled"],
                                   province_cmp["actual"])):
        ax.text(i - w / 2, m, f"{m:.0f}", ha="center", va="bottom", fontsize=7.6)
        ax.text(i + w / 2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(province_cmp["province"], rotation=15)
    ax.set_ylabel("Direct economic loss (CNY 100 mn)")
    ax.set_title("(b) Province-level: modelled vs reported")
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    ax.legend(fontsize=8)

    txt = (f"Total modelled: {calib.loss_after:,.1f} (CNY 100 mn)\n"
           f"Reported actual: {calib.target_loss:,.1f}\n"
           f"Relative error: {calib.rel_error_after * 100:+.2f}%")
    ax.text(0.98, 0.62, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.6, bbox=dict(boxstyle="round,pad=0.4", fc="#FDFEFE",
                                    ec=cfg.palette[0], lw=0.9))

    fig.suptitle("Fig.5  Lekima Loss Hindcast Validation",
                 fontsize=13, fontweight="bold", y=1.00)
    return _save(fig, "fig05_lekima_validation.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 6 — EP curves + annual loss distribution
# --------------------------------------------------------------------------- #


def plot_ep_and_distribution(
    ylt: YearLossTable, metrics: RiskMetrics, cfg: PlotConfig = PLOT
) -> str:
    """绘制 OEP/AEP 曲线与年度损失分布（含 VaR/TVaR 标注）。

    Args:
        ylt: 年度损失表。
        metrics: 风险度量。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.0))

    ax = axes[0]
    for name, series, color in (
        ("OEP (occurrence)", ylt.occurrence, cfg.palette[0]),
        ("AEP (aggregate)", ylt.aggregate, cfg.palette[4]),
    ):
        x, _, rp = ep_curve(series)
        sel = rp >= 1.05
        ax.plot(rp[sel], x[sel], lw=2.3, color=color, label=name)

    for rp_mark in (100.0, 250.0):
        v = metrics.oep_pml.get(rp_mark)
        if v is None:
            continue
        ax.axvline(rp_mark, color="#95A5A6", ls=":", lw=0.9)
        ax.plot([rp_mark], [v], marker="o", ms=7, color=cfg.palette[0],
                markeredgecolor="white", zorder=5)
        ax.annotate(f"{rp_mark:.0f}-yr OEP PML\n{v:,.0f}",
                    xy=(rp_mark, v), xytext=(rp_mark * 0.18, v * 1.06),
                    fontsize=8.4,
                    bbox=dict(boxstyle="round,pad=0.32", fc="#FDFEFE",
                              ec=cfg.palette[0], lw=0.9),
                    arrowprops=dict(arrowstyle="->", color=cfg.palette[0],
                                    lw=1.0))

    ax.axhline(metrics.aal, color=cfg.palette[2], ls="--", lw=1.6)
    ax.text(1.15, metrics.aal * 1.05, f"AAL = {metrics.aal:,.1f}",
            fontsize=8.6, color=cfg.palette[2], fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_xlabel("Return period (years, log scale)")
    ax.set_ylabel("Loss (CNY 100 mn)")
    ax.set_title("(a) Exceedance probability curves")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(loc="upper left")

    ax = axes[1]
    agg = ylt.aggregate
    upper = float(np.quantile(agg, 0.9985))
    bins = np.linspace(0.0, upper, 90)
    ax.hist(np.clip(agg, 0, upper), bins=bins, color=cfg.palette[1],
            alpha=0.80, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")

    marks = [
        (metrics.aal, cfg.palette[2], f"AAL = {metrics.aal:,.1f}"),
        (metrics.var[0.99], cfg.palette[3], f"VaR 99% = {metrics.var[0.99]:,.0f}"),
        (metrics.var[0.995], cfg.palette[5], f"VaR 99.5% = {metrics.var[0.995]:,.0f}"),
        (metrics.tvar, cfg.palette[4], f"TVaR 99% = {metrics.tvar:,.0f}"),
    ]
    for i, (xv, c, lbl) in enumerate(marks):
        ax.axvline(xv, color=c, ls="--", lw=1.8)
        ax.text(xv, ax.get_ylim()[1] * (0.42 ** (i * 0.55 + 0.15)), " " + lbl,
                rotation=90, va="top", ha="left", fontsize=8.2, color=c,
                fontweight="bold")

    ax.set_xlabel("Annual aggregate loss (CNY 100 mn)")
    ax.set_ylabel("Number of simulated years (log scale)")
    ax.set_title(f"(b) Annual loss distribution ({ylt.n_years:,} simulated years)")
    ax.grid(True, axis="y", ls=":", alpha=0.55)

    cap = (f"Loss-free years: {metrics.loss_free_prob * 100:.1f}%    "
           f"C-ROSS II capital (VaR99.5 - AAL): "
           f"{metrics.c_ross_capital:,.0f} (CNY 100 mn)")
    ax.text(0.98, 0.97, cap, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.4, bbox=dict(boxstyle="round,pad=0.38", fc="#FDFEFE",
                                    ec="#95A5A6", lw=0.8))

    fig.suptitle("Fig.6  Loss Distribution and Risk Metrics (Insured Basis)",
                 fontsize=13, fontweight="bold", y=1.00)
    return _save(fig, "fig06_ep_curves.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 7 — Reinsurance layer structure
# --------------------------------------------------------------------------- #


def plot_reinsurance_layers(
    layers: List[LayerResult], metrics: RiskMetrics, cfg: PlotConfig = PLOT
) -> str:
    """绘制再保险分层结构图与 ROL / Multiple 对比。

    Args:
        layers: 三层定价结果。
        metrics: 风险度量（用于标注 PML 参考线）。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    colors = [cfg.palette[0], cfg.palette[1], cfg.palette[3]]
    for i, (ly, c) in enumerate(zip(layers, colors)):
        ax.bar(0.0, ly.limit, bottom=ly.attachment, width=0.52, color=c,
               edgecolor="white", linewidth=1.4, alpha=0.92)
        mid = ly.attachment + ly.limit / 2.0
        ax.text(0.0, mid,
                f"{ly.name}\n{ly.limit:,.0f} xs {ly.attachment:,.0f}\n"
                f"ROL {ly.rate_on_line * 100:.2f}%  |  Mult {ly.multiple:.2f}x",
                ha="center", va="center", fontsize=8.6, color="white",
                fontweight="bold")
        ax.annotate(f"{ly.exhaustion:,.0f}  ({ly.exhaust_rp:.0f}-yr)",
                    xy=(0.27, ly.exhaustion), xytext=(0.40, ly.exhaustion),
                    fontsize=7.8, va="center", color=cfg.text_color)
        ax.annotate(f"{ly.attachment:,.0f}  ({ly.attach_rp:.0f}-yr)",
                    xy=(-0.27, ly.attachment), xytext=(-0.72, ly.attachment),
                    fontsize=7.8, va="center", color=cfg.text_color)

    ax.bar(0.0, layers[0].attachment, width=0.52, color="#D5DBDB",
           edgecolor="white", linewidth=1.4)
    ax.text(0.0, layers[0].attachment / 2.0, "Cedent retention",
            ha="center", va="center", fontsize=8.6, color="#566573",
            fontweight="bold")

    ax.axhline(metrics.oep_pml[100.0], color=cfg.palette[4], ls="--", lw=1.4)
    ax.text(0.62, metrics.oep_pml[100.0], " 100-yr OEP PML", fontsize=7.8,
            color=cfg.palette[4], va="bottom")

    ax.set_xlim(-0.85, 0.95)
    ax.set_xticks([])
    ax.set_ylabel("Loss (CNY 100 mn)")
    ax.set_title("(a) Excess-of-loss programme structure")
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    ax = axes[1]
    x = np.arange(len(layers))
    w = 0.34
    el_rate = [ly.el_rate * 100 for ly in layers]
    rol = [ly.rate_on_line * 100 for ly in layers]
    ax.bar(x - w / 2, el_rate, w, label="Expected loss rate (EL / limit)",
           color=cfg.palette[2], edgecolor="white")
    ax.bar(x + w / 2, rol, w, label="Rate on line (premium / limit)",
           color=cfg.palette[4], edgecolor="white")
    for i, ly in enumerate(layers):
        ax.text(i - w / 2, el_rate[i], f"{el_rate[i]:.2f}%", ha="center",
                va="bottom", fontsize=7.8)
        ax.text(i + w / 2, rol[i], f"{rol[i]:.2f}%", ha="center",
                va="bottom", fontsize=7.8)

    ax2 = ax.twinx()
    ax2.plot(x, [ly.multiple for ly in layers], marker="D", ms=8, lw=2.0,
             color=cfg.palette[0], label="Multiple (ROL / EL rate)")
    for i, ly in enumerate(layers):
        ax2.annotate(f"{ly.multiple:.2f}x", (i, ly.multiple),
                     xytext=(0, 11), textcoords="offset points",
                     ha="center", fontsize=8.4, fontweight="bold",
                     color=cfg.palette[0])
    ax2.set_ylabel("Multiple (x)", color=cfg.palette[0])
    ax2.tick_params(axis="y", colors=cfg.palette[0])
    ax2.set_ylim(0, max(ly.multiple for ly in layers) * 1.55)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ly.name}\n{ly.limit:,.0f} xs {ly.attachment:,.0f}"
                        for ly in layers], fontsize=8.4)
    ax.set_ylabel("Rate (% of limit)")
    ax.set_title("(b) Layer pricing: EL rate, ROL and multiple")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.suptitle("Fig.7  Catastrophe Excess-of-Loss Reinsurance Pricing",
                 fontsize=13, fontweight="bold", y=1.00)
    return _save(fig, "fig07_reinsurance_layers.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 8 — CAT bond spread sensitivity
# --------------------------------------------------------------------------- #


def plot_catbond_pricing(
    sens: pd.DataFrame,
    bond_index: CatBondResult,
    bond_param: CatBondResult,
    risk_free: float,
    cfg: PlotConfig = PLOT,
) -> str:
    """绘制 CAT bond 定价敏感性与两种触发结构对比。

    Args:
        sens: 起赔点敏感性表。
        bond_index: 指数触发债券定价结果。
        bond_param: 参数触发债券定价结果。
        risk_free: 无风险利率。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.4))

    ax = axes[0]
    ax.plot(sens["attach_rp"], sens["spread_lane"] * 1e4, marker="o", ms=6,
            lw=2.2, color=cfg.palette[0], label="Lane (2000) empirical model")
    ax.plot(sens["attach_rp"], sens["spread_wang"] * 1e4, marker="s", ms=6,
            lw=2.2, color=cfg.palette[4], label="Wang (2000) transform")
    ax.plot(sens["attach_rp"], sens["el"] * 1e4, marker="^", ms=5, lw=1.8,
            ls="--", color=cfg.palette[2], label="Expected loss (EL)")
    ax.fill_between(sens["attach_rp"], sens["el"] * 1e4,
                    sens["spread_lane"] * 1e4, color=cfg.palette[0], alpha=0.10)
    ax.set_xscale("log")
    ax.set_xlabel("Attachment return period (years, log scale)")
    ax.set_ylabel("Spread over risk-free (bp)")
    ax.set_title("(a) Attachment point vs spread")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(sens["el"] * 1e4, sens["spread_lane"] / np.maximum(sens["el"], 1e-9),
            marker="o", ms=6, lw=2.2, color=cfg.palette[0], label="Lane multiple")
    ax.plot(sens["el"] * 1e4, sens["spread_wang"] / np.maximum(sens["el"], 1e-9),
            marker="s", ms=6, lw=2.2, color=cfg.palette[4], label="Wang multiple")
    ax.axhline(1.0, color="#95A5A6", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Expected loss (bp, log scale)")
    ax.set_ylabel("Multiple = spread / EL")
    ax.set_title("(b) Pricing multiple vs expected loss")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = ["Industry\nloss index", "Parametric\n(landfall $P_c$)"]
    el_bp = [bond_index.expected_loss * 1e4, bond_param.expected_loss * 1e4]
    lane_bp = [bond_index.spread_lane * 1e4, bond_param.spread_lane * 1e4]
    wang_bp = [bond_index.spread_wang * 1e4, bond_param.spread_wang * 1e4]
    x = np.arange(2)
    w = 0.26
    ax.bar(x - w, el_bp, w, label="EL", color=cfg.palette[2], edgecolor="white")
    ax.bar(x, lane_bp, w, label="Spread (Lane)", color=cfg.palette[0],
           edgecolor="white")
    ax.bar(x + w, wang_bp, w, label="Spread (Wang)", color=cfg.palette[4],
           edgecolor="white")
    for xi, vals in zip(x, zip(el_bp, lane_bp, wang_bp)):
        for dx, v in zip((-w, 0.0, w), vals):
            ax.text(xi + dx, v, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("bp per annum")
    ax.set_title(f"(c) Trigger structures\n(risk-free = {risk_free * 100:.2f}%)")
    ax.grid(True, axis="y", ls=":", alpha=0.55)
    ax.legend(fontsize=8)

    fig.suptitle("Fig.8  Catastrophe Bond Pricing: Lane vs Wang Transform",
                 fontsize=13, fontweight="bold", y=1.01)
    return _save(fig, "fig08_catbond_pricing.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 9 — Basis risk
# --------------------------------------------------------------------------- #


def plot_basis_risk(
    basis_box: BasisRiskResult,
    basis_simple: BasisRiskResult,
    basis_naive: Optional[BasisRiskResult] = None,
    cfg: PlotConfig = PLOT,
) -> str:
    """绘制单一条件与多重条件参数触发的基差风险对比。

    Args:
        basis_box: 经验拟合位置箱体的多重条件触发结果。
        basis_simple: 单一条件 (仅气压) 触发的基差风险结果。
        basis_naive: 手工设定位置箱体的多重条件触发结果（可选）。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.6),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})

    rng = np.random.default_rng(7)
    n_show = min(basis_box.actual_loss.size, 6000)
    pick = rng.choice(basis_box.actual_loss.size, size=n_show, replace=False)

    panels = (
        (axes[0], basis_simple, "(a) Single-condition trigger: landfall $P_c$ only",
         cfg.palette[3]),
        (axes[1], basis_box,
         "(b) Multi-condition trigger: $P_c$ x fitted cat-in-a-box",
         cfg.palette[1]),
    )
    hi = float(max(np.quantile(basis_box.actual_loss, 0.9995),
                   np.quantile(basis_box.payout, 0.9995),
                   np.quantile(basis_simple.payout, 0.9995), 1.0))

    for ax, b, title, color in panels:
        ax.scatter(b.actual_loss[pick], b.payout[pick], s=11, alpha=0.30,
                   color=color, edgecolor="none")
        ax.plot([0, hi], [0, hi], color=cfg.palette[4], ls="--", lw=1.8,
                label="Perfect hedge (45$\\degree$)")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("Actual insured loss per event (CNY 100 mn)")
        ax.set_ylabel("Parametric payout per event (CNY 100 mn)")
        ax.set_title(title)
        ax.grid(True, ls=":", alpha=0.55)
        txt = (f"Pearson corr        = {b.correlation:.3f}\n"
               f"Spearman corr       = {b.rank_correlation:.3f}\n"
               f"Hedge effectiveness = {b.hedge_effectiveness * 100:.1f}%\n"
               f"P(under-payment)    = {b.prob_shortfall * 100:.1f}%\n"
               f"P(over-payment)     = {b.prob_windfall * 100:.1f}%")
        ax.text(0.035, 0.965, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=8.2, family="monospace",
                bbox=dict(boxstyle="round,pad=0.42", fc="#FDFEFE",
                          ec=color, lw=1.0))
        ax.legend(loc="lower right", fontsize=8.2)

    ax = axes[2]
    labels = ["Pearson\ncorrelation", "Spearman\ncorrelation",
              "Hedge\neffectiveness"]

    def _triple(b: BasisRiskResult) -> List[float]:
        """提取三项基差风险指标。

        Args:
            b: 基差风险结果。

        Returns:
            List[float]: [Pearson, Spearman, 对冲效率]。
        """
        return [b.correlation, b.rank_correlation, b.hedge_effectiveness]

    series = [("$P_c$ only", _triple(basis_simple), cfg.palette[3])]
    if basis_naive is not None:
        series.append(("$P_c$ x naive box", _triple(basis_naive), cfg.palette[6]))
    series.append(("$P_c$ x fitted box", _triple(basis_box), cfg.palette[1]))

    x = np.arange(3)
    w = 0.8 / len(series)
    for k, (lbl, vals, color) in enumerate(series):
        off = (k - (len(series) - 1) / 2.0) * w
        ax.bar(x + off, vals, w, label=lbl, color=color, edgecolor="white")
        for xi, v in zip(x, vals):
            ax.text(xi + off, v, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("Value (0-1)")
    ax.set_ylim(0, max(max(v) for _, v, _ in series) * 1.30)
    ax.set_title("(c) Trigger design drives basis risk")
    ax.grid(True, axis="y", ls=":", alpha=0.55)
    ax.legend(fontsize=7.8)

    fig.suptitle("Fig.9  Basis Risk: Single vs Multi-Condition Parametric Triggers",
                 fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, "fig09_basis_risk.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 10 — Portfolio efficient frontier
# --------------------------------------------------------------------------- #


def plot_portfolio(
    port: PortfolioResult, risk_free: float, cfg: PlotConfig = PLOT
) -> str:
    """绘制含/不含 CAT bond 的有效前沿与相关性矩阵。

    Args:
        port: 组合分析结果。
        risk_free: 无风险利率。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax = axes[0]
    fw, fo = port.frontier_with, port.frontier_without
    ax.plot(fo[:, 0] * 100, fo[:, 1] * 100, lw=2.6, color=cfg.palette[6],
            ls="--", label="Frontier: Equity + Bond")
    ax.plot(fw[:, 0] * 100, fw[:, 1] * 100, lw=2.8, color=cfg.palette[0],
            label="Frontier: Equity + Bond + CAT Bond")
    ax.fill_between(fw[:, 0] * 100, fw[:, 1] * 100,
                    np.interp(fw[:, 0], fo[:, 0], fo[:, 1],
                              left=np.nan, right=np.nan) * 100,
                    color=cfg.palette[0], alpha=0.10)

    for i, name in enumerate(port.asset_names):
        ax.scatter(port.sigma[i] * 100, port.mu[i] * 100, s=110, zorder=5,
                   color=cfg.palette[[0, 1, 4][i]], edgecolor="white",
                   linewidth=1.2)
        ax.annotate(name, (port.sigma[i] * 100, port.mu[i] * 100),
                    xytext=(8, -3), textcoords="offset points", fontsize=9,
                    fontweight="bold")

    for res, c, lbl in ((port.best_without, cfg.palette[6], "Max Sharpe (no CAT)"),
                        (port.best_with, cfg.palette[4], "Max Sharpe (with CAT)")):
        ax.scatter(res[0] * 100, res[1] * 100, marker="*", s=330, color=c,
                   edgecolor="white", linewidth=1.1, zorder=6, label=lbl)

    ax.scatter(0.0, risk_free * 100, marker="P", s=110, color="#2C3E50",
               zorder=6)
    ax.annotate(f"Risk-free {risk_free * 100:.2f}%", (0.0, risk_free * 100),
                xytext=(8, -12), textcoords="offset points", fontsize=8.5)

    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised expected return (%)")
    ax.set_title("(a) Mean-variance efficient frontier")
    ax.grid(True, ls=":", alpha=0.55)
    ax.legend(fontsize=8.4, loc="lower right")

    vo, ro, so, wo = port.best_without
    vw, rw, sw, ww = port.best_with
    txt = (f"Max-Sharpe without CAT bond\n"
           f"  weights  E/B      = {wo[0] * 100:.0f}% / {wo[1] * 100:.0f}%\n"
           f"  return / vol      = {ro * 100:.2f}% / {vo * 100:.2f}%\n"
           f"  Sharpe            = {so:.3f}\n\n"
           f"Max-Sharpe with CAT bond\n"
           f"  weights E/B/CAT   = {ww[0] * 100:.0f}% / {ww[1] * 100:.0f}%"
           f" / {ww[2] * 100:.0f}%\n"
           f"  return / vol      = {rw * 100:.2f}% / {vw * 100:.2f}%\n"
           f"  Sharpe            = {sw:.3f}\n"
           f"  improvement       = {port.sharpe_improvement:+.3f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.2, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FDFEFE",
                      ec=cfg.palette[0], lw=1.0))

    ax = axes[1]
    im = ax.imshow(port.corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(port.asset_names, rotation=20, ha="right")
    ax.set_yticklabels(port.asset_names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{port.corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(port.corr[i, j]) > 0.55 else "#1C2833")
    fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03).set_label(
        "Correlation", fontsize=9)
    ax.set_title("(b) Correlation matrix: CAT bond as a zero-beta asset")

    fig.suptitle("Fig.10  Portfolio Diversification Value of CAT Bonds",
                 fontsize=13, fontweight="bold", y=1.00)
    return _save(fig, "fig10_portfolio_frontier.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 11 — Stochastic event set overview
# --------------------------------------------------------------------------- #


def plot_event_set_overview(
    landfall_lat: np.ndarray,
    landfall_dp: np.ndarray,
    event_losses: np.ndarray,
    lekima_loss: float,
    freq_lambda: float,
    cfg: PlotConfig = PLOT,
) -> str:
    """绘制随机事件集的诊断图（登陆强度/位置分布、事件损失分布）。

    Args:
        landfall_lat: 登陆纬度 ``(N,)``。
        landfall_dp: 登陆 Delta_p ``(N,)`` (hPa)。
        event_losses: 事件直接经济损失 ``(N,)`` (亿元)。
        lekima_loss: 利奇马模拟损失 (亿元)，用于定位。
        freq_lambda: 年频率。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.0))

    ax = axes[0]
    ax.hist(landfall_dp, bins=55, color=cfg.palette[1], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvline(80.0, color=cfg.palette[4], lw=2.0, ls="--")
    ax.text(80.0, ax.get_ylim()[1] * 0.85, "  Lekima landfall\n  $\\Delta p$=80 hPa",
            fontsize=8.4, color=cfg.palette[4], fontweight="bold")
    exceed = float(np.mean(landfall_dp >= 80.0))
    ax.text(0.97, 0.60,
            f"P($\\Delta p \\geq$ 80) = {exceed * 100:.2f}%\n"
            f"landfall RP = {1.0 / max(exceed * freq_lambda, 1e-9):.0f} yr",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDFEFE",
                      ec=cfg.palette[4], lw=0.9))
    ax.set_xlabel("Landfall central pressure deficit $\\Delta p$ (hPa)")
    ax.set_ylabel("Number of synthetic events")
    ax.set_title("(a) Landfall intensity distribution")
    ax.grid(True, axis="y", ls=":", alpha=0.55)

    ax = axes[1]
    hb = ax.hexbin(landfall_lat, landfall_dp, gridsize=38, cmap=cfg.seq_cmap,
                   mincnt=1)
    ax.scatter([28.40], [80.0], marker="*", s=260, color=cfg.palette[4],
               edgecolor="white", linewidth=1.0, zorder=5, label="Lekima")
    fig.colorbar(hb, ax=ax, pad=0.02).set_label("Event count", fontsize=9)
    ax.set_xlabel("Landfall latitude (deg N)")
    ax.set_ylabel("Landfall $\\Delta p$ (hPa)")
    ax.set_title("(b) Joint landfall location-intensity density")
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[2]
    pos = event_losses[event_losses > 0.5]
    ax.hist(np.log10(pos), bins=60, color=cfg.palette[0], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvline(np.log10(max(lekima_loss, 1.0)), color=cfg.palette[4], lw=2.0,
               ls="--")
    ax.text(np.log10(max(lekima_loss, 1.0)), ax.get_ylim()[1] * 0.85,
            f"  Lekima\n  {lekima_loss:,.0f}", fontsize=8.4,
            color=cfg.palette[4], fontweight="bold")
    ax.set_xlabel("$\\log_{10}$ event direct economic loss (CNY 100 mn)")
    ax.set_ylabel("Number of synthetic events")
    ax.set_title(f"(c) Event loss severity ({event_losses.size:,} events)")
    ax.grid(True, axis="y", ls=":", alpha=0.55)

    fig.suptitle(
        f"Fig.11  Stochastic Event Set Diagnostics "
        f"($\\lambda$ = {freq_lambda:.1f} events/yr)",
        fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, "fig11_event_set.png", cfg)


# --------------------------------------------------------------------------- #
# Figure 12 — Climate change scenarios (module E)
# --------------------------------------------------------------------------- #

#: fig12 各情景配色（基准深蓝 -> 高排放紫红），与情景表顺序一一对应。
_CLIMATE_COLORS: Tuple[str, ...] = (
    "#12355B", "#3FA796", "#E9A03B", "#D1495B", "#7D53DE",
)


def plot_climate_scenarios(analysis: "ClimateAnalysis",
                           cfg: PlotConfig = PLOT) -> str:
    """绘制气候变化情景全景图（2x3 六联图，全英文标注）。

    子图内容：
        (a) 各情景 OEP 曲线族
        (b) 各重现期 PML 漂移（分组柱状）
        (c) PML100 变化的归因分解（堆叠柱状：气候/暴露/交互）
        (d) 重现期贬值：2020 年的 100 年一遇在未来变成几年一遇
        (e) CAT bond 利差漂移（Lane vs Wang，bp）
        (f) 强度信号不确定性扇形带（Knutson +1%~+10%）

    Args:
        analysis: ``climate.run_climate_scenarios`` 的返回结果。
        cfg: 绘图配置。

    Returns:
        str: 图片路径。
    """
    _init_style(cfg)
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.4))

    results = list(analysis.results)
    colors = [_CLIMATE_COLORS[i % len(_CLIMATE_COLORS)]
              for i in range(len(results))]
    labels = [r.scenario.label_en for r in results]
    non_base = [r for r in results if not r.scenario.is_baseline]
    nb_labels = [r.scenario.label_en for r in non_base]
    nb_colors = colors[1:len(non_base) + 1]
    head_rp = analysis.headline_rp

    # ---- (a) EP curve family ---- #
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        x, _, rp = ep_curve(r.occurrence_econ)
        sel = rp >= 1.05
        ax.plot(rp[sel], x[sel] / 1.0e4, lw=2.2, color=c,
                label=r.scenario.label_en,
                ls="-" if not r.scenario.is_baseline else "--")
    ax.axvline(head_rp, color="#95A5A6", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_xlabel("Return period (years, log scale)")
    ax.set_ylabel("OEP loss (CNY trillion)")
    ax.set_title("(a) Occurrence EP curves by climate scenario")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(loc="upper left", fontsize=8.2)

    # ---- (b) PML drift by return period ---- #
    ax = axes[0, 1]
    rps = list(analysis.return_periods)
    xpos = np.arange(len(rps), dtype=float)
    width = 0.8 / max(len(non_base), 1)
    for k, (r, c) in enumerate(zip(non_base, nb_colors)):
        vals = [r.pml_drift.get(rp, 0.0) * 100.0 for rp in rps]
        ax.bar(xpos + (k - (len(non_base) - 1) / 2.0) * width, vals,
               width=width * 0.92, color=c, edgecolor="white", linewidth=0.6,
               label=r.scenario.label_en)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{int(rp)}-yr" for rp in rps])
    ax.axhline(0.0, color="#5D6D7E", lw=1.0)
    ax.set_xlabel("OEP return period")
    ax.set_ylabel("PML change vs Baseline 2020 (%)")
    ax.set_title("(b) PML drift, total effect (climate + exposure)")
    ax.grid(True, axis="y", ls=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8.2)

    # ---- (c) Attribution decomposition ---- #
    ax = axes[0, 2]
    xa = np.arange(len(non_base), dtype=float)
    d_clim = np.array([r.attribution["delta_climate"] for r in non_base]) / 1.0e4
    d_exp = np.array([r.attribution["delta_exposure"] for r in non_base]) / 1.0e4
    d_int = np.array(
        [r.attribution["delta_interaction"] for r in non_base]) / 1.0e4
    ax.bar(xa, d_exp, width=0.56, color=cfg.palette[0],
           edgecolor="white", linewidth=0.6, label="Exposure growth")
    ax.bar(xa, d_clim, width=0.56, bottom=d_exp, color=cfg.palette[4],
           edgecolor="white", linewidth=0.6, label="Climate signal")
    ax.bar(xa, d_int, width=0.56, bottom=d_exp + d_clim, color=cfg.palette[3],
           edgecolor="white", linewidth=0.6, label="Interaction")
    for i, r in enumerate(non_base):
        a = r.attribution
        ax.text(xa[i], (d_exp[i] + d_clim[i] + d_int[i]) * 1.02,
                f"C {a['share_climate']:.0f}% | E {a['share_exposure']:.0f}%"
                f" | I {a['share_interaction']:.0f}%",
                ha="center", va="bottom", fontsize=7.4, fontweight="bold",
                color=cfg.text_color)
    ax.set_xticks(xa)
    ax.set_xticklabels(nb_labels, fontsize=8.2, rotation=12)
    ax.set_ylabel(f"Increase in {int(head_rp)}-yr PML (CNY trillion)")
    ax.set_title(f"(c) Attribution of the {int(head_rp)}-yr PML increase")
    ax.grid(True, axis="y", ls=":", alpha=0.55)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.set_ylim(0.0, float((d_exp + d_clim + d_int).max()) * 1.22)

    # ---- (d) Return-period depreciation ---- #
    ax = axes[1, 0]
    dep = [min(r.depreciated_rp, head_rp * 3.0) for r in non_base]
    bars = ax.bar(xa, dep, width=0.56, color=nb_colors,
                  edgecolor="white", linewidth=0.6)
    ax.axhline(head_rp, color=cfg.palette[0], ls="--", lw=1.6)
    ax.text(len(non_base) - 0.5, head_rp * 1.03,
            f"Baseline: {int(head_rp)}-yr event", ha="right", va="bottom",
            fontsize=8.4, color=cfg.palette[0], fontweight="bold")
    for b, r in zip(bars, non_base):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() * 1.03,
                f"{r.depreciated_rp:.0f} yr", ha="center", va="bottom",
                fontsize=8.6, fontweight="bold", color=cfg.text_color)
    ax.set_xticks(xa)
    ax.set_xticklabels(nb_labels, fontsize=8.2, rotation=12)
    ax.set_ylabel("New return period (years)")
    ax.set_title(f"(d) Return-period depreciation of today's "
                 f"{int(head_rp)}-yr loss")
    ax.grid(True, axis="y", ls=":", alpha=0.55)
    ax.set_ylim(0.0, head_rp * 1.28)

    # ---- (e) CAT bond spread drift ---- #
    ax = axes[1, 1]
    lane = [r.spread_drift_lane_bp for r in non_base]
    wang = [r.spread_drift_wang_bp for r in non_base]
    ax.bar(xa - 0.19, lane, width=0.36, color=cfg.palette[0],
           edgecolor="white", linewidth=0.6, label="Lane (2000) model")
    ax.bar(xa + 0.19, wang, width=0.36, color=cfg.palette[2],
           edgecolor="white", linewidth=0.6, label="Wang transform")
    for i, v in enumerate(lane):
        ax.text(xa[i] - 0.19, v * 1.03, f"+{v:.0f}", ha="center", va="bottom",
                fontsize=8.0, fontweight="bold", color=cfg.palette[0])
    ax.set_xticks(xa)
    ax.set_xticklabels(nb_labels, fontsize=8.2, rotation=12)
    ax.set_ylabel("Fair spread drift (bp)")
    ax.set_title("(e) CAT bond spread drift, constant 2020 exposure")
    ax.grid(True, axis="y", ls=":", alpha=0.55)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.set_ylim(0.0, max(lane + wang) * 1.22)

    # ---- (f) Intensity uncertainty fan ---- #
    ax = axes[1, 2]
    warm = np.array([r.scenario.warming_c for r in results], dtype=float)
    lo = np.array([r.uncertainty[0] for r in results]) / 1.0e4
    mid = np.array([r.uncertainty[1] for r in results]) / 1.0e4
    hi = np.array([r.uncertainty[2] for r in results]) / 1.0e4
    order = np.argsort(warm)
    ax.fill_between(warm[order], lo[order], hi[order], color=cfg.palette[1],
                    alpha=0.22, label="Knutson intensity range (+1% to +10%)")
    ax.plot(warm[order], mid[order], lw=2.4, color=cfg.palette[0],
            marker="o", ms=6, markeredgecolor="white",
            label="Median intensity (+5%)")
    ax.plot(warm[order], lo[order], lw=1.1, ls="--", color=cfg.palette[1])
    ax.plot(warm[order], hi[order], lw=1.1, ls="--", color=cfg.palette[1])
    for r, w, m in zip(results, warm, mid):
        ax.annotate(r.scenario.label_en, xy=(w, m),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=7.6, color=cfg.text_color)
    ax.set_xlabel("Global warming above pre-industrial (deg C)")
    ax.set_ylabel(f"{int(head_rp)}-yr OEP PML (CNY trillion)")
    ax.set_title("(f) Uncertainty fan from the intensity signal")
    ax.grid(True, ls=":", alpha=0.55)
    ax.legend(loc="upper left", fontsize=8.2)

    fig.suptitle(
        "Fig.12  Climate Change Scenario Analysis "
        "(IPCC AR6 WG1 Ch11 / Knutson et al. 2020 scaling)",
        fontsize=13.5, fontweight="bold", y=1.005)
    fig.tight_layout()
    return _save(fig, "fig12_climate_scenarios.png", cfg)


__all__ = [
    "plot_track", "plot_holland_profiles", "plot_wind_field",
    "plot_vulnerability", "plot_lekima_validation", "plot_ep_and_distribution",
    "plot_reinsurance_layers", "plot_catbond_pricing", "plot_basis_risk",
    "plot_portfolio", "plot_event_set_overview", "plot_climate_scenarios",
]
