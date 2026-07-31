"""全局配置模块：物理常数、蒙特卡洛设置、金融参数、绘图色板与路径。

本模块集中管理台风巨灾模型（Typhoon Catastrophe Model）四大模块
(Hazard / Exposure / Vulnerability / Financial) 所需的全部可调参数，
使用 ``dataclass`` 组织，便于在敏感性分析中整体替换。

参数来源与假设：
    * 物理常数：Holland (1980), Kaplan & DeMaria (1995), Willoughby & Rahn (2004)
    * 脆弱性：Emanuel (2011)
    * 金融：Lane (2000), Wang (2000, 2002), 中国偿二代二期 (C-ROSS II) 思路
    * 频率/强度统计：西北太平洋登陆华东台风的公开量级近似，非官方统计

注意：
    所有货币单位统一为 **亿元人民币 (100 million CNY)**。
    所有风速单位为 **m/s**，气压单位为 **hPa**，距离单位为 **km**。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# 路径配置
# --------------------------------------------------------------------------- #
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "outputs")

# --------------------------------------------------------------------------- #
# 物理常数
# --------------------------------------------------------------------------- #
AIR_DENSITY: float = 1.15          # 空气密度 rho (kg/m^3)，热带海洋边界层典型值
AMBIENT_PRESSURE: float = 1010.0   # 环境气压 Pn (hPa)
OMEGA_EARTH: float = 7.292115e-5   # 地球自转角速度 (rad/s)
EARTH_RADIUS_KM: float = 6371.0    # 地球平均半径 (km)
DEG2RAD: float = 3.141592653589793 / 180.0


@dataclass(frozen=True)
class HazardConfig:
    """危险性模块参数。

    Attributes:
        surface_reduction_factor: 梯度风 -> 10m 高度 1 分钟持续风的折减系数
            (Powell et al. 2003 给出 0.75~0.90，海面取 0.85)。
        gust_factor: 1 分钟持续风 -> 3 秒阵风 的阵风因子 (ASCE/WMO 建议 ~1.3)。
        holland_b_min: Holland 形状参数 B 的下界。
        holland_b_max: Holland 形状参数 B 的上界。
        rmw_min_km: 最大风速半径下界 (km)。
        rmw_max_km: 最大风速半径上界 (km)。
        asymmetry_alpha: 移动非对称修正系数，右半圆叠加 alpha * 移速。
        decay_vb_ms: Kaplan-DeMaria 登陆衰减的背景风速 Vb (m/s)。
        decay_r: Kaplan-DeMaria 登陆瞬间强度折减比 R。
        decay_alpha_per_hr: Kaplan-DeMaria 衰减率 alpha (1/hr)。
        max_radius_km: 风场计算的最大作用半径 (km)，超出记为 0。
    """

    surface_reduction_factor: float = 0.85
    gust_factor: float = 1.30
    holland_b_min: float = 0.80
    holland_b_max: float = 2.50
    rmw_min_km: float = 15.0
    rmw_max_km: float = 120.0
    asymmetry_alpha: float = 0.50
    decay_vb_ms: float = 7.0
    decay_r: float = 0.90
    decay_alpha_per_hr: float = 0.095
    max_radius_km: float = 800.0


@dataclass(frozen=True)
class StochasticConfig:
    """随机事件集 (Stochastic Event Set) 参数。

    Attributes:
        n_events: 合成台风事件总数。
        random_seed: 随机种子，保证结果可复现。
        annual_frequency_lambda: 年频率 lambda（泊松分布均值）。
            取 3.2 —— 1949-2020 年平均每年约有 3 个台风登陆或显著影响
            华东沿海 (浙江/上海/江苏/山东)，示例量级近似值。
        dp_median_hpa: 登陆时刻中心气压差 Delta_p 的对数正态中位数 (hPa)。
        dp_sigma_log: Delta_p 对数正态的对数标准差。
        dp_min_hpa: Delta_p 截断下界 (hPa)。
        dp_max_hpa: Delta_p 截断上界 (hPa)，对应中心气压约 905 hPa。
        landfall_lat_min: 登陆点纬度下界。
        landfall_lat_max: 登陆点纬度上界。
        heading_mean_deg: 移动方向均值（0=正北，正值向东，负值向西）。
        heading_std_deg: 移动方向标准差。
        translation_median_ms: 移速对数正态中位数 (m/s)。
        translation_sigma_log: 移速对数正态的对数标准差。
        n_steps_before: 登陆前的 6 小时路径点数。
        n_steps_after: 登陆后的 6 小时路径点数。
        step_hours: 路径点时间间隔 (hr)。
        n_simulation_years: 年度损失分布模拟年数。
        batch_size: 事件风场计算的分批大小（控制内存占用）。
    """

    n_events: int = 10_000
    random_seed: int = 20190810
    annual_frequency_lambda: float = 3.2
    dp_median_hpa: float = 26.0
    dp_sigma_log: float = 0.52
    dp_min_hpa: float = 8.0
    dp_max_hpa: float = 105.0
    landfall_lat_min: float = 25.5
    landfall_lat_max: float = 38.0
    heading_mean_deg: float = -18.0
    heading_std_deg: float = 24.0
    translation_median_ms: float = 6.2
    translation_sigma_log: float = 0.35
    n_steps_before: int = 5
    n_steps_after: int = 10
    step_hours: float = 6.0
    n_simulation_years: int = 50_000
    batch_size: int = 1_000


@dataclass(frozen=True)
class VulnerabilityConfig:
    """脆弱性模块参数。

    Attributes:
        v_thresh_ms: Emanuel (2011) 损失函数的起损风速阈值 (m/s)。
        v_half_initial_ms: 校准前的 V_half 初值 (m/s)，Emanuel 大西洋标定值 74.7。
        v_half_bracket: 校准搜索区间 (m/s)。
        building_v_half_factor: 各建筑类型的 V_half 相对系数（越小越脆弱）。
        lognormal_params: 各建筑类型对数正态脆弱性曲线 (mu_ms, sigma_log, mdr_max)。
        capital_output_ratio: 资本产出系数，暴露财产总额 = GDP * 该系数。
            中国资本存量/GDP 约 3.0~3.5，取 3.2。
        wind_exposed_fraction: 资本存量中可被风灾直接损毁的比例（建筑、
            设施、存货等），扣除土地与金融资产，取 0.45。
        flood_beta: 次生灾害（暴雨内涝）附加损失系数上限。
        flood_decay_km: 内涝影响随路径距离衰减的特征尺度 (km)。
        flood_dp_ref_hpa: 内涝强度归一化参考气压差 (hPa)。
        demand_surge_max: 需求激增 (demand surge) 最大加成。
        demand_surge_half: 需求激增半饱和损失规模 (亿元)。
        lekima_actual_loss: 利奇马中国大陆直接经济损失实际值 (亿元)。
        lekima_zhejiang_loss: 利奇马浙江省直接经济损失实际值 (亿元)。
        lekima_shandong_loss: 利奇马山东省直接经济损失实际值 (亿元)。
        lekima_jiangsu_loss: 利奇马江苏省直接经济损失实际值 (亿元)。
    """

    v_thresh_ms: float = 25.7
    v_half_initial_ms: float = 74.7
    v_half_bracket: Tuple[float, float] = (60.0, 420.0)
    building_v_half_factor: Dict[str, float] = field(
        default_factory=lambda: {
            "concrete": 1.28,      # 钢筋混凝土，抗风最好
            "masonry": 1.00,       # 砖混结构，基准
            "light_steel": 0.80,   # 轻钢/彩钢厂房
            "greenhouse": 0.52,    # 农业大棚，最脆弱
        }
    )
    lognormal_params: Dict[str, Tuple[float, float, float]] = field(
        default_factory=lambda: {
            # 建筑类型: (中位破坏风速 mu (m/s), 对数标准差 sigma, 最大毁损率)
            "concrete": (108.0, 0.42, 0.55),
            "masonry": (82.0, 0.44, 0.72),
            "light_steel": (64.0, 0.46, 0.85),
            "greenhouse": (38.0, 0.50, 1.00),
        }
    )
    capital_output_ratio: float = 3.2
    wind_exposed_fraction: float = 0.45
    flood_beta: float = 0.55
    flood_decay_km: float = 180.0
    flood_dp_ref_hpa: float = 60.0
    demand_surge_max: float = 0.18
    demand_surge_half: float = 400.0
    lekima_actual_loss: float = 537.2
    lekima_zhejiang_loss: float = 242.5
    lekima_shandong_loss: float = 165.0
    lekima_jiangsu_loss: float = 41.0


@dataclass(frozen=True)
class FinancialConfig:
    """金融工程模块参数。

    Attributes:
        risk_free_rate: 无风险利率，取 10 年期中国国债收益率量级 2.2%。
        var_levels: 需要计算的 VaR 置信水平。
        tvar_level: TVaR/CVaR 置信水平。
        c_ross_var_level: 偿二代二期 (C-ROSS II) 巨灾资本口径置信水平。
        pml_return_periods: 需要报告的重现期 (年)。
        insured_deductible_factor: 免赔额/限额导致的赔付折减比例。
        layer_return_periods: 三层再保险的 (起赔点 RP, 耗尽点 RP)。
        layer_sd_load: 再保险标准差保费原理的载荷系数 k。
        layer_expense_ratio: 再保险费用附加比例。
        catbond_attach_rp: CAT bond 起赔点重现期 (年)。
        catbond_exhaust_rp: CAT bond 耗尽点重现期 (年)。
        lane_gamma: Lane (2000) 经验定价模型系数 gamma。
        lane_alpha: Lane (2000) PFL 指数 alpha。
        lane_beta: Lane (2000) CEL 指数 beta。
        wang_lambda_market: Wang 变换的市场风险价格 lambda 文献参考值。
        parametric_pc_ladder: 参数触发赔付阶梯 [(中心气压上限 hPa, 赔付比例)]。
        parametric_lat_boxes: cat-in-a-box 登陆位置权重
            [(纬度下界, 纬度上界, 权重)]，反映暴露沿岸的集中程度。
        parametric_box_default: 落在所有 box 之外时的位置权重。
        equity_mu: 权益资产年化预期收益。
        equity_sigma: 权益资产年化波动率。
        bond_mu: 债券资产年化预期收益。
        bond_sigma: 债券资产年化波动率。
        corr_equity_bond: 权益-债券相关系数。
        corr_equity_cat: 权益-巨灾债券相关系数（近似零贝塔）。
        corr_bond_cat: 债券-巨灾债券相关系数。
        frontier_step: 有效前沿权重网格步长。
    """

    risk_free_rate: float = 0.022
    var_levels: Tuple[float, ...] = (0.99, 0.995)
    tvar_level: float = 0.99
    c_ross_var_level: float = 0.995
    pml_return_periods: Tuple[float, ...] = (10, 20, 50, 100, 200, 250, 500)
    insured_deductible_factor: float = 0.15
    layer_return_periods: Tuple[Tuple[float, float], ...] = (
        (5.0, 20.0),
        (20.0, 100.0),
        (100.0, 500.0),
    )
    layer_sd_load: float = 0.25
    layer_expense_ratio: float = 0.10
    catbond_attach_rp: float = 50.0
    catbond_exhaust_rp: float = 250.0
    lane_gamma: float = 0.55
    lane_alpha: float = 0.49
    lane_beta: float = 0.57
    wang_lambda_market: float = 0.45
    parametric_pc_ladder: Tuple[Tuple[float, float], ...] = (
        (920.0, 1.00),
        (935.0, 0.75),
        (950.0, 0.45),
        (965.0, 0.20),
    )
    parametric_lat_boxes: Tuple[Tuple[float, float, float], ...] = (
        (27.0, 29.5, 1.00),   # 浙南/台州-温州，暴露与历史登陆频次最高
        (29.5, 32.5, 0.95),   # 宁波-上海-苏州-南通，暴露规模最大
        (32.5, 35.0, 0.65),   # 盐城-连云港
        (35.0, 37.5, 0.70),   # 青岛-潍坊-烟台
        (25.0, 27.0, 0.50),   # 闽浙交界
    )
    parametric_box_default: float = 0.20
    equity_mu: float = 0.080
    equity_sigma: float = 0.180
    bond_mu: float = 0.032
    bond_sigma: float = 0.050
    corr_equity_bond: float = -0.15
    corr_equity_cat: float = 0.02
    corr_bond_cat: float = 0.00
    frontier_step: float = 0.02


@dataclass(frozen=True)
class PlotConfig:
    """绘图配置。

    Attributes:
        dpi: 输出分辨率。
        palette: 统一专业色板（深蓝-青-橙-红渐进）。
        seq_cmap: 连续型色图名称。
        bg: 画布背景色。
        grid_color: 网格线颜色。
        text_color: 主文字颜色。
    """

    dpi: int = 300
    palette: Tuple[str, ...] = (
        "#12355B",  # deep navy
        "#1B7B8C",  # teal
        "#3FA796",  # green teal
        "#E9A03B",  # amber
        "#D1495B",  # crimson
        "#7D53DE",  # violet
        "#6C757D",  # slate grey
        "#0F8B8D",  # cyan
    )
    seq_cmap: str = "YlGnBu"
    bg: str = "#FFFFFF"
    grid_color: str = "#D6DBDF"
    text_color: str = "#1C2833"


# --------------------------------------------------------------------------- #
# 单例配置对象
# --------------------------------------------------------------------------- #
HAZARD = HazardConfig()
STOCHASTIC = StochasticConfig()
VULNERABILITY = VulnerabilityConfig()
FINANCIAL = FinancialConfig()
PLOT = PlotConfig()

# --------------------------------------------------------------------------- #
# 简化的中国东部海岸线（land 在西侧，sea 在东侧）
# 仅用于陆地判别与底图绘制，非精确岸线，分辨率约 0.5 度
# --------------------------------------------------------------------------- #
COASTLINE: List[Tuple[float, float]] = [
    (24.0, 118.4), (25.0, 119.6), (26.0, 120.2), (27.0, 120.9),
    (27.5, 121.1), (28.0, 121.5), (28.5, 121.8), (29.0, 122.0),
    (29.5, 122.2), (30.0, 122.2), (30.5, 121.9), (31.0, 121.9),
    (31.5, 121.9), (32.0, 121.7), (33.0, 121.0), (34.0, 120.5),
    (35.0, 119.9), (35.6, 119.8), (36.1, 120.6), (36.6, 121.0),
    (37.0, 121.8), (37.4, 122.5), (37.8, 121.0), (38.2, 119.5),
    (39.0, 118.5), (40.0, 118.0),
]

__all__ = [
    "PROJECT_ROOT", "OUTPUT_DIR", "AIR_DENSITY", "AMBIENT_PRESSURE",
    "OMEGA_EARTH", "EARTH_RADIUS_KM", "DEG2RAD", "HazardConfig",
    "StochasticConfig", "VulnerabilityConfig", "FinancialConfig",
    "PlotConfig", "HAZARD", "STOCHASTIC", "VULNERABILITY", "FINANCIAL",
    "PLOT", "COASTLINE",
]
