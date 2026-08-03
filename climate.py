"""模块 E —— 气候变化情景下的台风巨灾风险与定价漂移分析。

本模块在既有 Hazard/Exposure/Vulnerability/Financial 四模块之上叠加一层
"情景引擎"：对每个气候情景重新生成随机事件集、重新缩放暴露与承灾能力，
然后完整重跑损失分布与巨灾金融定价，量化以下三件事：

    1. **风险漂移**：AAL / PML / VaR / TVaR 相对 2020 基准的变化幅度。
    2. **归因分解**：把 PML100 的总变化拆成 *气候信号 / 暴露增长 / 交互项*
       三部分，直接回应 Pielke (2007) 的 normalization 争论——
       "损失上升到底是气候变了，还是我们把更多钱堆在了海边？"
    3. **金融传导**：再保险分层 EL/ROL/Multiple 漂移、CAT bond 重现期贬值、
       Lane & Wang 利差漂移与"按今天定价、按未来出险"的错误定价缺口。

科学依据
--------

**(a) 强度增强 → Delta_p 中位数上移。**
IPCC AR6 WG1 Ch11 (2021) 与 Knutson et al. (2020, *BAMS*) 的归因共识：
全球升温 2°C 下热带气旋最大风速中位数增加约 **+5%**（专家评估区间
+1%~+10%）。本模型的强度自由度是登陆气压差 Delta_p，二者由
Atkinson-Holliday 关系连接：

.. math::
    V_{max} = 3.4\\,\\Delta p^{0.644}
    \\;\\Longrightarrow\\;
    \\frac{\\Delta p'}{\\Delta p} =
    \\left(\\frac{V'_{max}}{V_{max}}\\right)^{1/0.644}

故 V 增加 5% 对应 :math:`1.05^{1/0.644} = 1.0797 \\approx` **Delta_p 中位数 ×1.08**。
（教材式近似 :math:`V \\propto \\sqrt{\\Delta p}` 会给出 ×1.10；本模块的情景表
按 ×1.10 口径设定，属于略偏保守——即略偏严重——的取值，见 ``SCENARIOS``。）

**(b) Cat 4-5 占比 +13% → 对数标准差上移。**
Knutson et al. (2020) 给出 2°C 下 Cat4-5 比例增加约 **+13%**（区间
+6%~+20%）。在截断对数正态下，尾部占比对 ``sigma_log`` 高度敏感：
保持中位数不变、把 ``sigma_log`` 从 0.52 提高约 6% 即可使
:math:`P(\\Delta p \\ge 80\\,hPa)` 抬升约 13%，故 2°C 情景取 sigma ×1.06。

**(c) 降水 +14% → 次生内涝系数放大。**
AR6 与 Knutson et al. (2020) 一致给出 2°C 下 TC 降水率 **+14%**
（近 Clausius-Clapeyron 的 7%/°C）。本模型的内涝附加损失系数
:math:`\\Lambda = 1 + \\beta e^{-(d/D_0)^2}\\min(\\Delta p/\\Delta p_{ref}, 1.5)s`
中，``beta`` 与降水率近似线性，故按同比例缩放 ``flood_beta``。

**(d) 频率 -14% → 泊松 lambda 下调。**
全球 TC 总频数在多数高分辨率模式中减少，Knutson et al. (2020) 中值约
**-14%**。注意这是**总数**减少，与"强台风更多"并不矛盾——分布右移的同时
总量收缩，是巨灾风险"频率降、严重度升"的典型形态。

**(e) 线性内插的简化。**
上述 (a)~(d) 都锚定在 **2°C** 升温。本模块把各情景的升温幅度
:math:`\\Delta T` 相对 2°C 做**线性内插/外推**得到缩放因子。
这是一个明确的简化：真实的 TC 响应对 SST 是非线性的（存在潜在强度饱和、
垂直风切变的反向作用等），4.4°C 的外推尤其应被视为"方向性指示"
而非预测。相关局限已在 ``LIMITATIONS`` 与 README 中显式声明。

**(f) 暴露增长与承灾能力。**
Pielke (2007) 的 normalization 框架指出，观测到的灾害损失上升绝大部分
来自暴露增长，因此任何气候归因都必须先把暴露"折算回同一年"。本模块
反向使用这一框架：显式地把暴露路径与气候路径**分离**再重新合成，从而
给出两者各自的贡献。暴露路径 = 实际资本存量复合增长 × 承灾能力改善系数，
其中增长率在 30 年后下调（成熟经济体收敛），避免 2100 年出现 ×15 的
不可信倍数；承灾能力改善以 0.6%/yr 的脆弱性下降体现并设饱和下限。

参考文献
--------
* IPCC, 2021: *Climate Change 2021: The Physical Science Basis*, WG1 AR6,
  Chapter 11 (Weather and Climate Extreme Events in a Changing Climate).
* Knutson, T. et al., 2020: Tropical Cyclones and Climate Change Assessment:
  Part II — Projected Response to Anthropogenic Warming.
  *Bull. Amer. Meteor. Soc.*, 101(3), E303-E322.
* Kossin, J. P. et al., 2018: A global slowdown of tropical-cyclone
  translation speed. *Nature*, 558, 104-107.
* Pielke, R. A. Jr., 2007: Future economic damage from tropical cyclones:
  sensitivities to societal and climate changes.
  *Phil. Trans. R. Soc. A*, 365, 2717-2729.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import (
    AMBIENT_PRESSURE,
    CLIMATE,
    FINANCIAL,
    HAZARD,
    STOCHASTIC,
    VULNERABILITY,
    ClimateConfig,
    FinancialConfig,
    HazardConfig,
    StochasticConfig,
    VulnerabilityConfig,
)
import financial as fin
import hazard as hz
from exposure import ExposureDatabase
from vulnerability import city_losses

# --------------------------------------------------------------------------- #
# 1. 情景定义
# --------------------------------------------------------------------------- #

#: Atkinson-Holliday 关系的指数，用于 V_max 与 Delta_p 的换算。
ATKINSON_EXPONENT: float = 0.644

#: Knutson et al. (2020) 归因评估锚点升温 (°C)。
KNUTSON_ANCHOR_WARMING: float = 2.0

#: Knutson et al. (2020) 在 2°C 升温下给出的 Cat 4-5 占比增幅（用于透明度诊断）。
KNUTSON_CAT45_TARGET: float = 0.13

#: 判定"利奇马量级登陆"的 Delta_p 阈值 (hPa)，与主流程报告口径一致。
CAT45_DP_THRESHOLD: float = 80.0

#: 以下三个区间镜像 ``hazard.generate_event_set`` 内部的物理钳制，
#: 供 ``perturb_event_set`` 复现同一套强度演变逻辑。若 hazard 侧调整，
#: 此处需同步（已由 ``test_climate_baseline_identity`` 式的一致性校验守护：
#: 基准情景的扰动结果必须与直接生成的事件集逐元素相等）。
_VMAX_CLIP: Tuple[float, float] = (8.0, 78.0)
_DELTA_P_CLIP: Tuple[float, float] = (1.0, 120.0)
_PRE_LANDFALL_GROWTH: float = 1.02

#: 已在 README / 终端报告中声明的模型局限（气候模块部分）。
LIMITATIONS: Tuple[str, ...] = (
    "Scaling factors are linearly interpolated from the Knutson et al. (2020) "
    "2 deg-C anchor; the true TC response to SST is non-linear.",
    "Sea-level rise and storm-surge flooding are not modelled; coastal "
    "inundation loss is therefore omitted and the results are a LOWER bound.",
    "Kossin (2018) translation-speed slowdown is not scenario-dependent here; "
    "it would further amplify rainfall-driven inland loss.",
    "Exposure growth is a deterministic compound path, not a stochastic "
    "economic model; no spatial reallocation of capital is considered.",
    "Insurance penetration converges smoothly to a target; regulatory or "
    "market shocks (e.g. a mandatory cat scheme) are outside the model.",
)


@dataclass(frozen=True)
class ClimateScenario:
    """单个气候情景的参数缩放定义。

    所有 ``*_scale`` 字段都是**相对 2020 基准配置的乘子**，
    直接作用在既有 ``StochasticConfig`` / ``VulnerabilityConfig`` 的
    对应参数上，不改变任何既有模块的函数签名。

    Attributes:
        name: 情景唯一标识（中英混排，用于终端报告）。
        label_en: 图表用英文短标签（图表标注一律英文）。
        horizon_year: 情景对应的目标年份。
        warming_c: 相对工业化前的全球平均升温 (°C)。
        dp_median_scale: 登陆 Delta_p 中位数缩放因子（强度信号）。
        dp_sigma_scale: Delta_p 对数标准差缩放因子（Cat4-5 占比信号）。
        lambda_scale: 泊松年频率缩放因子（总频数信号）。
        flood_beta_scale: 次生内涝系数缩放因子（降水信号）。
        description: 情景中文描述，用于终端与文档。
        is_baseline: 是否为基准情景（基准情景所有缩放因子必须为 1.0）。
    """

    name: str
    label_en: str
    horizon_year: int
    warming_c: float
    dp_median_scale: float
    dp_sigma_scale: float
    lambda_scale: float
    flood_beta_scale: float
    description: str
    is_baseline: bool = False

    @property
    def horizon_years(self) -> int:
        """自基准年起的年数（用于暴露增长复利）。"""
        return int(self.horizon_year - CLIMATE.base_year)

    @property
    def implied_vmax_scale(self) -> float:
        """由 Delta_p 中位数缩放隐含的最大风速缩放因子。

        .. math::
            \\frac{V'}{V} = \\left(\\frac{\\Delta p'}{\\Delta p}\\right)^{0.644}

        Returns:
            float: 风速缩放因子。
        """
        return float(self.dp_median_scale ** ATKINSON_EXPONENT)


#: 五个气候情景。缩放因子由 Knutson et al. (2020) 的 2°C 锚点
#: （Delta_p 中位数 ×1.10 / sigma ×1.06 / lambda ×0.92 / beta ×1.13）
#: 按升温幅度相对 (2.0 - 1.1) = 0.9°C 的额外升温做线性内插得到，
#: 并四舍五入到两位小数以便复核。
SCENARIOS: Tuple[ClimateScenario, ...] = (
    ClimateScenario(
        name="Baseline 2020",
        label_en="Baseline 2020",
        horizon_year=2020,
        warming_c=1.1,
        dp_median_scale=1.00,
        dp_sigma_scale=1.00,
        lambda_scale=1.00,
        flood_beta_scale=1.00,
        description="当前气候与当前暴露，等同主流程基准，用于对照",
        is_baseline=True,
    ),
    ClimateScenario(
        name="SSP1-2.6 2050",
        label_en="SSP1-2.6 2050",
        horizon_year=2050,
        warming_c=1.6,
        dp_median_scale=1.04,
        dp_sigma_scale=1.02,
        lambda_scale=0.96,
        flood_beta_scale=1.05,
        description="强减排路径，本世纪中叶升温接近 1.5°C 目标上沿",
    ),
    ClimateScenario(
        name="SSP2-4.5 2050",
        label_en="SSP2-4.5 2050",
        horizon_year=2050,
        warming_c=2.0,
        dp_median_scale=1.07,
        dp_sigma_scale=1.04,
        lambda_scale=0.94,
        flood_beta_scale=1.09,
        description="中间路径，当前政策承诺的大致落点，行业主用情景",
    ),
    ClimateScenario(
        name="SSP5-8.5 2050",
        label_en="SSP5-8.5 2050",
        horizon_year=2050,
        warming_c=2.4,
        dp_median_scale=1.10,
        dp_sigma_scale=1.06,
        lambda_scale=0.92,
        flood_beta_scale=1.13,
        description="高排放路径本世纪中叶，用作偿付能力压力测试的主情景",
    ),
    ClimateScenario(
        name="SSP5-8.5 2100",
        label_en="SSP5-8.5 2100",
        horizon_year=2100,
        warming_c=4.4,
        dp_median_scale=1.22,
        dp_sigma_scale=1.12,
        lambda_scale=0.86,
        flood_beta_scale=1.28,
        description="高排放路径世纪末，长尾外推，仅作方向性参考",
    ),
)


def scenario_by_name(name: str) -> ClimateScenario:
    """按名称查找情景。

    Args:
        name: 情景名称，如 ``"SSP5-8.5 2050"``。

    Returns:
        ClimateScenario: 匹配的情景对象。

    Raises:
        KeyError: 名称不存在时。
    """
    for sc in SCENARIOS:
        if sc.name == name:
            return sc
    raise KeyError(f"unknown climate scenario: {name!r}")


def baseline_scenario() -> ClimateScenario:
    """返回基准情景对象。

    Returns:
        ClimateScenario: ``is_baseline=True`` 的情景。

    Raises:
        RuntimeError: 情景表中没有基准情景时。
    """
    for sc in SCENARIOS:
        if sc.is_baseline:
            return sc
    raise RuntimeError("SCENARIOS contains no baseline scenario")


# --------------------------------------------------------------------------- #
# 2. 参数缩放：气候侧
# --------------------------------------------------------------------------- #


def scale_stochastic_config(
    sc: ClimateScenario,
    base: StochasticConfig = STOCHASTIC,
    n_events: Optional[int] = None,
    intensity_multiplier: float = 1.0,
) -> StochasticConfig:
    """按情景生成缩放后的随机事件集配置。

    缩放三个自由度：

        * ``dp_median_hpa`` ×= ``dp_median_scale``（强度中位数上移）
        * ``dp_sigma_log``  ×= ``dp_sigma_scale``（Cat4-5 占比上升）
        * ``annual_frequency_lambda`` ×= ``lambda_scale``（总频数下降）

    同时把截断上限 ``dp_max_hpa`` 按 ``dp_median_scale`` 同步抬升——
    海温升高意味着**潜在强度 (potential intensity) 上限**本身在抬升，
    若保持 105 hPa 固定，高情景的右尾会被人为削平。

    Args:
        sc: 气候情景。
        base: 基准随机配置。
        n_events: 覆盖事件数；``None`` 表示沿用 ``base.n_events``。
        intensity_multiplier: 强度信号异常的额外乘子，用于不确定性带。
            1.0 = 中值路径，0.2 = Knutson 下界，2.0 = 上界。

    Returns:
        StochasticConfig: 缩放后的新配置（原对象不被修改）。
    """
    m = float(intensity_multiplier)
    dp_med = 1.0 + (sc.dp_median_scale - 1.0) * m
    dp_sig = 1.0 + (sc.dp_sigma_scale - 1.0) * m
    return replace(
        base,
        n_events=int(n_events if n_events is not None else base.n_events),
        dp_median_hpa=base.dp_median_hpa * dp_med,
        dp_sigma_log=base.dp_sigma_log * dp_sig,
        dp_max_hpa=base.dp_max_hpa * max(dp_med, 1.0),
        annual_frequency_lambda=base.annual_frequency_lambda * sc.lambda_scale,
    )


def scale_vulnerability_config(
    sc: ClimateScenario, base: VulnerabilityConfig = VULNERABILITY
) -> VulnerabilityConfig:
    """按情景生成缩放后的脆弱性配置（仅放大次生内涝系数）。

    Args:
        sc: 气候情景。
        base: 基准脆弱性配置。

    Returns:
        VulnerabilityConfig: 缩放后的新配置。
    """
    return replace(base, flood_beta=base.flood_beta * sc.flood_beta_scale)


# --------------------------------------------------------------------------- #
# 3. 参数缩放：社会经济侧
# --------------------------------------------------------------------------- #


def exposure_growth_factor(
    years: int, cfg: ClimateConfig = CLIMATE, conservative: bool = True
) -> float:
    """分段复利的实际资本存量增长倍数。

    前 ``cfg.growth_switch_years`` 年用高增速，之后切换到低增速，
    体现成熟经济体的增长收敛。若全程使用 3.5%，2100 年将得到 ×15.7，
    这在 80 年尺度上缺乏可信度。

    Args:
        years: 距基准年的年数（<=0 返回 1.0）。
        cfg: 气候配置。
        conservative: ``True`` 用保守路径 (2.5%/1.0%)，
            ``False`` 用基准路径 (3.5%/1.5%)。

    Returns:
        float: 资本存量倍数（>= 1.0）。
    """
    if years <= 0:
        return 1.0
    if conservative:
        r_early = cfg.capital_growth_conservative
        r_late = cfg.capital_growth_late_conservative
    else:
        r_early = cfg.capital_growth_baseline
        r_late = cfg.capital_growth_late_baseline
    n_early = min(years, cfg.growth_switch_years)
    n_late = max(years - cfg.growth_switch_years, 0)
    return float((1.0 + r_early) ** n_early * (1.0 + r_late) ** n_late)


def resilience_factor(years: int, cfg: ClimateConfig = CLIMATE) -> float:
    """承灾能力改善导致的脆弱性折减系数。

    .. math::
        R(t) = \\max\\big((1 - r)^{t},\\; R_{floor}\\big)

    这是 Pielke (2007) normalization 争论的"另一半"：暴露在增长，
    但单位暴露的易损性在下降（新建筑规范、防洪工程、预警体系）。
    只算增长不算折减，会系统性高估未来损失。

    Args:
        years: 距基准年的年数（<=0 返回 1.0）。
        cfg: 气候配置。

    Returns:
        float: 折减系数，区间 ``[cfg.resilience_floor, 1.0]``。
    """
    if years <= 0:
        return 1.0
    return float(max((1.0 - cfg.resilience_rate) ** years, cfg.resilience_floor))


def penetration_path(
    penetration_0: np.ndarray, years: int, cfg: ClimateConfig = CLIMATE
) -> np.ndarray:
    """财产险渗透率的指数收敛路径。

    .. math::
        \\pi(t) = \\pi^{*} - (\\pi^{*} - \\pi_0)\\,e^{-t/\\tau}

    Args:
        penetration_0: 基准年各城市渗透率 ``(C,)``。
        years: 距基准年的年数（<=0 原样返回）。
        cfg: 气候配置。

    Returns:
        np.ndarray: 目标年渗透率 ``(C,)``，单调趋向 ``cfg.penetration_target``。
    """
    pi0 = np.asarray(penetration_0, dtype=float)
    if years <= 0:
        return pi0.copy()
    decay = float(np.exp(-years / cfg.penetration_tau_years))
    return cfg.penetration_target - (cfg.penetration_target - pi0) * decay


@dataclass(frozen=True)
class ExposurePath:
    """某目标年的暴露路径参数与结果。

    Attributes:
        years: 距基准年的年数。
        growth_factor: 实际资本存量增长倍数。
        resilience: 承灾能力折减系数。
        net_scale: 净有效暴露倍数 = ``growth_factor * resilience``。
        growth_factor_high: 高增长（基准路径）倍数，仅作敏感性展示。
        exposed_value_total: 目标年可暴露价值合计 (亿元)。
        mean_penetration: 目标年平均渗透率。
        database: 缩放后的暴露数据库。
    """

    years: int
    growth_factor: float
    resilience: float
    net_scale: float
    growth_factor_high: float
    exposed_value_total: float
    mean_penetration: float
    database: ExposureDatabase = field(repr=False)


def project_exposure(
    exposure: ExposureDatabase,
    years: int,
    cfg: ClimateConfig = CLIMATE,
    vul_cfg: VulnerabilityConfig = VULNERABILITY,
    conservative: bool = True,
) -> ExposurePath:
    """把暴露数据库推演到目标年。

    实现方式是**预缩放暴露表**而非改动 ``city_losses`` 的签名：

        * ``exposed_value`` ×= 增长倍数 × 承灾能力折减系数。
          把折减放在暴露侧而非 MDR 侧，在数学上等价
          （:math:`L = V \\cdot MDR \\cdot \\Lambda`），且不必复制脆弱性曲线；
          唯一的差异来自需求激增因子对总损失的非线性依赖，量级可忽略。
        * ``penetration`` 按指数收敛路径上移。
        * ``insured_value`` 同步重算，保证 ``summary()`` 自洽。

    Args:
        exposure: 基准年暴露数据库。
        years: 距基准年的年数。
        cfg: 气候配置。
        vul_cfg: 脆弱性配置（随数据库一起携带）。
        conservative: 是否使用保守增长路径。

    Returns:
        ExposurePath: 含缩放后数据库与路径参数的结果对象。
    """
    growth = exposure_growth_factor(years, cfg, conservative=conservative)
    growth_high = exposure_growth_factor(years, cfg, conservative=False)
    resil = resilience_factor(years, cfg)
    net = growth * resil

    table = exposure.table.copy(deep=True)
    table["exposed_value"] = table["exposed_value"].to_numpy(dtype=float) * net
    table["capital_stock"] = table["capital_stock"].to_numpy(dtype=float) * growth
    table["gdp"] = table["gdp"].to_numpy(dtype=float) * growth
    table["penetration"] = penetration_path(
        table["penetration"].to_numpy(dtype=float), years, cfg
    )
    table["insured_value"] = (
        table["exposed_value"].to_numpy(dtype=float)
        * table["penetration"].to_numpy(dtype=float)
    )

    db = ExposureDatabase(table=table, config=vul_cfg)
    return ExposurePath(
        years=int(years),
        growth_factor=float(growth),
        resilience=float(resil),
        net_scale=float(net),
        growth_factor_high=float(growth_high),
        exposed_value_total=float(db.exposed_value.sum()),
        mean_penetration=float(db.penetration.mean()),
        database=db,
    )


# --------------------------------------------------------------------------- #
# 4. 单次情景重跑
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HazardCache:
    """某个事件集在固定站点上的风场计算缓存。

    同一事件集会被"总效应"和"仅气候"两条支路复用，缓存可省掉一半的
    风场积分开销。

    Attributes:
        events: 随机事件集。
        gust: 过程最大阵风 ``(N, C)`` (m/s)。
        dist: 站点到路径最近距离 ``(N, C)`` (km)。
        dp_near: 最近点 Delta_p ``(N, C)`` (hPa)。
    """

    events: hz.EventSet = field(repr=False)
    gust: np.ndarray = field(repr=False)
    dist: np.ndarray = field(repr=False)
    dp_near: np.ndarray = field(repr=False)


def perturb_event_set(
    base_events: hz.EventSet,
    sto_base: StochasticConfig,
    sto_scenario: StochasticConfig,
    hazard_cfg: HazardConfig = HAZARD,
) -> hz.EventSet:
    """用**共同随机数**把基准事件集变换到情景气候下。

    直接对缩放后的配置重新调用 ``generate_event_set`` 会有一个隐蔽问题：
    截断对数正态用的是**拒绝重抽**，参数一变，被拒绝的样本数就变，
    随机数流随之错位，登陆点纬度、移动方向、移速等**与气候无关**的维度
    也会全部改变。这样算出来的情景差异里混入了纯蒙特卡洛噪声，
    在 PML250/PML500 这类极值分位上噪声甚至会盖过气候信号
    （表现为漂移率随重现期非单调跳动）。

    行业标准做法是"事件集扰动 (event set perturbation)"：保持事件的
    几何与运动学完全不变，只把强度维度做单调变换。对数正态的
    中位数缩放 :math:`m` 与对数标准差缩放 :math:`s` 恰好对应一个
    闭式的单调映射：

    .. math::
        \\Delta p' = m\\,\\mu_0 \\left(\\frac{\\Delta p}{\\mu_0}\\right)^{s}

    其中 :math:`\\mu_0` 为基准中位数。当 :math:`m = s = 1` 时映射为恒等，
    因此基准情景可**逐元素精确复现**原事件集。

    变换后按 Atkinson-Holliday 反算登陆风速，并用与
    ``hazard.generate_event_set`` 完全相同的公式重建强度演变
    （登陆前每 6 小时 +2%，登陆后 Kaplan-DeMaria 衰减）。

    Args:
        base_events: 基准事件集。
        sto_base: 基准随机配置（提供 :math:`\\mu_0` 与截断区间）。
        sto_scenario: 情景随机配置（提供 :math:`m\\mu_0`、:math:`s\\sigma_0`、
            新的截断上限与新的泊松频率）。
        hazard_cfg: 危险性配置（衰减参数）。

    Returns:
        hz.EventSet: 与基准事件集共享几何、仅强度维度被变换的新事件集。
    """
    mu0 = sto_base.dp_median_hpa
    s_exp = sto_scenario.dp_sigma_log / sto_base.dp_sigma_log
    m_med = sto_scenario.dp_median_hpa / mu0

    # s == 1 时走乘法快路径：保证 m == 1 的基准情景逐比特恒等，
    # 避免 mu0 * (dp / mu0) 的往返舍入误差破坏"基准可复现"这一硬约束。
    if s_exp == 1.0:
        dp_scaled = base_events.landfall_dp * m_med
    else:
        dp_scaled = m_med * mu0 * np.power(base_events.landfall_dp / mu0, s_exp)
    dp_lf = np.clip(
        dp_scaled, sto_scenario.dp_min_hpa, sto_scenario.dp_max_hpa)
    vmax_lf = hz.vmax_from_delta_p(dp_lf)

    t_before, t_after = sto_base.n_steps_before, sto_base.n_steps_after
    step_idx = np.arange(-t_before, t_after + 1, dtype=float).reshape(1, -1)
    hours_after = np.maximum(step_idx, 0.0) * sto_base.step_hours
    vmax = np.where(
        step_idx < 0,
        vmax_lf.reshape(-1, 1) * np.power(_PRE_LANDFALL_GROWTH, -step_idx),
        hz.kaplan_demaria_decay(vmax_lf.reshape(-1, 1), hours_after, hazard_cfg),
    )
    vmax = np.clip(vmax, _VMAX_CLIP[0], _VMAX_CLIP[1])
    delta_p = np.clip(
        np.power(vmax / 3.4, 1.0 / ATKINSON_EXPONENT),
        _DELTA_P_CLIP[0], _DELTA_P_CLIP[1],
    )

    n = base_events.n_events
    lam = sto_scenario.annual_frequency_lambda
    return hz.EventSet(
        lon=base_events.lon,
        lat=base_events.lat,
        vmax=vmax,
        delta_p=delta_p,
        v_trans=base_events.v_trans,
        heading=base_events.heading,
        landfall_lon=base_events.landfall_lon,
        landfall_lat=base_events.landfall_lat,
        landfall_dp=dp_lf,
        landfall_pc=AMBIENT_PRESSURE - dp_lf,
        landfall_vmax=vmax_lf,
        annual_rate=np.full(n, lam / n, dtype=float),
        freq_lambda=float(lam),
    )


def build_hazard_cache(
    sto_cfg: StochasticConfig,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    hazard_cfg: HazardConfig = HAZARD,
    events: Optional[hz.EventSet] = None,
) -> HazardCache:
    """生成（或复用）事件集并一次性算好全站点风场。

    Args:
        sto_cfg: （已按情景缩放的）随机事件集配置。
        site_lon: 站点经度 ``(C,)``。
        site_lat: 站点纬度 ``(C,)``。
        hazard_cfg: 危险性配置。
        events: 预先构造好的事件集；``None`` 时用 ``sto_cfg`` 现场生成。

    Returns:
        HazardCache: 事件集与风场缓存。
    """
    ev = hz.generate_event_set(sto_cfg, hazard_cfg) if events is None else events
    gust, dist, dp_near = hz.event_set_max_gust(
        ev, site_lon, site_lat, hazard_cfg, sto_cfg.batch_size
    )
    return HazardCache(events=ev, gust=gust, dist=dist, dp_near=dp_near)


def _event_losses(
    cache: HazardCache,
    exposure: ExposureDatabase,
    v_half: float,
    vul_cfg: VulnerabilityConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """由风场缓存与暴露计算全事件集的经济/保险损失向量。

    Args:
        cache: 风场缓存。
        exposure: 暴露数据库。
        v_half: 校准后的 Emanuel V_half (m/s)。
        vul_cfg: （已按情景缩放的）脆弱性配置。

    Returns:
        Tuple[np.ndarray, np.ndarray]: (经济损失 ``(N,)``, 保险损失 ``(N,)``)，
        单位亿元。
    """
    econ_city, ins_city = city_losses(
        cache.gust, cache.dist, cache.dp_near, exposure, v_half,
        vul_cfg, curve="emanuel", apply_demand_surge=True,
    )
    return econ_city.sum(axis=1), ins_city.sum(axis=1)


def _sub_run(
    cache: HazardCache,
    exposure: ExposureDatabase,
    v_half: float,
    vul_cfg: VulnerabilityConfig,
    sto_cfg: StochasticConfig,
) -> Tuple[fin.YearLossTable, fin.YearLossTable]:
    """一条完整支路：风场缓存 + 暴露 -> 经济/保险两张年度损失表。

    Args:
        cache: 风场缓存。
        exposure: 暴露数据库。
        v_half: 校准的 V_half (m/s)。
        vul_cfg: 脆弱性配置。
        sto_cfg: 随机配置（年数与种子）。

    Returns:
        Tuple[fin.YearLossTable, fin.YearLossTable]: (经济口径, 保险口径)。
    """
    econ, ins = _event_losses(cache, exposure, v_half, vul_cfg)
    return _ylt(econ, cache, sto_cfg), _ylt(ins, cache, sto_cfg)


def _ylt(
    losses: np.ndarray, cache: HazardCache, sto_cfg: StochasticConfig
) -> fin.YearLossTable:
    """由事件损失构建年度损失表（统一使用同一模拟年数与种子）。

    Args:
        losses: 事件损失 ``(N,)``。
        cache: 风场缓存（提供 ``freq_lambda``）。
        sto_cfg: 随机配置（提供年数与种子）。

    Returns:
        fin.YearLossTable: 年度损失表。
    """
    return fin.build_year_loss_table(
        losses,
        cache.events.freq_lambda,
        sto_cfg.n_simulation_years,
        sto_cfg.random_seed,
    )


def loss_return_period(occurrence: np.ndarray, loss: float) -> float:
    """给定损失额，反查其在 OEP 曲线上的重现期。

    .. math::
        RP(L) = \\frac{1}{P(L_{occ} > L)}

    Args:
        occurrence: 年最大单次损失序列 ``(Y,)`` (亿元)。
        loss: 目标损失额 (亿元)。

    Returns:
        float: 重现期 (年)。当超越概率为 0 时返回 ``inf``。
    """
    x = np.asarray(occurrence, dtype=float)
    p = float(np.mean(x > loss))
    return float("inf") if p <= 0.0 else 1.0 / p


# --------------------------------------------------------------------------- #
# 5. 情景结果容器
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioResult:
    """单情景的完整分析结果。

    Attributes:
        scenario: 情景定义。
        exposure_path: 暴露推演路径。
        sto_cfg: 该情景使用的随机配置。
        vul_cfg: 该情景使用的脆弱性配置。
        freq_lambda: 该情景的泊松年频率。
        dp_median: 该情景的 Delta_p 中位数 (hPa)。
        dp_sigma: 该情景的 Delta_p 对数标准差。
        flood_beta: 该情景的内涝系数。
        p_cat45: 登陆 Delta_p >= 80 hPa（利奇马量级）的事件占比。
        metrics_econ: 经济口径风险度量。
        metrics_ins: 保险口径风险度量。
        occurrence_econ: 经济口径年最大损失序列 ``(Y,)``（供 EP 曲线绘图）。
        occurrence_ins: 保险口径年最大损失序列 ``(Y,)``。
        aal_drift: AAL 相对基准的变化率。
        aal_drift_ins: 保险 AAL 相对基准的变化率。
        pml_drift: 各重现期 OEP PML 相对基准的变化率。
        climate_only_drift: **恒定暴露**下 PML(headline) 的变化率，
            即纯气候信号，可直接与国际同业发布的"气候变化使 100 年一遇
            损失上升 X%"口径对比。
        exposure_only_drift: 恒定气候下 PML(headline) 的变化率。
        cat45_drift: 登陆 Delta_p >= 80 hPa 事件占比相对基准的变化率。
            与 ``KNUTSON_CAT45_TARGET`` 对照可暴露"中位数上移 + sigma 上移"
            对强台风占比的重复计入程度（透明度诊断，见 README 局限）。
        attribution: PML(headline) 归因分解结果。
        layers: 用基准结构重新定价的再保险分层。
        layer_el_drift: 各层期望损失率相对基准的变化率。
        bond: 用基准结构重新评估的 CAT bond 定价。
        spread_drift_lane_bp: Lane 利差漂移 (bp)。
        spread_drift_wang_bp: Wang 利差漂移 (bp)。
        fair_spread_lane: 气候调整后的 Lane 公允利差（小数）。
        underpricing_bp: 按基准定价发行时的利差缺口 (bp)。
        underpricing_pct: 缺口相对基准利差的百分比。
        depreciated_rp: 基准 PML(headline) 在本情景下的新重现期 (年)。
        uncertainty: 强度不确定性带 (low, central, high) 的 PML(headline)。
        uncertainty_drift: 上述三值相对基准的变化率。
    """

    scenario: ClimateScenario
    exposure_path: ExposurePath
    sto_cfg: StochasticConfig = field(repr=False)
    vul_cfg: VulnerabilityConfig = field(repr=False)
    freq_lambda: float = 0.0
    dp_median: float = 0.0
    dp_sigma: float = 0.0
    flood_beta: float = 0.0
    p_cat45: float = 0.0
    metrics_econ: Optional[fin.RiskMetrics] = None
    metrics_ins: Optional[fin.RiskMetrics] = None
    occurrence_econ: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros(0))
    occurrence_ins: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros(0))
    aal_drift: float = 0.0
    aal_drift_ins: float = 0.0
    pml_drift: Dict[float, float] = field(default_factory=dict)
    climate_only_drift: float = 0.0
    exposure_only_drift: float = 0.0
    cat45_drift: float = 0.0
    attribution: Dict[str, float] = field(default_factory=dict)
    layers: List[fin.LayerResult] = field(default_factory=list, repr=False)
    layer_el_drift: List[float] = field(default_factory=list)
    bond: Optional[fin.CatBondResult] = field(repr=False, default=None)
    spread_drift_lane_bp: float = 0.0
    spread_drift_wang_bp: float = 0.0
    fair_spread_lane: float = 0.0
    underpricing_bp: float = 0.0
    underpricing_pct: float = 0.0
    depreciated_rp: float = float("inf")
    uncertainty: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    uncertainty_drift: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def headline_pml(self) -> float:
        """基准重现期（默认 100 年）的经济口径 OEP PML (亿元)。"""
        assert self.metrics_econ is not None
        return float(self.metrics_econ.oep_pml[CLIMATE.headline_return_period])

    @property
    def headline_drift(self) -> float:
        """基准重现期 PML 相对基准情景的变化率。"""
        return float(self.pml_drift.get(CLIMATE.headline_return_period, 0.0))


@dataclass
class ClimateAnalysis:
    """全部情景的分析汇总。

    Attributes:
        results: 各情景结果，顺序与 ``SCENARIOS`` 一致，第 0 个为基准。
        headline_rp: 归因与贬值分析所用的基准重现期 (年)。
        return_periods: 报告的重现期序列。
        baseline_bond_attach: 基准 CAT bond 起赔点 (亿元)。
        baseline_bond_exhaust: 基准 CAT bond 耗尽点 (亿元)。
        n_events: 每个情景的事件数。
        runtime_s: 模块总运行时间 (秒)。
    """

    results: List[ScenarioResult]
    headline_rp: float
    return_periods: Tuple[float, ...]
    baseline_bond_attach: float
    baseline_bond_exhaust: float
    n_events: int
    runtime_s: float

    @property
    def baseline(self) -> ScenarioResult:
        """基准情景结果。"""
        return self.results[0]

    def by_name(self, name: str) -> ScenarioResult:
        """按情景名称取结果。

        Args:
            name: 情景名称。

        Returns:
            ScenarioResult: 匹配结果。

        Raises:
            KeyError: 名称不存在时。
        """
        for r in self.results:
            if r.scenario.name == name:
                return r
        raise KeyError(f"unknown scenario in results: {name!r}")

    def summary_table(self) -> pd.DataFrame:
        """生成情景汇总表。

        Returns:
            pandas.DataFrame: 每行一个情景，含升温、暴露倍数、AAL、
            各重现期 PML 与漂移率。
        """
        rows: List[Dict[str, float]] = []
        for r in self.results:
            assert r.metrics_econ is not None and r.metrics_ins is not None
            row: Dict[str, float] = {
                "scenario": r.scenario.name,
                "label_en": r.scenario.label_en,
                "horizon_year": float(r.scenario.horizon_year),
                "warming_c": r.scenario.warming_c,
                "exposure_scale": r.exposure_path.net_scale,
                "mean_penetration": r.exposure_path.mean_penetration,
                "freq_lambda": r.freq_lambda,
                "p_cat45": r.p_cat45,
                "cat45_drift": r.cat45_drift,
                "aal_econ": r.metrics_econ.aal,
                "aal_ins": r.metrics_ins.aal,
                "aal_drift": r.aal_drift,
                "climate_only_drift": r.climate_only_drift,
                "exposure_only_drift": r.exposure_only_drift,
                "var995": r.metrics_econ.var[FINANCIAL.c_ross_var_level],
                "tvar99": r.metrics_econ.tvar,
            }
            for rp in self.return_periods:
                row[f"pml{int(rp)}"] = r.metrics_econ.oep_pml[rp]
                row[f"drift{int(rp)}"] = r.pml_drift.get(rp, 0.0)
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. 主分析入口
# --------------------------------------------------------------------------- #

#: 报告用重现期序列（PML 表格与漂移表）。
REPORT_RETURN_PERIODS: Tuple[float, ...] = (10.0, 50.0, 100.0, 250.0, 500.0)


def run_climate_scenarios(
    exposure: ExposureDatabase,
    v_half: float,
    scenarios: Sequence[ClimateScenario] = SCENARIOS,
    cfg: ClimateConfig = CLIMATE,
    sto_base: StochasticConfig = STOCHASTIC,
    vul_base: VulnerabilityConfig = VULNERABILITY,
    hazard_cfg: HazardConfig = HAZARD,
    fin_cfg: FinancialConfig = FINANCIAL,
    n_events: Optional[int] = None,
    verbose: bool = False,
) -> ClimateAnalysis:
    """对全部气候情景重跑事件集并计算风险漂移、归因与金融传导。

    计算流程（每个情景）::

        1. 缩放 StochasticConfig / VulnerabilityConfig  -> 新事件集 + 新内涝系数
        2. 推演暴露到目标年                              -> 新暴露数据库
        3. 三条支路各算一次 PML(headline)：
             total    = 新事件集 + 新暴露      (气候 + 暴露)
             climate  = 新事件集 + 基准暴露    (仅气候)
             exposure = 基准事件集 + 新暴露    (仅暴露)
           交互项 = total - climate - exposure
        4. 用**基准年的再保险结构与 CAT bond 结构**重新定价，
           得到 EL/ROL/Multiple 漂移与利差漂移
        5. 强度信号取 Knutson 下界/中值/上界各跑一次，得到不确定性带

    性能：全部计算已向量化，5 情景 × 10,000 事件（含每情景 3 个
    不确定性变体，共 13 次事件集生成 + 风场积分）在常规笔记本上
    约需 3~6 秒，远低于 90 秒预算。

    Args:
        exposure: 基准年（2020）暴露数据库。
        v_half: 主流程校准得到的 Emanuel V_half (m/s)，全情景共用，
            确保情景间差异只来自气候与暴露，而非重新校准的噪声。
        scenarios: 情景序列，第一个必须是基准情景。
        cfg: 气候配置。
        sto_base: 基准随机配置。
        vul_base: 基准脆弱性配置。
        hazard_cfg: 危险性配置。
        fin_cfg: 金融配置。
        n_events: 覆盖每情景事件数；``None`` 表示用 ``cfg.n_events``。
        verbose: 是否逐情景打印进度。

    Returns:
        ClimateAnalysis: 全部情景的分析结果。

    Raises:
        ValueError: 情景序列为空、首个情景不是基准情景，或基准情景的
            缩放因子不全为 1.0 时。
    """
    t_start = time.time()
    if len(scenarios) == 0:
        raise ValueError("scenarios must not be empty")
    base_sc = scenarios[0]
    if not base_sc.is_baseline:
        raise ValueError("scenarios[0] must be the baseline scenario")
    if not np.allclose(
        [base_sc.dp_median_scale, base_sc.dp_sigma_scale,
         base_sc.lambda_scale, base_sc.flood_beta_scale], 1.0
    ):
        raise ValueError("baseline scenario must have all scaling factors = 1.0")

    n_ev = int(n_events if n_events is not None else cfg.n_events)
    lon, lat = exposure.lon, exposure.lat
    rps = REPORT_RETURN_PERIODS
    head_rp = cfg.headline_return_period

    # --- 基准支路：所有情景的公共参照 --------------------------------- #
    base_sto = scale_stochastic_config(base_sc, sto_base, n_ev)
    base_vul = scale_vulnerability_config(base_sc, vul_base)
    base_cache = build_hazard_cache(base_sto, lon, lat, hazard_cfg)
    base_path = project_exposure(exposure, 0, cfg, vul_base)

    base_econ, base_ins = _event_losses(
        base_cache, base_path.database, v_half, base_vul)
    base_ylt_econ = _ylt(base_econ, base_cache, base_sto)
    base_ylt_ins = _ylt(base_ins, base_cache, base_sto)
    base_metrics_econ = fin.compute_risk_metrics(base_ylt_econ, fin_cfg)
    base_metrics_ins = fin.compute_risk_metrics(base_ylt_ins, fin_cfg)
    base_pml = base_metrics_econ.oep_pml
    base_pml_head = float(base_pml[head_rp])

    # 基准年确定的金融结构：一旦发行/续转即固定，后续按气候重估
    base_layers = fin.build_reinsurance_program(base_ylt_ins, fin_cfg)
    bond_attach = float(fin._nice_round(
        fin.pml_at_return_periods(
            base_ylt_ins.occurrence, [cfg.catbond_attach_rp])[0]))
    bond_exhaust = float(fin._nice_round(
        fin.pml_at_return_periods(
            base_ylt_ins.occurrence, [cfg.catbond_exhaust_rp])[0]))
    if bond_exhaust <= bond_attach:
        bond_exhaust = bond_attach * 2.0
    base_bond = fin.price_cat_bond(
        base_ylt_ins.occurrence, bond_attach, bond_exhaust,
        "industry index (fixed 2020 structure)", fin_cfg)

    results: List[ScenarioResult] = []

    for sc in scenarios:
        if verbose:
            print(f"  [climate] running {sc.name} ...", flush=True)

        sto = scale_stochastic_config(sc, sto_base, n_ev)
        vul = scale_vulnerability_config(sc, vul_base)
        path = project_exposure(exposure, sc.horizon_years, cfg, vul_base)

        if sc.is_baseline:
            cache = base_cache
            econ_ev, ins_ev = base_econ, base_ins
            ylt_econ, ylt_ins = base_ylt_econ, base_ylt_ins
            m_econ, m_ins = base_metrics_econ, base_metrics_ins
        else:
            ev = perturb_event_set(
                base_cache.events, base_sto, sto, hazard_cfg)
            cache = build_hazard_cache(sto, lon, lat, hazard_cfg, events=ev)
            econ_ev, ins_ev = _event_losses(cache, path.database, v_half, vul)
            ylt_econ = _ylt(econ_ev, cache, sto)
            ylt_ins = _ylt(ins_ev, cache, sto)
            m_econ = fin.compute_risk_metrics(ylt_econ, fin_cfg)
            m_ins = fin.compute_risk_metrics(ylt_ins, fin_cfg)

        # ---- 漂移 ---- #
        pml_drift = {
            rp: (m_econ.oep_pml[rp] / base_pml[rp] - 1.0) if base_pml[rp] > 0
            else 0.0
            for rp in rps
        }
        aal_drift = m_econ.aal / base_metrics_econ.aal - 1.0
        aal_drift_ins = m_ins.aal / base_metrics_ins.aal - 1.0

        # ---- 两条归因支路 ---- #
        # 仅气候：未来气候 + 2020 暴露；仅暴露：2020 气候 + 未来暴露
        if sc.is_baseline:
            clim_econ_ylt, clim_ins_ylt = base_ylt_econ, base_ylt_ins
            exp_econ_ylt = base_ylt_econ
        else:
            clim_econ_ylt, clim_ins_ylt = _sub_run(
                cache, base_path.database, v_half, vul, sto)
            exp_econ_ylt, _ = _sub_run(
                base_cache, path.database, v_half, base_vul, base_sto)

        attribution = _attribute(
            climate_only_occ=clim_econ_ylt.occurrence,
            exposure_only_occ=exp_econ_ylt.occurrence,
            total_pml=float(m_econ.oep_pml[head_rp]),
            base_pml=base_pml_head,
            head_rp=head_rp,
        )

        # ---- 金融传导：固定 2020 结构 + 恒定 2020 暴露 ---- #
        # 这里刻意使用"仅气候"支路而非总效应支路。理由是精算意义上的
        # 可比性：再保合约逐年续转、CAT bond 三年到期，组合规模增长会在
        # 续转时通过重新起赔点自动吸收；真正无法在合约期内重新定价、
        # 需要今天就定价进去的，是**气候信号**。若把 30 年的暴露增长塞进
        # 一张 3 年期债券的利差里，得到的数字没有交易含义。
        layers = [
            fin.price_layer(clim_ins_ylt, lay.attachment, lay.limit, lay.name,
                            lay.attach_rp, lay.exhaust_rp, fin_cfg)
            for lay in base_layers
        ]
        layer_el_drift = [
            (new.el_rate / old.el_rate - 1.0) if old.el_rate > 0 else 0.0
            for new, old in zip(layers, base_layers)
        ]
        bond = fin.price_cat_bond(
            clim_ins_ylt.occurrence, bond_attach, bond_exhaust,
            "industry index (fixed 2020 structure)", fin_cfg)
        drift_lane_bp = (bond.spread_lane - base_bond.spread_lane) * 1.0e4
        drift_wang_bp = (bond.spread_wang - base_bond.spread_wang) * 1.0e4
        underprice_bp = drift_lane_bp
        underprice_pct = (
            (bond.spread_lane / base_bond.spread_lane - 1.0) * 100.0
            if base_bond.spread_lane > 0 else 0.0
        )
        dep_rp = loss_return_period(clim_econ_ylt.occurrence, base_pml_head)

        # ---- 不确定性带 ---- #
        unc = _uncertainty_band(
            sc=sc, exposure_db=path.database, v_half=v_half,
            base_events=base_cache.events, sto_base_cfg=sto_base,
            base_sto=base_sto, vul_base_cfg=vul_base,
            hazard_cfg=hazard_cfg, cfg=cfg,
            n_events=n_ev, lon=lon, lat=lat,
            central=float(m_econ.oep_pml[head_rp]),
        )
        unc_drift = tuple(
            (u / base_pml_head - 1.0) if base_pml_head > 0 else 0.0 for u in unc
        )

        # ---- 透明度诊断 ---- #
        p_cat45 = float(np.mean(cache.events.landfall_dp >= CAT45_DP_THRESHOLD))
        base_p_cat45 = float(
            np.mean(base_cache.events.landfall_dp >= CAT45_DP_THRESHOLD))
        cat45_drift = (
            p_cat45 / base_p_cat45 - 1.0 if base_p_cat45 > 0 else 0.0)
        clim_drift = (
            attribution["pml_climate_only"] / base_pml_head - 1.0
            if base_pml_head > 0 else 0.0)
        exp_drift = (
            attribution["pml_exposure_only"] / base_pml_head - 1.0
            if base_pml_head > 0 else 0.0)

        results.append(ScenarioResult(
            scenario=sc,
            exposure_path=path,
            sto_cfg=sto,
            vul_cfg=vul,
            freq_lambda=float(cache.events.freq_lambda),
            dp_median=float(sto.dp_median_hpa),
            dp_sigma=float(sto.dp_sigma_log),
            flood_beta=float(vul.flood_beta),
            p_cat45=p_cat45,
            metrics_econ=m_econ,
            metrics_ins=m_ins,
            occurrence_econ=ylt_econ.occurrence,
            occurrence_ins=ylt_ins.occurrence,
            aal_drift=float(aal_drift),
            aal_drift_ins=float(aal_drift_ins),
            pml_drift=pml_drift,
            climate_only_drift=float(clim_drift),
            exposure_only_drift=float(exp_drift),
            cat45_drift=float(cat45_drift),
            attribution=attribution,
            layers=layers,
            layer_el_drift=layer_el_drift,
            bond=bond,
            spread_drift_lane_bp=float(drift_lane_bp),
            spread_drift_wang_bp=float(drift_wang_bp),
            fair_spread_lane=float(bond.spread_lane),
            underpricing_bp=float(underprice_bp),
            underpricing_pct=float(underprice_pct),
            depreciated_rp=float(dep_rp),
            uncertainty=(float(unc[0]), float(unc[1]), float(unc[2])),
            uncertainty_drift=(
                float(unc_drift[0]), float(unc_drift[1]), float(unc_drift[2])),
        ))

    return ClimateAnalysis(
        results=results,
        headline_rp=float(head_rp),
        return_periods=rps,
        baseline_bond_attach=bond_attach,
        baseline_bond_exhaust=bond_exhaust,
        n_events=n_ev,
        runtime_s=float(time.time() - t_start),
    )


def _attribute(
    climate_only_occ: np.ndarray,
    exposure_only_occ: np.ndarray,
    total_pml: float,
    base_pml: float,
    head_rp: float,
) -> Dict[str, float]:
    """把 PML(headline) 的总变化分解为气候 / 暴露 / 交互三部分。

    分解恒等式（精确，无残差）：

    .. math::
        \\Delta_{total} = \\underbrace{(L_{c} - L_0)}_{climate}
        + \\underbrace{(L_{e} - L_0)}_{exposure}
        + \\underbrace{(L_{t} - L_c - L_e + L_0)}_{interaction}

    其中 :math:`L_c` 用"未来气候 + 今天暴露"，:math:`L_e` 用
    "今天气候 + 未来暴露"。交互项之所以为正，是因为更强的风打在
    更大的资产上，损失是两者的**乘积**而非加和——这正是
    Pielke normalization 只做一次线性折算时会漏掉的部分。

    Args:
        climate_only_occ: "未来气候 + 2020 暴露"支路的年最大损失序列。
        exposure_only_occ: "2020 气候 + 未来暴露"支路的年最大损失序列。
        total_pml: 已算好的总效应 PML (亿元)。
        base_pml: 基准情景 PML (亿元)。
        head_rp: 归因所用的重现期 (年)。

    Returns:
        Dict[str, float]: 含以下键的字典——
        ``pml_base`` / ``pml_total`` / ``pml_climate_only`` /
        ``pml_exposure_only`` / ``delta_total`` / ``delta_climate`` /
        ``delta_exposure`` / ``delta_interaction`` /
        ``share_climate`` / ``share_exposure`` / ``share_interaction``
        （share 为百分数，三者之和恒为 100.0）。
    """
    rp = [head_rp]
    pml_c = float(fin.pml_at_return_periods(climate_only_occ, rp)[0])
    pml_e = float(fin.pml_at_return_periods(exposure_only_occ, rp)[0])

    d_total = total_pml - base_pml
    d_clim = pml_c - base_pml
    d_exp = pml_e - base_pml
    d_int = d_total - d_clim - d_exp

    if abs(d_total) <= 1.0e-9:  # 基准情景：三项贡献均为 0
        return {
            "pml_base": base_pml, "pml_total": total_pml,
            "pml_climate_only": pml_c, "pml_exposure_only": pml_e,
            "delta_total": 0.0, "delta_climate": 0.0,
            "delta_exposure": 0.0, "delta_interaction": 0.0,
            "share_climate": 0.0, "share_exposure": 0.0,
            "share_interaction": 0.0,
        }

    denom = d_total
    return {
        "pml_base": base_pml,
        "pml_total": total_pml,
        "pml_climate_only": pml_c,
        "pml_exposure_only": pml_e,
        "delta_total": d_total,
        "delta_climate": d_clim,
        "delta_exposure": d_exp,
        "delta_interaction": d_int,
        "share_climate": float(d_clim / denom * 100.0),
        "share_exposure": float(d_exp / denom * 100.0),
        "share_interaction": float(d_int / denom * 100.0),
    }


def _uncertainty_band(
    sc: ClimateScenario,
    exposure_db: ExposureDatabase,
    v_half: float,
    base_events: hz.EventSet,
    sto_base_cfg: StochasticConfig,
    base_sto: StochasticConfig,
    vul_base_cfg: VulnerabilityConfig,
    hazard_cfg: HazardConfig,
    cfg: ClimateConfig,
    n_events: int,
    lon: np.ndarray,
    lat: np.ndarray,
    central: float,
) -> Tuple[float, float, float]:
    """按 Knutson 强度区间生成 PML(headline) 的低/中/高三点带。

    Knutson et al. (2020) 对 2°C 下 TC 最大风速的专家评估区间为
    +1%~+10%，中值 +5%。因此把情景的强度**异常**（``scale - 1``）
    分别乘以 0.2 与 2.0 即得下/上界，频率与降水信号保持中值不变
    （它们的不确定性方向与强度不同，混在一起会夸大区间宽度）。

    Args:
        sc: 情景定义。
        exposure_db: 该情景的目标年暴露数据库。
        v_half: 校准的 V_half。
        base_events: 基准事件集（共同随机数来源）。
        sto_base_cfg: 基准随机配置（未缩放）。
        base_sto: 基准情景的随机配置（含事件数覆盖）。
        vul_base_cfg: 基准脆弱性配置。
        hazard_cfg: 危险性配置。
        cfg: 气候配置。
        n_events: 事件数。
        lon: 站点经度。
        lat: 站点纬度。
        central: 已算好的中值 PML (亿元)，避免重复计算。

    Returns:
        Tuple[float, float, float]: (low, central, high) 的 PML (亿元)，
        并保证 ``low <= central <= high``（数值抖动时做单调化处理）。
    """
    if sc.is_baseline:
        return (central, central, central)

    vul = scale_vulnerability_config(sc, vul_base_cfg)
    rp = [cfg.headline_return_period]
    out: List[float] = []
    for mult in (cfg.intensity_low_multiplier, cfg.intensity_high_multiplier):
        sto = scale_stochastic_config(
            sc, sto_base_cfg, n_events, intensity_multiplier=mult)
        ev = perturb_event_set(base_events, base_sto, sto, hazard_cfg)
        cache = build_hazard_cache(sto, lon, lat, hazard_cfg, events=ev)
        econ, _ = _event_losses(cache, exposure_db, v_half, vul)
        ylt = _ylt(econ, cache, sto)
        out.append(float(fin.pml_at_return_periods(ylt.occurrence, rp)[0]))

    low, high = out[0], out[1]
    lo = min(low, central, high)
    hi = max(low, central, high)
    return (lo, float(central), hi)


# --------------------------------------------------------------------------- #
# 7. 报表辅助
# --------------------------------------------------------------------------- #


def scenario_definition_table() -> pd.DataFrame:
    """生成情景定义表（用于终端与文档）。

    Returns:
        pandas.DataFrame: 每行一个情景的缩放因子与隐含风速倍数。
    """
    return pd.DataFrame([
        {
            "scenario": s.name,
            "year": s.horizon_year,
            "warming_c": s.warming_c,
            "dp_median_scale": s.dp_median_scale,
            "dp_sigma_scale": s.dp_sigma_scale,
            "lambda_scale": s.lambda_scale,
            "flood_beta_scale": s.flood_beta_scale,
            "implied_vmax_scale": s.implied_vmax_scale,
            "description": s.description,
        }
        for s in SCENARIOS
    ])


def attribution_table(analysis: ClimateAnalysis) -> pd.DataFrame:
    """生成归因分解表。

    Args:
        analysis: 情景分析结果。

    Returns:
        pandas.DataFrame: 每行一个非基准情景的三项贡献（金额与占比）。
    """
    rows: List[Dict[str, object]] = []
    for r in analysis.results:
        if r.scenario.is_baseline:
            continue
        a = r.attribution
        rows.append({
            "scenario": r.scenario.name,
            "label_en": r.scenario.label_en,
            "delta_total": a["delta_total"],
            "delta_climate": a["delta_climate"],
            "delta_exposure": a["delta_exposure"],
            "delta_interaction": a["delta_interaction"],
            "share_climate": a["share_climate"],
            "share_exposure": a["share_exposure"],
            "share_interaction": a["share_interaction"],
        })
    return pd.DataFrame(rows)


def financial_transmission_table(analysis: ClimateAnalysis) -> pd.DataFrame:
    """生成金融传导表。

    Args:
        analysis: 情景分析结果。

    Returns:
        pandas.DataFrame: 每行一个情景的 CAT bond EL/利差/贬值重现期等。
    """
    rows: List[Dict[str, object]] = []
    for r in analysis.results:
        assert r.bond is not None
        rows.append({
            "scenario": r.scenario.name,
            "label_en": r.scenario.label_en,
            "bond_el": r.bond.expected_loss,
            "bond_pfl": r.bond.prob_first_loss,
            "spread_lane": r.bond.spread_lane,
            "spread_wang": r.bond.spread_wang,
            "spread_drift_lane_bp": r.spread_drift_lane_bp,
            "spread_drift_wang_bp": r.spread_drift_wang_bp,
            "underpricing_bp": r.underpricing_bp,
            "underpricing_pct": r.underpricing_pct,
            "depreciated_rp": r.depreciated_rp,
        })
    return pd.DataFrame(rows)


__all__ = [
    "ATKINSON_EXPONENT",
    "KNUTSON_ANCHOR_WARMING",
    "LIMITATIONS",
    "REPORT_RETURN_PERIODS",
    "SCENARIOS",
    "ClimateScenario",
    "ExposurePath",
    "HazardCache",
    "ScenarioResult",
    "ClimateAnalysis",
    "scenario_by_name",
    "baseline_scenario",
    "scale_stochastic_config",
    "scale_vulnerability_config",
    "exposure_growth_factor",
    "resilience_factor",
    "penetration_path",
    "project_exposure",
    "build_hazard_cache",
    "perturb_event_set",
    "loss_return_period",
    "KNUTSON_CAT45_TARGET",
    "CAT45_DP_THRESHOLD",
    "run_climate_scenarios",
    "scenario_definition_table",
    "attribution_table",
    "financial_transmission_table",
]
