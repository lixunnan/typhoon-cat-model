"""模块 D —— 巨灾金融工程 (Financial Module)。

覆盖内容：

1. **年度损失分布**：由事件集 + 泊松频率模拟年度 OEP / AEP。
2. **风险度量**：AAL、EP 曲线、各重现期 PML、VaR、TVaR/CVaR，
   并按中国偿二代二期 (C-ROSS II) 思路给出巨灾风险最低资本要求估算。
3. **再保险超赔分层定价 (Excess of Loss)**：期望赔付、Rate on Line、
   纯保费、加载保费、Multiple。
4. **巨灾债券 (CAT Bond) 定价**：指数触发与参数触发两种结构；
   Lane (2000) 市场经验模型与 Wang (2000/2002) 变换两种定价方法；
   起赔点敏感性分析。
5. **基差风险量化**：参数触发赔付 vs 实际损失，相关系数与对冲效率。
6. **投资组合视角**：CAT bond 的零贝塔分散化价值、夏普比率改善、
   有效前沿变化。

文献:
    Lane, M. N. (2000). Pricing risk transfer transactions.
        *ASTIN Bulletin*, 30(2), 259-293.
    Wang, S. S. (2000). A class of distortion operators for pricing financial
        and insurance risks. *Journal of Risk and Insurance*, 67(1), 15-36.
    Wang, S. S. (2002). A universal framework for pricing financial and
        insurance risks. *ASTIN Bulletin*, 32(2), 213-234.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from config import FINANCIAL, STOCHASTIC, FinancialConfig


# --------------------------------------------------------------------------- #
# 1. 年度损失分布
# --------------------------------------------------------------------------- #


@dataclass
class YearLossTable:
    """年度损失表 (Year Loss Table, YLT)。

    Attributes:
        occurrence: 每年最大单次事件损失 ``(Y,)`` (亿元)，用于 OEP。
        aggregate: 每年累计损失 ``(Y,)`` (亿元)，用于 AEP。
        n_events_per_year: 每年事件数 ``(Y,)``。
        event_index: 展开后每次事件对应的事件集索引 ``(E,)``。
        event_loss: 展开后每次事件的损失 ``(E,)`` (亿元)。
        event_year: 展开后每次事件所属年份 ``(E,)``。
        n_years: 模拟年数。
        freq_lambda: 泊松年频率。
    """

    occurrence: np.ndarray
    aggregate: np.ndarray
    n_events_per_year: np.ndarray
    event_index: np.ndarray
    event_loss: np.ndarray
    event_year: np.ndarray
    n_years: int
    freq_lambda: float

    @property
    def aal(self) -> float:
        """年均损失 AAL (Average Annual Loss)，单位亿元。"""
        return float(self.aggregate.mean())

    @property
    def std_aggregate(self) -> float:
        """年度累计损失的标准差 (亿元)。"""
        return float(self.aggregate.std(ddof=1))


def build_year_loss_table(
    event_losses: np.ndarray,
    freq_lambda: float,
    n_years: int = STOCHASTIC.n_simulation_years,
    random_seed: int = STOCHASTIC.random_seed,
) -> YearLossTable:
    """由事件损失与泊松频率构建年度损失表。

    模拟逻辑：每年事件数 :math:`n_y \\sim \\text{Poisson}(\\lambda)`，
    每次事件从事件集中等概率有放回抽取（事件集本身已按强度分布抽样，
    因此等概率抽取即代表频率-强度联合分布）。

    Args:
        event_losses: 事件集损失数组 ``(N,)`` (亿元)。
        freq_lambda: 泊松年频率参数。
        n_years: 模拟年数。
        random_seed: 随机种子。

    Returns:
        YearLossTable: 年度损失表。
    """
    rng = np.random.default_rng(random_seed + 7)
    losses = np.asarray(event_losses, dtype=float)
    counts = rng.poisson(freq_lambda, size=n_years)
    total = int(counts.sum())

    idx = rng.integers(0, losses.size, size=total)
    year_id = np.repeat(np.arange(n_years, dtype=np.int64), counts)
    ev_loss = losses[idx]

    aggregate = np.bincount(year_id, weights=ev_loss, minlength=n_years)
    occurrence = np.zeros(n_years, dtype=float)
    if total > 0:
        np.maximum.at(occurrence, year_id, ev_loss)

    return YearLossTable(
        occurrence=occurrence,
        aggregate=aggregate,
        n_events_per_year=counts,
        event_index=idx,
        event_loss=ev_loss,
        event_year=year_id,
        n_years=n_years,
        freq_lambda=float(freq_lambda),
    )


def ep_curve(losses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由年度损失序列构造超越概率 (EP) 曲线。

    Args:
        losses: 年度损失序列 ``(Y,)`` (亿元)。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - 降序排列的损失 ``(Y,)``
            - 对应的年超越概率 ``(Y,)``
            - 对应的重现期 (年) ``(Y,)``
    """
    x = np.sort(np.asarray(losses, dtype=float))[::-1]
    n = x.size
    rank = np.arange(1, n + 1, dtype=float)
    exceed_prob = rank / (n + 1.0)
    return x, exceed_prob, 1.0 / exceed_prob


def pml_at_return_periods(
    losses: np.ndarray, return_periods: Sequence[float]
) -> np.ndarray:
    """计算指定重现期的 PML (Probable Maximum Loss)。

    Args:
        losses: 年度损失序列 ``(Y,)`` (亿元)。
        return_periods: 重现期列表 (年)。

    Returns:
        np.ndarray: 对应各重现期的损失 (亿元)。
    """
    x = np.asarray(losses, dtype=float)
    q = 1.0 - 1.0 / np.asarray(return_periods, dtype=float)
    return np.quantile(x, np.clip(q, 0.0, 1.0), method="linear")


def var_tvar(losses: np.ndarray, level: float) -> Tuple[float, float]:
    """计算给定置信水平的 VaR 与 TVaR/CVaR。

    .. math::
        VaR_\\alpha = F^{-1}(\\alpha), \\qquad
        TVaR_\\alpha = E[L \\mid L > VaR_\\alpha]

    Args:
        losses: 年度损失序列 ``(Y,)`` (亿元)。
        level: 置信水平，如 0.99。

    Returns:
        Tuple[float, float]: (VaR, TVaR)，单位亿元。
    """
    x = np.asarray(losses, dtype=float)
    v = float(np.quantile(x, level, method="linear"))
    tail = x[x > v]
    t = float(tail.mean()) if tail.size > 0 else v
    return v, t


@dataclass
class RiskMetrics:
    """风险度量汇总。

    Attributes:
        aal: 年均损失 (亿元)。
        std: 年度损失标准差 (亿元)。
        oep_pml: 各重现期 OEP PML，键为重现期。
        aep_pml: 各重现期 AEP PML，键为重现期。
        var: 各置信水平 VaR (基于 AEP)，键为置信水平。
        tvar: TVaR (基于 AEP)。
        tvar_level: TVaR 置信水平。
        c_ross_capital: 偿二代二期口径巨灾最低资本要求 (亿元)。
        c_ross_level: 偿二代二期 VaR 置信水平。
        loss_free_prob: 无损失年份占比。
    """

    aal: float
    std: float
    oep_pml: Dict[float, float]
    aep_pml: Dict[float, float]
    var: Dict[float, float]
    tvar: float
    tvar_level: float
    c_ross_capital: float
    c_ross_level: float
    loss_free_prob: float


def compute_risk_metrics(
    ylt: YearLossTable, cfg: FinancialConfig = FINANCIAL
) -> RiskMetrics:
    """计算全套风险度量指标。

    偿二代二期 (C-ROSS II) 巨灾风险最低资本按 VaR(99.5%) 口径超出
    期望损失的部分估算：

    .. math::
        MCR_{cat} = VaR_{99.5\\%}(L_{agg}) - E[L_{agg}]

    Args:
        ylt: 年度损失表。
        cfg: 金融配置。

    Returns:
        RiskMetrics: 风险度量汇总对象。
    """
    rps = list(cfg.pml_return_periods)
    oep = pml_at_return_periods(ylt.occurrence, rps)
    aep = pml_at_return_periods(ylt.aggregate, rps)

    var_map: Dict[float, float] = {}
    for lv in cfg.var_levels:
        var_map[lv] = var_tvar(ylt.aggregate, lv)[0]
    _, tvar = var_tvar(ylt.aggregate, cfg.tvar_level)
    var_995 = var_tvar(ylt.aggregate, cfg.c_ross_var_level)[0]

    return RiskMetrics(
        aal=ylt.aal,
        std=ylt.std_aggregate,
        oep_pml={rp: float(v) for rp, v in zip(rps, oep)},
        aep_pml={rp: float(v) for rp, v in zip(rps, aep)},
        var=var_map,
        tvar=float(tvar),
        tvar_level=cfg.tvar_level,
        c_ross_capital=float(var_995 - ylt.aal),
        c_ross_level=cfg.c_ross_var_level,
        loss_free_prob=float(np.mean(ylt.aggregate <= 0.0)),
    )


# --------------------------------------------------------------------------- #
# 2. 再保险分层定价
# --------------------------------------------------------------------------- #


@dataclass
class LayerResult:
    """再保险单层定价结果。

    Attributes:
        name: 层名称。
        attachment: 起赔点 (亿元)。
        limit: 层限额 (亿元)。
        exhaustion: 耗尽点 = 起赔点 + 限额 (亿元)。
        attach_rp: 起赔点对应重现期 (年)。
        exhaust_rp: 耗尽点对应重现期 (年)。
        expected_loss: 年期望赔付 (亿元)。
        el_rate: 期望损失率 = 期望赔付 / 限额。
        std_recovery: 年赔付标准差 (亿元)。
        prob_attach: 起赔概率 (年触发概率)。
        prob_exhaust: 耗尽概率。
        pure_premium: 纯保费 (亿元)。
        loaded_premium: 加载保费 (亿元)。
        rate_on_line: ROL = 加载保费 / 限额。
        multiple: Multiple = ROL / EL rate。
    """

    name: str
    attachment: float
    limit: float
    exhaustion: float
    attach_rp: float
    exhaust_rp: float
    expected_loss: float
    el_rate: float
    std_recovery: float
    prob_attach: float
    prob_exhaust: float
    pure_premium: float
    loaded_premium: float
    rate_on_line: float
    multiple: float


def layer_annual_recovery(
    ylt: YearLossTable, attachment: float, limit: float
) -> np.ndarray:
    """计算 XoL 层的逐年赔付额（按次超赔，年内累加）。

    单次事件赔付：

    .. math::
        R_e = \\min\\big(\\max(L_e - A,\\,0),\\; \\text{Limit}\\big)

    年赔付为年内各次事件赔付之和（无年度限额/复效限制，属简化假设）。

    Args:
        ylt: 年度损失表。
        attachment: 起赔点 (亿元)。
        limit: 层限额 (亿元)。

    Returns:
        np.ndarray: 逐年赔付 ``(Y,)`` (亿元)。
    """
    rec = np.minimum(np.maximum(ylt.event_loss - attachment, 0.0), limit)
    return np.bincount(ylt.event_year, weights=rec, minlength=ylt.n_years)


def price_layer(
    ylt: YearLossTable,
    attachment: float,
    limit: float,
    name: str = "Layer",
    attach_rp: float = float("nan"),
    exhaust_rp: float = float("nan"),
    cfg: FinancialConfig = FINANCIAL,
) -> LayerResult:
    """对单层超赔再保险定价。

    采用标准差保费原理 (standard deviation principle) 加载，再叠加费用率：

    .. math::
        P = \\big(E[R] + k\\,\\sigma_R\\big)\\,(1 + e)

    其中 :math:`k` 为风险载荷系数，:math:`e` 为费用附加率。

    Args:
        ylt: 年度损失表。
        attachment: 起赔点 (亿元)。
        limit: 层限额 (亿元)。
        name: 层名称。
        attach_rp: 起赔点重现期（仅用于展示）。
        exhaust_rp: 耗尽点重现期（仅用于展示）。
        cfg: 金融配置。

    Returns:
        LayerResult: 该层的定价结果。
    """
    recovery = layer_annual_recovery(ylt, attachment, limit)
    el = float(recovery.mean())
    sd = float(recovery.std(ddof=1))
    pure = el
    loaded = (el + cfg.layer_sd_load * sd) * (1.0 + cfg.layer_expense_ratio)
    rol = loaded / limit
    el_rate = el / limit
    return LayerResult(
        name=name,
        attachment=float(attachment),
        limit=float(limit),
        exhaustion=float(attachment + limit),
        attach_rp=float(attach_rp),
        exhaust_rp=float(exhaust_rp),
        expected_loss=el,
        el_rate=el_rate,
        std_recovery=sd,
        prob_attach=float(np.mean(recovery > 0.0)),
        prob_exhaust=float(np.mean(recovery >= limit - 1.0e-9)),
        pure_premium=pure,
        loaded_premium=float(loaded),
        rate_on_line=float(rol),
        multiple=float(rol / el_rate) if el_rate > 0 else float("nan"),
    )


def build_reinsurance_program(
    ylt: YearLossTable, cfg: FinancialConfig = FINANCIAL
) -> List[LayerResult]:
    """按 OEP 重现期自动构造并定价三层超赔再保险方案。

    层的起赔点/耗尽点由 OEP 曲线的重现期分位数确定，并向上取整到
    便于交易的整数金额，保证结构随组合规模自适应。

    Args:
        ylt: 年度损失表。
        cfg: 金融配置。

    Returns:
        List[LayerResult]: 三层定价结果，按起赔点升序。
    """
    layers: List[LayerResult] = []
    for i, (a_rp, e_rp) in enumerate(cfg.layer_return_periods, start=1):
        a = float(pml_at_return_periods(ylt.occurrence, [a_rp])[0])
        e = float(pml_at_return_periods(ylt.occurrence, [e_rp])[0])
        a = _nice_round(a)
        e = _nice_round(e)
        if e <= a:
            e = a * 1.5
        layers.append(
            price_layer(ylt, a, e - a, f"Layer {i}", a_rp, e_rp, cfg)
        )
    return layers


def _nice_round(x: float) -> float:
    """将金额取整到便于交易的"整数"档位。

    Args:
        x: 原始金额 (亿元)。

    Returns:
        float: 取整后的金额 (亿元)。
    """
    if x <= 0:
        return 0.0
    mag = 10.0 ** np.floor(np.log10(x))
    step = mag / 2.0 if x / mag < 3 else mag
    return float(np.round(x / step) * step)


# --------------------------------------------------------------------------- #
# 3. CAT Bond 定价
# --------------------------------------------------------------------------- #


@dataclass
class CatBondResult:
    """巨灾债券定价结果。

    Attributes:
        trigger_type: 触发类型（``"industry index"`` / ``"parametric"``）。
        attachment: 起赔点 (亿元，参数触发时为等价损失口径)。
        exhaustion: 耗尽点 (亿元)。
        principal: 债券本金 (亿元)。
        expected_loss: 期望损失率 EL（占本金比例，年化）。
        prob_first_loss: 首次损失概率 PFL。
        prob_total_loss: 全损概率。
        cond_expected_loss: 条件期望损失 CEL = EL / PFL。
        spread_lane: Lane (2000) 模型给出的信用利差（小数）。
        spread_wang: Wang 变换给出的信用利差（小数）。
        wang_lambda_used: Wang 变换所用的市场风险价格 lambda。
        wang_lambda_implied: 由 Lane 利差反推的隐含 lambda。
        coupon_lane: Lane 口径票息 = rf + spread。
        coupon_wang: Wang 口径票息 = rf + spread。
        investor_expected_return_lane: 投资人期望收益 = 票息 - EL。
        investor_expected_return_wang: 同上，Wang 口径。
        multiple_lane: Multiple = spread / EL (Lane)。
        multiple_wang: Multiple = spread / EL (Wang)。
        loss_ratios: 逐年债券本金损失率 ``(Y,)``。
    """

    trigger_type: str
    attachment: float
    exhaustion: float
    principal: float
    expected_loss: float
    prob_first_loss: float
    prob_total_loss: float
    cond_expected_loss: float
    spread_lane: float
    spread_wang: float
    wang_lambda_used: float
    wang_lambda_implied: float
    coupon_lane: float
    coupon_wang: float
    investor_expected_return_lane: float
    investor_expected_return_wang: float
    multiple_lane: float
    multiple_wang: float
    loss_ratios: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


def tranche_loss_ratio(
    annual_index: np.ndarray, attachment: float, exhaustion: float
) -> np.ndarray:
    """给定年度触发指数序列，计算债券分层的本金损失率。

    .. math::
        LR = \\frac{\\min\\big(\\max(I - A,\\,0),\\; E - A\\big)}{E - A}

    Args:
        annual_index: 年度触发指数 ``(Y,)`` (亿元或等价口径)。
        attachment: 起赔点。
        exhaustion: 耗尽点。

    Returns:
        np.ndarray: 逐年本金损失率 ``(Y,)``，取值 [0, 1]。

    Raises:
        ValueError: 当 ``exhaustion <= attachment`` 时。
    """
    if exhaustion <= attachment:
        raise ValueError("exhaustion must exceed attachment")
    size = exhaustion - attachment
    x = np.asarray(annual_index, dtype=float)
    return np.clip((x - attachment) / size, 0.0, 1.0)


def lane_spread(
    pfl: float, cel: float, el: float, cfg: FinancialConfig = FINANCIAL
) -> float:
    """Lane (2000) 市场经验定价模型。

    Lane 将超额期望收益 (Expected Excess Return) 拟合为频率与
    严重度的 Cobb-Douglas 形式：

    .. math::
        EER = \\gamma \\cdot PFL^{\\alpha} \\cdot CEL^{\\beta}

        Spread = EL + EER

    其中 PFL 为首次损失概率 (Probability of First Loss)，
    CEL 为条件期望损失 (Conditional Expected Loss)。

    Args:
        pfl: 首次损失概率。
        cel: 条件期望损失（占本金比例）。
        el: 期望损失率（占本金比例）。
        cfg: 金融配置。

    Returns:
        float: 信用利差（小数形式）。
    """
    if pfl <= 0.0 or cel <= 0.0:
        return 0.0
    eer = cfg.lane_gamma * (pfl ** cfg.lane_alpha) * (cel ** cfg.lane_beta)
    return float(el + eer)


def wang_transform_price(loss_ratios: np.ndarray, lam: float) -> float:
    """Wang (2000/2002) 变换下的风险调整期望损失。

    对损失分布 :math:`F` 施加畸变算子：

    .. math::
        F^*(x) = \\Phi\\big(\\Phi^{-1}(F(x)) - \\lambda\\big)

    其中 :math:`\\lambda > 0` 为市场风险价格 (Sharpe-like)，
    风险中性价格为：

    .. math::
        E^*[Y] = \\int_0^1 \\big(1 - F^*(y)\\big)\\, dy

    Args:
        loss_ratios: 逐年本金损失率 ``(Y,)``，取值 [0, 1]。
        lam: 市场风险价格 lambda。

    Returns:
        float: 风险调整期望损失率（即 Wang 口径的 spread）。
    """
    y = np.clip(np.asarray(loss_ratios, dtype=float), 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, 1001)
    # 经验 CDF：F(g) = P(Y <= g)
    cdf = np.searchsorted(np.sort(y), grid, side="right") / float(y.size)
    cdf = np.clip(cdf, 1.0e-12, 1.0 - 1.0e-12)
    surv_star = 1.0 - norm.cdf(norm.ppf(cdf) - lam)
    return float(np.trapezoid(surv_star, grid))


def implied_wang_lambda(
    loss_ratios: np.ndarray, target_spread: float
) -> float:
    """由目标利差反推 Wang 变换的隐含市场风险价格 lambda。

    Args:
        loss_ratios: 逐年本金损失率 ``(Y,)``。
        target_spread: 目标利差（小数），通常取 Lane 模型或市场报价。

    Returns:
        float: 隐含 lambda；若无法求解则返回 ``nan``。
    """
    def obj(lam: float) -> float:
        return wang_transform_price(loss_ratios, lam) - target_spread

    try:
        if obj(0.0) > 0.0:
            return 0.0
        if obj(3.0) < 0.0:
            return float("nan")
        return float(brentq(obj, 0.0, 3.0, xtol=1.0e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def price_cat_bond(
    annual_index: np.ndarray,
    attachment: float,
    exhaustion: float,
    trigger_type: str = "industry index",
    cfg: FinancialConfig = FINANCIAL,
) -> CatBondResult:
    """对巨灾债券进行 Lane 与 Wang 双方法定价。

    Args:
        annual_index: 年度触发指数序列 ``(Y,)``。
        attachment: 起赔点。
        exhaustion: 耗尽点。
        trigger_type: 触发结构描述。
        cfg: 金融配置。

    Returns:
        CatBondResult: 定价结果对象。
    """
    lr = tranche_loss_ratio(annual_index, attachment, exhaustion)
    el = float(lr.mean())
    pfl = float(np.mean(lr > 0.0))
    ptl = float(np.mean(lr >= 1.0 - 1.0e-12))
    cel = el / pfl if pfl > 0.0 else 0.0

    s_lane = lane_spread(pfl, cel, el, cfg)
    s_wang = wang_transform_price(lr, cfg.wang_lambda_market)
    lam_implied = implied_wang_lambda(lr, s_lane)

    return CatBondResult(
        trigger_type=trigger_type,
        attachment=float(attachment),
        exhaustion=float(exhaustion),
        principal=float(exhaustion - attachment),
        expected_loss=el,
        prob_first_loss=pfl,
        prob_total_loss=ptl,
        cond_expected_loss=cel,
        spread_lane=s_lane,
        spread_wang=s_wang,
        wang_lambda_used=cfg.wang_lambda_market,
        wang_lambda_implied=lam_implied,
        coupon_lane=cfg.risk_free_rate + s_lane,
        coupon_wang=cfg.risk_free_rate + s_wang,
        investor_expected_return_lane=cfg.risk_free_rate + s_lane - el,
        investor_expected_return_wang=cfg.risk_free_rate + s_wang - el,
        multiple_lane=s_lane / el if el > 0 else float("nan"),
        multiple_wang=s_wang / el if el > 0 else float("nan"),
        loss_ratios=lr,
    )


def location_box_weight(
    landfall_lat: np.ndarray, cfg: FinancialConfig = FINANCIAL
) -> np.ndarray:
    """cat-in-a-box 登陆位置权重。

    按登陆纬度落入的"箱体"赋予不同权重，反映沿海暴露的空间集中程度。
    这是把单一强度条件升级为**多重触发条件**的关键一步。

    Args:
        landfall_lat: 登陆纬度 ``(N,)`` (deg N)。
        cfg: 金融配置，含纬度箱体定义。

    Returns:
        np.ndarray: 位置权重 ``(N,)``，取值 (0, 1]。
    """
    lat = np.asarray(landfall_lat, dtype=float)
    w = np.full_like(lat, cfg.parametric_box_default)
    for lo, hi, weight in cfg.parametric_lat_boxes:
        w = np.where((lat >= lo) & (lat < hi), weight, w)
    return w


@dataclass
class LocationBoxDesign:
    """经验拟合的 cat-in-a-box 登陆位置权重设计。

    Attributes:
        edges: 纬度分箱边界 ``(B+1,)``。
        weights: 各箱体权重 ``(B,)``，已归一化到最大值 1。
        mean_loss: 各箱体的条件平均损失 ``(B,)`` (亿元)。
        counts: 各箱体的事件数 ``(B,)``。
    """

    edges: np.ndarray
    weights: np.ndarray
    mean_loss: np.ndarray
    counts: np.ndarray

    def apply(self, landfall_lat: np.ndarray) -> np.ndarray:
        """将设计好的箱体权重应用到给定登陆纬度。

        Args:
            landfall_lat: 登陆纬度 ``(N,)`` (deg N)。

        Returns:
            np.ndarray: 位置权重 ``(N,)``。
        """
        idx = np.clip(
            np.digitize(np.asarray(landfall_lat, dtype=float), self.edges) - 1,
            0, self.weights.size - 1,
        )
        return self.weights[idx]


def design_location_box(
    landfall_lat: np.ndarray,
    event_loss: np.ndarray,
    n_bins: int = 10,
    lat_range: Optional[Tuple[float, float]] = None,
) -> LocationBoxDesign:
    """由模型损失经验拟合 cat-in-a-box 的位置权重。

    实务中 cat-in-a-box 的箱体与档位并非拍脑袋设定，而是依据分保组合的
    暴露分布来设计。本函数按登陆纬度分箱，取各箱体的**条件平均损失**
    并归一化作为权重：

    .. math::
        w_b = \\frac{E[L \\mid \\phi \\in b]}{\\max_{b'} E[L \\mid \\phi \\in b']}

    Args:
        landfall_lat: 事件登陆纬度 ``(N,)`` (deg N)。
        event_loss: 事件损失 ``(N,)`` (亿元)。
        n_bins: 纬度分箱数。
        lat_range: 纬度范围 (min, max)；``None`` 时取数据范围。

    Returns:
        LocationBoxDesign: 拟合好的箱体设计。
    """
    lat = np.asarray(landfall_lat, dtype=float)
    loss = np.asarray(event_loss, dtype=float)
    lo, hi = lat_range if lat_range is not None else (lat.min(), lat.max())
    edges = np.linspace(lo, hi + 1.0e-9, n_bins + 1)
    idx = np.clip(np.digitize(lat, edges) - 1, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins).astype(float)
    sums = np.bincount(idx, weights=loss, minlength=n_bins)
    mean_loss = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
    peak = mean_loss.max()
    weights = mean_loss / peak if peak > 0 else np.zeros(n_bins)
    return LocationBoxDesign(edges=edges, weights=weights,
                             mean_loss=mean_loss, counts=counts)


def parametric_payout_ratio(
    landfall_pc: np.ndarray,
    cfg: FinancialConfig = FINANCIAL,
    landfall_lat: Optional[np.ndarray] = None,
    box_design: Optional[LocationBoxDesign] = None,
) -> np.ndarray:
    """参数触发赔付比例。

    单一条件结构（仅给出 ``landfall_pc``）：按登陆中心气压阶梯赋予赔付档位，
    登陆中心气压越低，赔付比例越高。

    多重条件 cat-in-a-box 结构（同时给出 ``landfall_lat``）：在气压阶梯的
    基础上乘以登陆位置权重，只有"强度足够 **且** 登陆在高暴露箱体内"才
    触发高额赔付，从而显著压缩基差风险：

    .. math::
        P_r = L(P_c) \\times W(\\phi_{landfall})

    Args:
        landfall_pc: 登陆时刻中心气压 ``(N,)`` (hPa)。
        cfg: 金融配置，含赔付阶梯与箱体定义。
        landfall_lat: 登陆纬度 ``(N,)`` (deg N)。为 ``None`` 时退化为
            仅按气压的单一条件结构。
        box_design: 经验拟合的箱体设计。给出时使用拟合权重，
            否则回退到 ``cfg.parametric_lat_boxes`` 的手工设定权重。

    Returns:
        np.ndarray: 赔付比例 ``(N,)``，取值 [0, 1]。
    """
    pc = np.asarray(landfall_pc, dtype=float)
    payout = np.zeros_like(pc)
    # 阶梯自低压到高压依次赋值，后面的档位不覆盖已赋更高档位
    for threshold, ratio in sorted(cfg.parametric_pc_ladder, key=lambda t: t[0]):
        payout = np.where((pc <= threshold) & (payout == 0.0), ratio, payout)
    if landfall_lat is not None:
        weight = (box_design.apply(landfall_lat) if box_design is not None
                  else location_box_weight(landfall_lat, cfg))
        payout = payout * weight
    return payout


def catbond_attachment_sensitivity(
    ylt: YearLossTable,
    attach_rps: Sequence[float],
    exhaust_multiple: float = 5.0,
    cfg: FinancialConfig = FINANCIAL,
) -> pd.DataFrame:
    """CAT bond 起赔点敏感性分析：attachment RP vs spread。

    Args:
        ylt: 年度损失表。
        attach_rps: 起赔点重现期序列 (年)。
        exhaust_multiple: 耗尽点重现期 = 起赔点重现期 x 该倍数。
        cfg: 金融配置。

    Returns:
        pandas.DataFrame: 列为 ``attach_rp / attachment / exhaustion /
        el / pfl / cel / spread_lane / spread_wang``。
    """
    rows: List[Dict[str, float]] = []
    for rp in attach_rps:
        a = float(pml_at_return_periods(ylt.occurrence, [rp])[0])
        e = float(pml_at_return_periods(
            ylt.occurrence, [min(rp * exhaust_multiple, ylt.n_years / 2.0)]
        )[0])
        if e <= a:
            e = a * 1.4
        res = price_cat_bond(ylt.occurrence, a, e, "industry index", cfg)
        rows.append({
            "attach_rp": float(rp),
            "attachment": a,
            "exhaustion": e,
            "el": res.expected_loss,
            "pfl": res.prob_first_loss,
            "cel": res.cond_expected_loss,
            "spread_lane": res.spread_lane,
            "spread_wang": res.spread_wang,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. 基差风险
# --------------------------------------------------------------------------- #


@dataclass
class BasisRiskResult:
    """基差风险量化结果。

    Attributes:
        correlation: 参数触发赔付与实际损失的 Pearson 相关系数。
        rank_correlation: Spearman 秩相关系数。
        hedge_effectiveness: 对冲效率 = 1 - Var(L - P) / Var(L)。
        optimal_notional: 使残差方差最小的最优名义本金 (亿元)。
        mean_shortfall: 平均对冲缺口 E[L - P] (亿元)。
        prob_shortfall: 赔付不足（L > P）的概率。
        prob_windfall: 超额赔付（P > L）的概率。
        actual_loss: 用于比较的实际损失序列 ``(E,)``。
        payout: 参数触发赔付序列 ``(E,)``。
    """

    correlation: float
    rank_correlation: float
    hedge_effectiveness: float
    optimal_notional: float
    mean_shortfall: float
    prob_shortfall: float
    prob_windfall: float
    actual_loss: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    payout: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


def analyse_basis_risk(
    actual_loss: np.ndarray, payout_ratio: np.ndarray
) -> BasisRiskResult:
    """量化参数触发结构的基差风险。

    最优名义本金由最小二乘给出：

    .. math::
        q^* = \\frac{\\mathrm{Cov}(L, P_r)}{\\mathrm{Var}(P_r)}

    对冲效率定义为方差削减比例：

    .. math::
        HE = 1 - \\frac{\\mathrm{Var}(L - q^* P_r)}{\\mathrm{Var}(L)}

    Args:
        actual_loss: 实际损失序列 ``(E,)`` (亿元)。
        payout_ratio: 参数触发赔付比例序列 ``(E,)``，取值 [0, 1]。

    Returns:
        BasisRiskResult: 基差风险结果对象。
    """
    l = np.asarray(actual_loss, dtype=float)
    p = np.asarray(payout_ratio, dtype=float)

    var_p = float(np.var(p))
    q = float(np.cov(l, p, ddof=1)[0, 1] / var_p) if var_p > 1.0e-12 else 0.0
    payout = q * p
    resid = l - payout

    var_l = float(np.var(l))
    he = 1.0 - float(np.var(resid)) / var_l if var_l > 1.0e-12 else 0.0

    if np.std(l) > 1e-12 and np.std(p) > 1e-12:
        corr = float(np.corrcoef(l, p)[0, 1])
        rl = pd.Series(l).rank().to_numpy()
        rp = pd.Series(p).rank().to_numpy()
        rank_corr = float(np.corrcoef(rl, rp)[0, 1])
    else:
        corr, rank_corr = 0.0, 0.0

    return BasisRiskResult(
        correlation=corr,
        rank_correlation=rank_corr,
        hedge_effectiveness=he,
        optimal_notional=q,
        mean_shortfall=float(np.mean(resid)),
        prob_shortfall=float(np.mean(resid > 0.0)),
        prob_windfall=float(np.mean(resid < 0.0)),
        actual_loss=l,
        payout=payout,
    )


# --------------------------------------------------------------------------- #
# 5. 投资组合视角
# --------------------------------------------------------------------------- #


@dataclass
class PortfolioResult:
    """投资组合分析结果。

    Attributes:
        asset_names: 资产名称列表。
        mu: 各资产年化预期收益 ``(A,)``。
        sigma: 各资产年化波动率 ``(A,)``。
        corr: 相关系数矩阵 ``(A, A)``。
        frontier_with: 含 CAT bond 的有效前沿 (vol, ret) ``(K, 2)``。
        frontier_without: 不含 CAT bond 的有效前沿 ``(K, 2)``。
        best_with: 含 CAT bond 的最大夏普组合 (vol, ret, sharpe, weights)。
        best_without: 不含 CAT bond 的最大夏普组合。
        sharpe_improvement: 夏普比率绝对提升。
    """

    asset_names: List[str]
    mu: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray
    frontier_with: np.ndarray
    frontier_without: np.ndarray
    best_with: Tuple[float, float, float, np.ndarray]
    best_without: Tuple[float, float, float, np.ndarray]
    sharpe_improvement: float


def _weight_grid(n_assets: int, step: float) -> np.ndarray:
    """生成单纯形上的权重网格（长期只做多，权重和为 1）。

    Args:
        n_assets: 资产数量（2 或 3）。
        step: 网格步长。

    Returns:
        np.ndarray: shape ``(K, n_assets)`` 的权重矩阵。
    """
    m = int(round(1.0 / step))
    if n_assets == 2:
        a = np.arange(m + 1, dtype=float)
        w = np.column_stack([a, m - a]) / m
        return w
    a, b = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
    a, b = a.ravel(), b.ravel()
    ok = (a + b) <= m
    a, b = a[ok], b[ok]
    c = m - a - b
    return np.column_stack([a, b, c]).astype(float) / m


def _frontier(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, n_bins: int = 60
) -> np.ndarray:
    """由权重网格提取有效前沿上沿（每个波动率分箱取最高收益）。

    Args:
        weights: 权重矩阵 ``(K, A)``。
        mu: 预期收益 ``(A,)``。
        cov: 协方差矩阵 ``(A, A)``。
        n_bins: 波动率分箱数。

    Returns:
        np.ndarray: shape ``(B, 2)`` 的 (波动率, 收益) 数组，按波动率升序。
    """
    ret = weights @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    bins = np.linspace(vol.min(), vol.max(), n_bins + 1)
    idx = np.clip(np.digitize(vol, bins) - 1, 0, n_bins - 1)
    out: List[Tuple[float, float]] = []
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        j = np.argmax(ret[sel])
        out.append((float(vol[sel][j]), float(ret[sel][j])))
    arr = np.array(out, dtype=float)
    return arr[np.argsort(arr[:, 0])] if arr.size else np.zeros((0, 2))


def _best_sharpe(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float
) -> Tuple[float, float, float, np.ndarray]:
    """在权重网格中寻找最大夏普比率组合。

    Args:
        weights: 权重矩阵 ``(K, A)``。
        mu: 预期收益 ``(A,)``。
        cov: 协方差矩阵 ``(A, A)``。
        rf: 无风险利率。

    Returns:
        Tuple[float, float, float, np.ndarray]: (波动率, 收益, 夏普比率, 权重)。
    """
    ret = weights @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    vol = np.maximum(vol, 1.0e-9)
    sharpe = (ret - rf) / vol
    j = int(np.argmax(sharpe))
    return float(vol[j]), float(ret[j]), float(sharpe[j]), weights[j].copy()


def analyse_portfolio(
    catbond_mu: float,
    catbond_sigma: float,
    cfg: FinancialConfig = FINANCIAL,
) -> PortfolioResult:
    """评估 CAT bond 加入传统股债组合后的分散化价值。

    CAT bond 的收益驱动因子（台风登陆强度）与宏观经济周期基本独立，
    因此与股票、债券的相关系数近似为零（"零贝塔"资产）。本函数
    在均值-方差框架下量化其对有效前沿与夏普比率的改善。

    Args:
        catbond_mu: CAT bond 年化预期收益（票息 - EL）。
        catbond_sigma: CAT bond 年化收益波动率。
        cfg: 金融配置。

    Returns:
        PortfolioResult: 组合分析结果。
    """
    names = ["Equity", "Bond", "CAT Bond"]
    mu = np.array([cfg.equity_mu, cfg.bond_mu, catbond_mu], dtype=float)
    sig = np.array([cfg.equity_sigma, cfg.bond_sigma,
                    max(catbond_sigma, 1.0e-4)], dtype=float)
    corr = np.array([
        [1.0, cfg.corr_equity_bond, cfg.corr_equity_cat],
        [cfg.corr_equity_bond, 1.0, cfg.corr_bond_cat],
        [cfg.corr_equity_cat, cfg.corr_bond_cat, 1.0],
    ], dtype=float)
    cov = corr * np.outer(sig, sig)

    w3 = _weight_grid(3, cfg.frontier_step)
    w2 = _weight_grid(2, cfg.frontier_step)
    cov2 = cov[:2, :2]
    mu2 = mu[:2]

    best_w = _best_sharpe(w3, mu, cov, cfg.risk_free_rate)
    best_wo = _best_sharpe(w2, mu2, cov2, cfg.risk_free_rate)

    return PortfolioResult(
        asset_names=names,
        mu=mu,
        sigma=sig,
        corr=corr,
        frontier_with=_frontier(w3, mu, cov),
        frontier_without=_frontier(w2, mu2, cov2),
        best_with=best_w,
        best_without=best_wo,
        sharpe_improvement=float(best_w[2] - best_wo[2]),
    )


__all__ = [
    "YearLossTable", "build_year_loss_table", "ep_curve",
    "pml_at_return_periods", "var_tvar", "RiskMetrics", "compute_risk_metrics",
    "LayerResult", "layer_annual_recovery", "price_layer",
    "build_reinsurance_program", "CatBondResult", "tranche_loss_ratio",
    "lane_spread", "wang_transform_price", "implied_wang_lambda",
    "price_cat_bond", "parametric_payout_ratio", "location_box_weight",
    "LocationBoxDesign", "design_location_box",
    "catbond_attachment_sensitivity", "BasisRiskResult", "analyse_basis_risk",
    "PortfolioResult", "analyse_portfolio",
]
