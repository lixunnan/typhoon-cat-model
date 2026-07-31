"""模块 A —— 危险性 (Hazard)。

包含三部分内容：

1. **利奇马 (Lekima, 1909) 历史 best-track 数据**：2019-08-04 生成于西北太平洋，
   08-10 01:45 (BJT) 以超强台风级（中心气压 930 hPa，近中心最大风力 16 级
   约 52 m/s）在浙江温岭城南镇登陆；随后北上穿越浙江、江苏，出海后于
   08-11 20:50 在山东青岛黄岛区二次登陆（9 级约 23 m/s）；08-13 在渤海湾消散。

2. **Holland (1980) 参数化梯度风场模型**，含移动非对称修正与
   Kaplan & DeMaria (1995) 登陆衰减。

3. **随机事件集 (Stochastic Event Set)**：蒙特卡洛生成 N 场合成台风，
   登陆强度服从截断对数正态分布，年频率服从泊松分布。

文献:
    Holland, G. J. (1980). An analytic model of the wind and pressure
        profiles in hurricanes. *Monthly Weather Review*, 108(8), 1212-1218.
    Kaplan, J., & DeMaria, M. (1995). A simple empirical model for predicting
        the decay of tropical cyclone winds after landfall.
        *Journal of Applied Meteorology*, 34(11), 2499-2512.
    Willoughby, H. E., & Rahn, M. E. (2004). Parametric representation of the
        primary hurricane vortex. Part I. *Monthly Weather Review*, 132, 3033-3048.
    Atkinson, G. D., & Holliday, C. R. (1977). Tropical cyclone minimum sea
        level pressure / maximum sustained wind relationship for the western
        North Pacific. *Monthly Weather Review*, 105, 421-427.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    AIR_DENSITY,
    AMBIENT_PRESSURE,
    COASTLINE,
    DEG2RAD,
    EARTH_RADIUS_KM,
    HAZARD,
    OMEGA_EARTH,
    STOCHASTIC,
    HazardConfig,
    StochasticConfig,
)

# --------------------------------------------------------------------------- #
# 1. 利奇马历史路径
# --------------------------------------------------------------------------- #

# (北京时间, 经度 E, 纬度 N, 中心气压 hPa, 近中心最大风速 m/s, 备注)
# 数据为公开报道与 CMA/JTWC best-track 量级的近似重构，仅供方法学演示。
_LEKIMA_RAW = [
    ("2019-08-04 08:00", 137.5, 14.2, 1000.0, 18.0, ""),
    ("2019-08-04 14:00", 136.6, 14.5,  998.0, 20.0, ""),
    ("2019-08-04 20:00", 135.7, 14.9,  996.0, 23.0, ""),
    ("2019-08-05 02:00", 134.8, 15.3,  994.0, 25.0, ""),
    ("2019-08-05 08:00", 133.9, 15.8,  990.0, 28.0, ""),
    ("2019-08-05 14:00", 133.0, 16.3,  985.0, 30.0, ""),
    ("2019-08-05 20:00", 132.1, 16.9,  980.0, 33.0, ""),
    ("2019-08-06 02:00", 131.2, 17.5,  975.0, 35.0, ""),
    ("2019-08-06 08:00", 130.3, 18.2,  968.0, 38.0, ""),
    ("2019-08-06 14:00", 129.5, 18.9,  960.0, 42.0, ""),
    ("2019-08-06 20:00", 128.7, 19.6,  950.0, 45.0, ""),
    ("2019-08-07 02:00", 127.9, 20.3,  940.0, 50.0, ""),
    ("2019-08-07 08:00", 127.1, 21.0,  930.0, 55.0, ""),
    ("2019-08-07 14:00", 126.4, 21.7,  925.0, 58.0, ""),
    ("2019-08-07 20:00", 125.8, 22.4,  920.0, 60.0, "peak intensity"),
    ("2019-08-08 02:00", 125.2, 23.1,  915.0, 62.0, "peak intensity"),
    ("2019-08-08 08:00", 124.6, 23.8,  915.0, 62.0, "peak intensity"),
    ("2019-08-08 14:00", 124.1, 24.6,  918.0, 60.0, ""),
    ("2019-08-08 20:00", 123.6, 25.4,  920.0, 58.0, ""),
    ("2019-08-09 02:00", 123.2, 26.1,  925.0, 55.0, ""),
    ("2019-08-09 08:00", 122.8, 26.8,  925.0, 55.0, ""),
    ("2019-08-09 14:00", 122.4, 27.3,  928.0, 53.0, ""),
    ("2019-08-09 20:00", 121.9, 27.9,  930.0, 52.0, ""),
    ("2019-08-10 01:45", 121.40, 28.40, 930.0, 52.0, "LANDFALL-1 Wenling, Zhejiang"),
    ("2019-08-10 08:00", 120.90, 29.20, 960.0, 40.0, ""),
    ("2019-08-10 14:00", 120.50, 30.00, 975.0, 33.0, ""),
    ("2019-08-10 20:00", 120.30, 30.80, 982.0, 30.0, ""),
    ("2019-08-11 02:00", 120.50, 31.90, 985.0, 28.0, ""),
    ("2019-08-11 08:00", 120.90, 33.00, 988.0, 25.0, "re-emerged over Yellow Sea"),
    ("2019-08-11 14:00", 121.00, 34.40, 990.0, 25.0, ""),
    ("2019-08-11 20:50", 120.00, 35.90, 992.0, 23.0, "LANDFALL-2 Huangdao, Qingdao"),
    ("2019-08-12 02:00", 119.90, 36.50,  996.0, 20.0, ""),
    ("2019-08-12 08:00", 119.60, 37.20,  998.0, 18.0, ""),
    ("2019-08-12 14:00", 119.30, 37.80, 1000.0, 15.0, ""),
    ("2019-08-12 20:00", 119.20, 38.20, 1002.0, 15.0, ""),
    ("2019-08-13 02:00", 119.20, 38.50, 1004.0, 12.0, ""),
    ("2019-08-13 08:00", 119.30, 38.60, 1006.0, 10.0, "dissipated over Bohai Bay"),
]


@dataclass
class Track:
    """台风路径容器。

    所有数组长度一致，均为 shape ``(T,)``。

    Attributes:
        lon: 中心经度 (deg E)。
        lat: 中心纬度 (deg N)。
        pc: 中心气压 (hPa)。
        vmax: 近中心最大 1 分钟持续风速 (m/s，10 m 高度)。
        time_hours: 相对起始时刻的小时数 (hr)。
        name: 路径名称。
        note: 每个路径点的备注。
    """

    lon: np.ndarray
    lat: np.ndarray
    pc: np.ndarray
    vmax: np.ndarray
    time_hours: np.ndarray
    name: str = "track"
    note: Optional[np.ndarray] = None

    @property
    def delta_p(self) -> np.ndarray:
        """中心气压差 Delta_p = Pn - Pc (hPa)，下限 1 hPa。"""
        return np.maximum(AMBIENT_PRESSURE - self.pc, 1.0)

    @property
    def n_points(self) -> int:
        """路径点数。"""
        return int(self.lon.size)

    def translation_speed(self) -> np.ndarray:
        """逐点移动速度 (m/s)，端点用邻近差分。

        Returns:
            np.ndarray: shape ``(T,)`` 的移速数组，单位 m/s。
        """
        dx = np.gradient(self.lon) * 111.32 * np.cos(self.lat * DEG2RAD)
        dy = np.gradient(self.lat) * 110.57
        dt = np.gradient(self.time_hours) * 3600.0
        dt = np.where(np.abs(dt) < 1e-6, 1.0, dt)
        return np.sqrt(dx ** 2 + dy ** 2) * 1000.0 / dt

    def heading_deg(self) -> np.ndarray:
        """逐点移动方向（气象方位角，0=正北，顺时针为正，单位 deg）。

        Returns:
            np.ndarray: shape ``(T,)`` 的方位角数组。
        """
        dx = np.gradient(self.lon) * 111.32 * np.cos(self.lat * DEG2RAD)
        dy = np.gradient(self.lat) * 110.57
        return np.degrees(np.arctan2(dx, dy))


def load_lekima_track() -> Track:
    """载入 2019 年台风利奇马 (Lekima, 1909) 的 best-track 重构数据。

    Returns:
        Track: 含 37 个路径点的路径对象，覆盖生成 (08-04) 到消散 (08-13)。

    Note:
        数据为基于公开报道与 best-track 量级的近似重构，非官方发布数据集，
        仅用于方法学演示。两次登陆点已固定为温岭 (121.40E, 28.40N) 与
        青岛黄岛 (120.00E, 35.90N)。
    """
    times = pd.to_datetime([r[0] for r in _LEKIMA_RAW])
    t0 = times[0]
    hours = np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)
    return Track(
        lon=np.array([r[1] for r in _LEKIMA_RAW], dtype=float),
        lat=np.array([r[2] for r in _LEKIMA_RAW], dtype=float),
        pc=np.array([r[3] for r in _LEKIMA_RAW], dtype=float),
        vmax=np.array([r[4] for r in _LEKIMA_RAW], dtype=float),
        time_hours=hours,
        name="Typhoon Lekima (1909)",
        note=np.array([r[5] for r in _LEKIMA_RAW], dtype=object),
    )


def lekima_landfall_indices() -> Tuple[int, int]:
    """返回利奇马两次登陆点在路径数组中的索引。

    Returns:
        Tuple[int, int]: (温岭登陆索引, 青岛黄岛登陆索引)。
    """
    notes = [r[5] for r in _LEKIMA_RAW]
    i1 = next(i for i, s in enumerate(notes) if s.startswith("LANDFALL-1"))
    i2 = next(i for i, s in enumerate(notes) if s.startswith("LANDFALL-2"))
    return i1, i2


# --------------------------------------------------------------------------- #
# 2. 地理工具与陆地判别
# --------------------------------------------------------------------------- #

_COAST_LAT = np.array([p[0] for p in COASTLINE], dtype=float)
_COAST_LON = np.array([p[1] for p in COASTLINE], dtype=float)


def coast_longitude(lat: np.ndarray) -> np.ndarray:
    """给定纬度，线性插值得到简化海岸线的经度。

    Args:
        lat: 纬度数组 (deg N)。

    Returns:
        np.ndarray: 对应的海岸线经度 (deg E)。
    """
    return np.interp(np.asarray(lat, dtype=float), _COAST_LAT, _COAST_LON)


def is_land(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """判断点是否位于陆地（简化：中国东部海岸线以西为陆地）。

    Args:
        lon: 经度数组 (deg E)。
        lat: 纬度数组 (deg N)。

    Returns:
        np.ndarray: 布尔数组，True 表示陆地。

    Note:
        该判别为一维岸线近似，忽略了山东半岛东突、舟山群岛等细节，
        对模型的宏观损失估计影响有限，属已声明的局限性之一。
    """
    return np.asarray(lon, dtype=float) < coast_longitude(lat)


def haversine_km(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """球面大圆距离 (km)，支持 numpy 广播。

    Args:
        lon1: 起点经度 (deg)。
        lat1: 起点纬度 (deg)。
        lon2: 终点经度 (deg)。
        lat2: 终点纬度 (deg)。

    Returns:
        np.ndarray: 距离 (km)。
    """
    p1 = np.asarray(lat1, dtype=float) * DEG2RAD
    p2 = np.asarray(lat2, dtype=float) * DEG2RAD
    dphi = p2 - p1
    dlam = (np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float)) * DEG2RAD
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """从点 1 指向点 2 的气象方位角 (0=北, 顺时针为正, deg)。

    Args:
        lon1: 起点经度 (deg)。
        lat1: 起点纬度 (deg)。
        lon2: 终点经度 (deg)。
        lat2: 终点纬度 (deg)。

    Returns:
        np.ndarray: 方位角 (deg)。
    """
    lat_mid = (np.asarray(lat1, dtype=float) + np.asarray(lat2, dtype=float)) / 2.0
    dx = (np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float)) * \
        111.32 * np.cos(lat_mid * DEG2RAD)
    dy = (np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float)) * 110.57
    return np.degrees(np.arctan2(dx, dy))


def coriolis_parameter(lat: np.ndarray) -> np.ndarray:
    """科氏参数 f = 2 * Omega * sin(phi) (1/s)。

    Args:
        lat: 纬度 (deg N)。

    Returns:
        np.ndarray: 科氏参数，绝对值下限 1e-6 以避免赤道奇异。
    """
    f = 2.0 * OMEGA_EARTH * np.sin(np.abs(np.asarray(lat, dtype=float)) * DEG2RAD)
    return np.maximum(f, 1.0e-6)


# --------------------------------------------------------------------------- #
# 3. Holland (1980) 风场
# --------------------------------------------------------------------------- #


def radius_max_wind_km(
    vmax_ms: np.ndarray, lat: np.ndarray, cfg: HazardConfig = HAZARD
) -> np.ndarray:
    """Willoughby & Rahn (2004) 最大风速半径经验式。

    公式:
        .. math::
            R_{mw} = 46.4 \\; \\exp(-0.0155\\,V_{max} + 0.0169\\,|\\phi|)

    其中 :math:`V_{max}` 为 10 m 高度 1 分钟持续风 (m/s)，:math:`\\phi` 为纬度 (deg)。

    Args:
        vmax_ms: 最大持续风速 (m/s)。
        lat: 纬度 (deg N)。
        cfg: 危险性配置。

    Returns:
        np.ndarray: 最大风速半径 (km)，裁剪到 [rmw_min_km, rmw_max_km]。
    """
    rmw = 46.4 * np.exp(-0.0155 * np.asarray(vmax_ms, dtype=float)
                        + 0.0169 * np.abs(np.asarray(lat, dtype=float)))
    return np.clip(rmw, cfg.rmw_min_km, cfg.rmw_max_km)


def holland_b_physical(
    vmax_gradient_ms: np.ndarray,
    delta_p_hpa: np.ndarray,
    cfg: HazardConfig = HAZARD,
) -> np.ndarray:
    """Holland (1980) 形状参数 B 的物理一致解。

    由 :math:`V_{max,g} = \\sqrt{B \\Delta p / (\\rho e)}` 反解:

    .. math::
        B = \\frac{\\rho\\, e\\, V_{max,g}^2}{\\Delta p}

    这样构造保证 Holland 廓线在 :math:`r = R_{mw}` 处的梯度风恰好等于给定
    :math:`V_{max,g}`，避免风廓线与观测强度不自洽。

    Args:
        vmax_gradient_ms: 梯度层最大风速 (m/s)。
        delta_p_hpa: 中心气压差 (hPa)。
        cfg: 危险性配置。

    Returns:
        np.ndarray: 形状参数 B，裁剪到 [holland_b_min, holland_b_max]。
    """
    dp_pa = np.maximum(np.asarray(delta_p_hpa, dtype=float), 1.0) * 100.0
    b = AIR_DENSITY * np.e * np.asarray(vmax_gradient_ms, dtype=float) ** 2 / dp_pa
    return np.clip(b, cfg.holland_b_min, cfg.holland_b_max)


def holland_b_vickery(
    vmax_ms: np.ndarray,
    rmw_km: np.ndarray,
    lat: np.ndarray,
    cfg: HazardConfig = HAZARD,
) -> np.ndarray:
    """Vickery 类经验式估计 Holland B（作为物理解的对照方案）。

    .. math::
        B = 1.0036 + 0.0173 V_{max} - 0.0313 \\ln R_{mw} + 0.0087 \\phi

    Args:
        vmax_ms: 最大持续风速 (m/s)。
        rmw_km: 最大风速半径 (km)。
        lat: 纬度 (deg N)。
        cfg: 危险性配置。

    Returns:
        np.ndarray: 形状参数 B，裁剪到配置边界。
    """
    b = (1.0036
         + 0.0173 * np.asarray(vmax_ms, dtype=float)
         - 0.0313 * np.log(np.maximum(np.asarray(rmw_km, dtype=float), 1.0))
         + 0.0087 * np.abs(np.asarray(lat, dtype=float)))
    return np.clip(b, cfg.holland_b_min, cfg.holland_b_max)


def holland_gradient_wind(
    r_km: np.ndarray,
    rmw_km: np.ndarray,
    delta_p_hpa: np.ndarray,
    b_shape: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Holland (1980) 梯度风廓线。

    .. math::
        V_g(r) = \\sqrt{\\frac{B}{\\rho}\\left(\\frac{R_{mw}}{r}\\right)^{B}
        \\Delta p \\; e^{-(R_{mw}/r)^{B}} + \\left(\\frac{r f}{2}\\right)^2}
        - \\frac{r f}{2}

    Args:
        r_km: 到台风中心的距离 (km)，可广播。
        rmw_km: 最大风速半径 (km)。
        delta_p_hpa: 中心气压差 (hPa)。
        b_shape: Holland 形状参数 B。
        lat: 台风中心纬度 (deg N)，用于科氏参数。

    Returns:
        np.ndarray: 梯度风速 (m/s)，非负。
    """
    r_m = np.maximum(np.asarray(r_km, dtype=float), 1.0) * 1000.0
    rmw_m = np.asarray(rmw_km, dtype=float) * 1000.0
    dp_pa = np.maximum(np.asarray(delta_p_hpa, dtype=float), 1.0) * 100.0
    f = coriolis_parameter(lat)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = np.power(rmw_m / r_m, b_shape)
        ratio = np.clip(ratio, 0.0, 700.0)          # 防止 exp 下溢/溢出
        core = (b_shape / AIR_DENSITY) * ratio * dp_pa * np.exp(-ratio)
        rf2 = r_m * f / 2.0
        vg = np.sqrt(np.maximum(core + rf2 ** 2, 0.0)) - rf2
    return np.nan_to_num(np.maximum(vg, 0.0), nan=0.0, posinf=0.0, neginf=0.0)


def kaplan_demaria_decay(
    v0_ms: np.ndarray, hours_since_landfall: np.ndarray, cfg: HazardConfig = HAZARD
) -> np.ndarray:
    """Kaplan & DeMaria (1995) 登陆后强度指数衰减。

    .. math::
        V(t) = V_b + (R\\,V_0 - V_b)\\, e^{-\\alpha t}

    Args:
        v0_ms: 登陆时刻最大风速 (m/s)。
        hours_since_landfall: 登陆后经过的小时数（负值表示未登陆）。
        cfg: 危险性配置。

    Returns:
        np.ndarray: 衰减后的最大风速 (m/s)。
    """
    t = np.maximum(np.asarray(hours_since_landfall, dtype=float), 0.0)
    v = cfg.decay_vb_ms + (cfg.decay_r * np.asarray(v0_ms, dtype=float)
                           - cfg.decay_vb_ms) * np.exp(-cfg.decay_alpha_per_hr * t)
    return np.maximum(v, 0.0)


def vmax_from_delta_p(delta_p_hpa: np.ndarray) -> np.ndarray:
    """Atkinson & Holliday (1977) 西北太平洋气压-风速关系。

    .. math::
        V_{max} = 3.4\\,(P_n - P_c)^{0.644}\\;[\\mathrm{m/s}]

    Args:
        delta_p_hpa: 中心气压差 (hPa)。

    Returns:
        np.ndarray: 10 m 高度 1 分钟持续最大风速 (m/s)。
    """
    return 3.4 * np.power(np.maximum(np.asarray(delta_p_hpa, dtype=float), 0.1), 0.644)


def wind_at_sites(
    track: Track,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    cfg: HazardConfig = HAZARD,
    return_gust: bool = True,
) -> np.ndarray:
    """计算单条路径在若干站点上的逐时刻地面风速。

    流程：Holland 梯度风 -> 表面折减 -> 移动非对称修正 -> (可选)阵风因子。

    Args:
        track: 台风路径。
        site_lon: 站点经度数组 shape ``(S,)``。
        site_lat: 站点纬度数组 shape ``(S,)``。
        cfg: 危险性配置。
        return_gust: True 返回 3 秒阵风；False 返回 1 分钟持续风。

    Returns:
        np.ndarray: shape ``(T, S)`` 的风速矩阵 (m/s)。
    """
    site_lon = np.asarray(site_lon, dtype=float).reshape(1, -1)
    site_lat = np.asarray(site_lat, dtype=float).reshape(1, -1)

    c_lon = track.lon.reshape(-1, 1)
    c_lat = track.lat.reshape(-1, 1)
    dp = track.delta_p.reshape(-1, 1)
    vmax_sfc = track.vmax.reshape(-1, 1)
    vmax_grad = vmax_sfc / cfg.surface_reduction_factor

    rmw = radius_max_wind_km(vmax_sfc, c_lat, cfg)
    b_shape = holland_b_physical(vmax_grad, dp, cfg)

    r = haversine_km(c_lon, c_lat, site_lon, site_lat)
    vg = holland_gradient_wind(r, rmw, dp, b_shape, c_lat)
    v_sfc = vg * cfg.surface_reduction_factor

    # 移动非对称修正：右半圆 (相对移动方向顺时针 90 度) 叠加 alpha * 移速
    v_trans = track.translation_speed().reshape(-1, 1)
    heading = track.heading_deg().reshape(-1, 1)
    brg = bearing_deg(c_lon, c_lat, site_lon, site_lat)
    delta = np.radians(brg - heading)
    asym = cfg.asymmetry_alpha * v_trans * np.sin(delta)
    v_sfc = np.maximum(v_sfc + asym * np.exp(-np.maximum(r - rmw, 0.0) / 250.0), 0.0)

    v_sfc = np.where(r > cfg.max_radius_km, 0.0, v_sfc)
    return v_sfc * cfg.gust_factor if return_gust else v_sfc


def max_wind_field_over_track(
    track: Track,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    cfg: HazardConfig = HAZARD,
    return_gust: bool = True,
) -> np.ndarray:
    """给定站点在整条路径过程中经历的最大风速。

    Args:
        track: 台风路径。
        site_lon: 站点经度数组 shape ``(S,)``。
        site_lat: 站点纬度数组 shape ``(S,)``。
        cfg: 危险性配置。
        return_gust: True 返回最大 3 秒阵风 (m/s)；False 返回最大 1 分钟持续风。

    Returns:
        np.ndarray: shape ``(S,)`` 的过程最大风速 (m/s)。
    """
    return wind_at_sites(track, site_lon, site_lat, cfg, return_gust).max(axis=0)


def min_distance_to_track(
    track: Track, site_lon: np.ndarray, site_lat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """站点到路径的最近距离及该处的中心气压差。

    Args:
        track: 台风路径。
        site_lon: 站点经度数组 shape ``(S,)``。
        site_lat: 站点纬度数组 shape ``(S,)``。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 最近距离 (km)，shape ``(S,)``
            - 最近路径点处的 Delta_p (hPa)，shape ``(S,)``
    """
    r = haversine_km(
        track.lon.reshape(-1, 1), track.lat.reshape(-1, 1),
        np.asarray(site_lon, dtype=float).reshape(1, -1),
        np.asarray(site_lat, dtype=float).reshape(1, -1),
    )
    idx = np.argmin(r, axis=0)
    return r[idx, np.arange(r.shape[1])], track.delta_p[idx]


def wind_field_grid(
    track: Track,
    lon_range: Tuple[float, float] = (117.0, 126.0),
    lat_range: Tuple[float, float] = (25.0, 34.0),
    n_grid: int = 140,
    cfg: HazardConfig = HAZARD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算规则网格上的过程最大阵风场（用于二维等值线绘图）。

    Args:
        track: 台风路径。
        lon_range: 经度范围 (min, max)。
        lat_range: 纬度范围 (min, max)。
        n_grid: 每个方向的网格数。
        cfg: 危险性配置。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (LON 网格 ``(n,n)``, LAT 网格 ``(n,n)``, 最大阵风 ``(n,n)``, 单位 m/s)。
    """
    lons = np.linspace(lon_range[0], lon_range[1], n_grid)
    lats = np.linspace(lat_range[0], lat_range[1], n_grid)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    vmax = max_wind_field_over_track(
        track, grid_lon.ravel(), grid_lat.ravel(), cfg, return_gust=True
    )
    return grid_lon, grid_lat, vmax.reshape(grid_lon.shape)


# --------------------------------------------------------------------------- #
# 4. 随机事件集
# --------------------------------------------------------------------------- #


@dataclass
class EventSet:
    """合成台风随机事件集（向量化存储）。

    数组维度约定：``N`` = 事件数，``T`` = 每事件路径点数。

    Attributes:
        lon: 路径经度 ``(N, T)``。
        lat: 路径纬度 ``(N, T)``。
        vmax: 逐点最大持续风速 ``(N, T)`` (m/s)。
        delta_p: 逐点中心气压差 ``(N, T)`` (hPa)。
        v_trans: 逐点移速 ``(N, T)`` (m/s)。
        heading: 逐点移动方位角 ``(N, T)`` (deg)。
        landfall_lon: 登陆点经度 ``(N,)``。
        landfall_lat: 登陆点纬度 ``(N,)``。
        landfall_dp: 登陆时刻 Delta_p ``(N,)`` (hPa)。
        landfall_pc: 登陆时刻中心气压 ``(N,)`` (hPa)。
        landfall_vmax: 登陆时刻最大风速 ``(N,)`` (m/s)。
        annual_rate: 每个事件的年发生率 ``(N,)``，总和 = lambda。
        freq_lambda: 泊松年频率参数。
    """

    lon: np.ndarray
    lat: np.ndarray
    vmax: np.ndarray
    delta_p: np.ndarray
    v_trans: np.ndarray
    heading: np.ndarray
    landfall_lon: np.ndarray
    landfall_lat: np.ndarray
    landfall_dp: np.ndarray
    landfall_pc: np.ndarray
    landfall_vmax: np.ndarray
    annual_rate: np.ndarray
    freq_lambda: float

    @property
    def n_events(self) -> int:
        """事件总数。"""
        return int(self.lon.shape[0])


def _sample_truncated_lognormal(
    rng: np.random.Generator,
    size: int,
    median: float,
    sigma_log: float,
    low: float,
    high: float,
) -> np.ndarray:
    """截断对数正态抽样（拒绝重抽，保证严格落在 [low, high]）。

    Args:
        rng: numpy 随机数发生器。
        size: 样本量。
        median: 对数正态中位数（即 exp(mu)）。
        sigma_log: 对数标准差。
        low: 截断下界。
        high: 截断上界。

    Returns:
        np.ndarray: shape ``(size,)`` 的样本。
    """
    out = np.empty(size, dtype=float)
    filled = 0
    mu = np.log(median)
    while filled < size:
        draw = rng.lognormal(mean=mu, sigma=sigma_log, size=max(size, 1024))
        ok = draw[(draw >= low) & (draw <= high)]
        take = min(ok.size, size - filled)
        out[filled:filled + take] = ok[:take]
        filled += take
    return out


def generate_event_set(
    cfg: StochasticConfig = STOCHASTIC, hazard_cfg: HazardConfig = HAZARD
) -> EventSet:
    """蒙特卡洛生成合成台风随机事件集。

    随机维度与分布假设（示例参数，来源为西北太平洋登陆华东台风的公开量级）：
        * 登陆强度 Delta_p ~ 截断对数正态(median=26 hPa, sigma_log=0.52,
          截断区间 [8, 105] hPa)。该设定下 Delta_p > 80 hPa（即中心气压
          <= 930 hPa，利奇马量级）的登陆概率约 1.5%~2%，对应约 15~20 年一遇。
        * 年频率 ~ Poisson(lambda = 3.2)。
        * 登陆点纬度 ~ Uniform(25.5N, 38.0N)，经度取该纬度的海岸线经度。
        * 移动方位角 ~ Normal(-18 deg, 24 deg)（西北西—北北西为主）。
        * 移速 ~ 对数正态(median=6.2 m/s)。
        * 登陆后强度按 Kaplan-DeMaria (1995) 指数衰减；登陆前保持登陆强度
          并向外海略强 (每 6 小时 +2%)。

    Args:
        cfg: 随机事件集配置。
        hazard_cfg: 危险性配置（用于衰减参数）。

    Returns:
        EventSet: 含 ``cfg.n_events`` 场合成台风的事件集。
    """
    rng = np.random.default_rng(cfg.random_seed)
    n = cfg.n_events
    t_before, t_after = cfg.n_steps_before, cfg.n_steps_after
    n_t = t_before + t_after + 1

    dp_lf = _sample_truncated_lognormal(
        rng, n, cfg.dp_median_hpa, cfg.dp_sigma_log, cfg.dp_min_hpa, cfg.dp_max_hpa
    )
    vmax_lf = vmax_from_delta_p(dp_lf)

    lf_lat = rng.uniform(cfg.landfall_lat_min, cfg.landfall_lat_max, size=n)
    lf_lon = coast_longitude(lf_lat)

    heading0 = rng.normal(cfg.heading_mean_deg, cfg.heading_std_deg, size=n)
    heading0 = np.clip(heading0, -70.0, 40.0)
    v_trans0 = rng.lognormal(
        np.log(cfg.translation_median_ms), cfg.translation_sigma_log, size=n
    )
    v_trans0 = np.clip(v_trans0, 2.0, 16.0)

    # 每步方向做小扰动，形成弯曲路径
    step_idx = np.arange(-t_before, t_after + 1, dtype=float).reshape(1, -1)
    jitter = rng.normal(0.0, 4.0, size=(n, n_t)).cumsum(axis=1) * 0.35
    heading = heading0.reshape(-1, 1) + jitter
    v_trans = np.repeat(v_trans0.reshape(-1, 1), n_t, axis=1)

    # 位移积分：以登陆点为原点，沿方位角前后推演
    step_km = v_trans * cfg.step_hours * 3600.0 / 1000.0
    dx = step_km * np.sin(np.radians(heading))
    dy = step_km * np.cos(np.radians(heading))
    lf_col = t_before
    cum_x = np.cumsum(dx, axis=1)
    cum_y = np.cumsum(dy, axis=1)
    rel_x = cum_x - cum_x[:, [lf_col]]
    rel_y = cum_y - cum_y[:, [lf_col]]

    lat = lf_lat.reshape(-1, 1) + rel_y / 110.57
    lon = lf_lon.reshape(-1, 1) + rel_x / (111.32 * np.cos(lf_lat.reshape(-1, 1) * DEG2RAD))

    # 强度演变
    hours_after = np.maximum(step_idx, 0.0) * cfg.step_hours
    vmax = np.where(
        step_idx < 0,
        vmax_lf.reshape(-1, 1) * np.power(1.02, -step_idx),
        kaplan_demaria_decay(vmax_lf.reshape(-1, 1), hours_after, hazard_cfg),
    )
    vmax = np.clip(vmax, 8.0, 78.0)
    # 由风速反推 Delta_p，保持与 Atkinson-Holliday 关系一致
    delta_p = np.clip(np.power(vmax / 3.4, 1.0 / 0.644), 1.0, 120.0)

    annual_rate = np.full(n, cfg.annual_frequency_lambda / n, dtype=float)

    return EventSet(
        lon=lon, lat=lat, vmax=vmax, delta_p=delta_p,
        v_trans=v_trans, heading=heading,
        landfall_lon=lf_lon, landfall_lat=lf_lat,
        landfall_dp=dp_lf, landfall_pc=AMBIENT_PRESSURE - dp_lf,
        landfall_vmax=vmax_lf,
        annual_rate=annual_rate, freq_lambda=cfg.annual_frequency_lambda,
    )


def event_set_max_gust(
    events: EventSet,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    cfg: HazardConfig = HAZARD,
    batch_size: int = STOCHASTIC.batch_size,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向量化计算全事件集在各站点的过程最大阵风、最近距离与近场强度。

    实现要点：使用 numpy 广播在 ``(B, T, S)`` 上一次性计算，分批控制内存，
    避免 Python 双层循环。B=批大小，T=路径点数，S=站点数。

    Args:
        events: 随机事件集。
        site_lon: 站点经度 shape ``(S,)``。
        site_lat: 站点纬度 shape ``(S,)``。
        cfg: 危险性配置。
        batch_size: 每批处理的事件数。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - 最大 3 秒阵风 ``(N, S)`` (m/s)
            - 站点到路径最近距离 ``(N, S)`` (km)
            - 最近点处 Delta_p ``(N, S)`` (hPa)
    """
    s_lon = np.asarray(site_lon, dtype=float).reshape(1, 1, -1)
    s_lat = np.asarray(site_lat, dtype=float).reshape(1, 1, -1)
    n, n_s = events.n_events, s_lon.shape[2]

    gust = np.zeros((n, n_s), dtype=float)
    dmin = np.zeros((n, n_s), dtype=float)
    dp_near = np.zeros((n, n_s), dtype=float)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        c_lon = events.lon[start:end, :, None]
        c_lat = events.lat[start:end, :, None]
        dp = events.delta_p[start:end, :, None]
        v_sfc_max = events.vmax[start:end, :, None]
        v_grad_max = v_sfc_max / cfg.surface_reduction_factor

        rmw = radius_max_wind_km(v_sfc_max, c_lat, cfg)
        b_shape = holland_b_physical(v_grad_max, dp, cfg)

        r = haversine_km(c_lon, c_lat, s_lon, s_lat)
        vg = holland_gradient_wind(r, rmw, dp, b_shape, c_lat)
        v = vg * cfg.surface_reduction_factor

        brg = bearing_deg(c_lon, c_lat, s_lon, s_lat)
        delta = np.radians(brg - events.heading[start:end, :, None])
        asym = cfg.asymmetry_alpha * events.v_trans[start:end, :, None] * np.sin(delta)
        v = np.maximum(v + asym * np.exp(-np.maximum(r - rmw, 0.0) / 250.0), 0.0)
        v = np.where(r > cfg.max_radius_km, 0.0, v)

        gust[start:end] = v.max(axis=1) * cfg.gust_factor
        k = np.argmin(r, axis=1)
        rows = np.arange(end - start)[:, None]
        cols = np.arange(n_s)[None, :]
        dmin[start:end] = r[rows, k, cols]
        dp_near[start:end] = dp[:, :, 0][rows, k]

    return gust, dmin, dp_near


__all__ = [
    "Track", "EventSet", "load_lekima_track", "lekima_landfall_indices",
    "coast_longitude", "is_land", "haversine_km", "bearing_deg",
    "coriolis_parameter", "radius_max_wind_km", "holland_b_physical",
    "holland_b_vickery", "holland_gradient_wind", "kaplan_demaria_decay",
    "vmax_from_delta_p", "wind_at_sites", "max_wind_field_over_track",
    "min_distance_to_track", "wind_field_grid", "generate_event_set",
    "event_set_max_gust",
]
