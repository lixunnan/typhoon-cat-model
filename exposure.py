"""模块 B —— 承灾体暴露 (Exposure)。

内置华东沿海（浙江 / 上海 / 江苏 / 山东）14 个重点地市的暴露数据库，
字段包括经纬度、GDP、可保财产总额、财产险渗透率与建筑类型占比。

.. warning::
    **本模块所有数值均为公开量级的近似示例数据，非官方统计口径，
    仅供巨灾模型方法学演示，不得用于任何实际定价、监管或投资决策。**

主要近似来源与构造方式：
    * GDP：2019 年各市地区生产总值的公开量级（亿元）。台州市已扣除温岭市，
      避免与单列的温岭市重复计算。
    * 可保财产总额 = GDP x 资本产出系数(3.2) x 风灾可暴露比例(0.45)。
      资本产出系数取中国资本存量/GDP 的经验区间 3.0~3.5；风灾可暴露比例
      表示资本存量中建筑物、构筑物、设备与存货等可被风灾直接损毁的份额。
    * 财产险渗透率：中国财产险深度低，商业与工业财产投保率高于居民，
      各省取 0.5%~2.0% 区间内的差异化示例值。
    * 建筑类型占比：按各市产业结构（工业/农业/城镇化率）定性构造。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import VULNERABILITY, VulnerabilityConfig

BUILDING_TYPES: Tuple[str, ...] = ("concrete", "masonry", "light_steel", "greenhouse")

BUILDING_TYPE_LABELS: Dict[str, str] = {
    "concrete": "Reinforced Concrete",
    "masonry": "Masonry / Brick-Concrete",
    "light_steel": "Light Steel / Industrial Shed",
    "greenhouse": "Agricultural Greenhouse",
}

# (城市英文名, 中文名, 省份, 经度E, 纬度N, GDP亿元, 财产险渗透率,
#  混凝土, 砖混, 轻钢, 农业大棚)
_EXPOSURE_RAW: List[Tuple[str, str, str, float, float, float, float,
                          float, float, float, float]] = [
    ("Wenling",     "温岭",   "Zhejiang", 121.36, 28.37,  1085.0, 0.0130,
     0.34, 0.39, 0.20, 0.07),
    ("Taizhou-ZJ",  "台州",   "Zhejiang", 121.43, 28.66,  4049.0, 0.0135,
     0.40, 0.35, 0.19, 0.06),
    ("Wenzhou",     "温州",   "Zhejiang", 120.70, 27.99,  6606.0, 0.0140,
     0.44, 0.34, 0.17, 0.05),
    ("Ningbo",      "宁波",   "Zhejiang", 121.55, 29.87, 11985.0, 0.0160,
     0.52, 0.26, 0.19, 0.03),
    ("Hangzhou",    "杭州",   "Zhejiang", 120.16, 30.27, 15373.0, 0.0170,
     0.61, 0.24, 0.13, 0.02),
    ("Zhoushan",    "舟山",   "Zhejiang", 122.11, 30.02,  1372.0, 0.0125,
     0.42, 0.33, 0.22, 0.03),
    ("Shanghai",    "上海",   "Shanghai", 121.47, 31.23, 38155.0, 0.0195,
     0.68, 0.18, 0.12, 0.02),
    ("Suzhou",      "苏州",   "Jiangsu",  120.62, 31.30, 19236.0, 0.0165,
     0.57, 0.21, 0.19, 0.03),
    ("Nantong",     "南通",   "Jiangsu",  120.86, 32.01,  9383.0, 0.0130,
     0.47, 0.28, 0.20, 0.05),
    ("Yancheng",    "盐城",   "Jiangsu",  120.16, 33.35,  5703.0, 0.0105,
     0.38, 0.29, 0.21, 0.12),
    ("Lianyungang", "连云港", "Jiangsu",  119.22, 34.60,  3139.0, 0.0095,
     0.36, 0.30, 0.22, 0.12),
    ("Qingdao",     "青岛",   "Shandong", 120.38, 36.07, 11741.0, 0.0140,
     0.53, 0.24, 0.19, 0.04),
    ("Weifang",     "潍坊",   "Shandong", 119.16, 36.71,  5688.0, 0.0085,
     0.30, 0.27, 0.21, 0.22),
    ("Yantai",      "烟台",   "Shandong", 121.45, 37.46,  7653.0, 0.0100,
     0.44, 0.27, 0.20, 0.09),
]

# 各省暴雨内涝易损系数（相对值，1.0 为基准）。
# 利奇马在浙江（临海城区被淹）与山东（潍坊寿光设施农业内涝）造成的
# 洪涝损失占比显著高于江苏与上海。
PROVINCE_RAIN_SUSCEPTIBILITY: Dict[str, float] = {
    "Zhejiang": 1.20,
    "Shanghai": 0.80,
    "Jiangsu": 0.90,
    "Shandong": 1.30,
}


@dataclass
class ExposureDatabase:
    """暴露数据库封装。

    Attributes:
        table: 承载全部字段的 ``pandas.DataFrame``。
        config: 脆弱性/暴露换算配置。
    """

    table: pd.DataFrame
    config: VulnerabilityConfig = VULNERABILITY

    @property
    def n_cities(self) -> int:
        """城市数量。"""
        return int(len(self.table))

    @property
    def lon(self) -> np.ndarray:
        """城市经度数组 ``(C,)``。"""
        return self.table["lon"].to_numpy(dtype=float)

    @property
    def lat(self) -> np.ndarray:
        """城市纬度数组 ``(C,)``。"""
        return self.table["lat"].to_numpy(dtype=float)

    @property
    def exposed_value(self) -> np.ndarray:
        """风灾可暴露财产价值数组 ``(C,)``，单位亿元。"""
        return self.table["exposed_value"].to_numpy(dtype=float)

    @property
    def penetration(self) -> np.ndarray:
        """财产险渗透率数组 ``(C,)``。"""
        return self.table["penetration"].to_numpy(dtype=float)

    @property
    def rain_susceptibility(self) -> np.ndarray:
        """省级暴雨内涝易损系数数组 ``(C,)``。"""
        return self.table["rain_susceptibility"].to_numpy(dtype=float)

    @property
    def building_shares(self) -> np.ndarray:
        """建筑类型占比矩阵 ``(C, 4)``，列顺序同 ``BUILDING_TYPES``。"""
        return self.table[list(BUILDING_TYPES)].to_numpy(dtype=float)

    @property
    def names(self) -> List[str]:
        """城市英文名列表。"""
        return self.table["city"].tolist()

    def province_mask(self, province: str) -> np.ndarray:
        """返回指定省份的布尔掩码。

        Args:
            province: 省份英文名，如 ``"Zhejiang"``。

        Returns:
            np.ndarray: shape ``(C,)`` 的布尔数组。
        """
        return (self.table["province"] == province).to_numpy()

    def summary(self) -> pd.DataFrame:
        """按省份汇总暴露规模。

        Returns:
            pandas.DataFrame: 含 GDP、可暴露价值、可保价值的省级汇总表。
        """
        grp = self.table.groupby("province", as_index=False).agg(
            gdp_bn=("gdp", "sum"),
            exposed_value=("exposed_value", "sum"),
            insured_value=("insured_value", "sum"),
            n_cities=("city", "count"),
        )
        return grp.sort_values("exposed_value", ascending=False).reset_index(drop=True)


def load_exposure(config: VulnerabilityConfig = VULNERABILITY) -> ExposureDatabase:
    """构建华东沿海重点地市暴露数据库。

    换算关系：
        .. math::
            V_{exposed} = GDP \\times \\kappa_{K/Y} \\times \\theta_{wind}

            V_{insured} = V_{exposed} \\times \\pi_{penetration}

    其中 :math:`\\kappa_{K/Y}` 为资本产出系数，:math:`\\theta_{wind}` 为
    风灾可暴露比例，:math:`\\pi` 为财产险渗透率。

    Args:
        config: 脆弱性/暴露换算配置。

    Returns:
        ExposureDatabase: 含 14 个城市的暴露数据库。

    Note:
        示例数据，非官方统计，仅供方法学演示。
    """
    cols = ["city", "city_cn", "province", "lon", "lat", "gdp", "penetration",
            *BUILDING_TYPES]
    df = pd.DataFrame(_EXPOSURE_RAW, columns=cols)

    # 建筑占比归一化，防止手工输入的舍入误差
    shares = df[list(BUILDING_TYPES)].to_numpy(dtype=float)
    shares = shares / shares.sum(axis=1, keepdims=True)
    for i, b in enumerate(BUILDING_TYPES):
        df[b] = shares[:, i]

    df["capital_stock"] = df["gdp"] * config.capital_output_ratio
    df["exposed_value"] = df["capital_stock"] * config.wind_exposed_fraction
    df["insured_value"] = df["exposed_value"] * df["penetration"]
    df["rain_susceptibility"] = df["province"].map(PROVINCE_RAIN_SUSCEPTIBILITY)
    df["rain_susceptibility"] = df["rain_susceptibility"].fillna(1.0)
    return ExposureDatabase(table=df, config=config)


__all__ = [
    "BUILDING_TYPES", "BUILDING_TYPE_LABELS", "PROVINCE_RAIN_SUSCEPTIBILITY",
    "ExposureDatabase", "load_exposure",
]
