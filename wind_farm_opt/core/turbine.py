"""风机模型与功率曲线规范化。

功率曲线约定（标量与数组查询完全一致）
------------------------------------

规范化后的功率曲线满足：

1. 风速严格递增、有限且非负；功率有限且非负。
2. 乱序输入按风速重排；完全相同的重复点去重；同一风速对应不同
   功率的重复点无法消除歧义，直接拒绝。
3. 开头、结尾的零功率采样点只保留一个边界锚点（若存在），其余剥离。

派生参数定义：

- ``cut_in_speed``（切入风速）：正功率区之前最后一个零功率锚点的风速；
  若曲线无零锚点，则取第一个正功率点。风速 ``v < cut_in`` 或
  ``v == cut_in``（锚点处功率为 0）时功率为 0，锚点之后按曲线插值。
- ``rated_power``（额定功率）：曲线的最大功率。
- ``rated_speed``（首次达到额定）：第一个功率达到额定功率的采样点风速。
- 额定平台：从首次达到额定到最后一个有效发电点的连续区间，区间内
  功率必须保持额定；达到额定后又回落的曲线存在歧义，拒绝使用。
- ``cut_out_speed``（切出风速）：最后一个正功率（有效发电）采样点的
  风速。``v == cut_out`` 取该点功率（通常为额定），``v > cut_out``
  立即归零，不把切出后的零功率点误认为切出风速。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# 判定“达到额定功率/额定平台保持”的相对容差
RATED_RTOL = 1e-9
RATED_ATOL = 1e-9


def normalize_power_curve(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将原始功率曲线规范化为风速严格递增、可直接插值的形式。

    Parameters
    ----------
    points : np.ndarray
        形状 (N, 2) 的功率曲线，第一列风速 (m/s)，第二列功率 (kW)。
        允许乱序；允许完全重复的采样点。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (wind_speeds, powers)：排序去重后的风速与功率，已剥离首尾多余
        的零功率点，仅保留切入零功率锚点（若原曲线提供）。

    Raises
    ------
    ValueError
        曲线形状/点数非法、含 NaN 或 inf、风速为负、功率为负、
        同一风速对应多个不同功率（歧义重复点），或没有正功率点。
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("功率曲线必须是形状为 (N, 2) 的数组")
    if arr.shape[0] < 2:
        raise ValueError(f"功率曲线至少需要 2 个采样点，当前为 {arr.shape[0]} 个")
    if not np.all(np.isfinite(arr)):
        raise ValueError("功率曲线包含 NaN 或无穷大，必须全部为有限值")

    ws, pw = arr[:, 0], arr[:, 1]

    if np.any(ws < 0.0):
        raise ValueError("风速必须非负，功率曲线中存在负风速采样点")
    if np.any(pw < 0.0):
        raise ValueError("功率必须非负，功率曲线中存在负功率采样点")

    # 消除乱序：按风速稳定排序。
    order = np.argsort(ws, kind="stable")
    ws, pw = ws[order], pw[order]

    # 消除重复风速：功率一致则去重，不一致则歧义拒绝。
    speeds = [float(ws[0])]
    powers = [float(pw[0])]
    for i in range(1, len(ws)):
        if ws[i] == ws[i - 1]:
            if pw[i] != pw[i - 1]:
                raise ValueError(
                    f"风速 {ws[i]:g} m/s 对应多个不同功率 "
                    f"({pw[i - 1]:g} 与 {pw[i]:g} kW)，重复点存在歧义"
                )
            continue
        speeds.append(float(ws[i]))
        powers.append(float(pw[i]))

    ws_n = np.asarray(speeds, dtype=np.float64)
    pw_n = np.asarray(powers, dtype=np.float64)

    positive_idx = np.where(pw_n > 0.0)[0]
    if len(positive_idx) == 0:
        raise ValueError("功率曲线中没有正功率点，无法确定运行区间")

    first_pos = int(positive_idx[0])
    last_pos = int(positive_idx[-1])

    # 切入锚点：正功率区前的最后一个零功率点（若有）。
    start = first_pos
    if first_pos > 0 and pw_n[first_pos - 1] == 0.0:
        start = first_pos - 1

    return ws_n[start : last_pos + 1], pw_n[start : last_pos + 1]


def derive_curve_parameters(
    wind_speeds: np.ndarray, powers: np.ndarray
) -> Tuple[float, float, float, float]:
    """从规范化后的功率曲线派生运行参数。

    Returns
    -------
    tuple[float, float, float, float]
        (cut_in_speed, rated_speed, cut_out_speed, rated_power)

    Raises
    ------
    ValueError
        首次达到额定功率后出现功率回落，额定平台不连续。
    """
    rated_power = float(np.max(powers))
    first_rated = int(np.argmax(np.isclose(
        powers, rated_power, rtol=RATED_RTOL, atol=RATED_ATOL
    )))

    platform = powers[first_rated:]
    if not np.all(np.isclose(
        platform, rated_power, rtol=RATED_RTOL, atol=RATED_ATOL
    )):
        raise ValueError(
            "功率曲线在首次达到额定功率后出现回落，额定平台不连续，"
            "无法唯一确定额定运行区间"
        )

    cut_in_speed = float(wind_speeds[0])
    rated_speed = float(wind_speeds[first_rated])
    cut_out_speed = float(wind_speeds[-1])
    return cut_in_speed, rated_speed, cut_out_speed, rated_power


def interp_power_curve(
    wind_speed: float | np.ndarray,
    wind_speeds: np.ndarray,
    powers: np.ndarray,
) -> float | np.ndarray:
    """按统一边界约定在规范化功率曲线上插值。

    风速低于曲线起点（切入锚点）或高于曲线终点（切出点）时功率为 0；
    标量与数组输入遵循完全相同的约定。
    """
    ws = np.asarray(wind_speed, dtype=np.float64)
    result = np.interp(ws, wind_speeds, powers, left=0.0, right=0.0)
    return float(result) if ws.ndim == 0 else result


@dataclass
class Turbine:
    """风机参数类。

    Parameters
    ----------
    name : str
        风机名称/型号
    hub_height : float
        轮毂高度 (m)
    rotor_diameter : float
        转子直径 (m)
    thrust_coefficient : float
        推力系数 Ct (0-1)
    power_curve : np.ndarray
        功率曲线，形状为 (N, 2)，第一列为风速 (m/s)，第二列为功率 (kW)。
        构造时会被规范化（排序、去重、剥离首尾零功率点）。
    position : Optional[Tuple[float, float]]
        风机位置 (x, y) (m)，可选

    规范化后可通过同名属性读取切入/额定/切出风速与额定功率。
    """

    name: str
    hub_height: float
    rotor_diameter: float
    thrust_coefficient: float
    power_curve: np.ndarray
    position: Optional[Tuple[float, float]] = None

    cut_in_speed: float = field(init=False)
    rated_speed: float = field(init=False)
    cut_out_speed: float = field(init=False)
    rated_power: float = field(init=False)

    def __post_init__(self) -> None:
        ws, pw = normalize_power_curve(self.power_curve)
        self.power_curve = np.column_stack([ws, pw])

        self.cut_in_speed, self.rated_speed, self.cut_out_speed, self.rated_power = \
            derive_curve_parameters(ws, pw)

        if not (0.0 < self.thrust_coefficient <= 1.0):
            raise ValueError(f"推力系数必须在 (0, 1] 范围内，当前为 {self.thrust_coefficient}")

    @property
    def _curve_speeds(self) -> np.ndarray:
        return self.power_curve[:, 0]

    @property
    def _curve_powers(self) -> np.ndarray:
        return self.power_curve[:, 1]

    @property
    def rated_platform(self) -> Tuple[float, float]:
        """额定平台的风速区间 (m/s)：从首次达到额定到切出风速。"""
        return (self.rated_speed, self.cut_out_speed)

    def power(self, wind_speed: float | np.ndarray) -> float | np.ndarray:
        """根据风速计算功率。

        边界约定（标量与数组一致）：

        - ``v < cut_in_speed``：0；
        - 切入锚点（功率为 0 的起点）处为 0，其后线性插值；
        - 额定平台 ``rated_speed <= v <= cut_out_speed``：额定功率；
        - ``v == cut_out_speed``：该点功率（通常为额定）；
        - ``v > cut_out_speed``：0。

        Parameters
        ----------
        wind_speed : float | np.ndarray
            风速 (m/s)

        Returns
        -------
        float | np.ndarray
            功率 (kW)，输出形状与输入一致。
        """
        return interp_power_curve(
            wind_speed, self._curve_speeds, self._curve_powers
        )

    def summary(self) -> dict:
        """返回对外摘要：型号、额定功率与派生运行边界。"""
        return {
            "name": self.name,
            "rated_power_kW": self.rated_power,
            "rated_power_MW": self.rated_power / 1e3,
            "cut_in_speed": self.cut_in_speed,
            "rated_speed": self.rated_speed,
            "cut_out_speed": self.cut_out_speed,
            "rated_platform": [self.rated_speed, self.cut_out_speed],
            "hub_height": self.hub_height,
            "rotor_diameter": self.rotor_diameter,
        }

    @property
    def rotor_area(self) -> float:
        """风轮扫掠面积 (m^2)。"""
        return np.pi * (self.rotor_diameter / 2.0) ** 2


def _build_curve(
    cut_in: float,
    rated_speed: float,
    cut_out: float,
    rated_power: float,
    speed_max: float = 30.0,
) -> np.ndarray:
    """按物理边界构造整数网格功率曲线，并补齐边界节点。

    上升段采用 ``P = Pr * ((v - ci) / (vr - ci))^3``；切入节点功率为 0
    （作为切入锚点），额定节点至切出节点为额定平台，切出后补零功率点
    （规范化时会被剥离）。
    """
    speeds = set(np.arange(0.0, speed_max + 1.0, 1.0).tolist())
    speeds.update([float(cut_in), float(rated_speed), float(cut_out)])
    speeds = np.array(sorted(speeds), dtype=np.float64)

    powers = np.zeros_like(speeds)
    for i, v in enumerate(speeds):
        if v <= cut_in or v > cut_out:
            powers[i] = 0.0
        elif v < rated_speed:
            powers[i] = rated_power * ((v - cut_in) / (rated_speed - cut_in)) ** 3
        else:
            powers[i] = rated_power

    return np.column_stack([speeds, powers])


def create_default_turbine(model: str = "V164-9.5MW") -> Turbine:
    """创建默认风机模型。

    Parameters
    ----------
    model : str
        风机型号，可选 "V164-9.5MW" 或 "V126-3.45MW"

    Returns
    -------
    Turbine
        风机实例
    """
    if model == "V164-9.5MW":
        power_curve = _build_curve(
            cut_in=4.0, rated_speed=11.5, cut_out=25.0, rated_power=9500.0
        )
        return Turbine(
            name="V164-9.5MW",
            hub_height=105.0,
            rotor_diameter=164.0,
            thrust_coefficient=0.80,
            power_curve=power_curve,
        )

    elif model == "V126-3.45MW":
        power_curve = _build_curve(
            cut_in=3.5, rated_speed=12.0, cut_out=25.0, rated_power=3450.0
        )
        return Turbine(
            name="V126-3.45MW",
            hub_height=87.0,
            rotor_diameter=126.0,
            thrust_coefficient=0.82,
            power_curve=power_curve,
        )

    else:
        raise ValueError(f"未知的风机型号: {model}")
