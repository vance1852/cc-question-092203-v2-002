"""风机模型定义。

功率曲线规范化与参数推导约定
----------------------------

输入功率曲线为形状 ``(N, 2)`` 的数组（风速 m/s, 功率 kW），允许包含
切入前与切出后的零功率采样点。构造时按以下规则规范化：

1. 所有值必须为有限值（拒绝 ``NaN``/``Inf``）。
2. 按风速升序排序。
3. 重复风速：对应功率完全相同则去重；功率不同则无法消除歧义，直接拒绝。
4. 规范化后风速必须严格递增；功率不允许出现负值。
5. 必须至少存在一个正功率点，且正功率点在风速轴上构成连续区间
   （发电区间内部出现零功率将使切出点产生歧义，拒绝）。

派生参数定义：

- ``cut_in_speed``  切入风速：首个正功率采样点之前紧邻的零功率采样点
  （即功率由零转正的边界）；若曲线第一个点即为正功率，则取该点。
- ``rated_power``   额定功率：发电区间内功率的最大值。
- ``rated_speed``   首次达到额定的风速：第一个功率不小于
  ``rated_power * (1 - RATED_TOL)`` 的采样点，此后进入额定平台。
- ``cut_out_speed`` 切出风速：最后一个正功率采样点（最后有效发电区间
  的末端）；其后的零功率点只是停机采样，不参与边界判定。

查询约定（标量与数组完全一致）：

- ``v < cut_in`` 或 ``v > cut_out``：功率为 0；
- ``cut_in <= v <= cut_out``：在规范化曲线采样点之间线性插值；
- 在切出点 ``v == cut_out`` 返回该点功率（额定平台功率），
  绝不与切出后的零功率采样点插值。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

#: 判定“达到额定”的相对容差
RATED_TOL = 1e-3


@dataclass
class _NormalizedCurve:
    """规范化后的功率曲线及其派生参数。"""

    points: np.ndarray  # shape (M, 2)，风速严格递增
    cut_in_speed: float
    rated_speed: float
    cut_out_speed: float
    rated_power: float
    cut_in_index: int  # 切入边界点在 points 中的索引（可能为零功率点）
    cut_out_index: int  # 切出点（最后正功率点）在 points 中的索引


def normalize_power_curve(power_curve: np.ndarray) -> _NormalizedCurve:
    """规范化功率曲线并推导运行边界参数。

    Parameters
    ----------
    power_curve : np.ndarray
        形状为 (N, 2) 的数组，第一列为风速 (m/s)，第二列为功率 (kW)。

    Returns
    -------
    _NormalizedCurve
        规范化结果与切入/额定/切出参数。

    Raises
    ------
    ValueError
        曲线形状非法、含非有限值或负功率、重复风速对应功率不一致、
        没有正功率点，或发电区间不连续。
    """
    raw = np.asarray(power_curve, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("功率曲线必须是形状为 (N, 2) 的数组")
    if raw.shape[0] < 2:
        raise ValueError("功率曲线至少需要 2 个采样点")

    if not np.all(np.isfinite(raw)):
        raise ValueError("功率曲线包含 NaN 或 Inf 等非有限值，请清理后重试")

    # 1. 按风速升序排序（乱序输入对用户透明）。
    order = np.argsort(raw[:, 0], kind="mergesort")
    ws_all = raw[order, 0]
    pw_all = raw[order, 1]

    # 2. 处理重复风速：功率一致则去重，否则歧义无法消除。
    keep = np.ones(len(ws_all), dtype=bool)
    for i in range(1, len(ws_all)):
        if ws_all[i] == ws_all[i - 1]:
            if pw_all[i] != pw_all[i - 1]:
                raise ValueError(
                    f"风速 {ws_all[i]:g} m/s 对应多个不同功率 "
                    f"({pw_all[i - 1]:g} 与 {pw_all[i]:g} kW)，存在歧义，"
                    "请消除重复点后重试"
                )
            keep[i] = False
    ws = ws_all[keep]
    pw = pw_all[keep]

    # 3. 严格递增（去重后仍不递增说明数据本身异常）。
    if np.any(np.diff(ws) <= 0.0):
        raise ValueError("功率曲线的风速列必须严格递增")

    # 4. 拒绝负功率。
    if np.any(pw < 0.0):
        raise ValueError("功率曲线不允许出现负功率")

    # 5. 正功率点必须存在且构成连续区间。
    positive_idx = np.where(pw > 0.0)[0]
    if len(positive_idx) == 0:
        raise ValueError("功率曲线中没有正功率点，无法推导运行边界")

    first_pos = int(positive_idx[0])
    last_pos = int(positive_idx[-1])
    if np.any(pw[first_pos:last_pos + 1] <= 0.0):
        raise ValueError(
            "发电区间内部出现零功率点，有效发电区间不连续，"
            "无法唯一确定切出风速"
        )

    # --- 派生参数 ---
    rated_power = float(np.max(pw[first_pos:last_pos + 1]))

    # 切入：首个正功率点前紧邻的零功率点；没有则取首个正功率点。
    cut_in_index = first_pos
    if first_pos > 0 and pw[first_pos - 1] == 0.0:
        cut_in_index = first_pos - 1

    # 首次达到额定（额定平台起点）。
    rated_threshold = rated_power * (1.0 - RATED_TOL)
    reached = np.where(pw[first_pos:last_pos + 1] >= rated_threshold)[0]
    rated_index = first_pos + int(reached[0])

    return _NormalizedCurve(
        points=np.column_stack([ws, pw]),
        cut_in_speed=float(ws[cut_in_index]),
        rated_speed=float(ws[rated_index]),
        cut_out_speed=float(ws[last_pos]),
        rated_power=rated_power,
        cut_in_index=cut_in_index,
        cut_out_index=last_pos,
    )


def _interpolate_power(
    wind_speed: float | np.ndarray,
    norm: _NormalizedCurve,
) -> float | np.ndarray:
    """按统一边界约定查询功率（标量与数组同规则）。"""
    ws_points = norm.points[:, 0]
    pw_points = norm.points[:, 1]

    v = np.asarray(wind_speed, dtype=np.float64)
    # 仅在 [切入, 切出] 闭区间内插值；区间外（含 NaN）一律为 0。
    # 使用完整规范化曲线插值但用掩码截断，保证 v == cut_out 时
    # 取到切出点本身的功率，而不会与切出后的零功率点发生插值。
    in_band = (v >= norm.cut_in_speed) & (v <= norm.cut_out_speed)
    interpolated = np.interp(v, ws_points, pw_points, left=0.0, right=0.0)
    result = np.where(in_band, interpolated, 0.0)

    if v.ndim == 0:
        return float(result)
    return result


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
        允许乱序、含重复（同功率）点或切入/切出外的零功率点，构造时
        自动规范化；歧义数据将抛出 ValueError。
    position : Optional[Tuple[float, float]]
        风机位置 (x, y) (m)，可选

    派生属性（init=False）
    ---------------------
    cut_in_speed : float
        切入风速 (m/s)
    rated_speed : float
        首次达到额定的风速 (m/s)
    cut_out_speed : float
        切出风速 (m/s)，取最后一个正功率采样点
    rated_power : float
        额定功率 (kW)
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
        self._normalized = normalize_power_curve(self.power_curve)
        # 对外暴露规范化后的曲线（排序、去重、含切入前/切出后的零点）。
        self.power_curve = self._normalized.points.copy()

        self.cut_in_speed = self._normalized.cut_in_speed
        self.rated_speed = self._normalized.rated_speed
        self.cut_out_speed = self._normalized.cut_out_speed
        self.rated_power = self._normalized.rated_power

        if not (0.0 < self.thrust_coefficient <= 1.0):
            raise ValueError(f"推力系数必须在 (0, 1] 范围内，当前为 {self.thrust_coefficient}")

    def power(self, wind_speed: float | np.ndarray) -> float | np.ndarray:
        """根据风速计算功率。

        标量与数组遵循同一边界约定：

        - ``v < cut_in_speed`` 或 ``v > cut_out_speed``：返回 0；
        - ``cut_in_speed <= v <= cut_out_speed``：线性插值；
        - ``v == cut_out_speed``：返回切出点功率（额定值），
          不与切出后的零功率采样点插值。

        Parameters
        ----------
        wind_speed : float | np.ndarray
            风速 (m/s)

        Returns
        -------
        float | np.ndarray
            功率 (kW)，形状与入参一致
        """
        return _interpolate_power(wind_speed, self._normalized)

    @property
    def rotor_area(self) -> float:
        """风轮扫掠面积 (m^2)。"""
        return np.pi * (self.rotor_diameter / 2.0) ** 2

    def spec_summary(self) -> dict:
        """返回与派生参数相互吻合的对外机型摘要。"""
        return {
            "name": self.name,
            "rated_power_kw": self.rated_power,
            "rated_power_mw": self.rated_power / 1e3,
            "cut_in_speed_mps": self.cut_in_speed,
            "rated_speed_mps": self.rated_speed,
            "cut_out_speed_mps": self.cut_out_speed,
            "hub_height_m": float(self.hub_height),
            "rotor_diameter_m": float(self.rotor_diameter),
            "thrust_coefficient": float(self.thrust_coefficient),
        }


def _build_curve(
    cut_in: float,
    rated_speed: float,
    cut_out: float,
    rated_power: float,
    speed_max: float,
) -> np.ndarray:
    """按设计边界构造功率曲线。

    整数风速网格上额外补入切入、额定、切出三个关键节点，保证派生
    参数与设计值一致；切出后保留零功率采样点（用于回归切出边界）。
    """
    grid = np.arange(0.0, speed_max + 1.0, 1.0)
    speeds = np.unique(np.concatenate([grid, [cut_in, rated_speed, cut_out]]))
    powers = np.zeros_like(speeds)

    for i, ws in enumerate(speeds):
        if ws < cut_in or ws > cut_out:
            powers[i] = 0.0
        elif ws <= rated_speed:
            powers[i] = rated_power * ((ws - cut_in) / (rated_speed - cut_in)) ** 3
        else:
            powers[i] = rated_power

    # 关键节点处强制精确取值，消除浮点误差。
    powers[np.searchsorted(speeds, rated_speed)] = rated_power
    powers[speeds > cut_out] = 0.0
    powers[speeds < cut_in] = 0.0

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
            cut_in=4.0, rated_speed=11.5, cut_out=25.0,
            rated_power=9500.0, speed_max=30.0,
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
            cut_in=3.5, rated_speed=12.0, cut_out=25.0,
            rated_power=3450.0, speed_max=25.0,
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
