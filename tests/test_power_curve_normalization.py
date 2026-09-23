"""功率曲线规范化规则的回归测试。"""

import numpy as np
import pytest

from wind_farm_opt.core.turbine import Turbine, normalize_power_curve


def _curve(points):
    return np.asarray(points, dtype=np.float64)


# ---------- 基础规范化 ----------

def test_unsorted_points_are_sorted():
    curve = _curve([
        [12.0, 1000.0],
        [3.0, 0.0],
        [4.0, 100.0],
        [25.0, 1000.0],
    ])
    norm = normalize_power_curve(curve)
    np.testing.assert_array_equal(norm.points[:, 0], [3.0, 4.0, 12.0, 25.0])


def test_duplicate_wind_speed_same_power_is_deduplicated():
    curve = _curve([
        [3.0, 0.0],
        [4.0, 100.0],
        [4.0, 100.0],
        [12.0, 1000.0],
    ])
    norm = normalize_power_curve(curve)
    assert list(norm.points[:, 0]).count(4.0) == 1


def test_duplicate_wind_speed_conflicting_power_rejected():
    curve = _curve([
        [3.0, 0.0],
        [4.0, 100.0],
        [4.0, 200.0],
        [12.0, 1000.0],
    ])
    with pytest.raises(ValueError, match="歧义"):
        normalize_power_curve(curve)


def test_nan_rejected():
    curve = _curve([
        [3.0, 0.0],
        [4.0, np.nan],
        [12.0, 1000.0],
    ])
    with pytest.raises(ValueError, match="非有限"):
        normalize_power_curve(curve)


def test_inf_rejected():
    curve = _curve([
        [3.0, 0.0],
        [4.0, np.inf],
        [12.0, 1000.0],
    ])
    with pytest.raises(ValueError, match="非有限"):
        normalize_power_curve(curve)


def test_negative_power_rejected():
    curve = _curve([
        [3.0, 0.0],
        [3.5, -10.0],
        [12.0, 1000.0],
    ])
    with pytest.raises(ValueError, match="负功率"):
        normalize_power_curve(curve)


def test_non_increasing_wind_speed_rejected():
    # 去重后仍不递增（理论上不会发生，但显式保证严格递增）
    curve = _curve([
        [3.0, 0.0],
        [5.0, 200.0],
        [4.0, 100.0],
        [5.0, 300.0],
    ])
    with pytest.raises(ValueError):
        normalize_power_curve(curve)


def test_no_positive_power_rejected():
    curve = _curve([[0.0, 0.0], [25.0, 0.0]])
    with pytest.raises(ValueError, match="没有正功率"):
        normalize_power_curve(curve)


def test_zero_power_inside_generating_range_rejected():
    # 4 发电、10 突然归零、12 又发电 —— 切出点不唯一
    curve = _curve([
        [3.0, 0.0],
        [4.0, 100.0],
        [10.0, 0.0],
        [12.0, 1000.0],
        [25.0, 1000.0],
    ])
    with pytest.raises(ValueError, match="不连续"):
        normalize_power_curve(curve)


def test_bad_shape_rejected():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        normalize_power_curve(np.zeros((5, 3)))
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        normalize_power_curve(np.zeros(5))


def test_too_few_points_rejected():
    with pytest.raises(ValueError, match="至少需要 2 个"):
        normalize_power_curve(_curve([[4.0, 100.0]]))


# ---------- 派生参数定义 ----------

def test_derived_parameters_basic():
    # 3 为切入前零点，4 首次发电，25 最后正功率点，26 为停机采样
    curve = _curve([
        [0.0, 0.0],
        [3.0, 0.0],
        [4.0, 100.0],
        [10.0, 800.0],
        [12.0, 1000.0],
        [25.0, 1000.0],
        [26.0, 0.0],
        [30.0, 0.0],
    ])
    norm = normalize_power_curve(curve)
    assert norm.cut_in_speed == 3.0          # 转正边界（紧邻零点）
    assert norm.cut_out_speed == 25.0        # 最后正功率点，绝不是 30
    assert norm.rated_power == 1000.0
    assert norm.rated_speed == 12.0          # 首次达到额定


def test_rated_speed_is_first_reach_not_last():
    curve = _curve([
        [3.0, 0.0],
        [4.0, 100.0],
        [11.0, 999.0],
        [12.0, 1000.0],
        [25.0, 1000.0],
        [26.0, 0.0],
    ])
    norm = normalize_power_curve(curve)
    assert norm.rated_speed == 11.0


def test_cut_in_is_first_positive_when_no_leading_zero():
    curve = _curve([
        [4.0, 100.0],
        [12.0, 1000.0],
        [25.0, 1000.0],
    ])
    norm = normalize_power_curve(curve)
    assert norm.cut_in_speed == 4.0
    assert norm.cut_out_speed == 25.0


def test_turbine_stores_normalized_curve():
    curve = _curve([
        [25.0, 1000.0],
        [26.0, 0.0],
        [4.0, 100.0],
        [3.0, 0.0],
    ])
    t = Turbine("X", 80.0, 100.0, 0.8, curve)
    assert np.all(np.diff(t.power_curve[:, 0]) > 0)
