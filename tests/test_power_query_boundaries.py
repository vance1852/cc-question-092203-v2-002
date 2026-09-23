"""功率查询边界约定的回归测试（标量与数组一致）。"""

import numpy as np
import pytest

from wind_farm_opt.core.turbine import Turbine


@pytest.fixture
def turbine():
    # 切入边界 3（零点）/4（首正），平台 12~25，切出后 26、30 为停机点
    curve = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [4.0, 100.0],
        [8.0, 500.0],
        [12.0, 1000.0],
        [25.0, 1000.0],
        [26.0, 0.0],
        [30.0, 0.0],
    ])
    return Turbine("X", 80.0, 100.0, 0.8, curve)


def test_before_cut_in_scalar(turbine):
    assert turbine.power(0.0) == 0.0
    assert turbine.power(2.9) == 0.0


def test_before_cut_in_array(turbine):
    out = turbine.power(np.array([-1.0, 0.0, 2.9]))
    np.testing.assert_array_equal(out, np.zeros(3))


def test_cut_in_region_interpolates(turbine):
    # 3(0) 与 4(100) 之间线性
    assert turbine.power(3.5) == pytest.approx(50.0)
    arr = turbine.power(np.array([3.0, 3.5, 4.0]))
    np.testing.assert_allclose(arr, [0.0, 50.0, 100.0])


def test_plateau_scalar_and_array(turbine):
    assert turbine.power(15.0) == 1000.0
    assert turbine.power(20.0) == 1000.0
    arr = turbine.power(np.array([12.0, 18.0, 25.0]))
    np.testing.assert_array_equal(arr, np.full(3, 1000.0))


def test_cut_out_point_is_plateau_power(turbine):
    # 关键回归：切出点必须是额定功率，不能与 26 m/s 的零点插值
    assert turbine.power(25.0) == 1000.0
    arr = turbine.power(np.array([24.99, 25.0, 25.01]))
    # 25.01 已在切出后，必须立即归零（阶跃停机约定）
    np.testing.assert_allclose(arr, [1000.0, 1000.0, 0.0])


def test_after_cut_out_is_zero(turbine):
    assert turbine.power(26.0) == 0.0
    assert turbine.power(30.0) == 0.0
    assert turbine.power(50.0) == 0.0
    arr = turbine.power(np.array([25.5, 26.0, 40.0]))
    np.testing.assert_array_equal(arr, np.zeros(3))


def test_scalar_and_array_share_convention(turbine):
    speeds = np.array([0.0, 3.0, 3.5, 4.0, 12.0, 25.0, 25.0001, 26.0, 30.0])
    array_out = turbine.power(speeds)
    scalar_out = np.array([turbine.power(float(v)) for v in speeds])
    np.testing.assert_array_equal(array_out, scalar_out)


def test_old_bug_no_false_power_after_cutout(turbine):
    # 旧实现 np.interp 在 25(1000) 与 26(0) 之间插值出 500
    assert turbine.power(25.5) == 0.0


def test_query_does_not_mutate_input():
    curve = np.array([[3.0, 0.0], [4.0, 100.0], [25.0, 1000.0]])
    t = Turbine("X", 80.0, 100.0, 0.8, curve)
    speeds = np.array([2.0, 10.0, 30.0])
    t.power(speeds)
    np.testing.assert_array_equal(speeds, np.array([2.0, 10.0, 30.0]))
