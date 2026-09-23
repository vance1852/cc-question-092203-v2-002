"""功率曲线规范化、参数推导与边界查询的回归测试。"""

import numpy as np
import pytest

from wind_farm_opt.core.turbine import (
    Turbine,
    create_default_turbine,
    normalize_power_curve,
    derive_curve_parameters,
    interp_power_curve,
)


# 一条带切入锚点、上升段、额定平台和切出后零点的物理曲线
def _sample_curve():
    ws = np.arange(0.0, 31.0, 1.0)
    pw = np.zeros_like(ws)
    for i, v in enumerate(ws):
        if 4.0 <= v <= 25.0:
            if v < 12.0:
                pw[i] = 3000.0 * ((v - 4.0) / 8.0) ** 3
            else:
                pw[i] = 3000.0
    return np.column_stack([ws, pw])


def _make_turbine(curve):
    return Turbine(
        name="TEST",
        hub_height=80.0,
        rotor_diameter=100.0,
        thrust_coefficient=0.8,
        power_curve=curve,
    )


# ---------------------------------------------------------------------------
# 内置机型：额定功率不变，切出风速不再被末尾零点污染
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,rated,cut_in,rated_speed,cut_out",
    [
        ("V164-9.5MW", 9500.0, 4.0, 11.5, 25.0),
        ("V126-3.45MW", 3450.0, 3.5, 12.0, 25.0),
    ],
)
class TestBuiltinTurbines:
    def test_derived_parameters(self, model, rated, cut_in, rated_speed, cut_out):
        t = create_default_turbine(model)
        assert t.rated_power == rated
        assert t.cut_in_speed == cut_in
        assert t.rated_speed == rated_speed
        # 回归：旧实现把 30 m/s 的零功率点当成切出风速
        assert t.cut_out_speed == cut_out
        assert t.power_curve[-1, 0] == cut_out
        assert t.power_curve[-1, 1] == rated

    def test_summary_matches_attributes(self, model, rated, cut_in, rated_speed, cut_out):
        t = create_default_turbine(model)
        s = t.summary()
        assert s["rated_power_kW"] == rated
        assert s["cut_in_speed"] == cut_in
        assert s["rated_speed"] == rated_speed
        assert s["cut_out_speed"] == cut_out
        assert s["rated_platform"] == [rated_speed, cut_out]

    def test_boundary_conventions(self, model, rated, cut_in, rated_speed, cut_out):
        t = create_default_turbine(model)
        # 切入前与切入锚点
        assert t.power(0.0) == 0.0
        assert t.power(cut_in - 0.001) == 0.0
        assert t.power(cut_in) == 0.0
        # 首次达到额定
        assert t.power(rated_speed) == rated
        # 平台内
        assert t.power((rated_speed + cut_out) / 2.0) == rated
        # 切出点仍发电，切出后立即归零（关键回归点）
        assert t.power(cut_out) == rated
        assert t.power(cut_out + 1e-9) == 0.0
        assert t.power(30.0) == 0.0


def test_builtin_rated_powers_unchanged():
    assert create_default_turbine("V164-9.5MW").rated_power == 9500.0
    assert create_default_turbine("V126-3.45MW").rated_power == 3450.0


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_trailing_and_leading_zeros_stripped_but_anchor_kept(self):
        ws, pw = normalize_power_curve(_sample_curve())
        # 0~3 m/s 的零点被剥离，4 m/s 的切入锚点保留
        assert ws[0] == 4.0 and pw[0] == 0.0
        # 26~30 m/s 的切出后零点全部剥离
        assert ws[-1] == 25.0 and pw[-1] == 3000.0

    def test_unsorted_points_are_sorted(self):
        curve = _sample_curve()
        shuffled = curve[np.random.default_rng(0).permutation(len(curve))]
        ws, pw = normalize_power_curve(shuffled)
        assert np.all(np.diff(ws) > 0)
        assert ws[0] == 4.0 and ws[-1] == 25.0

        t1, t2 = _make_turbine(curve), _make_turbine(shuffled)
        probe = np.linspace(0.0, 30.0, 61)
        np.testing.assert_allclose(t1.power(probe), t2.power(probe))

    def test_exact_duplicate_points_deduplicated(self):
        curve = _sample_curve()
        dup = np.vstack([curve, curve[5], curve[5]])
        ws, pw = normalize_power_curve(dup)
        assert np.all(np.diff(ws) > 0)

    def test_conflicting_duplicate_wind_speeds_rejected(self):
        curve = _sample_curve()
        # 追加一个同风速但不同功率的点（排序后形成歧义重复点）
        bad = np.vstack([curve, [[6.0, 999.0]]])
        with pytest.raises(ValueError, match="歧义"):
            normalize_power_curve(bad)

    def test_negative_power_rejected(self):
        bad = _sample_curve()
        bad[3, 1] = -1.0
        with pytest.raises(ValueError, match="负功率"):
            normalize_power_curve(bad)

    def test_negative_wind_speed_rejected(self):
        bad = _sample_curve()
        bad[0, 0] = -0.5
        with pytest.raises(ValueError, match="非负"):
            normalize_power_curve(bad)

    @pytest.mark.parametrize("modify", [
        lambda c: np.insert(c, 2, [5.0, np.nan], axis=0),
        lambda c: np.insert(c, 2, [np.inf, 100.0], axis=0),
    ])
    def test_nonfinite_values_rejected(self, modify):
        with pytest.raises(ValueError, match="有限"):
            normalize_power_curve(modify(_sample_curve()))

    def test_no_positive_power_rejected(self):
        zeros = np.column_stack([np.arange(5.0), np.zeros(5)])
        with pytest.raises(ValueError, match="正功率"):
            normalize_power_curve(zeros)

    def test_bad_shape_rejected(self):
        with pytest.raises(ValueError, match="形状"):
            normalize_power_curve(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="采样点"):
            normalize_power_curve(np.array([[3.0, 0.0]]))

    def test_curve_without_cut_in_anchor_uses_first_positive(self):
        # 没有零功率锚点：切入风速取第一个正功率点
        ws, pw = normalize_power_curve(
            np.array([[4.0, 100.0], [8.0, 1000.0], [12.0, 2000.0],
                      [16.0, 2000.0]])
        )
        ci, vr, co, pr = derive_curve_parameters(ws, pw)
        assert ci == 4.0 and vr == 12.0 and co == 16.0 and pr == 2000.0


# ---------------------------------------------------------------------------
# 额定平台与歧义曲线
# ---------------------------------------------------------------------------

class TestRatedPlatform:
    def test_power_drop_after_rated_rejected(self):
        curve = np.array([
            [3.0, 0.0], [5.0, 1000.0], [10.0, 3000.0],
            [12.0, 3000.0], [14.0, 2000.0], [16.0, 3000.0],
        ])
        with pytest.raises(ValueError, match="额定平台"):
            _make_turbine(curve)

    def test_no_platform_single_rated_point_accepted(self):
        # 仅最后一个点达到额定，平台退化为单点也是合法平台
        t = _make_turbine(np.array([
            [3.0, 0.0], [5.0, 1000.0], [10.0, 2500.0], [12.0, 3000.0],
        ]))
        assert t.rated_speed == 12.0 == t.cut_out_speed
        assert t.rated_platform == (12.0, 12.0)
        assert t.power(12.0) == 3000.0
        assert t.power(12.1) == 0.0


# ---------------------------------------------------------------------------
# 标量 / 数组查询的一致性
# ---------------------------------------------------------------------------

class TestQueryConsistency:
    @pytest.mark.parametrize("model", ["V164-9.5MW", "V126-3.45MW"])
    def test_scalar_matches_array_everywhere(self, model):
        t = create_default_turbine(model)
        probes = np.array([
            -1.0, 0.0,
            t.cut_in_speed - 1e-6, t.cut_in_speed,
            0.5 * (t.cut_in_speed + t.rated_speed),
            t.rated_speed,
            0.5 * (t.rated_speed + t.cut_out_speed),
            t.cut_out_speed - 1e-6, t.cut_out_speed,
            t.cut_out_speed + 1e-6, 30.0, 100.0,
        ])
        scalar_results = np.array([t.power(float(v)) for v in probes])
        array_results = t.power(probes)
        np.testing.assert_array_equal(scalar_results, array_results)

    def test_scalar_returns_python_float(self):
        t = create_default_turbine("V164-9.5MW")
        assert isinstance(t.power(8.0), float)
        assert isinstance(t.power(np.float64(8.0)), float)

    def test_interpolated_value_on_rise_segment(self):
        t = create_default_turbine("V164-9.5MW")
        expected = 9500.0 * ((5.0 - 4.0) / (11.5 - 4.0)) ** 3
        assert t.power(5.0) == pytest.approx(expected)

    def test_module_level_interp_shares_convention(self):
        ws, pw = normalize_power_curve(_sample_curve())
        assert interp_power_curve(25.0, ws, pw) == 3000.0
        assert interp_power_curve(25.0001, ws, pw) == 0.0
        arr = interp_power_curve(np.array([3.9, 4.0, 20.0, 26.0]), ws, pw)
        np.testing.assert_array_equal(arr, [0.0, 0.0, 3000.0, 0.0])
