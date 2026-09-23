"""内置机型、派生参数、AEP 查表与对外摘要一致性的回归测试。"""

import numpy as np
import pytest

from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wake import JensenWake, compute_wake_interactions
from wind_farm_opt.core.wind_resource import create_simple_wind_resource
from wind_farm_opt.farm.aep import AEPCalculator


@pytest.mark.parametrize(
    "model,rated_kw,cut_in,rated_speed,cut_out",
    [
        ("V164-9.5MW", 9500.0, 4.0, 11.5, 25.0),
        ("V126-3.45MW", 3450.0, 3.5, 12.0, 25.0),
    ],
)
def test_builtin_derived_parameters(model, rated_kw, cut_in, rated_speed, cut_out):
    t = create_default_turbine(model)
    # 额定功率不允许改变
    assert t.rated_power == rated_kw
    assert t.cut_in_speed == cut_in
    assert t.rated_speed == rated_speed
    # 核心回归：切出风速必须是 25，不能是曲线上最后一个采样点
    assert t.cut_out_speed == cut_out


@pytest.mark.parametrize("model", ["V164-9.5MW", "V126-3.45MW"])
def test_builtin_power_boundaries(model):
    t = create_default_turbine(model)
    assert t.power(t.cut_out_speed) == pytest.approx(t.rated_power)
    assert t.power(25.5) == 0.0
    assert t.power(30.0) == 0.0
    assert t.power(0.0) == 0.0
    arr = t.power(np.array([0.0, 25.0, 25.5, 30.0]))
    np.testing.assert_allclose(arr, [0.0, t.rated_power, 0.0, 0.0])


@pytest.mark.parametrize("model", ["V164-9.5MW", "V126-3.45MW"])
def test_spec_summary_matches_derived(model):
    t = create_default_turbine(model)
    s = t.spec_summary()
    assert s["name"] == model
    assert s["rated_power_kw"] == t.rated_power
    assert s["rated_power_mw"] == t.rated_power / 1e3
    assert s["cut_in_speed_mps"] == t.cut_in_speed
    assert s["rated_speed_mps"] == t.rated_speed
    assert s["cut_out_speed_mps"] == t.cut_out_speed


def _far_apart_calculator(model="V164-9.5MW", n=2, speed_step=0.5):
    turbines = [create_default_turbine(model) for _ in range(n)]
    wr = create_simple_wind_resource(num_sectors=4, uniform=True, mean_speed=9.0)
    return turbines, wr, AEPCalculator(
        turbines=turbines,
        wind_resource=wr,
        wake_model=JensenWake(wake_decay=0.07),
        speed_step=speed_step,
        speed_max=30.0,
    )


def test_aep_lookup_matches_turbine_power():
    """AEP 内部查表必须与 Turbine.power 使用同一边界约定。"""
    turbines, _, calc = _far_apart_calculator()
    for i, t in enumerate(turbines):
        np.testing.assert_array_equal(
            calc._power_lookup[i], t.power(calc._speed_centers)
        )
    # 切出之后的积分节点全部为零功率
    above = calc._speed_centers > turbines[0].cut_out_speed
    assert np.all(calc._power_lookup[:, above] == 0.0)
    # 最接近切出点的中心节点为额定功率
    idx = np.argmin(np.abs(calc._speed_centers - 25.0))
    assert calc._power_lookup[0, idx] == pytest.approx(9500.0)


def test_gross_aep_matches_manual_integration():
    """无尾流时 gross/net AEP 与直接用 power() 积分相互吻合。"""
    turbines, wr, calc = _far_apart_calculator(n=2, speed_step=0.5)
    # 两台风机相距 20 km，尾流可忽略
    positions = np.array([[0.0, 0.0], [20000.0, 0.0]])
    result = calc.compute_farm_aep(positions, return_details=False)

    centers = calc._speed_centers
    step = calc.speed_step
    manual_total_kw = 0.0
    for s_idx, sector in enumerate(wr.sectors):
        prob = wr.weibull_pdf(centers, s_idx) * step
        for t in turbines:
            manual_total_kw += np.sum(t.power(centers) * prob * sector.frequency * 8760.0)

    assert result.gross_aep == pytest.approx(manual_total_kw / 1e3, rel=1e-10)
    # 相距足够远，净 AEP ≈ 理论 AEP
    assert result.net_aep == pytest.approx(result.gross_aep, rel=1e-9)
    # 装机容量 = 2 × 9.5 MW
    assert result.total_installed_capacity == pytest.approx(19.0)


def test_evaluate_layout_matches_full_computation():
    turbines, wr, calc = _far_apart_calculator(n=2, speed_step=1.0)
    positions = np.array([[0.0, 0.0], [600.0, 0.0]])
    fast = calc.evaluate_layout(positions)
    full = calc.compute_farm_aep(positions, return_details=False)
    assert fast == pytest.approx(full.net_aep, rel=1e-12)


def test_wake_interaction_zero_above_cutout():
    """自由来流超过切出时，功率查询必须为 0（不与停机点插值）。"""
    t = create_default_turbine("V164-9.5MW")
    # 2 km 下游亏损约 7.5%：自由来流 28 m/s 与受扰速度 ~25.9 m/s
    # 均已超过 25 m/s 切出。
    positions = np.array([[0.0, 0.0], [2000.0, 0.0]])
    interactions = compute_wake_interactions(
        positions=positions,
        wind_direction=270.0,
        rotor_diameters=np.array([t.rotor_diameter, t.rotor_diameter]),
        thrust_coefficients=np.array([t.thrust_coefficient, t.thrust_coefficient]),
        wake_model=JensenWake(0.07),
        power_curves=[t.power_curve, t.power_curve],
        free_stream_speed=28.0,
    )
    # 旧实现会在 25(额定) 与 26(0) 之间为受扰速度 25.9 插值出虚假功率；
    # 两者都在切出后，功率差必须为 0。
    assert interactions, "顺风向两台风机应产生相互作用记录"
    for it in interactions:
        assert it.affected_power == 0.0
