"""AEP 查表与尾流交互必须遵循与 Turbine.power 相同的边界约定。"""

import numpy as np
import pytest

from wind_farm_opt.core.turbine import Turbine, create_default_turbine
from wind_farm_opt.core.wake import JensenWake, compute_wake_interactions
from wind_farm_opt.core.wind_resource import create_simple_wind_resource
from wind_farm_opt.farm.aep import AEPCalculator


def _calculator(model="V164-9.5MW", n=4, speed_step=0.5):
    turbines = [create_default_turbine(model) for _ in range(n)]
    wr = create_simple_wind_resource(num_sectors=4, uniform=True, mean_speed=8.0)
    return AEPCalculator(
        turbines=turbines,
        wind_resource=wr,
        wake_model=JensenWake(0.07),
        speed_step=speed_step,
        speed_max=30.0,
    )


def test_power_lookup_matches_turbine_power():
    calc = _calculator()
    t = calc.turbines[0]
    np.testing.assert_allclose(
        calc._power_lookup[0], t.power(calc._speed_centers)
    )


def test_lookup_is_zero_at_and_above_cut_out_bin():
    calc = _calculator(model="V164-9.5MW")
    t = calc.turbines[0]
    centers = calc._speed_centers
    above = centers > t.cut_out_speed
    assert np.all(calc._power_lookup[0][above] == 0.0)
    # 切出点所在 bin 仍是额定功率（最后一个有效发电点）
    at = np.argmin(np.abs(centers - t.cut_out_speed))
    assert abs(centers[at] - t.cut_out_speed) <= calc.speed_step
    assert calc._power_lookup[0][at] == t.rated_power


def test_evaluate_layout_equals_farm_result_net_aep():
    calc = _calculator()
    positions = np.array([[0.0, 0.0], [800.0, 0.0],
                          [0.0, 800.0], [800.0, 800.0]])
    farm = calc.compute_farm_aep(positions)
    fast = calc.evaluate_layout(positions)
    assert fast == pytest.approx(farm.net_aep, rel=1e-12)


def test_aep_identical_for_shuffled_duplicated_curve():
    """乱序 + 完全重复点的曲线规范化后，AEP 必须与整洁曲线一致。"""

    def make_calc(curve_factory, model):
        turbines = [
            Turbine(
                name=model,
                hub_height=105.0,
                rotor_diameter=164.0,
                thrust_coefficient=0.80,
                power_curve=curve_factory(),
            )
            for _ in range(3)
        ]
        wr = create_simple_wind_resource(num_sectors=4, uniform=True, mean_speed=9.0)
        return AEPCalculator(
            turbines=turbines, wind_resource=wr,
            wake_model=JensenWake(0.07), speed_step=0.5,
        )

    base = create_default_turbine("V164-9.5MW")
    clean = base.power_curve.copy()

    rng = np.random.default_rng(1)
    messy = np.vstack([clean, clean[::3]])
    messy = messy[rng.permutation(len(messy))]

    positions = np.array([[0.0, 0.0], [700.0, 0.0], [350.0, 700.0]])
    aep_clean = make_calc(lambda: clean, "V164").compute_farm_aep(positions)
    aep_messy = make_calc(lambda: messy, "V164").compute_farm_aep(positions)
    assert aep_messy.gross_aep == pytest.approx(aep_clean.gross_aep, rel=1e-12)
    assert aep_messy.net_aep == pytest.approx(aep_clean.net_aep, rel=1e-12)


def test_wake_interactions_respect_cut_out_with_unsorted_curve():
    base = create_default_turbine("V164-9.5MW")
    clean = base.power_curve
    rng = np.random.default_rng(2)
    messy = clean[rng.permutation(len(clean))]

    def run(curves, spacing, free_stream):
        positions = np.array([[0.0, 0.0], [spacing, 0.0]])
        return compute_wake_interactions(
            positions=positions,
            wind_direction=270.0,
            rotor_diameters=np.array([164.0, 164.0]),
            thrust_coefficients=np.array([0.8, 0.8]),
            wake_model=JensenWake(0.07),
            power_curves=curves,
            free_stream_speed=free_stream,
        )

    # 乱序曲线与整洁曲线的交互结果必须逐一对等（不会被静默扭曲）
    near = run([clean, clean], 800.0, 15.0)
    near_messy = run([messy, messy], 800.0, 15.0)
    assert near and len(near) == len(near_messy)
    for a, b in zip(near, near_messy):
        assert a.affected_power == pytest.approx(b.affected_power)

    # 远间距、来流略高于切出：受扰后仍高于切出，双方均停机
    far = run([clean, clean], 4000.0, 26.0)
    assert far
    for item in far:
        assert item.affected_power == 0.0
