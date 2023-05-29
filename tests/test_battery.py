import random
from datetime import datetime, timedelta

import pytest
from helper import fuzzy_equal, generate_datetime_range

from mybo.battery import Battery
from mybo.constraints import ConstraintScenario
from mybo.model import OptimisationModel
from mybo.objective import ObjectiveScenario


@pytest.fixture
def ideal_battery() -> Battery:
    return Battery(
        id="ideal",
        charge_rate=1,
        discharge_rate=-1,
        max_capacity=10,
        min_capacity=0,
        charge_efficiency=1,
        discharge_efficiency=1,
        cycles_per_day=1,
        initial_capacity=5,
        throughput_cost=0,
        operating_losses=0,
        constraint_scenario=ConstraintScenario["SimpleConstraints"],
        objective_scenario=ObjectiveScenario["SimpleSpotMarketRevenue"],
    )


@pytest.fixture
def nonsymmetric_inefficient_battery() -> Battery:
    return Battery(
        id="inefficient",
        charge_rate=1,
        discharge_rate=-1,
        max_capacity=10,
        min_capacity=0,
        charge_efficiency=0.8,
        discharge_efficiency=0.85,
        cycles_per_day=1,
        initial_capacity=5,
        throughput_cost=0,
        operating_losses=0,
        constraint_scenario=ConstraintScenario["SimpleConstraints"],
        objective_scenario=ObjectiveScenario["SimpleSpotMarketRevenue"],
    )


@pytest.fixture
def lossy_battery() -> Battery:
    return Battery(
        id="lossy",
        charge_rate=1,
        discharge_rate=-1,
        max_capacity=10,
        min_capacity=0,
        charge_efficiency=1,
        discharge_efficiency=1,
        cycles_per_day=1,
        initial_capacity=5,
        throughput_cost=0,
        operating_losses=0.05,  # 5% losses
        constraint_scenario=ConstraintScenario["SimpleConstraints"],
        objective_scenario=ObjectiveScenario["SimpleSpotMarketRevenue"],
    )


@pytest.fixture
def nonsymmetric_inefficient_lossy_battery() -> Battery:
    return Battery(
        id="nil_battery",
        charge_rate=1,
        discharge_rate=-1,
        max_capacity=10,
        min_capacity=0,
        charge_efficiency=0.8,
        discharge_efficiency=0.85,
        cycles_per_day=1,
        initial_capacity=5,
        throughput_cost=0,
        operating_losses=0.05,  # 5% losses
        constraint_scenario=ConstraintScenario["SimpleConstraints"],
        objective_scenario=ObjectiveScenario["SimpleSpotMarketRevenue"],
    )


@pytest.fixture
def one_battery_model(ideal_battery: Battery) -> OptimisationModel:
    return OptimisationModel(
        root_profile=None,
        scenario_profile=None,
        node=[ideal_battery],
        index_var="index",
        opt_method="deterministic",
        scenario_options=None,
        solver_options={"solver": "glpk"},
        model_options=None,
    )


def test_deterministic_one_battery_model_discharges_at_high_prices(one_battery_model: OptimisationModel) -> None:
    """TODO"""

    # test 5 minute interval with high price, incentivised to discharge
    start_time = datetime(2023, 1, 1, 0)
    delta = timedelta(minutes=5)

    root_index = generate_datetime_range(start_time, start_time, delta)
    prices = [1000 for _ in range(len(root_index))]

    one_battery_model.root_profile = {"index": root_index, "price": prices}

    ef = one_battery_model.solve()
    soln = ef.get_root_solution()
    df = one_battery_model.root_solution_to_dataframe(soln)

    # assert that we discharge and not charge
    assert df["discharging"].iloc[0] < 0
    assert df["charging"].iloc[0] == 0


def test_deterministic_one_battery_model_charges_at_low_prices(one_battery_model: OptimisationModel) -> None:
    """TODO"""

    # test 5 minute interval with low price, incentivised to charge
    start_time = datetime(2023, 1, 1, 0)
    delta = timedelta(minutes=5)

    root_index = generate_datetime_range(start_time, start_time, delta)
    prices = [-10]

    one_battery_model.root_profile = {"index": root_index, "price": prices}

    ef = one_battery_model.solve()
    soln = ef.get_root_solution()
    df = one_battery_model.root_solution_to_dataframe(soln)

    # assert that we charge and not discharge
    assert df["discharging"].iloc[0] == 0
    assert df["charging"].iloc[0] > 0


def test_init_capacity_tracks_capacity_of_previous_timestep(one_battery_model: OptimisationModel) -> None:
    """TODO"""

    start_time = datetime(2023, 1, 1, 0)
    end_time = datetime(2023, 1, 1, 1)
    delta = timedelta(minutes=5)

    root_index = generate_datetime_range(start_time, end_time, delta)
    prices = [random.randint(-100, 100) for _ in range(len(root_index))]

    one_battery_model.root_profile = {"index": root_index, "price": prices}

    ef = one_battery_model.solve()
    soln = ef.get_root_solution()
    df = one_battery_model.root_solution_to_dataframe(soln)

    # assert that init_capacity at timestep i > 0 == capacity at time i-1
    system: Battery = one_battery_model.get_node_object_by_id("ideal")
    assert df.init_capacity.iloc[0] == system.initial_capacity
    assert all(df.init_capacity.iloc[1:].values == df.capacity.shift(1).iloc[1:].values)


@pytest.mark.parametrize(
    "battery, price",
    [
        (pytest.lazy_fixture("nonsymmetric_inefficient_battery"), -10),
        (pytest.lazy_fixture("nonsymmetric_inefficient_battery"), 10),
        (pytest.lazy_fixture("lossy_battery"), -10),
        (pytest.lazy_fixture("lossy_battery"), 10),
        (pytest.lazy_fixture("ideal_battery"), -10),
        (pytest.lazy_fixture("ideal_battery"), 10),
        (pytest.lazy_fixture("nonsymmetric_inefficient_lossy_battery"), -10),
        (pytest.lazy_fixture("nonsymmetric_inefficient_lossy_battery"), 10),
    ],
)
def test_power_action_respects_action_duration(
    battery: Battery, price: int, one_battery_model: OptimisationModel
) -> None:
    """TODO"""

    start_time = datetime(2023, 1, 1, 0)
    delta = timedelta(minutes=5)

    root_index = generate_datetime_range(start_time, start_time, delta)
    prices = [price]

    one_battery_model.node = [battery]
    one_battery_model.root_profile = {"index": root_index, "price": prices}

    ef = one_battery_model.solve()
    soln = ef.get_root_solution()
    df = one_battery_model.root_solution_to_dataframe(soln)

    change_in_capacity = df.init_capacity.iloc[0] + (
        df.charging.iloc[0] * battery.charge_efficiency + df.discharging.iloc[0] * battery.discharge_efficiency
    ) * (1 - battery.operating_losses) * (one_battery_model.model.duration_map[0] / 60)

    assert fuzzy_equal(df.capacity.iloc[0], change_in_capacity)


def test_duration_maps_calculated_correctly(one_battery_model: OptimisationModel) -> None:
    """TODO"""

    start_time = datetime(2023, 1, 1, 0)
    end_time = datetime(2023, 1, 1, 0, 5)
    delta = timedelta(minutes=10)

    root_index = generate_datetime_range(start_time, end_time, delta)
    prices = [10 for _ in range(len(root_index))]
    one_battery_model.root_profile = {"index": root_index, "price": prices}
    one_battery_model.solve()
    calculated = list(one_battery_model.model.duration_map.values())

    # if long enough to calculate durations, do proper assertion
    if len(root_index) > 1:
        assert calculated == [delta.seconds / 60 for i in range(len(root_index))]
    else:
        assert calculated == [5]  # assuming the assumed value is 5. too hard to get it
