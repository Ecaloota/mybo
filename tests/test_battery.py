from datetime import datetime, timedelta

import pytest
from helper import generate_datetime_range

from mybo.battery import Battery
from mybo.constraints import ConstraintScenario
from mybo.model import OptimisationModel
from mybo.objective import ObjectiveScenario


@pytest.fixture
def ideal_battery() -> Battery:
    return Battery(
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


def test_build_one_battery_model(one_battery_model: OptimisationModel) -> None:
    """TODO"""

    # test 5 minute interval with high price, incentivised to discharge
    start_time = datetime(2023, 1, 1, 0)
    delta = timedelta(minutes=5)

    root_index = generate_datetime_range(start_time, start_time, delta)
    prices = [[1000 for _ in range(len(root_index))]]

    one_battery_model.root_profile = {"index": root_index, "price": prices}

    ef = one_battery_model.solve()
    soln = ef.get_root_solution()
    one_battery_model.root_solution_to_dataframe(soln)
