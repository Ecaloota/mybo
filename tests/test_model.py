import random
from datetime import datetime, timedelta

import pytest
from helper import generate_datetime_range

from mybo.model import OptimisationModel


@pytest.fixture
def base_opt_model():
    return OptimisationModel(
        root_profile=None,
        scenario_profile=None,
        node=None,
        index_var="index",
        opt_method="deterministic",
        scenario_options=None,
        solver_options=None,
        model_options=None,
    )


@pytest.mark.parametrize(
    "start_time, end_time, delta",
    [
        (datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 1), timedelta(minutes=5)),
    ],
)
def test_get_root_index_from_optimisation_model(
    start_time: datetime, end_time: datetime, delta: timedelta, base_opt_model: OptimisationModel
) -> None:
    """TODO"""

    root_index = generate_datetime_range(start_time, end_time, delta)
    base_opt_model.root_profile = {"index": root_index}
    ind = base_opt_model._get_index_from_root_profile()

    assert len(root_index) == len(ind)
    assert list(ind) == list(range(len(ind)))


@pytest.mark.parametrize(
    "start_time, end_time, delta",
    [
        (datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 1), timedelta(minutes=5)),
    ],
)
def test_get_root_prices_from_optimisation_model(
    start_time: datetime, end_time: datetime, delta: timedelta, base_opt_model: OptimisationModel
) -> None:
    """TODO"""

    root_index = generate_datetime_range(start_time, end_time, delta)
    prices = [random.randint(0, 10) for i in range(len(root_index))]
    base_opt_model.root_profile = {"index": root_index, "price": prices}

    # this is required to assign the root_index required by the root_price Var
    assigned = list(base_opt_model.model.root_price.values())
    assert assigned == base_opt_model.root_profile.get("price")
