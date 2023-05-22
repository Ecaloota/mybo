# import pyomo.environ as pyo
# import pytest

# from mybo.battery import Battery


# @pytest.fixture
# def battery():
#     return Battery(
#         charge_rate=1,
#         discharge_rate=-1,
#         max_capacity=10,
#         min_capacity=0,
#         charge_efficiency=1,
#         discharge_efficiency=1,
#         cycles_per_day=1,
#         initial_capacity=5,
#         throughput_cost=0,
#         operating_losses=0,
#     )


# @pytest.mark.parametrize("prices", [[-100], [100]])
# def test_one_state_only(battery: Battery, prices: list[int]) -> None:
#     """Assert that the battery may only be in one state at some time"""

#     model = pyo.ConcreteModel()
#     model.time = pyo.RangeSet(0, len(prices) - 1)

#     battery.build_model(model)

#     model.prices = pyo.Param(model.time, initialize=prices)
#     model.objective = pyo.Objective(
#         rule=sum(model.prices[i] * (model.charging[i] + model.discharging[i]) for i in model.time), sense=pyo.minimize
#     )

#     solver = pyo.SolverFactory("glpk")
#     solver.solve(model)

#     # if we provided incentive to charge, ensure we charge
#     if prices[0] < 0:
#         assert (
#             model.action["charge", 0].value == 1.0
#             and model.action["discharge", 0] != 1.0
#             and model.action["idle", 0] != 1.0
#         )

#     elif prices[0] > 0:
#         assert (
#             model.action["charge", 0].value != 1.0
#             and model.action["discharge", 0] == 0.0
#             and model.action["idle", 0] != 1.0
#         )

#     print("wow")
