from enum import Enum

import pyomo.environ as pyo


def spot_market_arbitrage_objective(model: pyo.ConcreteModel) -> None:
    """TODO is this name appropriate?"""

    def _spot_market_arbitrage_rule(model: pyo.ConcreteModel):
        """"""
        return pyo.quicksum(
            model.scenario_prices[i] * (model.charging[i] + model.discharging[i]) for i in model.scenario_index
        )

    # for i in model.time:
    model.objectives.add(expr=_spot_market_arbitrage_rule(model))


# def objective_rule(model: ConcreteModel):
#     """Defines the objective for our optimisation from our model"""

#     # import_rate = -100
#     # export_rate = -100
#     # import_tariff = sum(import_rate * model.charging[t] for t in model.time)
#     # export_tariff = sum(export_rate * model.discharging[t] for t in model.time)

#     # throughput_cost = sum(
#     #     model.throughput_cost * (model.charging[t] - model.discharging[t])
#     #     for t in model.time
#     # )

#     # when charging, cost > 0
#     cost = sum(model.prices[i] * (model.charging[i] + model.discharging[i]) for i in model.time)

#     # return revenue + throughput_cost
#     # return revenue + throughput_cost + import_tariff + export_tariff
#     # return cost + throughput_cost
#     return cost


class ObjectiveScenario(Enum):
    SimpleSpotMarketRevenue = [spot_market_arbitrage_objective]
