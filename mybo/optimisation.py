import random
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import mpisppy.utils.sputils as sputils
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from battery import Battery
from constraints import ConstraintScenario
from mpisppy.opt.ef import ExtensiveForm

# from objective import objective_rule
from prices import generate_fake_price_data

# prices = generate_fake_price_data(
#     N=1,
#     seed=102,
#     num_points=70,
#     noise_std=0.3,
#     phase_shift=0.2,
#     autocorrelation_coefficient=0.98,
# )

# 1. Get or generate fake price forecasts
# 2. Get or generate associated confidence intervals
# 3. Generate scenarios and probabilities from confidence intervals
# 4. Normalise probabilities if sum of scenario probabilities is less than 1.0
# 5. Pass scenarios to scenario_creator function for mpi-sppy


def generate_values_and_probability(confidence_intervals):
    """
    Given a list of confidence intervals, we generate a set of values (assumed uniform)
    and calculate the probability of the set of values occurring.
    """

    # Calculate the values within the confidence intervals
    values = []
    overall_probability = 1.0

    for confidence_interval in confidence_intervals:
        lower_bound, upper_bound = confidence_interval
        value = random.uniform(lower_bound, upper_bound)
        values.append(value)
        probability = 1 / (upper_bound - lower_bound)
        overall_probability *= probability

    return values, overall_probability


# prices = [[10, 10, 10, 20, 20, 20, 10, 10, 10, 10]]
global_prices = [10, 10, 10, 20, 20, 20, 10, 10, 10, 10]


def scenario_creator(scenario_name):
    if scenario_name == "good":
        prices = [10, 10, 10, 20, 20, 20, 10, 10, 10, 10]
        prob = 0.2
    elif scenario_name == "average":
        prices = [10, 10, 10, 0, 0, 0, 10, 10, 10, 10]
        prob = 0.7
    elif scenario_name == "bad":
        prices = [10, 10, 10, -20, -20, -20, 10, 10, 10, 10]
        prob = 0.1
    else:
        raise ValueError("Unrecognized scenario name")

    model = build_model(prices)

    # decompose as a two-stage problem. We tell attach_root_node which part of the
    # objective function and which Vars belong to the first stage
    sputils.attach_root_node(model, firstobj=model.objective, varlist=[model.charging, model.discharging])
    model._mpisppy_probability = prob
    return model


def build_model(prices):
    model = pyo.ConcreteModel()
    model.time = pyo.RangeSet(0, len(prices) - 1)
    model.prices = pyo.Param(model.time, initialize={t: prices[t] for t in model.time})
    model.big_m = 1000

    battery_instance = Battery(
        charge_rate=10,
        discharge_rate=-10,
        max_capacity=100,
        min_capacity=0,
        charge_efficiency=1,
        discharge_efficiency=1,
        cycles_per_day=20,
        initial_capacity=30,
        throughput_cost=0,
        operating_losses=0.0,  # 0.017 == 1.7%
        constraint_scenario=ConstraintScenario["SimpleConstraints"],
    )

    battery_instance.build_model(model)

    # objective
    model.objective = pyo.Objective(
        expr=pyo.quicksum(model.prices[i] * (model.charging[i] + model.discharging[i]) for i in model.time),
        sense=pyo.minimize,
    )

    return model


if __name__ == "__main__":
    options = {"solver": "glpk"}
    all_scenario_names = ["good", "average", "bad"]
    ef = ExtensiveForm(options, all_scenario_names, scenario_creator)
    results = ef.solve_extensive_form()

    fig, ax = plt.subplots()

    dfs = []

    # make this a function
    soln = ef.get_root_solution()

    result_dict = defaultdict(list)
    for key, value in soln.items():
        result = re.match(r"(\w+)\[(\d+)\]", key)
        result_dict[result.groups()[0]].append(value)

    stochastic_df = pd.DataFrame(result_dict)
    stochastic_df["power"] = stochastic_df["charging"] + stochastic_df["discharging"]
    stochastic_df["prices"] = global_prices
    stochastic_df["time"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    stochastic_df.name = "stochastic_optimal"
    dfs.append(stochastic_df)

    print(f"Stochastic Optimal Revenue = {ef.get_objective_value()}")

    # this gives each of the scenario ConcreteModel objects (subject to stochasticity)
    # for sname, smodel in ef.scenarios():
    #     power_values = [pyo.value(smodel.power[key]) for key in smodel.power]
    #     prices_values = [pyo.value(smodel.prices[key]) for key in smodel.prices]
    #     time_values = [t for t in smodel.time]
    #     df = pd.DataFrame({"time": time_values, "power": power_values, "prices": prices_values})
    #     df.name = sname
    #     dfs.append(df)

    # this gives the deterministic solution
    model = build_model(global_prices)
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model)

    print(f"Deterministic Revenue = {pyo.value(model.objective)}")

    power_values = [pyo.value(model.power[key]) for key in model.power]
    # capacity_values = [pyo.value(model.capacity[key]) for key in model.capacity]
    prices_values = [pyo.value(model.prices[key]) for key in model.prices]
    time_values = [t for t in model.time]

    df = pd.DataFrame({"time": time_values, "power": power_values, "prices": prices_values})
    df.name = "deterministic"
    dfs.append(df)

    for df in dfs:
        # Plot the dataframes
        ax.plot(df["time"], df["power"], "--", label=f"{df.name} - Power", alpha=0.4)
        ax.plot(df["time"], df["prices"], label=f"{df.name} - Prices")

    ax.legend()
    plt.show()
