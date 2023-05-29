from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from mybo.battery import Battery
from mybo.constraints import BATTERY_BASE_CONSTRAINTS, ConstraintSet
from mybo.model import OptimisationModel
from mybo.objective import ObjectiveSet
from mybo.tariff import MarketExportTariff, MarketImportTariff

# from mybo.prices import generate_fake_price_data

# prices = generate_fake_price_data(
#     N=1,
#     seed=102,
#     num_points=70,
#     noise_std=0.3,
#     phase_shift=0.2,
#     autocorrelation_coefficient=0.98,
# )

base_constraints = []

battery = Battery(
    id="simple",
    charge_rate=100,
    discharge_rate=-100,
    max_capacity=100,
    min_capacity=0,
    charge_efficiency=1,
    discharge_efficiency=1,
    cycles_per_day=20,
    initial_capacity=30,
    throughput_cost=0,
    operating_losses=0.0,  # 0.017 == 1.7%
    constraints=ConstraintSet(constraints=BATTERY_BASE_CONSTRAINTS),
    objectives=ObjectiveSet(objectives=[MarketImportTariff, MarketExportTariff]),
)

delta = timedelta(minutes=5)
start_time = datetime(2023, 1, 1)
prices = [10, 10, 10, 20, 20, 20, 10, 10, 10, 10]
times = [start_time + delta * i for i in range(len(prices))]

root = {
    "price": prices,
    "time": times,
    "confidence_intervals": [
        (7, 13),
        (7, 13),
        (7, 13),
        (15, 25),
        (15, 25),
        (15, 25),
        (7, 13),
        (7, 13),
        (7, 13),
        (7, 13),
    ],
}

stoch_model = OptimisationModel(
    root_profile=root,
    scenario_profile=None,
    node=[battery],
    index_var="time",
    opt_method="stochastic",
    scenario_options={"num_scenarios": 10},
    solver_options={"solver": "glpk"},
    model_options=None,
)

ef = stoch_model.solve()

fig, ax = plt.subplots()
dfs = []

soln = ef.get_root_solution()
stochastic_df = stoch_model.root_solution_to_dataframe(soln)

stochastic_df["power"] = stochastic_df["charging"] + stochastic_df["discharging"]
stochastic_df["price"] = root.get("price")
stochastic_df["time"] = times
stochastic_df.name = "stochastic_optimal"
dfs.append(stochastic_df)

print(f"Stochastic Optimal Revenue = {ef.get_objective_value()}")

det_model = OptimisationModel(
    root_profile=root,
    scenario_profile=None,
    node=[battery],
    index_var="time",
    opt_method="deterministic",
    scenario_options=None,
    solver_options={"solver": "glpk"},
    model_options=None,
)
det_ef = det_model.solve()
det_soln = det_ef.get_root_solution()
det_df = det_model.root_solution_to_dataframe(det_soln)

det_df["power"] = det_df["charging"] + det_df["discharging"]
det_df["price"] = root.get("price")
det_df["time"] = times
det_df.name = "det_optimal"
dfs.append(det_df)

print(f"Deterministic Optimal Revenue = {det_ef.get_objective_value()}")

for df in dfs:
    # Plot the dataframes
    ax.step(df["time"], df["power"], "--", where="post", label=f"{df.name} - Power", alpha=0.4)
    ax.step(df["time"], df["price"], where="post", label=f"{df.name} - Price")
    # ax.plot(df["time"], df["capacity"], label=f"{df.name} - Capacity")
    ax.plot(df["time"], df["init_capacity"], label=f"{df.name} - Init Capacity")

ax.legend()
plt.show()
