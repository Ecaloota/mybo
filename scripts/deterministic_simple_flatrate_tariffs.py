from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from mybo.battery import Battery
from mybo.constraints import BATTERY_BASE_CONSTRAINTS, ConstraintSet
from mybo.model import OptimisationModel
from mybo.objective import ObjectiveSet
from mybo.tariff import FlatRateExportTariff, FlatRateImportTariff, MarketExportTariff, MarketImportTariff

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

import_tariff_battery = Battery(
    id="simple_tariffs",
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
    objectives=ObjectiveSet(objectives=[MarketImportTariff, MarketExportTariff, FlatRateImportTariff(rate=100)]),
)

export_tariff_battery = Battery(
    id="simple_tariffs",
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
    objectives=ObjectiveSet(objectives=[MarketImportTariff, MarketExportTariff, FlatRateExportTariff(rate=100)]),
)


delta = timedelta(minutes=5)
start_time = datetime(2023, 1, 1)
prices = [10, 10, 10, -20, -20, -20, 10, 10, 10, 10]
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

fig, ax = plt.subplots()
dfs = []

tariff_model = OptimisationModel(
    root_profile=root,
    scenario_profile=None,
    node=[import_tariff_battery],
    index_var="time",
    opt_method="deterministic",
    scenario_options=None,
    solver_options={"solver": "glpk"},
    model_options=None,
)
tariff_ef = tariff_model.solve()
tariff_soln = tariff_ef.get_root_solution()
tariff_df = tariff_model.root_solution_to_dataframe(tariff_soln)

tariff_df["power"] = tariff_df["charging"] + tariff_df["discharging"]
tariff_df["price"] = root.get("price")
tariff_df["time"] = times
tariff_df.name = "det_import_tariff"
dfs.append(tariff_df)

print(f"Deterministic Optimal Revenue with Tariff = {tariff_ef.get_objective_value()}")


export_tariff_model = OptimisationModel(
    root_profile=root,
    scenario_profile=None,
    node=[export_tariff_battery],
    index_var="time",
    opt_method="deterministic",
    scenario_options=None,
    solver_options={"solver": "glpk"},
    model_options=None,
)
export_tariff_ef = export_tariff_model.solve()
export_tariff_soln = export_tariff_ef.get_root_solution()
export_tariff_df = export_tariff_model.root_solution_to_dataframe(export_tariff_soln)

export_tariff_df["power"] = export_tariff_df["charging"] + export_tariff_df["discharging"]
export_tariff_df["price"] = root.get("price")
export_tariff_df["time"] = times
export_tariff_df.name = "det_export_tariff"
dfs.append(export_tariff_df)

print(f"Deterministic Optimal Revenue with EXPORT Tariff = {export_tariff_ef.get_objective_value()}")


non_tariff_model = OptimisationModel(
    root_profile=root,
    scenario_profile=None,
    node=[battery],
    index_var="time",
    opt_method="deterministic",
    scenario_options=None,
    solver_options={"solver": "glpk"},
    model_options=None,
)
non_tariff_ef = non_tariff_model.solve()
non_tariff_soln = non_tariff_ef.get_root_solution()
non_tariff_df = non_tariff_model.root_solution_to_dataframe(non_tariff_soln)

non_tariff_df["power"] = non_tariff_df["charging"] + non_tariff_df["discharging"]
non_tariff_df["price"] = root.get("price")
non_tariff_df["time"] = times
non_tariff_df.name = "det_non_tariff"
dfs.append(non_tariff_df)

print(f"Deterministic Optimal Revenue without Tariff = {non_tariff_ef.get_objective_value()}")


for df in dfs:
    # Plot the dataframes
    ax.step(df["time"], df["power"], "--", where="post", label=f"{df.name} - Power", alpha=0.4)
    ax.step(df["time"], df["price"], where="post", label=f"{df.name} - Price")
    # ax.plot(df["time"], df["capacity"], label=f"{df.name} - Capacity")
    ax.plot(df["time"], df["init_capacity"], label=f"{df.name} - Init Capacity")

ax.legend()
plt.show()
