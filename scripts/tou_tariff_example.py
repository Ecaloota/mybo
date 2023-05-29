from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from mybo.battery import Battery
from mybo.constraints import BATTERY_BASE_CONSTRAINTS, ConstraintSet
from mybo.model import OptimisationModel
from mybo.objective import ObjectiveSet
from mybo.tariff import MarketExportTariff, MarketImportTariff, TouCharge, TouImportTariff

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

fig, ax = plt.subplots()
dfs = []


def tou_rule(model_index: list[datetime]):
    st = datetime(2022, 1, 1)
    et = datetime(2024, 1, 1)

    applies_mask = []
    for i in model_index:
        if st <= i <= et:
            applies_mask.append(1)
        else:
            applies_mask.append(0)

    return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    # return np.array(applies_mask)


tou_tariff = TouImportTariff(charges=[TouCharge(rate=100, rule=tou_rule(model_index=times))], off_peak_rate=0)

tou_battery = Battery(
    id="tou_tariff_battery",
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
    objectives=ObjectiveSet(objectives=[MarketImportTariff, MarketExportTariff, tou_tariff]),
)

tariff_model = OptimisationModel(
    root_profile=root,
    scenario_profile=None,
    node=[tou_battery],
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
tariff_df.name = "det_tariff"
dfs.append(tariff_df)

print(f"Deterministic Optimal Revenue with Tariff = {tariff_ef.get_objective_value()}")


battery = Battery(
    id="tou_tariff_battery",
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
