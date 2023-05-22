from typing import Any

import pyomo.environ as pyo
from pydantic import BaseModel


class Battery(BaseModel):
    """Battery TODO"""

    charge_rate: float  # kW
    discharge_rate: float  # kW
    max_capacity: float  # kWh
    min_capacity: float  # kWh
    charge_efficiency: float  # unitless proportion
    discharge_efficiency: float  # unitless proportion
    cycles_per_day: float
    initial_capacity: float
    throughput_cost: float
    operating_losses: float  # intended to be pct < 0
    constraint_scenario: Any  # TODO fix this?
    objective_scenario: Any  # TODO fix this annotation?

    @property
    def name(self):
        # TODO to disambiguate multiple batteries
        pass

    def _create_vars(self, model: pyo.ConcreteModel) -> None:
        """"""

        # TODO assert the model.time index exists

        model.power = pyo.Var(
            model.time, within=pyo.Reals, bounds=(self.discharge_rate, self.charge_rate), initialize=0
        )
        model.charging = pyo.Var(model.time, within=pyo.NonNegativeReals, bounds=(0.0, self.charge_rate), initialize=0)
        model.discharging = pyo.Var(
            model.time, within=pyo.NonPositiveReals, bounds=(self.discharge_rate, 0.0), initialize=0
        )
        model.capacity = pyo.Var(
            model.time,
            within=pyo.NonNegativeReals,
            bounds=(self.min_capacity, self.max_capacity),
            initialize=self.initial_capacity,
        )
        model.is_charging = pyo.Var(model.time, within=pyo.Binary)

    def _create_params(self, model: pyo.ConcreteModel) -> None:
        """"""

        model.throughput_cost = pyo.Param(initialize=self.throughput_cost)
        model.charge_efficiency = pyo.Param(initialize=self.charge_efficiency)
        model.discharge_efficiency = pyo.Param(initialize=self.discharge_efficiency)
        model.initial_capacity = pyo.Param(initialize=self.initial_capacity)
        model.operating_losses = pyo.Param(initialize=self.operating_losses)

    def _apply_constraints(self, model: pyo.ConcreteModel) -> None:
        """"""

        for constr in self.constraint_scenario.value:
            constr(model)

    def _define_objectives(self, model: pyo.ConcreteModel) -> None:
        """"""

        for obj in self.objective_scenario.value:
            obj(model)

    def build_model(self, model: pyo.ConcreteModel):
        """"""

        self._create_vars(model)
        self._create_params(model)
        self._apply_constraints(model)
        self._define_objectives(model)
