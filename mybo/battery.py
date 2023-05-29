import pyomo.environ as pyo
from pydantic import BaseModel

from mybo.constraints import ConstraintSet
from mybo.objective import ObjectiveSet


class Battery(BaseModel):
    """Battery TODO"""

    id: str  # unique identifier
    charge_rate: float  # kW
    discharge_rate: float  # kW
    max_capacity: float  # kWh
    min_capacity: float  # kWh
    charge_efficiency: float  # unitless proportion
    discharge_efficiency: float  # unitless proportion
    cycles_per_day: float
    initial_capacity: float
    throughput_cost: float
    operating_losses: float  # intended to be pct > 0
    constraints: ConstraintSet
    objectives: ObjectiveSet

    def _create_vars(self, model: pyo.ConcreteModel) -> None:
        """"""

        model.power = pyo.Var(
            model.scenario_index, within=pyo.Reals, bounds=(self.discharge_rate, self.charge_rate), initialize=0
        )
        model.charging = pyo.Var(
            model.scenario_index, within=pyo.NonNegativeReals, bounds=(0.0, self.charge_rate), initialize=0
        )
        model.discharging = pyo.Var(
            model.scenario_index, within=pyo.NonPositiveReals, bounds=(self.discharge_rate, 0.0), initialize=0
        )
        model.capacity = pyo.Var(
            model.scenario_index,
            within=pyo.NonNegativeReals,
            bounds=(self.min_capacity, self.max_capacity),
            initialize=self.initial_capacity,
        )
        model.is_charging = pyo.Var(model.scenario_index, within=pyo.Binary)

        model.init_capacity = pyo.Var(
            model.scenario_index,
            within=pyo.NonNegativeReals,
            bounds=(self.min_capacity, self.max_capacity),
            initialize=self.initial_capacity,
        )

    def _create_params(self, model: pyo.ConcreteModel) -> None:
        """"""

        model.throughput_cost = pyo.Param(initialize=self.throughput_cost)
        model.charge_efficiency = pyo.Param(initialize=self.charge_efficiency)
        model.discharge_efficiency = pyo.Param(initialize=self.discharge_efficiency)
        model.initial_capacity = pyo.Param(initialize=self.initial_capacity)
        model.operating_losses = pyo.Param(initialize=self.operating_losses)

    def _apply_constraints(self, model: pyo.ConcreteModel) -> None:
        """"""

        for i in model.scenario_index:
            self.constraints.set_constraint(model, i)

    def _define_objectives(self, model: pyo.ConcreteModel) -> None:
        """"""

        self.objectives.set_objective(model)

    def build_model(self, model: pyo.ConcreteModel):
        """"""

        self._create_vars(model)
        self._create_params(model)
        self._apply_constraints(model)
        self._define_objectives(model)
