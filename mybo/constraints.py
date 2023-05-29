# from enum import Enum
from typing import Any

import pyomo.environ as pyo
from pydantic import BaseModel

MINUTES_PER_HOUR = 60


class Constraint(BaseModel):
    def _constraint_rule(self, model: pyo.ConcreteModel, idx: int):
        raise NotImplementedError


class ConstraintSet(BaseModel):
    constraints: list[Any]

    def set_constraint(self, model: pyo.ConcreteModel, idx: int):
        cons: Constraint
        for cons in self.constraints:
            model.constraints.add(expr=cons._constraint_rule(model, idx))


class CapacityConstraint(Constraint):
    """Adds the constraint that the model capacity must change according to the
    capacity constraint rule to the model constraints list"""

    def _constraint_rule(self, model: pyo.ConcreteModel, idx: int):
        def _change_in_capacity_rule(model: pyo.ConcreteModel, idx: int):
            return (
                (model.charging[idx] * model.charge_efficiency + model.discharging[idx] * model.discharge_efficiency)
                * (1 - model.operating_losses)
                * (model.duration_map[idx] / MINUTES_PER_HOUR)
            )

        prev_capacity = model.initial_capacity if idx == 0 else model.capacity[idx - 1]
        return model.capacity[idx] == prev_capacity + _change_in_capacity_rule(model, idx)


class InitialCapacityConstraint(Constraint):
    """Defines the constraint that the initial capacity at some time must always equal
    the capacity at the previous timestep, or the initial_capacity at the start of the index"""

    def _constraint_rule(self, model: pyo.ConcreteModel, idx: int):
        prev_capacity = model.initial_capacity if idx == 0 else model.capacity[idx - 1]
        return model.init_capacity[idx] == prev_capacity


class PowerBalanceConstraint(Constraint):
    """Adds the constraint that the model power must always equal the sum of the
    model positive power (charging) and negative power (discharging) to the
    model constraints list"""

    def _constraint_rule(self, model: pyo.ConcreteModel, idx: int):
        return model.charging[idx] + model.discharging[idx] == model.power[idx]


class ExclusiveActionConstraint(Constraint):
    """Adds a pair of constraints to the model constraints list that together
    outline that the system may only charge OR discharge or remain idle at some time
    i, never some combination at the same time."""

    def _constraint_rule(self, model: pyo.ConcreteModel, idx: int):
        return (model.charging[idx] <= model.big_m * model.is_charging[idx]) and (
            model.discharging[idx] >= model.big_m * model.is_charging[idx] - 1
        )


BATTERY_BASE_CONSTRAINTS = [
    CapacityConstraint(),
    InitialCapacityConstraint(),
    PowerBalanceConstraint(),
    InitialCapacityConstraint(),
]
