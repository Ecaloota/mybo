from typing import Any

import pyomo.environ as pyo
from pydantic import BaseModel


class Objective(BaseModel):
    """Generic base class for holding tariff information"""

    def _objective_rule(self, model: pyo.ConcreteModel):
        raise NotImplementedError


class ObjectiveSet(BaseModel):
    objectives: list[Any]

    def set_objective(self, model):
        def objective_rule(model):
            return sum(obj._objective_rule(model) for obj in self.objectives)

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
