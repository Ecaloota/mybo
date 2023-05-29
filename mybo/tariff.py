from typing import Any

import numpy as np
import pyomo.environ as pyo
from pydantic import BaseModel

from mybo.objective import Objective

MINUTES_PER_HOUR = 60


class FlatRateImportTariff(Objective):
    """A tariff which is charged against import at a fixed rate. A rate > 0 defines a loss
    of revenue for energy import (e.g. how much we are charged to import)"""

    rate: float

    def _objective_rule(self, model: pyo.ConcreteModel):
        return pyo.quicksum(
            self.rate * model.charging[i] * model.duration_map[i] / MINUTES_PER_HOUR for i in model.scenario_index
        )


class FlatRateExportTariff(Objective):
    """A tariff which is charged against export at a fixed rate. A rate > 0 defines a loss
    of revenue for energy export (e.g. how much we are charged to export). Note the -1 term
    in the objective_rule definition"""

    rate: float

    def _objective_rule(self, model: pyo.ConcreteModel):
        return pyo.quicksum(
            -1 * self.rate * model.discharging[i] * model.duration_map[i] / MINUTES_PER_HOUR
            for i in model.scenario_index
        )


class MarketImportTariff(Objective):
    """A tariff which is charged against import at the spot market rate at some time"""

    # @staticmethod
    def _objective_rule(model: pyo.ConcreteModel):
        return pyo.quicksum(
            model.scenario_price[i] * model.charging[i] * (model.duration_map[i] / MINUTES_PER_HOUR)
            for i in model.scenario_index
        )


class MarketExportTariff(Objective):
    """A tariff which is charged against export at the spot market rate at some time"""

    # @staticmethod
    def _objective_rule(model: pyo.ConcreteModel):
        return pyo.quicksum(
            model.scenario_price[i] * model.discharging[i] * (model.duration_map[i] / MINUTES_PER_HOUR)
            for i in model.scenario_index
        )


class TouCharge(BaseModel):
    """A tariff which applies at the given rate when the rule returns true, and does
    not apply otherwise."""

    rate: float
    rule: Any

    @property
    def charge(self):
        return np.array(self.rate * self.rule)


class TouImportTariff(Objective):
    """A TOU ImportTariff applies any of a list of TouCharges to optimisation intervals
    according to their rules, and optionally applies an off_peak_rate to any intervals
    which have no charges applied.
    """

    charges: list[TouCharge]
    off_peak_rate: float

    @property
    def off_peak_rate_applies(self):
        all_rules = [a.rule for a in self.charges]
        return [all(not arr[i] for arr in all_rules) for i in range(len(all_rules[0]))]

    def _objective_rule(self, model: pyo.ConcreteModel):
        total_charge = sum(x.charge for x in self.charges)
        total_charge[self.off_peak_rate_applies] = self.off_peak_rate

        return pyo.quicksum(
            total_charge[i] * model.charging[i] * (model.duration_map[i] / MINUTES_PER_HOUR)
            for i in model.scenario_index
        )


class TouExportTariff(Objective):
    # TODO
    pass
