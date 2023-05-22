from enum import Enum

import pyomo.environ as pyo


def capacity_constraint(model: pyo.ConcreteModel) -> None:
    """Adds the constraint that the model capacity must change according to the
    capacity constraint rule to the model constraints list"""

    # TODO check that model contains time index?

    def _capacity_constraint_rule(model: pyo.ConcreteModel, i: int):
        """The capacity at some time i must be equal to the previous capacity plus
        the change in capacity, where change in capacity is given by the change
        in capacity rule"""

        # TODO check that model contains charging, charge_efficiency, discharging,
        # discharge_efficiency, initial_capacity, capacity, and operating_losses

        def _change_in_capacity_rule(model: pyo.ConcreteModel, i: int):
            """The capacity at some time i is given by the below expression"""

            return model.charging[i] * model.charge_efficiency + model.discharging[i] * model.discharge_efficiency

        prev_capacity = model.initial_capacity if i == 0 else model.capacity[i - 1] * (1 - model.operating_losses)
        return model.capacity[i] == prev_capacity + _change_in_capacity_rule(model, i)

    for i in model.time:
        model.constraints.add(expr=_capacity_constraint_rule(model, i))


def power_balance_constraint(model: pyo.ConcreteModel) -> None:
    """Adds the constraint that the model power must always equal the sum of the
    model positive power (charging) and negative power (discharging) to the
    model constraints list"""

    # TODO check that model contains time index

    def _power_balance_constraint_rule(model: pyo.ConcreteModel, i: int):
        """Ensures that the sum of the charging and discharging power values
        must equal the total power"""

        # TODO check that model contains charging, discharging, and power

        return model.charging[i] + model.discharging[i] == model.power[i]

    for i in model.time:
        model.constraints.add(expr=_power_balance_constraint_rule(model, i))


def exclusive_action_constraint(model: pyo.ConcreteModel):
    """Adds a pair of constraints to the model constraints list that together
    outline that the system may only charge OR discharge or remain idle at some time
    i, never some combination at the same time."""

    def _one_action_constraint_one(model: pyo.ConcreteModel, i: int):
        return model.charging[i] <= model.big_m * model.is_charging[i]

    def _one_action_constraint_two(model: pyo.ConcreteModel, i: int):
        return model.discharging[i] >= model.big_m * (model.is_charging[i] - 1)

    model.exclusive_action_constraint = pyo.ConstraintList()
    for i in model.time:
        model.exclusive_action_constraint.add(expr=_one_action_constraint_one(model, i))
        model.exclusive_action_constraint.add(expr=_one_action_constraint_two(model, i))


# def _charge_once_constraint(self, model: pyo.ConcreteModel, i):
#     """"""
#     return sum(model.is_charging[i] for i in model.time) <= 1

# def _discharge_once_constraint(self, model: pyo.ConcreteModel, i):
#     """"""
#     return sum(model.is_discharging[i] for i in model.time) <= 1

# def _cycle_limit_constraint(self, model, i):
#     return sum(model.charging[i] + -1 * model.discharging[i] for i in model.time) <= self.cycles_per_day * (
#         -1 * self.discharge_rate + self.charge_rate
#     )

# def _charge_limit_constraint(self, model, i):
#     return sum(model.charging[i] for i in model.time) <= self.cycles_per_day * self.charge_rate

# def _discharge_limit_constraint(self, model, i):
#     return sum(-1 * model.discharging[i] for i in model.time) <= self.cycles_per_day * -1 * self.discharge_rate

# def _hard_discharge_limit_constraint(self, model, i):
#     return sum(-1 * model.discharging[i] for i in model.time) == self.cycles_per_day

# def _hard_charge_limit_constraint(self, model, i):
#     return sum(model.charging[i] for i in model.time) == self.cycles_per_day


class ConstraintScenario(Enum):
    SimpleConstraints = [capacity_constraint, power_balance_constraint, exclusive_action_constraint]
