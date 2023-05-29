import logging
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import mpisppy.utils.sputils as sputils
import pandas as pd
import pyomo.environ as pyo
from mpisppy.opt.ef import ExtensiveForm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OptimisationModel(BaseModel):
    """
    An OptimisationModel is built from an input profile (prices, loads, index, etc)
    and objects on node. Constraints and objectives objects are initialised, and
    each object on node is then built. Scenario generation types are requested
    and the corresponding scenarios are defined and initialised.
    the model is then solved, and the model defines methods to access the results.

    Note that the root index and root vars are defined only so we have a fixed reference
    against which perturbation can be performed independently when generating
    scenarios. When defining objectives and node Vars etc., one should define them
    against scenario Vars.

    profile: input information, contains prices, time index, etc
    node: list of objects in the model
    opt_method: one of ['stochastic', 'deterministic']
    index_var: optional str, determines which keys in the root profile are used to make optimisation Vars
    scenario_options: determines how scenarios are generated. Only relevant when opt_method is 'stochastic'
    solver_options: determines how the solver is invoked
    model_options: determines how the model is defined. Contains a lot of random info.
    """

    root_profile: Any  # required for instantiation
    scenario_profile: Optional[Any]  # not required
    node: Any
    opt_method: Any
    index_var: Optional[str]  # must exist as a key in root profile TODO
    scenario_options: Optional[dict]
    solver_options: Optional[dict]
    model_options: Optional[dict]

    @property
    def model(self) -> pyo.ConcreteModel:
        """
        The ConcreteModel instantiation occurs here. We define variables which
        are used by child objects and by methods of the OptimisationModel class.
        """

        m = pyo.ConcreteModel()
        m.big_m = 1000  # or get it from model_options if defined TODO
        m.constraints = pyo.ConstraintList()
        m.objectives = pyo.ObjectiveList()

        m.root_index = self._get_index_from_root_profile()
        m.scenario_index = self._get_index_from_root_profile()  # scenarios will always have the same length as root
        m.duration_map = self._calculate_duration_map(m)

        # we build optimisation vars from keys in root profile
        var_list = [k for k in list(self.root_profile.keys()) if k != self.index_var]
        for var in var_list:
            setattr(m, f"root_{var}", self._get_var_from_root_profile(var, m))

            if self.scenario_profile is not None:
                setattr(m, f"scenario_{var}", self._get_var_from_scenario_profile(var, m))

        # build child objects
        if self.node is not None:
            for child in self.node:
                child.build_model(m)

        return m

    def _get_index_from_root_profile(self) -> pyo.Set:
        """The model is defined over a simple 0-indexed set. One can access timedelta
        information by indexing against this set"""
        return pyo.RangeSet(0, len(self.root_profile[self.index_var]) - 1)

    def _get_var_from_root_profile(self, var: str, model: pyo.ConcreteModel) -> pyo.Param:
        # TODO error catching
        return pyo.Param(model.root_index, initialize=self.root_profile.get(var), domain=pyo.Any)

    def _get_var_from_scenario_profile(self, var: str, model: pyo.ConcreteModel) -> pyo.Param:
        # TODO error catching
        return pyo.Param(model.scenario_index, initialize=self.scenario_profile.get(var), domain=pyo.Any)

    def _create_stochastic_scenario(self, scenario_name: str) -> pyo.ConcreteModel:
        """We decompose the optimisation problem into two stages, and tell
        attach_root_node which part of the objective function and which Vars
        belong to the first stage e.g. which are non-anticipative, or nonants."""

        self.scenario_profile = self._generate_scenario_profile_from_root()
        model = self.model
        sputils.attach_root_node(
            model,
            firstobj=model.objective,
            varlist=[model.charging, model.discharging, model.capacity, model.init_capacity],
        )
        model._mpisppy_probability = self.scenario_profile.get("scenario_probability")

        return model

    def _create_deterministic_scenario(self, scenario_name: str) -> pyo.ConcreteModel:
        """Functionally identical to create_stochastic_scenario, but we give the
        scenario a probability of unity."""

        self.scenario_profile = self.root_profile
        model = self.model
        sputils.attach_root_node(
            model,
            firstobj=model.objective,
            varlist=[model.charging, model.discharging, model.capacity, model.init_capacity],
        )
        model._mpisppy_probability = 1.0
        return model

    def _generate_scenario_profile_from_root(self):
        """"""
        # we have root profile prices. we can bootstrap confidence intervals
        # or use existing confidence intervals to generate samples
        # which form the basis for a scenario

        scenario_profile = {}

        ci = self.root_profile.get("confidence_intervals")
        prices = self.root_profile.get("price")

        # generate sampled prices from confidence intervals on root prices
        if ci is not None:
            prob = 1.0
            sample_prices = []

            for idx, _ in enumerate(prices):
                lower_bound, upper_bound = ci[idx]
                random_value = random.uniform(lower_bound, upper_bound)
                prob *= 1 / (upper_bound - lower_bound)
                sample_prices.append(random_value)

            scenario_profile = {
                "time": self.root_profile.get("time"),
                "price": sample_prices,
                "scenario_probability": prob,
            }

            return scenario_profile

        else:
            raise NotImplementedError

    def _normalise_extensive_form_scenarios(self, ef: ExtensiveForm) -> None:
        """"""

        # I believe the ExtensiveForm object does this automatically
        # but why risk it

        probs = []
        for scenario in ef.scenarios():
            probs.append(scenario[1]._mpisppy_probability)

        total = sum(probs)
        normalized_numbers = [num / total for num in probs]
        for idx, scenario in enumerate(ef.scenarios()):
            scenario[1]._mpisppy_probability = normalized_numbers[idx]

    def _solve_stochastic_problem(self) -> ExtensiveForm:
        """Creates n_scenarios of the form expected by mpi-sppy.
        Returns the solved ExtensiveForm object"""

        # create a list of scenario names from info in the scenario_options dict
        num_scenarios = self.scenario_options.get("num_scenarios")
        scenarios = [f"scenario_{i}" for i in range(num_scenarios)]

        ef = ExtensiveForm(self.solver_options, scenarios, self._create_stochastic_scenario)
        self._normalise_extensive_form_scenarios(ef)

        ef.solve_extensive_form()  # this could fail, catch that
        return ef

    def _solve_deterministic_problem(self) -> ExtensiveForm:
        """Creates n_scenarios of the form expected by mpi-sppy. Functionally identical
        to _solve_stochastic_problem but with n_scenarios = 1 and with no bootstrapping
        from the root_profile. Returns the solved ExtensiveForm object"""

        ef = ExtensiveForm(self.solver_options, ["deterministic"], self._create_deterministic_scenario)
        ef.solve_extensive_form()  # this could fail, catch that
        return ef

    def _calculate_duration_map(self, model: pyo.ConcreteModel, assumed: float = 5) -> list[float]:
        """Determine or assume the length of simulation intervals from the given
        root_profile, in minutes. If the index contains one (or zero? TODO) interval(s),
        we assume its duration, regardless of the type of the elements in the index. If
        the length of the index is > 1 and its elements are datetimes, we calculate their
        duration and pad out the resulting list. Otherwise, we just assume a duration
        map of lengths assumed."""

        idx = self.root_profile[self.index_var]

        if len(idx) <= 1:
            logger.warning(f"Cannot determine duration_map. Assumed to be {assumed} minutes")
            durations = [assumed]
        elif all(isinstance(x, datetime) for x in idx):
            durations = [(idx[i + 1] - idx[i]).total_seconds() / 60 for i in range(len(idx) - 1)]
            durations.append(durations[-1])
        else:
            durations = [assumed for _ in range(len(idx))]

        return pyo.Param(model.scenario_index, initialize=durations, domain=pyo.Any)

    def solve(self) -> ExtensiveForm:
        """Passes to the appropriate solver method"""

        if self.opt_method == "stochastic":
            soln = self._solve_stochastic_problem()

        elif self.opt_method == "deterministic":
            soln = self._solve_deterministic_problem()

        else:
            raise NotImplementedError

        return soln

    def root_solution_to_dataframe(self, solution: dict) -> pd.DataFrame:
        """Converts the solution from a successful call of ef.get_root_solution
        (where ef is a solved instance of ExtensiveForm) to a dataframe"""

        result = defaultdict(list)
        for k, v in solution.items():
            matches = re.match(r"(\w+)\[(\d+)\]", k)
            new_key = matches.groups()[0]
            result[new_key].append(v)

        if "price" in self.root_profile.keys():
            result["price"] = self.root_profile.get("price")

        df = pd.DataFrame(result, index=self.root_profile.get(self.index_var))

        return df

    def get_node_object_by_id(self, id: str):
        """TODO"""

        obj = [i for i in self.node if i.id == id]
        return obj[0] if obj is not None else None
