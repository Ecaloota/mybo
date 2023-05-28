import pyomo.environ as pyo
from pydantic import BaseModel


class FlatRateImportTariff(BaseModel):
    """A tariff which is charged against import at a fixed rate"""

    rate: float

    def _objective(self, model: pyo.ConcreteModel):
        # TODO doesn't account for interval durations (i.e. 5/60 term)
        return sum(self.rate * model.charging[t] for t in model.time)


class FlatRateExportTariff(BaseModel):
    """A tariff which is charged against export at a fixed rate"""

    rate: float

    def _objective(self, model: pyo.ConcreteModel):
        # TODO as above
        return sum(self.rate * model.discharging[t] for t in model.time)


class MarketImportTariff:
    """A tariff which is charged against import at the spot market rate at some time"""

    pass


class MarketExportTariff:
    """A tariff which is charged against export at the spot market rate at some time"""

    pass


class TouImportTariff:
    # TODO
    pass


class TouExportTariff:
    # TODO
    pass
