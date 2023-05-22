# My Battery Optimiser

Simulates a simple battery system which can participate in energy arbitrage.
The input prices can be treated stochastically with respect to spot prices by uniformly sampling provided confidence intervals, or deterministically with respect to spot prices.

See `scripts/simple_example.py` for example usage.

## Overview
1. Get or generate fake price forecasts
2. Get or generate associated confidence intervals
3. Generate scenarios and probabilities from confidence intervals
4. (Normalise probabilities if sum of scenario probabilities is less than 1.0)
5. Pass scenarios and probabilities to scenario_creator function for mpi-sppy
6. Solve model subject to stochastic input prices
7. Solve deterministic model subject to original price forecasts


## Wishlist
0. Test existing features (scenario generation and normalisation; charge and discharge direction wrt prices; obvious error possibilities)
1. FCAS participation
2. Limit throughput
3. Require the battery to generate 'spread' plans, (single charge / single discharge per day). Is this a good idea?
4. Handle unsymmetric sampling of confidence distributions
5. Generate unsymmetric confidence distributions from real prices


## Known Issues
1. There is an off-by-one error in formulation. For example, initialising a battery with
init_capacity = 10 (kWh) and charge_power = 5 (kW) can sometimes generate a capacity at time t = 0 with 15 kWh (charge_efficiency = 1 also).