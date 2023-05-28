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

## Usage

```bash
pip install -r requirements.txt
pip install -e .
```

## Wishlist
0. Test existing features (scenario generation and normalisation; charge and discharge direction wrt prices; obvious error possibilities; stochastic model)
1. FCAS participation
2. Limit throughput
3. Require the battery to generate 'spread' plans, (single charge / single discharge per day). Is this a good idea?
4. Handle unsymmetric sampling of confidence distributions
5. Generate unsymmetric confidence distributions from real prices


## TODO

1. Can we get the best outcome in the highest number of plausible scenarios (rather than the best outcome for all)? How do we formulate this?
2. Uniform sampling doesn't consider autocorrelation between prices. This will produce weird scenarios. Can we autocorrelate between scenarios?
3. We can forecast background load reasonably well (we have good characterisations of uncertainty). Can we use this formulation to solve for stochastic load and solar to analyse different outcomes (i.e. how we choose to charge batteries?)? This is in the context of TOU tariffs, and optimising battery capacity wrt those TOUs and uncertain loads and solar profiles.