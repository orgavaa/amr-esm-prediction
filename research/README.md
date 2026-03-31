# Research / Exploratory Modules

**These modules are exploratory work — not part of the published analysis.**

They contain promising directions that require additional validation before
they can support published claims. They are preserved here for transparency
and future development.

## Contents

### `models/epistasis_scorer.py`
Pairwise epistasis computed from ESM-2 masked marginals on mutant backgrounds.
The technique is novel (score mutation B after introducing mutation A into the
sequence), but the interpretation is unvalidated — the delta-LLR may reflect
transformer context effects rather than biophysical epistasis. Requires
experimental validation against combinatorial DMS data.

### `models/emergence_simulator.py`
Kinetic Monte Carlo Wright-Fisher simulation driven by ESM-2 fitness + epistasis.
In testing, the simulation **reduced prediction accuracy** for katG (concordance
0.833 → 0.333 vs simple |LLR| ranking), indicating the population dynamics
layer introduces noise rather than signal for small mutation sets.

### `models/binding_disruption.py`
Drug-binding disruption prediction from structural proximity and ESM-2 embedding
deltas at binding sites. Functional but the de novo panel design achieves 0%
recall for most targets except the trivial single-position case (katG).

### `experiments/emergence_forecast.py`
Full emergence forecasting pipeline (epistasis + KMC + lineage stratification).
Results for 7 MTB targets are in `results/emergence/` if previously generated.

### `experiments/denovo_design.py`
De novo diagnostic panel design for pipeline drugs (BTZ043, telacebec).
Predictions cannot be validated without clinical resistance data for these
Phase 2 compounds. Preserved as future work.

## Status

These modules are importable and runnable but their outputs should be treated
as hypotheses, not validated results. The published analysis uses only:
- `models/esm2_scorer.py`
- `experiments/retrospective.py`
- `experiments/prospective.py`
- `experiments/panel_design.py`
