"""Kinetic Monte Carlo simulator for resistance mutation emergence.

Simulates a bacterial population under drug pressure to predict which
resistance mutations emerge first, second, etc. Uses ESM-2 fitness
landscape + pairwise epistasis matrix to compute genotype fitness.

The simulation is a Gillespie-type stochastic process:
- Population is a collection of genotypes (mutation combinations)
- Each generation, new mutations arise at rate mu * N
- Fitness of each genotype = product of individual mutation fitnesses
  corrected by pairwise epistatic interactions
- Drug selection: genotypes above resistance threshold survive

This adapts the approach of the PNAS 2024 HIV kinetic coevolutionary
model to protein language model fitness predictions.

References:
    Gillespie (1977): Exact stochastic simulation of coupled reactions.
    PNAS 2024: Kinetic coevolutionary models predict HIV DRM acquisition.
    Torres Ortiz et al. (Nat Comm 2021): MTB temporal resistance ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Parameters for emergence simulation."""

    population_size: int = 10_000
    mutation_rate_per_site: float = 2.5e-10  # per base per generation (MTB)
    generations: int = 1000
    drug_concentration_mic_ratio: float = 1.0  # [drug] / MIC_wt
    selection_strength: float = 5.0  # sigmoid steepness
    n_replicates: int = 100
    seed: int = 42
    detection_threshold: float = 0.01  # frequency to count as "emerged"


@dataclass
class EmergenceEvent:
    """A single mutation emergence event in one replicate."""

    mutation: str
    generation: int
    frequency: float
    background: list[str]
    replicate: int


@dataclass
class EmergenceResult:
    """Aggregated emergence results across replicates."""

    mutation: str
    median_emergence_gen: float
    mean_emergence_gen: float
    std_emergence_gen: float
    emergence_probability: float  # fraction of replicates where it emerged
    first_emergence_events: list[EmergenceEvent]


class MutationTrajectorySimulator:
    """Simulate resistance emergence using fitness landscape + epistasis.

    The fitness of a genotype with mutations {m1, m2, ...} is:

        w(genotype) = w0 * exp(sum_i s_i + sum_{i<j} e_{ij})

    where:
        s_i = selection coefficient of mutation i (derived from LLR)
        e_{ij} = epistatic correction between mutations i and j
        w0 = wildtype fitness (1.0 without drug, reduced under drug)

    Drug selection adds a fitness bonus for resistant genotypes:

        w_drug(genotype) = w(genotype) * (1 + benefit * resistance_level)

    where resistance_level is 1 if ANY mutation in the genotype is known
    to confer resistance, 0 otherwise.
    """

    def __init__(
        self,
        mutations: list[str],
        llr_values: dict[str, float],
        epistasis_matrix: np.ndarray | None = None,
        resistance_mutations: set[str] | None = None,
    ):
        """
        Args:
            mutations: List of possible resistance mutations.
            llr_values: Dict mapping mutation -> LLR value.
            epistasis_matrix: N x N matrix where [i,j] = epistasis(i given j).
                If None, assumes additive model (no epistasis).
            resistance_mutations: Set of mutations that confer drug resistance.
                If None, all mutations are assumed to confer resistance.
        """
        self.mutations = mutations
        self.n_mutations = len(mutations)
        self.mut_to_idx = {m: i for i, m in enumerate(mutations)}
        self.llr_values = llr_values
        self.epistasis_matrix = epistasis_matrix
        self.resistance_mutations = resistance_mutations or set(mutations)

        # Convert LLR to selection coefficients
        # LLR < 0 means wildtype is preferred; |LLR| is the fitness cost
        # s = -|LLR| / scale maps to a selection coefficient
        self._selection_coefficients = np.array([
            -abs(llr_values.get(m, 0.0)) / 10.0  # scale to reasonable range
            for m in mutations
        ])

    def _genotype_fitness(
        self,
        genotype: np.ndarray,
        drug_benefit: float,
    ) -> float:
        """Compute fitness of a genotype (binary vector of mutations).

        Args:
            genotype: Binary array of length n_mutations (1 = mutation present).
            drug_benefit: Fitness benefit from drug resistance.

        Returns:
            Relative fitness (wildtype = 1.0).
        """
        # Additive fitness component
        fitness_log = np.dot(genotype, self._selection_coefficients)

        # Epistatic corrections
        if self.epistasis_matrix is not None:
            present = np.where(genotype)[0]
            for i in present:
                for j in present:
                    if i != j:
                        fitness_log += self.epistasis_matrix[i, j] / 10.0

        fitness = np.exp(fitness_log)

        # Drug selection benefit
        if drug_benefit > 0:
            has_resistance = any(
                self.mutations[i] in self.resistance_mutations
                for i in np.where(genotype)[0]
            )
            if has_resistance:
                fitness *= (1.0 + drug_benefit)

        return fitness

    def simulate_trajectory(
        self,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> list[EmergenceEvent]:
        """Run one replicate of the emergence simulation.

        Uses a Wright-Fisher-like model with mutation and selection.
        Tracks genotype frequencies over generations.

        Returns:
            List of EmergenceEvent for mutations that cross detection threshold.
        """
        n = config.population_size
        drug_benefit = config.drug_concentration_mic_ratio * config.selection_strength

        # State: count of each genotype in population
        # Start with all wildtype (no mutations)
        # Represent genotypes as tuples of mutation indices for efficiency
        genotype_counts: dict[tuple[int, ...], int] = {(): n}

        events = []
        emerged = set()

        # Effective per-site mutation probability per individual per generation
        mu = config.mutation_rate_per_site * 3e6  # genome size ~3 Mb for MTB
        # Scale to per-target-site: we only care about our specific mutations
        mu_per_mutation = mu / (3e6 / self.n_mutations)
        mu_per_mutation = max(mu_per_mutation, 1e-6)  # floor for simulation

        for gen in range(config.generations):
            # Calculate fitness for each genotype
            genotype_list = list(genotype_counts.keys())
            counts = np.array([genotype_counts[g] for g in genotype_list])
            fitnesses = np.zeros(len(genotype_list))

            for i, g in enumerate(genotype_list):
                gvec = np.zeros(self.n_mutations)
                for idx in g:
                    gvec[idx] = 1
                fitnesses[i] = self._genotype_fitness(gvec, drug_benefit)

            # Selection: sample next generation proportional to fitness * count
            weights = fitnesses * counts
            total_weight = weights.sum()
            if total_weight <= 0:
                break

            probs = weights / total_weight
            new_counts_raw = rng.multinomial(n, probs)
            new_genotype_counts: dict[tuple[int, ...], int] = {}

            for i, g in enumerate(genotype_list):
                if new_counts_raw[i] > 0:
                    new_genotype_counts[g] = new_counts_raw[i]

            # Mutation: each individual has a chance to gain a new mutation
            mutations_to_add: dict[tuple[int, ...], int] = {}
            for g, count in list(new_genotype_counts.items()):
                n_new_mutations = rng.binomial(count, mu_per_mutation)
                if n_new_mutations > 0 and len(g) < self.n_mutations:
                    # Pick which mutation to add
                    available = [i for i in range(self.n_mutations) if i not in g]
                    if available:
                        for _ in range(n_new_mutations):
                            new_mut_idx = rng.choice(available)
                            new_g = tuple(sorted(g + (new_mut_idx,)))
                            mutations_to_add[new_g] = mutations_to_add.get(new_g, 0) + 1
                            new_genotype_counts[g] = max(0, new_genotype_counts[g] - 1)

            for g, count in mutations_to_add.items():
                new_genotype_counts[g] = new_genotype_counts.get(g, 0) + count

            # Remove zero-count genotypes
            genotype_counts = {g: c for g, c in new_genotype_counts.items() if c > 0}

            # Check for emergence events
            for g, count in genotype_counts.items():
                freq = count / n
                if freq >= config.detection_threshold:
                    for mut_idx in g:
                        mut = self.mutations[mut_idx]
                        if mut not in emerged:
                            emerged.add(mut)
                            bg = [self.mutations[i] for i in g if i != mut_idx]
                            events.append(EmergenceEvent(
                                mutation=mut, generation=gen,
                                frequency=round(freq, 4),
                                background=bg, replicate=-1,
                            ))

        return events

    def estimate_emergence_times(
        self,
        config: SimulationConfig,
    ) -> list[EmergenceResult]:
        """Run multiple replicates and compute emergence time distributions.

        Returns:
            List of EmergenceResult, one per mutation, sorted by median
            emergence time (earliest first).
        """
        rng = np.random.default_rng(config.seed)

        # Collect emergence times per mutation across replicates
        emergence_times: dict[str, list[int]] = {m: [] for m in self.mutations}
        all_events: dict[str, list[EmergenceEvent]] = {m: [] for m in self.mutations}

        for rep in range(config.n_replicates):
            events = self.simulate_trajectory(config, rng)
            seen = set()
            for event in events:
                if event.mutation not in seen:
                    seen.add(event.mutation)
                    event.replicate = rep
                    emergence_times[event.mutation].append(event.generation)
                    all_events[event.mutation].append(event)

            if (rep + 1) % 20 == 0:
                logger.info("  Replicate %d/%d", rep + 1, config.n_replicates)

        # Aggregate results
        results = []
        for mut in self.mutations:
            times = emergence_times[mut]
            if times:
                results.append(EmergenceResult(
                    mutation=mut,
                    median_emergence_gen=float(np.median(times)),
                    mean_emergence_gen=float(np.mean(times)),
                    std_emergence_gen=float(np.std(times)),
                    emergence_probability=len(times) / config.n_replicates,
                    first_emergence_events=all_events[mut][:5],
                ))
            else:
                results.append(EmergenceResult(
                    mutation=mut,
                    median_emergence_gen=float("inf"),
                    mean_emergence_gen=float("inf"),
                    std_emergence_gen=0.0,
                    emergence_probability=0.0,
                    first_emergence_events=[],
                ))

        results.sort(key=lambda r: r.median_emergence_gen)
        return results


def find_dominant_pathways(
    results: list[EmergenceResult],
    min_frequency: float = 0.1,
) -> list[dict]:
    """Identify the most common mutation acquisition orderings.

    Looks at which mutations co-occur in emergence events and
    their temporal ordering across replicates.

    Returns:
        List of pathway dicts with ordering and frequency.
    """
    from collections import Counter

    # Extract pairwise orderings from emergence events
    ordering_counts: Counter[tuple[str, str]] = Counter()

    # Group events by replicate
    events_by_rep: dict[int, list[EmergenceEvent]] = {}
    for r in results:
        for event in r.first_emergence_events:
            events_by_rep.setdefault(event.replicate, []).append(event)

    for rep, events in events_by_rep.items():
        events_sorted = sorted(events, key=lambda e: e.generation)
        for i, e1 in enumerate(events_sorted):
            for e2 in events_sorted[i + 1:]:
                ordering_counts[(e1.mutation, e2.mutation)] += 1

    # Convert to pathways
    pathways = []
    total_reps = len(events_by_rep)
    for (m1, m2), count in ordering_counts.most_common(20):
        freq = count / max(total_reps, 1)
        if freq >= min_frequency:
            pathways.append({
                "first": m1,
                "second": m2,
                "frequency": round(freq, 3),
                "count": count,
            })

    return pathways
