"""ESM-2 epistasis scorer — compute fitness effects on mutant backgrounds.

Core novel operation: score mutation B on a protein sequence where mutation A
has already been introduced. The delta between LLR(B|wt) and LLR(B|A) is
the epistatic effect.

    epistasis(A,B) = LLR(B | seq_with_A) - LLR(B | wt_seq)

    > 0: synergistic (B is MORE tolerated when A is present)
    < 0: antagonistic (B is LESS tolerated when A is present)
    = 0: additive (no epistasis)

This is novel — the PNAS 2024 HIV paper (kinetic coevolutionary models)
used a Potts model fit to MSA. We compute epistasis directly from ESM-2's
masked marginal probabilities, which implicitly capture higher-order
structural and evolutionary effects.

References:
    PNAS 2024: Kinetic coevolutionary models predict DRM acquisition rates
    from epistatic networks in HIV protease, RT, integrase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from models.esm2_scorer import ESM2Scorer, LLRResult, parse_aa_mutation

logger = logging.getLogger(__name__)


@dataclass
class EpistaticEffect:
    """Pairwise epistatic interaction between two mutations."""

    mutation_a: str
    mutation_b: str
    gene: str
    llr_b_on_wt: float        # LLR(B) scored on wildtype background
    llr_b_on_a: float         # LLR(B) scored on background with A present
    epistasis: float           # delta = llr_b_on_a - llr_b_on_wt
    epistasis_type: str        # "synergistic", "antagonistic", "neutral"


def introduce_mutations(
    protein_sequence: str,
    mutations: list[str],
) -> str:
    """Apply a list of mutations to a protein sequence.

    Args:
        protein_sequence: Wildtype sequence.
        mutations: List of mutation strings like ["S450L", "H445Y"].

    Returns:
        Mutated sequence with all substitutions applied.
    """
    seq = list(protein_sequence)
    for mut in mutations:
        ref_aa, position, alt_aa = parse_aa_mutation(mut)
        pos_idx = position - 1
        if pos_idx < 0 or pos_idx >= len(seq):
            raise ValueError(f"Position {position} out of range for sequence of length {len(seq)}")
        seq[pos_idx] = alt_aa
    return "".join(seq)


def score_on_mutant_background(
    scorer: ESM2Scorer,
    protein_sequence: str,
    background_mutations: list[str],
    query_mutation: str,
    gene: str = "",
) -> tuple[LLRResult, LLRResult]:
    """Score a query mutation on both wildtype and mutant backgrounds.

    Args:
        scorer: Pre-loaded ESM2Scorer.
        protein_sequence: Wildtype protein sequence.
        background_mutations: Mutations already present in the background.
        query_mutation: The mutation to score.
        gene: Gene name for labeling.

    Returns:
        (result_on_wt, result_on_mutant) — LLRResult on each background.
    """
    # Score on wildtype
    result_wt = scorer.score_mutation(protein_sequence, query_mutation, gene=gene)

    # Introduce background mutations and score
    mutant_seq = introduce_mutations(protein_sequence, background_mutations)
    result_mut = scorer.score_mutation(mutant_seq, query_mutation, gene=gene)

    return result_wt, result_mut


def compute_pairwise_epistasis(
    scorer: ESM2Scorer,
    protein_sequence: str,
    mutations: list[str],
    gene: str = "",
) -> tuple[list[EpistaticEffect], np.ndarray]:
    """Compute all pairwise epistatic interactions among mutations.

    For N mutations, computes N*(N-1) directed interactions:
    for each pair (A, B), score B on wildtype and on A-mutant background.

    Args:
        scorer: Pre-loaded ESM2Scorer.
        protein_sequence: Wildtype protein sequence.
        mutations: List of mutation strings.
        gene: Gene name.

    Returns:
        (effects, matrix) where:
        - effects: list of EpistaticEffect for all pairs
        - matrix: N x N numpy array where matrix[i,j] = epistasis of j given i
                  (diagonal is 0, matrix[i,j] = LLR(j|i) - LLR(j|wt))
    """
    n = len(mutations)
    logger.info("Computing pairwise epistasis: %d mutations = %d pairs for %s",
                n, n * (n - 1), gene)

    # First, score all mutations on wildtype background
    wt_results = scorer.score_batch(protein_sequence, mutations, gene=gene)
    wt_llrs = {m: r.llr for m, r in zip(mutations, wt_results)}

    # Check for position conflicts (two mutations at same position)
    positions = {}
    for mut in mutations:
        _, pos, _ = parse_aa_mutation(mut)
        positions.setdefault(pos, []).append(mut)

    effects = []
    matrix = np.zeros((n, n))

    for i, mut_a in enumerate(mutations):
        _, pos_a, _ = parse_aa_mutation(mut_a)

        # Create background with mutation A
        bg_seq = introduce_mutations(protein_sequence, [mut_a])

        for j, mut_b in enumerate(mutations):
            if i == j:
                continue

            _, pos_b, _ = parse_aa_mutation(mut_b)

            # Skip if same position (can't have two mutations at same residue)
            if pos_a == pos_b:
                continue

            # Score B on A-mutant background
            result_b_on_a = scorer.score_mutation(bg_seq, mut_b, gene=gene)
            llr_b_on_a = result_b_on_a.llr
            llr_b_on_wt = wt_llrs[mut_b]
            delta = llr_b_on_a - llr_b_on_wt

            if abs(delta) < 0.1:
                etype = "neutral"
            elif delta > 0:
                etype = "synergistic"
            else:
                etype = "antagonistic"

            effect = EpistaticEffect(
                mutation_a=mut_a, mutation_b=mut_b, gene=gene,
                llr_b_on_wt=round(llr_b_on_wt, 6),
                llr_b_on_a=round(llr_b_on_a, 6),
                epistasis=round(delta, 6),
                epistasis_type=etype,
            )
            effects.append(effect)
            matrix[i, j] = delta

        if (i + 1) % 5 == 0:
            logger.info("  Computed background %d/%d (%s)", i + 1, n, mut_a)

    return effects, matrix


def compute_pathway_fitness(
    scorer: ESM2Scorer,
    protein_sequence: str,
    ordered_path: list[str],
    gene: str = "",
) -> list[dict]:
    """Compute cumulative fitness along an ordered mutation path.

    For path [A, B, C], computes:
    - LLR(A | wt)
    - LLR(B | wt+A)
    - LLR(C | wt+A+B)

    This traces whether each step in the evolutionary path is accessible
    (low |LLR| = low fitness cost at that step).

    Args:
        scorer: Pre-loaded ESM2Scorer.
        protein_sequence: Wildtype protein sequence.
        ordered_path: Mutations in acquisition order.
        gene: Gene name.

    Returns:
        List of dicts with step-by-step fitness data.
    """
    steps = []
    current_seq = protein_sequence
    accumulated = []

    for i, mutation in enumerate(ordered_path):
        result = scorer.score_mutation(current_seq, mutation, gene=gene)

        steps.append({
            "step": i + 1,
            "mutation": mutation,
            "background": list(accumulated),
            "llr": result.llr,
            "abs_llr": result.abs_llr,
            "accessible": result.abs_llr < 5.0,  # heuristic threshold
        })

        # Apply this mutation to the sequence for next step
        current_seq = introduce_mutations(current_seq, [mutation])
        accumulated.append(mutation)

    return steps
