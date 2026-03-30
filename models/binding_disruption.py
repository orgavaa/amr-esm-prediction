"""Drug-binding disruption prediction from structure and ESM-2 embeddings.

Two complementary approaches to predict which mutations disrupt drug binding:

Approach A (Structural): Distance from mutation to drug-binding residues.
    Closer = more likely to disrupt binding. Uses co-crystal PDB structures.
    No ML, no training data — pure physics prior.

Approach B (Embedding): Change in ESM-2 representation at the binding site
    when a mutation is introduced. Large embedding delta = large structural
    perturbation at binding site. Uses ESM-2 layer-33 representations.
    This can capture allosteric effects that distance alone misses.

Combined score:
    disruption = alpha * structural_score + (1 - alpha) * embedding_score

References:
    ConPLex (PNAS 2023): Contrastive PLM + drug embedding for DTI.
    AMRscope (bioRxiv 2025): ESM-2 embeddings for AMR variant triage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from models.esm2_scorer import ESM2Scorer, parse_aa_mutation

logger = logging.getLogger(__name__)


@dataclass
class BindingDisruptionScore:
    """Combined binding disruption prediction for a mutation."""

    mutation: str
    position: int
    structural_score: float   # 0 = far from binding site, 1 = in binding site
    embedding_score: float    # L2 distance of embedding change at binding site
    combined_score: float     # weighted combination
    min_distance_to_binding: int  # closest binding residue (in sequence positions)


class StructuralBindingPredictor:
    """Predict binding disruption from proximity to drug-binding residues.

    Uses a sigmoid function on sequence distance (or 3D distance if PDB
    available) to estimate disruption probability. Simple but effective
    for most drug targets where resistance mutations cluster near the
    binding site.

    For targets without PDB structures, uses sequence distance as proxy
    for 3D distance (valid for globular proteins with well-defined pockets).
    """

    def __init__(
        self,
        binding_residues: list[int],
        distance_threshold: float = 5.0,
        sigmoid_steepness: float = 1.0,
    ):
        """
        Args:
            binding_residues: 1-indexed positions of drug-contacting residues.
            distance_threshold: Midpoint of sigmoid (in residue positions).
            sigmoid_steepness: Controls sharpness of transition.
        """
        self.binding_residues = set(binding_residues)
        self.distance_threshold = distance_threshold
        self.sigmoid_steepness = sigmoid_steepness

    def score_position(self, position: int) -> tuple[float, int]:
        """Score a single position's binding disruption potential.

        Args:
            position: 1-indexed residue position.

        Returns:
            (score, min_distance) where score is in [0, 1].
        """
        if position in self.binding_residues:
            return 1.0, 0

        # Minimum sequence distance to any binding residue
        min_dist = min(abs(position - br) for br in self.binding_residues)

        # Sigmoid: high score near binding site, low far away
        score = 1.0 / (1.0 + np.exp(self.sigmoid_steepness * (min_dist - self.distance_threshold)))

        return float(score), min_dist

    def score_mutations(
        self,
        mutations: list[str],
    ) -> list[tuple[str, float, int]]:
        """Score multiple mutations.

        Returns:
            List of (mutation, score, min_distance) tuples.
        """
        results = []
        for mut in mutations:
            _, pos, _ = parse_aa_mutation(mut)
            score, dist = self.score_position(pos)
            results.append((mut, score, dist))
        return results


class EmbeddingBindingPredictor:
    """Predict binding disruption from ESM-2 embedding changes.

    Computes the change in ESM-2 layer-33 representation at drug-binding
    residues when a mutation is introduced. Large changes at the binding
    site indicate structural/functional perturbation.

    This captures effects that sequence distance misses:
    - Allosteric mutations (far in sequence, close in 3D)
    - Mutations that alter protein dynamics rather than structure
    """

    def __init__(self, scorer: ESM2Scorer):
        self.scorer = scorer

    def compute_binding_site_embedding(
        self,
        protein_sequence: str,
        binding_residues: list[int],
    ) -> np.ndarray:
        """Extract mean ESM-2 representation at binding site residues.

        Args:
            protein_sequence: Full protein sequence.
            binding_residues: 1-indexed positions of binding site.

        Returns:
            Mean embedding vector at binding site [embed_dim].
        """
        repr_tensor = self.scorer.get_representations(protein_sequence)
        # repr_tensor shape: [seq_len, 1280]

        # Extract binding site residues (convert to 0-indexed)
        indices = [pos - 1 for pos in binding_residues
                   if 0 <= pos - 1 < repr_tensor.shape[0]]

        if not indices:
            return np.zeros(repr_tensor.shape[1])

        binding_repr = repr_tensor[indices].mean(dim=0)
        return binding_repr.cpu().numpy()

    def score_mutation(
        self,
        protein_sequence: str,
        mutation: str,
        binding_residues: list[int],
        wt_embedding: np.ndarray | None = None,
    ) -> float:
        """Score binding disruption from embedding change.

        Args:
            protein_sequence: Wildtype protein sequence.
            mutation: Mutation string like 'S450L'.
            binding_residues: 1-indexed binding site positions.
            wt_embedding: Pre-computed wildtype binding site embedding
                (avoids redundant forward pass when scoring many mutations).

        Returns:
            L2 distance between WT and mutant binding site embeddings.
        """
        if wt_embedding is None:
            wt_embedding = self.compute_binding_site_embedding(
                protein_sequence, binding_residues,
            )

        # Create mutant sequence
        ref_aa, pos, alt_aa = parse_aa_mutation(mutation)
        pos_idx = pos - 1
        mutant_seq = protein_sequence[:pos_idx] + alt_aa + protein_sequence[pos_idx + 1:]

        # Get mutant embedding at binding site
        mut_embedding = self.compute_binding_site_embedding(
            mutant_seq, binding_residues,
        )

        # L2 distance
        delta = np.linalg.norm(wt_embedding - mut_embedding)
        return float(delta)

    def score_mutations_batch(
        self,
        protein_sequence: str,
        mutations: list[str],
        binding_residues: list[int],
    ) -> list[tuple[str, float]]:
        """Score multiple mutations efficiently (shared WT embedding).

        Returns:
            List of (mutation, embedding_delta) tuples.
        """
        wt_embedding = self.compute_binding_site_embedding(
            protein_sequence, binding_residues,
        )

        results = []
        for i, mut in enumerate(mutations):
            delta = self.score_mutation(
                protein_sequence, mut, binding_residues,
                wt_embedding=wt_embedding,
            )
            results.append((mut, delta))

            if (i + 1) % 20 == 0:
                logger.info("  Scored %d/%d mutations (embedding)", i + 1, len(mutations))

        return results


def score_binding_disruption(
    mutations: list[str],
    binding_residues: list[int],
    protein_sequence: str | None = None,
    scorer: ESM2Scorer | None = None,
    alpha: float = 0.5,
    use_embedding: bool = True,
) -> list[BindingDisruptionScore]:
    """Combined structural + embedding binding disruption scoring.

    Args:
        mutations: List of mutation strings.
        binding_residues: 1-indexed drug-binding positions.
        protein_sequence: Required for embedding approach.
        scorer: Required for embedding approach.
        alpha: Weight for structural score (1-alpha for embedding).
        use_embedding: Whether to use embedding approach.

    Returns:
        List of BindingDisruptionScore, sorted by combined score (descending).
    """
    # Structural scores
    struct_pred = StructuralBindingPredictor(binding_residues)
    struct_results = {mut: (score, dist) for mut, score, dist
                      in struct_pred.score_mutations(mutations)}

    # Embedding scores (optional)
    emb_results = {}
    if use_embedding and protein_sequence and scorer:
        emb_pred = EmbeddingBindingPredictor(scorer)
        emb_raw = emb_pred.score_mutations_batch(
            protein_sequence, mutations, binding_residues,
        )
        # Normalize embedding scores to [0, 1]
        deltas = [d for _, d in emb_raw]
        max_delta = max(deltas) if deltas else 1.0
        emb_results = {mut: d / max(max_delta, 1e-8) for mut, d in emb_raw}

    # Combine
    results = []
    for mut in mutations:
        struct_score, min_dist = struct_results.get(mut, (0.0, 999))
        emb_score = emb_results.get(mut, 0.0) if use_embedding else 0.0

        if use_embedding:
            combined = alpha * struct_score + (1 - alpha) * emb_score
        else:
            combined = struct_score

        _, pos, _ = parse_aa_mutation(mut)
        results.append(BindingDisruptionScore(
            mutation=mut,
            position=pos,
            structural_score=round(struct_score, 4),
            embedding_score=round(emb_score, 4),
            combined_score=round(combined, 4),
            min_distance_to_binding=min_dist,
        ))

    results.sort(key=lambda r: -r.combined_score)
    return results
