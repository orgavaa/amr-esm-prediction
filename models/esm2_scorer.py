"""ESM-2 masked marginal log-likelihood ratio scorer.

Computes the evolutionary fitness cost of amino acid substitutions using
ESM-2's masked marginal probability:

    LLR = log P(alt_aa | context) - log P(ref_aa | context)

where context = full protein sequence with the target position masked.

Key insight: LLR < 0 means the wildtype is preferred by evolution.
|LLR| magnitude reflects the fitness cost of the substitution.
Clinically prevalent AMR mutations have LOW |LLR| (conservative),
rare mutations have HIGH |LLR| (disruptive).

References:
    Meier et al. (2021) "Language models enable zero-shot prediction of
    the effects of mutations on protein function." NeurIPS.
    Brandes et al. (2023) "Genome-wide prediction of disease variant
    effects with a deep protein language model." Nature Genetics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LLRResult:
    """Result of a single ESM-2 LLR computation."""

    gene: str
    mutation: str
    ref_aa: str
    position: int
    alt_aa: str
    llr: float
    abs_llr: float
    ref_log_prob: float
    alt_log_prob: float
    ref_aa_actual: str  # What was actually at that position in the sequence
    status: str  # "computed", "no_protein", "position_out_of_range"

    @property
    def label(self) -> str:
        return f"{self.gene}_{self.mutation}"


def parse_aa_mutation(mut_str: str) -> tuple[str, int, str]:
    """Parse amino acid substitution string.

    Args:
        mut_str: Mutation in format like 'S450L' (ref_aa + position + alt_aa)

    Returns:
        (ref_aa, position, alt_aa) — position is 1-indexed.
    """
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut_str)
    if not m:
        raise ValueError(f"Cannot parse AA mutation: {mut_str!r}")
    return m.group(1), int(m.group(2)), m.group(3)


class ESM2Scorer:
    """ESM-2 masked marginal scorer for protein fitness prediction.

    Loads the model once and scores multiple mutations efficiently.
    Each mutation requires one forward pass (~0.5s on RTX 2070 for 650M model).

    Usage:
        scorer = ESM2Scorer()
        result = scorer.score_mutation(protein_seq, "S450L")
        results = scorer.score_batch(protein_seq, ["S450L", "H445Y", "D435V"])
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model, self.alphabet, self.batch_converter = self._load_model()

    def _resolve_device(self, device: str | None) -> torch.device:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _load_model(self):
        try:
            import esm
        except ImportError:
            raise ImportError(
                "ESM-2 not installed. Install via: pip install fair-esm"
            )

        logger.info("Loading %s...", self.model_name)
        load_fn = getattr(esm.pretrained, self.model_name)
        model, alphabet = load_fn()
        batch_converter = alphabet.get_batch_converter()
        model = model.to(self.device).eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "%s loaded on %s (%.0fM params)",
            self.model_name, self.device, n_params / 1e6,
        )
        return model, alphabet, batch_converter

    def score_mutation(
        self,
        protein_sequence: str,
        mutation: str,
        gene: str = "",
    ) -> LLRResult:
        """Score a single amino acid substitution.

        Args:
            protein_sequence: Full wildtype protein sequence.
            mutation: Substitution string like 'S450L'.
            gene: Gene name (for labeling only).

        Returns:
            LLRResult with LLR and per-AA log-probabilities.
        """
        ref_aa, position, alt_aa = parse_aa_mutation(mutation)
        pos_idx = position - 1

        if pos_idx < 0 or pos_idx >= len(protein_sequence):
            return LLRResult(
                gene=gene, mutation=mutation,
                ref_aa=ref_aa, position=position, alt_aa=alt_aa,
                llr=float("nan"), abs_llr=float("nan"),
                ref_log_prob=float("nan"), alt_log_prob=float("nan"),
                ref_aa_actual="", status="position_out_of_range",
            )

        actual_aa = protein_sequence[pos_idx]
        if actual_aa != ref_aa:
            logger.debug(
                "Ref mismatch at %s pos %d: expected %s, found %s",
                gene, position, ref_aa, actual_aa,
            )

        # Mask target position and get log-probs
        masked_seq = (
            protein_sequence[:pos_idx] + "<mask>" + protein_sequence[pos_idx + 1:]
        )
        _, _, tokens = self.batch_converter([("protein", masked_seq)])
        tokens = tokens.to(self.device)

        with torch.no_grad():
            logits = self.model(tokens, repr_layers=[], return_contacts=False)["logits"]

        # Token position = pos_idx + 1 (CLS token at index 0)
        log_probs = F.log_softmax(logits[0, pos_idx + 1], dim=-1)
        ref_lp = log_probs[self.alphabet.get_idx(ref_aa)].item()
        alt_lp = log_probs[self.alphabet.get_idx(alt_aa)].item()
        llr = alt_lp - ref_lp

        return LLRResult(
            gene=gene, mutation=mutation,
            ref_aa=ref_aa, position=position, alt_aa=alt_aa,
            llr=round(llr, 6), abs_llr=round(abs(llr), 6),
            ref_log_prob=round(ref_lp, 6), alt_log_prob=round(alt_lp, 6),
            ref_aa_actual=actual_aa, status="computed",
        )

    def score_batch(
        self,
        protein_sequence: str,
        mutations: list[str],
        gene: str = "",
    ) -> list[LLRResult]:
        """Score multiple mutations on the same protein.

        Shares a single masked forward pass per position when multiple
        mutations target the same position (e.g., S450L, S450W, S450F).

        Args:
            protein_sequence: Full wildtype protein sequence.
            mutations: List of substitution strings like ['S450L', 'S450W'].
            gene: Gene name (for labeling).

        Returns:
            List of LLRResult, one per mutation, in INPUT ORDER.
        """
        # Group mutations by position for efficiency, tracking input index
        by_position: dict[int, list[tuple[int, str, str, str]]] = {}
        for idx, mut in enumerate(mutations):
            ref_aa, position, alt_aa = parse_aa_mutation(mut)
            by_position.setdefault(position, []).append((idx, mut, ref_aa, alt_aa))

        # Build results indexed by input position
        results: list[tuple[int, LLRResult]] = []
        for position, muts in sorted(by_position.items()):
            pos_idx = position - 1

            if pos_idx < 0 or pos_idx >= len(protein_sequence):
                for idx, mut, ref_aa, alt_aa in muts:
                    results.append((idx, LLRResult(
                        gene=gene, mutation=mut,
                        ref_aa=ref_aa, position=position, alt_aa=alt_aa,
                        llr=float("nan"), abs_llr=float("nan"),
                        ref_log_prob=float("nan"), alt_log_prob=float("nan"),
                        ref_aa_actual="", status="position_out_of_range",
                    )))
                continue

            actual_aa = protein_sequence[pos_idx]

            # One forward pass per position
            masked_seq = (
                protein_sequence[:pos_idx] + "<mask>" + protein_sequence[pos_idx + 1:]
            )
            _, _, tokens = self.batch_converter([("protein", masked_seq)])
            tokens = tokens.to(self.device)

            with torch.no_grad():
                logits = self.model(
                    tokens, repr_layers=[], return_contacts=False
                )["logits"]

            log_probs = F.log_softmax(logits[0, pos_idx + 1], dim=-1)

            for idx, mut, ref_aa, alt_aa in muts:
                ref_lp = log_probs[self.alphabet.get_idx(ref_aa)].item()
                alt_lp = log_probs[self.alphabet.get_idx(alt_aa)].item()
                llr = alt_lp - ref_lp

                results.append((idx, LLRResult(
                    gene=gene, mutation=mut,
                    ref_aa=ref_aa, position=position, alt_aa=alt_aa,
                    llr=round(llr, 6), abs_llr=round(abs(llr), 6),
                    ref_log_prob=round(ref_lp, 6), alt_log_prob=round(alt_lp, 6),
                    ref_aa_actual=actual_aa, status="computed",
                )))

        # Return in input order
        results.sort(key=lambda t: t[0])
        return [r for _, r in results]

    def score_full_landscape(
        self,
        protein_sequence: str,
        positions: list[int] | None = None,
        gene: str = "",
    ) -> dict[int, dict[str, float]]:
        """Score ALL possible substitutions at given positions.

        This is the core of pre-emptive prediction: for each position,
        compute LLR for all 19 non-wildtype amino acids. The resulting
        landscape predicts which resistance mutations are evolutionarily
        accessible BEFORE any clinical data exists.

        Args:
            protein_sequence: Full wildtype protein sequence.
            positions: 1-indexed positions to scan. If None, scans entire protein.
            gene: Gene name (for logging).

        Returns:
            Dict mapping position -> {alt_aa: LLR} for all 19 substitutions.
        """
        AA = "ACDEFGHIKLMNPQRSTVWY"

        if positions is None:
            positions = list(range(1, len(protein_sequence) + 1))

        landscape = {}
        for i, position in enumerate(positions):
            pos_idx = position - 1
            if pos_idx < 0 or pos_idx >= len(protein_sequence):
                continue

            wt_aa = protein_sequence[pos_idx]

            # Mask and forward pass
            masked_seq = (
                protein_sequence[:pos_idx] + "<mask>" + protein_sequence[pos_idx + 1:]
            )
            _, _, tokens = self.batch_converter([("protein", masked_seq)])
            tokens = tokens.to(self.device)

            with torch.no_grad():
                logits = self.model(
                    tokens, repr_layers=[], return_contacts=False
                )["logits"]

            log_probs = F.log_softmax(logits[0, pos_idx + 1], dim=-1)
            wt_lp = log_probs[self.alphabet.get_idx(wt_aa)].item()

            position_landscape = {}
            for aa in AA:
                if aa == wt_aa:
                    continue
                alt_lp = log_probs[self.alphabet.get_idx(aa)].item()
                position_landscape[aa] = round(alt_lp - wt_lp, 6)

            landscape[position] = position_landscape

            if (i + 1) % 50 == 0:
                logger.info(
                    "  [%s] Scanned %d/%d positions", gene, i + 1, len(positions)
                )

        return landscape

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()
