"""Experiment 5: De novo diagnostic panel design for pipeline drugs.

The core pipeline:
    1. ESM-2 LLR landscape → fitness filter (which mutations are tolerated?)
    2. Binding disruption score → resistance filter (which fit mutations escape the drug?)
    3. Intersection → predicted diagnostic panel

Novel claim: design a diagnostic panel for a drug still in clinical trials,
using ONLY the target protein sequence + drug binding site — no clinical
resistance data needed.

Validation:
    - Leave-one-drug-out on existing drugs (recall, precision, F1 vs WHO)
    - BDQ retrospective (Rv0678: drug approved 2012, mutations catalogued 2022)
    - Pipeline predictions: BTZ043 (dprE1), telacebec (qcrB)

Usage:
    python -m experiments.denovo_design
    python -m experiments.denovo_design --leave-one-out
    python -m experiments.denovo_design --pipeline --drug BTZ043
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from data.drug_targets.targets import DRUG_TARGETS, PIPELINE_TARGETS, get_target
from data.who_catalogue.catalogue import get_mutations_by_gene, WHO_AA_MUTATIONS
from models.esm2_scorer import ESM2Scorer, parse_aa_mutation
from models.binding_disruption import (
    score_binding_disruption,
    StructuralBindingPredictor,
    BindingDisruptionScore,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/denovo")


def denovo_panel_design(
    protein_sequence: str,
    binding_residues: list[int],
    scorer: ESM2Scorer,
    k: int = 5,
    fitness_percentile: float = 25.0,
    alpha: float = 0.5,
    use_embedding: bool = False,
    hotspot_positions: list[int] | None = None,
    gene: str = "",
) -> dict:
    """Design a diagnostic panel from ESM-2 + binding disruption.

    Pipeline:
        1. Score full landscape at hotspot positions
        2. Fitness filter: keep mutations below fitness_percentile
        3. Resistance filter: score binding disruption
        4. Combined ranking
        5. Return top-k panel

    Args:
        protein_sequence: Target protein sequence.
        binding_residues: Drug-contacting positions (1-indexed).
        scorer: Pre-loaded ESM2Scorer.
        k: Panel size.
        fitness_percentile: Keep mutations with |LLR| below this percentile.
        alpha: Weight for structural binding score (vs embedding).
        use_embedding: Use ESM-2 embedding-based binding predictor.
        hotspot_positions: Positions to scan (None = binding residues ± 10).
        gene: Gene name for logging.

    Returns:
        Dict with panel mutations, scores, and metadata.
    """
    # Determine scan positions
    if hotspot_positions is None:
        # Scan binding residues ± 10 positions
        all_pos = set()
        for br in binding_residues:
            for offset in range(-10, 11):
                pos = br + offset
                if 1 <= pos <= len(protein_sequence):
                    all_pos.add(pos)
        hotspot_positions = sorted(all_pos)

    logger.info("Scanning %d positions for %s", len(hotspot_positions), gene)

    # Step 1: Full landscape scan
    landscape = scorer.score_full_landscape(
        protein_sequence, positions=hotspot_positions, gene=gene,
    )

    # Flatten to mutation list
    all_mutations = []
    for pos, aa_scores in landscape.items():
        wt = protein_sequence[pos - 1]
        for alt, llr in aa_scores.items():
            all_mutations.append({
                "mutation": f"{wt}{pos}{alt}",
                "position": pos,
                "wt": wt,
                "alt": alt,
                "llr": llr,
                "abs_llr": abs(llr),
            })

    # Step 2: Fitness filter
    # Two thresholds: strict for non-binding positions, relaxed for binding site
    binding_set = set(binding_residues)
    abs_llrs = [m["abs_llr"] for m in all_mutations]
    strict_threshold = np.percentile(abs_llrs, fitness_percentile)
    relaxed_threshold = np.percentile(abs_llrs, 75.0)  # top 75% for binding site

    fit_mutations = []
    for m in all_mutations:
        if m["position"] in binding_set:
            if m["abs_llr"] <= relaxed_threshold:
                fit_mutations.append(m)
        else:
            if m["abs_llr"] <= strict_threshold:
                fit_mutations.append(m)

    n_at_site = sum(1 for m in fit_mutations if m["position"] in binding_set)
    logger.info("Fitness filter: %d/%d pass (%d at binding site, %d flanking). "
                "Thresholds: site<=%.3f, flank<=%.3f",
                len(fit_mutations), len(all_mutations), n_at_site,
                len(fit_mutations) - n_at_site, relaxed_threshold, strict_threshold)

    # Step 3: Binding disruption scoring
    fit_mut_strings = [m["mutation"] for m in fit_mutations]
    binding_scores = score_binding_disruption(
        fit_mut_strings,
        binding_residues,
        protein_sequence=protein_sequence if use_embedding else None,
        scorer=scorer if use_embedding else None,
        alpha=alpha,
        use_embedding=use_embedding,
    )
    binding_lookup = {s.mutation: s for s in binding_scores}

    # Step 4: Two-tier ranking
    # Tier 1: Mutations AT binding residues — ranked by |LLR| ascending
    #         (these already have structural relevance; rank by fitness)
    # Tier 2: Mutations NEAR binding residues — ranked by combined score
    #
    # Rationale: mutations directly at drug-contacting positions are most
    # likely to confer resistance. Among those, the fittest survive.
    # Flanking mutations are secondary candidates.
    binding_set = set(binding_residues)

    for m in fit_mutations:
        bs = binding_lookup.get(m["mutation"])
        m["binding_score"] = bs.combined_score if bs else 0.0
        m["structural_score"] = bs.structural_score if bs else 0.0
        m["min_dist_to_binding"] = bs.min_distance_to_binding if bs else 999
        m["at_binding_site"] = m["position"] in binding_set

    # Tier 1: at binding site, sorted by |LLR| ascending (fittest first)
    tier1 = sorted(
        [m for m in fit_mutations if m["at_binding_site"]],
        key=lambda m: m["abs_llr"],
    )
    # Tier 2: near binding site, sorted by combined score
    tier2_pool = [m for m in fit_mutations if not m["at_binding_site"]]
    if tier2_pool:
        max_binding = max(m["binding_score"] for m in tier2_pool) or 1.0
        max_llr = max(m["abs_llr"] for m in tier2_pool) or 1.0
        for m in tier2_pool:
            binding_norm = m["binding_score"] / max_binding
            fitness_norm = 1.0 - (m["abs_llr"] / max_llr)
            m["combined_score"] = alpha * binding_norm + (1 - alpha) * fitness_norm
    tier2 = sorted(tier2_pool, key=lambda m: -m.get("combined_score", 0))

    # Assign combined scores for tier1 (binding_score=1.0 for at-site)
    for i, m in enumerate(tier1):
        m["combined_score"] = 1.0 - (i * 0.001)  # preserve ordering

    # Merge: tier 1 first, then tier 2
    fit_mutations = tier1 + tier2

    # Step 5: Top-k panel
    panel = fit_mutations[:k]

    return {
        "gene": gene,
        "n_positions_scanned": len(hotspot_positions),
        "n_total_mutations": len(all_mutations),
        "n_after_fitness_filter": len(fit_mutations),
        "fitness_threshold_llr": round(strict_threshold, 4),
        "fitness_percentile": fitness_percentile,
        "alpha": alpha,
        "panel_size": k,
        "panel": [
            {
                "mutation": m["mutation"],
                "abs_llr": round(m["abs_llr"], 4),
                "binding_score": round(m.get("binding_score", 0), 4),
                "combined_score": round(m.get("combined_score", 0), 4),
                "min_dist_to_binding": m.get("min_dist_to_binding", 999),
            }
            for m in panel
        ],
    }


def leave_one_drug_out(
    protein_sequences: dict[str, str],
    scorer: ESM2Scorer,
    k: int = 5,
    alpha: float = 0.5,
) -> dict:
    """Leave-one-drug-out validation on existing drugs.

    For each drug with known WHO resistance mutations, pretend it's new:
    design a panel using only sequence + binding site, then compare
    to the WHO catalogue.

    Returns dict with per-drug recall, precision, F1.
    """
    results = {}

    for target in DRUG_TARGETS:
        protein_seq = protein_sequences.get(target.key)
        if protein_seq is None:
            continue

        who_muts = get_mutations_by_gene(target.organism, target.gene)
        if len(who_muts) < 2:
            continue

        who_set = {m.mutation for m in who_muts}
        k_use = min(k, len(who_muts) * 2)  # allow 2x WHO panel size

        logger.info("\n--- Leave-one-out: %s %s (%s) ---", target.organism, target.gene, target.drug)

        panel_result = denovo_panel_design(
            protein_seq, target.binding_residues, scorer,
            k=k_use, gene=target.gene,
        )

        # Compare to WHO
        panel_set = {m["mutation"] for m in panel_result["panel"]}
        true_positives = panel_set & who_set
        recall = len(true_positives) / len(who_set) if who_set else 0
        precision = len(true_positives) / len(panel_set) if panel_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[target.key] = {
            "organism": target.organism,
            "gene": target.gene,
            "drug": target.drug,
            "mechanism": target.resistance_mechanism,
            "n_who_mutations": len(who_set),
            "panel_size": len(panel_set),
            "true_positives": sorted(true_positives),
            "false_positives": sorted(panel_set - who_set),
            "false_negatives": sorted(who_set - panel_set),
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "f1": round(f1, 3),
            "panel": panel_result["panel"],
        }

        logger.info(
            "  WHO: %d mutations | Panel: %d | TP=%d | Recall=%.2f | Precision=%.2f | F1=%.2f",
            len(who_set), len(panel_set), len(true_positives), recall, precision, f1,
        )
        if true_positives:
            logger.info("  Recovered: %s", ", ".join(sorted(true_positives)))

    return results


def pipeline_drug_predictions(
    protein_sequences: dict[str, str],
    scorer: ESM2Scorer,
    k: int = 10,
) -> dict:
    """Generate de novo panels for pipeline drugs (no clinical resistance data).

    These are the NOVEL predictions — the primary output of this module.
    """
    results = {}

    for target in PIPELINE_TARGETS:
        protein_seq = protein_sequences.get(target.key)
        if protein_seq is None:
            logger.warning("No protein sequence for %s — need to add to sequences.json", target.key)
            continue

        logger.info("\n=== PIPELINE PREDICTION: %s (%s) ===", target.drug, target.gene)
        logger.info("  Mechanism: %s", target.resistance_mechanism)
        logger.info("  Binding residues: %s", target.binding_residues[:10])
        logger.info("  PDB: %s", target.pdb_id)

        panel_result = denovo_panel_design(
            protein_seq, target.binding_residues, scorer,
            k=k, gene=target.gene,
        )

        results[target.key] = {
            "organism": target.organism,
            "gene": target.gene,
            "drug": target.drug,
            "drug_class": target.drug_class,
            "pdb_id": target.pdb_id,
            "approval_status": "Phase 2",
            "prediction": panel_result,
            "note": "NO clinical resistance data used — purely predictive",
        }

        logger.info("  Predicted panel (%d mutations):", len(panel_result["panel"]))
        for m in panel_result["panel"]:
            logger.info(
                "    %s: |LLR|=%.3f, binding=%.3f, combined=%.3f",
                m["mutation"], m["abs_llr"], m["binding_score"], m["combined_score"],
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="De novo diagnostic panel design")
    parser.add_argument("--leave-one-out", action="store_true",
                        help="Validate on existing drugs")
    parser.add_argument("--pipeline", action="store_true",
                        help="Predict for pipeline drugs")
    parser.add_argument("--drug", type=str, default=None,
                        help="Specific pipeline drug (e.g., BTZ043)")
    parser.add_argument("--panel-size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for structural binding score")
    parser.add_argument("--protein-sequences", type=Path,
                        default=Path("data/protein_sequences/sequences.json"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump({
            "experiment": "denovo_design",
            "timestamp": datetime.now().isoformat(),
            "panel_size": args.panel_size,
            "alpha": args.alpha,
        }, f, indent=2)

    with open(args.protein_sequences) as f:
        protein_sequences = json.load(f)

    scorer = ESM2Scorer()

    # Leave-one-drug-out validation
    if args.leave_one_out or (not args.pipeline):
        logger.info("=" * 60)
        logger.info("LEAVE-ONE-DRUG-OUT VALIDATION")
        logger.info("=" * 60)

        loo_results = leave_one_drug_out(
            protein_sequences, scorer,
            k=args.panel_size, alpha=args.alpha,
        )

        with open(RESULTS_DIR / "leave_one_out.json", "w") as f:
            json.dump(loo_results, f, indent=2)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("LEAVE-ONE-OUT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"{'Target':>25} {'Mechanism':>25} {'Recall':>8} {'Precision':>10} {'F1':>6}")
        logger.info("-" * 80)

        recalls = []
        for key, r in sorted(loo_results.items()):
            logger.info(
                f"{key:>25} {r['mechanism']:>25} {r['recall']:>8.2f} {r['precision']:>10.2f} {r['f1']:>6.2f}"
            )
            recalls.append(r["recall"])

        if recalls:
            logger.info("-" * 80)
            logger.info(f"{'Mean':>25} {'':>25} {np.mean(recalls):>8.2f}")

    # Pipeline predictions
    if args.pipeline:
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE DRUG PREDICTIONS (no clinical data used)")
        logger.info("=" * 60)

        pipeline_results = pipeline_drug_predictions(
            protein_sequences, scorer, k=args.panel_size,
        )

        with open(RESULTS_DIR / "pipeline_predictions.json", "w") as f:
            json.dump(pipeline_results, f, indent=2)

    scorer.cleanup()


if __name__ == "__main__":
    main()
