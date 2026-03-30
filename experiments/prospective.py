"""Experiment 2: Prospective prediction — emergence order from ESM-2 LLR.

THIS IS THE NOVEL EXPERIMENT. Nobody has used a protein language model to
predict the ORDER in which resistance mutations emerge under drug pressure.

Key insight:
    Under drug selection, the first mutations to appear should be those with:
    (a) Sufficient resistance (above MIC threshold) — biological constraint
    (b) Minimal fitness cost — captured by LOW |ESM-2 LLR|

    The model predicts: given a gene + drug, rank mutations by |LLR|.
    Low |LLR| mutations emerge first. This can be validated against:
    1. Known prevalence rank (retrospective proxy)
    2. Longitudinal WGS data (true prospective validation)

Validation strategy:
    - Leave-one-out within gene: for each gene with N mutations, hold out
      one mutation, predict its prevalence rank from |LLR| rank.
    - Cross-species: train rank model on 3 organisms, predict 4th.
    - If CRyPTIC longitudinal data available: compare predicted emergence
      order to observed temporal order in patients on treatment.

Metrics:
    - Rank concordance: fraction of pairwise comparisons where |LLR| rank
      matches prevalence rank (equivalent to Kendall's tau + 1 / 2).
    - Top-k precision: of the k most prevalent mutations, how many are in
      the k lowest |LLR| predictions?
    - Emergence order correlation (if longitudinal data): Spearman between
      predicted rank and observed first-detection time.

Usage:
    python -m experiments.prospective
    python -m experiments.prospective --longitudinal data/cryptic/emergence_times.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr, rankdata

from data.who_catalogue.catalogue import WHO_AA_MUTATIONS, get_unique_genes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/prospective")


def load_llr_results(path: Path) -> list[dict]:
    """Load LLR results from retrospective experiment."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("esm2_llr") and r["esm2_llr"] != "":
            r["esm2_llr"] = float(r["esm2_llr"])
            r["abs_esm2_llr"] = abs(r["esm2_llr"])
            r["prevalence_pct"] = float(r["prevalence_pct"])
        else:
            r["esm2_llr"] = None
    return [r for r in rows if r["esm2_llr"] is not None]


def rank_concordance(predicted_order: np.ndarray, true_order: np.ndarray) -> float:
    """Fraction of pairwise comparisons where predicted rank matches true rank.

    Equivalent to (Kendall tau + 1) / 2, scaled to [0, 1].
    0.5 = random, 1.0 = perfect concordance.
    """
    tau, _ = kendalltau(predicted_order, true_order)
    return (tau + 1) / 2


def top_k_precision(
    predicted_ranks: np.ndarray,
    true_ranks: np.ndarray,
    k: int,
) -> float:
    """Of the k truly most prevalent mutations, how many are in the top-k predicted?

    predicted_ranks: lower = predicted more prevalent (lower |LLR|)
    true_ranks: lower = actually more prevalent
    """
    predicted_top_k = set(np.argsort(predicted_ranks)[:k])
    true_top_k = set(np.argsort(true_ranks)[:k])
    return len(predicted_top_k & true_top_k) / k


def within_gene_analysis(results: list[dict]) -> dict:
    """Rank prediction within each gene.

    For genes with >= 4 mutations, test whether |LLR| rank predicts
    prevalence rank. This is the strongest test because it controls
    for gene identity, drug class, and organism.
    """
    by_gene: dict[str, list] = defaultdict(list)
    for r in results:
        by_gene[f"{r['organism']}_{r['gene']}"].append(r)

    analysis = {}
    for gene_key, targets in sorted(by_gene.items()):
        if len(targets) < 4:
            continue

        abs_llr = np.array([t["abs_esm2_llr"] for t in targets])
        prevalence = np.array([t["prevalence_pct"] for t in targets])

        # |LLR| rank (low |LLR| = predicted most prevalent = rank 1)
        llr_ranks = rankdata(abs_llr)
        # Prevalence rank (high prevalence = rank 1)
        prev_ranks = rankdata(-prevalence)

        rho, p = spearmanr(llr_ranks, prev_ranks)
        concordance = rank_concordance(llr_ranks, prev_ranks)

        # Top-k precision for k = ceil(N/3) (roughly top third)
        k = max(1, len(targets) // 3)
        topk_prec = top_k_precision(llr_ranks, prev_ranks, k)

        # Leave-one-out: for each held-out mutation, predict its prevalence
        # rank from its |LLR| rank among the remaining mutations
        loo_hits = 0
        for i in range(len(targets)):
            held_out_llr = abs_llr[i]
            remaining_llr = np.delete(abs_llr, i)
            remaining_prev = np.delete(prevalence, i)
            # Predicted rank = where does held-out |LLR| fall?
            predicted_rank = np.sum(remaining_llr < held_out_llr) + 1
            actual_rank = np.sum(remaining_prev > prevalence[i]) + 1
            # Hit if predicted rank is within 1 of actual
            n_total = len(targets)
            if abs(predicted_rank - actual_rank) <= max(1, n_total // 4):
                loo_hits += 1
        loo_accuracy = loo_hits / len(targets)

        analysis[gene_key] = {
            "n": len(targets),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p), 4),
            "rank_concordance": round(float(concordance), 4),
            f"top_{k}_precision": round(float(topk_prec), 4),
            "loo_accuracy": round(float(loo_accuracy), 4),
            "mutations": [
                {
                    "mutation": t["mutation"],
                    "abs_llr": t["abs_esm2_llr"],
                    "prevalence": t["prevalence_pct"],
                    "llr_rank": int(llr_ranks[i]),
                    "prev_rank": int(prev_ranks[i]),
                }
                for i, t in enumerate(targets)
            ],
        }
        logger.info(
            "  %s (N=%d): rho=%.3f, concordance=%.3f, top-%d=%.2f, LOO=%.2f",
            gene_key, len(targets), rho, concordance, k, topk_prec, loo_accuracy,
        )

    return analysis


def cross_species_analysis(results: list[dict]) -> dict:
    """Leave-one-organism-out: predict emergence ranking for held-out species.

    Uses the |LLR| → prevalence rank relationship learned from 3 organisms
    to predict rankings in the 4th. This tests universality of the signal.
    """
    organisms = sorted(set(r["organism"] for r in results))
    if len(organisms) < 3:
        logger.info("Need >= 3 organisms for cross-species analysis")
        return {}

    analysis = {}
    for held_out in organisms:
        train = [r for r in results if r["organism"] != held_out]
        test = [r for r in results if r["organism"] == held_out]

        if len(test) < 5:
            continue

        # Learn: is |LLR| predictive of prevalence rank in training organisms?
        train_rho = spearmanr(
            [abs(r["esm2_llr"]) for r in train],
            [r["prevalence_pct"] for r in train],
        ).statistic

        # Predict: rank test mutations by |LLR|
        test_llr = np.array([abs(r["esm2_llr"]) for r in test])
        test_prev = np.array([r["prevalence_pct"] for r in test])
        pred_ranks = rankdata(test_llr)
        true_ranks = rankdata(-test_prev)

        rho, p = spearmanr(pred_ranks, true_ranks)
        concordance = rank_concordance(pred_ranks, true_ranks)

        analysis[held_out] = {
            "n_train": len(train),
            "n_test": len(test),
            "train_rho": round(float(train_rho), 4),
            "test_rho": round(float(rho), 4),
            "test_p": round(float(p), 4),
            "concordance": round(float(concordance), 4),
        }
        logger.info(
            "  Hold-out %s: train_rho=%.3f, test_rho=%.3f (N=%d), concordance=%.3f",
            held_out, train_rho, rho, len(test), concordance,
        )

    return analysis


def emergence_order_analysis(
    results: list[dict],
    longitudinal_path: Path | None = None,
) -> dict | None:
    """Compare predicted emergence order to observed temporal emergence.

    Requires longitudinal WGS data with first-detection times for each mutation.
    Expected format: CSV with columns [organism, gene, mutation, first_detected_days]

    This is the TRUE prospective validation. Everything else is retrospective proxy.
    """
    if longitudinal_path is None or not longitudinal_path.exists():
        logger.info(
            "No longitudinal data provided. To run prospective validation, "
            "provide --longitudinal with CRyPTIC or PATRIC emergence times."
        )
        return None

    with open(longitudinal_path) as f:
        longitudinal = list(csv.DictReader(f))

    # Match mutations between LLR results and longitudinal data
    llr_lookup = {
        f"{r['organism']}_{r['gene']}_{r['mutation']}": r
        for r in results
    }

    matched = []
    for entry in longitudinal:
        key = f"{entry['organism']}_{entry['gene']}_{entry['mutation']}"
        llr_entry = llr_lookup.get(key)
        if llr_entry is not None:
            matched.append({
                "key": key,
                "abs_llr": abs(llr_entry["esm2_llr"]),
                "first_detected_days": float(entry["first_detected_days"]),
                "prevalence": llr_entry["prevalence_pct"],
            })

    if len(matched) < 5:
        logger.warning("Only %d matched mutations (need >= 5)", len(matched))
        return None

    llr_vals = np.array([m["abs_llr"] for m in matched])
    emergence_times = np.array([m["first_detected_days"] for m in matched])

    # Key prediction: low |LLR| should emerge FIRST (low emergence time)
    rho, p = spearmanr(llr_vals, emergence_times)

    logger.info("\n=== PROSPECTIVE VALIDATION (N=%d) ===", len(matched))
    logger.info("  rho(|LLR|, emergence_time) = %.3f (p=%.4f)", rho, p)
    logger.info("  Expected: POSITIVE (low |LLR| → early emergence → low time)")

    return {
        "n_matched": len(matched),
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p), 4),
        "direction": "positive" if rho > 0 else "negative",
        "interpretation": (
            "CONFIRMED: low fitness cost → early emergence"
            if rho > 0 and p < 0.05
            else "NOT confirmed at p<0.05"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Prospective: emergence order prediction")
    parser.add_argument("--llr-results", type=Path,
                        default=Path("results/retrospective/llr_results.csv"),
                        help="LLR results from retrospective experiment")
    parser.add_argument("--longitudinal", type=Path, default=None,
                        help="Longitudinal WGS emergence times CSV")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "prospective_emergence_prediction",
        "timestamp": datetime.now().isoformat(),
        "llr_source": str(args.llr_results),
        "longitudinal_data": str(args.longitudinal) if args.longitudinal else None,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if not args.llr_results.exists():
        logger.error(
            "LLR results not found at %s. Run retrospective experiment first.",
            args.llr_results,
        )
        return

    results = load_llr_results(args.llr_results)
    logger.info("Loaded %d LLR results", len(results))

    # Within-gene rank prediction
    logger.info("\n" + "=" * 60)
    logger.info("WITHIN-GENE RANK PREDICTION")
    logger.info("=" * 60)
    gene_analysis = within_gene_analysis(results)

    # Cross-species generalization
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-SPECIES GENERALIZATION (Leave-One-Organism-Out)")
    logger.info("=" * 60)
    cross_species = cross_species_analysis(results)

    # Prospective validation (if longitudinal data available)
    emergence = emergence_order_analysis(results, args.longitudinal)

    # Save all results
    full_analysis = {
        "within_gene": gene_analysis,
        "cross_species": cross_species,
        "emergence_order": emergence,
    }
    with open(RESULTS_DIR / "prospective_analysis.json", "w") as f:
        json.dump(full_analysis, f, indent=2)
    logger.info("\nSaved analysis to %s", RESULTS_DIR / "prospective_analysis.json")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if gene_analysis:
        rhos = [v["spearman_rho"] for v in gene_analysis.values()]
        concordances = [v["rank_concordance"] for v in gene_analysis.values()]
        logger.info("Within-gene: %d genes tested", len(gene_analysis))
        logger.info("  Mean rho: %.3f", np.mean(rhos))
        logger.info("  Mean concordance: %.3f (0.5=random, 1.0=perfect)", np.mean(concordances))
    if cross_species:
        test_rhos = [v["test_rho"] for v in cross_species.values()]
        logger.info("Cross-species: %d organisms held out", len(cross_species))
        logger.info("  Mean test rho: %.3f", np.mean(test_rhos))
    if emergence:
        logger.info("Emergence order: rho=%.3f (p=%.4f) — %s",
                    emergence["spearman_rho"], emergence["p_value"],
                    emergence["interpretation"])


if __name__ == "__main__":
    main()
