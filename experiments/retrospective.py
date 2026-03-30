"""Experiment 1: Retrospective validation — |ESM-2 LLR| vs clinical prevalence.

This is the validation experiment. It confirms that ESM-2 masked marginal
LLR captures evolutionary constraint that correlates with clinical prevalence
of AMR mutations. This is a KNOWN relationship (Meier et al. 2021, Brandes
et al. 2023) — the purpose here is to validate our pipeline and establish
baselines before the novel experiments.

Key hypothesis:
    Clinically prevalent AMR mutations have LOW |ESM-2 LLR| (conservative
    substitutions preserving protein function). Rare mutations have HIGH
    |LLR| (disruptive, high fitness cost).

    Expected: negative Spearman correlation between |LLR| and prevalence.

Analysis:
    1. Per-organism Spearman rho (within-species consistency)
    2. Per-gene Spearman rho (within-gene resolution)
    3. Pooled cross-species correlation (universal signal)
    4. Permutation test for robust p-values
    5. Partial correlation controlling for gene identity

Usage:
    python -m experiments.retrospective
    python -m experiments.retrospective --analysis-only  # use cached LLR
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from data.who_catalogue.catalogue import (
    WHO_AA_MUTATIONS,
    get_mutations_by_organism,
    get_unique_genes,
)
from models.esm2_scorer import ESM2Scorer, LLRResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/retrospective")


def compute_all_llr(
    protein_sequences: dict[str, str],
    scorer: ESM2Scorer | None = None,
) -> list[dict]:
    """Compute ESM-2 LLR for all WHO catalogue AA substitutions.

    Groups mutations by gene to reuse forward passes for same-position mutations.
    """
    if scorer is None:
        scorer = ESM2Scorer()

    results = []
    for org, gene in get_unique_genes():
        key = f"{org}_{gene}"
        protein_seq = protein_sequences.get(key)

        from data.who_catalogue.catalogue import get_mutations_by_gene
        gene_muts = get_mutations_by_gene(org, gene)

        if protein_seq is None:
            logger.warning("No protein sequence for %s, skipping %d mutations", key, len(gene_muts))
            for m in gene_muts:
                results.append({
                    "organism": m.organism, "gene": m.gene,
                    "mutation": m.mutation, "drug": m.drug,
                    "prevalence_pct": m.prevalence_pct,
                    "who_tier": m.who_tier,
                    "esm2_llr": None, "abs_esm2_llr": None,
                    "status": "no_protein",
                })
            continue

        # Score all mutations for this gene in one batch
        mutations = [m.mutation for m in gene_muts]
        llr_results = scorer.score_batch(protein_seq, mutations, gene=gene)

        for m, lr in zip(gene_muts, llr_results):
            results.append({
                "organism": m.organism, "gene": m.gene,
                "mutation": m.mutation, "drug": m.drug,
                "prevalence_pct": m.prevalence_pct,
                "who_tier": m.who_tier,
                "esm2_llr": lr.llr,
                "abs_esm2_llr": lr.abs_llr,
                "ref_log_prob": lr.ref_log_prob,
                "alt_log_prob": lr.alt_log_prob,
                "ref_aa_actual": lr.ref_aa_actual,
                "status": lr.status,
            })

    computed = [r for r in results if r["status"] == "computed"]
    logger.info("Computed LLR for %d / %d mutations", len(computed), len(results))
    return results


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Permutation test for Spearman correlation.

    More robust than parametric p-value for small N.
    Returns observed rho and permutation p-value.
    """
    rng = np.random.default_rng(seed)
    observed_rho = spearmanr(x, y).statistic
    count = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_rho = spearmanr(x, y_perm).statistic
        if abs(perm_rho) >= abs(observed_rho):
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return observed_rho, p_value


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for Spearman rho."""
    rng = np.random.default_rng(seed)
    rhos = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rhos.append(spearmanr(x[idx], y[idx]).statistic)
    alpha = (1 - ci) / 2
    return np.quantile(rhos, alpha), np.quantile(rhos, 1 - alpha)


def correlation_analysis(results: list[dict]) -> dict:
    """Full statistical analysis of |LLR| vs prevalence."""
    computed = [r for r in results if r.get("esm2_llr") is not None]
    analysis = {"per_organism": {}, "per_gene": {}, "pooled": None}

    # Per organism
    by_org: dict[str, list] = defaultdict(list)
    for r in computed:
        by_org[r["organism"]].append(r)

    for org_id, targets in sorted(by_org.items()):
        x = np.array([abs(t["esm2_llr"]) for t in targets])
        y = np.array([t["prevalence_pct"] for t in targets])

        if len(x) >= 5:
            rho, perm_p = permutation_test(x, y)
            ci_lo, ci_hi = bootstrap_ci(x, y)
            analysis["per_organism"][org_id] = {
                "n": len(x),
                "spearman_rho": round(float(rho), 4),
                "permutation_p": round(float(perm_p), 4),
                "ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
                "significant": perm_p < 0.05,
            }
            logger.info(
                "  %s (N=%d): rho=%.3f, p=%.4f [%.3f, %.3f] %s",
                org_id, len(x), rho, perm_p, ci_lo, ci_hi,
                "*" if perm_p < 0.05 else "",
            )

    # Per gene (within organism)
    by_gene: dict[str, list] = defaultdict(list)
    for r in computed:
        by_gene[f"{r['organism']}_{r['gene']}"].append(r)

    for gene_key, targets in sorted(by_gene.items()):
        if len(targets) >= 4:
            x = np.array([abs(t["esm2_llr"]) for t in targets])
            y = np.array([t["prevalence_pct"] for t in targets])
            rho, p = spearmanr(x, y)
            analysis["per_gene"][gene_key] = {
                "n": len(targets),
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p), 4),
            }
            logger.info("  %s (N=%d): rho=%.3f, p=%.4f", gene_key, len(targets), rho, p)

    # Pooled
    x_all = np.array([abs(r["esm2_llr"]) for r in computed])
    y_all = np.array([r["prevalence_pct"] for r in computed])
    if len(x_all) >= 10:
        rho, perm_p = permutation_test(x_all, y_all)
        ci_lo, ci_hi = bootstrap_ci(x_all, y_all)
        analysis["pooled"] = {
            "n": len(x_all),
            "spearman_rho": round(float(rho), 4),
            "permutation_p": round(float(perm_p), 4),
            "ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        }
        logger.info(
            "\n  POOLED (N=%d): rho=%.3f, p=%.4f [%.3f, %.3f]",
            len(x_all), rho, perm_p, ci_lo, ci_hi,
        )

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Retrospective: |ESM-2 LLR| vs prevalence")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Use cached LLR results, skip computation")
    parser.add_argument("--protein-sequences", type=Path,
                        default=Path("data/protein_sequences/sequences.json"),
                        help="Path to protein sequences JSON")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "experiment": "retrospective_validation",
        "timestamp": datetime.now().isoformat(),
        "model": "esm2_t33_650M_UR50D",
        "n_mutations": len(WHO_AA_MUTATIONS),
        "analysis_only": args.analysis_only,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.analysis_only:
        cache_path = RESULTS_DIR / "llr_results.csv"
        if not cache_path.exists():
            raise FileNotFoundError(f"No cached results at {cache_path}")
        with open(cache_path) as f:
            results = list(csv.DictReader(f))
        for r in results:
            if r.get("esm2_llr") and r["esm2_llr"] != "":
                r["esm2_llr"] = float(r["esm2_llr"])
                r["abs_esm2_llr"] = abs(r["esm2_llr"])
                r["prevalence_pct"] = float(r["prevalence_pct"])
            else:
                r["esm2_llr"] = None
    else:
        if not args.protein_sequences.exists():
            logger.error(
                "Protein sequences not found at %s. "
                "Provide sequences.json with format: {\"organism_gene\": \"MSEQ...\"}",
                args.protein_sequences,
            )
            return

        with open(args.protein_sequences) as f:
            protein_sequences = json.load(f)
        logger.info("Loaded %d protein sequences", len(protein_sequences))

        scorer = ESM2Scorer()
        results = compute_all_llr(protein_sequences, scorer)
        scorer.cleanup()

        # Save raw results
        computed = [r for r in results if r["status"] == "computed"]
        if computed:
            out_path = RESULTS_DIR / "llr_results.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=computed[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logger.info("Saved LLR results to %s", out_path)

    # Correlation analysis
    logger.info("\n" + "=" * 60)
    logger.info("CORRELATION: |ESM-2 LLR| vs CLINICAL PREVALENCE")
    logger.info("=" * 60)
    analysis = correlation_analysis(results)

    with open(RESULTS_DIR / "correlation_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Saved analysis to %s", RESULTS_DIR / "correlation_analysis.json")

    # Summary
    computed = [r for r in results if r.get("esm2_llr") is not None]
    if computed:
        llrs = [r["esm2_llr"] for r in computed]
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info("Computed: %d / %d mutations", len(computed), len(results))
        logger.info("|LLR| range: [%.3f, %.3f]", min(abs(l) for l in llrs), max(abs(l) for l in llrs))
        logger.info("|LLR| median: %.3f", np.median([abs(l) for l in llrs]))
        neg = sum(1 for l in llrs if l < 0)
        logger.info("LLR < 0 (WT preferred): %d/%d (%.0f%%)", neg, len(llrs), 100 * neg / len(llrs))


if __name__ == "__main__":
    main()
