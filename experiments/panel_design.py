"""Experiment 3: Pre-emptive diagnostic panel design from ESM-2 LLR alone.

THIS IS THE CLINICAL APPLICATION. The novel claim:
    "We can design a diagnostic panel for a pathogen we've never surveilled
    by ranking mutations by |ESM-2 LLR| alone."

Currently, diagnostic panels (like Xpert MTB/RIF, WHO-recommended LPA) are
designed by surveilling patient populations for years, tabulating mutation
frequencies, and selecting the most prevalent. This requires:
    - Large-scale WGS studies (CRyPTIC: 10,000+ isolates, 5+ years)
    - Geographic-specific data (prevalence varies by region)
    - Repeated updates as mutation landscape shifts

ESM-2 LLR provides a PHYSICS-BASED PRIOR that is:
    - Available before any clinical data exists
    - Stable across geographies (protein fitness is universal)
    - Computable in seconds (one forward pass per position)

Experiment design:
    1. For each drug target gene, score ALL 19 possible substitutions at
       known resistance hotspot positions using ESM-2 LLR.
    2. Rank by |LLR| (ascending) — low fitness cost mutations are predicted
       to be most prevalent.
    3. Evaluate: does the LLR-only panel achieve comparable coverage to the
       WHO surveillance-based panel?

Coverage metric:
    A panel "covers" a resistant isolate if it detects >= 1 mutation present
    in that isolate. Panel coverage = fraction of resistant isolates covered.

    WHO panels achieve ~95% coverage for first-line drugs in MTB.
    Question: what coverage does a LLR-ranked panel of the same size achieve?

Usage:
    python -m experiments.panel_design
    python -m experiments.panel_design --gene rpoB --organism mtb --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from data.who_catalogue.catalogue import (
    WHO_AA_MUTATIONS,
    get_mutations_by_gene,
    get_unique_genes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/panel_design")


def compute_panel_coverage(
    panel_mutations: list[str],
    all_mutations: list[dict],
) -> float:
    """Estimate panel coverage using prevalence as proxy for population frequency.

    Coverage = sum of prevalence_pct for mutations in the panel,
    capped at 100%. This approximates the fraction of resistant isolates
    that carry at least one panel mutation.

    Note: this is an APPROXIMATION. True coverage requires isolate-level data
    (co-occurrence of mutations). For single-locus resistance (like rpoB for RIF),
    this approximation is accurate because mutations are typically mutually exclusive.
    """
    panel_set = set(panel_mutations)
    covered = sum(
        m["prevalence_pct"] for m in all_mutations
        if m["mutation"] in panel_set
    )
    return min(covered, 100.0)


def design_llr_panel(
    gene_results: list[dict],
    k: int,
) -> dict:
    """Design a k-mutation diagnostic panel ranked by |LLR|.

    Returns panel mutations and expected coverage.
    """
    # Sort by |LLR| ascending — lowest fitness cost first
    sorted_by_llr = sorted(gene_results, key=lambda r: r["abs_esm2_llr"])
    panel = sorted_by_llr[:k]
    panel_mutations = [r["mutation"] for r in panel]

    coverage = compute_panel_coverage(panel_mutations, gene_results)

    return {
        "panel_mutations": panel_mutations,
        "panel_abs_llr": [r["abs_esm2_llr"] for r in panel],
        "panel_prevalence": [r["prevalence_pct"] for r in panel],
        "coverage_pct": round(coverage, 2),
        "k": k,
    }


def design_prevalence_panel(
    gene_results: list[dict],
    k: int,
) -> dict:
    """Design a k-mutation panel ranked by clinical prevalence (gold standard)."""
    sorted_by_prev = sorted(gene_results, key=lambda r: -r["prevalence_pct"])
    panel = sorted_by_prev[:k]
    panel_mutations = [r["mutation"] for r in panel]

    coverage = compute_panel_coverage(panel_mutations, gene_results)

    return {
        "panel_mutations": panel_mutations,
        "panel_prevalence": [r["prevalence_pct"] for r in panel],
        "coverage_pct": round(coverage, 2),
        "k": k,
    }


def design_random_panel(
    gene_results: list[dict],
    k: int,
    n_trials: int = 1000,
    seed: int = 42,
) -> dict:
    """Random baseline: average coverage of k random mutations."""
    rng = np.random.default_rng(seed)
    coverages = []
    for _ in range(n_trials):
        idx = rng.choice(len(gene_results), size=min(k, len(gene_results)), replace=False)
        panel_mutations = [gene_results[i]["mutation"] for i in idx]
        cov = compute_panel_coverage(panel_mutations, gene_results)
        coverages.append(cov)
    return {
        "mean_coverage_pct": round(float(np.mean(coverages)), 2),
        "std_coverage_pct": round(float(np.std(coverages)), 2),
        "k": k,
    }


def per_gene_panel_comparison(results: list[dict]) -> dict:
    """Compare LLR-based vs prevalence-based panel design for each gene."""
    by_gene: dict[str, list] = defaultdict(list)
    for r in results:
        by_gene[f"{r['organism']}_{r['gene']}"].append(r)

    analysis = {}
    for gene_key, targets in sorted(by_gene.items()):
        if len(targets) < 3:
            continue

        # Test panel sizes from 1 to N
        n = len(targets)
        gene_analysis = {
            "n_known_mutations": n,
            "panel_comparisons": [],
        }

        for k in range(1, n + 1):
            llr_panel = design_llr_panel(targets, k)
            prev_panel = design_prevalence_panel(targets, k)
            random_panel = design_random_panel(targets, k)

            gene_analysis["panel_comparisons"].append({
                "k": k,
                "llr_coverage": llr_panel["coverage_pct"],
                "llr_mutations": llr_panel["panel_mutations"],
                "prevalence_coverage": prev_panel["coverage_pct"],
                "prevalence_mutations": prev_panel["panel_mutations"],
                "random_coverage_mean": random_panel["mean_coverage_pct"],
                "random_coverage_std": random_panel["std_coverage_pct"],
                "llr_vs_prevalence_gap": round(
                    prev_panel["coverage_pct"] - llr_panel["coverage_pct"], 2
                ),
            })

        # Key metric: at what k does LLR panel reach 90% coverage?
        llr_90 = None
        prev_90 = None
        for comp in gene_analysis["panel_comparisons"]:
            if llr_90 is None and comp["llr_coverage"] >= 90.0:
                llr_90 = comp["k"]
            if prev_90 is None and comp["prevalence_coverage"] >= 90.0:
                prev_90 = comp["k"]

        gene_analysis["k_for_90pct_llr"] = llr_90
        gene_analysis["k_for_90pct_prevalence"] = prev_90
        gene_analysis["panel_size_overhead"] = (
            (llr_90 - prev_90) if llr_90 is not None and prev_90 is not None else None
        )

        analysis[gene_key] = gene_analysis

        # Log key comparison at k=3
        k3 = min(3, n)
        comp = gene_analysis["panel_comparisons"][k3 - 1]
        logger.info(
            "  %s (N=%d): k=%d → LLR=%.1f%% vs Prev=%.1f%% vs Random=%.1f%% [gap=%.1f%%]",
            gene_key, n, k3,
            comp["llr_coverage"], comp["prevalence_coverage"],
            comp["random_coverage_mean"], comp["llr_vs_prevalence_gap"],
        )

    return analysis


def full_landscape_panel(
    protein_sequences: dict[str, str],
    organism: str,
    gene: str,
    hotspot_positions: list[int],
    k: int = 5,
) -> dict | None:
    """Design a panel from the FULL mutation landscape (all 19 AAs per position).

    This is the TRUE pre-emptive design: we don't use the WHO catalogue at all.
    We scan all possible substitutions at known drug-binding positions and
    rank by |LLR| to predict which mutations will emerge.

    This is what you'd do for a drug still in clinical trials.
    """
    from models.esm2_scorer import ESM2Scorer

    protein_key = f"{organism}_{gene}"
    protein_seq = protein_sequences.get(protein_key)
    if protein_seq is None:
        logger.warning("No protein for %s", protein_key)
        return None

    scorer = ESM2Scorer()
    landscape = scorer.score_full_landscape(
        protein_seq, positions=hotspot_positions, gene=gene,
    )
    scorer.cleanup()

    # Flatten landscape into ranked mutations
    all_mutations = []
    for position, aa_scores in landscape.items():
        wt_aa = protein_seq[position - 1]
        for alt_aa, llr in aa_scores.items():
            all_mutations.append({
                "position": position,
                "mutation": f"{wt_aa}{position}{alt_aa}",
                "wt_aa": wt_aa,
                "alt_aa": alt_aa,
                "llr": llr,
                "abs_llr": abs(llr),
            })

    # Rank by |LLR| — lowest first (most likely to emerge)
    all_mutations.sort(key=lambda m: m["abs_llr"])

    # The panel = top-k lowest |LLR| mutations
    panel = all_mutations[:k]
    non_panel = all_mutations[k:]

    # Check which panel mutations are in WHO catalogue
    from data.who_catalogue.catalogue import get_mutations_by_gene
    known = {m.mutation for m in get_mutations_by_gene(organism, gene)}

    for m in panel:
        m["in_who_catalogue"] = m["mutation"] in known

    return {
        "organism": organism,
        "gene": gene,
        "n_positions_scanned": len(hotspot_positions),
        "n_total_mutations": len(all_mutations),
        "panel_size": k,
        "panel": panel,
        "n_panel_in_who": sum(1 for m in panel if m["in_who_catalogue"]),
        "all_mutations": all_mutations,  # Full landscape for analysis
    }


# Known drug-binding hotspot positions for pre-emptive scanning
# These come from crystal structures, not surveillance data
HOTSPOT_POSITIONS = {
    ("mtb", "rpoB"): list(range(430, 460)),       # RRDR: codons 430-460
    ("mtb", "katG"): [315],                        # S315 catalytic site
    ("mtb", "gyrA"): list(range(88, 96)),          # QRDR: codons 88-95
    ("mtb", "embB"): [306, 354, 406, 497],         # Ethambutol binding pocket
    ("mtb", "pncA"): list(range(1, 186)),           # Entire enzyme (small, 186 aa)
    ("ecoli", "gyrA"): list(range(81, 88)),        # QRDR
    ("ecoli", "parC"): list(range(78, 85)),        # QRDR
    ("saureus", "gyrA"): list(range(82, 88)),      # QRDR
    ("ngonorrhoeae", "penA"): list(range(310, 550)), # PBP2 transpeptidase domain
}


def main():
    parser = argparse.ArgumentParser(description="Pre-emptive diagnostic panel design")
    parser.add_argument("--llr-results", type=Path,
                        default=Path("results/retrospective/llr_results.csv"),
                        help="LLR results from retrospective experiment")
    parser.add_argument("--protein-sequences", type=Path,
                        default=Path("data/protein_sequences/sequences.json"),
                        help="Protein sequences for full landscape scan")
    parser.add_argument("--full-landscape", action="store_true",
                        help="Run full landscape scan (requires ESM-2 + GPU)")
    parser.add_argument("--gene", type=str, default=None,
                        help="Specific gene for full landscape (e.g., rpoB)")
    parser.add_argument("--organism", type=str, default=None,
                        help="Specific organism for full landscape (e.g., mtb)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Panel size for full landscape design")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "preemptive_panel_design",
        "timestamp": datetime.now().isoformat(),
        "full_landscape": args.full_landscape,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Part 1: Compare LLR-ranked vs prevalence-ranked panels using WHO data
    if args.llr_results.exists():
        import csv
        with open(args.llr_results) as f:
            results = list(csv.DictReader(f))
        for r in results:
            if r.get("esm2_llr") and r["esm2_llr"] != "":
                r["esm2_llr"] = float(r["esm2_llr"])
                r["abs_esm2_llr"] = abs(r["esm2_llr"])
                r["prevalence_pct"] = float(r["prevalence_pct"])
            else:
                r["esm2_llr"] = None

        computed = [r for r in results if r["esm2_llr"] is not None]

        logger.info("=" * 60)
        logger.info("PANEL COMPARISON: LLR-ranked vs Prevalence-ranked vs Random")
        logger.info("=" * 60)
        comparison = per_gene_panel_comparison(computed)

        with open(RESULTS_DIR / "panel_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info("Saved panel comparison to %s", RESULTS_DIR / "panel_comparison.json")

        # Aggregate statistics
        gaps = []
        for gene_key, gene_data in comparison.items():
            for comp in gene_data["panel_comparisons"]:
                if comp["k"] <= 3:
                    gaps.append(comp["llr_vs_prevalence_gap"])

        if gaps:
            logger.info("\nAggregate (k<=3):")
            logger.info("  Mean coverage gap (prevalence - LLR): %.1f%%", np.mean(gaps))
            logger.info("  Max gap: %.1f%%", max(gaps))
            logger.info("  Genes where LLR matches prevalence: %d/%d",
                       sum(1 for g in gaps if g == 0), len(gaps))
    else:
        logger.info("No LLR results found. Run retrospective experiment first.")

    # Part 2: Full landscape scan (the novel part — pre-emptive design)
    if args.full_landscape:
        if not args.protein_sequences.exists():
            logger.error("Protein sequences required for full landscape scan")
            return

        with open(args.protein_sequences) as f:
            proteins = json.load(f)

        if args.organism and args.gene:
            targets = [(args.organism, args.gene)]
        else:
            targets = [
                (org, gene) for (org, gene) in HOTSPOT_POSITIONS.keys()
            ]

        logger.info("\n" + "=" * 60)
        logger.info("FULL LANDSCAPE SCAN — Pre-emptive Panel Design")
        logger.info("=" * 60)

        for organism, gene in targets:
            positions = HOTSPOT_POSITIONS.get((organism, gene))
            if positions is None:
                continue

            logger.info("\n--- %s %s (%d hotspot positions) ---", organism, gene, len(positions))
            result = full_landscape_panel(
                proteins, organism, gene, positions, k=args.top_k,
            )

            if result is not None:
                logger.info("  Panel (%d mutations):", result["panel_size"])
                for m in result["panel"]:
                    who_tag = " [WHO]" if m["in_who_catalogue"] else " [NEW]"
                    logger.info(
                        "    %s: |LLR|=%.3f%s", m["mutation"], m["abs_llr"], who_tag,
                    )
                logger.info(
                    "  %d/%d panel mutations in WHO catalogue",
                    result["n_panel_in_who"], result["panel_size"],
                )

                out_path = RESULTS_DIR / f"landscape_{organism}_{gene}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
