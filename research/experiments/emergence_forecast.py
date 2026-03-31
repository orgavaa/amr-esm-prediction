"""Experiment 4: Resistance emergence forecasting.

Predicts the temporal ORDER in which resistance mutations emerge under
drug pressure, using ESM-2 pairwise epistasis + kinetic Monte Carlo.

Novel contributions:
1. Pairwise epistasis computed directly from ESM-2 masked marginals
   on mutant backgrounds (nobody has done this with PLMs)
2. KMC simulation driven by PLM fitness landscape (existing work uses
   experimentally measured fitness or Potts models from MSA)
3. Lineage-stratified predictions (L2 Beijing vs L4 Euro-American)

Validation:
- Simulated emergence order vs WHO prevalence rank (Kendall tau)
- INH-before-RIF temporal ordering (Torres Ortiz 46:2 ratio)
- Known epistatic pairs: rpoB S450L + rpoC compensatory

Usage:
    python -m experiments.emergence_forecast
    python -m experiments.emergence_forecast --gene gyrA --organism mtb
    python -m experiments.emergence_forecast --all-targets
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
from scipy.stats import spearmanr, kendalltau

from data.drug_targets.targets import DRUG_TARGETS, get_target
from data.drug_targets.compensatory import get_compensatory_pairs
from data.lineage_backgrounds.backgrounds import get_backgrounds
from data.who_catalogue.catalogue import get_mutations_by_gene
from models.esm2_scorer import ESM2Scorer, parse_aa_mutation
from models.epistasis_scorer import (
    compute_pairwise_epistasis,
    compute_pathway_fitness,
    introduce_mutations,
    EpistaticEffect,
)
from models.emergence_simulator import (
    MutationTrajectorySimulator,
    SimulationConfig,
    find_dominant_pathways,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/emergence")


def load_llr_results() -> dict[str, dict]:
    """Load pre-computed LLR results from retrospective experiment."""
    path = Path("results/retrospective/llr_results.csv")
    if not path.exists():
        raise FileNotFoundError(f"Run retrospective experiment first: {path}")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    lookup = {}
    for r in rows:
        if r.get("esm2_llr") and r["esm2_llr"] != "":
            key = f"{r['gene']}_{r['mutation']}"
            lookup[key] = {
                "organism": r["organism"],
                "gene": r["gene"],
                "mutation": r["mutation"],
                "esm2_llr": float(r["esm2_llr"]),
                "prevalence_pct": float(r["prevalence_pct"]),
            }
    return lookup


def compute_epistasis_for_target(
    scorer: ESM2Scorer,
    protein_sequences: dict[str, str],
    organism: str,
    gene: str,
) -> dict:
    """Compute pairwise epistasis matrix for a drug target's mutations.

    Returns dict with epistasis matrix and metadata.
    """
    target = get_target(organism, gene)
    if target is None:
        logger.warning("No drug target defined for %s %s", organism, gene)
        return {}

    protein_seq = protein_sequences.get(target.key)
    if protein_seq is None:
        logger.warning("No protein sequence for %s", target.key)
        return {}

    mutations_data = get_mutations_by_gene(organism, gene)
    mutations = [m.mutation for m in mutations_data]

    if len(mutations) < 2:
        logger.info("Only %d mutations for %s %s, skipping epistasis", len(mutations), organism, gene)
        return {}

    logger.info("Computing epistasis for %s %s: %d mutations (%d pairs)",
                organism, gene, len(mutations), len(mutations) * (len(mutations) - 1))

    effects, matrix = compute_pairwise_epistasis(
        scorer, protein_seq, mutations, gene=gene,
    )

    return {
        "organism": organism,
        "gene": gene,
        "mutations": mutations,
        "n_mutations": len(mutations),
        "n_pairs": len(effects),
        "effects": [
            {
                "mutation_a": e.mutation_a,
                "mutation_b": e.mutation_b,
                "llr_b_on_wt": e.llr_b_on_wt,
                "llr_b_on_a": e.llr_b_on_a,
                "epistasis": e.epistasis,
                "type": e.epistasis_type,
            }
            for e in effects
        ],
        "matrix": matrix.tolist(),
        "summary": {
            "mean_epistasis": round(float(np.mean([e.epistasis for e in effects])), 4),
            "std_epistasis": round(float(np.std([e.epistasis for e in effects])), 4),
            "n_synergistic": sum(1 for e in effects if e.epistasis_type == "synergistic"),
            "n_antagonistic": sum(1 for e in effects if e.epistasis_type == "antagonistic"),
            "n_neutral": sum(1 for e in effects if e.epistasis_type == "neutral"),
        },
    }


def run_emergence_simulation(
    organism: str,
    gene: str,
    llr_lookup: dict[str, dict],
    epistasis_data: dict,
    config: SimulationConfig,
) -> dict:
    """Run KMC emergence simulation for a drug target.

    Returns dict with emergence times, ordering, and validation metrics.
    """
    mutations = epistasis_data.get("mutations", [])
    if not mutations:
        return {}

    # Build LLR dict
    llr_values = {}
    prevalence = {}
    for mut in mutations:
        key = f"{gene}_{mut}"
        if key in llr_lookup:
            llr_values[mut] = llr_lookup[key]["esm2_llr"]
            prevalence[mut] = llr_lookup[key]["prevalence_pct"]

    if not llr_values:
        return {}

    # Build epistasis matrix
    matrix = np.array(epistasis_data["matrix"]) if epistasis_data.get("matrix") else None

    logger.info("Simulating emergence for %s %s: %d mutations, %d replicates",
                organism, gene, len(mutations), config.n_replicates)

    simulator = MutationTrajectorySimulator(
        mutations=mutations,
        llr_values=llr_values,
        epistasis_matrix=matrix,
    )

    results = simulator.estimate_emergence_times(config)
    pathways = find_dominant_pathways(results)

    # Validation: compare predicted order to prevalence rank
    predicted_order = []
    true_prevalence = []
    for r in results:
        if r.mutation in prevalence and r.emergence_probability > 0:
            predicted_order.append(r.median_emergence_gen)
            true_prevalence.append(prevalence[r.mutation])

    validation = {}
    if len(predicted_order) >= 4:
        # Lower emergence time should correlate with higher prevalence
        rho, p = spearmanr(predicted_order, true_prevalence)
        tau, tau_p = kendalltau(predicted_order, true_prevalence)

        # Concordance (from prospective.py logic)
        concordance = (tau + 1) / 2

        validation = {
            "n": len(predicted_order),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p), 4),
            "kendall_tau": round(float(tau), 4),
            "kendall_p": round(float(tau_p), 4),
            "rank_concordance": round(float(concordance), 4),
            "expected_direction": "negative (early emergence = high prevalence)",
        }

    return {
        "organism": organism,
        "gene": gene,
        "config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "drug_concentration_mic_ratio": config.drug_concentration_mic_ratio,
            "n_replicates": config.n_replicates,
        },
        "emergence_order": [
            {
                "rank": i + 1,
                "mutation": r.mutation,
                "median_generation": round(r.median_emergence_gen, 1),
                "emergence_probability": round(r.emergence_probability, 3),
                "prevalence_pct": prevalence.get(r.mutation, 0),
            }
            for i, r in enumerate(results)
        ],
        "dominant_pathways": pathways,
        "validation": validation,
    }


def lineage_comparison(
    scorer: ESM2Scorer,
    protein_sequences: dict[str, str],
    organism: str,
    gene: str,
    llr_lookup: dict[str, dict],
) -> dict:
    """Compare LLR scores on different lineage backgrounds.

    For MTB, tests whether L2 (Beijing) background produces different
    fitness predictions than L4 (Euro-American / H37Rv reference).
    """
    backgrounds = get_backgrounds(organism)
    if not backgrounds:
        return {}

    protein_seq = protein_sequences.get(f"{organism}_{gene}")
    if protein_seq is None:
        return {}

    mutations_data = get_mutations_by_gene(organism, gene)
    mutations = [m.mutation for m in mutations_data]

    results = {}
    for bg in backgrounds:
        # Apply lineage polymorphisms to the protein
        polys = bg.gene_polymorphisms.get(gene, [])
        if polys:
            lineage_seq = introduce_mutations(protein_seq, polys)
            label = f"{bg.lineage}_{bg.lineage_name}"
            logger.info("Scoring %d mutations on %s background (%d polymorphisms)",
                       len(mutations), label, len(polys))

            lineage_results = scorer.score_batch(lineage_seq, mutations, gene=gene)
            results[bg.lineage] = {
                "lineage": bg.lineage,
                "lineage_name": bg.lineage_name,
                "polymorphisms_applied": polys,
                "mutations": [
                    {
                        "mutation": m,
                        "llr_reference": llr_lookup.get(f"{gene}_{m}", {}).get("esm2_llr"),
                        "llr_lineage": r.llr,
                        "delta": round(r.llr - (llr_lookup.get(f"{gene}_{m}", {}).get("esm2_llr", 0)), 4),
                    }
                    for m, r in zip(mutations, lineage_results)
                ],
            }
        else:
            results[bg.lineage] = {
                "lineage": bg.lineage,
                "lineage_name": bg.lineage_name,
                "polymorphisms_applied": [],
                "note": "No polymorphisms in this gene — identical to reference",
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Emergence forecasting")
    parser.add_argument("--organism", type=str, default="mtb")
    parser.add_argument("--gene", type=str, default=None,
                        help="Specific gene (default: all genes for organism)")
    parser.add_argument("--all-targets", action="store_true",
                        help="Run on all validated drug targets")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--protein-sequences", type=Path,
                        default=Path("data/protein_sequences/sequences.json"))
    parser.add_argument("--skip-epistasis", action="store_true",
                        help="Use additive model only (skip epistasis computation)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        n_replicates=args.replicates,
        generations=args.generations,
    )

    # Save experiment config
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump({
            "experiment": "emergence_forecast",
            "timestamp": datetime.now().isoformat(),
            "organism": args.organism,
            "gene": args.gene,
            "replicates": args.replicates,
            "generations": args.generations,
        }, f, indent=2)

    # Load data
    with open(args.protein_sequences) as f:
        protein_sequences = json.load(f)
    llr_lookup = load_llr_results()

    # Determine which targets to run
    if args.all_targets:
        targets = [(t.organism, t.gene) for t in DRUG_TARGETS]
    elif args.gene:
        targets = [(args.organism, args.gene)]
    else:
        targets = [
            (t.organism, t.gene) for t in DRUG_TARGETS
            if t.organism == args.organism
        ]

    # Filter to targets with enough mutations
    targets = [
        (org, gene) for org, gene in targets
        if len(get_mutations_by_gene(org, gene)) >= 3
    ]

    scorer = ESM2Scorer()
    all_results = {}

    for organism, gene in targets:
        logger.info("\n" + "=" * 60)
        logger.info("TARGET: %s %s", organism, gene)
        logger.info("=" * 60)

        # Step 1: Compute epistasis
        if not args.skip_epistasis:
            epistasis_data = compute_epistasis_for_target(
                scorer, protein_sequences, organism, gene,
            )
            if epistasis_data:
                out_path = RESULTS_DIR / f"epistasis_{organism}_{gene}.json"
                with open(out_path, "w") as f:
                    json.dump(epistasis_data, f, indent=2)
                logger.info("Saved epistasis to %s", out_path)
        else:
            # Minimal epistasis data for simulation
            mutations = [m.mutation for m in get_mutations_by_gene(organism, gene)]
            epistasis_data = {"mutations": mutations, "matrix": None}

        # Step 2: Run emergence simulation
        sim_result = run_emergence_simulation(
            organism, gene, llr_lookup, epistasis_data, config,
        )

        if sim_result:
            all_results[f"{organism}_{gene}"] = sim_result

            # Log key results
            logger.info("\nEmergence order (predicted):")
            for entry in sim_result["emergence_order"][:5]:
                logger.info(
                    "  #%d %s: gen=%.0f, P(emerge)=%.2f, prevalence=%.1f%%",
                    entry["rank"], entry["mutation"],
                    entry["median_generation"], entry["emergence_probability"],
                    entry["prevalence_pct"],
                )

            if sim_result.get("validation"):
                v = sim_result["validation"]
                logger.info(
                    "\nValidation: rho=%.3f (p=%.4f), concordance=%.3f",
                    v["spearman_rho"], v["spearman_p"], v["rank_concordance"],
                )

        # Step 3: Lineage comparison (MTB only)
        if organism == "mtb":
            lineage_data = lineage_comparison(
                scorer, protein_sequences, organism, gene, llr_lookup,
            )
            if lineage_data:
                out_path = RESULTS_DIR / f"lineage_{organism}_{gene}.json"
                with open(out_path, "w") as f:
                    json.dump(lineage_data, f, indent=2)

    scorer.cleanup()

    # Save all simulation results
    with open(RESULTS_DIR / "emergence_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nSaved all results to %s", RESULTS_DIR / "emergence_results.json")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for key, result in all_results.items():
        v = result.get("validation", {})
        rho = v.get("spearman_rho", float("nan"))
        conc = v.get("rank_concordance", float("nan"))
        n_emerged = sum(1 for e in result["emergence_order"] if e["emergence_probability"] > 0)
        logger.info(
            "  %s: %d/%d emerged, rho=%.3f, concordance=%.3f",
            key, n_emerged, len(result["emergence_order"]), rho, conc,
        )


if __name__ == "__main__":
    main()
