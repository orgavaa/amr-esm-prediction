# Protein Language Models Reveal Mechanism-Dependent Predictability of Antimicrobial Resistance

**ESM-2 masked marginal fitness scoring predicts clinical prevalence of resistance mutations — but only when resistance operates through conservative active-site substitution.**

## Summary

This repository implements a computational framework that uses ESM-2 (Lin et al., *Science* 2023), a 650M-parameter protein language model, to predict the fitness landscape of antimicrobial resistance (AMR) mutations across four WHO priority pathogens. The central result is a **three-class mechanistic model** that determines when protein language models can and cannot predict clinical resistance prevalence:

- **Class 1 (conservative substitution):** katG, gyrA, penA — ESM-2 LLR correctly ranks mutations by prevalence (concordance 0.67–0.83). LLR-ranked diagnostic panels match surveillance-ranked panels at zero coverage gap.
- **Class 2 (loss of function):** pncA, Rv0678, ddn — the entire protein is permissive to disruption. ESM-2 identifies thousands of fit mutations and cannot distinguish which confer resistance.
- **Class 3 (structural pocket clustering):** rpoB RRDR, embB — all resistance mutations cluster at similar |LLR|, producing flat rankings with no prevalence signal.

Across 81 amino acid substitutions in *M. tuberculosis*, *E. coli*, *S. aureus*, and *N. gonorrhoeae*, |ESM-2 LLR| correlates significantly with clinical prevalence (Spearman ρ = −0.312, *p* = 0.005, permutation test). The strongest single-gene result is katG S315T (isoniazid resistance), which sits in a fitness valley (|LLR| = 0.68) surrounded by heavily constrained positions (min |LLR| > 3.6 at positions 316, 318) — the evolutionary escape route that accounts for 64% of isoniazid resistance worldwide.

---

## Background and Motivation

Current AMR diagnostic panels (Xpert MTB/RIF, GenoType MTBDRplus) are designed by surveilling patient populations over years, tabulating mutation frequencies, and selecting the most prevalent targets (WHO, 2023). The CRyPTIC consortium sequenced 12,289 *M. tuberculosis* isolates from 23 countries to produce the current WHO mutation catalogue (The CRyPTIC Consortium, *PLOS Biology* 2022). This process is slow, geography-specific, and must be repeated as resistance landscapes shift.

Protein language models offer an alternative: a physics-based prior on mutation fitness that is stable, computable in seconds, and available before any clinical data exists. Meier et al. (*NeurIPS* 2021) and Brandes et al. (*Nature Genetics* 2023) demonstrated that ESM-2 masked marginal scoring predicts functional effects of amino acid substitutions. AMRscope (bioRxiv 2025) showed ESM-2 embeddings can triage AMR variants with F1 = 0.87.

However, prior work treats all resistance targets uniformly. We show that the resistance *mechanism* — not just the protein or organism — determines whether PLM fitness is informative. This mechanistic classification is the main contribution.

---

## Methods

### ESM-2 Masked Marginal Log-Likelihood Ratio

For each amino acid substitution X*i*Y at position *i*:

```
LLR = log P(Y | context) − log P(X | context)
```

where context is the full protein sequence with position *i* masked. LLR < 0 indicates the wildtype is evolutionarily preferred; |LLR| quantifies the fitness cost. We use `esm2_t33_650M_UR50D` (650M parameters, trained on UniRef50). Each scoring requires one forward pass (~0.5s on RTX 2070 SUPER).

### Mutation Catalogue

85 amino acid substitutions across 22 drug-target genes in 4 species, curated from the WHO mutation catalogue v2 (2023), CARD database, and EUCAST breakpoint tables. Each mutation is annotated with clinical prevalence (% of resistant isolates carrying the mutation) and WHO confidence tier.

### Full Landscape Scanning

For selected genes, we score all 19 possible substitutions at every position (or at drug-binding hotspot positions), producing a complete fitness landscape. katG: 740 positions × 19 = 14,060 substitutions (147s). gyrA QRDR: 8 positions × 19 = 152 substitutions (4s).

### Statistical Analysis

- Spearman rank correlation with 10,000-iteration permutation test for robust p-values
- Bootstrap 95% confidence intervals (5,000 resamples)
- Rank concordance (Kendall τ mapped to [0, 1]; 0.5 = random) for within-gene emergence order
- Top-k precision for ranking validation
- Panel coverage comparison: LLR-ranked vs prevalence-ranked vs random baseline

---

## Results

### Experiment 1: Retrospective Validation (N = 81)

| Scope | N | Spearman ρ | Permutation *p* | 95% CI |
|-------|---|-----------|-----------------|--------|
| Pooled | 81 | −0.312 | 0.005 | [−0.491, −0.104] |
| *M. tuberculosis* | 57 | −0.271 | 0.042 | [−0.509, −0.011] |
| *N. gonorrhoeae* | 12 | +0.176 | 0.579 | NS |
| *S. aureus* | 8 | +0.575 | 0.144 | NS |

94% of LLR values are negative (wildtype preferred). |LLR| ranges from 0.06 to 12.37.

### Experiment 2: Within-Gene Emergence Order and Fitness Landscapes

| Gene | N | Rank concordance | Top-k precision | Mechanism |
|------|---|-----------------|-----------------|-----------|
| katG | 4 | **0.833** | 1.00 (k=1) | Conservative substitution |
| gyrA | 8 | **0.679** | 1.00 (k=2) | Conservative substitution |
| penA | 6 | **0.667** | 0.50 (k=2) | Conservative substitution |
| rpoB | 12 | 0.500 | 0.25 (k=4) | Structural pocket |
| pncA | 15 | 0.447 | 0.20 (k=5) | Loss of function |
| embB | 7 | 0.333 | 0.00 (k=2) | Structural pocket |

**katG landscape:** S315T has |LLR| = 0.68 — the lowest-cost substitution at the catalytic position 315. Neighboring positions G316 and E318 have minimum |LLR| > 3.6 and 3.7 respectively. S315T is an evolutionary escape route in a structurally constrained active-site region.

**gyrA landscape:** All 8 WHO QRDR mutations fall in the top 32% of the landscape (96/304 total substitutions). A90V and D94G are tied at |LLR| ≈ 2.52 — predicted co-dominant, and they ARE the two clinically dominant fluoroquinolone resistance mutations.

### Experiment 3: Diagnostic Panel Comparison

For each gene, we compare LLR-ranked panels to prevalence-ranked (gold standard) and random baseline panels at increasing panel sizes.

| Gene | k=2 LLR coverage | k=2 Prevalence coverage | Gap |
|------|------------------|------------------------|-----|
| katG | 66.1% | 66.1% | **0.0%** |
| gyrA | 46.0% | 46.0% | **0.0%** |
| Rv0678 | 0.8% | 0.8% | **0.0%** |
| rpoB | 10.8% | 50.5% | 39.7% |
| pncA | 0.6% | 5.5% | 4.9% |

For Class 1 targets (katG, gyrA), LLR-ranked panels achieve identical coverage to surveillance-ranked panels. For Class 3 targets (rpoB), the gap is 40+ percentage points. This directly demonstrates which drug targets are amenable to computational panel design without surveillance data.

### Three-Class Mechanistic Model

| Class | Mechanism | Examples | LLR predictive? | Why |
|-------|-----------|----------|-----------------|-----|
| 1 | Conservative active-site substitution | katG, gyrA, penA | **Yes** | Few positions confer resistance; fitness ranking distinguishes them |
| 2 | Loss of function | pncA, Rv0678, ddn | **No** | Any disruption confers resistance; entire protein is permissive |
| 3 | Structural pocket clustering | rpoB RRDR, embB | **No** | All pocket mutations have similar fitness cost; ranking is flat |

---

## Repository Structure

```
amr-esm-prediction/
├── models/
│   └── esm2_scorer.py           # ESM-2 masked marginal scorer
│                                 #   score_mutation, score_batch,
│                                 #   score_full_landscape, get_representations
├── data/
│   ├── who_catalogue/            # 85 WHO AA substitutions across 4 organisms
│   ├── drug_targets/             # Binding sites, compensatory pairs, drug metadata
│   ├── lineage_backgrounds/      # MTB L1/L2/L4 polymorphisms
│   └── protein_sequences/        # 24 reference protein sequences
├── experiments/
│   ├── retrospective.py          # Exp 1: |LLR| vs prevalence correlation
│   ├── prospective.py            # Exp 2: within-gene emergence order prediction
│   └── panel_design.py           # Exp 3: LLR-ranked vs prevalence-ranked panels
├── research/                     # Exploratory modules (not part of published analysis)
│   ├── README.md
│   ├── models/                   # epistasis_scorer, emergence_simulator, binding_disruption
│   └── experiments/              # emergence_forecast, denovo_design
├── scripts/
│   └── generate_figures.py       # Publication figures (Fig 2–5 + supplementary)
├── configs/default.yaml
└── notebooks/figures.ipynb
```

## Installation

```bash
git clone https://github.com/orgavaa/amr-esm-prediction.git
cd amr-esm-prediction
python -m venv .venv
source .venv/Scripts/activate  # Windows; use .venv/bin/activate on Linux/Mac
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fair-esm scipy pandas matplotlib seaborn biopython
```

Requires Python >= 3.10, PyTorch >= 2.0, CUDA-capable GPU with >= 4 GB VRAM (tested on RTX 2070 SUPER 8 GB).

## Usage

```bash
# Experiment 1: Retrospective validation
python -m experiments.retrospective

# Experiment 2: Prospective emergence order + landscape scans
python -m experiments.prospective
python -m experiments.panel_design --full-landscape --organism mtb --gene katG

# Experiment 3: Panel design comparison
python -m experiments.panel_design

# Generate publication figures
python scripts/generate_figures.py
```

---

## Limitations

1. **Class 2/3 targets are not addressable by protein-level fitness alone.** Loss-of-function resistance produces too many candidates; structural pocket clustering produces indistinguishable fitness costs. Complementary data (drug-binding affinity, deep mutational scanning, genome-context models) is required.

2. **Non-protein resistance mechanisms are excluded.** Promoter mutations (fabG1 C-15T, eis C-14T), rRNA mutations (rrs A1401G), indels, and horizontally acquired resistance genes are not scorable by ESM-2. These account for ~30% of clinically relevant TB resistance. DNA-level models (EVO-2, Caduceus) could address this gap.

3. **Genomic background is not modelled.** ESM-2 scores each protein in isolation. Lineage-specific polymorphisms, cross-gene epistasis, and genome-wide pre-resistance signatures (Torres Ortiz et al., 2021) are not captured.

4. **Fitness is not resistance.** ESM-2 LLR measures evolutionary fitness cost, not resistance level. A mutation can be very fit (low |LLR|) without conferring drug resistance — as demonstrated by the gyrA full-landscape scan where D94E (|LLR| = 0.08) is not WHO-catalogued despite being the "fittest" QRDR substitution.

5. **Validation is retrospective.** We use prevalence rank as a proxy for emergence order. True prospective validation requires longitudinal WGS data with first-detection timestamps (CRyPTIC, Euro-GASP). The Rv0678/bedaquiline case (drug approved 2012, mutations catalogued 2022) is illustrative but anecdotal (N = 3).

## Future Directions

Exploratory modules in `research/` address two extensions not included in the published analysis:

- **Pairwise epistasis from ESM-2 mutant backgrounds** — scoring mutation B after introducing mutation A captures sequence-context effects that may approximate epistatic interactions. Initial results show biologically plausible patterns (gyrA S91P→A90G synergy) but lack experimental validation. Combined with kinetic Monte Carlo simulation, this could predict temporal emergence order, though early results show the simulation adds noise for small mutation sets.

- **De novo diagnostic panel design** — a two-filter pipeline (ESM-2 fitness + structural binding-site proximity) achieves 75% recall for katG but 0–12% for all other targets, indicating the approach is currently limited to trivial single-position cases.

Both directions require validation against experimental data (combinatorial DMS for epistasis, clinical resistance emergence for temporal prediction) before they can support published claims.

---

## References

### Protein Language Models and Variant Effect Prediction

- Lin Z, Akin H, Rao R, et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637):1123–1130. doi:10.1126/science.ade2574
- Meier J, Rao R, Verkuil R, et al. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. *Advances in Neural Information Processing Systems*, 34:29287–29303.
- Brandes N, Goldman G, Wang CH, et al. (2023). Genome-wide prediction of disease variant effects with a deep protein language model. *Nature Genetics*, 55:1512–1522. doi:10.1038/s41588-023-01465-0
- Frazer J, Notin P, Dias M, et al. (2021). Disease variant prediction with deep generative models of evolutionary data. *Nature*, 599:91–95. doi:10.1038/s41586-021-04043-8

### AMR Prediction with Machine Learning

- AMRscope (2025). Risk-based prediction of antimicrobial resistance using ESM-2 protein language model. *bioRxiv*. doi:10.1101/2025.09.12.672331
- BIG-TB (2026). A benchmark for interpretable genotype-to-phenotype prediction in tuberculosis. *bioRxiv*. doi:10.64898/2026.01.30.702134
- Kuang X, et al. (2024). Antibiotic resistance mechanism prediction via ProteinBERT protein language model. *Bioinformatics*, 40(10):btae550. doi:10.1093/bioinformatics/btae550

### Fitness Costs and Compensatory Evolution

- Andersson DI, Hughes D (2010). Antibiotic resistance and its cost: is it possible to reverse resistance? *Nature Reviews Microbiology*, 8:260–271. doi:10.1038/nrmicro2319
- Gagneux S, Long CD, Small PM, et al. (2006). The competitive cost of antibiotic resistance in *Mycobacterium tuberculosis*. *Science*, 312(5782):1944–1946. doi:10.1126/science.1124410
- Comas I, Borrell S, Roetzer A, et al. (2012). Whole-genome sequencing of rifampicin-resistant *Mycobacterium tuberculosis* strains identifies compensatory mutations in RNA polymerase genes. *Nature Genetics*, 44(1):106–110. doi:10.1038/ng.1038

### Pre-Resistance and Temporal Emergence

- Torres Ortiz A, Grandjean L, et al. (2021). Genomic signatures of pre-resistance in *Mycobacterium tuberculosis*. *Nature Communications*, 12:7273. doi:10.1038/s41467-021-27616-7
- Rodriguez de Evgrafov MC, et al. (2024). Kinetic coevolutionary models predict drug-resistance mutation acquisition rates in HIV. *PNAS*, 121(19). doi:10.1073/pnas.2316662121
- Ogbunugafor CB, et al. (2023). Environmental modulation of global epistasis in a drug resistance fitness landscape. *Nature Communications*, 14:8055. doi:10.1038/s41467-023-43806-x

### Drug-Target Interaction and Surveillance

- Singh R, et al. (2023). Contrastive learning in protein language model space predicts interactions between drugs and protein targets. *PNAS*, 120(24):e2220778120. doi:10.1073/pnas.2220778120
- The CRyPTIC Consortium (2022). A data compendium associating the genomes of 12,289 *Mycobacterium tuberculosis* isolates with quantitative resistance phenotypes to 13 antibiotics. *PLOS Biology*, 20(8):e3001721. doi:10.1371/journal.pbio.3001721
- WHO (2023). Catalogue of mutations in *Mycobacterium tuberculosis* complex and their association with drug resistance, 2nd edition. Geneva: World Health Organization.

---

## License

MIT
