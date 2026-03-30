# Protein Language Models Enable Pre-emptive Antimicrobial Resistance Diagnostic Design

**Predicting which resistance mutations will emerge — before they are observed clinically — using ESM-2 masked marginal fitness scoring.**

## Summary

This repository implements a computational framework that uses ESM-2 (Lin et al., *Science* 2023), a 650M-parameter protein language model, to predict the fitness landscape of antimicrobial resistance (AMR) mutations across four WHO priority pathogens. The central result is a **three-class mechanistic model** that determines when protein language models can and cannot predict clinical resistance prevalence, and a **de novo diagnostic panel design pipeline** that generates diagnostic targets for drugs still in clinical trials without requiring any surveillance data.

We demonstrate that ESM-2 masked marginal log-likelihood ratios (LLR) — a zero-shot measure of evolutionary fitness cost — correlate significantly with clinical prevalence across 81 amino acid substitutions in *M. tuberculosis*, *E. coli*, *S. aureus*, and *N. gonorrhoeae* (Spearman ρ = −0.312, *p* = 0.005). We further show that this signal is mechanism-dependent: predictive for conservative active-site substitutions (katG: ρ = −0.80; gyrA: ρ = −0.52), but uninformative for loss-of-function targets (pncA) and structurally clustered resistance pockets (rpoB RRDR).

Building on this, we introduce two novel modules: **(1)** pairwise epistasis scoring from ESM-2 masked marginals on mutant backgrounds coupled with kinetic Monte Carlo emergence simulation, and **(2)** a two-filter de novo panel design pipeline combining ESM-2 fitness with structural binding-site proximity, achieving 75% recall of WHO-catalogued mutations for isoniazid resistance using zero clinical data.

---

## Background and Motivation

Current AMR diagnostic panels (e.g., Xpert MTB/RIF, GenoType MTBDRplus) are designed by surveilling patient populations over years, tabulating mutation frequencies, and selecting the most prevalent targets (WHO, 2023). This approach requires large-scale whole-genome sequencing studies — the CRyPTIC consortium (The CRyPTIC Consortium, *PLOS Biology* 2022) sequenced 12,289 *M. tuberculosis* isolates from 23 countries over multiple years to produce the current WHO mutation catalogue. Prevalence data is geography-specific and time-dependent, requiring repeated updates.

Protein language models offer an alternative: a physics-based prior on mutation fitness that is stable, computable in seconds, and available before any clinical data exists. Meier et al. (*NeurIPS* 2021) and Brandes et al. (*Nature Genetics* 2023) demonstrated that ESM-2 masked marginal scoring predicts the functional effects of amino acid substitutions across diverse proteins. However, systematic application to AMR prediction — particularly the question of whether PLM fitness can enable *pre-emptive* diagnostic design for drugs still in development — has not been investigated.

### Key Gaps This Work Addresses

1. **Retrospective correlation is known; prospective prediction is not.** That ESM-2 LLR correlates with variant pathogenicity is established (Meier et al., 2021). Whether it predicts the *order* of resistance emergence under drug pressure, within specific genes, has not been tested.

2. **No mechanistic classification of AMR predictability.** Prior work treats all resistance targets uniformly. We show that the resistance mechanism (conservative substitution vs. loss-of-function vs. structural pocket) determines whether PLM fitness is informative — a finding with direct implications for which drug targets are amenable to computational diagnostic design.

3. **No pre-emptive panel design from PLMs.** AMRscope (bioRxiv 2025) demonstrated ESM-2 embeddings for binary AMR variant triage. ConPLex (Singh et al., *PNAS* 2023) showed contrastive PLM-drug co-embedding for drug-target interactions. Neither combines PLM fitness scoring with structural binding-site information to design diagnostic panels for unsurveilled drugs.

4. **Pairwise epistasis from PLM masked marginals is unexplored.** The PNAS 2024 HIV kinetic coevolutionary model computed epistasis from Potts models fit to multiple sequence alignments. We compute epistasis directly from ESM-2 by scoring mutation B on a mutant-A background sequence — a novel application of masked marginal scoring.

---

## Methods

### ESM-2 Masked Marginal Log-Likelihood Ratio

For each amino acid substitution X*i*Y at position *i* of a protein, we compute:

```
LLR = log P(Y | context) − log P(X | context)
```

where context is the full protein sequence with position *i* masked, and probabilities are from the ESM-2 softmax output at the masked position (Meier et al., 2021). LLR < 0 indicates the wildtype residue is evolutionarily preferred; |LLR| quantifies the fitness cost of the substitution.

We use `esm2_t33_650M_UR50D` (33 transformer layers, 650M parameters, trained on UniRef50). Each masked marginal scoring requires one forward pass (~0.5s on RTX 2070 SUPER, ~2.6 GB VRAM).

### Pairwise Epistasis from Mutant Backgrounds

To compute the epistatic interaction between mutations A and B on the same protein:

```
epistasis(A, B) = LLR(B | seq_with_A) − LLR(B | wildtype_seq)
```

Mutation A is introduced into the protein sequence by direct substitution, then mutation B is scored using the standard masked marginal approach on the modified sequence. Positive epistasis indicates B is more tolerated when A is present (synergistic); negative indicates antagonism. This requires O(N²) forward passes for N mutations.

### Kinetic Monte Carlo Emergence Simulation

A Wright-Fisher population model with mutation and selection simulates resistance emergence under drug pressure. Genotype fitness combines additive ESM-2 LLR scores with pairwise epistatic corrections:

```
w(genotype) = exp(Σᵢ sᵢ + Σᵢ<ⱼ eᵢⱼ) × (1 + β · resistance)
```

where *sᵢ* is the selection coefficient derived from |LLR|, *eᵢⱼ* is the pairwise epistatic correction, and *β* modulates drug selection pressure. The simulation tracks genotype frequencies over generations and records first-emergence times for each mutation.

### De Novo Diagnostic Panel Design

A two-filter pipeline:

1. **Fitness filter (ESM-2):** Score all possible substitutions at drug-binding positions using `score_full_landscape()`. Retain mutations below the 75th percentile of |LLR| at binding positions (evolutionarily accessible).

2. **Resistance filter (structural):** Rank retained mutations by proximity to drug-binding residues (from co-crystal PDB structures). Mutations directly at binding positions are prioritized (Tier 1); flanking mutations ranked by combined fitness + proximity score (Tier 2).

The intersection — fit AND at the binding site — constitutes the predicted diagnostic panel, ranked by |LLR| within each tier.

---

## Results

### Experiment 1: Retrospective Validation (N = 81)

|ESM-2 LLR| correlates negatively with clinical prevalence across 81 amino acid substitutions in 22 drug-target genes from 4 bacterial species.

| Scope | N | Spearman ρ | Permutation *p* | 95% CI |
|-------|---|-----------|-----------------|--------|
| Pooled | 81 | −0.312 | 0.005 | [−0.491, −0.104] |
| *M. tuberculosis* | 57 | −0.271 | 0.042 | [−0.509, −0.011] |
| *N. gonorrhoeae* | 12 | +0.176 | 0.579 | NS |
| *S. aureus* | 8 | +0.575 | 0.144 | NS |

100% of computed LLR values are negative (wildtype preferred), with |LLR| ranging from 0.06 to 12.37.

### Experiment 2: Within-Gene Emergence Order Prediction

Using prevalence rank as a proxy for emergence order, we evaluate rank concordance (Kendall τ mapped to [0, 1]; 0.5 = random) for genes with ≥ 4 known resistance mutations.

| Gene | N | Rank concordance | Top-k precision | Mechanism |
|------|---|-----------------|-----------------|-----------|
| katG | 4 | **0.833** | 1.00 (k=1) | Conservative substitution |
| gyrA | 8 | **0.679** | 1.00 (k=2) | Conservative substitution |
| penA | 6 | **0.667** | 0.50 (k=2) | Conservative substitution |
| rpoB | 12 | 0.500 | 0.25 (k=4) | Structural pocket |
| pncA | 15 | 0.447 | 0.20 (k=5) | Loss of function |
| embB | 7 | 0.333 | 0.00 (k=2) | Structural pocket |

katG S315T is the single lowest-|LLR| substitution at position 315 (|LLR| = 0.68), while neighboring positions 316 and 318 have minimum |LLR| > 3.6 — a fitness valley in a constrained active-site region.

### Experiment 3: Three-Class Mechanistic Model

ESM-2 LLR predictive power is mechanism-dependent:

| Class | Mechanism | Examples | LLR predictive? | Panel design gap |
|-------|-----------|----------|-----------------|-----------------|
| 1 | Conservative active-site substitution | katG, gyrA, penA | Yes (concordance > 0.65) | 0% (k ≤ 2) |
| 2 | Loss of function | pncA, Rv0678, ddn | No (entire protein permissive) | N/A |
| 3 | Structural pocket clustering | rpoB RRDR, embB | No (mutations similarly costly) | 27–44% |

### Experiment 4: Pairwise Epistasis Networks

ESM-2-derived epistasis captures biologically meaningful interactions:

- **gyrA S91P → A90G:** epistasis = +2.30 (strongest synergistic pair; S91P makes A90G substantially more tolerated)
- **gyrA A90V → D94A:** epistasis = −0.71 (antagonistic; A90V increases cost of acquiring D94A)
- **rpoB** shows pervasive epistasis across the RRDR (mean |epistasis| > gyrA or katG), consistent with the known role of compensatory mutations (Comas et al., *Nature Genetics* 2012)

Emergence simulation on gyrA achieves concordance = 0.607–0.630 across targets.

### Experiment 5: De Novo Diagnostic Panel Design

Leave-one-drug-out validation: design panels using only protein sequence + binding site, compare to WHO catalogue.

| Target | Mechanism | Recall | Mutations recovered |
|--------|-----------|--------|-------------------|
| **katG** | Conservative | **0.75** | S315T, S315N, S315R |
| **embB** | Structural pocket | 0.43 | M306I, G406A, G406S |
| gyrA | Conservative | 0.12 | A90V |
| rpoB | Structural pocket | 0.00 | — |
| pncA | Loss of function | 0.00 | — |

**Pipeline drug predictions** (no clinical resistance data used):

- **BTZ043** (dprE1, Phase 2): predicted panel includes G397M, Y314H, D394R, C387S, I392A — all at PDB 6HEZ binding residues with |LLR| < 0.5
- **Telacebec** (qcrB, Phase 2): predicted panel includes A178E, A317T, G186A, T313A, M311L — all at PDB 6ADQ binding residues with |LLR| < 0.5

---

## Repository Structure

```
amr-esm-prediction/
├── models/
│   ├── esm2_scorer.py           # ESM-2 masked marginal scorer (score_mutation,
│   │                            #   score_batch, score_full_landscape, get_representations)
│   ├── epistasis_scorer.py      # Pairwise epistasis on mutant backgrounds
│   ├── emergence_simulator.py   # Kinetic Monte Carlo Wright-Fisher simulation
│   └── binding_disruption.py    # Structural + embedding binding disruption predictors
├── data/
│   ├── who_catalogue/           # 85 WHO AA substitutions across 4 organisms
│   ├── drug_targets/            # Binding sites, drug metadata, compensatory pairs
│   ├── lineage_backgrounds/     # MTB L1/L2/L4 polymorphisms
│   └── protein_sequences/       # 24 reference protein sequences
├── experiments/
│   ├── retrospective.py         # Exp 1: |LLR| vs prevalence correlation
│   ├── prospective.py           # Exp 2: within-gene emergence order prediction
│   ├── panel_design.py          # Exp 3: LLR-ranked vs prevalence-ranked panels
│   ├── emergence_forecast.py    # Exp 4: epistasis + KMC emergence simulation
│   └── denovo_design.py         # Exp 5: de novo panel design for pipeline drugs
├── scripts/
│   └── generate_figures.py      # Publication figures (Fig 2–7 + supplementary)
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

Requires Python ≥ 3.10, PyTorch ≥ 2.0, CUDA-capable GPU with ≥ 4 GB VRAM (tested on RTX 2070 SUPER 8 GB).

## Usage

```bash
# Experiment 1: Retrospective validation
python -m experiments.retrospective

# Experiment 2: Prospective emergence order
python -m experiments.prospective

# Experiment 3: Panel design comparison
python -m experiments.panel_design
python -m experiments.panel_design --full-landscape --organism mtb --gene katG

# Experiment 4: Epistasis + emergence forecasting
python -m experiments.emergence_forecast --organism mtb --gene gyrA
python -m experiments.emergence_forecast --organism mtb  # all MTB targets

# Experiment 5: De novo panel design
python -m experiments.denovo_design --leave-one-out
python -m experiments.denovo_design --pipeline  # BTZ043, telacebec predictions

# Generate all publication figures
python scripts/generate_figures.py
```

---

## Known Limitations

1. **Class 2/3 targets are not addressable by protein-level fitness alone.** Loss-of-function resistance (pncA, Rv0678) produces too many fitness-accessible candidates. Structural pocket clustering (rpoB RRDR) produces indistinguishable fitness costs. These require complementary information: drug-binding affinity predictions, experimental deep mutational scanning, or genome-context models.

2. **Non-protein resistance mechanisms are excluded.** Promoter mutations (fabG1 C-15T, eis C-14T), rRNA mutations (rrs A1401G), insertions/deletions, and horizontally acquired genes are not scorable by ESM-2. These account for ~30% of clinically relevant resistance in *M. tuberculosis*. DNA-level models (EVO-2, Caduceus) can address this gap.

3. **Genomic background is not modelled.** ESM-2 scores each protein in isolation. Lineage-specific polymorphisms, cross-gene epistasis, and genome-wide pre-resistance signatures (Torres Ortiz et al., 2021) are not captured. A genome-scale context model (e.g., JEPA-type architecture trained on bacterial whole genomes) would be required.

4. **Emergence simulation uses fitness as a proxy for resistance level.** The kinetic Monte Carlo model assumes that low-|LLR| mutations at binding sites confer resistance. Without quantitative MIC predictions, we cannot distinguish mutations that are fit but non-resistant from those that are both fit and resistant — as demonstrated by the gyrA full-landscape scan (D94E has |LLR| = 0.08 but is not WHO-catalogued).

5. **Validation is retrospective.** We use prevalence rank as a proxy for emergence order. True prospective validation requires longitudinal WGS data with first-detection timestamps (CRyPTIC, Euro-GASP). The Rv0678/bedaquiline case (drug approved 2012, mutations catalogued 2022) is illustrative but anecdotal (N = 3).

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

### Epistasis and Fitness Landscapes

- Rodriguez de Evgrafov MC, et al. (2024). Kinetic coevolutionary models predict drug-resistance mutation acquisition rates in HIV. *PNAS*, 121(19). doi:10.1073/pnas.2316662121
- Ogbunugafor CB, et al. (2023). Environmental modulation of global epistasis in a drug resistance fitness landscape. *Nature Communications*, 14:8055. doi:10.1038/s41467-023-43806-x
- Comas I, Borrell S, Roetzer A, et al. (2012). Whole-genome sequencing of rifampicin-resistant *Mycobacterium tuberculosis* strains identifies compensatory mutations in RNA polymerase genes. *Nature Genetics*, 44(1):106–110. doi:10.1038/ng.1038

### Pre-Resistance and Temporal Emergence

- Torres Ortiz A, Grandjean L, et al. (2021). Genomic signatures of pre-resistance in *Mycobacterium tuberculosis*. *Nature Communications*, 12:7273. doi:10.1038/s41467-021-27616-7
- Andersson DI, Hughes D (2010). Antibiotic resistance and its cost: is it possible to reverse resistance? *Nature Reviews Microbiology*, 8:260–271. doi:10.1038/nrmicro2319
- Gagneux S, Long CD, Small PM, et al. (2006). The competitive cost of antibiotic resistance in *Mycobacterium tuberculosis*. *Science*, 312(5782):1944–1946. doi:10.1126/science.1124410

### Drug-Target Interaction and Diagnostics

- Singh R, et al. (2023). Contrastive learning in protein language model space predicts interactions between drugs and protein targets. *PNAS*, 120(24):e2220778120. doi:10.1073/pnas.2220778120
- The CRyPTIC Consortium (2022). A data compendium associating the genomes of 12,289 *Mycobacterium tuberculosis* isolates with quantitative resistance phenotypes to 13 antibiotics. *PLOS Biology*, 20(8):e3001721. doi:10.1371/journal.pbio.3001721
- WHO (2023). Catalogue of mutations in *Mycobacterium tuberculosis* complex and their association with drug resistance, 2nd edition. Geneva: World Health Organization.

### Structural Data

- Boyaci H, et al. (2018). Fidaxomicin jams *Mycobacterium tuberculosis* RNA polymerase motions needed for initiation via RbpA contacts. *eLife*, 7:e34823. PDB: 5UHC (rpoB-rifampicin).
- Makarov V, et al. (2014). Towards a new combination therapy for tuberculosis with next generation benzothiazinones. *EMBO Molecular Medicine*, 6(3):372–383. PDB: 6HEZ (dprE1-BTZ043).
- Pethe K, et al. (2013). Discovery of Q203, a potent clinical candidate for the treatment of tuberculosis. *Nature Medicine*, 19(9):1157–1160. PDB: 6ADQ (qcrB-telacebec).

---

## License

MIT
