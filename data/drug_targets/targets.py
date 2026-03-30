"""Drug target metadata — binding sites, resistance mechanisms, drug info.

Curated from crystal structures, WHO catalogue, and published literature.
Each DrugTarget links a gene to its drug, known binding residues, and
resistance mechanism classification (from our three-class model).

Sources:
    rpoB-RIF: PDB 5UHC (Boyaci et al., Nature 2018)
    gyrA-FQ: PDB 5BS8 (Blower et al., PNAS 2016)
    katG-INH: Zhao et al., JACS 2006 (catalytic mechanism)
    embB-EMB: Safi et al., AAC 2008 (codon 306 binding)
    pncA-PZA: Petrella et al., PLOS ONE 2011 (active site)
    Rv0678-BDQ: Hartkoorn et al., Nature Med 2014
    dprE1-BTZ043: PDB 6HEZ (Makarov et al., EMBO 2014)
    qcrB-telacebec: PDB 6ADQ (Pethe et al., Nature Med 2013)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DrugTarget:
    """A drug target gene with binding site and mechanism metadata."""

    organism: str
    gene: str
    drug: str
    drug_class: str
    binding_residues: list[int]
    resistance_mechanism: str  # "conservative_substitution", "loss_of_function", "structural_pocket"
    pdb_id: str | None = None
    mic_wildtype_ug_ml: float | None = None
    description: str = ""

    @property
    def key(self) -> str:
        return f"{self.organism}_{self.gene}"


# Validated drug targets (existing drugs with known resistance)
DRUG_TARGETS: list[DrugTarget] = [
    # ── M. tuberculosis ──
    DrugTarget(
        organism="mtb", gene="rpoB", drug="rifampicin", drug_class="rifamycin",
        binding_residues=list(range(430, 460)),  # RRDR codons 430-460
        resistance_mechanism="structural_pocket",
        pdb_id="5UHC", mic_wildtype_ug_ml=0.5,
        description="RNA polymerase beta subunit; RIF binds in RRDR pocket",
    ),
    DrugTarget(
        organism="mtb", gene="katG", drug="isoniazid", drug_class="isonicotinic_acid_hydrazide",
        binding_residues=[315],  # S315 is the catalytic site
        resistance_mechanism="conservative_substitution",
        pdb_id=None, mic_wildtype_ug_ml=0.1,
        description="Catalase-peroxidase; INH prodrug activator; S315T is dominant",
    ),
    DrugTarget(
        organism="mtb", gene="gyrA", drug="moxifloxacin", drug_class="fluoroquinolone",
        binding_residues=list(range(88, 96)),  # QRDR codons 88-95
        resistance_mechanism="conservative_substitution",
        pdb_id="5BS8", mic_wildtype_ug_ml=0.25,
        description="DNA gyrase subunit A; FQ binds at QRDR",
    ),
    DrugTarget(
        organism="mtb", gene="embB", drug="ethambutol", drug_class="ethylenediamine",
        binding_residues=[306, 354, 406, 497],
        resistance_mechanism="structural_pocket",
        pdb_id=None, mic_wildtype_ug_ml=2.5,
        description="Arabinosyltransferase; EMB binds near codon 306",
    ),
    DrugTarget(
        organism="mtb", gene="pncA", drug="pyrazinamide", drug_class="pyrazine",
        binding_residues=list(range(1, 187)),  # Entire enzyme (186 aa)
        resistance_mechanism="loss_of_function",
        pdb_id=None, mic_wildtype_ug_ml=50.0,
        description="Pyrazinamidase; PZA prodrug activator; LOF confers resistance",
    ),
    DrugTarget(
        organism="mtb", gene="Rv0678", drug="bedaquiline", drug_class="diarylquinoline",
        binding_residues=list(range(1, 166)),  # Entire regulator (165 aa)
        resistance_mechanism="loss_of_function",
        pdb_id=None, mic_wildtype_ug_ml=0.25,
        description="MmpR5 transcriptional repressor; LOF derepresses efflux pump",
    ),
    DrugTarget(
        organism="mtb", gene="ddn", drug="delamanid", drug_class="nitroimidazole",
        binding_residues=list(range(1, 152)),  # Entire enzyme (151 aa)
        resistance_mechanism="loss_of_function",
        pdb_id=None, mic_wildtype_ug_ml=0.006,
        description="Deazaflavin-dependent nitroreductase; LOF confers resistance",
    ),
    # ── E. coli ──
    DrugTarget(
        organism="ecoli", gene="gyrA", drug="ciprofloxacin", drug_class="fluoroquinolone",
        binding_residues=list(range(81, 88)),
        resistance_mechanism="conservative_substitution",
        pdb_id=None, mic_wildtype_ug_ml=0.008,
        description="DNA gyrase subunit A; QRDR",
    ),
    # ── N. gonorrhoeae ──
    DrugTarget(
        organism="ngonorrhoeae", gene="penA", drug="ceftriaxone", drug_class="cephalosporin",
        binding_residues=list(range(310, 550)),  # PBP2 transpeptidase domain
        resistance_mechanism="conservative_substitution",
        pdb_id=None, mic_wildtype_ug_ml=0.004,
        description="PBP2; mosaic alleles from commensal Neisseria",
    ),
]

# Pipeline drug targets (for de novo prediction — no clinical resistance data)
PIPELINE_TARGETS: list[DrugTarget] = [
    DrugTarget(
        organism="mtb", gene="dprE1", drug="BTZ043", drug_class="benzothiazinone",
        binding_residues=[314, 316, 367, 387, 389, 392, 394, 397],  # From PDB 6HEZ
        resistance_mechanism="conservative_substitution",
        pdb_id="6HEZ", mic_wildtype_ug_ml=0.001,
        description="Decaprenylphosphoryl-beta-D-ribose oxidase; BTZ043 covalent inhibitor; Phase 2",
    ),
    DrugTarget(
        organism="mtb", gene="qcrB", drug="telacebec", drug_class="imidazopyridine",
        binding_residues=[173, 175, 176, 178, 182, 186, 311, 313, 317],  # From PDB 6ADQ
        resistance_mechanism="conservative_substitution",
        pdb_id="6ADQ", mic_wildtype_ug_ml=0.003,
        description="Cytochrome bc1 complex subunit B; Q203/telacebec target; Phase 2",
    ),
]


def get_target(organism: str, gene: str) -> DrugTarget | None:
    """Look up a drug target by organism and gene."""
    for t in DRUG_TARGETS + PIPELINE_TARGETS:
        if t.organism == organism and t.gene == gene:
            return t
    return None


def get_targets_by_mechanism(mechanism: str) -> list[DrugTarget]:
    """Get all targets with a given resistance mechanism."""
    return [t for t in DRUG_TARGETS if t.resistance_mechanism == mechanism]
