"""Compensatory mutation pairs for AMR fitness restoration.

When a resistance mutation imposes a fitness cost, compensatory mutations
at secondary sites can restore fitness without reverting resistance.
These create evolutionary "pathways" that lock in resistance.

Sources:
    rpoB-rpoC: Comas et al. (Nat Genet 2012); Sherman et al. (J Bacteriol 2001)
    katG-ahpC: Sherman et al. (J Infect Dis 1996)
    gyrA-gyrB: Aubry et al. (AAC 2006)
    rpsL-rrs: Sander et al. (Mol Microbiol 2002)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompensatoryPair:
    """A primary resistance gene and its compensatory partner."""

    organism: str
    primary_gene: str
    compensatory_gene: str
    known_compensatory_mutations: list[str]
    mechanism: str
    reference: str

    @property
    def key(self) -> str:
        return f"{self.organism}_{self.primary_gene}_{self.compensatory_gene}"


COMPENSATORY_PAIRS: list[CompensatoryPair] = [
    # ── rpoB → rpoC/rpoA (rifampicin) ──
    # rpoB RRDR mutations (especially S450L) impose fitness cost on RNA polymerase.
    # Compensatory mutations in rpoC (beta' subunit) or rpoA (alpha subunit)
    # restore transcription efficiency.
    CompensatoryPair(
        organism="mtb",
        primary_gene="rpoB",
        compensatory_gene="rpoC",
        known_compensatory_mutations=["V483A", "V483G", "W484G", "N698S"],
        mechanism="Restore RNA polymerase processivity after rpoB RRDR disruption",
        reference="Comas et al., Nat Genet 2012; Brandis & Hughes, MBE 2013",
    ),
    # ── katG → ahpC (isoniazid) ──
    # katG S315T reduces catalase-peroxidase activity. ahpC promoter upregulation
    # compensates for loss of oxidative stress defense.
    CompensatoryPair(
        organism="mtb",
        primary_gene="katG",
        compensatory_gene="ahpC",
        known_compensatory_mutations=[],  # Promoter mutations, not AA substitutions
        mechanism="Upregulate alkyl hydroperoxide reductase to compensate katG loss",
        reference="Sherman et al., J Infect Dis 1996",
    ),
    # ── gyrA → gyrB (fluoroquinolones) ──
    # gyrA QRDR mutations alter DNA gyrase. gyrB mutations may compensate
    # or add resistance.
    CompensatoryPair(
        organism="mtb",
        primary_gene="gyrA",
        compensatory_gene="gyrB",
        known_compensatory_mutations=["E501D", "N499D"],
        mechanism="Restore gyrase function or add FQ resistance",
        reference="Aubry et al., AAC 2006",
    ),
    # ── rpsL → rrs (streptomycin) ──
    CompensatoryPair(
        organism="mtb",
        primary_gene="rpsL",
        compensatory_gene="rrs",
        known_compensatory_mutations=[],  # rRNA mutations
        mechanism="Restore ribosome function",
        reference="Sander et al., Mol Microbiol 2002",
    ),
]


def get_compensatory_pairs(organism: str, primary_gene: str) -> list[CompensatoryPair]:
    """Get all compensatory pairs for a given primary resistance gene."""
    return [
        p for p in COMPENSATORY_PAIRS
        if p.organism == organism and p.primary_gene == primary_gene
    ]
