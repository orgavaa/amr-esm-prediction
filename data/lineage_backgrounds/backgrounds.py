"""MTB lineage-specific protein polymorphisms.

ESM-2 masked marginal scores depend on the full sequence context.
Different MTB lineages carry fixed polymorphisms that change this context,
potentially producing lineage-specific fitness predictions.

Torres Ortiz et al. (Nat Comm 2021) showed that lineage 2 (Beijing) has
higher hazard of acquiring resistance than lineage 4 (Euro-American),
partially explained by pre-existing polymorphisms.

Sources:
    Comas et al. (Nat Genet 2013): MTB phylogeography
    Torres Ortiz et al. (Nat Comm 2021): Pre-resistance signatures
    Stucki et al. (Nat Genet 2016): Lineage-specific adaptation
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LineageBackground:
    """Lineage-specific polymorphisms in drug target genes."""

    organism: str
    lineage: str
    lineage_name: str
    gene_polymorphisms: dict[str, list[str]]  # gene -> list of fixed AA changes
    description: str


MTB_LINEAGE_BACKGROUNDS: list[LineageBackground] = [
    LineageBackground(
        organism="mtb",
        lineage="L2",
        lineage_name="Beijing",
        gene_polymorphisms={
            # L2 (Beijing) lineage-defining polymorphisms in drug target genes
            # These are FIXED differences from H37Rv (L4 reference)
            "gyrA": [],      # QRDR is conserved across lineages
            "rpoB": [],      # RRDR is conserved
            "katG": ["R463L"],  # Common L2 polymorphism — near but not at S315
            "pncA": [],      # Highly variable even within lineages
            "embB": [],
        },
        description=(
            "Beijing lineage (East Asia, Russia). Higher mutation rate, "
            "associated with drug resistance outbreaks. katG R463L is "
            "a lineage marker, not a resistance mutation."
        ),
    ),
    LineageBackground(
        organism="mtb",
        lineage="L4",
        lineage_name="Euro-American",
        gene_polymorphisms={
            "gyrA": [],
            "rpoB": [],
            "katG": [],   # H37Rv is L4 — this is the reference
            "pncA": [],
            "embB": [],
        },
        description=(
            "Euro-American lineage (global). H37Rv reference strain belongs "
            "to L4. Our wildtype protein sequences are L4 by definition."
        ),
    ),
    LineageBackground(
        organism="mtb",
        lineage="L1",
        lineage_name="Indo-Oceanic",
        gene_polymorphisms={
            "gyrA": [],
            "rpoB": [],
            "katG": ["R463L"],  # Shared with L2
            "pncA": [],
            "embB": [],
        },
        description=(
            "Indo-Oceanic lineage (South/Southeast Asia). Less associated "
            "with drug resistance than L2."
        ),
    ),
]


def get_backgrounds(organism: str) -> list[LineageBackground]:
    """Get all lineage backgrounds for an organism."""
    if organism == "mtb":
        return MTB_LINEAGE_BACKGROUNDS
    return []
