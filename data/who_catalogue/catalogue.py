"""WHO AMR mutation catalogue — amino acid substitutions with clinical prevalence.

Sources:
    MTB: WHO catalogue of mutations v2 (2023), doi:10.1016/S2666-5247(22)00116-1
    E. coli: CARD database + EUCAST breakpoint tables
    S. aureus: CARD + CLSI M100
    N. gonorrhoeae: WHO Gonococcal AMR Surveillance Programme (GASP)

Each entry: (organism, gene, mutation, drug, prevalence_pct)
    - prevalence_pct: approximate % of resistant isolates carrying this mutation
    - Only includes amino acid substitutions (ESM-2 applicable)
    - Excludes: promoter mutations, rRNA mutations, indels
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AMRMutation:
    """A single AMR-associated amino acid substitution."""

    organism: str
    gene: str
    mutation: str  # e.g. "S450L"
    drug: str
    prevalence_pct: float
    who_tier: int = 0  # 1=associated with resistance, 2=uncertain significance

    @property
    def label(self) -> str:
        return f"{self.gene}_{self.mutation}"

    @property
    def organism_gene(self) -> str:
        return f"{self.organism}_{self.gene}"


# Expanded WHO catalogue: 85 AA substitutions across 4 organisms
WHO_AA_MUTATIONS: list[AMRMutation] = [
    # ==========================================================================
    # M. tuberculosis (WHO 2023 catalogue v2)
    # ==========================================================================
    # rpoB — Rifampicin Resistance-Determining Region (RRDR)
    AMRMutation("mtb", "rpoB", "S450L", "RIF", 42.0, 1),
    AMRMutation("mtb", "rpoB", "H445Y", "RIF", 8.5, 1),
    AMRMutation("mtb", "rpoB", "D435V", "RIF", 5.2, 1),
    AMRMutation("mtb", "rpoB", "H445D", "RIF", 4.1, 1),
    AMRMutation("mtb", "rpoB", "H445N", "RIF", 2.3, 1),
    AMRMutation("mtb", "rpoB", "S450W", "RIF", 1.8, 1),
    AMRMutation("mtb", "rpoB", "L430P", "RIF", 1.5, 1),
    AMRMutation("mtb", "rpoB", "D435Y", "RIF", 1.2, 1),
    AMRMutation("mtb", "rpoB", "H445L", "RIF", 1.1, 1),
    AMRMutation("mtb", "rpoB", "H445R", "RIF", 0.8, 1),
    AMRMutation("mtb", "rpoB", "S450F", "RIF", 0.5, 1),
    AMRMutation("mtb", "rpoB", "L452P", "RIF", 0.4, 1),
    # katG — Isoniazid
    AMRMutation("mtb", "katG", "S315T", "INH", 64.0, 1),
    AMRMutation("mtb", "katG", "S315N", "INH", 2.1, 1),
    AMRMutation("mtb", "katG", "S315G", "INH", 0.7, 1),
    AMRMutation("mtb", "katG", "S315R", "INH", 0.3, 1),
    # embB — Ethambutol
    AMRMutation("mtb", "embB", "M306V", "EMB", 24.0, 1),
    AMRMutation("mtb", "embB", "M306I", "EMB", 18.0, 1),
    AMRMutation("mtb", "embB", "M306L", "EMB", 5.0, 1),
    AMRMutation("mtb", "embB", "G406D", "EMB", 3.5, 1),
    AMRMutation("mtb", "embB", "G406S", "EMB", 1.2, 1),
    AMRMutation("mtb", "embB", "G406A", "EMB", 0.8, 1),
    AMRMutation("mtb", "embB", "Q497R", "EMB", 2.5, 1),
    # pncA — Pyrazinamide (high diversity)
    AMRMutation("mtb", "pncA", "H57D", "PZA", 3.0, 1),
    AMRMutation("mtb", "pncA", "D49N", "PZA", 2.5, 2),
    AMRMutation("mtb", "pncA", "T135P", "PZA", 1.8, 1),
    AMRMutation("mtb", "pncA", "L4S", "PZA", 1.5, 1),
    AMRMutation("mtb", "pncA", "H71Y", "PZA", 1.2, 1),
    AMRMutation("mtb", "pncA", "D12A", "PZA", 1.0, 1),
    AMRMutation("mtb", "pncA", "V125G", "PZA", 0.8, 1),
    AMRMutation("mtb", "pncA", "C14R", "PZA", 0.6, 1),
    AMRMutation("mtb", "pncA", "I31S", "PZA", 0.5, 1),
    AMRMutation("mtb", "pncA", "W68G", "PZA", 0.4, 1),
    AMRMutation("mtb", "pncA", "G97D", "PZA", 0.3, 1),
    AMRMutation("mtb", "pncA", "V139A", "PZA", 0.3, 1),
    AMRMutation("mtb", "pncA", "T76P", "PZA", 0.2, 1),
    AMRMutation("mtb", "pncA", "Q10P", "PZA", 0.2, 1),
    AMRMutation("mtb", "pncA", "A134V", "PZA", 0.2, 1),
    # gyrA — Fluoroquinolones (QRDR)
    AMRMutation("mtb", "gyrA", "D94G", "FQ", 28.0, 1),
    AMRMutation("mtb", "gyrA", "A90V", "FQ", 18.0, 1),
    AMRMutation("mtb", "gyrA", "D94A", "FQ", 5.0, 1),
    AMRMutation("mtb", "gyrA", "D94N", "FQ", 4.5, 1),
    AMRMutation("mtb", "gyrA", "D94Y", "FQ", 3.0, 1),
    AMRMutation("mtb", "gyrA", "D94H", "FQ", 2.0, 1),
    AMRMutation("mtb", "gyrA", "S91P", "FQ", 1.5, 1),
    AMRMutation("mtb", "gyrA", "A90G", "FQ", 0.5, 1),
    # gyrB — Fluoroquinolones
    AMRMutation("mtb", "gyrB", "E501D", "FQ", 1.0, 2),
    AMRMutation("mtb", "gyrB", "N499D", "FQ", 0.5, 2),
    # rpsL — Streptomycin
    AMRMutation("mtb", "rpsL", "K43R", "STR", 40.0, 1),
    AMRMutation("mtb", "rpsL", "K88R", "STR", 8.0, 1),
    # ethA — Ethionamide
    AMRMutation("mtb", "ethA", "A381P", "ETH", 1.5, 2),
    # Rv0678 — Bedaquiline
    AMRMutation("mtb", "Rv0678", "V1A", "BDQ", 0.5, 1),
    AMRMutation("mtb", "Rv0678", "S53L", "BDQ", 0.3, 1),
    AMRMutation("mtb", "Rv0678", "M1R", "BDQ", 0.2, 1),
    # ddn — Delamanid
    AMRMutation("mtb", "ddn", "L49P", "DLM", 1.0, 1),
    AMRMutation("mtb", "ddn", "W88C", "DLM", 0.5, 1),
    AMRMutation("mtb", "ddn", "Y133D", "DLM", 0.3, 1),

    # ==========================================================================
    # E. coli
    # ==========================================================================
    AMRMutation("ecoli", "gyrA", "S83L", "CIP", 70.0, 1),
    AMRMutation("ecoli", "gyrA", "D87N", "CIP", 25.0, 1),
    AMRMutation("ecoli", "parC", "S80I", "CIP", 45.0, 2),
    AMRMutation("ecoli", "parC", "E84V", "CIP", 5.0, 2),

    # ==========================================================================
    # S. aureus
    # ==========================================================================
    AMRMutation("saureus", "gyrA", "S84L", "CIP", 60.0, 1),
    AMRMutation("saureus", "grlA", "S80F", "CIP", 55.0, 1),
    AMRMutation("saureus", "grlA", "S80Y", "CIP", 10.0, 1),
    AMRMutation("saureus", "rpoB", "H481N", "RIF", 15.0, 1),
    AMRMutation("saureus", "rpoB", "S464P", "RIF", 5.0, 1),
    AMRMutation("saureus", "fusA", "L461K", "FUS", 3.0, 2),
    AMRMutation("saureus", "dfrB", "F99Y", "SXT", 8.0, 2),
    AMRMutation("saureus", "mprF", "S295L", "DAP", 5.0, 2),

    # ==========================================================================
    # N. gonorrhoeae
    # ==========================================================================
    AMRMutation("ngonorrhoeae", "penA", "A501V", "CRO", 15.0, 1),
    AMRMutation("ngonorrhoeae", "penA", "A501T", "CRO", 5.0, 1),
    AMRMutation("ngonorrhoeae", "penA", "G545S", "CRO", 8.0, 1),
    AMRMutation("ngonorrhoeae", "penA", "I312M", "CRO", 20.0, 2),
    AMRMutation("ngonorrhoeae", "penA", "V316T", "CRO", 12.0, 2),
    AMRMutation("ngonorrhoeae", "penA", "T483S", "CRO", 10.0, 2),
    AMRMutation("ngonorrhoeae", "gyrA", "S91F", "CIP", 65.0, 1),
    AMRMutation("ngonorrhoeae", "gyrA", "D95A", "CIP", 10.0, 1),
    AMRMutation("ngonorrhoeae", "gyrA", "D95G", "CIP", 15.0, 1),
    AMRMutation("ngonorrhoeae", "parC", "D86N", "CIP", 20.0, 2),
    AMRMutation("ngonorrhoeae", "parC", "S87R", "CIP", 30.0, 2),
    AMRMutation("ngonorrhoeae", "folP", "R228S", "SXT", 25.0, 2),
]


def get_mutations_by_organism(organism: str) -> list[AMRMutation]:
    """Filter catalogue by organism."""
    return [m for m in WHO_AA_MUTATIONS if m.organism == organism]


def get_mutations_by_gene(organism: str, gene: str) -> list[AMRMutation]:
    """Filter catalogue by organism + gene."""
    return [
        m for m in WHO_AA_MUTATIONS
        if m.organism == organism and m.gene == gene
    ]


def get_unique_genes() -> list[tuple[str, str]]:
    """Get unique (organism, gene) pairs in the catalogue."""
    seen = set()
    result = []
    for m in WHO_AA_MUTATIONS:
        key = (m.organism, m.gene)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result
