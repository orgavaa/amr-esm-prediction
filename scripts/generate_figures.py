"""Generate all publication figures for ESM-2 AMR fitness landscape paper.

Figures:
    Fig 2: Retrospective — |LLR| vs clinical prevalence scatter
    Fig 3: Fitness landscapes — katG (A,B) and gyrA QRDR (C) heatmaps
    Fig 4: Three-class model — concordance by resistance mechanism
    Fig 5: Panel design — LLR-ranked vs prevalence-ranked coverage curves

Usage:
    python scripts/generate_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ORG_LABELS = {
    "mtb": "M. tuberculosis",
    "ecoli": "E. coli",
    "saureus": "S. aureus",
    "ngonorrhoeae": "N. gonorrhoeae",
}
ORG_COLORS = {
    "mtb": "#E64B35",
    "ecoli": "#4DBBD5",
    "saureus": "#00A087",
    "ngonorrhoeae": "#3C5488",
}
ORG_MARKERS = {
    "mtb": "o",
    "ecoli": "s",
    "saureus": "D",
    "ngonorrhoeae": "^",
}


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Retrospective — |LLR| vs clinical prevalence
# ═══════════════════════════════════════════════════════════════════════
def fig2_retrospective():
    df = pd.read_csv(RESULTS / "retrospective" / "llr_results.csv")
    df = df[df["esm2_llr"].notna()].copy()
    df["abs_llr"] = df["esm2_llr"].abs()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 2]})

    # ── Panel A: Scatter ──
    ax = axes[0]
    for org_id in ["mtb", "ecoli", "saureus", "ngonorrhoeae"]:
        g = df[df["organism"] == org_id]
        if len(g) == 0:
            continue
        ax.scatter(
            g["abs_llr"], g["prevalence_pct"],
            label=f'{ORG_LABELS[org_id]} (n={len(g)})',
            color=ORG_COLORS[org_id],
            marker=ORG_MARKERS[org_id],
            s=45, alpha=0.75, edgecolors="white", linewidth=0.4, zorder=3,
        )

    # Trend line (LOWESS or simple)
    from numpy.polynomial.polynomial import polyfit
    x_all = df["abs_llr"].values
    y_all = df["prevalence_pct"].values
    rho, p = spearmanr(x_all, y_all)

    # Regression line for visual
    coeffs = polyfit(x_all, y_all, 1)
    x_line = np.linspace(0, x_all.max() * 1.05, 100)
    ax.plot(x_line, coeffs[0] + coeffs[1] * x_line, "--", color="gray",
            linewidth=1.2, alpha=0.6, zorder=1)

    # Annotate key mutations
    highlights = [
        ("katG_S315T", "S315T", 8, 12),
        ("gyrA_S83L", "gyrA S83L", 6, -10),
    ]
    for label_col, text, dx, dy in highlights:
        row = df[df["mutation"] == text.split()[-1]]
        if len(row) == 0:
            # Try gene_mutation label
            row = df[(df["gene"] + "_" + df["mutation"]) == label_col]
        if len(row) > 0:
            r = row.iloc[0]
            ax.annotate(
                text, xy=(r["abs_llr"], r["prevalence_pct"]),
                xytext=(r["abs_llr"] + dx * 0.08, r["prevalence_pct"] + dy * 0.5),
                fontsize=8, fontstyle="italic", color="gray",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

    ax.set_xlabel("|ESM-2 LLR|  (evolutionary fitness cost)")
    ax.set_ylabel("Clinical prevalence (%)")
    ax.set_title(f"A) |ESM-2 LLR| vs clinical prevalence\n"
                 f"Spearman $\\rho$ = {rho:.3f}, p = {p:.4f}, N = {len(df)}")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(left=-0.3)
    ax.set_ylim(bottom=-2)

    # ── Panel B: Per-organism bar ──
    ax2 = axes[1]
    with open(RESULTS / "retrospective" / "correlation_analysis.json") as f:
        corr = json.load(f)

    orgs = []
    rhos_bar = []
    colors_bar = []
    ns = []
    sigs = []

    for org_id in ["mtb", "ngonorrhoeae", "saureus"]:
        if org_id in corr["per_organism"]:
            d = corr["per_organism"][org_id]
            orgs.append(ORG_LABELS[org_id])
            rhos_bar.append(d["spearman_rho"])
            colors_bar.append(ORG_COLORS[org_id])
            ns.append(d["n"])
            sigs.append(d.get("significant", False))

    # Add pooled
    if corr["pooled"]:
        orgs.append("Pooled")
        rhos_bar.append(corr["pooled"]["spearman_rho"])
        colors_bar.append("#333333")
        ns.append(corr["pooled"]["n"])
        sigs.append(corr["pooled"]["permutation_p"] < 0.05)

    bars = ax2.barh(range(len(orgs)), rhos_bar, color=colors_bar, edgecolor="black",
                    linewidth=0.5, height=0.6)
    ax2.set_yticks(range(len(orgs)))
    ax2.set_yticklabels(orgs, fontsize=10)
    ax2.set_xlabel("Spearman $\\rho$")
    ax2.set_title("B) Per-organism correlation")
    ax2.axvline(0, color="black", linewidth=0.5)

    for i, (bar, n, sig) in enumerate(zip(bars, ns, sigs)):
        x_pos = bar.get_width()
        sign = 1 if x_pos >= 0 else -1
        label = f"n={n}" + (" *" if sig else "")
        ax2.text(x_pos + sign * 0.02, bar.get_y() + bar.get_height() / 2,
                 label, va="center", ha="left" if x_pos >= 0 else "right", fontsize=9)

    ax2.set_xlim(-0.7, 0.7)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_retrospective.png")
    plt.savefig(FIGURES / "fig2_retrospective.pdf")
    plt.close()
    print("  Fig 2 saved.")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Fitness landscapes — katG + gyrA
# ═══════════════════════════════════════════════════════════════════════
def fig3_landscapes():
    with open(RESULTS / "panel_design" / "landscape_mtb_katG.json") as f:
        katg = json.load(f)
    with open(RESULTS / "panel_design" / "landscape_mtb_gyrA_qrdr.json") as f:
        gyra = json.load(f)
    with open("data/protein_sequences/sequences.json") as f:
        seqs = json.load(f)

    katg_prot = seqs["mtb_katG"]
    gyra_prot = seqs["mtb_gyrA"]
    AA = "ACDEFGHIKLMNPQRSTVWY"

    pos_stats = {int(k): v for k, v in katg["position_stats"].items()}
    positions = sorted(pos_stats.keys())
    min_llr = np.array([pos_stats[p]["min_abs_llr"] for p in positions])

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 2, 2], width_ratios=[1, 1],
                           hspace=0.35, wspace=0.3)

    # ── Panel A: katG full position plot (spans both columns) ──
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.fill_between(positions, min_llr, alpha=0.25, color="steelblue")
    ax_a.plot(positions, min_llr, linewidth=0.4, color="steelblue", alpha=0.8)

    # Highlight active site
    ax_a.axvspan(310, 320, alpha=0.12, color="#E64B35", label="Active site (310–320)")

    # S315T annotation
    p315 = positions.index(315)
    ax_a.scatter([315], [min_llr[p315]], color="#E64B35", s=120, zorder=5,
                 edgecolors="black", linewidth=1)
    ax_a.annotate(
        "S315T\n|LLR| = 0.68",
        xy=(315, min_llr[p315]),
        xytext=(380, min_llr[p315] + 2.0),
        arrowprops=dict(arrowstyle="-|>", color="#E64B35", lw=1.5),
        fontsize=12, color="#E64B35", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E64B35", alpha=0.9),
    )

    ax_a.set_xlabel("katG residue position", fontsize=12)
    ax_a.set_ylabel("Min |ESM-2 LLR|\n(lowest-cost substitution)", fontsize=12)
    ax_a.set_title("A)  katG fitness landscape — S315T is the evolutionary escape route",
                    fontsize=14, fontweight="bold", loc="left")
    ax_a.legend(fontsize=10, loc="upper right")
    ax_a.set_xlim(1, 740)
    ax_a.set_ylim(bottom=0)

    # ── Panel B: katG active site heatmap ──
    ax_b = fig.add_subplot(gs[1, 0])
    region_b = list(range(310, 321))
    hm_b = np.full((len(AA), len(region_b)), np.nan)
    labels_b = []
    for j, pos in enumerate(region_b):
        wt = katg_prot[pos - 1]
        labels_b.append(f"{wt}{pos}")
        for mut in katg["all_mutations"]:
            if mut["position"] == pos:
                hm_b[AA.index(mut["alt"]), j] = mut["abs_llr"]

    im_b = ax_b.imshow(np.clip(hm_b, 0, 12), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=12)

    # WHO mutations
    for alt, pos in [("T", 315), ("N", 315), ("G", 315), ("R", 315)]:
        ax_b.plot(region_b.index(pos), AA.index(alt), "k*", markersize=11)
    # Wildtype
    for j, pos in enumerate(region_b):
        wt = katg_prot[pos - 1]
        if wt in AA:
            ax_b.plot(j, AA.index(wt), "ws", markersize=7,
                      markeredgecolor="black", markeredgewidth=0.8)

    ax_b.set_xticks(range(len(region_b)))
    ax_b.set_xticklabels(labels_b, fontsize=8, rotation=45, ha="right")
    ax_b.set_yticks(range(len(AA)))
    ax_b.set_yticklabels(list(AA), fontsize=7, fontfamily="monospace")
    ax_b.set_ylabel("Substituted AA")
    ax_b.set_title("B)  katG active site (310–320)", fontweight="bold", loc="left")
    plt.colorbar(im_b, ax=ax_b, shrink=0.7, pad=0.02, label="|LLR|")

    # ── Panel C: gyrA QRDR heatmap ──
    ax_c = fig.add_subplot(gs[1, 1])
    region_c = list(range(88, 96))
    hm_c = np.full((len(AA), len(region_c)), np.nan)
    labels_c = []
    for j, pos in enumerate(region_c):
        wt = gyra_prot[pos - 1]
        labels_c.append(f"{wt}{pos}")
        for mut in gyra["all_mutations"]:
            if mut["position"] == pos:
                hm_c[AA.index(mut["alt"]), j] = mut["abs_llr"]

    im_c = ax_c.imshow(np.clip(hm_c, 0, 12), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=12)

    who_gyra = [("V", 90), ("G", 90), ("P", 91), ("G", 94), ("A", 94),
                ("N", 94), ("Y", 94), ("H", 94)]
    for alt, pos in who_gyra:
        if pos in region_c:
            ax_c.plot(region_c.index(pos), AA.index(alt), "k*", markersize=11)
    for j, pos in enumerate(region_c):
        wt = gyra_prot[pos - 1]
        if wt in AA:
            ax_c.plot(j, AA.index(wt), "ws", markersize=7,
                      markeredgecolor="black", markeredgewidth=0.8)

    ax_c.set_xticks(range(len(region_c)))
    ax_c.set_xticklabels(labels_c, fontsize=8, rotation=45, ha="right")
    ax_c.set_yticks(range(len(AA)))
    ax_c.set_yticklabels(list(AA), fontsize=7, fontfamily="monospace")
    ax_c.set_ylabel("Substituted AA")
    ax_c.set_title("C)  gyrA QRDR (88–95)", fontweight="bold", loc="left")
    plt.colorbar(im_c, ax=ax_c, shrink=0.7, pad=0.02, label="|LLR|")

    # ── Legend row (spans both columns) ──
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    ax_leg.text(0.15, 0.75, "* WHO-catalogued resistance mutation", fontsize=11,
                transform=ax_leg.transAxes, va="center")
    ax_leg.text(0.15, 0.45, "Green = low fitness cost (evolutionarily accessible)",
                fontsize=11, transform=ax_leg.transAxes, va="center", color="#2ca02c")
    ax_leg.text(0.15, 0.15, "Red = high fitness cost (evolutionarily constrained)",
                fontsize=11, transform=ax_leg.transAxes, va="center", color="#d62728")

    # Key stats box
    stats_text = (
        "katG S315T:  |LLR| = 0.68  (rank 1,384 / 14,060 = top 9.8%)\n"
        "Neighbor G316:  min |LLR| = 6.47  (9.5x higher cost)\n"
        "gyrA A90V + D94G:  |LLR| = 2.52  (tied — co-dominant clinically)"
    )
    ax_leg.text(0.55, 0.45, stats_text, fontsize=10, fontfamily="monospace",
                transform=ax_leg.transAxes, va="center",
                bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    plt.savefig(FIGURES / "fig3_landscapes.png")
    plt.savefig(FIGURES / "fig3_landscapes.pdf")
    plt.close()
    print("  Fig 3 saved.")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Three-class model — concordance + mechanism
# ═══════════════════════════════════════════════════════════════════════
def fig4_three_class():
    with open(RESULTS / "prospective" / "prospective_analysis.json") as f:
        prosp = json.load(f)

    gene_data = prosp["within_gene"]

    # Classify genes by mechanism
    classes = {
        "Conservative\nsubstitution": {
            "genes": ["mtb_katG", "mtb_gyrA", "ngonorrhoeae_penA"],
            "color": "#00A087",
        },
        "Loss of\nfunction": {
            "genes": ["mtb_pncA"],
            "color": "#E64B35",
        },
        "Structural\npocket": {
            "genes": ["mtb_rpoB", "mtb_embB"],
            "color": "#3C5488",
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # ── Panel A: Rank concordance by gene ──
    ax = axes[0]
    all_genes = []
    all_conc = []
    all_colors = []
    for cls_name, cls_info in classes.items():
        for gene in cls_info["genes"]:
            if gene in gene_data:
                all_genes.append(gene.replace("_", "\n"))
                all_conc.append(gene_data[gene]["rank_concordance"])
                all_colors.append(cls_info["color"])

    bars = ax.bar(range(len(all_genes)), all_conc, color=all_colors,
                  edgecolor="black", linewidth=0.5, width=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Random")
    ax.set_xticks(range(len(all_genes)))
    ax.set_xticklabels(all_genes, fontsize=8)
    ax.set_ylabel("Rank concordance")
    ax.set_ylim(0, 1.05)
    ax.set_title("A)  Emergence order prediction", fontweight="bold", loc="left")
    ax.legend(fontsize=9)

    # Add N labels
    for i, gene_key in enumerate(
        [g for cls in classes.values() for g in cls["genes"] if g in gene_data]
    ):
        n = gene_data[gene_key]["n"]
        ax.text(i, all_conc[i] + 0.03, f"n={n}", ha="center", fontsize=8)

    # ── Panel B: Top-k precision ──
    ax2 = axes[1]
    genes_for_topk = ["mtb_katG", "mtb_gyrA", "ngonorrhoeae_penA", "mtb_rpoB", "mtb_pncA"]
    topk_vals = []
    topk_labels = []
    topk_colors = []
    for gene in genes_for_topk:
        if gene not in gene_data:
            continue
        gd = gene_data[gene]
        # Find top_k precision key
        for key in gd:
            if key.startswith("top_") and key.endswith("_precision"):
                topk_vals.append(gd[key])
                k_val = key.split("_")[1]
                topk_labels.append(f"{gene.split('_')[1]}\n(k={k_val})")
                for cls_name, cls_info in classes.items():
                    if gene in cls_info["genes"]:
                        topk_colors.append(cls_info["color"])
                        break
                break

    ax2.bar(range(len(topk_vals)), topk_vals, color=topk_colors,
            edgecolor="black", linewidth=0.5, width=0.7)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xticks(range(len(topk_vals)))
    ax2.set_xticklabels(topk_labels, fontsize=8)
    ax2.set_ylabel("Top-k precision")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("B)  Top-k precision", fontweight="bold", loc="left")

    # ── Panel C: Three-class schematic ──
    ax3 = axes[2]
    ax3.axis("off")

    class_info = [
        ("Class 1: Conservative substitution", "#00A087",
         "katG S315T, gyrA QRDR\nESM-2 LLR works\nPanel gap: 0%"),
        ("Class 2: Loss of function", "#E64B35",
         "pncA, Rv0678\nESM-2 LLR fails\n(entire protein permissive)"),
        ("Class 3: Structural pocket", "#3C5488",
         "rpoB RRDR\nESM-2 LLR fails\n(mutations similarly costly)"),
    ]

    for i, (title, color, desc) in enumerate(class_info):
        y = 0.82 - i * 0.32
        ax3.add_patch(plt.Rectangle((0.02, y - 0.08), 0.96, 0.25,
                                     facecolor=color, alpha=0.15,
                                     edgecolor=color, linewidth=2,
                                     transform=ax3.transAxes))
        ax3.text(0.06, y + 0.10, title, fontsize=11, fontweight="bold",
                 color=color, transform=ax3.transAxes, va="top")
        ax3.text(0.06, y - 0.02, desc, fontsize=9,
                 transform=ax3.transAxes, va="top", linespacing=1.4)

    ax3.set_title("C)  Three-class AMR model", fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(FIGURES / "fig4_three_class.png")
    plt.savefig(FIGURES / "fig4_three_class.pdf")
    plt.close()
    print("  Fig 4 saved.")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Panel design — coverage curves + Rv0678 case
# ═══════════════════════════════════════════════════════════════════════
def fig5_panel_design():
    with open(RESULTS / "panel_design" / "panel_comparison.json") as f:
        panels = json.load(f)

    # Select genes with enough mutations for meaningful curves
    show_genes = ["mtb_katG", "mtb_gyrA", "mtb_rpoB", "mtb_pncA",
                  "ngonorrhoeae_penA", "mtb_embB"]
    show_genes = [g for g in show_genes if g in panels]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    mechanism_colors = {
        "mtb_katG": "#00A087", "mtb_gyrA": "#00A087",
        "ngonorrhoeae_penA": "#00A087",
        "mtb_rpoB": "#3C5488", "mtb_embB": "#3C5488",
        "mtb_pncA": "#E64B35",
    }

    for idx, gene_key in enumerate(show_genes):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        data = panels[gene_key]["panel_comparisons"]
        n_muts = panels[gene_key]["n_known_mutations"]

        ks = [d["k"] for d in data]
        llr_cov = [d["llr_coverage"] for d in data]
        prev_cov = [d["prevalence_coverage"] for d in data]
        rand_cov = [d["random_coverage_mean"] for d in data]
        rand_std = [d["random_coverage_std"] for d in data]

        # Prevalence = gold standard
        ax.plot(ks, prev_cov, "o-", color="#2ca02c", linewidth=2,
                markersize=5, label="Prevalence (gold std)", zorder=3)
        # LLR
        ax.plot(ks, llr_cov, "s-", color=mechanism_colors.get(gene_key, "blue"),
                linewidth=2, markersize=5, label="ESM-2 |LLR|", zorder=3)
        # Random
        ax.fill_between(ks,
                         np.array(rand_cov) - np.array(rand_std),
                         np.array(rand_cov) + np.array(rand_std),
                         alpha=0.15, color="gray")
        ax.plot(ks, rand_cov, "--", color="gray", linewidth=1, label="Random")

        ax.axhline(90, color="red", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.set_title(gene_key.replace("_", " ") + f"  (N={n_muts})", fontsize=11)
        ax.set_xlabel("Panel size (k)")
        ax.set_ylim(-2, 105)

        if col == 0:
            ax.set_ylabel("Coverage (%)")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

        # Shade gap at k=3
        k3 = min(3, n_muts) - 1
        gap = data[k3]["llr_vs_prevalence_gap"]
        if abs(gap) > 1:
            ax.annotate(
                f"gap={gap:.0f}%", xy=(k3 + 1, (llr_cov[k3] + prev_cov[k3]) / 2),
                fontsize=8, color="red", ha="center",
            )

    # Hide unused subplot if needed
    for idx in range(len(show_genes), 6):
        row, col = divmod(idx, 3)
        axes[row, col].set_visible(False)

    fig.suptitle("Diagnostic Panel Coverage: ESM-2 LLR-ranked vs Surveillance-ranked",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig5_panel_design.png")
    plt.savefig(FIGURES / "fig5_panel_design.pdf")
    plt.close()
    print("  Fig 5 saved.")


# ═══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: Rv0678 bedaquiline case
# ═══════════════════════════════════════════════════════════════════════
def fig_supp_rv0678():
    df = pd.read_csv(RESULTS / "retrospective" / "llr_results.csv")
    rv = df[df["gene"] == "Rv0678"].copy()
    rv = rv[rv["esm2_llr"].notna()]
    rv["abs_llr"] = rv["esm2_llr"].abs()
    rv = rv.sort_values("abs_llr")

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["#00A087", "#4DBBD5", "#E64B35"]
    for i, (_, row) in enumerate(rv.iterrows()):
        ax.barh(i, row["abs_llr"], color=colors[i], edgecolor="black",
                linewidth=0.5, height=0.5)
        ax.text(row["abs_llr"] + 0.2, i,
                f'prevalence = {row["prevalence_pct"]:.1f}%',
                va="center", fontsize=10)

    ax.set_yticks(range(len(rv)))
    ax.set_yticklabels(rv["mutation"].values, fontsize=12, fontweight="bold")
    ax.set_xlabel("|ESM-2 LLR|  (fitness cost)")
    ax.set_title("Rv0678 (bedaquiline) — predicted emergence order\n"
                 "Low |LLR| = emerges first (drug approved 2012, mutations catalogued 2022)",
                 fontsize=11)
    ax.invert_yaxis()

    # Arrow showing predicted order
    ax.text(0.5, -0.6, "Predicted earliest", fontsize=9,
            color="#00A087", fontweight="bold", ha="center")
    ax.text(8, -0.6, "Predicted latest", fontsize=9,
            color="#E64B35", fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig(FIGURES / "fig_supp_rv0678.png")
    plt.savefig(FIGURES / "fig_supp_rv0678.pdf")
    plt.close()
    print("  Fig Supp Rv0678 saved.")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures...")
    fig2_retrospective()
    fig3_landscapes()
    fig4_three_class()
    fig5_panel_design()
    fig_supp_rv0678()
    print(f"\nAll figures saved to {FIGURES}/")
    for f in sorted(FIGURES.glob("*")):
        print(f"  {f.name}")
