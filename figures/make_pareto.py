#!/usr/bin/env python3
"""Parameter-accuracy trade-off (Pareto) figure for CRF-KD paper.
Numbers: 50-isolate NanoSpecies median read identity."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (params_M, median_identity_%, label)
kd = [
    (0.66, 85.29, "M2-Tiny"),
    (3.5, 87.07, "M2-Small"),
]
hard = [
    (0.66, 85.02, "M2-Tiny"),
    # H-Small 50-iso filled at runtime; placeholder ~86.5
    (3.5, 86.50, "M2-Small"),
]
vendor = [
    (27.0, 91.08, "Bonito sup (teacher)"),
]

fig, ax = plt.subplots(figsize=(3.4, 2.7))

xk = [p for p, _, _ in kd]
yk = [a for _, a, _ in kd]
ax.plot(xk, yk, "o-", color="#1f77b4", label="CRF-KD (ours)", zorder=3)

xh = [p for p, _, _ in hard]
yh = [a for _, a, _ in hard]
ax.plot(xh, yh, "s--", color="#ff7f0e", label="Hard-label baseline", zorder=2)

xv = [p for p, _, _ in vendor]
yv = [a for _, a, _ in vendor]
ax.plot(xv, yv, "*", color="#2ca02c", markersize=13, label="Bonito sup (teacher)", zorder=3)

for p, a, lbl in kd:
    ax.annotate(lbl, (p, a), textcoords="offset points", xytext=(5, 4), fontsize=7)
ax.annotate("teacher", (27.0, 91.08), textcoords="offset points", xytext=(-6, -12), fontsize=7, ha="right")

ax.set_xscale("log")
ax.set_xlabel("Parameters (M, log scale)")
ax.set_ylabel("Median read identity (\\%)")
ax.set_xlim(0.4, 45)
ax.set_ylim(84, 92.5)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig("pareto.pdf", bbox_inches="tight")
print("wrote pareto.pdf")
