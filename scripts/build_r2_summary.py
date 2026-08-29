"""Aggregate R1 & R2 KD eval.csv files into Markdown / LaTeX / TSV tables.

Outputs land in logs/r2_summary/:
  - overall.md / overall.tex / overall.tsv     # 6-config headline table
  - per_isolate.md / per_isolate.tex / .tsv    # 8 isolates x 6 configs
  - deltas.md                                  # KD vs Hard, Small vs Tiny, R2 vs R1
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "r2_summary"
OUT.mkdir(parents=True, exist_ok=True)

# (label, short, ckpt-dir, epochs, model, training)
CONFIGS = [
    ("R1 · H · Tiny",        "R1-H",      "checkpoints_kd/r1_h",            3, "Tiny",  "hard"),
    ("R1 · KD · Tiny",       "R1-KD-T",   "checkpoints_kd/r1_kdjp",         3, "Tiny",  "KD"),
    ("R1 · KD · Small",      "R1-KD-S",   "checkpoints_kd/r1_m2small_kdjp", 3, "Small", "KD"),
    ("R2 · H · Tiny",        "R2-H",      "checkpoints_kd/r2_h",            5, "Tiny",  "hard"),
    ("R2 · KD · Tiny",       "R2-KD-T",   "checkpoints_kd/r2_kdjp_tiny",    5, "Tiny",  "KD"),
    ("R2 · KD · Small",      "R2-KD-S",   "checkpoints_kd/r2_kdjp_small",   5, "Small", "KD"),
]


ISOLATES_ORDER = [
    "Klebsiella_pneumoniae_INF078",
    "Klebsiella_pneumoniae_INF007",
    "Klebsiella_pneumoniae_KSB1_6F",
    "Klebsiella_pneumoniae_KSB1_7E",
    "Escherichia_coli_MSB2_1A",
    "Acinetobacter_baumannii_AYP-A2",
    "Haemophilus_parainfluenzae_M1C146_1",
    "Moraxella_lincolnii_51409",
]
PRETTY = {
    "Klebsiella_pneumoniae_INF078":         "K.p. INF078",
    "Klebsiella_pneumoniae_INF007":         "K.p. INF007",
    "Klebsiella_pneumoniae_KSB1_6F":        "K.p. KSB1-6F",
    "Klebsiella_pneumoniae_KSB1_7E":        "K.p. KSB1-7E",
    "Escherichia_coli_MSB2_1A":             "E. coli MSB2-1A",
    "Acinetobacter_baumannii_AYP-A2":       "A. baum. AYP-A2",
    "Haemophilus_parainfluenzae_M1C146_1":  "H. para. M1C146",
    "Moraxella_lincolnii_51409":            "M. linc. 51409",
}


def load_eval(path: Path):
    rows = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            rows[r["isolate"]] = (
                int(r["n_reads"]),
                float(r["median"]),
                float(r["mean"]),
            )
    return rows


def pct(x: float) -> str:
    return f"{x * 100:.2f}"


def build_overall(data):
    lines_md = [
        "| # | Run | Model | Loss | Epochs | Median (%) | Mean (%) | n reads |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines_tsv = ["#\trun\tmodel\tloss\tepochs\tmedian\tmean\tn_reads"]
    lines_tex = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"\# & Run & Model & Loss & Epochs & Median (\%) & Mean (\%) & $n$ reads \\",
        r"\midrule",
    ]

    best_med = max(data[c[1]]["OVERALL"][1] for c in CONFIGS)
    best_mean = max(data[c[1]]["OVERALL"][2] for c in CONFIGS)

    for i, (label, short, _, ep, model, loss) in enumerate(CONFIGS, 1):
        n, med, mean = data[short]["OVERALL"]
        med_md = f"**{pct(med)}**" if med == best_med else pct(med)
        mean_md = f"**{pct(mean)}**" if mean == best_mean else pct(mean)
        med_tex = rf"\textbf{{{pct(med)}}}" if med == best_med else pct(med)
        mean_tex = rf"\textbf{{{pct(mean)}}}" if mean == best_mean else pct(mean)
        lines_md.append(
            f"| {i} | {label} | {model} | {loss} | {ep} | {med_md} | {mean_md} | {n} |"
        )
        lines_tsv.append(
            f"{i}\t{label}\t{model}\t{loss}\t{ep}\t{pct(med)}\t{pct(mean)}\t{n}"
        )
        lines_tex.append(
            f"{i} & {label} & {model} & {loss} & {ep} & {med_tex} & {mean_tex} & {n} \\\\"
        )
    lines_tex += [r"\bottomrule", r"\end{tabular}"]

    (OUT / "overall.md").write_text("\n".join(lines_md) + "\n")
    (OUT / "overall.tsv").write_text("\n".join(lines_tsv) + "\n")
    (OUT / "overall.tex").write_text("\n".join(lines_tex) + "\n")


def build_per_isolate(data):
    headers = [PRETTY[i] for i in ISOLATES_ORDER] + ["**OVERALL**"]
    cols = ISOLATES_ORDER + ["OVERALL"]

    md = ["### Median read identity (%) per isolate", ""]
    md.append("| Run | " + " | ".join(headers) + " |")
    md.append("|---" * (len(headers) + 1) + "|")
    tsv = ["run\t" + "\t".join(cols)]
    tex = [
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        " & ".join(["Run"] + [PRETTY.get(c, c) for c in cols]) + r" \\",
        r"\midrule",
    ]

    # Per-column best for highlight
    best_per_col = {c: max(data[s[1]][c][1] for s in CONFIGS) for c in cols}

    for label, short, _, _, _, _ in CONFIGS:
        med_cells_md, med_cells_tsv, med_cells_tex = [], [], []
        for c in cols:
            v = data[short][c][1]
            cell = pct(v)
            if v == best_per_col[c]:
                med_cells_md.append(f"**{cell}**")
                med_cells_tex.append(rf"\textbf{{{cell}}}")
            else:
                med_cells_md.append(cell)
                med_cells_tex.append(cell)
            med_cells_tsv.append(cell)
        md.append(f"| {label} | " + " | ".join(med_cells_md) + " |")
        tsv.append(f"{label}\t" + "\t".join(med_cells_tsv))
        tex.append(f"{label} & " + " & ".join(med_cells_tex) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]

    md.append("")
    md.append("### Mean read identity (%) per isolate")
    md.append("")
    md.append("| Run | " + " | ".join(headers) + " |")
    md.append("|---" * (len(headers) + 1) + "|")
    best_per_col_mean = {c: max(data[s[1]][c][2] for s in CONFIGS) for c in cols}
    for label, short, _, _, _, _ in CONFIGS:
        cells_md = []
        for c in cols:
            v = data[short][c][2]
            cell = pct(v)
            cells_md.append(f"**{cell}**" if v == best_per_col_mean[c] else cell)
        md.append(f"| {label} | " + " | ".join(cells_md) + " |")

    (OUT / "per_isolate.md").write_text("\n".join(md) + "\n")
    (OUT / "per_isolate.tsv").write_text("\n".join(tsv) + "\n")
    (OUT / "per_isolate.tex").write_text("\n".join(tex) + "\n")


def build_deltas(data):
    def med(short, iso="OVERALL"):
        return data[short][iso][1]

    rows = []

    rows.append(("R1: KD-Tiny  vs  H-Tiny    (KD gain @3ep)",   med("R1-KD-T") - med("R1-H")))
    rows.append(("R1: KD-Small vs  KD-Tiny  (Tiny→Small @3ep)", med("R1-KD-S") - med("R1-KD-T")))
    rows.append(("R2: KD-Tiny  vs  H-Tiny    (KD gain @5ep)",   med("R2-KD-T") - med("R2-H")))
    rows.append(("R2: KD-Small vs  KD-Tiny  (Tiny→Small @5ep)", med("R2-KD-S") - med("R2-KD-T")))
    rows.append(("R2 vs R1 — H-Tiny    (longer training)",       med("R2-H")    - med("R1-H")))
    rows.append(("R2 vs R1 — KD-Tiny   (longer training)",       med("R2-KD-T") - med("R1-KD-T")))
    rows.append(("R2 vs R1 — KD-Small  (longer training)",       med("R2-KD-S") - med("R1-KD-S")))

    md = ["### Headline deltas (overall median, percentage points)", ""]
    md.append("| Comparison | Δ median (pp) |")
    md.append("|---|---:|")
    for name, d in rows:
        md.append(f"| {name} | {'+' if d >= 0 else ''}{d * 100:.2f} |")

    (OUT / "deltas.md").write_text("\n".join(md) + "\n")


def main():
    data = {}
    for label, short, ckpt, *_ in CONFIGS:
        path = ROOT / ckpt / "eval.csv"
        if not path.exists():
            raise SystemExit(f"missing eval.csv: {path}")
        data[short] = load_eval(path)

    build_overall(data)
    build_per_isolate(data)
    build_deltas(data)
    print(f"wrote {len(list(OUT.iterdir()))} files into {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
