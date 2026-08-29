### Phase 2 main result on NanoSpecies-50

| Model | n_iso | Median (%) | Mean (%) | Δ to teacher (pp) |
|---|---:|---:|---:|---:|
| Bonito sup (teacher) | 50 | 91.03 | 90.42 | — |
| M2_TINY  (hard-only) | 50 | 84.95 | 84.43 | -6.08 |
| M2_TINY  (CRF-KD ours) | 50 | 85.25 | 84.74 | -5.78 |
| M2_SMALL (hard-only) | 50 | 87.70 | 87.15 | -3.33 |
| M2_SMALL (CRF-KD ours) | 50 | 87.05 | 86.53 | -3.98 |

**Headline:** mean-of-medians of M2_TINY (CRF-KD) = 85.25 %, M2_SMALL (CRF-KD) = 87.05 %, against Bonito sup 91.03 %.
