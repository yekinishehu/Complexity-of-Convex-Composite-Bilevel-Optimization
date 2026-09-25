# Supplementary code: The Complexity of Convex Composite Bilevel Optimization — A Trichotomy

Verification suite and experiment code for the manuscript

> Y. Shehu, *The Complexity of Convex Composite Bilevel Optimization: A Trichotomy*.
> Submitted to Mathematics of Operations Research.
> Preprint/manuscript: `bilevel_trichotomy_MOR.tex` (not included in this repository;
> the paper is the primary artifact — this repository contains only the code).

## Contents

| File | Purpose |
|---|---|
| `supplement_verification.py` | All verification checks **and** all numerical experiments (E1–E7). The single source of truth for every number tabulated in the paper's Tables 5–7. |
| `apply_fixes.py` | Optional. Reproduces the revised manuscript from the original draft by applying the documented fixes (old→new blocks, each asserted). Not needed to verify the results. |
| `requirements.txt` | Python dependencies (NumPy, Matplotlib only). |

## Quickstart

```bash
pip install -r requirements.txt
python supplement_verification.py
```

Expected output:

- `ALL 166 CHECKS PASSED` — the check count is asserted to equal **166**, the number
  stated twice in the manuscript (end of §1.3 and Appendix C). If the assertion fires,
  the manuscript text must be synced to the printed count.
- `figures/exp1_uniform_barrier.pdf` … `figures/exp7_deep_pair.pdf` and
  `figures/fig_frontier.pdf` — every figure referenced by the manuscript
  (`exp7_deep_pair.pdf` is generated but not referenced; it backs Table 7).
- A printed block with the regenerated constants (`rho`, `tau`, `rho*R*sqrt(L/tau)`,
  `rho^2/tau`, `L_om*L*R^2/tau`), all five rows of Table 5, the fitted slopes,
  the E4 floor constants, the E5 medians and relative errors, the E6 endpoint
  errors, and E7's `theta_9`/loss columns. **These printed values are the values
  tabulated in the paper** (Tables 5–7 and the related sentences in §5.2); the code
  is the source of truth, not the other way around.

## What the 166 checks verify

- **Group A (141) — closed-form identities.** Every load-bearing algebraic step of
  the proofs: the `dd_i(chi)` closed form (3.13) and the polynomial bounds (a)–(b)
  of Thm 3.20(iii); the ramp separation sum and index count (Thm 3.22, including the
  exact threshold `n >= 62`); the restricted-gap inequality `gap(k+2) >= L/(48k)`
  on the full window `k <= n/8` (Thm 3.5(i) repair); the discriminant, phase and
  telescoping bounds of the FBi-PG trajectory analysis (Lemmas 3.6–3.7, Thm 3.4);
  the FISTA scalars `t_s` bounds; the per-block product-chain gap at block length
  `3k` (Lemma 3.20: `L(k-1)/(4(3k+1)(k+3)) >= L/(48(k+2))`); the joint moving-target
  bound (Thm 3.31); the E4/E5/E7 constants; and the value-relation counterexamples
  of Thm 3.16.
- **Group B (12) — transcript identity.** Under adversarial adaptive zero-respecting
  probing, the complete transcripts of the singleton pair and of the tower pair
  (two towers differing only in the scale-`i` bump height) are *exactly* identical
  through round `2^i`, with `theta_i = 0` exactly, at scales `i in {2,4,6}`
  (Lemma 3.1 and its tower analogue). All zeros are exact: the chain gradient is
  computed as `(L/4)(T y - e_1)` so the zero-chain property holds bitwise.
- **Group C (3) — rotation support invariant.** Exact simulation of the pulled-back
  recursion of Lemma 3.19: the pulled-back query at stage `j` has support
  `subseteq {1..j}` (growing by exactly one coordinate per stage) and the report,
  processed as one further query, extends the support by at most two coordinates
  (i.e. `k+2` at round `k`).

These checks complement, and do not replace, the written proofs (Appendix C of the
paper states this explicitly).

## Reproducibility

- Single dependency stack: NumPy + Matplotlib. No network access, no GPU.
- All randomness is seeded (`numpy.random.default_rng` with fixed seeds), so every
  check, figure, and table value is deterministic across runs and platforms.
- Pure-Python exact recurrences are used wherever closed forms exist (chain
  restricted minima, `dd_i` identities), so the checks are validation of the
  algebra, not merely of one numerical pipeline.

## Layout expectations

The manuscript expects `figures/` relative to the `.tex`
(`\graphicspath{{figures/}}`). After running the script once, either keep this
repository's `figures/` next to the `.tex`, or copy the generated PDFs into your
paper's `figures/` directory.

## Citing

If you use this code, please cite the paper (MOR submission, 2026). A citable
archive (Zenodo DOI) will be linked here once minted.
