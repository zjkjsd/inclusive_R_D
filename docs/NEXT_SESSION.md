# Prompt for the next session

Last updated: 2026-11-18. Copy the block below into a new Claude Code session in this repository.
Replace this file when the work moves on; the discussion records (`docs/B0_PIECE2_REVIEW.md`,
`docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`) keep the history.

---

> I'm continuing the inclusive tagged R(D+) analysis (Belle II), Part B (BBbar background model),
> step B0: matching the EvtGen decay file with PDG. Pieces 1 and 2 are DONE. We now start **piece 3:
> exclusive modes without a PDG measurement**, and how they enter the background model.
>
> Read first:
> - `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`, sections B.6 (the reweighting model and the
>   first-model parameters r_meas, r_SL, N₁, N₂, a) and B.13 (B0 in five pieces: the piece-2 summary
>   and the piece-3 facts and proposal, written 2026-11-18).
> - `docs/B0_PIECE2_REVIEW.md`, section 1 (all piece-2 decisions with current values) and section 3
>   (register of the exceptions file). The rest of that file is the dated record of the piece-2
>   discussion; read a section when a point refers to it.
>
> Code and data: `scripts/tools/decay_table.py` (decay-file parser, classification),
> `scripts/tools/pdg_match.py` (PDG comparison; run `python3 scripts/tools/pdg_match.py`),
> `scripts/tools/pdg_review_cases.py` (writes `docs/B0_PIECE2_CASES.md`),
> `Samples/dec/pdg_exceptions.json` (28 hand corrections, each with its reason),
> `Samples/dec/pdg_bf_B_2026.json` (cached PDG values), outputs in `Notebooks/Plotting/csv_output/`
> (`b_mode_pdg_table.csv`, `b_mode_pdg_review.csv`, `b_mode_pdg_nodes.csv`,
> `b_mode_isospin_check.csv`, and the sample tables `bbbar_modes_*.csv`). Tests:
> `python3 scripts/tests/test_pdg_match.py` and `python3 scripts/tests/test_decay_table.py`.
>
> Open points carried over: N1 (uncertainty of lines with no PDG entry at all; proposed c = 1 ± 1),
> part of piece 3; the semileptonic gap modes (non-resonant D(\*)πℓν kept at c = 1 ± 1, D(\*)ππℓν at
> their remainders), to be replaced by the official Belle II gap-mode recommendation that I will
> provide; totals that stay above PDG by decision (B⁰ → J/ψK⁰π⁺π⁻, B⁰ → D⁰D̄⁰K⁺π⁻, the D(\*)πℓν
> totals).
>
> How I want to work:
> - One point at a time. For each point, show me the facts (with decay-file line numbers and PDG
>   ids, and the source papers when a number matters) and the options with your recommendation, and
>   wait for my decision.
> - After each decision, record it (status, decision, reason) in a review file for piece 3, in the
>   style of `docs/B0_PIECE2_REVIEW.md`: a decision table with current values at the top, a dated
>   record below. Implement only when I say so, rerun the scripts and the tests, and tell me what
>   changed (before/after for every line whose value moved).
> - Raise anything abnormal instead of fixing it silently. Check numbers against their source
>   (decay file, PDG via pdgapi.lbl.gov/pdgLive, the papers on arXiv/INSPIRE) rather than from memory,
>   and say so when you could not check something.
> - Dates: write the actual date of the day the work is done; check it.
> - If a web site you need is blocked, tell me the host and I will allow it.
>
> Start with point 1 of the piece-3 proposal in B.13: whether the exclusive double-charm lines
> without a measurement and the PYTHIA ≥ 2 charm decays form one class (one scale factor), and what
> inclusive measurements can constrain it. Begin by checking the facts of the proposal (the shares,
> the inclusive B → D D̄ K X, B → D_s X and B → D D̄ X rates against the decay file's sums).
