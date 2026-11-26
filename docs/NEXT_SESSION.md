# Prompt for the next session

Last updated: 2026-11-20. Copy the block below into a new Claude Code session in this repository.
Replace this file when the work moves on; the discussion records (`docs/B0_PIECE2_RECORD.md`,
`docs/B0_PIECE3_REVIEW.md`, `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`) keep the history.

---

> I'm continuing the inclusive tagged R(D+) analysis (Belle II), Part B (BBbar background model),
> step B0. Pieces 1–3 are DONE: the decay file is matched with PDG 2026 (piece 2), and the exclusive
> lines without a measurement and the PYTHIA decays are assigned to model classes with priors
> (piece 3). We now start **piece 4: priors and how the corrections enter the model**.
>
> Read first:
> - `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`, sections B.6 (the reweighting model; its parameter
>   table now holds the piece-3 classes and priors) and B.13 (B0 in five pieces; the piece-4 row).
> - `docs/B0_PIECE3_REVIEW.md`: section 1 (decision table with current values) and section 4 (piece-3
>   summary and the list of what passes to piece 4). Read a dated section (3.1–3.8) when a point
>   refers to it.
> - `docs/B0_PIECE2_REVIEW.md`: section 1 (decisions), 2 (rules as implemented), 3 (register of the
>   exceptions file), 4 (open items). The dated piece-2 discussion is `docs/B0_PIECE2_RECORD.md`.
> - `docs/REFERENCES.md` (the papers and notes behind the decisions) and
>   `docs/DECAY_FILE_ISSUES_FOR_BELLE2.md` (the report of decay-file problems, to be sent to Belle II).
>
> Code and data: `scripts/tools/decay_table.py` (decay-file parser, classification),
> `scripts/tools/pdg_match.py` (PDG comparison; `python3 scripts/tools/pdg_match.py`),
> `scripts/tools/pdg_review_cases.py`, `Samples/dec/pdg_exceptions.json` (100 hand corrections, each with
> its reason), `Samples/dec/pdg_bf_B_2026.json` (cached PDG values), outputs in
> `Notebooks/Plotting/csv_output/` (`b_mode_pdg_table.csv` and the other `b_mode_*.csv`, the sample
> tables `bbbar_modes_*.csv`, `truth_b_decays_summary.csv`). Generator truth of MC16rd B decays:
> `Samples/mc_truth/truth_{mixed,charged}.root` (one file each, experiment 18), written by
> `scripts/mc_truth/b_decay_truth_basf2.py`, analysed by `scripts/mc_truth/truth_analysis.py`
> (needs `pip install uproot scipy` in a fresh container). Tests:
> `python3 scripts/tests/test_pdg_match.py` and `python3 scripts/tests/test_decay_table.py`.
>
> Piece 4 topics (from `docs/B0_PIECE3_REVIEW.md` section 4 and B.13):
> - c applied as event weights, or only as prior widths, or both (the B → X_sγ note uses nominal
>   weights plus one ±1σ template per corrected mode; B.13);
> - correlations between modes (e.g. the BaBar 2011 D D̄ K modes share systematics; the D_s lines of
>   3.4 share the LHCb normalization);
> - the isospin parameter for PYTHIA single-charm decays (3.5: PYTHIA makes no D\*\*; definition,
>   rate conservation, wide prior from BaBar 2007 per flavor);
> - a composition nuisance for the D(\*)D̄(\*)K\* lines (3.4) and B⁰ → D⁰D̄⁰K⁺π⁻ (S042.609) as a
>   closure check including PYTHIA;
> - the D₁(2420) decay table (3.7: D₁ → Dππ about 20 % too high relative to D\*π; event weight on
>   `D_genMotherPDG` for D⁺ directly from D₁, mainly in B → D₁ℓν);
> - r_SL: one parameter or a split (D, D\*, narrow D\*\*, broad D\*\* + gap); τ modes (their MC BF comes
>   from a theory R(D\*), so a PDG correction would be circular);
> - the semileptonic gap modes (non-resonant D(\*)πℓν at c = 1 ± 1, D(\*)ππℓν at their remainders,
>   D(\*)ηℓν), to be replaced by the official Belle II gap-mode recommendation that I will provide.
>
> How I want to work:
> - Before the options of each point, explain the context briefly: what problem the point solves and
>   where it sits in the model (also as a short context paragraph in the review file).
> - One point at a time. For each point, show me the facts (with decay-file line numbers and PDG
>   ids, and the source papers when a number matters) and the options with your recommendation, and
>   wait for my decision.
> - After each decision, record it (status, decision, reason) in a review file for piece 4, in the
>   style of `docs/B0_PIECE3_REVIEW.md`: a decision table with current values at the top, a dated
>   record below. Implement only when I say so, rerun the scripts and the tests, and tell me what
>   changed (before/after for every value that moved).
> - Raise anything abnormal instead of fixing it silently; add decay-file problems to
>   `docs/DECAY_FILE_ISSUES_FOR_BELLE2.md` and critical references to `docs/REFERENCES.md`.
> - Check numbers against their source (decay file, PDG via pdgapi.lbl.gov, the papers on
>   arXiv/INSPIRE) rather than from memory, and say so when you could not check something. The
>   basf2 source is on GitHub (belle2/basf2, tag light-2609-luna; belle2/externals).
> - Dates: write the actual date of the day the work is done; check it.
> - If a web site you need is blocked, tell me the host and I will allow it. New directories with
>   files to commit may need a whitelist line in `.gitignore` (it ignores everything by default).
>
> Start with an overview of piece 4: list its points in the order you propose, with the facts that
> decide the order, and wait for my agreement before the first point.
