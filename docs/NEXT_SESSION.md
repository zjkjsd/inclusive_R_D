# Prompt for the next session

Last updated: 2026-11-27. Copy the block below into a new Claude Code session in this repository.
Replace this file when the work moves on; the discussion records (`docs/B0_PIECE2_RECORD.md`,
`docs/B0_PIECE3_REVIEW.md`, `docs/B0_PIECE4_REVIEW.md`, `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`) keep
the history.

---

> I'm continuing the inclusive tagged R(D+) analysis (Belle II; D⁺ → K⁻π⁺π⁺, signal B⁰ → D⁻τν, not
> FEI-tagged, MC16rd, decay file `Samples/dec/DECAY_BELLE2_rel08.DEC`), Part B (BBbar background model),
> step B0. Pieces 1–3 are DONE. **Piece 4 (priors and how the corrections enter the model) is decided**:
> all points 4.1–4.8 are AGREED, and the rate corrections of 4.5–4.7 are implemented in
> `scripts/tools/pdg_match.py` and `Samples/dec/pdg_exceptions.json` (166 hand corrections). What remains
> of piece 4 is code: the event weights (ε of 4.3, θ of 4.4, the D₁ table of 4.5), the semileptonic
> parameters r_SL and r_broad+gap, the 4.7 composition nuisances and gap-shape proxy, and the toys of
> 4.1/4.2 (c drawn from the source model, propagated through the linearized tuning into eigen-NPs for
> the SR prior). Piece 5 (code layout: separate modules, merged into `utilities.py` at the end) is AGREED.
>
> Read first:
> - `docs/B0_PIECE4_REVIEW.md`: section 1 (decision table with current values) and section 4 (piece-4
>   summary and what passes on). Read a dated section (3.1–3.8) when a point refers to it; each opens
>   with a context paragraph.
> - `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`: the procedure table (B0–B9, C1–C2), B.6 (the reweighting
>   model and its parameter table) and B.13 (B0 in five pieces).
> - `docs/B0_PIECE3_REVIEW.md` section 4 and `docs/B0_PIECE2_REVIEW.md` sections 1–4 for the earlier
>   pieces; `docs/REFERENCES.md`; `docs/DECAY_FILE_ISSUES_FOR_BELLE2.md`; `docs/NEXT_REPROCESSING.md`
>   (variables for the next ntuple production, including the MCAmbulance and HAMMER inputs of 4.8);
>   `docs/SLME_NONRESONANT_DPILNU_NOTE.md` (the non-resonant D(\*)πℓν question I raised with SLME).
>
> Code and data: `scripts/tools/decay_table.py` (decay-file parser, classification),
> `scripts/tools/pdg_match.py` (PDG comparison and the per-line c; `python3 scripts/tools/pdg_match.py`;
> exception actions include `external`, `sub_decay`, `override`, `tau_ratio`), `Samples/dec/pdg_exceptions.json`,
> `Samples/dec/pdg_bf_B_2026.json` (cached PDG values), outputs in `Notebooks/Plotting/csv_output/`
> (`b_mode_pdg_table.csv`, `b_mode_pdg_review.csv`, `b_mode_pdg_nodes.csv`, `b_mode_isospin_check.csv`, the
> sample tables `bbbar_modes_*.csv`, `truth_b_decays_summary.csv`). The BBbar weights of the current tuning
> are in `utilities.py` (`prepare_bbbar_reweighting`, `apply_bbbar_weights`) and
> `5_BBbkg_weights_optuna_minuit.py`. Generator truth of MC16rd B decays: `Samples/mc_truth/truth_{mixed,charged}.root`,
> analysed by `scripts/mc_truth/truth_analysis.py` (needs `pip install uproot scipy`). The candidate
> ntuples are not in the cloud container; code that needs them is tested on the mode tables or run by me.
> Tests: `python3 scripts/tests/test_pdg_match.py` (38) and `python3 scripts/tests/test_decay_table.py` (15).
>
> Points carried forward (piece-4 summary): B3 Fisher run for r_SL, r_broad+gap, ε and the comb FF size;
> piece-7 closure checks (S042.609, ΛcΛ̄cK, D_sD_sK, the D(\*)πℓν node tension); at B9, whether the comb part
> of B⁰ → D⁻τν follows the fitted R(D⁺) (the D τν correction is `bkg_model_only`, never in the SR signal
> template or the signal fit; SM R(D) is used), sharing NPs with the SR templates, the SR to-do list of
> review 3.8 (HAMMER and MCAmbulance on every semileptonic B decay of the event, workspace modifiers), and
> carrying the global weights into the SR; the inclusive D-decay composition for Part C / H3.
>
> How I want to work:
> - PDF files made for me use the US Letter page size.
> - Before the options of each point, explain the context briefly: what problem the point solves and
>   where it sits in the model (also as a short context paragraph in the record).
> - One point at a time. For each point, show me the facts (with decay-file line numbers and PDG
>   ids, and the source papers when a number matters) and the options with your recommendation, and
>   wait for my decision.
> - After each decision, record it (status, decision, reason) in the record file, with a decision
>   table with current values at the top and a dated record below. Implement only when I say so, rerun
>   the scripts and the tests, and tell me what changed (before/after for every value that moved).
> - Raise anything abnormal instead of fixing it silently. The pyhf workspace and the SR fit are
>   preliminary: a missing systematic or modifier there is a to-do item to list, not an anomaly.
>   Add decay-file problems to `docs/DECAY_FILE_ISSUES_FOR_BELLE2.md` and critical references to
>   `docs/REFERENCES.md`. Belle II internal documents (PDFs) are never committed.
> - Check numbers against their source (decay file, PDG via pdgapi.lbl.gov, the papers on
>   arXiv/INSPIRE, the basf2 source on GitHub: belle2/basf2, tag light-2609-luna or release-08) rather
>   than from memory, and say so when you could not check something.
> - Dates: write the actual date of the day the work is done; check it.
> - If a web site you need is blocked, tell me the host and I will allow it (gitlab.cern.ch was not
>   reachable). New directories with files to commit may need a whitelist line in `.gitignore` (it
>   ignores everything by default).
>
> Start with an overview of what remains of B0 (coding the piece-4 model) and of the next step (B1,
> composition study): propose the order and the code layout, with the facts that decide them, and wait
> for my agreement before the first point.
