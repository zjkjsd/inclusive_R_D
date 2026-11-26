# References

Register of the documents that decided something in Part B (step B0) of
`docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md` and in the review files (`docs/B0_PIECE2_REVIEW.md` and its dated record `docs/B0_PIECE2_RECORD.md`; piece
3 later). **Only critical references:** a document enters when a decision or a correction value
rests on what it says beyond the PDG average (a number, a definition, a correction of the listing).
Papers that only appear as inputs of a PDG average, or that were read as background, are not listed;
the PDG identifier in the review file is enough for them.

Created 2026-11-18. Identifiers (arXiv, journal, INSPIRE, PDG document id) taken on 2026-11-18 from
the pdgapi.lbl.gov listings and the INSPIRE API.

**No PDF files in this repository.** The repository is public on GitHub. Public papers are cited by
their identifiers. Belle II internal documents must not be committed here (deleting them later does
not remove them from the history); they are listed by note number and version, and the copy is kept
in a private place. The version is the one printed in the document, not the one in a file name.

## 1. Belle II internal documents

| Document | Title, authors | Version | Copy kept | Decisions | What we took |
|---|---|---|---|---|---|
| BELLE2-NOTE-PH-2022-046 (number from the file name; the title page reads "BELLE2-NOTE-PH-2022-xxx") | Modelling of the neutral B decays used in hadronic B-tagging; M. Liu, V. S. Vobbilisetti, K. Trabelsi | DRAFT v0.x.x, 7 October 2022 | private: to be filled in | G3/G4, D1, M2 | B⁰ tag-mode corrections (non-resonant lines set to 0 or to the remainder "∪" of the inclusive total; unmeasured lines set to 0, BF given back to PYTHIA); our rel08 decay file already has them ("LMH 06/2023") |
| BELLE2-NOTE-PH-2022-002 | Modelling of the charged B decays used in hadronic B-tagging; G. de Marino, S. El Dine Hammoud, K. Trabelsi, V. S. Vobbilisetti, V. Bertacchi | v1.0.1, 16 October 2023 | private: to be filled in | G3/G4, D1 | B⁺ tag-mode corrections ("NR 06/2023" lines 4252–4254, 4301, 4315–4321) |
| BELLE2-NOTE-PH-2025-051 | B → X_sγ with hadronic tagging (MC15rd) | v0.4, 18 September 2026; over 200 pages in three files; file 2 (from section 6.3) read on 2026-11-18 | private: to be filled in | piece 3 proposal (point 1), piece 4; pointer to P5 | section 6.3, tables 18 and 19: weights per corrected mode with one nuisance parameter each; "unmatched" modes scaled by one common free factor (≈ 1.2) |
| Belle II xwiki, Physics SLMissing / MC Samples WG1: "Correction of BF(B to Xc l ν) in generic MC" | F. Bernlochner, M. Prim (HFLAV 2023 table), maintained by R. van Tonder | version of 2026-11-17 (printed by the user 2026-11-26) | private: printout with the user | piece 4: 4.7 | the official Belle II semileptonic recipe: HFLAV 2023 isospin values, non-resonant D(\*)πℓν masked, gap filled with D(\*)ηℓν 0.88 / 0.92 % (100 % uncertainty in the text) |
| M. Prim, "Update of b → c ℓν MC Branching Fractions & Gap Model" (indico event 12225, GapUpdate.pdf) | M. Prim | slides (preliminary), received 2026-11-26 | private: with the user | piece 4: 4.7 | plan for the generic decay file (gap through intermediate resonances, Goity–Roberts modes removed; 25/75 Dη : D\*η considered) |
| BELLE2-NOTE-PH-2026-035 | Search for B → D(\*)ηℓν with hadronic FEI tagging; D. Ivanov, L. Wang, T. Koga, Q.-D. Zhou, T. Iijima | Version 8, 1 November 2026 (MC16rd) | private: to be filled in | piece 4: 4.1, 4.2, 4.6 (4.7 next) | nominal weights + SysVar eigen-NPs, PYTHIA at 1 ± 100 % (4.1); PDG D\* correlation matrix (4.2); Table 4.4 SLME D\*\* and gap values, compared in 4.6 (we chose HFLAV) |
| BELLE2-NOTE-PH-2025-031 | Measurement of R(D\*) and P_τ in hadronic 1-prong τ decays with the hadronic FEI; I. Tsaklidis et al. | Version 4.2, 8 December 2025 (MC15rd); parts 1, 2, 4 read | private: to be filled in | piece 4: 4.1, 4.6 | SysVar procedure (4.1); R(D\*\*) source chain to BLR 2018, τ gap definition (4.6) |
| BELLE2-NOTE-PH-2023-037 | Probing lepton universality in inclusive semileptonic B decays, R(X_e/μ) and R(X_τ/ℓ); H. Junkerkalefeld et al. | v5.04, 9 August 2023 (MC14ri_a) | private: to be filled in | piece 4: 4.6 (4.7 next) | 2022 version of the semileptonic recipe (its decay-file column equals our release 08); D\*\*τν correction precedent |
| BELLE2-PUB-PH-2025-024 | Studies of the decays B⁰ → D\*⁻ηπ⁺ and B⁺ → D̄⁰ηπ⁺ in Belle and Belle II | v12, JHEP draft (plots preliminary); not on arXiv, INSPIRE or in PDG on 2026-11-18 | private: to be filled in | P5 | B(B⁰ → D\*⁻ηπ⁺) = (2.78 ± 0.24 ± 0.19)·10⁻⁴, B(B⁺ → D̄⁰ηπ⁺) = (5.70 ± 0.26 ± 0.44)·10⁻⁴ (abstract; pp. 25, 27); D_s band 1.90–2.05 GeV vetoed (about 5 %, the extra uncertainty) |

## 2. Data sources and generator

| Source | Version | Where | Decisions |
|---|---|---|---|
| Review of Particle Physics (PDG), B⁰ and B⁺ branching fractions | 2026 edition | `pdg` package, cached in `Samples/dec/pdg_bf_B_2026.json`; listings with footnotes at `https://pdgapi.lbl.gov/listings/<id>` | every c of piece 2 |
| Belle II decay file | release 08 (MC16) | `Samples/dec/DECAY_BELLE2_rel08.DEC`, `Samples/dec/evt_rel08.pdl` | BF_MC; all line numbers |
| Belle II EvtGen patch `evtgen-02-00-01-avoid-double-counting.patch` | applied in belle2/externals `Makefile.targets` since 2021-03-17 (EvtGen R02-00-01) | `https://github.com/belle2/externals`, `patches/` | piece 1 (exclusive vs PYTHIA), P5: PYTHIA redecays a B whose first-generation daughters equal, as a set, any non-PYTHIA line of its table (zero-BF lines included) |

## 3. Publications

| Reference | arXiv | Journal | INSPIRE | PDG id | Read | Decisions | What we took |
|---|---|---|---|---|---|---|---|
| BaBar, Study of inclusive B⁻ and B̄⁰ decays to flavor-tagged D, D_s and Λ_c | hep-ex/0606026 | PRD 75, 072002 (2007) | 719111 | AUBERT 2007N | 2026-11-19 (via the piece-3 check) | piece 3: 3.2, 3.5 | per-flavor upper- and lower-vertex D⁰, D⁺, D_s, Λ_c multiplicities (Table IV): the inputs of the N₂, r_s prior and of the D⁺/D⁰ check; still the only per-flavor measurement (2026-11-20) |
| Belle, e⁺e⁻ → B_s⁰B̄_s⁰X cross section from 10.63 to 11.02 GeV | 2305.10098 | JHEP 08 (2023) 131 | 2660525 | ZHUKOVA 2023 | 2026-11-19 | piece 3: 3.2, 3.5 | B → D_s± X = 11.28 ± 0.43 %, B → D⁰/D̄⁰ X = 66.63 ± 1.77 % (Υ(4S), multiplicities, average of B⁺ and B⁰) |
| LHCb, Amplitude analysis of B⁺ → D\*⁻D_s⁺π⁺ (branching fraction and D\*\* fit fractions) | 2405.00098 | JHEP 08 (2024) 165 | 2782517 | – (not in PDG 2026) | 2026-11-20 | piece 3: 3.4 | B(B⁺ → D\*⁻D_s⁺π⁺)/B(B⁰ → D\*⁻D_s⁺) = 0.173 ± 0.012; B(D\*⁻D_s\*⁺π⁺)/B(D\*⁻D_s⁺π⁺) = 1.32 ± 0.16; fit fractions D₁(2420) 74.8 %, D₁(2430) 14.7 %, D₂\* 11.7 %: c of the 12 D\*\*D_s lines |
| LHCb, Amplitude analyses of B⁰ → D̄⁰D_s⁺π⁻ and B⁺ → D⁻D_s⁺π⁺ | 2212.02717 | PRD 108, 012017 (2023) | 2682639 | – (not in PDG 2026) | 2026-11-20 | piece 3: 3.4 | fit fractions (D₂\* 22 %, virtual D\* 15 %, Dπ S-wave 40–53 %): with the D₂\* bridge, c of the 8 three-body D_s Dπ lines |
| Belle, First observation of B → D̄₁(→ D̄π⁺π⁻)ℓ⁺ν and measurement of the B → D(\*)ππℓν branching fractions | 2211.09833 | PRD 107, 092003 (2023) | 2512112 | MEIER 2023 | 2026-11-20 | piece 3: 3.4, 3.7 | B → D₁ℓν through D₁ → Dππ and D₁ → D\*π in one analysis: D₁ → Dππ/D\*π (B⁺ 0.42 ± 0.07, B⁰ 0.33 ± 0.08) |
| LHCb, Branching fractions for B(s) → D(s)πππ and Λ_b → Λ_cπππ | 1109.6831 | PRD 84, 092001 (2011) | 930219 | AAIJ 2011E | 2026-11-20 | piece 3: 3.4, 3.7 | B⁺ → D̄₁⁰π⁺ through D̄₁ → D̄⁰π⁺π⁻ (non-D\*) and D\*⁻π⁺ in one analysis: ratio 0.29 ± 0.09 |
| HFLAV, Averages of b-hadron, c-hadron and τ-lepton properties as of 2023 | 2411.18639 | PRD 113, 012008 (2026) | – | – | 2026-11-23 | piece 4: 4.6 | B(B⁰ → D\*ℓν) 4.90 ± 0.11 % (eq. 90), B(B⁰ → Dℓν) 2.11 ± 0.05 % (eq. 97), D\*\*ℓν products (Tables 34–37), SM R(D) 0.296, R(D\*) 0.254, inclusive 10.63 ± 0.15 % (eq. 106): the 4.6 overrides and τ ratios |
| LHCb, Evidence for B⁻ → D\*\*⁰τ⁻ν̄ | 2501.14943 | PRL 135, 021802 (2025) | – | – | 2026-11-26 (via the 4.6 check) | piece 4: 4.6 | R(D\*\*⁰₁,₂) = 0.13 ± 0.04 against SM 0.09 ± 0.02 (× 1.44): the narrow D\*\*τν ratios |
| Bernlochner, Ligeti, Robinson, Model independent analysis of semileptonic B decays to D\*\* for arbitrary new physics | 1711.03110 | PRD 97, 075011 (2018) | – | – | 2026-11-26 (via the 4.6 check) | piece 4: 4.6 | R(D₀\*) 0.08 ± 0.03, R(D₁′) 0.05 ± 0.02, R(D₁) 0.10 ± 0.02, R(D₂\*) 0.07 ± 0.01 (eq. 38) |
| LHCb, Amplitude analysis of B⁺ → D⁺D⁻K⁺ | 2009.00026 | PRD 102, 112003 (2020) | 1814324 | AAIJ 2020AI | 2026-11-17, 2026-11-18 | P1, P2 | Table 4 fit fractions (ψ(3770) 14.5 %, non-resonant 24.2 %, components without a decay-file line 85.5 %): the D⁺D⁻K⁺ direct line 4216 keeps its remainder; the tension with the ψ(3770)K⁺ products (P1 widening) |
| LHCb, Observation of new charmonium(-like) states in B⁺ → D\*±D∓K⁺ | 2406.03156 | PRL 133, 131902 (2024) | 2794793 | AAIJ 2024AB | 2026-11-18 | P2 | NR waves ≈ 61 % of the PDG sum; the rest in resonances without a line: lines 4217, 4218 keep their remainder |
| LHCb, Amplitude analysis of B⁺ → ψ(2S)K⁺π⁺π⁻ | 2407.12475 | JHEP 01 (2025) 054 | 2808551 | AAIJ 2025Q | 2026-11-17 | E6; interference allowance of G3/G4 | S041.858 is one of three K₁(1270) waves (excluded); 25 fit fractions summing to 102.7 ± 8.6 % |
| Belle, Observation of B⁺ → ψ(3770)K⁺ | hep-ex/0307061 | PRL 93, 051803 (2004) | 623994 | CHISTOV 2004 | PDG listing | P1 (E14) | its total is the sum of its two products and is superseded in them, yet still drives S041.360: the total is excluded |
| Belle, B⁰ → D(\*)±D∓ branching fractions and CP asymmetries | 1203.6647 | PRD 85, 091106 (2012) | 1095407 | ROHRKEN 2012 | 2026-11-16 | D3 (E4, E5) | B(B⁰ → D\*±D∓) = (6.14 ± 0.29 ± 0.50)·10⁻⁴ is the sum of both charge states |
| BaBar, B⁰ → D(\*)±K⁰π∓ branching fraction and Dalitz distribution | hep-ex/0412040 | PRL 95, 171802 (2005) | 667017 | AUBERT,BE 2005B | 2026-11-17 | G6 (E15) | both charge combinations measured together |
| Belle, Observation of B → D(\*)K⁻K(\*)⁰ | hep-ex/0207041 | PLB 542, 171 (2002) | 590700 | DRUTSKOY 2002 | 2026-11-17 | G6 (E16) | B⁻ → D\*⁰K⁻K\*⁰; the decay file's B⁺ line conjugated without flipping the K\*⁰: vetoed |
| LHCb, Study of charmonium decays to K⁰_SKπ in the B → (K⁰_SKπ)K channels | 2304.14891 (v2) | PRD 108, 032010 (2023) | 2655326 | AAIJ 2023AH | 2026-11-18 | P4 | the four-body totals keep charmonium (Tables 20, 21); their normalization (and that of χ_c0K⁰π⁺) is 2.89 times the paper's own ratios × reference (eq. 35, Tables 21–24; our reading, unconfirmed): S041.797–799 excluded |
| BaBar, Measurement of the B → D̄(\*)D(\*)K branching fractions | 1011.3929 | PRD 83, 032004 (2011) | 877816 | DEL-AMO-SANCHEZ 2011B | 2026-11-18 | P6; piece 3 | the D(\*)D̄(\*)K totals include resonant modes (D_s1(2536), ψ(3770), X(3872)): D_s1(2536) folded into eight totals |
| Belle, B⁰ → D\*⁺D\*⁻K⁰_S branching fraction and time-dependent CP asymmetry | 0706.2045 | PRD 76, 072004 (2007) | 753101 | DALSENO 2007 | 2026-11-18 | P6 (S042.266) | BF from all candidates; the D_s1(2536) region removed only as a check of the CP fit: D_s1 inside the total |
| BaBar, B⁰ → D\*⁺D\*⁻K⁰_S branching fraction and time-dependent CP asymmetry | hep-ex/0608016 | PRD 74, 091101 (2006) | 723331 | AUBERT,B 2006Q | 2026-11-18 | P6 (S042.266) | D_s1(2536) fraction of the total 0.092; vetoed only in the time-dependent analysis: D_s1 inside the total |
| BaBar, Study of B̄⁰ → Λ_c⁺p̄π⁺π⁻ and its intermediate states | 1302.0191 | PRD 87, 092004 (2013) | 1217425 | LEES 2013H | 2026-11-18 | P2, P6 | "non-resonant" = not through Σ_c(2455/2520); the total includes the Σ_c modes (Table VII) |
| Belle, Charmed baryonic decays B̄⁰ → Σ_c p̄π | hep-ex/0608025 | PRD 75, 011101 (2007) | 723504 | PARK 2007 | 2026-11-18 | P2, P6 | same definition, by subtraction of the four Σ_c modes (Table II) |

## 4. Adding a row

Add a row when a document decides something: identifiers from INSPIRE or the PDG listing, the
version printed in the document, the date read, the decision labels as in the review file, and the
numbers taken with page or table. Remove a row if the decision it supported is replaced and nothing
else rests on it.
