# References

Register of the documents behind the decisions of Part B (step B0) of
`docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md` and of the review files (`docs/B0_PIECE2_REVIEW.md`;
piece 3 later). One row per document: what it is, where to get it, when we read it, which decisions
rest on it, and what we took from it.

Created 2026-11-18. Identifiers (arXiv, journal, INSPIRE, PDG document id) were taken on 2026-11-18
from the pdgapi.lbl.gov listings and the INSPIRE API, not from memory.

**No PDF files in this repository.** The repository is public on GitHub. Public papers are cited by
their arXiv / journal / INSPIRE identifiers. Belle II internal documents (notes, drafts) must not be
committed here (deleting them later does not remove them from the history): they are listed by note
number and version, and kept in a private place (Belle II internal document system; or a private
repository or drive). A row says where the copy we read is kept.

Status of a row: **read** (the paper itself was read for the decision, with date); **listing**
(only the PDG listing — values, footnotes — was used; the paper was not read).

## 1. Belle II internal documents

| Document | Title, authors | Version read | Read | Copy kept | Used for | What we took |
|---|---|---|---|---|---|---|
| BELLE2-NOTE-PH-2022-046 (number from the file name; the title page reads "BELLE2-NOTE-PH-2022-xxx") | Modelling of the neutral B decays used in hadronic B-tagging; M. Liu, V. S. Vobbilisetti, K. Trabelsi | DRAFT v0.x.x, 7 October 2022 | 2026-11-17 | provided by the user (file `Hadronic_FEI_correction1.pdf`); private location: to be filled in | G3/G4 (review 5a), D1, M2; B.13 | B⁰ tag-mode corrections: non-resonant lines set to 0 or to the remainder of the inclusive total ("∪"); unmeasured lines set to 0 and the BF given back to PYTHIA; our rel08 decay file already contains these changes ("LMH 06/2023") |
| BELLE2-NOTE-PH-2022-002 | Modelling of the charged B decays used in hadronic B-tagging; G. de Marino, S. El Dine Hammoud, K. Trabelsi, V. S. Vobbilisetti, V. Bertacchi | v1.0.1, 16 October 2023 | 2026-11-17 | provided by the user (file `Hadronic_FEI_correction2.pdf`); private location: to be filled in | as above | B⁺ tag-mode corrections ("NR 06/2023" lines, e.g. 4252–4254, 4301, 4315–4321) |
| BELLE2-NOTE-PH-2025-051 | B → X_sγ with hadronic tagging (MC15rd) | the excerpt read starts at section 6.3 (file `BELLE2-NOTE-PH-2025-051_v5_part2.pdf`); B.13 calls it v0.4, but the excerpt has no title page, so the version is not confirmed | 2026-11-18 | provided by the user; private location: to be filled in | piece 3 (point 1), piece 4; P5 | section 6.3, tables 18 and 19: per-mode weights with one nuisance parameter each; "unmatched" modes scaled by a common free factor (≈ 1.2); their weights for D D_s0\*, D̄⁰ρ⁺, D\*a₁ agree with our c |
| BELLE2-PUB-PH-2025-024 | Studies of the decays B⁰ → D\*⁻ηπ⁺ and B⁺ → D̄⁰ηπ⁺ in Belle and Belle II; Belle II collaboration | v12, JHEP draft, plots marked preliminary; not on arXiv or INSPIRE, not in PDG (2026-11-18) | 2026-11-18 | provided by the user; private location: to be filled in | P5; B.13 (D(\*)ηℓν and the gap) | B(B⁰ → D\*⁻ηπ⁺) = (2.78 ± 0.24 ± 0.19)·10⁻⁴, B(B⁺ → D̄⁰ηπ⁺) = (5.70 ± 0.26 ± 0.44)·10⁻⁴ (abstract; pp. 25, 27); D_s band 1.90–2.05 GeV vetoed; resonance content; D\*\*/D′ → D(\*)η limit |

## 2. Data sources

| Source | Version | Where | Used for |
|---|---|---|---|
| Review of Particle Physics (PDG), B⁰ and B⁺ branching fractions | 2026 edition | `pdg` Python package (`pdg_match.build_pdg_table`), cached in `Samples/dec/pdg_bf_B_2026.json`; listings with measurements and footnotes at `https://pdgapi.lbl.gov/listings/<id>` | every c of piece 2 |
| Belle II decay file | release 08 (MC16) | `Samples/dec/DECAY_BELLE2_rel08.DEC`, particle table `Samples/dec/evt_rel08.pdl` | BF_MC; line numbers in all documents |

## 3. Publications

### 3.1 Double-charm b → cc̄s modes (D D̄ K, D\* D K, ψ(3770)K)

| Reference | arXiv | Journal | INSPIRE | PDG id | Status | Used for | What we took |
|---|---|---|---|---|---|---|---|
| LHCb, Amplitude analysis of B⁺ → D⁺D⁻K⁺ | 2009.00026 | PRD 102, 112003 (2020) | 1814324 | AAIJ 2020AI | read 2026-11-17, 2026-11-18 | P1, G5, P2 | Table 4 fit fractions: ψ(3770) 14.5, χ_c0(3930) 3.7, χ_c2(3930) 7.2, ψ(4040) 5.0, ψ(4160) 6.6, ψ(4415) 9.2, X₀(2900) 5.6, X₁(2900) 30.6, non-resonant 24.2 % (= PDG S041.765) |
| LHCb, Observation of new charmonium(-like) states in B⁺ → D\*±D∓K⁺ | 2406.03156 | PRL 133, 131902 (2024) | 2794793 | AAIJ 2024AB | read 2026-11-18 | P2 | Table 1: NR(1⁻⁻, 0⁻⁻, 1⁺⁺, 0⁻⁺) waves ≈ 50–55 % of the fit fraction (61 % of the PDG sum S041.337+338); η_c(3945), h_c(4000), χ_c1(4010), h_c(4300), T_cs̄ (PDG S041.822–833) |
| BaBar, Measurement of the B → D̄(\*)D(\*)K branching fractions | 1011.3929 | PRD 83, 032004 (2011) | 877816 | DEL-AMO-SANCHEZ 2011B | read 2026-11-18 | G5, P6; piece 3 (inclusive D D̄ K) | the 22 D(\*)D̄(\*)K totals, "including nonresonant and resonant modes" (D_s1(2536), ψ(3770), X(3872) inside) |
| BaBar, Study of resonances in exclusive B decays to D̄(\*)D(\*)K | 0708.1565 | PRD 77, 011102 (2008) | 758167 | AUBERT 2008B | listing | P1 | ψ(3770)K⁺ × B(ψ → D D̄) products |
| BaBar, Dalitz plot analyses of B⁰ → D⁻D⁰K⁺ and B⁺ → D̄⁰D⁰K⁺ | 1412.6751 | PRD 91, 052002 (2015) | 1335269 | LEES 2015C | listing | P1 | ψ(3770)K⁺ × B(ψ → D⁰D̄⁰) |
| Belle, Observation of B⁺ → ψ(3770)K⁺ | hep-ex/0307061 | PRL 93, 051803 (2004) | 623994 | CHISTOV 2004 | listing | P1 (E14) | its total = sum of its two products |
| Belle, Observation of a new D_sJ meson in B⁺ → D̄⁰D⁰K⁺ | 0707.3491 | PRL 100, 092001 (2008) | 756554 | BRODZICKA 2008 | listing | P1 | ψ(3770)K⁺ × B(ψ → D⁰D̄⁰) |
| BaBar, Absolute branching fractions of B± → K± X_cc̄ | 1911.11740 | PRL 124, 152001 (2020) | 1767137 | LEES 2020C | listing | P1 | recoil ψ(3770)K⁺ |
| Belle, Absolute branching fractions of B⁺ → X_cc̄K⁺ | 1709.06108 | PRD 97, 012005 (2018) | 1624416 | KATO 2018 | listing | P1 | recoil ψ(3770)K⁺ limit |
| Belle, B⁰ → D(\*)±D∓ branching fractions and CP asymmetries | 1203.6647 | PRD 85, 091106 (2012) | 1095407 | ROHRKEN 2012 | read 2026-11-16 | D3 (E4, E5) | B(B⁰ → D\*±D∓) = (6.14 ± 0.29 ± 0.50)·10⁻⁴, both charge states together |
| Belle, Observation of B → D(\*)K⁻K(\*)⁰ | hep-ex/0207041 | PLB 542, 171 (2002) | 590700 | DRUTSKOY 2002 | read 2026-11-17 | G6 (E16) | B⁻ → D\*⁰K⁻K\*⁰; conjugated without flipping the K\*⁰ in the decay file |
| Belle II, B̄ → D(\*)K⁻K(\*)⁰(S) branching fractions | 2406.06277 | JHEP 08 (2024) 206 | 2796563 | ADACHI 2024L | listing | G6 | second input of S041.289 |
| BaBar, B⁰ → D(\*)±K⁰π∓ branching fraction and Dalitz distribution | hep-ex/0412040 | PRL 95, 171802 (2005) | 667017 | AUBERT,BE 2005B | read 2026-11-17 | G6 (E15) | both charge combinations measured together (K_S) |

### 3.2 D(\*)nπ, D\*\* (direct lines, G3/G4, D1)

| Reference | arXiv | Journal | INSPIRE | Status | Used for | What we took |
|---|---|---|---|---|---|---|
| LHCb, Dalitz plot analysis of B⁰ → D̄⁰π⁺π⁻ | 1505.01710 | PRD 92, 032002 (2015) | 1367306 | read 2026-11-17 | G3/G4 (5c) | resonance content of D̄⁰π⁺π⁻ |
| Belle, Study of B̄⁰ → D⁰π⁺π⁻ | hep-ex/0611054 | PRD 76, 012006 (2007) | 733011 | read 2026-11-17 | G3/G4 (5c) | same |
| LHCb, Amplitude analysis of B⁻ → D⁺π⁻π⁻ | 1608.01289 | PRD 94, 072001 (2016) | 1479164 | read 2026-11-17 | G3/G4 (5c) | S-wave fraction of D⁻π⁺π⁺ |
| Belle, Study of B⁻ → D\*\*⁰π⁻ (D\*\*⁰ → D(\*)⁺π⁻) | hep-ex/0307021 | PRD 69, 112002 (2004) | 623102 | read 2026-11-17 | G3/G4 (5c) | same |
| BaBar, Dalitz plot analysis of B⁻ → D⁺π⁻π⁻ | 0901.1291 | PRD 79, 112004 (2009) | 810694 | read 2026-11-17 | G3/G4 (5c) | same |
| LHCb, First observation of B̄⁰ → D⁺K⁻π⁺π⁻ and B⁻ → D⁰K⁻π⁺π⁻ | 1201.4402 | PRL 108, 161801 (2012) | 1085516 | listing | D1 | input of the D⁻π⁺π⁺π⁻ total S042.88 |

### 3.3 Charmonium + K(π)

| Reference | arXiv | Journal | INSPIRE | Status | Used for | What we took |
|---|---|---|---|---|---|---|
| LHCb, Amplitude analysis of B⁺ → ψ(2S)K⁺π⁺π⁻ | 2407.12475 | JHEP 01 (2025) 054 | 2808551 | read 2026-11-17 | E6; interference allowance (5a) | Table 1: three K₁(1270) waves, 25 fit fractions summing to 102.7 ± 8.6 % |
| CDF, A study of B⁰ → J/ψK(\*)⁰π⁺π⁻ | hep-ex/0108022 | PRL 88, 071801 (2002) | 561343 | read 2026-11-18 (P2 check) | G7, P6 | ψ(2S)K(\*)⁰ counted inside J/ψK\*⁰π⁺π⁻ |
| Belle, K⁺π⁺π⁻ in B⁺ → J/ψK⁺π⁺π⁻ and ψ(2S)K⁺π⁺π⁻ | 1009.5256 | PRD 83, 032005 (2011) | 871475 | read 2026-11-18 (P2 check) | P6 | ψ(2S)K vetoed |
| Belle, Search for a J/ψη resonance in B± → J/ψηK± | 1209.0886 | conference version | 1184481 | read 2026-11-18 (P2 check) | P6 | result quoted excluding ψ(2S)K |
| LHCb, Charmonium decays to K⁰_SKπ in B → (K⁰_SKπ)K | 2304.14891 | PRD 108, 032010 (2023) | 2655326 | listing | P4 | S041.797–799 (to be read before P4 is decided) |

### 3.4 Charmed baryons

| Reference | arXiv | Journal | INSPIRE | PDG id | Status | Used for | What we took |
|---|---|---|---|---|---|---|---|
| BaBar, Study of B̄⁰ → Λ_c⁺p̄π⁺π⁻ and its intermediate states | 1302.0191 | PRD 87, 092004 (2013) | 1217425 | LEES 2013H | read 2026-11-18 | P2, P6 | Table VII: Σ_c(2455)⁰ 9.1, Σ_c(2455)⁺⁺ 21.3, Σ_c(2520)⁰ 2.2, Σ_c(2520)⁺⁺ 11.5, non-Σ_c 79, total 123 (10⁻⁵); "non-resonant" = not through Σ_c(2455/2520) |
| Belle, Charmed baryonic decays B̄⁰ → Σ_c p̄π | hep-ex/0608025 | PRD 75, 011101 (2007) | 723504 | PARK 2007 | read 2026-11-18 | P2, P6 | Table II; "non-resonant" by subtraction of the four Σ_c modes; total includes them |

### 3.5 Charmless (P2 charmless lines, P3, P6; share 0)

| Reference | arXiv | Journal | INSPIRE | Status | Used for | What we took |
|---|---|---|---|---|---|---|
| BaBar, CP violation in Dalitz-plot analyses of B⁰ → K⁺K⁻K⁰_S, B⁺ → K⁺K⁻K⁺, B⁺ → K⁰_SK⁰_SK⁺ | 1201.5897 | PRD 85, 112010 (2012) | 1086537 | listing (footnote) | P6 | totals keep φ and χ_c0 |
| BaBar, B⁺ → K\*⁺K⁺K⁻ and related | hep-ex/0607113 | PRD 74, 051104 (2006) | 722714 | read 2026-11-18 (P2 check) | P6 | charmless total includes φK\* |
| BaBar, B⁰ → K\*⁰K⁺K⁻ and related | 0708.2543 | PRD 76, 071104 (2007) | 758594 | read 2026-11-18 (P2 check) | P6 | same |
| Belle, Evidence for B⁰ → K⁺K⁻π⁰ | 1304.5312 | PRD 87, 091101 (2013) | 1229032 | read 2026-11-18 (P2 check) | P6 | only the D⁰ window vetoed |
| BaBar, Observation of B⁺ → K⁺π⁰π⁰ | 1005.3717 | – | 855083 | read 2026-11-18 (P2 check) | P6 | only K_S vetoed |

## 4. Adding a row

Add a row when a document enters a decision: identifiers from INSPIRE or the PDG listing, the
version read, the date, the decision labels (as in the review file), and the numbers taken with
page or table. Internal documents: note number, version, and where the private copy is.
