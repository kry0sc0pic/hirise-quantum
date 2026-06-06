# Academic Peer Review Report — `paper/main.tex`

**Paper title:** Quantum Similarity Learning for Mars Terrain Retrieval via Entangled-Pair Encoding
**Submitting venue (declared):** IEEE Transactions on Geoscience and Remote Sensing
**Review mode:** `full` (EIC + R1 Methodology + R2 Domain + R3 Perspective + Devil's Advocate + Editorial Synthesizer)
**Review date:** 2026-06-06
**Reviewer team:** simulated panel (academic-paper-reviewer v1.9.1)
**Relation to prior review:** Re-evaluation post-Commits e538308, b0b2d38, 1773101, 8155dbc, 35c353e. Train/val/test contamination, MAP@K inflation, and validation-metric proxy issues from REVIEW_REPORT.md (2026-05-30) are confirmed resolved. This report scores the current draft afresh.

---

## Phase 0 — Field Analysis & Reviewer Configuration

| Dimension | Determination |
|-----------|---------------|
| Primary discipline | Quantum machine learning (PQC-based metric learning) |
| Secondary discipline | Planetary remote sensing (HiRISE imagery, Mars terrain analysis) |
| Research paradigm | Empirical computational study; supervised retrieval; ablation-driven |
| Methodology type | Hybrid classical–quantum (PennyLane simulator); Siamese CNN + PQC head; triplet loss |
| Target tier | IEEE Transactions (TGRS — Q1 remote-sensing methodology) |
| Paper maturity | Submission-ready draft. All ablation cells now populated with test-set macro-MAP@10 (commit b0b2d38). Architecture (Fig 2) and circuit (Fig 3) figures are now real (commits 35c353e, e538308). |
| Length | ~14 pages double-column (typical TGRS) |
| Author count | Single author (atypical for TGRS; not disqualifying) |

### Reviewer panel (dynamically configured)

| # | Role | Configured identity |
|---|------|--------------------|
| **EIC** | Editor-in-Chief | Associate editor at IEEE TGRS; deep-learning-for-remote-sensing background; gatekeeper for scope, originality envelope, and submission fit |
| **R1** | Methodology Reviewer | ML methodologist with retrieval-evaluation, ablation-design, and statistical-significance expertise; familiar with PyTorch/PennyLane training pipelines |
| **R2** | Domain Reviewer (QML) | Quantum machine learning theorist; familiar with PQC expressivity, barren plateaus (Cerezo et al.), SLIQ, data re-uploading, and PennyLane primitives |
| **R3** | Perspective Reviewer | Planetary scientist / HiRISE end-user; operational PDS catalog workflow; interested in actual deployability for science-team queries |
| **DA** | Devil's Advocate | Adversarial reader: contribution-claim plausibility, ablation-vs-narrative coherence, statistical-strength testing, cherry-picking detection |

---

## Phase 1 — Five Independent Reviews

> **IRON RULE applied:** Reviewers operate from non-overlapping perspectives and do not cross-reference each other's reports.

---

### Review 1 — Editor-in-Chief

**Recommendation:** Major Revision
**Confidence:** 4/5

**Summary assessment (200 words).** The manuscript presents the first application of quantum metric learning to HiRISE Mars terrain retrieval, using a Siamese CNN + entangled-pair PQC head trained with triplet loss. The five-factor ablation framework is structurally well-conceived, and the manuscript is substantially more mature than the version reviewed on 2026-05-30 — all placeholder result cells are now filled, and the architecture and circuit figures have been added. Scope is acceptable for TGRS though heavily weighted toward QML methodology (~60% of body); a more natural fit might be *npj Quantum Information*, *Quantum Machine Intelligence*, or *IEEE Transactions on Quantum Engineering*, but TGRS is defensible given the planetary-remote-sensing application framing. The headline retrieval numbers (micro-MAP@10 = 0.879, macro-MAP@10 = 0.533) are honestly reported, and the macro/micro distinction is correctly handled. **The decisive concerns** are: (1) the paper's own ablations (A3–A5) show that the chosen "default" configuration (N=8, L=2, class-pair mining) is not the best configuration found — yet the headline numbers are reported for the default rather than the best-ablation cell, and (2) the contribution claims in the abstract are inconsistent with the magnitudes of the reported deltas. These are substantive but fixable. Major Revision is appropriate.

**Strengths.**

- **S1: Clean retrieval framing.** The paper correctly identifies classification accuracy as a poor surrogate for the operationally relevant retrieval task, and explicitly disavows the 96% classification baseline (main.tex:204, 674–675). This is exactly the methodological care TGRS expects.
- **S2: Macro-MAP@10 chosen as primary metric.** Given 83.6% class imbalance, macro-MAP is the correct primary metric. The paper highlights this and resists the temptation to lead with micro-MAP (main.tex:666–667).
- **S3: Internal honesty.** The paper reports A4 (class-pair mining) and A5 (qubit count) ablations as *not* supporting the authors' claims, rather than burying them. This is reviewer-friendly and increases trust.
- **S4: Reproducibility scaffolding.** Stratified 70/15/15 split with seed 42 and a `scripts/make_splits.py` reference, plus a `default.qubit` + `diff_method="backprop"` appendix snippet (main.tex:1082–1095).

**Weaknesses (EIC-level).**

- **W1: Headline configuration is not the best-ablation configuration.** *Severity: Major.* Table I reports macro-MAP@10 = 0.533 for the default \qts{} configuration (N=8, L=2, class-pair mining). Table II shows that N=4, L=3, with random mining individually outperform — and the discussion at main.tex:1001–1002 even concedes "the default L=2 is a suboptimal choice." A reader will reasonably ask why the abstract and conclusion sell 0.533 when the authors' own results suggest a configuration that would score higher exists. *Required fix:* either re-run with the best-ablation hyperparameters and update Table I (preferred), or add an explicit "why we kept the default" justification in §5 with results to back it up.
- **W2: Scope-fit borderline for TGRS.** *Severity: Major.* Sections 3, 4.3, and the appendix are dominated by QML mechanics (PQC formalism, StronglyEntanglingLayers, barren plateaus, PennyLane API). For TGRS readership, the planetary-remote-sensing content needs to anchor the narrative more — e.g., what *new science* does this retrieval system enable that the wagstaff2021mars classifier cannot? *Required fix:* add a §6 sub-discussion on operational retrieval use cases (e.g., concrete query examples, integration with PDS workflows).
- **W3: Missing comparison to classical retrieval baselines.** *Severity: Major.* The only classical comparison is (a) Classical MLP head (A1) and (b) ResNet50 fine-tuned. Neither is a standard image-retrieval baseline. TGRS readers will expect at least one of: pretrained-CNN + cosine similarity (no triplet training), deep image hashing (DSH, HashNet), or DELF-style local-feature retrieval. *Required fix:* add 1–2 standard retrieval baselines.
- **W4: Originality claim wording.** *Severity: Minor.* "First quantum metric learning for planetary terrain" is true but narrow. The methodological novelty (interleaved $R_X$/$R_Y$ vs SLIQ's sequential $R_Z$/$R_X$) is a small variation, and the discussion at main.tex:932–938 actually acknowledges the SLIQ comparison may be confounded by axis choice. *Suggested fix:* in the abstract, soften "first quantum similarity learning system for planetary terrain retrieval" to something quantifiable.

**Detailed comments.**

*Title & Abstract.* Title is accurate and descriptive. Abstract is information-dense and now reports actual numbers (good), but oversells the contribution given the Δ=0.005 vs classical MLP (Table I).

*Introduction.* Strong motivation; the "retrieval, not classification" framing is the paper's best contribution and is well-argued.

*Related Work.* Well-curated; coverage of HiRISE prior work, QML foundations, SLIQ, classical metric learning, and barren plateaus is appropriate.

*Method.* The interleaved encoding is clearly explained with the SU(2) matrix expansion (main.tex:454–458), but the entanglement argument is mechanistic rather than measured (no entanglement entropy, no Schmidt rank — see R2 review).

*Experiments.* Train/val/test split now correct (commit history confirms `make_splits.py` was added). Single seed (42) is still a concern (see R1 review).

*Results.* Table I is honest. Table II is the paper's most informative artifact and unfortunately also where its narrative weakens.

*Discussion.* The discussion subsection on A3 (main.tex:990–1002) admits "the default L=2 is a suboptimal choice." This is honest but should propagate back to the headline numbers.

*Conclusion.* Repeats the abstract; could be tightened.

**Questions for the author.**

1. Why is the headline configuration N=8, L=2, class-pair mining when A3 (L=3 at 0.556) and A5 (N=4 at 0.540) both outperform? Did you re-run \qts{} at (N=4, L=3, random mining)?
2. Was the choice of QML methodology over standard retrieval baselines (DSH, HashNet, DELF) deliberate? If so, on what grounds?
3. Could the paper's narrative survive replacing the QML head with the Classical MLP (A1) given Δ=0.005?

**Dimension scores (universal):**

| Dimension | Score | Descriptor | Notes |
|-----------|-------|------------|-------|
| Originality | 60 | Adequate | Domain-novel, methodologically narrow |
| Methodological Rigor | 55 | Weak/Adequate boundary | Single seed; suboptimal default; conflated A2 |
| Evidence Sufficiency | 58 | Weak/Adequate | Missing standard retrieval baselines; ablation contradictions |
| Argument Coherence | 65 | Adequate | Internally honest but headline-vs-ablation mismatch |
| Writing Quality | 78 | Strong | Clean IEEE prose, good figures |
| **Weighted average** | **61.0** | **Major Revision** | |

---

### Review 2 — Methodology Reviewer

**Recommendation:** Major Revision
**Confidence:** 5/5

**Summary assessment (240 words).** The methodology has been substantially upgraded since the prior review: the train/val/test contamination is fixed (stratified 70/15/15 per-class split, seed 42, test reserved exclusively for final evaluation, main.tex:599–606); validation-set checkpoint selection now uses macro-MAP@10 rather than the triplet-margin proxy (main.tex:604–606); and class imbalance is now correctly handled by reporting macro-MAP@10 as the primary metric. These are non-trivial corrections and they materially strengthen the manuscript. **However**, several methodological issues remain: (1) *single-seed reporting throughout*: every cell in Tables I and II is a single-run number, with no variance estimate, no error bars, no multiple-seed mean ± std. The smallest reported deltas (Δ=0.005 between Classical MLP and Quantum head, Δ=0.005 between random mining and class-pair mining) are well within the typical seed-to-seed variance of a 50-epoch triplet-loss CNN trained on a moderately small dataset, and cannot be interpreted as evidence either way. (2) *Ablation defaults disagree with ablation winners*: the paper's default N=8, L=2 with class-pair mining is dominated by A3 (L=3), A5 (N=4), and A4 (random mining); this is not a methodology bug but it does mean the headline number (0.533) understates what the authors' own framework can achieve. (3) *A2 ablation conflates two factors*: interleaved $R_X$/$R_Y$ vs sequential $R_Z$/$R_X$ changes both interleaving and axis choice; the paper acknowledges this (main.tex:932–938) but doesn't disentangle. Major Revision.

**Strengths.**

- **S1: Train/val/test discipline is now correct.** Stratified per-class split, seed pinned, test reserved (main.tex:599–606). Major upgrade.
- **S2: Macro-MAP@10 chosen as primary, properly justified.** Class-imbalance handling is exemplary; the 83.6% "other" class is acknowledged as a motivator for macro reporting (main.tex:619–622, 661–667).
- **S3: Validation-metric and reported-metric now aligned.** Checkpoint selection on macro-MAP@10 (computed every 5 epochs on val) matches the headline test metric (main.tex:604–606).
- **S4: Implementation snippet in appendix.** The QNode code (main.tex:1082–1095) with the `inputs[..., :N]` vs `inputs[:N]` PennyLane TorchLayer detail is exactly the kind of reproducibility note that is usually omitted.

**Weaknesses.**

- **W1: No variance reporting; single seed.** *Severity: Critical.*
  **Problem:** Every reported number in Tables I and II is a single training run with seed 42. The deltas in A1 (Δ=0.005, classical vs quantum), A4 (Δ=0.005, random vs class-pair), and several A3/A5 cells (Δ in [0.003, 0.023]) are interpreted as substantive in the discussion (e.g., "Random mining marginally outperforms class-pair mining"). Without 3+ seeds and a reported std, these deltas cannot be distinguished from training noise. A 5-seed estimate of standard deviation of macro-MAP@10 for a triplet-loss CNN on ~73k patches at this batch size is typically 0.01–0.03 — i.e., larger than the differences being interpreted.
  **Why it matters:** The paper's central narrative (interleaved encoding is the dominant driver, head type and mining are negligible) rests on these deltas being real. Without variance, the narrative is unsupported.
  **Suggestion:** Re-run each of the 10 ablation cells × 3 seeds (minimum), report mean ± std, and update the discussion paragraphs that lean on small deltas. If compute is the blocker, prioritize the cells where the discussion makes specific claims about the delta direction (A1, A2, A4 most urgently).
  **Severity:** Critical.

- **W2: A2 ablation conflates interleaving with axis choice.** *Severity: Major.*
  **Problem:** The A2 condition compares interleaved $R_X(\pi e_a)$, $R_Y(\pi e_b)$ (qts) against sequential $R_Z(\pi e_a)$, $R_X(\pi e_b)$ (SLIQ-style). The discussion at main.tex:932–935 acknowledges that $R_Z|0\rangle = |0\rangle$ (up to global phase), so the first encoding block of the SLIQ-style baseline has no effect on the local-Z measurements that ultimately determine the loss. This means the A2 comparison is partially measuring an artifact of axis choice, not encoding ordering. The headline claim "encoding strategy as the primary driver" (main.tex:79, 740–743) therefore conflates two factors.
  **Why it matters:** The Δ=0.061 between \qts{} and SLIQ-style is the largest reported delta in the paper and the only one that survives the single-seed concern; it is the load-bearing finding. If it is partly an axis artifact, the contribution claim weakens substantially.
  **Suggestion:** Add an A2.5 condition: sequential $R_X$/$R_Y$ (same axes as qts, sequential ordering) and/or interleaved $R_Z$/$R_X$. This isolates ordering from axis. Estimate cost: ~2 additional runs.
  **Severity:** Major.

- **W3: Headline configuration is not the best-ablation configuration.** *Severity: Major.*
  **Problem:** Table II shows (N=4, L=1, random mining) regions of the ablation grid that individually outperform the (N=8, L=2, class-pair) default reported in Table I. A reasonable reader will combine these and ask: what does (N=4, L=3, random) score? The paper does not report this combined-best cell.
  **Why it matters:** The macro-MAP@10 = 0.533 headline is leaving performance on the table. A revised configuration could plausibly score 0.55–0.57 based on the marginal effects.
  **Suggestion:** Run \qts{} at (N=4, L=3, random mining) and report. If it outperforms 0.533, update Table I and the abstract. Caveat the result as "best-of-ablation" rather than retroactively claiming it as the default.
  **Severity:** Major.

- **W4: ResNet50 baseline training protocol is underspecified.** *Severity: Major.*
  **Problem:** Section 5.3 says ResNet50 is "pretrained on ImageNet" and is a "strong baseline" (main.tex:680), and the Discussion mentions "differential learning rates, grayscale-to-RGB channel repeat" (main.tex:949). The actual training procedure — frozen vs fine-tuned backbone, head architecture, learning rate, epochs, weight decay, augmentation — is not specified anywhere in the paper. The claim "ResNet50 achieves the lowest macro-MAP@10 of all models" (0.378) is being asked to do significant narrative work (it appears in abstract, results, discussion, and conclusion), but the reader has no way to assess whether ResNet50 was given a fair shake.
  **Why it matters:** A weak ResNet50 baseline inflates the *qts* narrative. A well-tuned ResNet50 (e.g., with proper triplet-loss fine-tuning, grayscale-aware first-layer reinitialization, and a similar number of training epochs as \qts{}) may score considerably higher.
  **Suggestion:** Add an "Implementation Details — ResNet50 baseline" subsection in §5.4 with: backbone state (frozen/unfrozen), first-conv reinitialization, head architecture, optimizer + LR + schedule, epochs, augmentation pipeline. Justify each choice.
  **Severity:** Major.

- **W5: Barren-plateau claim is overreached.** *Severity: Major.*
  **Problem:** Section 4.3.3 and the Results subsection on gradient norms (main.tex:859–884) claim to "verify empirically" Cerezo et al.'s prediction that local observables prevent barren plateaus. But Cerezo's theorem is a *scaling* claim — gradients decay polynomially in $N$ for local observables vs exponentially for global. The paper only tests N=8. Showing that gradients stay above $10^{-5}$ at a single qubit count does not verify the scaling behavior; it merely shows non-vanishing at one operating point.
  **Why it matters:** The "barren plateau mitigation" is claimed as Contribution 3 (main.tex:163–168). It is currently supported only by a single-N observation.
  **Suggestion:** Either (a) re-run gradient-norm analysis across N ∈ {4, 6, 8, 10, 12} and plot mean abs gradient vs N (would verify the scaling), or (b) soften the claim to "we observe that local-Z measurements produce non-vanishing gradients at N=8" (more honest, less load-bearing).
  **Severity:** Major.

- **W6: MAP@K definition still ambiguous.** *Severity: Minor.*
  **Problem:** §5.2 says "MAP@10 is averaged over all test queries" but the per-query AP@K formula is not given. The paper should explicitly state whether AP@K normalizes by `min(K, total_relevant_in_DB)` or by `relevant_in_top_K`. These two definitions can produce noticeably different aggregate numbers.
  **Suggestion:** Add one line to §5.2 with the formula and a citation to a standard IR reference (Manning et al., or the IR textbook of choice).
  **Severity:** Minor.

- **W7: Discussion of computational cost is hand-waved.** *Severity: Minor.*
  **Problem:** §6 "Computational cost" paragraph (main.tex:976–987) acknowledges that quantum simulation is "substantially slower per epoch" but provides no numbers — "wall-clock comparison is deferred to future work." For a paper whose entire enterprise is comparing quantum vs classical similarity heads, this is the comparison most readers will want.
  **Suggestion:** Even a single line ("Quantum head: X minutes/epoch on CPU; Classical MLP head: Y minutes/epoch on the same hardware") would resolve this. The data must exist; record it.
  **Severity:** Minor.

**Detailed comments.**

*Methodology / Research Design.* Soundly upgraded since the prior review. The remaining issues are above the threshold for Major Revision because the lack of variance and the A2 confound undermine the central comparative claims.

*Sampling strategy.* Stratified per-class split is correct. With only 231 impact_ejecta and 476 spider patches, the per-class test split has 35 and 71 patches respectively — small enough that per-class AP estimates are themselves high-variance.

*Statistical reporting.* Currently absent. Adding 3-seed mean ± std is the single highest-impact fix.

*Reproducibility.* Code snippet in appendix is helpful; full repository link not given (or this reviewer missed it). Add a Code Availability statement.

**Questions for the author.**

1. What is the seed-to-seed std of macro-MAP@10 for \qts{} (e.g., over 5 seeds)? Without this, every reported delta < ~0.02 is uninterpretable.
2. Have you run \qts{} with sequential $R_X$/$R_Y$ encoding (same axes, different ordering)? This is the cleanest disambiguation of the A2 finding.
3. What is the wall-clock training time for \qts{} vs the Classical MLP baseline on the same hardware?
4. Was ResNet50 fine-tuned end-to-end with triplet loss, or only its last layer? Was the first conv re-initialized to accept single-channel grayscale input, or were grayscale images repeated across RGB channels?
5. Is `scripts/make_splits.py` and the full training repo planned for public release?

**Dimension scores:**

| Dimension | Score | Descriptor |
|-----------|-------|------------|
| Originality | 60 | Adequate |
| Methodological Rigor | 50 | Weak (single seed, conflated A2, unverified BP claim) |
| Evidence Sufficiency | 55 | Weak (missing baselines, no variance, missing wall-clock) |
| Argument Coherence | 65 | Adequate |
| Writing Quality | 80 | Strong |
| **Weighted average** | **59.5** | **Major Revision** |

---

### Review 3 — Domain Reviewer (QML)

**Recommendation:** Major Revision
**Confidence:** 5/5

**Summary assessment (220 words).** This paper sits in the data-re-uploading + entangled-pair-encoding corner of the PQC literature, and at the level of citation coverage it is competent: Cerezo et al. (2021) on cost-function-dependent barren plateaus, McClean et al. (2018), SLIQ, Mari et al. (2020) transfer learning, Schuld & Killoran (2019) on feature Hilbert spaces, and PennyLane (Bergholm et al. 2018) are all present. The interleaved $R_X$/$R_Y$ encoding is a sensible variation on SLIQ's two-block design, and the SU(2) matrix expansion (main.tex:454–458) is the right mechanistic argument to ground the "non-commuting joint encoding" claim. **However**, the QML methodology has three substantive weaknesses that the QML community will flag: (1) the entanglement claim is mechanistic rather than measured — no entanglement entropy, no Schmidt rank, no concentration analysis is reported; the paper asserts that interleaved encoding creates "richer entanglement" without quantifying it; (2) the barren plateau verification is single-N, which does not verify Cerezo's scaling prediction (see R1's W5); and (3) the framing of the A2 finding as evidence that "the encoding strategy is the dominant factor" is undermined by the paper's own acknowledgement (main.tex:932–935) that the $R_Z|0\rangle = |0\rangle$ degeneracy may be doing most of the work. Major Revision.

**Strengths.**

- **S1: Correct identification of Cerezo et al. (2021) as the load-bearing barren-plateau result.** Far too many QML papers cite McClean (2018) without distinguishing global vs local cost dependence; this paper gets it right (main.tex:262–270, 489–507).
- **S2: SU(2) matrix expansion is a clean argument for joint encoding.** The explicit demonstration that $R_Y(\beta)R_X(\alpha)$ produces a matrix with off-diagonal entries mixing $\alpha$ and $\beta$ multiplicatively (main.tex:454–458) is the right level of formal argument for the "non-linear joint function" claim.
- **S3: Correct PennyLane primitives.** `StronglyEntanglingLayers` weight shape (L, N, 3), `default.qubit` simulator with `diff_method="backprop"`, and the `inputs[..., :N]` vs `inputs[:N]` TorchLayer batching detail are all correct and useful for reproduction (main.tex:1082–1102).
- **S4: Honest reporting of A2 confound in discussion.** Acknowledging the $R_Z$-on-$|0\rangle$ degeneracy (main.tex:932–935) is intellectually honest and rare in this literature.

**Weaknesses.**

- **W1: Entanglement claim is unmeasured.** *Severity: Major.*
  **Problem:** The paper claims interleaved encoding produces "richer entanglement structure conditioned on both embeddings simultaneously" (main.tex:462–465) and "deeper circuits create multi-hop cross-qubit correlations" (main.tex:998–1000). These are mechanistic intuitions, not measurements. Standard quantifications (von Neumann entanglement entropy, Schmidt rank across bipartitions, concentrating Frobenius norm, or even pairwise mutual information of the measured Pauli-Z values) are absent.
  **Why it matters:** The Δ=0.061 finding (interleaved vs SLIQ) is being explained in terms of entanglement, but the explanation has no observational support. A reader cannot distinguish "more entanglement" from "just a different axis choice" without measurement.
  **Suggestion:** For at least one trained \qts{} checkpoint and one SLIQ-style checkpoint, compute the half-chain von Neumann entanglement entropy averaged over a held-out batch of test pairs, and report. Or, compute the Schmidt rank distribution. Or, less ambitiously, plot the pairwise mutual information of $\langle Z_i \rangle, \langle Z_j \rangle$ across qubit pairs.
  **Severity:** Major.

- **W2: Barren plateau "verification" is single-N.** *Severity: Major.*
  **Problem:** Cerezo et al.'s theorem is a *scaling* statement. The empirical "verification" reported (gradient norm stays above $10^{-5}$ at N=8) is a single-point observation. The paper claims (main.tex:166–168) "we verify empirically that quantum layer gradients remain above the vanishing threshold throughout training" — but to verify the *theorem*, you need to show polynomial decay at increasing N, not just non-vanishing at one N.
  **Why it matters:** This is Contribution 3 of the paper and is positioned alongside the other contributions in the introduction. As currently reported it does not carry that weight.
  **Suggestion:** Add a gradient-norm vs N plot for N ∈ {4, 6, 8, 10, 12}, ideally with a polynomial fit overlay. If this is computationally infeasible, soften the claim to "we observe non-vanishing gradients at our operating point of N=8."
  **Severity:** Major.

- **W3: StronglyEntanglingLayers is one of many ansatz choices.** *Severity: Minor.*
  **Problem:** The choice of `StronglyEntanglingLayers` is made without comparison to alternatives (e.g., HardwareEfficientLayers, BasicEntanglerLayers, problem-inspired ansatze). Given that L=3 outperforms L=2, a reader will wonder whether a different ansatz at fixed parameter count would do better still.
  **Why it matters:** The paper is making a claim about a specific PQC; the generality to other ansatze is unaddressed.
  **Suggestion:** Either add a small comparison to `BasicEntanglerLayers` (1 line per layer; fewer parameters) or acknowledge the choice as a limitation in §6.
  **Severity:** Minor.

- **W4: NISQ-readiness discussion is abstract.** *Severity: Minor.*
  **Problem:** §6 briefly mentions shot noise (main.tex:984–986) but does not estimate the shot budget required to achieve the reported gradient quality on real hardware. Given that the paper invokes "NISQ" framing (main.tex:233) via the SLIQ citation, an estimate of shots/parameter/epoch would be valuable.
  **Suggestion:** Even a back-of-envelope estimate ("for $\epsilon = 10^{-3}$ gradient precision at N=8, approximately 10^6 shots per parameter per epoch are required") would calibrate reader expectations.
  **Severity:** Minor.

- **W5: $\pi e^{(i)}$ angle scaling not justified.** *Severity: Minor.*
  **Problem:** The encoding uses $R_X(\pi e_a^{(i)})$ with the $\pi$ scaling factor, but the choice of $\pi$ vs any other constant is not motivated. Given $e^{(i)} \in [-1, 1]$, the rotation angle is in $[-\pi, \pi]$ — full circle. This avoids periodicity issues, but a sentence stating the rationale would help.
  **Suggestion:** One line in §3.1 or §4.3.1.
  **Severity:** Minor.

**Detailed comments.**

*Background.* §3 PQC formalism is clean. Hilbert space dimension $2^8 = 256$ correctly stated.

*Method §4.3.1 Encoding.* The SU(2) expansion is correct. The interleaved-vs-sequential framing is the right way to motivate the contribution, modulo the W1 confound.

*Method §4.3.2 Ansatz.* StronglyEntanglingLayers with 3*N*L = 48 parameters at (N=8, L=2) is correct. The "brick-layer pattern of CZ" description (main.tex:484) is accurate.

*Appendix.* QNode code is correct and the `inputs[..., :N]` note is genuinely useful.

**Questions for the author.**

1. Can you report the von Neumann entanglement entropy (or Schmidt rank) of the trained \qts{} state vs the trained SLIQ-style state, averaged over a batch of held-out pairs? This would directly test whether interleaved encoding produces "richer" entanglement.
2. Have you plotted PQC parameter gradient norm vs N to verify the polynomial-decay prediction, rather than only confirming non-vanishing at N=8?
3. Did you experiment with alternative ansatze (HardwareEfficient, BasicEntangler) at matched parameter count?

**Dimension scores:**

| Dimension | Score | Descriptor |
|-----------|-------|------------|
| Originality | 62 | Adequate (encoding variant + barren-plateau-aware design) |
| Methodological Rigor | 60 | Adequate (correct primitives; entanglement claim unmeasured) |
| Evidence Sufficiency | 58 | Weak (single-N BP test, no ansatz comparison) |
| Argument Coherence | 70 | Strong (clean QML formal argument) |
| Writing Quality | 80 | Strong |
| Literature Integration (R2 focus) | 75 | Strong (Cerezo et al. correctly cited; SLIQ correctly characterized; PennyLane source correct) |
| **Weighted average** | **63.5** | **Major Revision (borderline)** |

---

### Review 4 — Perspective Reviewer (Planetary Science)

**Recommendation:** Major Revision
**Confidence:** 4/5

**Summary assessment (190 words).** From the perspective of a HiRISE end-user, the paper has a credible motivation (retrieval-by-similarity for science queries, rover trafficability comparison, change detection) but does not deliver a system the planetary science community could deploy as-is. The choice of $64\times64$ grayscale patches reduces the data to a fraction of HiRISE's actual resolution and discards all of the multi-band CRISM/CTX/CaSSIS context that real terrain analysis uses. The "operational deployment" framing (main.tex:104–106) cites the wagstaff2021mars PDS deployment but does not demonstrate a path from this work to a comparable operational artifact. The macro-MAP@10 = 0.533 is honestly reported but is not in itself an indicator that the system would be useful for real science queries — for example, a 53% precision-at-10 on a query for "regions like this swiss-cheese terrain" would still surface 4–5 irrelevant patches per query, which is a non-trivial review burden. The interesting cross-disciplinary contribution is the framing — that retrieval, not classification, is the right operational task. Major Revision to address scope/utility framing and add at least one realistic operational scenario.

**Strengths.**

- **S1: Retrieval-vs-classification framing is operationally correct.** §1 (main.tex:108–116) gets the science-team workflow exactly right: scientists want "find similar regions," not "classify this region."
- **S2: Honest about ResNet50 transfer limitations.** The discussion at main.tex:944–958 of why ImageNet features fail on HiRISE (overhead vs perspective, grayscale vs RGB, geomorphic vs semantic) is well-reasoned and matches the planetary-science literature on cross-domain transfer.
- **S3: Confusable class pairs are correctly identified.** Bright/dark dune and spider/swiss cheese are indeed the classes that geologists confuse most often, particularly at small patch scales (main.tex:572–574). This is a real domain insight.

**Weaknesses.**

- **W1: No operational scenario worked end-to-end.** *Severity: Major.*
  **Problem:** The paper motivates retrieval with three operational use cases (scientific queries, rover path planning, change detection — main.tex:111–114), but none of these are demonstrated. There is no example query patch shown, no retrieved-results visualization for a specific scientific question, no integration with an existing planetary-science workflow.
  **Why it matters:** TGRS readers are remote-sensing practitioners. A retrieval paper without a worked retrieval example will read as methodology-only and the planetary-science motivation will feel grafted-on.
  **Suggestion:** Add one figure showing: query patch (e.g., a known active aeolian dune migration site) → top-10 retrieved patches → expert evaluation of which are scientifically relevant. Even one such example makes the operational story concrete.
  **Severity:** Major.

- **W2: $64\times64$ grayscale patches lose most of HiRISE's information.** *Severity: Major.*
  **Problem:** HiRISE produces 25 cm/pixel imagery; the paper resamples to $64\times64$ grayscale, which throws away resolution, multi-color channel structure, and any context beyond a ~16 m × 16 m square. The wagstaff2021mars dataset uses this format for classification convenience, but a retrieval paper that aims to support real scientific queries should at least discuss the resolution loss and the fact that the patch size makes some terrain distinctions (spider topology, dune cross-strata) effectively invisible.
  **Why it matters:** A reader from the HiRISE community will immediately wonder whether $64\times64$ patches are operationally meaningful, or whether the entire benchmark is a proxy for a real retrieval task.
  **Suggestion:** Add a paragraph in §6 acknowledging that the HiRISE v3 patch format is a benchmark-convenience choice, and that operational deployment would need to handle full-resolution tiled queries. Optionally, run a single supplementary experiment at $128\times128$ to gesture at the resolution-sensitivity question.
  **Severity:** Major.

- **W3: No multi-instrument or multi-temporal extension demonstrated.** *Severity: Minor.*
  **Problem:** The conclusion (main.tex:1067–1071) mentions cross-instrument retrieval (HiRISE → CTX, CaSSIS) and multi-temporal change detection as future work. For a paper claiming planetary-science motivation, demonstrating at least one of these would substantially strengthen the case.
  **Severity:** Minor.

- **W4: Class taxonomy is treated as ground truth.** *Severity: Minor.*
  **Problem:** The paper inherits the 8-class HiRISE v3 taxonomy as ground truth (main.tex:99–102) and the macro-MAP@10 evaluation treats class-membership as the retrieval relevance criterion. But the operational query is rarely "find me more swiss_cheese patches" — it is "find me regions like this one." A real evaluation would use expert-annotated query–result relevance pairs.
  **Why it matters:** Macro-MAP@10 over taxonomic labels measures whether the model has learned the existing taxonomy, not whether it surfaces operationally useful neighbors.
  **Suggestion:** Acknowledge this as a limitation in §6.
  **Severity:** Minor.

**Questions for the author.**

1. Can you show one operational query scenario — e.g., a specific dune-migration study site as the query and the top retrieved patches as a manually evaluated answer?
2. At $64\times64$, is the retrieval system practically useful, or is the patch size a benchmark-convenience choice that would not generalize to real science queries on full-resolution tiles?
3. Have you discussed this work with HiRISE / PDS team members? Their input on whether macro-MAP@10 = 0.533 is operationally adequate would substantially strengthen the paper.

**Dimension scores:**

| Dimension | Score | Descriptor |
|-----------|-------|------------|
| Originality | 68 | Adequate (operational framing is the real contribution) |
| Methodological Rigor | 60 | Adequate (proper splits; ResNet50 fairness contested) |
| Evidence Sufficiency | 55 | Weak (no operational scenario worked) |
| Argument Coherence | 65 | Adequate |
| Writing Quality | 78 | Strong |
| Significance & Impact (R3 focus) | 58 | Weak/Adequate (retrieval framing is significant; deployment path is missing) |
| **Weighted average** | **62.7** | **Major Revision** |

---

### Review 5 — Devil's Advocate

> *Per skill protocol: this report does not score. It challenges.*
> *Affirmation (1 sentence, for fairness): The paper is intellectually honest in reporting ablations that undermine its own claims (A4, A5), which is unusual and admirable in this literature.*

#### Strongest Counter-Argument (280 words)

**The paper's own ablation table refutes its central thesis.**

The thesis is that an entangled-pair quantum similarity head provides a structurally motivated advantage over classical metric learning for fine-grained planetary terrain retrieval. The paper builds this case through five ablations (A1–A5) and reports macro-MAP@10 = 0.533 for the default \qts{} configuration.

But examine the ablations honestly. A1 shows the quantum head beats the classical MLP by **Δ = 0.005** macro-MAP@10. A4 shows the paper's domain-specific hard negative mining strategy is *outperformed* by random mining (Δ = 0.005 in the wrong direction). A5 shows N = 4 qubits beats N = 8, and N = 12 is worst — the opposite of what a quantum-advantage narrative predicts. A3 shows L = 3 beats L = 2, meaning the chosen default is sub-optimal.

The single delta that survives — A2's Δ = 0.061 between interleaved and SLIQ-style — is then acknowledged by the paper itself (main.tex:932–938) to be partially attributable to the fact that $R_Z|0\rangle = |0\rangle$, so SLIQ's first encoding block is essentially a no-op for the chosen measurement scheme. **The A2 baseline is degenerate. The paper compares against an axis choice that cannot activate, not against a genuine sequential encoding.**

If a hostile reader stripped the paper to its evidence — five ablations, four of which are null-or-negative-direction, and one of which is comparing against a known-degenerate baseline — the contribution claim ("first quantum metric learning system that meaningfully improves planetary terrain retrieval") does not survive. A more honest framing is: "first quantum metric learning system applied to planetary terrain, with the finding that the quantum head is not meaningfully better than a classical MLP at this scale." That is still publishable, but it is not the paper that has been written.

#### Issue List

##### CRITICAL

| # | Dimension | Issue | Location |
|---|-----------|-------|----------|
| C1 | Data–Conclusion Mismatch | A1 ablation shows Δ = 0.005 macro-MAP@10 (quantum vs classical MLP) — within typical seed-to-seed variance. Paper claims quantum head is one of five core contributions. | Tables I, II; main.tex:738–743 |
| C2 | Logic Chain Break | A2 baseline is the SLIQ-style encoding with $R_Z$/$R_X$, where the first $R_Z$ block has no effect on local Pauli-Z measurements (paper acknowledges this at main.tex:932–935). The Δ = 0.061 finding is therefore partly an artifact of the degenerate baseline, not of interleaved-vs-sequential ordering. The headline "encoding strategy is the primary driver" claim conflates these. | main.tex:79, 740–743, 932–938 |
| C3 | Foundation Collapse | The paper's own Table II shows that A3 (L=3), A5 (N=4), and A4 (random mining) each individually outperform the default configuration. The "headline" macro-MAP@10 = 0.533 is therefore for a configuration that the paper's own framework identifies as sub-optimal. The discussion concedes "the default L=2 is a suboptimal choice" (main.tex:1001–1002) but the abstract, results section, and conclusion all lead with 0.533. | main.tex:74–76, 722–724, 1052–1057 |

##### MAJOR

| # | Dimension | Issue | Location |
|---|-----------|-------|----------|
| M1 | Cherry-Picking | The abstract and conclusion lead with the macro-MAP@10 = 0.533 result and the Δ = 0.061 vs SLIQ. The four ablations whose results undermine the contribution (A1, A3 picking wrong default, A4, A5) are de-emphasized. The paper is not factually inaccurate but the narrative emphasis is one-sided. | Abstract, main.tex:1052–1062 |
| M2 | Overgeneralization | "Domain-specific metric learning substantially outperforms large-scale transfer learning for minority planetary terrain classes" (main.tex:80–83). This is overreach — only one transfer-learning baseline (ResNet50, with unspecified fine-tuning) was tested. | Abstract, conclusion |
| M3 | Confirmation Bias | The barren-plateau "verification" tests only N = 8 and concludes the local-observable strategy "suppresses vanishing gradients." But Cerezo's theorem is a scaling claim that requires varying N to verify. The single-N test cannot distinguish "Cerezo's theorem holds" from "we picked an N where everything works." | main.tex:166–168, 859–884 |
| M4 | "So What?" | If the genuine finding of the paper is that quantum vs classical at this scale is a wash (Δ = 0.005), the operational implication is that one should use the classical MLP head, which is faster, cheaper, and equally accurate. The paper does not engage with this conclusion. | Discussion §6 generally |
| M5 | Alternative Paths Not Considered | The paper does not consider that the gain over SLIQ (Δ = 0.061) might be achievable classically by appropriate data augmentation or by sequential vs interleaved feature concatenation in the MLP head. The QML claim is therefore not properly isolated. | main.tex:737–743 |

##### MINOR

| # | Dimension | Issue | Location |
|---|-----------|-------|----------|
| m1 | Stakeholder Blind Spot | The planetary-science end-user perspective is invoked rhetorically but not engaged (no PDS team consultation noted, no science-team query examples). | §1, §6 |
| m2 | Logic Chain | "Macro-MAP gap versus SLIQ-style encoding (Δ = 0.061) isolates the encoding strategy as the primary driver" (main.tex:78–79). The word "isolates" is too strong; the gap demonstrates the gap, but the discussion later admits the cause is confounded with axis choice. | Abstract |
| m3 | Hidden Assumption | The paper assumes throughout that the HiRISE v3 8-class taxonomy is the right relevance criterion for retrieval. A scientifically motivated query is rarely "find me more class-c examples." | §5.2 |

#### Ignored Alternative Explanations / Paths

1. **The Δ = 0.061 A2 gap is just an axis choice.** A clean test: interleaved $R_Z$/$R_Y$ vs sequential $R_Z$/$R_Y$. If the gap disappears, the paper's central claim is gone. If it survives, the claim strengthens.
2. **The Δ = 0.005 A1 gap is noise.** A 5-seed reanalysis may show overlapping confidence intervals.
3. **The ResNet50 baseline is undertuned.** A properly fine-tuned ResNet50 with triplet loss and grayscale-aware first-layer surgery may match or exceed \qts{}, in which case the "transfer learning doesn't work" finding is wrong.
4. **The N = 4 winner suggests the CNN encoder is the bottleneck.** If a tiny 4-qubit head outperforms a 12-qubit head, the embedding-space discrimination is dominated by what the CNN does in the last layer, not by the quantum mechanics downstream.

#### Missing Stakeholder Perspectives

- **HiRISE / PDS operations team** (would the system actually be deployed?)
- **Quantum hardware engineers** (is this trainable on NISQ devices at the shot budget that the local-observable choice implies?)
- **The IR / image-retrieval community** (is macro-MAP@10 = 0.533 competitive with existing classical deep-retrieval methods?)

#### Unexamined Premise

The paper assumes that *better quantum encoding* is the right axis along which to seek progress on this task. An equally well-motivated paper would ask: given that we have access to overhead grayscale imagery with extreme class imbalance and confusable geological subcategories, **what is the right inductive bias?** The quantum-circuit machinery may simply be the wrong tool for this problem; the gains attributed to it may be attainable more cheaply through better data augmentation, larger encoders, or per-class loss weighting.

#### Observations (Non-Defects)

- The internal honesty in reporting A4 and A5 against the authors' interest is genuinely admirable.
- The paper would benefit from a one-line statement of the negative findings in the abstract — currently the negative results are only visible to readers who reach §6.
- The implementation appendix note (PennyLane `inputs[..., :N]` vs `inputs[:N]`) is genuinely useful and is the kind of detail that should appear in more QML papers.

---

## Phase 2 — Editorial Decision

### Decision: **Major Revision**

*Rationale for decision (300 words).*

All four scoring reviewers (EIC, R1, R2, R3) independently recommend Major Revision; the Devil's Advocate finds three CRITICAL issues, which — per the skill's IRON RULE — preclude an Accept decision. The weighted dimension scores converge in the 59–63 range, which is the upper-Major / lower-Minor boundary; the CRITICAL DA findings tip the decision toward Major rather than Minor.

The decision is **not** Reject, because the paper has genuine merits that survive the panel's scrutiny:

1. The retrieval-vs-classification operational framing is correct and contributes a useful methodological reframing for HiRISE work.
2. The methodology has been substantially upgraded from the prior review (train/val/test discipline now correct, macro-MAP@10 chosen as primary metric, val/test metric alignment fixed).
3. The internal honesty in reporting A4 and A5 against the authors' interest is unusual and admirable.
4. The QML technical content (Cerezo correctly invoked, SU(2) expansion correct, PennyLane primitives correctly used) is competent.

The decision is **not** Minor Revision, because the issues to be addressed include:

- A central claim ("interleaved encoding is the dominant driver") that is acknowledged in the discussion to be confounded with an axis-choice degeneracy in the baseline — this requires an additional ablation (A2.5: interleaved-vs-sequential at matched axes).
- A single-seed evaluation across all 10 ablation cells, where several deltas under interpretation are 0.005–0.023 and within plausible seed variance.
- A headline configuration that the authors' own ablations show to be sub-optimal (default N=8, L=2, class-pair mining is dominated by N=4, L=3, random).
- A barren-plateau "verification" tested at only one qubit count.
- Missing standard image-retrieval baselines and a missing operational scenario.

These are fixable within a Major Revision cycle (6–8 weeks).

### Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| EIC | IEEE TGRS associate editor (deep-learning RS) | Major Revision | 4/5 |
| R1 | Methodology Reviewer | Major Revision | 5/5 |
| R2 | QML Domain Reviewer | Major Revision (borderline Minor) | 5/5 |
| R3 | Planetary Science Perspective Reviewer | Major Revision | 4/5 |
| DA | Devil's Advocate | Three CRITICAL findings → no Accept | n/a |

### Consensus Analysis

**[CONSENSUS-5]** (all five reviewers agree):
1. The default (headline) configuration is not the best ablation configuration; this internal inconsistency must be resolved (EIC W1, R1 W3, DA C3).
2. Single-seed evaluation is insufficient evidence for the small deltas under interpretation (R1 W1, DA C1).
3. The A2 ablation is confounded by the $R_Z|0\rangle = |0\rangle$ degeneracy and needs an additional axis-matched control (R1 W2, R2 W1 implicitly, DA C2).

**[CONSENSUS-3]** (3/4 of the scoring reviewers agree):
1. The ResNet50 baseline is under-specified and may be unfairly weak (EIC W3, R1 W4, DA M2).
2. The barren-plateau verification is overreached from a single-N observation (R1 W5, R2 W2, DA M3).

### Points of Disagreement

**Disagreement 1: Severity of "headline-vs-ablation" mismatch.**
- R1, R2, EIC: Major — fixable with a re-run at the best-ablation config.
- DA: Critical — undermines core thesis if not resolved.
- **Editor's resolution:** Treat as Critical for purposes of the Required Revisions list (R1 must be addressed), but Major in severity terms. Both framings agree the issue must be resolved before publication.

**Disagreement 2: Scope fit for TGRS.**
- EIC: Borderline — the methodology-heavy body may fit better at *npj Quantum Information* or *Quantum Machine Intelligence*.
- R3: Borderline — the operational planetary-science claims need more substance.
- R1, R2: Implicit acceptance of TGRS as venue.
- **Editor's resolution:** TGRS is acceptable if W2 (operational scenario) and W3 (RS-standard baselines) are addressed in revision. If the author prefers, *IEEE Transactions on Quantum Engineering* would also be an appropriate destination.

---

### Required Revisions (Must Fix)

| # | Revision Item | Source | Severity | Section | Estimated Effort |
|---|--------------|--------|----------|---------|-----------------|
| **R1** | Re-run \qts{} at the best-ablation configuration (e.g., N=4, L=3, random mining) and update Table I + abstract. Or, document why the default is kept despite being dominated. | EIC W1, R1 W3, DA C3 | Critical | §5, §6, Abstract | 3–5 days |
| **R2** | Add A2.5 ablation: axis-matched sequential encoding (e.g., sequential $R_X$/$R_Y$ at same axes as \qts{}, plus optionally interleaved $R_Z$/$R_Y$) to disentangle interleaving from axis choice. | R1 W2, R2 W4 implicit, DA C2 | Critical | §5.3 | 2–3 days |
| **R3** | Multi-seed reanalysis (minimum 3 seeds per cell) for all 10 ablation cells. Report mean ± std. Update discussion to reflect which deltas survive seed noise. | R1 W1, DA C1, M1 | Critical | §5, §6, Tables I/II | 5–7 days (compute) |
| **R4** | Fully specify ResNet50 baseline training protocol: backbone state, first-conv re-initialization, optimizer/LR/schedule, augmentation. If under-tuned, re-tune. | EIC W3, R1 W4, DA M2 | Major | §5.3, §5.4 | 2–3 days |
| **R5** | Barren-plateau verification across multiple N (∈ {4, 6, 8, 10, 12}) OR soften the claim to "observed at our N=8 operating point." | R1 W5, R2 W2, DA M3 | Major | §4.3.3, §5.4 | 2–4 days |
| **R6** | Add at least one worked operational retrieval scenario: query patch + top-K retrieved + manual relevance assessment. | EIC W2, R3 W1 | Major | New §6 subsection | 2 days |
| **R7** | Add at least one standard image-retrieval baseline (e.g., deep image hashing, or DELF-style, or learned cosine similarity without quantum head and without triplet pretraining). | EIC W3 | Major | §5.3 | 2–3 days |
| **R8** | State the AP@K formula explicitly in §5.2 with a textbook citation. | R1 W6 | Minor | §5.2 | 0.5 day |
| **R9** | Report wall-clock training time per epoch for \qts{} vs the Classical MLP baseline on identical hardware. | R1 W7 | Minor | §6 | 0.5 day |
| **R10** | Add a Code Availability statement and confirm planned public release. | R1 (questions) | Minor | New §7 / acknowledgments | 0.5 day |

### Suggested Revisions (Should Fix)

| # | Revision Item | Source | Priority | Section |
|---|--------------|--------|----------|---------|
| S1 | Add entanglement-quantification measurement (von Neumann entropy or Schmidt rank) comparing \qts{} vs SLIQ-style states. | R2 W1 | P2 | New §5.4 subsection |
| S2 | Acknowledge that $64\times64$ patches lose most of HiRISE's actual information; discuss patch-size sensitivity. | R3 W2 | P2 | §6 |
| S3 | Compare against an alternative ansatz (e.g., BasicEntanglerLayers) at matched parameter count. | R2 W3 | P2 | §5 or §6 |
| S4 | Tighten the conclusion (currently restates the abstract). | EIC | P3 | §7 |
| S5 | Add NISQ-shot-budget estimate. | R2 W4 | P3 | §6 |
| S6 | Soften "first quantum similarity learning system for planetary terrain" language. | EIC W4 | P3 | Abstract, §1 |
| S7 | Acknowledge that the 8-class taxonomy as relevance criterion is a benchmark convenience rather than an operational query model. | R3 W4 | P3 | §5.2 or §6 |

### Revision Roadmap

**Priority 1 — Structural revisions** *(estimated 14–22 days)*
- [ ] R1: Re-run / re-justify headline configuration
- [ ] R2: A2.5 axis-matched ablation
- [ ] R3: Multi-seed reanalysis
- [ ] R4: ResNet50 baseline re-specification (and re-tuning if needed)
- [ ] R5: Multi-N barren-plateau plot or claim softening
- [ ] R6: Operational retrieval scenario figure
- [ ] R7: Standard image-retrieval baseline

**Priority 2 — Content supplementation** *(estimated 3–5 days)*
- [ ] S1: Entanglement quantification
- [ ] S2: Patch-size discussion
- [ ] S3: Alternative ansatz comparison

**Priority 3 — Text and formatting** *(estimated 1–2 days)*
- [ ] R8: AP@K formula statement
- [ ] R9: Wall-clock comparison
- [ ] R10: Code availability statement
- [ ] S4: Conclusion tightening
- [ ] S5: NISQ shot-budget estimate
- [ ] S6: Originality-claim language softening
- [ ] S7: Taxonomy-as-relevance caveat

**Total estimated effort:** 4–6 weeks (within standard Major Revision window of 6–8 weeks).

### Revision Deadline

- **Recommended deadline:** 2026-08-01 (8 weeks from review date)
- **Re-review:** Yes — Major Revision requires re-review (`/ars-reviewer re-review` mode after revision)
- **Maximum rounds:** Per editorial policy, 2 rounds of Major Revision then accept-or-reject decision

### Response Letter Instructions

Please use the R→A→C format (Reviewer comment → Author response → Change description with new page/line numbers) for every Required and Suggested item. Track changes in the revised manuscript. Where the revision is "we chose not to address this," provide an explicit justification.

### Closing

We encourage the author to carefully consider the reviewers' comments and submit a substantially revised manuscript. The paper presents an interesting application of quantum metric learning to planetary terrain retrieval, with several strong methodological choices (correct retrieval framing, proper macro-MAP@10 reporting, honest ablation reporting). The required revisions address a coherent set of issues — the relationship between the headline numbers and the ablation table, the identification of which specific design decisions actually matter, and the proper scoping of the claims against the evidence. We look forward to receiving the revision and assessing whether the issues have been resolved.

---

## Appendix: Score Summary

| Reviewer | Originality | Methodology | Evidence | Coherence | Writing | Weighted |
|----------|-------------|-------------|----------|-----------|---------|----------|
| EIC | 60 | 55 | 58 | 65 | 78 | **61.0** |
| R1 (Methodology) | 60 | 50 | 55 | 65 | 80 | **59.5** |
| R2 (QML Domain) | 62 | 60 | 58 | 70 | 80 | **63.5** |
| R3 (Perspective) | 68 | 60 | 55 | 65 | 78 | **62.7** |
| **Panel mean** | **62.5** | **56.3** | **56.5** | **66.3** | **79.0** | **61.7** |

**Decision mapping:** Weighted average 61.7 → Major Revision band (50–64). DA CRITICAL findings present → Accept precluded; Major Revision confirmed.

---

*End of review report.*
