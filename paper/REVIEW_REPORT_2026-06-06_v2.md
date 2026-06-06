# Academic Peer Review Report (v2) — `paper/main.tex`

**Paper title:** Quantum Similarity Learning for Mars Terrain Retrieval via Entangled-Pair Encoding
**Submitting venue (declared):** IEEE Transactions on Geoscience and Remote Sensing
**Review mode:** `full` (EIC + R1 Methodology + R2 Domain (QML) + R3 Perspective (Planetary Science) + Devil's Advocate + Editorial Synthesizer)
**Review date:** 2026-06-06 (second pass, same day)
**Reviewer team:** simulated panel (academic-paper-reviewer v1.9.1)
**Working tree state:** HEAD = `5acb4d1` with substantial uncommitted edits to `paper/main.tex` (~464 deletions / 254 insertions vs HEAD). This review evaluates the **working-tree draft**, not the committed version.
**Relation to prior review (`REVIEW_REPORT_2026-06-06.md`):** Re-evaluation of the *same-day-edited* draft. The prior report scored the committed snapshot. The draft has since been edited to address several Required Revisions narratively (R8 AP@K formula, R4 ResNet50 specification, R5 BP claim softening, R10 code availability, S6 originality softening, plus new "Variance and replication" and "Limitations" paragraphs). No new ablation cells have been run.

---

## Phase 0 — Field Analysis & Reviewer Configuration

| Dimension | Determination |
|-----------|---------------|
| Primary discipline | Quantum machine learning (PQC-based metric learning) |
| Secondary discipline | Planetary remote sensing (HiRISE imagery, Mars terrain) |
| Research paradigm | Empirical computational study; supervised retrieval; ablation-driven |
| Methodology type | Hybrid classical–quantum (PennyLane simulator); Siamese CNN + PQC head; triplet loss |
| Target tier | IEEE Transactions (TGRS — Q1 remote-sensing methodology) |
| Paper maturity | Working-tree draft. Several Major findings from the prior review are now *disclosed* in the manuscript text but not *empirically resolved*. |
| Length | ~12–14 pages double-column (IEEEtran journal) |
| Author count | Single author block (anonymous) |

### Reviewer panel (carried forward from prior pass; identities unchanged)

| # | Role | Configured identity |
|---|------|--------------------|
| **EIC** | Editor-in-Chief | Associate editor at IEEE TGRS; deep-learning-for-remote-sensing background |
| **R1** | Methodology Reviewer | ML methodologist with retrieval-evaluation, ablation-design, statistical-significance expertise |
| **R2** | Domain Reviewer (QML) | QML theorist; PQC expressivity, barren plateaus, SLIQ, PennyLane primitives |
| **R3** | Perspective Reviewer | Planetary scientist / HiRISE end-user; PDS catalog workflow |
| **DA** | Devil's Advocate | Adversarial reader: contribution-vs-evidence coherence, cherry-picking, statistical strength |

---

## Phase 1 — Five Independent Reviews

> **IRON RULE applied:** reviewers operate from non-overlapping perspectives and do not cross-reference each other's reports.

---

### Review 1 — Editor-in-Chief

**Recommendation:** Major Revision (borderline Minor)
**Confidence:** 4/5

**Summary assessment (220 words).** The current draft has visibly improved relative to the committed version reviewed earlier today. The author has added an explicit AP@K formula (eq. 5), a "Variance and replication" paragraph that openly acknowledges single-seed limitations and labels A1/A4 deltas as directional, a fully specified ResNet50 baseline protocol, a softened barren-plateau claim ("consistent with the prediction" rather than "we verify"), an explicit limitations subsection, a "Data Availability" section, and a Discussion paragraph that openly discloses the $R_Z|0\rangle$ degeneracy in the A2 baseline. The abstract now reports both $L=2$ (default, 0.533) and $L=3$ (0.556), and the conclusion identifies $N=4, L=3$ random-mining as the immediate next experiment. **This is a substantively more honest manuscript.** However, every one of these improvements is *disclosure-only*: the underlying empirical work (multi-seed runs, A2.5 axis-matched control, combined-best ablation, operational retrieval scenario, standard image-retrieval baseline, wall-clock comparison, multi-$N$ BP plot) has not been performed. The paper is now an honest report of a partially confounded result rather than an evidence-complete contribution. For TGRS, that is one step short of acceptable: the central claim ("encoding strategy is the dominant factor") is still load-bearing and is now explicitly admitted to be confounded with axis choice. Major Revision remains the right call, but the gap to Minor is now narrower.

**Strengths.**

- **S1: Honest "Variance and replication" paragraph (main.tex:612–620).** Explicitly labels A1 ($\Delta=0.005$) and A4 ($\Delta=0.005$) as within typical seed-to-seed variance and as *directional, not definitive*. This is exactly the language a methodologist would write.
- **S2: A2 confound now disclosed in Discussion (main.tex:793–802).** "Attributable jointly to interleaved ordering and axis choice" — and the A2.5 follow-up is named explicitly. This is the same finding the prior reviewer panel flagged as Critical; calling it out in the main text rather than hiding it in a footnote is the right move.
- **S3: ResNet50 baseline now reproducible (main.tex:600–610).** Backbone state (ImageNet-pretrained), input handling (grayscale→3-channel replication, no first-layer reinit), optimizer ($\alpha_{\text{backbone}}=10^{-4}$, $\alpha_{\text{head}}=10^{-3}$), head architecture (2048→8 linear + $\ell_2$), epochs/batch/split all stated.
- **S4: BP claim softened correctly (main.tex:150–156, 779–783).** "We observe empirically that quantum layer gradients remain above the vanishing threshold throughout training at $N=8$, consistent with the cost-function-dependent plateau prediction." The "consistent with" phrasing is honest; the introduction of a *qualitative-distinction* paragraph (main.tex:386–391) correctly anchors the local-vs-global contrast as scaling, not as point-verification.
- **S5: Limitations subsection (main.tex:838–847).** Single-run estimates, A2 confound, missing standard retrieval baselines, full-resolution operational gap — all named.

**Weaknesses (EIC-level).**

- **W1: Disclosure ≠ resolution.** *Severity: Major.* Every one of the Required Revisions from the prior pass is now acknowledged in the text, but the underlying empirical work has not been done. A reader at TGRS will see "this is a confound" and "we recommend multi-seed replication before drawing strong conclusions" and reasonably ask: why is this paper being submitted before that work is done? *Required fix:* perform at least the (N=4, L=3, random) combined-best run, the A2.5 axis-matched control, and 3-seed replication for A1 and A4 (the cells where small deltas drive the discussion). Without these, the contribution claim is honest-but-thin.
- **W2: Headline number is still the dominated default.** *Severity: Major.* The abstract leads with macro-MAP@10 = 0.533 for the default ($N=8, L=2$), then parenthetically mentions $L=3$ reaches 0.556. Table I reports the default. A reader is left to assemble the "best" picture from Table II. *Required fix:* either re-cast Table I to include the $L=3$ row and lead the abstract with 0.556, or commit to a single defended best configuration after running (N=4, L=3, random).
- **W3: Operational retrieval scenario still missing.** *Severity: Major.* §1 motivates retrieval with three operational use cases (scientific queries, rover path planning, change detection), but no query-and-top-K example is shown in §6. For a TGRS submission whose framing is operational, one worked retrieval example would carry significant weight.
- **W4: Standard image-retrieval baselines still acknowledged-only.** *Severity: Major.* §5.3 (main.tex:577–580) and Limitations (main.tex:843–845) both name deep supervised hashing and local-feature retrieval as deferred. A TGRS reader will want at least one such baseline to justify the QML framing.

**Detailed comments.**

*Title & Abstract.* Title remains accurate. Abstract is now considerably more honest — it concedes the $L=3$ improvement and uses "encoding strategy accounts for the largest performance gap" (correctly limited to a *gap* claim, not a *driver* claim). The "first quantum metric learning system for planetary terrain" phrasing is still in §1 Contributions but has been narrowed.

*Introduction.* Strong. The retrieval-vs-classification reframing remains the paper's best contribution.

*Related Work.* Adequate but compact for TGRS standards. SLIQ's discussion could mention what the SLIQ paper itself reports as its result, for context.

*Method.* Now includes a qualitative-distinction paragraph (main.tex:386–391) that correctly frames Cerezo's theorem as a *scaling* claim. This is the right level of QML literacy for a TGRS audience.

*Experiments.* Variance paragraph (main.tex:612–620) is the methodologically most important addition since the prior pass. The "Implementation Details" section's PennyLane TorchLayer note (main.tex:591–596) remains valuable.

*Results.* Table I and Table II are honest. The text at main.tex:646–651 explicitly characterizes the A2 gap as "the dominant factor" while admitting "marginal advantage over … classical MLP" — this is internally honest, but tightens the W1 problem.

*Discussion.* Genuinely improved: the A2 confound is openly disclosed, A4 is admitted as within-noise, A3/A5 are interpreted carefully, and the Limitations subsection enumerates the deferred work.

*Conclusion.* Now identifies the combined-best (N=4, L=3, random) configuration as the immediate next experiment. This is the right call but it should be done *before* submission, not *named in* the submission.

**Questions for the author.**

1. Why submit before running the (N=4, L=3, random) cell and the A2.5 axis-matched control? Both are inexpensive — 2–3 runs each on CPU. Their absence shapes the entire review.
2. The abstract reports $L=3$ reaches 0.556 but Table I and the conclusion's headline number is 0.533. Is the intent that the default $L=2$ should remain the "default" for narrative simplicity? If so, this should be stated explicitly.
3. Is there a reason the operational retrieval scenario (query patch + top-K + manual assessment) cannot be produced from existing eval outputs? The retrieval engine clearly exists.

**Dimension scores (universal):**

| Dimension | Score | Descriptor | Notes |
|-----------|-------|------------|-------|
| Originality | 60 | Adequate | Unchanged — domain-novel, methodologically narrow |
| Methodological Rigor | 62 | Adequate | Up from 55 — variance paragraph + ResNet50 spec + softened BP claim |
| Evidence Sufficiency | 60 | Adequate | Up from 58 — better disclosure, no new evidence added |
| Argument Coherence | 70 | Strong | Up from 65 — A2 confound now acknowledged in main text |
| Writing Quality | 80 | Strong | Up from 78 — Limitations subsection, qualitative-distinction paragraph |
| **Weighted average** | **64.5** | **Major Revision (borderline Minor)** | |

---

### Review 2 — Methodology Reviewer

**Recommendation:** Major Revision (with substantially reduced scope)
**Confidence:** 5/5

**Summary assessment (240 words).** The methodology has been further upgraded since the morning's pass. The author has added the per-query AP@K formula explicitly (eq. 5, main.tex:540–545), which closes a Minor issue. The "Variance and replication" paragraph (main.tex:612–620) explicitly identifies the A1 and A4 deltas as within typical seed variance and recommends 3-seed replication — the *correct* methodological position, but a position rather than evidence. The ResNet50 baseline is now fully specified at main.tex:600–610: ImageNet-pretrained backbone, grayscale-to-RGB replication with no first-layer reinitialization, end-to-end triplet fine-tuning, separate learning rates ($10^{-4}$ backbone, $10^{-3}$ head), 50 epochs matched to \qts{}. The BP claim has been softened from "we verify empirically" to "we observe empirically … consistent with the prediction" and the qualitative-distinction paragraph (main.tex:386–391) reframes Cerezo as a scaling claim. The Discussion's A2 paragraph (main.tex:793–802) openly attributes the $\Delta=0.061$ gap "jointly to interleaved ordering and axis choice" and names A2.5 as the immediate priority. The Limitations subsection (main.tex:838–847) lists single-seed estimates, the A2 confound, missing retrieval baselines, and the full-resolution operational gap. **What is missing is the empirical work itself.** A methodology reviewer must distinguish "this paper now correctly *describes* its evidence" from "this paper now *has* the necessary evidence." On the former, the manuscript is materially better; on the latter, no new cells have been run. Major Revision.

**Strengths.**

- **S1: AP@K formula explicit (eq. 5).** $\mathrm{AP}@K(q) = \frac{1}{\min(K, r_q)} \sum_{k=1}^{K} P(k)\cdot\mathbf{1}[\text{rank-}k\text{ result is relevant}]$ — the standard $\min(K, r_q)$ normalization is stated. This was a Minor item in the prior pass and is now resolved.
- **S2: Variance acknowledgment is methodologically correct.** The paragraph at main.tex:612–620 explicitly names $|\Delta|\lesssim 0.01$ deltas as within seed variance and labels them as directional. This is exactly the language ML methodology reviewers want. It does not substitute for multi-seed runs, but it eliminates the prior-pass concern that the manuscript was over-interpreting noise.
- **S3: ResNet50 baseline now fully reproducible (main.tex:600–610).** Sufficient detail for an independent re-run. The "no first-layer reinitialization" disclosure is the key piece: it means the ResNet50 baseline is being given a slightly unfavorable input pipeline (grayscale replicated rather than first-conv adapted), which a fair reviewer can either accept as a documented choice or flag for revision.
- **S4: BP claim correctly softened (main.tex:150–156, 779–783).** No more "we verify"; now "consistent with the prediction." This pairs with the qualitative-distinction paragraph (main.tex:386–391) which correctly frames Cerezo's result as polynomial-vs-exponential scaling rather than a point claim.
- **S5: Limitations subsection (main.tex:838–847).** Single-run estimates, A2 confound, absent standard retrieval baselines, full-resolution operational gap — all named.

**Weaknesses.**

- **W1: Multi-seed replication still not run.** *Severity: Critical → Major.*
  **Problem:** The paper now acknowledges that $|\Delta|\lesssim 0.01$ deltas in A1 and A4 are within seed variance, but no multi-seed evidence has been added. The downgrade from Critical to Major reflects the fact that the *interpretation* is no longer over-claimed; the *evidence* remains thin.
  **Why it matters:** A reader who accepts the disclaimer ("treat as directional") can no longer use these cells to update their belief about quantum-vs-classical or random-vs-class-pair mining. The paper's empirical contribution is therefore *narrower* than the abstract suggests: the only delta that survives is A2 ($\Delta=0.061$), and that is itself confounded (see W2). Effectively the paper has 0–1 statistically defensible findings.
  **Suggestion:** Run 3 seeds for A1 (Classical vs Quantum) and A4 (random vs class-pair) at minimum. These are the cells where the discussion makes specific directional claims. ~6 additional runs.
  **Severity:** Major (downgraded from Critical because the manuscript now correctly characterizes the evidence).

- **W2: A2.5 axis-matched ablation still not run.** *Severity: Major.*
  **Problem:** The Discussion (main.tex:793–802) names the A2 confound explicitly: $\RZ|0\rangle = |0\rangle$ up to global phase, so SLIQ's first $\RZ$ block is a no-op for local-$\PauliZ$ measurements. This makes the $\Delta=0.061$ between \qts{} and SLIQ-style partially an axis artifact rather than an interleaved-vs-sequential ordering effect. A2.5 (sequential $\RX$/$\RY$ at matched axes) would disentangle the two. The manuscript names this as an "immediate priority" but does not run it.
  **Why it matters:** The A2 gap is now the *only* delta the paper can interpret as substantive. With A2.5 unrun, the headline encoding claim is a confounded measurement.
  **Suggestion:** Run A2.5: same encoder, same ansatz, same measurements, but sequential $\RX(\pi e_a^{(i)})$ on all wires followed by $\RY(\pi e_b^{(i)})$ on all wires (no interleaving). One additional training run.
  **Severity:** Major.

- **W3: Combined-best configuration (N=4, L=3, random mining) still not evaluated.** *Severity: Major.*
  **Problem:** The conclusion (main.tex:867–870) names this as the immediate next experiment. Table II shows that $N=4$, $L=3$, and random mining each individually beat the default. The combined run is one training session. Its absence means the headline number (0.533, default) is still dominated by the abstract's own parenthetical (0.556 at $L=3$) and Table II's $N=4$ (0.540).
  **Why it matters:** If the combined-best score lands at, e.g., 0.56–0.58, the paper materially strengthens. If it lands at the default's 0.533 or below — also informative; the marginal effects don't compose. Either way, the experiment is cheap.
  **Severity:** Major.

- **W4: Standard image-retrieval baselines deferred rather than added.** *Severity: Major.*
  **Problem:** §5.3 (main.tex:577–580) and Limitations (main.tex:843–845) acknowledge that deep supervised hashing, local-feature retrieval, and other standard image-retrieval baselines are absent. A reasonable defense exists ("this is a quantum-vs-classical ablation paper, not a retrieval-benchmark paper"), but TGRS readers will still want at least one such baseline.
  **Suggestion:** Add one — DSH (Deep Supervised Hashing) or a pretrained-CNN + cosine-similarity baseline (no triplet training). Either is implementable in 1–2 days.
  **Severity:** Major.

- **W5: Wall-clock comparison still absent.** *Severity: Minor.*
  **Problem:** No epoch-time numbers for \qts{} vs Classical MLP on the same hardware. This is a one-line addition.
  **Suggestion:** "Quantum head: X seconds/epoch on CPU; Classical MLP head: Y seconds/epoch on same hardware."
  **Severity:** Minor.

- **W6: Multi-$N$ BP plot still absent.** *Severity: Minor (downgraded from Major).*
  **Problem:** The single-$N$ gradient norm plot is now correctly framed as "consistent with" rather than "verifying" Cerezo's scaling prediction. The downgrade reflects that the claim is now appropriately scoped. A multi-$N$ plot would still strengthen the paper but is no longer load-bearing.
  **Severity:** Minor.

**Detailed comments.**

*Methodology / Research Design.* Substantively better in framing; the empirical gaps remain.

*Sampling strategy.* Stratified per-class split is correct. Test split (15% of 73,031 = ~10,955 patches) is large enough for credible aggregate MAP@10 estimates; per-class AP@K for the rare classes (spider 71 test patches, impact_ejecta 35 test patches) remains high-variance, which is why the macro-MAP@10 deltas should be treated as directional.

*Statistical reporting.* Now correctly framed but evidentially thin.

*Reproducibility.* Code Availability statement now exists (main.tex:872–880). Acknowledgement of AI assistance (Claude) is in compliance with IEEE policy and is admirable.

**Questions for the author.**

1. Multi-seed runs for A1 (Classical vs Quantum) and A4 (random vs class-pair) would each take roughly the same wall time as a single \qts{} training run, since the encoder dominates the cost. What is the blocker to running these before resubmission?
2. The A2.5 axis-matched ablation is one additional training run. Can it be added to the revision?
3. What is the wall-clock time per epoch for the \qts{} default vs the Classical MLP baseline on the same hardware?
4. The ResNet50 baseline uses "no first-layer reinitialization" with grayscale replicated to 3 channels. Did you test the alternative (first-conv reinitialized to single-channel)? If so, did it improve the macro-MAP@10?

**Dimension scores:**

| Dimension | Score | Descriptor |
|-----------|-------|------------|
| Originality | 60 | Adequate |
| Methodological Rigor | 60 | Adequate (up from 50 — correct framing of variance, BP scaling, A2 confound) |
| Evidence Sufficiency | 55 | Weak (unchanged — no new cells run) |
| Argument Coherence | 72 | Strong (up from 65 — limitations + A2 confound disclosed) |
| Writing Quality | 82 | Strong (up from 80 — qualitative-distinction paragraph) |
| **Weighted average** | **63.5** | **Major Revision (borderline Minor)** |

---

### Review 3 — Domain Reviewer (QML)

**Recommendation:** Minor Revision (borderline Major)
**Confidence:** 5/5

**Summary assessment (220 words).** From a QML perspective the manuscript has improved on the two issues that mattered most. (1) The barren-plateau framing has been corrected: the introduction now says "consistent with the cost-function-dependent plateau prediction for shallow circuits" (main.tex:154–156), and the Method section adds an explicit qualitative-distinction paragraph (main.tex:386–391) characterizing Cerezo's theorem correctly as polynomial-vs-exponential gradient scaling. The gradient-norm figure is now described as "consistent with the … prediction … scaling verification across multiple $N$ is deferred to future work" — the right epistemic level. (2) The A2 confound is openly disclosed in the Discussion (main.tex:793–802): "$\RZ|0\rangle$ differs from $|0\rangle$ only by a global phase, so SLIQ's first $\RZ$ block is a no-op for local-$\PauliZ$ measurements." That is exactly the QML-mechanical reality and the paper now states it. **The remaining QML-substantive issue is that the entanglement claim is still mechanistic rather than measured.** The paper argues from $[\RX, \RY] \neq 0$ that the per-qubit state is a nonlinear joint function (which is correct), but does not compute any entanglement quantifier (entropy, Schmidt rank, mutual information) on the trained circuits to *demonstrate* the claim. With the BP and A2 framings now corrected, that becomes the principal residual QML gap.

**Strengths.**

- **S1: Cerezo theorem now correctly framed as scaling (main.tex:386–391).** The qualitative-distinction paragraph — "polynomial suppression allows gradients to remain informative as the system scales, while exponential suppression renders gradient-based optimization infeasible" — is exactly the right level of formal claim for a TGRS audience.
- **S2: BP claim correctly softened.** "We observe empirically that quantum layer gradients remain above the vanishing threshold throughout training at $N=8$, consistent with the cost-function-dependent plateau prediction" (main.tex:150–156). The "consistent with" phrasing eliminates the overreach the prior reviewer panel flagged.
- **S3: A2 confound openly disclosed (main.tex:793–802).** The $\RZ|0\rangle = |0\rangle$ degeneracy is named explicitly in the main Discussion, not buried in a footnote. This is unusual honesty in this literature and earns trust.
- **S4: SU(2) mechanism is correctly presented.** Equation 6 (main.tex:344–349) and the non-commutativity argument (main.tex:350–354) are technically sound and at the right level of abstraction.
- **S5: PennyLane primitives correctly used.** `default.qubit` + `diff_method="backprop"` + `StronglyEntanglingLayers` (3NL params) + the `inputs[..., :N]` TorchLayer batching detail (main.tex:585–596) are all correct.

**Weaknesses.**

- **W1: Entanglement claim is still unmeasured.** *Severity: Major.*
  **Problem:** The mechanistic argument ($[\RX, \RY] \neq 0 \Rightarrow$ nonlinear joint state) is correct but does not quantify *how much* entanglement the trained circuit produces, nor whether the SLIQ-style circuit produces less. Standard QML quantifications — half-chain von Neumann entanglement entropy averaged over a batch, Schmidt rank distribution, pairwise mutual information of $\langle Z_i\rangle$ measurements — are absent.
  **Why it matters:** The paper's interpretive narrative says the encoding strategy creates "richer entanglement" (implicit in the "nonlinear joint function" framing). With A2 now disclosed as confounded with axis choice, the *only* way to separate "interleaved ordering creates more entanglement" from "axis choice changes which measurements activate" is to measure entanglement directly.
  **Suggestion:** For one trained \qts{} checkpoint and one SLIQ-style checkpoint, compute the half-chain von Neumann entanglement entropy averaged over a held-out batch of test pairs. Single supplementary figure. Cheap and decisive.
  **Severity:** Major.

- **W2: Single-N BP test is no longer overreached, but a multi-N plot would still strengthen.** *Severity: Minor (downgraded from Major).*
  **Problem:** With the claim now softened to "consistent with" and scaling verification explicitly deferred, the single-N observation is no longer load-bearing. A multi-N plot would convert "consistent with" to "directly observed" but is not required for the revised claim.
  **Severity:** Minor.

- **W3: No comparison to alternative ansatze.** *Severity: Minor.*
  **Problem:** `StronglyEntanglingLayers` is one of several PennyLane ansatze. With $L=3$ outperforming $L=2$, a natural question is whether `BasicEntanglerLayers` (fewer params per layer) at matched parameter count would do better. No comparison provided.
  **Suggestion:** Acknowledge as a limitation, or run one comparison.
  **Severity:** Minor.

- **W4: $\pi$ rotation scaling motivated implicitly only.** *Severity: Minor.*
  **Problem:** Rotation angles are $\pi e_i \in [-\pi, \pi]$ given $e_i \in [-1, 1]$. The "full-circle" coverage is a natural choice but should be motivated explicitly in §4.3.1 — e.g., "to avoid periodicity collapse for features near the unit-sphere boundary."
  **Severity:** Minor.

- **W5: NISQ-readiness discussion still abstract.** *Severity: Minor.*
  **Problem:** The Implementation Details section notes `diff_method="backprop"` with "no shot noise" (main.tex:586–588). A back-of-envelope shot-budget estimate for hardware execution would calibrate reader expectations and connect to the SLIQ paper's NISQ framing.
  **Severity:** Minor.

**Detailed comments.**

*Background.* §3 PQC formalism is clean. Hilbert space dimension correctly stated ($2^N$).

*Method §4.3.1.* The SU(2) reasoning is technically correct. With the A2 confound now disclosed in §7, the mechanistic argument is appropriately scoped.

*Method §4.3.2.* StronglyEntanglingLayers correctly described ("brick-layer pattern of CZ between neighboring and periodically-shifted qubit pairs"). $3NL=48$ parameter count correct.

*Method §4.3.3.* The qualitative-distinction paragraph is the most substantive QML addition since the prior pass and substantially raises the manuscript's QML rigor.

*Figures.* Figure 2 (architecture) and Figure 3 (circuit, quantikz) are clean.

*Appendix.* The TorchLayer batching note (main.tex:591–596) is genuinely useful — exactly the kind of reproducibility detail that should appear in more QML papers.

**Questions for the author.**

1. Can you report the half-chain von Neumann entanglement entropy (or Schmidt rank distribution) averaged over a batch of trained \qts{} states vs trained SLIQ-style states? This is the cleanest way to test the "richer entanglement" interpretation.
2. Did you experiment with `BasicEntanglerLayers` at matched parameter count? If so, what did you find?
3. For hardware deployment at the operating point ($N=8, L=2$), what shot budget does the local-observable measurement scheme imply for gradient precision $\epsilon \sim 10^{-3}$?

**Dimension scores:**

| Dimension | Score | Descriptor |
|-----------|-------|------------|
| Originality | 62 | Adequate |
| Methodological Rigor | 68 | Adequate-strong (up from 60 — BP framing corrected, qualitative-distinction paragraph) |
| Evidence Sufficiency | 60 | Adequate (up from 58 — A2 confound disclosed; entanglement still unmeasured) |
| Argument Coherence | 76 | Strong (up from 70 — internal consistency between Method and Discussion much improved) |
| Writing Quality | 82 | Strong |
| Literature Integration | 78 | Strong (Cerezo, SLIQ, McClean, Schuld-Killoran, PennyLane all correctly framed) |
| **Weighted average** | **68.5** | **Minor Revision (borderline Major)** |

---

### Review 4 — Perspective Reviewer (Planetary Science)

**Recommendation:** Major Revision
**Confidence:** 4/5

**Summary assessment (200 words).** The planetary-science framing is structurally unchanged: the retrieval-over-classification reframing remains the paper's clearest contribution from an operational perspective, and the discussion of ImageNet-to-Mars transfer limitations (main.tex:804–812) is well-reasoned and matches the cross-domain transfer literature. But none of the prior-pass operational concerns have been empirically addressed: no worked operational scenario, no full-resolution patch exploration, no PDS team consultation noted, no multi-instrument or multi-temporal extension. The Limitations subsection (main.tex:838–847) does name the operational gap ("Operational HiRISE retrieval would require full-resolution tiled queries, and end-to-end PDS workflow integration is deferred to future work") — which is honest, but for a TGRS submission that places "operational deployment" in its motivation, deferral-as-limitation is the weaker option. A planetary-science end-user reading the paper will still come away thinking *this is a methodology paper with planetary-science framing*, not *this is a planetary-science retrieval system*. Major Revision to add at least one operational worked example. The bar here is low — a single qualitative figure with a query patch and top-K retrieved patches assessed by the author would suffice.

**Strengths.**

- **S1: Retrieval framing remains operationally correct.** The §1 motivation matches how HiRISE science teams actually work.
- **S2: ResNet50 transfer-failure discussion is well-grounded.** The four reasons given (channel, viewpoint, structure, imbalance) are exactly the cross-domain transfer literature's findings.
- **S3: Confusable-pair identification is geologically credible.** Bright/dark dune and spider/swiss-cheese are real operational confusion modes (main.tex:455–458).
- **S4: Honest operational limitations.** The Limitations subsection acknowledges full-resolution and PDS integration as deferred. Not ideal, but honest.

**Weaknesses.**

- **W1: No operational scenario worked end-to-end.** *Severity: Major.*
  **Problem:** The retrieval engine clearly exists (Figure 4 confusion matrices are produced from real eval outputs). A query-and-top-K visualization for one specific science question (e.g., aeolian dune migration site) is a single additional figure.
  **Why it matters:** TGRS readers will close the paper without seeing what the system actually returns for a query.
  **Suggestion:** Add Figure 6: query patch → top-10 retrieved patches → 1–2 sentences of expert assessment per result.
  **Severity:** Major.

- **W2: $64\times64$ grayscale patches still unjustified for operational use.** *Severity: Major.*
  **Problem:** The Limitations subsection mentions full-resolution as deferred work, but the paper does not engage with *why* $64\times64$ is operationally meaningful. The wagstaff2021mars dataset uses this format for classification convenience; a retrieval paper inherits the choice without testing whether it is operationally adequate.
  **Suggestion:** Add a paragraph (~150 words) in Discussion explaining the patch-size choice, its operational implications, and what additional information would be available at $128\times128$ or full-resolution.
  **Severity:** Major.

- **W3: No multi-instrument extension demonstrated.** *Severity: Minor.*
  **Problem:** Cross-instrument retrieval (HiRISE → CTX, CaSSIS) is named in the conclusion (main.tex:868–870) as a long-range direction. Even a small toy experiment with a different sensor would substantially strengthen the planetary-science framing.
  **Severity:** Minor.

- **W4: Class taxonomy as relevance criterion still unexamined.** *Severity: Minor.*
  **Problem:** Macro-MAP@10 over the 8-class HiRISE v3 taxonomy is the relevance criterion. Real operational queries ("find regions like this dune migration site") rarely match the taxonomy cleanly. The Limitations subsection does not address this.
  **Suggestion:** One paragraph in §6 noting that taxonomic relevance is a benchmark proxy for operational relevance.
  **Severity:** Minor.

**Questions for the author.**

1. Can you show one operational query scenario worked end-to-end? The retrieval engine exists; this is a figure-generation task, not a re-training task.
2. Did you discuss this work with HiRISE / PDS team members? Their input on whether macro-MAP@10 ≈ 0.53–0.56 is operationally adequate would substantially strengthen the paper.
3. Is the $64\times64$ patch size a benchmark-convenience choice, or is there an operational reason it would generalize to full-resolution queries?

**Dimension scores:**

| Dimension | Score | Descriptor |
|-----------|-------|------------|
| Originality | 68 | Adequate (operational framing remains the real contribution) |
| Methodological Rigor | 62 | Adequate (proper splits; ResNet50 now specified) |
| Evidence Sufficiency | 55 | Weak (no operational scenario; full-resolution unexplored) |
| Argument Coherence | 68 | Adequate (Limitations subsection helps) |
| Writing Quality | 78 | Strong |
| Significance & Impact (R3 focus) | 60 | Adequate (retrieval framing is significant; deployment path missing) |
| **Weighted average** | **64.5** | **Major Revision (low end)** |

---

### Review 5 — Devil's Advocate

> *Per skill protocol: this report does not score. It challenges.*
> *Affirmation (1 sentence, for fairness): the manuscript is now substantially more honest than the version reviewed earlier today — the "Variance and replication" paragraph, the A2 confound disclosure, and the BP claim softening are all unforced improvements that strengthen the paper's epistemic credibility.*

#### Strongest Counter-Argument (290 words)

**The paper has converted its empirical gaps into limitations, but the central claim still rests on a single confounded delta.**

After the morning's revisions, the manuscript's evidence consists of exactly two findings: (1) a $\Delta=0.061$ macro-MAP@10 gap between interleaved and SLIQ-style encoding, and (2) a $\Delta=0.155$ macro-MAP@10 gap between \qts{} and ResNet50. Every other delta in the ablation grid is now explicitly labeled as within seed-to-seed variance or as confounded.

But the $\Delta=0.061$ gap (A2) is itself confounded — the paper now openly says so. The Discussion at main.tex:793–802 attributes the gap "jointly to interleaved ordering and axis choice" because $\RZ|0\rangle = |0\rangle$ makes SLIQ's first encoding block a no-op. **The single delta the paper can defend as substantive is partially an axis artifact.** A2.5 (sequential $\RX$/$\RY$) would resolve this in one training run; the run has not been performed.

The $\Delta=0.155$ vs ResNet50 (B) is the paper's strongest empirical finding, but it is a finding about classical transfer learning failure, not about quantum metric learning. A reader who accepts the paper's own variance disclaimers concludes: "domain-specific metric learning beats ImageNet transfer on minority Mars classes" — true, and worth publishing, but the QML head is not load-bearing in that argument.

**The honest one-sentence summary of the evidence is:** a quantum similarity head produces equivalent retrieval performance to a classical MLP head (A1: $\Delta=0.005$), a small advantage over a known-degenerate sequential-encoding baseline (A2: $\Delta=0.061$, confounded with axis), and is substantially better than naive ImageNet transfer (B: $\Delta=0.155$). This is a publishable finding but it is not the paper that has been written. The abstract still leads with "the first quantum similarity learning system for planetary terrain retrieval" and "the encoding strategy accounts for the largest performance gap." The first is true and modest; the second is now explicitly confounded.

#### Issue List

##### CRITICAL → MAJOR (downgraded after revisions)

| # | Dimension | Issue | Location | Status vs prior review |
|---|-----------|-------|----------|------------------------|
| ~~C1~~ M1' | Data–Conclusion Mismatch | A1 ablation Δ=0.005 still treated as a finding in §6 ("a marginal advantage over an equivalently parameterized classical MLP"), but the Variance paragraph correctly labels it directional. The downgrade reflects that the manuscript now correctly *scopes* the claim. | main.tex:646–651, 612–620 | Resolved by disclosure |
| ~~C2~~ M2' | Logic Chain Break | A2 baseline confound now openly disclosed in §7 Discussion. The downgrade reflects honest disclosure; the underlying methodological issue remains because A2.5 has not been run. | main.tex:793–802 | Disclosure only; experiment still needed |
| C3 (RETAINED) | Foundation Collapse | The headline configuration ($N=8, L=2$) is still dominated by individual cells (A3 $L=3$ at 0.556, A5 $N=4$ at 0.540, A4 random at 0.538). The conclusion now names the combined-best (N=4, L=3, random) as the next experiment but does not run it. The abstract straddles ("0.533, with $L=3$ reaching 0.556"). | Abstract, Table I, main.tex:867–870 | Still Critical — one inexpensive run resolves it |

##### MAJOR

| # | Dimension | Issue | Location |
|---|-----------|-------|----------|
| M1 | Cherry-Picking (Reduced) | The abstract and conclusion lead with macro-MAP@10 = 0.533 for the default. Table I shows only the default. Table II contains the $L=3$ → 0.556 cell. The framing is now more honest than the prior pass (the abstract concedes $L=3$ reaches 0.556) but still privileges the default. | Abstract, Table I |
| M2 | Overgeneralization (Reduced) | "Domain-specific metric learning … outperforms large-scale transfer for minority planetary terrain classes" (main.tex:863–865) — still only one transfer-learning baseline tested. | Conclusion |
| M3 | Confirmation Bias (Resolved) | The single-N BP test is no longer used to "verify" Cerezo's theorem; it is now framed as "consistent with the prediction" with scaling verification deferred. Acceptable. | main.tex:150–156, 779–783 |
| M4 | "So What?" (Retained) | If A1 is within noise, the operational implication is "use the classical MLP." The paper now states the variance caveat but does not draw the implication. | Discussion §6 |
| M5 | Alternative Paths Not Considered (Retained) | The paper does not consider that the A2 gap might be achievable classically by appropriate input augmentation or sequential-vs-interleaved feature concatenation. | main.tex:646–651 |
| M6 (NEW) | Disclosure-as-Resolution | Six prior-pass Required Revisions have been converted from "fix" to "acknowledged limitation." This is honest but tests reviewer patience — at some point, the limitations subsection becomes a substitute for empirical work. | Limitations (main.tex:838–847) |

##### MINOR

| # | Dimension | Issue | Location |
|---|-----------|-------|----------|
| m1 | Stakeholder Blind Spot | Planetary-science end-user perspective still rhetorical. | §1, §6 |
| m2 | Logic Chain (Reduced) | "Encoding strategy … is the dominant factor for minority-class retrieval, while the quantum circuit itself provides only a marginal advantage" (main.tex:648–651) — this is now an honest scoping rather than overreach, but the word "dominant" should be qualified given the A2 confound. | Results §6.1 |
| m3 | Hidden Assumption | Taxonomic relevance as operational relevance — still unaddressed. | §5.2 |
| m4 (NEW) | Self-Critique-as-Defense | The manuscript now anticipates many reviewer criticisms and acknowledges them in advance. This is good practice, but the reviewer must be careful not to reward acknowledgment as a substitute for resolution. | Limitations (main.tex:838–847) |

#### Ignored Alternative Explanations / Paths

1. **The Δ=0.061 A2 gap may collapse entirely under A2.5.** One additional training run resolves this.
2. **The Δ=0.005 A1 gap may flip sign under multi-seed averaging.** Three additional runs per cell resolve this.
3. **The ResNet50 baseline may be undertuned because of grayscale→RGB replication without first-layer reinitialization.** This is now disclosed (main.tex:602–603), but the alternative was not tested.
4. **The N=4 winner in A5 suggests the CNN encoder, not the quantum head, drives the embedding-space discrimination.** The paper does not engage with this implication.

#### Missing Stakeholder Perspectives

- **HiRISE / PDS operations team** (would the system actually be deployed?)
- **Quantum hardware engineers** (NISQ shot-budget implications for the local-observable scheme)
- **IR / image-retrieval community** (how does macro-MAP@10 ≈ 0.53–0.56 compare to existing planetary-imagery retrieval baselines?)

#### Unexamined Premise

The paper continues to assume that *better quantum encoding* is the right axis along which to seek progress. An equally well-motivated paper would ask: given that the dataset has extreme class imbalance and visually confusable subcategories, what is the right inductive bias? The quantum-circuit machinery may simply be the wrong tool; the gains may be attainable more cheaply through better augmentation, larger encoders, or per-class loss weighting.

#### Observations (Non-Defects)

- The morning's revisions are unusually disciplined: every disclosure is correctly scoped, no claim has been weakened beyond what the evidence requires, and the manuscript now reads as the work of an author who has genuinely internalized the prior review feedback.
- The IEEE AI-use disclosure (main.tex:884–891) is appropriately compliant.
- The "Variance and replication" paragraph (main.tex:612–620) is the kind of methodological honesty that is rare in this literature and should be the rule rather than the exception.

---

## Phase 2 — Editorial Decision

### Decision: **Major Revision** *(narrow margin to Minor Revision)*

*Rationale for decision (320 words).*

Three of four scoring reviewers recommend Major Revision; R2 (QML Domain) recommends Minor Revision (borderline Major). The Devil's Advocate retains one CRITICAL finding (C3: headline-vs-best-ablation mismatch), which — per IRON RULE #4 — precludes Accept. The weighted dimension scores have moved upward from the morning's pass (panel mean weighted score 65.3 vs 61.7), but remain in the upper-Major / lower-Minor band (60–74).

**Why Major rather than Minor.** Three of the prior pass's Required Revisions — multi-seed replication, A2.5 axis-matched control, and the (N=4, L=3, random) combined-best ablation — have been converted from "fix" to "acknowledged limitation" rather than performed. Each is inexpensive (1–6 additional training runs). The headline encoding claim is now explicitly confounded in the Discussion but not empirically resolved. The operational retrieval scenario and standard image-retrieval baselines are still missing.

**Why not Reject.** The manuscript has visibly improved on multiple fronts. The variance acknowledgment, the BP claim softening, the qualitative-distinction paragraph, the ResNet50 specification, the A2 confound disclosure, the Limitations subsection, the AP@K formula, and the Data Availability statement together represent a substantial uplift in scientific rigor. The author has demonstrated they understood the prior review and responded coherently. The retrieval-vs-classification reframing remains the paper's best contribution.

**Why not Minor.** A Minor Revision recommendation would require that the remaining changes be primarily textual or limited-scope. The needed changes here — A2.5 ablation, multi-seed runs for A1 and A4, combined-best run, one operational scenario figure — require new experiments, not just text edits. The author's own conclusion names (N=4, L=3, random) as the immediate next experiment; submitting *before* running it is what tips the decision back to Major.

The estimated revision effort has dropped from 4–6 weeks to **2–3 weeks** thanks to the morning's text-level improvements.

### Reviewer Summary

| Reviewer | Role | Recommendation | Confidence | Δ vs prior pass |
|----------|------|---------------|------------|-----------------|
| EIC | IEEE TGRS associate editor | Major Revision (borderline Minor) | 4/5 | +3.5 score |
| R1 | Methodology Reviewer | Major Revision (reduced scope) | 5/5 | +4.0 score |
| R2 | QML Domain Reviewer | Minor Revision (borderline Major) | 5/5 | +5.0 score |
| R3 | Planetary Science Perspective | Major Revision (low end) | 4/5 | +1.8 score |
| DA | Devil's Advocate | One retained CRITICAL → Accept precluded | n/a | C1, C2 downgraded; C3 retained |

### Consensus Analysis

**[CONSENSUS-4]** (all four scoring reviewers agree):
1. The (N=4, L=3, random mining) combined-best ablation must be run before resubmission (EIC W2, R1 W3, DA C3, R2 implicit).

**[CONSENSUS-3]**:
1. A2.5 axis-matched ablation must be run to disentangle interleaved-vs-sequential ordering from axis choice (EIC W1, R1 W2, DA C2 retained).
2. Multi-seed runs for A1 and A4 are needed before any directional interpretation can stand (R1 W1, DA M1, EIC W1).
3. At least one operational retrieval scenario or standard image-retrieval baseline is needed for TGRS scope (EIC W3+W4, R3 W1).

**[CONSENSUS-2]**:
1. Entanglement quantification would resolve a substantive QML question (R2 W1, DA "ignored alternative" #1).

### Points of Disagreement

**Disagreement 1: Severity of remaining empirical gaps.**
- EIC, R1, R3, DA: Major — text disclosure is not a substitute for ablation runs.
- R2: Minor — the QML-substantive issues (BP framing, A2 confound, qualitative-distinction paragraph) are now correctly scoped; the missing experiments are evidence-thinning, not framing errors.
- **Editor's resolution:** Major Revision. R2's position is internally consistent and the QML rigor has improved markedly, but the empirical gaps remain too substantial for Minor — and the missing experiments are inexpensive.

**Disagreement 2: TGRS scope fit.**
- EIC: Marginal — operational scenario and retrieval baselines needed; otherwise consider quantum-engineering venue.
- R2: Acceptable — QML rigor now meets TGRS standards.
- R3: Borderline — planetary-science framing still rhetorical.
- **Editor's resolution:** TGRS remains acceptable conditional on W3 (operational scenario) and W4 (standard retrieval baseline). If the author prefers, *IEEE Transactions on Quantum Engineering* or *Quantum Machine Intelligence* would also welcome the work without these specific additions.

---

### Required Revisions (Must Fix)

| # | Revision Item | Source | Severity | Section | Effort |
|---|--------------|--------|----------|---------|--------|
| **R1** | Run the combined-best configuration ($N=4$, $L=3$, random mining). Update Table I (or add a "best of ablation" row) and reconsider the abstract's headline number. | EIC W2, R1 W3, DA C3, conclusion (main.tex:867–870) | Critical | §6, Abstract, Table I | 1 day |
| **R2** | Run A2.5 axis-matched ablation: sequential $\RX$/$\RY$ at matched axes as \qts{}, plus optionally interleaved $\RZ$/$\RY$. Update the A2 discussion (main.tex:793–802) with the result. | R1 W2, DA C2 retained | Critical | §5.3, §7 | 1–2 days |
| **R3** | Run 3 seeds for A1 (Classical MLP vs Quantum) and A4 (random vs class-pair mining). Report mean ± std. Update the Variance paragraph (main.tex:612–620) with the empirical std. | R1 W1, DA M1, EIC W1 | Critical | §5, Tables I/II | 3–4 days compute |
| **R4** | Add one operational worked retrieval scenario: query patch (e.g., known dune migration site) → top-10 retrieved patches → 1–2 sentences of expert assessment per result. | EIC W3, R3 W1 | Major | New §6 subsection | 1 day |
| **R5** | Add at least one standard image-retrieval baseline (e.g., DSH/deep supervised hashing, or pretrained-CNN + cosine similarity without triplet training). | EIC W4, R1 W4 | Major | §5.3 | 2–3 days |
| **R6** | Add wall-clock training time per epoch for \qts{} vs Classical MLP on identical hardware. One line in §6. | R1 W5 | Minor | §6 | 0.5 day |
| **R7** | Reconcile the headline configuration with the abstract: if the default $N=8, L=2$ is kept, state explicitly why (narrative simplicity, parameter-count matching, etc.); otherwise update Table I and the abstract. | EIC W2, DA C3 retained | Major | Abstract, §6, §8 | 0.5 day |
| **R8** | Add a paragraph in §7 Discussion explaining the $64\times64$ patch-size choice and its operational implications. | R3 W2 | Major | §7 | 0.5 day |

### Suggested Revisions (Should Fix)

| # | Revision Item | Source | Priority |
|---|--------------|--------|----------|
| S1 | Compute and report entanglement quantification (half-chain von Neumann entropy or Schmidt rank distribution) on trained \qts{} vs SLIQ-style checkpoints. | R2 W1 | P2 |
| S2 | Compare to `BasicEntanglerLayers` ansatz at matched parameter count, or acknowledge as limitation. | R2 W3 | P2 |
| S3 | Add multi-$N$ BP plot for completeness (claim no longer requires it but it would strengthen). | R2 W2 | P3 |
| S4 | Motivate the $\pi$ rotation scaling explicitly in §4.3.1. | R2 W4 | P3 |
| S5 | Back-of-envelope NISQ shot-budget estimate. | R2 W5 | P3 |
| S6 | Add one paragraph on taxonomic relevance as a benchmark proxy for operational relevance. | R3 W4 | P3 |

### Revision Roadmap

**Priority 1 — Empirical work** *(estimated 7–11 days)*
- [ ] R1: Combined-best ($N=4, L=3$, random mining) ablation cell
- [ ] R2: A2.5 axis-matched ablation
- [ ] R3: 3-seed runs for A1 and A4 (6 runs total)
- [ ] R5: One standard image-retrieval baseline

**Priority 2 — Content additions** *(estimated 2 days)*
- [ ] R4: Operational worked retrieval scenario figure
- [ ] R6: Wall-clock comparison
- [ ] R7: Headline reconciliation
- [ ] R8: Patch-size discussion paragraph

**Priority 3 — QML strengthening (optional but recommended)** *(estimated 2–3 days)*
- [ ] S1: Entanglement quantification
- [ ] S2: Ansatz comparison
- [ ] S3–S6: Minor textual additions

**Total estimated effort:** 2–3 weeks (down from 4–6 weeks at the morning's pass).

### Revision Deadline

- **Recommended deadline:** 2026-07-04 (4 weeks; tight but tracked given that all changes are well-scoped)
- **Re-review:** Yes — Major Revision requires re-review (`/ars-reviewer re-review` mode)
- **Maximum rounds:** Per editorial policy, 2 rounds of Major Revision then accept-or-reject

### Closing

The author has responded to the morning's review with substantial textual revisions that materially improve the manuscript's epistemic credibility. The Variance and replication paragraph, the open disclosure of the A2 confound, the softened BP claim, the qualitative-distinction paragraph framing Cerezo correctly as scaling, the fully specified ResNet50 baseline, the Limitations subsection, the AP@K formula, and the Data Availability statement are all unforced improvements that demonstrate genuine engagement with the review.

What now separates the paper from acceptance is a small, well-scoped set of additional experiments — the combined-best ablation cell, the A2.5 axis-matched control, and 3-seed replication for A1 and A4 — plus one operational worked retrieval scenario and one standard image-retrieval baseline. These are inexpensive and the author's own conclusion already identifies the first of them as the immediate next step. We encourage the author to perform these experiments rather than continue to convert empirical gaps into limitations, and we look forward to a revision that closes the gap to Minor Revision territory.

---

## Appendix: Score Summary

| Reviewer | Originality | Methodology | Evidence | Coherence | Writing | Weighted | Δ vs morning |
|----------|-------------|-------------|----------|-----------|---------|----------|--------------|
| EIC | 60 | 62 | 60 | 70 | 80 | **64.5** | +3.5 |
| R1 (Methodology) | 60 | 60 | 55 | 72 | 82 | **63.5** | +4.0 |
| R2 (QML Domain) | 62 | 68 | 60 | 76 | 82 | **68.5** | +5.0 |
| R3 (Perspective) | 68 | 62 | 55 | 68 | 78 | **64.5** | +1.8 |
| **Panel mean** | **62.5** | **63.0** | **57.5** | **71.5** | **80.5** | **65.3** | **+3.6** |

**Decision mapping:** Weighted average 65.3 → upper end of Major Revision band (50–64) / lower end of Minor Revision band (65–79). DA C3 (foundation-collapse: headline-vs-best-ablation mismatch) retained → Accept precluded; Major Revision confirmed but at the threshold to Minor.

**Comparative band shift since morning:** panel mean rose from 61.7 (Major-mid) to 65.3 (Major/Minor boundary). The decision boundary held because three Required Revisions remain empirical rather than textual.

---

*End of review report.*
