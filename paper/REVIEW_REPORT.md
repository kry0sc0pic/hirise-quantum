# Academic Peer Review Report — `paper/main.tex`

**Paper title:** Quantum Similarity Learning for Mars Terrain Retrieval via Entangled-Pair Encoding
**Submitting venue (declared):** IEEE Transactions on Geoscience and Remote Sensing
**Review mode:** `full` (5 reviewers + EIC + synthesizer)
**Review date:** 2026-05-30
**Reviewer team:** simulated panel (academic-paper-reviewer v1.9.1)

---

## Phase 0 — Field Analysis & Reviewer Configuration

| Dimension | Determination |
|-----------|---------------|
| Primary discipline | Quantum machine learning (PQC-based metric learning) |
| Secondary discipline | Planetary remote sensing (HiRISE imagery) |
| Research paradigm | Empirical / computational (Siamese architecture + ablation study) |
| Methodology type | Hybrid classical–quantum simulation; supervised retrieval |
| Target tier | IEEE Transactions (top-tier IEEE journal, double-column) |
| Paper maturity | **Pre-submission draft — incomplete experimental results (placeholder `\res{X.XX}` values throughout Tables I and II; two of three figures are empty `\fbox{}` placeholders)** |

### Reviewer panel

| # | Role | Configured identity |
|---|------|--------------------|
| EIC | Editor-in-Chief | Associate editor at IEEE TGRS with background in deep-learning–based remote sensing; gatekeeper for scope and submission readiness |
| R1 | Methodology Reviewer | ML methodologist with focus on retrieval evaluation protocols and reproducibility |
| R2 | Domain Reviewer | Quantum machine learning theorist; familiar with PQC expressivity, barren plateaus, and quantum kernels |
| R3 | Perspective Reviewer | Planetary scientist / HiRISE user; operational PDS workflow expertise |
| DA | Devil's Advocate | Adversarial reader hunting for inflated claims, missing baselines, and cherry-picked framing |

---

## Phase 1 — Five Independent Reviews

> **IRON RULE applied:** Reviewers do not cross-reference each other. Each operates from a distinct lens.

---

### Review 1 — Editor-in-Chief

**Lens:** journal fit, originality envelope, submission readiness, overall significance.

**Summary judgement.** The paper would make a credible contribution to *IEEE TGRS* **if** complete experimental results existed, but in its current state it is not a finished manuscript. Tables I and II contain `\res{X.XX}` placeholders for every comparator condition; the architecture and circuit figures are empty boxes; the discussion of the "key" ablation A2 is written in the future tense ("we expect"). The cover letter has not been drafted. An EIC seeing this submission would desk-reject before peer review.

**Scope fit.** IEEE TGRS publishes geoscience/remote-sensing methodology; quantum ML is acceptable when motivated by a remote-sensing problem. The motivation here (retrieval on HiRISE) is appropriate, though the technical core (PQC encoding, barren plateaus, StronglyEntanglingLayers) reads more like a quantum-ML methods paper. *IEEE Transactions on Quantum Engineering* or *Quantum Machine Intelligence* would be a more natural fit for a methods-first reading; *TGRS* fit depends on whether the retrieval framing and HiRISE-specific findings carry the paper. As written, ~70% of the technical body is QML methodology — borderline for TGRS scope.

**Originality.** The application (quantum similarity on planetary terrain) is genuinely novel; no prior work targets this combination. The methodological novelty is narrower than the framing suggests: the interleaved $R_X/R_Y$ encoding is a small variation on SLIQ's sequential encoding, and the "non-commuting rotations create joint encoding" argument applies to any data re-uploading PQC. The barren-plateau-mitigation contribution is borrowing Cerezo et al. (2021) rather than a new result.

**Readership relevance.** TGRS readers care about (i) operational utility and (ii) honest comparison to remote-sensing baselines. The paper has neither: no rover-mission scenario is concretely worked out, and no classical remote-sensing baselines (e.g., pre-trained ResNet features, learned hashing, classical Siamese networks) appear.

**Strengths.**
- Clear motivation for retrieval framing over classification.
- Well-structured five-factor ablation design (if it were executed).
- Reproducibility-friendly appendix with code snippet.
- Honest about the 96% classification baseline being incomparable.

**Weaknesses (EIC-level).**
1. **Submission readiness:** placeholders, empty figures, no comparator numbers.
2. **Scope drift:** technical body is QML-heavy for TGRS.
3. **Originality framing is overstated:** the "first" claim is true only for the HiRISE+quantum-similarity intersection; the methodological delta over SLIQ is small.
4. **Missing classical remote-sensing baselines:** TGRS reviewers will ask "vs. a fine-tuned ResNet50?" and the paper has no answer.

**EIC dimension scores (0–100, per `quality_rubrics.md`):**

| Dimension | Score | Note |
|-----------|-------|------|
| Originality | 48 | Modest methodological delta; domain-application novelty real but narrow |
| Significance | 40 | Cannot assess until baselines exist |
| Scope fit (TGRS) | 50 | Acceptable but borderline |
| Presentation | 55 | Writing is clean; figures and tables are unfinished |
| Readiness | **20** | Hard ceiling — placeholders block submission |

**EIC tentative decision: Reject (not ready for review).** Re-evaluate after Tables I/II are completed with real numbers and the manuscript is figure-complete.

---

### Review 2 — Methodology Reviewer

**Lens:** experimental rigor, evaluation protocol, statistical validity, reproducibility.

**Critical issue 1 — train/val/test contamination (CRITICAL).**
Inspection of the implementation (`src/train.py:147–153`) shows the dataset is split 85/15 train/val with `torch.utils.data.random_split`. There is **no held-out test set**. The reported MAP@10 = 0.962 is produced by `eval/eval.py:evaluate_model`, which calls `compute_embeddings(model.encoder, data_root, ...)`. `data_root` is the **entire HiRISE directory**, i.e., the union of training and validation images. The MAP@10 query set therefore contains the training set, and the retrieval "database" being searched also contains the training set. This is a methodological invalidation: the headline number cannot distinguish memorization from generalization. *This issue alone prevents accept or minor-revision decisions* and must be fixed before any claim of retrieval quality is credible.

**Critical issue 2 — class-imbalance inflation of MAP@10 (CRITICAL).**
The "other" class is 83.6% of HiRISE v3 (Table I in the paper). MAP@10 on a uniformly sampled query set with this distribution is heavily biased upward: ~84% of queries are "other" patches, and ~84% of the top-10 retrieved patches will be "other" by base rate alone. A trivial constant embedding ($\phi(x) = c$ for all $x$) would obtain a high MAP@10 on this dataset because the retrieved set is dominated by the majority class. Report **macro-averaged** MAP@10 (per-class then averaged), or stratified per-class metrics, or evaluate only on the seven minority classes. Without this, the 0.962 number is uninterpretable.

**Critical issue 3 — single run, no variance reporting (MAJOR).**
The paper reports a single training run. No random seed is pinned (no `torch.manual_seed`, `np.random.seed`, `random.seed` in the training pipeline). No confidence intervals, error bars, or multiple-seed averaging. For an IEEE Transactions retrieval benchmark, the minimum bar is 3–5 seeds with mean ± std on the primary metric.

**Critical issue 4 — validation accuracy proxy ≠ MAP@10 (MAJOR).**
Training (`src/train.py:101–111`) selects the best checkpoint by `Pr[s_ap > s_an]` — a triplet-margin proxy — but the reported headline metric is MAP@10. These are not equivalent. The proxy can prefer checkpoints that are good at the binary ranking task but suboptimal for the K-ranked retrieval task. Either select on the actual evaluation metric, or document the discrepancy and motivate it.

**Issue 5 — MAP@K definition ambiguity (MAJOR).**
Section 5.2 says MAP@10 is "averaged over all test queries", but the AP-per-query definition is not given in the paper. The implementation (`eval/eval.py:113–132`) uses `precisions = np.cumsum(rel) / (np.arange(k) + 1)` and `ap = (precisions * rel).sum() / rel.sum()`. This is a non-standard definition: standard AP@K normalizes by `min(K, total_relevant_in_DB)`, not by relevant-in-top-K. With the current implementation, a query with 0 relevant in top-10 gets AP=0; a query with 1 relevant in top-10 gets AP=1/(rank); a query with all 10 relevant gets AP=1.0. The definition should be stated explicitly in §5.2 and matched to a textbook reference (e.g., Manning et al. IR textbook).

**Issue 6 — barren-plateau "$\Omega(1/64)$ vs $O(1/256)$" comparison is incoherent (MAJOR).**
Section 4.3.3 writes: *"At N=8, the local lower bound is Ω(1/64), whereas the global upper bound is O(1/256)—a 4× difference that compounds across all parameters and epochs."* You cannot subtract or take ratios of $\Omega$ and $O$ bounds — they bound different quantities (worst-case lower vs. worst-case upper). The actual content of Cerezo et al. 2021 is the **asymptotic** distinction between $\mathrm{poly}(N)^{-1}$ (local) and $2^{-N}$ (global). Restate without the "4×" arithmetic; cite Theorem 1 (or whichever) of Cerezo et al. with the exact bound. As written, this passage will be flagged by any QML-aware reviewer.

**Issue 7 — COBYLA fallback safeguard is undocumented in results (MINOR).**
Section 4.5 introduces the COBYLA gradient-free fallback and states "This safeguard was not triggered in any reported experiment." This is good, but the gradient-norm history plot (Fig. 4) only goes to 50 epochs. State explicitly: across how many seeds, across which ablation conditions, was COBYLA never triggered? If only the headline run, then this claim is overgeneralized.

**Issue 8 — triplet construction details missing (MINOR).**
How many triplets per epoch? Is it one triplet per anchor (yielding `len(dataset)` triplets/epoch)? `HiRISETripletDataset.__getitem__` (`src/dataset.py:81–89`) returns one triplet per index → 73,031 triplets/epoch at batch 16 → 4565 steps/epoch. State this. Also: is the anchor sampled uniformly over **samples** or uniformly over **classes**? The code uniform-samples by index, so it inherits the class-imbalance bias.

**Issue 9 — image normalization is misdescribed (MINOR).**
Section 5.1 says "images are … normalized to $[-\pi, \pi]$ (mean 0.5, std $1/(2\pi)$ on the unit scale)". The code (`src/dataset.py:31`) uses `Normalize(mean=[0.5], std=[1.0 / (2.0 * np.pi)])`, which maps $[0,1]$ to roughly $[-\pi, \pi]$. This is then fed into `RX(\pi e_a)` rotations — so the rotation angle becomes up to $\pi \cdot \pi = \pi^2 \approx 9.87$ rad, **not** the natural $[-\pi, \pi]$ rotation domain. This is a numerical inconsistency between the prose, the rotation gates, and the normalization. Either the prose or the encoding gate definition is wrong.

**Issue 10 — hyperparameters not justified (MINOR).**
$\alpha_{\mathrm{CNN}} = 10^{-3}$ and $\alpha_{\mathrm{PQC}} = 10^{-2}$ are stated but not motivated. Margin $m = 0.2$ is also unmotivated. Either cite the source for these defaults (SLIQ? PennyLane tutorial?) or report a brief sensitivity study.

**Reproducibility scorecard.**
- [x] Code path implied (appendix gives QNode snippet)
- [ ] Code release (no GitHub/Zenodo link)
- [ ] Random seed
- [ ] Data split files
- [ ] Hyperparameter justification
- [x] Software versions (PyTorch 2.1, PennyLane 0.45)
- [ ] Hardware (only "CPU" — which CPU, how many cores, wall-time?)

**Methodology decision: Major Revision (effectively Reject in current form).** Issues 1–6 are individually publication-blocking.

---

### Review 3 — Domain Reviewer (Quantum Machine Learning)

**Lens:** QML literature coverage, theoretical correctness, contribution to the QML field.

**Positioning relative to QML literature.**
- SLIQ (Silver et al. 2023) is correctly identified as the closest prior work. Good.
- Quantum kernels (Schuld & Killoran 2019, Havlíček et al. 2019, Schuld 2021 *PRL*) are touched on in passing but never quantitatively compared. Quantum metric learning is essentially quantum-kernel learning with a triplet loss — that connection should be made explicit, with a citation to Lloyd, Schuld, et al. 2020 (already in your bib) on *Quantum Embeddings for Machine Learning*, which is the strongest theoretical companion to your approach.
- Recent work on **trainable quantum embeddings** (Pérez-Salinas et al. 2020, *Quantum* — "data reuploading", arXiv:1907.02085) is missing. Their data-reuploading universal classifier *is* a form of interleaved-encoding scheme and would be the natural antecedent for your Contribution 2. Failure to cite this is the most damaging literature gap.
- Quantum Siamese networks have been explored beyond SLIQ — see, e.g., Beer et al. 2020 *Nat. Commun.* on training deep quantum NNs (Siamese variant), and Hubregtsen et al. 2022 on kernel-based quantum classifiers. The "no prior work has applied quantum similarity functions to planetary terrain data" claim is true; the broader claim about quantum similarity in general should be softened.

**Theoretical correctness of Contribution 2 (entangled-pair encoding).**
The technical claim — that $R_Y(\beta) R_X(\alpha) |0\rangle$ is a nonlinear joint function of $(\alpha, \beta)$ — is correct. The matrix expansion in Eq. (after eq:encoding) is correct. **However**, the *significance* of this claim is overstated for two reasons:

1. *Single-qubit state space is only 2-real-dimensional.* The post-encoding qubit state $|\psi_i\rangle \in \mathbb{S}^2$ (Bloch sphere) is parameterized by $(\alpha, \beta) \in [-\pi, \pi]^2$, the **same** parameter count as sequential encoding $R_X(\alpha)$ then later $R_X(\beta)$. The difference is in the **map** $(\alpha, \beta) \mapsto |\psi_i\rangle$, not in expressive capacity at the single-qubit level. The expressivity advantage, if any, materializes only after entangling layers process these per-qubit states. The paper conflates the two.

2. *Sequential encoding can also produce joint state preparation.* SLIQ uses $R_Z(\pi e_a^{(i)})$ then $R_X(\pi e_b^{(i)})$, separated by no entangling gate. Two non-commuting rotations on the same wire compose to a joint state — exactly the same as your $R_X$ then $R_Y$. The actual distinction with SLIQ is **the choice of axes** ($R_Z R_X$ vs $R_X R_Y$), not whether the encoding is "joint vs sequential". The current framing — "interleaved (joint) vs sequential (independent)" — is technically misleading; both are joint. Recommend reframing as *"alternative axis choices for joint per-qubit encoding"* with a careful discussion of why $R_X/R_Y$ might differ from $R_Z/R_X$ (hint: $R_Z|0\rangle = |0\rangle$ up to phase, so SLIQ's $R_Z$ on $|0\rangle$ is a no-op for measurement-axis observables — this is the actual technical asymmetry, and it deserves to be the central argument). As written, A2 will not survive a careful QML reviewer.

**Theoretical correctness of Contribution 3 (barren plateau mitigation).**
Citing Cerezo et al. 2021 (Nature Comm. 12:1791) for local-observable BP mitigation is correct. Two issues:

- The "4× factor" arithmetic (Sec. 4.3.3) is wrong as flagged by Methodology Review (Issue 6). Use the asymptotic statement.
- At $N=8$, $L=2$, the BP theory is **not in the asymptotic regime** the theorem describes. The Cerezo et al. result kicks in for $L = O(\log N)$ in the limit of large $N$ with random initialization. For $N=8$, $L=2$, both local and global gradients are likely non-vanishing in practice regardless of measurement choice. The empirical gradient-norm plot (Fig. 4) confirms gradients are healthy — but this is weak evidence for the BP mitigation claim, because the same plot under global measurement would also show healthy gradients at this scale. To support Contribution 3 you need either (a) a head-to-head ablation: same circuit, local vs global observable, plot both gradient curves; or (b) scale the experiment to $N=12$ where the difference might be visible; or (c) demote the claim to "we adopt the recommended local-observable design" without claiming empirical mitigation evidence.

**Cited paper details.**
- `cerezo2021costbarren` in bib has `pages = {1791}`. Verify: the actual Nature Communications paper is "Cost-function-dependent barren plateaus in shallow parametrized quantum circuits" by Cerezo, Sone, Volkoff, Cincio, Coles — Nat. Commun. 12:1791 (2021). DOI 10.1038/s41467-021-21728-w. The bib entry is correct as published. (Note: an earlier session draft of the project summary references "12:6961" which would be wrong — the bib has the right number.)
- Bergholm et al. PennyLane: should be cited as the *software* and ideally one of its companion methods papers; arXiv reference is OK as `@misc`.

**Missing references (must cite).**
1. **Pérez-Salinas et al. 2020**, "Data re-uploading for a universal quantum classifier", *Quantum* 4:226. Direct antecedent for interleaved-encoding schemes.
2. **Schuld 2021**, "Supervised quantum machine learning models are kernel methods", arXiv:2101.11020. Provides the formal kernel reading of your method.
3. **Larocca et al. 2024**, "A review of barren plateaus in variational quantum computing", *Nature Reviews Physics*. Updated synthesis of the BP landscape — Cerezo et al. 2021 is now five years old.
4. **Holmes et al. 2022**, "Connecting ansatz expressibility to gradient magnitudes", *PRX Quantum* 3:010313. Expressibility–BP trade-off; directly relevant to your StronglyEntanglingLayers choice.

**Strengths.**
- Honest about being a hybrid classical–quantum architecture; doesn't claim quantum advantage.
- Correct identification of the local-observable measurement strategy as relevant.
- The mathematical derivation of $R_Y R_X$ is laid out cleanly.

**Weaknesses.**
- A2 framing is technically misleading (joint-vs-sequential).
- Contribution 3 evidence is weak at $N=8$.
- Missing data-reuploading literature (Pérez-Salinas) is the most embarrassing gap.
- "4× difference" arithmetic is mathematically wrong.

**Domain decision: Major Revision.**

---

### Review 4 — Perspective Reviewer (Planetary Science / Cross-Disciplinary)

**Lens:** does this paper matter to the planetary remote-sensing community? Is the framing of the operational problem sound?

**Motivation — strong.** The pivot from classification to retrieval is well-motivated. Operational HiRISE workflows at JPL/USGS do use similarity-based queries ("find more terrains like this exemplar"), and the existing CNN classifiers are not directly suited to that. Naming concrete use cases (aeolian activity search, rover trafficability, change detection) is good. Reference to Rothrock et al. 2016 (SPOC) for the rover trafficability angle is appropriate.

**Geological soundness of "confusable pairs".**
- bright_dune vs dark_dune: yes, well-known confusion in HiRISE classification; the distinction is primarily albedo and secondarily composition (mafic vs felsic mineralogy). They are geomorphically very similar.
- spider vs swiss_cheese: both are CO₂-sublimation features in the south polar terrain. **Spiders** are radial branching channels (araneiform terrain), formed at the base of seasonal CO₂ ice. **Swiss cheese** is positive-relief flat-floored pits in residual CO₂ ice. These are visually distinct at high resolution but can be confused at the 64×64 patch scale. Acceptable as a "hard pair" choice.
- Missing pair worth flagging: **slope_streak vs (dust devil tracks or albedo features)**. Slope streaks have well-documented confusion with seasonal dark albedo features and dust devil tracks. The fact that this isn't a "hard pair" in your study suggests the confusion matrix isn't comprehensive.

**Operational utility — weak.**
- The HiRISE archive has ~70K labeled patches but ~100M unlabeled tiles when full strips are considered. The system as built operates only on 64×64 labeled patches. The bridge to operational tile-search is not discussed.
- The output of the system is a similarity score, but no demonstration of *how a scientist would use it*. Show a query → top-10 retrieval gallery with actual HiRISE patches, for at least one query per minority class. This is standard in image-retrieval papers and would be the single most impactful figure for a TGRS audience.
- The cross-instrument extension (CTX, CaSSIS) in the conclusion is interesting but speculative; no resolution-matching protocol is described.

**Comparison to community standards.**
The PDS Imaging Atlas (Wagstaff et al. 2018) is the operational deployment cited. For TGRS readers, a more competitive baseline than "Classical head with quantum encoder's 2-conv CNN" would be:
- **Pre-trained ResNet50 features** fine-tuned with triplet loss (current industry default for image retrieval).
- **DINO / DINOv2 self-supervised features** without any fine-tuning (zero-shot retrieval baseline).
- A simple **per-class mean image + cosine similarity** sanity check.

None of these appear. A planetary science reviewer will ask: "Why use a quantum head when a fine-tuned ResNet retriever is a strong, well-understood baseline?"

**Cross-disciplinary impact.**
- Quantum ML community will care about the encoding ablation (A2). Planetary science community will care about retrieval gallery + operational integration.
- These are two different papers. The current draft tries to do both and underserves both.
- Recommendation: pick a primary audience. For TGRS, lead with the retrieval task, foreground concrete query-result galleries, and treat the PQC head as one of several methodological options. For QML venues, lead with the encoding ablation and use HiRISE as a substantive evaluation testbed.

**Strengths.**
- Genuine domain motivation; not pure ML chauvinism applied to "any benchmark".
- Correct identification of confusable terrain pairs (modulo the slope_streak omission).
- HiRISE class distribution table is accurate (matches NASA-released splits).

**Weaknesses.**
- No qualitative retrieval examples (query → top-K gallery).
- No strong remote-sensing baseline (ResNet/DINO/cosine).
- Gap from labeled patch → operational tile search is hand-waved.
- Cross-instrument generalization is speculative.

**Perspective decision: Major Revision.**

---

### Review 5 — Devil's Advocate

**Lens:** strongest counter-argument; cherry-picking; overgeneralization; logical gaps; "so what?" test.

**Strongest Counter-Argument (the one a hostile reviewer will make).**

*"The headline number MAP@10 = 0.962 has four independent inflation mechanisms stacked on top of each other:*
*(1) it is evaluated on the entire dataset including training images, not a held-out test set;*
*(2) it is dominated by the 'other' class which is 83.6% of the data and trivially retrievable;*
*(3) it is a single-seed point estimate with no variance reporting;*
*(4) the checkpoint was selected on a related-but-not-identical proxy metric (triplet ranking accuracy on the same val set).*
*All other claims in the paper — about the quantum head, the interleaved encoding, the barren-plateau-suppression strategy, the hard-negative mining — rest on this number. The supporting ablation table is entirely placeholder. Therefore, the paper's empirical content is currently zero usable data points."*

This is the argument an adversarial reviewer will write, and it is correct.

**Cherry-picking checks.**
- Single-seed reporting: ✗ (confirmed — no seed pinning in `src/train.py`).
- Best-of-N checkpoint selection on val set, reported as test: ✗ (confirmed — val and "test" are the same data).
- Metric chosen post-hoc: not verifiable (the paper claims MAP@10 was primary from the start), but the val-selection metric (triplet ranking accuracy) differs, which is suspicious.
- Comparison only to favorable baselines: ✗ (no ResNet, no DINO, no quantum-kernel SVM).

**Confirmation-bias detection.**
The discussion of A2 (the "key" ablation) is written as *"we expect the interleaved model to outperform the sequential model because..."*. No null hypothesis is entertained beyond a pro-forma sentence. The expectation is preregistered in writing but the experiment is not done. This is the textbook setup for confirmation bias — when the results come in, any direction will be rationalized as supporting the hypothesis. Recommend: write down in advance what counts as evidence against the interleaved-encoding hypothesis (e.g., "if interleaved MAP@10 is within ±0.005 of sequential MAP@10 across 5 seeds, we conclude no encoding-strategy effect").

**Logic-chain validation.**

| Claim | Evidence offered | Holds? |
|-------|-----------------|--------|
| Quantum head improves retrieval | None — placeholder for classical row | No |
| Interleaved beats sequential | None — placeholder for SLIQ row | No |
| Hard mining helps | None — placeholder for random-mining row | No |
| Depth 2 is the sweet spot | None — placeholders for L=1, L=3 | No |
| N=8 is the sweet spot | None — placeholders for N=4, N=12 | No |
| BP suppression works empirically | Fig. 4 grad-norm > threshold | Weak (no global-observable comparator) |
| QTerrainSim achieves MAP@10=0.962 | One CSV row | Weak (eval on training data; class imbalance) |

Of seven empirical claims, *zero* are supported by completed comparisons.

**Overgeneralization detection.**
- "First quantum metric learning architecture applied to planetary terrain data" — true and narrow; acceptable.
- "First application of quantum similarity learning to planetary terrain data" — same as above, duplicated.
- "Demonstrating strong retrieval performance" (abstract) — not demonstrated without baselines.
- "Opens a research thread at the intersection of quantum machine learning and planetary science" — premature; one paper does not open a thread.
- "$L=O(\log N)$-depth circuits" lower bound applied to $L=2, N=8$ — this is technically inside the regime ($\log 8 = 3$), but only barely.

**Ignored alternative explanations.**
1. *MAP@10 = 0.962 is achievable by a frozen pre-trained ImageNet feature extractor + cosine similarity* — this baseline is absent. If true, the quantum head adds nothing.
2. *The encoder does all the work; the quantum head is a passive linear-readout layer* — the post-PQC linear+sigmoid (`sigma(w·f + b)`) is highly flexible and could mask quantum-head ineffectiveness. Ablate the post-processing layer (replace with fixed sum-of-Z) to isolate the quantum head's contribution.
3. *The hard-negative mining is what produces the gains, not the quantum head at all* — A4 placeholder.

**Missing stakeholder perspectives.**
- **JPL HiRISE operators**: would not deploy a system without an integration story for full HiRISE strips.
- **NISQ hardware vendors**: would care about real-hardware results; paper defers all hardware experiments.
- **Reproducibility-sensitive reviewers (NeurIPS-style)**: would demand seed, data split, hyperparameter justification.
- **Quantum-skeptical classical-ML reviewers**: would demand a parameter-matched classical comparison (your `Linear(2N, 1) + Sigmoid` baseline has only $2N+1=17$ parameters, vs. the quantum head's 48 PQC parameters + post-processing; the classical baseline is **underparameterized by design** and will lose unfairly).

**"So what?" test.**
Suppose every result comes in exactly as predicted: QTerrainSim 0.962, Classical 0.85, SLIQ 0.91. What would that change? It would establish that on one Mars dataset, with one CNN backbone, with one training schedule, a particular quantum head beats two specific baselines. It would not establish quantum advantage; it would not establish generalization to other planetary domains; it would not establish a route to hardware deployment. The paper needs to articulate the **decision** a reader makes differently because of these results. Currently it does not.

**Devil's Advocate Issue List**

| Severity | Dimension | Location | Issue |
|----------|-----------|----------|-------|
| **CRITICAL** | Evaluation validity | `eval/eval.py:289–313` & §6.1 | Test set = training set ∪ val set |
| **CRITICAL** | Metric semantics | §5.2 & §6.1 | MAP@10 dominated by majority class; macro-MAP not reported |
| **CRITICAL** | Empirical evidence | Tables I & II | 0/7 comparative claims supported; placeholder values |
| **CRITICAL** | Headline figures | Figs. 1, 2, 5 | Architecture, circuit, confusion-matrix figures empty `\fbox{}` |
| **MAJOR** | Reproducibility | training code | No seed; single run |
| **MAJOR** | A2 framing | §3.3 & §4.3.1 | Joint-vs-sequential dichotomy is technically misleading |
| **MAJOR** | Missing baseline | §5.3 | No fine-tuned ResNet / DINO / quantum kernel comparator |
| **MAJOR** | Parameter mismatch | §5.3 (Classical baseline) | Classical head has 17 params vs quantum head's ~50+; unfair |
| **MAJOR** | BP claim strength | §4.3.3 | "$\Omega(1/64)$ vs $O(1/256)$ = 4×" is mathematically incoherent |
| **MAJOR** | Encoding normalization | §5.1 & code | Image normalization yields $\pi \cdot \pi \approx 9.87$ rad rotations, not $[-\pi,\pi]$ |
| **MAJOR** | Literature gap | §2 | Pérez-Salinas et al. 2020 (data reuploading) missing |
| **MINOR** | Triplet construction | §4.4 | Sampling distribution not specified |
| **MINOR** | Hyperparameter justification | §4.5 | $\alpha_{CNN}, \alpha_{PQC}, m$ unmotivated |
| **MINOR** | Hardware reporting | §5.4 | "CPU" only; no wall-clock |
| **MINOR** | Conclusion overreach | §8 | "Opens a research thread" — premature |

**Observations (non-defects):** the LaTeX is clean; the prose is readable; the code is well-organized; the choice of `default.qubit` with backprop is appropriate for the scale.

---

## Phase 2 — Editorial Synthesis & Decision

### Consensus Matrix (5 reviewers)

| Issue | EIC | R1-Method | R2-Domain | R3-Persp | DA | Consensus |
|-------|-----|-----------|-----------|----------|-----|-----------|
| Placeholder results disqualify submission | ✓ | ✓ | — | — | ✓ | **3/5 strong** |
| Test set = training set (no held-out split) | — | ✓ CRIT | — | — | ✓ CRIT | **2/5 CRITICAL** |
| Class-imbalance inflation of MAP@10 | — | ✓ CRIT | — | — | ✓ CRIT | **2/5 CRITICAL** |
| Single-seed / no variance reporting | — | ✓ MAJ | — | — | ✓ MAJ | **2/5 MAJOR** |
| Missing classical remote-sensing baselines (ResNet/DINO) | ✓ | — | — | ✓ | ✓ | **3/5** |
| A2 "joint-vs-sequential" framing technically wrong | — | — | ✓ MAJ | — | ✓ MAJ | **2/5 MAJOR** |
| BP "4×" arithmetic is incoherent | — | ✓ MAJ | ✓ MAJ | — | ✓ MAJ | **3/5 MAJOR** |
| Empty `\fbox{}` figures block submission | ✓ | — | — | — | ✓ CRIT | **2/5 CRITICAL** |
| Missing data-reuploading literature | — | — | ✓ MAJ | — | — | **1/5** |
| Image normalization rotation-domain mismatch | — | ✓ MAJ | — | — | ✓ MAJ | **2/5 MAJOR** |
| No qualitative query-gallery figure | — | — | — | ✓ | — | **1/5** |
| Scope drift toward QML venue | ✓ | — | — | ✓ | — | **2/5** |

### Devil's Advocate CRITICAL findings — IRON RULE check
DA flagged FOUR CRITICAL issues (evaluation validity, metric semantics, missing empirical evidence, empty figures). **Per Checkpoint Rule #4, Editorial Decision cannot be Accept.**

### Disagreements & Arbitration
- **EIC vs reviewers on QML venue suitability.** EIC notes scope drift; R3 concurs. R2 implicitly assumes QML venue. *Arbitration:* this is the author's call after revision. Document the target audience in the cover letter.
- **No disagreement** on the four CRITICAL issues — all unanimous where applicable.

### Editorial Decision Letter

> *(For: Krishaay Jois, "Quantum Similarity Learning for Mars Terrain Retrieval via Entangled-Pair Encoding")*

Dear Mr. Jois,

Thank you for submitting your manuscript to *IEEE Transactions on Geoscience and Remote Sensing*. After review by an editorial board member and four peer reviewers (including a devil's advocate panel member), I regret to inform you that the manuscript in its current form **cannot be accepted for review**, and I am returning it as a **desk reject with invitation to resubmit**.

The principal reason is that the paper is structurally a draft rather than a completed submission: Tables I and II — which contain the central comparative claims of the paper — are populated with `\res{X.XX}` placeholder macros for every comparator condition (Classical head, SLIQ-style encoding, circuit depths $L \in \{1,3\}$, random-negative mining, qubit counts $N \in \{4, 12\}$). Three of five figures are empty `\fbox{}` placeholders. Of seven distinct empirical claims made in the paper, zero are supported by completed comparisons. The single completed configuration (QTerrainSim default) reports a high MAP@10 that, on closer inspection of the supporting code, is computed on the union of training and validation data and is heavily inflated by the 83.6% "other" class prior.

The conceptual framing and the writing are above the bar for an IEEE Transactions submission, and the experimental design (the five-factor ablation) is well-constructed. We would welcome a resubmission once the experimental program is executed and the methodological issues below are addressed.

Required revisions (in priority order):

1. **Establish a held-out test set.** Re-partition HiRISE v3 into train/val/test (recommended 70/15/15), pin the splits to disk, and report all headline numbers on the test set only.
2. **Report macro-MAP@10** (per-class then averaged) in addition to micro-MAP@10. Strongly consider reporting metrics restricted to the seven minority classes.
3. **Multi-seed evaluation.** Report mean ± std over ≥3 seeds for every entry in Tables I and II.
4. **Complete the five-factor ablation.** Tables I and II must have no `\res{}` macros. The A2 (Sequential vs Interleaved) ablation in particular is the only experiment that supports your central methodological claim and must be reported.
5. **Add a strong classical baseline.** A pre-trained ResNet50 or DINOv2 backbone fine-tuned with triplet loss on the same training split. Without this, the paper cannot make a "quantum vs classical" claim that TGRS readers will take seriously.
6. **Add a parameter-matched classical head.** Your current `Linear(2N,1)+Sigmoid` head has only 17 parameters vs the PQC head's ~50+. Match parameter counts (e.g., `Linear(2N, 64) → ReLU → Linear(64, 1) → Sigmoid` has ~1100 params, closer to the spirit of "what does the quantum head buy you").
7. **Complete the figures.** Replace `\fbox{}` boxes with: architecture diagram (TikZ or vector), Quantikz circuit diagram, and the actual hard-pair confusion matrices.
8. **Add a qualitative retrieval gallery.** Query patch on the left, top-10 retrieved patches on the right, one row per minority class. This is the single most impactful figure for a TGRS audience.
9. **Repair the barren-plateau argument** (§4.3.3). The "$\Omega(1/64)$ vs $O(1/256)$ = 4×" passage is mathematically incoherent. Restate using the asymptotic content of Cerezo et al. 2021. Optionally, add a local-vs-global observable head-to-head ablation to substantiate Contribution 3.
10. **Reframe Contribution 2.** The "joint (ours) vs sequential (SLIQ)" dichotomy is technically misleading — both schemes produce joint per-qubit states. The actual distinction is the **axis choice** ($R_X/R_Y$ vs $R_Z/R_X$), and the technical asymmetry hinges on $R_Z|0\rangle = |0\rangle$ being a no-op for $\langle Z \rangle$. Rewrite the contribution and the ablation discussion accordingly.
11. **Cite Pérez-Salinas et al. 2020** (*Quantum* 4:226, data re-uploading) as the principal antecedent for interleaved-encoding schemes. Also add Schuld 2021 (kernel view of QML), Holmes et al. 2022 (expressibility vs BP), and Larocca et al. 2024 (BP review).
12. **Reconcile the image normalization** (§5.1) with the rotation gate definition (§3.3.1). Either the prose ("normalized to $[-\pi, \pi]$") or the gate ($R_X(\pi e)$) needs to change; with the current code, encoding angles reach $\pi^2 \approx 9.87$ rad.

Optional but recommended:

13. Pin random seeds in code and release the data-split files.
14. Provide a code release (GitHub + Zenodo DOI).
15. Add a brief sensitivity study or citation justification for $\alpha_{CNN}, \alpha_{PQC}, m$.
16. Add wall-clock timing per epoch on specified hardware (currently only "CPU").
17. Consider whether *IEEE Transactions on Quantum Engineering* or *Quantum Machine Intelligence* would be a better venue if the methodology remains the principal contribution after revision.

I encourage you to take the time needed to complete the experimental program rigorously rather than to resubmit quickly with patched numbers. A correctly executed version of this study has clear potential to contribute to both the QML and planetary-remote-sensing communities; a hurried one will not.

**Decision: Reject (resubmission encouraged after major revision).**

Sincerely,
Editor-in-Chief, IEEE TGRS (simulated)
On behalf of the review panel

---

### Revision Roadmap (Prioritized — Directly Consumable by `academic-paper` Revision Mode)

#### Tier 1 — Publication blockers (must fix before any resubmission)

- [ ] **R1.1** Create disk-pinned 70/15/15 train/val/test split files (`data/splits/{train,val,test}.json`); update `dataset.py` and `train.py` to consume them.
- [ ] **R1.2** Update `eval/eval.py:compute_embeddings` to accept a split file and evaluate only on the test split. Re-run the headline experiment on the test split.
- [ ] **R1.3** Implement macro-MAP@10 (per-class averaging) in `eval/eval.py`; add to results table. Add minority-classes-only MAP@10 as a secondary number.
- [ ] **R1.4** Pin random seeds and re-run the headline configuration 3× (or 5×). Report mean ± std.
- [ ] **R1.5** Execute the full A1–A5 ablation grid (12 unique configurations × ≥3 seeds = ≥36 runs). Populate Tables I and II.
- [ ] **R1.6** Add a strong classical baseline: pre-trained ResNet50 or DINOv2 + triplet head, same training schedule. Add as a row in Table I.
- [ ] **R1.7** Add a parameter-matched classical head (`Linear(2N,64)→ReLU→Linear(64,1)→Sigmoid`) as a separate baseline.
- [ ] **R1.8** Create the three placeholder figures: architecture diagram (TikZ), circuit diagram (Quantikz), confusion matrix from real data.
- [ ] **R1.9** Add a qualitative retrieval gallery figure (one row per minority class).

#### Tier 2 — Technical correctness (must fix; methodology-critical)

- [ ] **R2.1** Rewrite §4.3.3 BP argument; remove the "$\Omega/O$" arithmetic; cite Cerezo et al. with the actual asymptotic claim. Optionally add local-vs-global observable ablation experiment.
- [ ] **R2.2** Rewrite Contribution 2 (§1.2, §4.3.1, §7) to reframe the SLIQ comparison as an **axis-choice** distinction, not a joint-vs-sequential distinction. Highlight the $R_Z|0\rangle = |0\rangle$ asymmetry as the actual technical hinge.
- [ ] **R2.3** Reconcile image normalization (§5.1) with encoding rotation domain. Either change the normalization or change the encoding (e.g., $R_X(e_a)$ without the $\pi$ prefactor, or normalize to $[-1, 1]$).
- [ ] **R2.4** Add Pérez-Salinas et al. 2020, Schuld 2021, Holmes et al. 2022, Larocca et al. 2024 to bibliography and discuss in §2.
- [ ] **R2.5** Pre-register the A2 hypothesis test: state in advance what counts as evidence against the interleaved-encoding hypothesis.

#### Tier 3 — Quality of submission (should fix; reviewer-friendliness)

- [ ] **R3.1** Clarify MAP@K definition in §5.2 with a formal equation (cite Manning et al. IR textbook).
- [ ] **R3.2** Document checkpoint-selection metric vs reported metric divergence; ideally select on test-MAP@10 (using a separate dev split).
- [ ] **R3.3** Document triplet construction (#triplets/epoch, anchor sampling distribution).
- [ ] **R3.4** Justify hyperparameters ($\alpha_{CNN}, \alpha_{PQC}, m$) — either cite SLIQ/PennyLane defaults or run a brief sensitivity study.
- [ ] **R3.5** Add hardware/wall-clock reporting (CPU model, cores, epoch time).
- [ ] **R3.6** Confirm COBYLA-not-triggered claim across all ablation runs, not just the headline.
- [ ] **R3.7** Soften "opens a research thread" → "demonstrates feasibility" or similar.
- [ ] **R3.8** Cover-letter decision: TGRS, TQE, or QMI? — articulate the target audience.

#### Tier 4 — Optional polish

- [ ] **R4.1** Code release (GitHub + Zenodo DOI), released splits, scripts to reproduce every table row.
- [ ] **R4.2** Add slope_streak vs (albedo features) as a third hard pair if confusion-matrix analysis supports it.
- [ ] **R4.3** Brief discussion of how the system would integrate with full HiRISE strip search at PDS scale.

**Estimated effort.** Tier 1 alone is on the order of 2–4 weeks of compute + analysis (ablation grid + multi-seed re-runs + figure production). Tier 2 is 1–2 weeks of writing + theoretical re-derivation + optional small experiment. Tier 3 is a few days. Total realistic resubmission timeline: **6–8 weeks** of focused effort.

---

## Phase 2.5 — Revision Coaching (Socratic Dialogue)

The Editorial Decision is *Reject with invitation to resubmit*. Per the skill protocol, the EIC offers Socratic guidance to help you formulate a revision strategy. You may skip this and proceed directly to the roadmap above by saying "just fix it" — otherwise, the EIC's coaching questions:

1. **Overall positioning.** After reading the panel's response, what surprised you the most? Was it the evaluation-protocol critique (test set = training set), the framing critique (joint-vs-sequential is misleading), or something else?
2. **Core issue focus.** Of the four CRITICAL issues — held-out test set, class-imbalance metric inflation, missing empirical evidence (Tables I/II placeholders), empty figures — which do you think you can address fastest? Which is the most expensive?
3. **Revision strategy.** If you could only address **three** roadmap items in the next two weeks, which three would have the highest payoff for the strength of the paper?
4. **Counter-argument response.** The Devil's Advocate's core counter-argument is: *"the headline number rests on four independent inflation mechanisms, and every comparative claim is currently empty"*. What is the strongest single response you can make to this — and what would make that response credible (an experiment, a re-analysis, a re-framing)?
5. **Implementation planning.** Is the right next move (a) execute the ablation grid as-is and report whatever results come in honestly, or (b) re-design the experiments first (proper splits, strong baselines) and then execute? The first is faster; the second is more defensible. Which fits your timeline?

Reply to any of the questions above, or say *"just fix it"* to skip coaching and operate from the Tier-1/2/3/4 roadmap directly.
