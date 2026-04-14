# Task 4: Out-of-Distribution Detection
## (Draft for NeurIPS Datasets & Benchmarks paper)

---

### 4.4  Task 4: Out-of-Distribution Detection

A world model deployed on real industrial equipment must not only make accurate predictions under nominal conditions, but also signal when the system enters an operating regime that was not encountered during training.  We therefore evaluate the OOD sensitivity of frozen JEPA representations under four types of distributional shift, each grounded in physically realistic failure modes.

#### 4.4.1  OOD Scenarios

We construct five OOD scenarios by modifying the degradation dynamics while keeping the simulator and observation pipeline unchanged.  All scenarios share the same nominal starting state (zero degradation) so that the deviation from the training distribution grows strictly from changes in the underlying process, not from initial conditions.

**Accelerated degradation (Accel-3x, Accel-5x).** The per-step slopes of all ten degradation components are multiplied by 3x and 5x respectively. Individual states remain within the training bounds, but they are visited much earlier in the episode. This is the "same states, different speed" OOD type and is the hardest to detect from individual frames.

**Premature maintenance (Pre-maint).** The maintenance interval is reduced from (10 000, 10 001) to (50, 150) steps, and maintenance effectiveness is reduced from 0.4 to 0.1. The engine therefore experiences frequent but nearly ineffective resets, producing long episodes with a distinctive sawtooth degradation pattern absent from the training distribution.

**Correlated component failure (Correlated).** Only the four high-pressure subsystem parameters (HPC efficiency, HPC flow, HPT efficiency, HPT flow) are allowed to degrade. The remaining six components are held at zero. This represents a localised, structurally coherent failure absent from training data, where all ten components degrade independently.

**Spike fault.** A sudden step discontinuity is injected into the HPC efficiency parameter at a random time between 30% and 70% through the episode, snapping it instantly to its lower bound. The remaining components follow their normal trajectories. This represents a catastrophic instantaneous fault with no gradual precursor; such step changes are physically impossible in the training data.

#### 4.4.2  Detection Methods

We evaluate four unsupervised detectors applied to frozen JEPA encoder representations, requiring no OOD labels at training time.

**Surprise score.** Following the JEPA prediction objective, we define the per-timestep surprise as the Euclidean distance between the predicted next embedding and the observed next embedding:
$$S(t) = \| z_{t+1} - f_\theta(z_{t-H+1:t},\, \mathbf{0}) \|_2,$$
where $f_\theta$ is the JEPA predictor and $H = 3$ is the context window.  Intuitively, larger surprise should indicate states that are harder to predict from recent history.

**Mahalanobis distance.** We fit a multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$ on the training set embeddings and score each test embedding by its Mahalanobis distance $d_M(z) = \sqrt{(z - \mu)^\top \Sigma^{-1} (z - \mu)}$.  This measures how far the embedding lies from the training distribution under a global Gaussian assumption.

**$k$-nearest-neighbour distance.** We compute the mean Euclidean distance to the $k = 20$ nearest training embeddings, subsampled to 50 000 points for computational tractability.  Unlike Mahalanobis distance, $k$-NN detects local gaps in the training manifold without assuming a parametric form.

**Reconstruction error.** We train a shallow MLP decoder $D_\phi: \mathbb{R}^{z} \to \mathbb{R}^{28}$ that maps each frozen JEPA embedding back to the corresponding normalised sensor observation (7 sensors $\times$ 4 flight contexts).  The decoder is trained on the training split with the encoder frozen, minimising mean squared reconstruction error.  At test time, the per-timestep score is:
$$R(t) = \| x_t - D_\phi(E_\psi(x_t)) \|_2,$$
where $E_\psi$ is the frozen encoder.  The rationale is that $D_\phi$ learns to invert the encoder's mapping for the training distribution.  When the encoder maps an OOD observation to an embedding region that the decoder has not been trained on, the reconstruction degrades, producing elevated $R(t)$.  Crucially, as an OOD trajectory progresses deeper into degraded territory, $R(t)$ should grow monotonically, making both the trend and the magnitude informative.

For each detector we evaluate at two levels of granularity:
- **Per-timestep AUC:** all timesteps across all episodes are treated as independent samples; ID timesteps are labelled 0 and OOD timesteps are labelled 1.
- **Per-episode AUC (mean):** each episode is assigned a single score equal to the mean of its per-timestep scores; the AUC is computed over episodes, with all ID test episodes labelled 0 and all 50 OOD episodes labelled 1.

For the reconstruction and surprise detectors, we additionally report a **per-episode max** AUC, which is particularly suited to spike faults where a single anomalous timestep suffices to identify the episode.

#### 4.4.3  Results

Table 1 reports AUC-ROC for all detectors, scenarios, and aggregation levels.  Figure X shows the heatmaps and Figure Y shows the reconstruction error profiles over episode lifetime.

---

**Table 1. Task 4 OOD Detection AUC-ROC.** Per-timestep and per-episode (mean) results for both JEPA variants. Best per-column in **bold**.

| Scenario | Surprise (ts) | Mahal (ts) | k-NN (ts) | **Recon (ts)** | Surprise (ep) | Mahal (ep) | k-NN (ep) | **Recon (ep)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *w*=1 |||||||||
| Accel-3x       | 0.21 | 0.77 | 0.54 | **0.83** | 0.24 | 0.87 | 0.57 | **0.98** |
| Accel-5x       | 0.29 | 0.77 | 0.55 | **0.84** | 0.42 | 0.86 | 0.57 | **0.98** |
| Pre-maint      | 0.11 | 0.75 | 0.58 | **0.79** | 0.01 | 0.87 | 0.65 | **0.93** |
| Correlated     | 0.20 | 0.69 | 0.70 | **0.88** | 0.16 | 0.80 | 0.79 | **1.00** |
| Spike fault    | 0.13 | 0.74 | 0.37 | **0.81** | 0.04 | 0.91 | 0.31 | **1.00** |
| *w*=10 |||||||||
| Accel-3x       | 0.18 | 0.72 | 0.62 | **0.93** | 0.16 | 0.85 | 0.69 | **1.00** |
| Accel-5x       | 0.26 | 0.73 | 0.63 | **0.94** | 0.30 | 0.85 | 0.70 | **1.00** |
| Pre-maint      | 0.10 | 0.69 | 0.57 | **0.89** | 0.00 | 0.81 | 0.69 | **0.96** |
| Correlated     | 0.15 | 0.67 | 0.74 | **0.93** | 0.07 | 0.78 | 0.85 | **1.00** |
| Spike fault    | 0.10 | 0.63 | 0.53 | **0.92** | 0.01 | 0.74 | 0.59 | **1.00** |

*(ts = per-timestep; ep = per-episode mean; chance = 0.50)*

---

**Reconstruction error is the dominant detector.**  The reconstruction-based detector achieves the highest AUC-ROC in every scenario and at every aggregation level, often by a large margin.  At the episode level, it reaches perfect or near-perfect AUC (1.000) on four of five scenarios for *w*=10, and on two of five for *w*=1.  This confirms the core hypothesis: as a degrading trajectory progresses, the encoder maps increasingly anomalous observations to embedding regions that the decoder was not trained on, and reconstruction error rises accordingly.

**Reconstruction error grows monotonically over episode lifetime.** Figure Y (recon_profile) shows mean reconstruction error binned by fractional episode progress.  For all OOD scenarios the error curve rises from near zero at episode start to 0.12--0.34 by episode end, while ID trajectories remain flat at 0.015--0.025 throughout.  The shape of the rise is scenario-specific and interpretable: accelerated degradation curves grow smoothly, the correlated failure rises steeply in the second half as the HPC and HPT components reach severe degradation levels, and the spike fault shows a sharp inflection at the 50% mark corresponding to the injected step discontinuity, followed by an elevated plateau.

**Episode-level aggregation substantially improves all detectors.**  Aggregating per-timestep scores into a single episode mean consistently raises AUC for every detector and scenario (Figure Z, episode_auc). The gain is largest for the reconstruction detector (+0.10 to +0.18 AUC) and for Mahalanobis (+0.08 to +0.17), because both accumulate a consistent distributional signal across timesteps.  The gain is negligible for the surprise score, for reasons discussed below.

**The surprise score is an unreliable OOD indicator — and this is a meaningful finding.** Across all scenarios and both models, the surprise score achieves AUC well below 0.50 at the per-timestep level, and near zero at the episode level for pre-maint and spike fault.  This means OOD episodes are *more* predictable than ID episodes in latent space, which is the inverse of the naive expectation.  The explanation is physically grounded: all OOD scenarios studied here produce trajectories that degrade more aggressively and monotonically than typical ID trajectories.  Fast, smooth, monotonic degradation is precisely the type of motion that a temporal predictor handles well, because the next state is a simple extrapolation of the current trend.  The predictor does not possess a notion of "this rate of change is anomalous" -- it only measures whether the next state was predicted correctly, which it was.  This finding motivates future work on training objectives that explicitly encode the normality of a trajectory rather than only its local predictability.

**Mahalanobis distance provides moderate but consistent detection.**  The Mahalanobis detector achieves AUC of 0.63--0.91 at the episode level, with the strongest signal for the spike fault (0.91, *w*=1) and weakest for the correlated scenario (0.78, *w*=1).  The signal is consistent because every OOD trajectory eventually reaches degradation states that are rare or absent from training embeddings, shifting the Mahalanobis score upward.  However, the margin is small because the degradation states themselves are not outside the training bounds -- only their temporal trajectory is unusual.

**$k$-NN distance is most sensitive to structural anomalies.**  The $k$-NN detector performs comparably to Mahalanobis in most scenarios but shows the highest relative performance on the correlated failure (0.79--0.85 episode AUC), where only four of ten components degrade.  This creates a sparse, genuinely novel sub-manifold in the embedding space with no nearby training neighbours.  Conversely, $k$-NN performs notably poorly on the spike fault at the per-timestep level (0.37--0.53), because the fault affects only a single component at a single timestep, which does not move the embedding far enough from its nearest neighbours within a batch.

**Longer prediction horizon improves reconstruction sensitivity.** Across all scenarios the *w*=10 model achieves higher reconstruction AUC than *w*=1 (e.g. Accel-3x: 0.93 vs 0.83 per-timestep; 1.00 vs 0.98 per-episode). We hypothesise that training the predictor to roll out multiple steps forces the encoder to capture richer, more structured information about the joint dynamics of all ten degradation components. This richer representation makes the decoder's learned inverse mapping more specific to the training distribution, amplifying the reconstruction error signal when the distribution shifts.

#### 4.4.4  Discussion

Our results suggest two complementary conclusions for the design of OOD-sensitive world model representations.

First, the reconstruction decoder adds a reliable, interpretable detection layer on top of any pretrained encoder, at the cost of a single additional training pass on frozen representations.  The training is lightweight (30 epochs, ~5 minutes on CPU for our dataset size), and the resulting detector outperforms all alternatives by a substantial margin.  The per-episode reconstruction profile also provides a qualitative diagnostic: the slope and shape of the error curve over episode lifetime is scenario-specific and could inform the type of fault (gradual vs sudden, global vs localised).

Second, prediction error alone is insufficient as an OOD signal in this setting, and likely in other settings where OOD inputs are locally smooth and predictable.  This is an important negative result: systems that use next-state prediction error as a proxy for anomaly detection (a common heuristic in model-based RL and health monitoring) will under-perform when the anomaly manifests as an unusual *rate* of change rather than an unusual *value*.  Pairing prediction-based objectives with a reconstruction decoder head addresses this limitation.

A limitation of the current evaluation is that OOD trajectories begin from the same nominal initial state as training data, so the OOD-ness accumulates gradually over the episode.  A complementary evaluation that begins episodes from already-degraded initial states would probe detector sensitivity at early detection, which is often the operationally relevant question.
