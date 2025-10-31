# Reinforcement Learning Math Notes for LLM Training

These notes condense the math needed to implement the reinforcement learning (RL) algorithms described in the following papers:
- **ProRL** (Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models, arXiv:2505.24864v1)
- **DAPO** (DAPO: An Open-Source LLM Reinforcement Learning System at Scale, arXiv:2503.14476v2)
- **VinePPO** (VinePPO: Refining Credit Assignment in RL Training of LLMs, arXiv:2410.01679v2)

The equations retain each paper's notation so you can cross-reference easily.

## 1. Setup & Notation
- Prompts \(x\sim\mathcal{D}\), intra-episode states \(s_t\), actions/tokens \(a_t\), and generated completions \(y\) form trajectories \(\tau=(s_0,a_0,\dots,s_T)\).
- Policies: online policy \(\pi_\theta\), behavior/old policy \(\pi_{\theta_{\text{old}}}\), and reference policy \(\pi_{\text{ref}}\).
- Rewards \(\mathcal{R}(x;y)\) are task-specific (verifiable rule-based in DAPO, BoN/variance-reduced credits in VinePPO).

General KL-regularized RL objective (VinePPO Eq. (1), ProRL Eq. (4)):

$$
J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}[\mathcal{R}(x;y)]-\beta\,\mathrm{KL}[\pi_\theta\,\Vert\,\pi_{\text{ref}}].
$$

- ProRL keeps \(\beta>0\) and periodically resets \(\pi_{\text{ref}}\) to the current policy to avoid KL dominance while maintaining entropy.
- DAPO sets \(\beta=0\) when long-chain-of-thought divergence from the initialization is desirable.

## 2. Policy Gradient & Surrogates
### 2.1 Vanilla Policy Gradient (VinePPO Eq. (2))
$$
\mathbf{g}_{\mathrm{pg}}=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A(s_t,a_t)\right].
$$

### 2.2 Probability Ratios (DAPO Eq. (6))
$$
r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,\,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,\,o_{i,<t})}.
$$

### 2.3 PPO-Style Surrogates
Standard PPO surrogate (DAPO Eq. (1)):
$$
\mathcal{L}_{\mathrm{PPO}}(\theta)=\mathbb{E}\left[\min\bigl(r_t\hat{A}_t,\; \text{clip}(r_t,1-\varepsilon,1+\varepsilon)\hat{A}_t\bigr)\right].
$$

GRPO surrogate over grouped rollouts (ProRL Eq. (1), DAPO Eq. (5)):
$$
\mathcal{L}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\min\bigl(r_\theta(\tau)A(\tau),\; \text{clip}(r_\theta(\tau),1-\varepsilon,1+\varepsilon)A(\tau)\bigr)\right].
$$

DAPO decoupled clip with dynamic bandwidths (DAPO Eq. (8)/(10)):
$$
\mathcal{J}_{\mathrm{DAPO}}(\theta)=\mathbb{E}_{(q,a),\{o_i\}_{i=1}^G}\Biggl[\frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}\min\Bigl(r_{i,t}\hat{A}_{i,t},\;\text{clip}\bigl(r_{i,t},\,1-\varepsilon_{\text{low}},\,1+\varepsilon_{\text{high}}\bigr)\hat{A}_{i,t}\Bigr)\Biggr].
$$
- Token-level weighting \(1/\sum_i|o_i|\) balances gradient contributions across sequences of different length.
- Asymmetric \((\varepsilon_{\text{low}},\varepsilon_{\text{high}})\) (ProRL Eq. (3)) mitigates entropy collapse by permitting larger boosts on good tokens than penalties on bad ones.

Dynamic sampling ensures a mixture of correct and incorrect completions (DAPO Eq. (8)):
$$
0 < \bigl|\{o_i\mid\texttt{is\_equivalent}(a,o_i)\}\bigr| < G.
$$

## 3. Advantage Estimation & Normalization
### 3.1 Generalized Advantage Estimation (DAPO Eq. (2), Eq. (3))
$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}=\sum_{\ell=0}^{\infty}(\gamma\lambda)^{\ell}\,\delta_{t+\ell},\quad \delta_{t}=R_{t}+\gamma V(s_{t+1})-V(s_t).
$$

### 3.2 Group-Level Normalization (ProRL Eq. (2), DAPO Eq. (4)/(9))
For trajectory \(\tau\) grouped with peers \(G(\tau)\):
$$
A(\tau)=\frac{R_{\tau}-\operatorname{mean}(\{R_i\}_{i\in G(\tau)})}{\operatorname{std}(\{R_i\}_{i\in G(\tau)})}.
$$
Token-level variant:
$$
\hat{A}_{i,t}=\frac{R_i-\operatorname{mean}(\{R_j\}_{j=1}^{G})}{\operatorname{std}(\{R_j\}_{j=1}^{G})}.
$$
This shared normalization supplies a low-variance credit signal when multiple responses per prompt are available.

### 3.3 Monte Carlo Baselines (VinePPO Appendix Eq. (5)–(7))
$$
\hat{V}_{\mathrm{MC}}(s_t)=\frac{1}{K}\sum_{k=1}^K R(\eta_k),\quad
\hat{A}_{\mathrm{MC}}(s_t,a_t)=r(s_t,a_t)+\gamma\hat{V}_{\mathrm{MC}}(s_{t+1})-\hat{V}_{\mathrm{MC}}(s_t),
$$
with terminal reward assignment
$$
r_t=r(s_t,a_t)=
\begin{cases}
\mathcal{R}(x;y) & t=T-1,\\
0 & \text{otherwise.}
\end{cases}
$$

### 3.4 Truncated Returns for Bias–Variance Trade-off (VinePPO Eq. (17))
$$
\hat{A}^{(k)}_t=\sum_{\ell=0}^{k-1}\gamma^{\ell}\delta_{t+\ell}=r_t+\gamma r_{t+1}+\dots+\gamma^{k-1}r_{t+k-1}+\gamma^{k}\hat{V}_{\phi}(s_{t+k})-\hat{V}_{\phi}(s_t).
$$

## 4. Value Function Training (VinePPO Eq. (4), Eq. (11)–(12))
Value regression with clipped targets stabilizes critic updates:
$$
\mathcal{L}_V(\phi)=\mathbb{E}_{\tau\sim\pi_\theta}\Biggl[\frac{1}{2T}\sum_{t=0}^{T-1}\max\Bigl(\bigl\|\hat{V}_\phi(s_t)-G_t\bigr\|^2,\; \bigl\|\operatorname{clip}(\hat{V}_\phi(s_t),\hat{V}_{\phi_k}(s_t)-\epsilon',\hat{V}_{\phi_k}(s_t)+\epsilon')-G_t\bigr\|^2\Bigr)\Biggr].
$$

## 5. Reward Specification & Shaping
- **Rule-based outcome reward** (DAPO Eq. (7)):
  $$
  R(\hat{y},y)=\begin{cases}1,& \texttt{is\_equivalent}(\hat{y},y)\\ -1,& \text{otherwise}\end{cases}
  $$
  Easily extensible to partial credit by altering the indicator rule.
- **Overlength shaping** (DAPO Eq. (13)) to discourage cached truncation artifacts:
  $$
  R_{\text{length}}(y)=
  \begin{cases}
  0,& |y|\le L_{\text{max}}-L_{\text{cache}}\\
  \dfrac{(L_{\text{max}}-L_{\text{cache}})-|y|}{L_{\text{cache}}},& L_{\text{max}}-L_{\text{cache}}<|y|\le L_{\text{max}}\\
  -1,& |y|>L_{\text{max}}
  \end{cases}
  $$
  Combine with the outcome reward before normalization to keep signal consistent across groups.

## 6. Regularization & Stability Tools
- **KL control**: retain \(\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\text{ref}})\) (ProRL Eq. (4), VinePPO Eq. (8)) when you need to bound divergence. A per-token Monte Carlo estimator keeps the penalty cheap (VinePPO Eq. (10)):
  $$
  \hat{\mathrm{KL}}(\theta)=\frac{\pi_{\text{ref}}(a_t\mid s_t)}{\pi_\theta(a_t\mid s_t)}-\log\frac{\pi_{\text{ref}}(a_t\mid s_t)}{\pi_\theta(a_t\mid s_t)}-1.
  $$
- **Reference resets**: ProRL periodically hard-resets \(\pi_{\text{ref}}\leftarrow\pi_\theta\) to prevent the KL term from freezing learning while still preserving entropy.
- **Entropy preservation**: the asymmetric clip window \([1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}}]\) allows larger positive updates, slowing entropy collapse (ProRL Eq. (3)).
- **Token-level loss balancing** (DAPO): decouple per-token gradients so that long completions do not dominate training.
- **Bridging to supervised MLE**: VinePPO shows REINFORCE aligns with maximizing log-likelihood on reward-1 examples (Eq. (19)), which can initialize or regularize training toward high-reward data.

## 7. Implementation Checklist
1. **Collect rollouts**: sample \(G\) completions per prompt under \(\pi_{\theta_{\text{old}}}\); ensure mixed correctness via the dynamic sampling constraint.
2. **Score rewards**: evaluate task reward plus any shaping terms (length penalty, coverage bonuses, etc.).
3. **Normalize credits**: compute group-normalized advantages, optionally blend with GAE or truncated returns for better credit assignment.
4. **Critic update**: regress \(V_\phi\) using clipped targets and Monte Carlo baselines.
5. **Actor update**: compute ratios, apply decoupled clip with asymmetric bounds, and optimize \(\mathcal{J}_{\mathrm{DAPO}}\) (or \(\mathcal{L}_{\mathrm{GRPO}}\) with a KL term if using ProRL-style regularization).
6. **Stabilize**: monitor entropy and KL; adjust \((\varepsilon_{\text{low}},\varepsilon_{\text{high}})\), \(\beta\), or trigger a reference reset when the KL penalty overwhelms the policy gradient.
7. **Iterate**: refresh \(\pi_{\theta_{\text{old}}}\leftarrow\pi_\theta\) and repeat.

These formulas form a minimal yet complete toolkit for reproducing the RL algorithms described in ProRL, DAPO, and VinePPO. Mix and match modules (advantage normalization, asymmetric clipping, KL resets, value clipping) according to the stability/quality trade-offs your RL pipeline requires.
