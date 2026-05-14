# Theory: Label Noise and Its Effect on Generalisation

## 1. Problem Setting

In supervised classification, we have training data $\{(x_i, y_i)\}_{i=1}^n$ drawn from a distribution $\mathcal{D}$ over $\mathcal{X} \times \{1, \ldots, K\}$. The goal is to learn a classifier $f: \mathcal{X} \to \{1, \ldots, K\}$ that minimises the **population risk**:

$$R(f) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(f(x), y)]$$

where $\ell$ is a loss function (e.g., 0–1 loss, cross-entropy).

In practice, we minimise the **empirical risk** over the training set:

$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(x_i), y_i)$$

**The problem**: When labels are noisy, we observe corrupted labels $\tilde{y}$ instead of true labels $y$. The learner minimises the *noisy* empirical risk, which is a biased estimator of the true risk.

---

## 2. Noise Transition Matrix

Label noise is modelled via a **transition matrix** $T \in \mathbb{R}^{K \times K}$ where:

$$T_{ij} = P(\tilde{y} = j \mid y = i)$$

$T$ is row-stochastic: $\sum_j T_{ij} = 1$ for all $i$.

### Symmetric (Uniform) Noise

Each label flips to any other class with equal probability:

$$T_{ij} = \begin{cases} 1 - \eta & \text{if } i = j \\ \frac{\eta}{K - 1} & \text{if } i \neq j \end{cases}$$

### Asymmetric (Class-Conditional) Noise

Flips are class-dependent. Example: in MNIST, `7` is often confused with `1`, but `0` is rarely confused with `5`. A simple model: class $i$ flips to class $(i+1) \mod K$ with probability $\eta$.

---

## 3. Risk Under Label Noise

The **noisy risk** can be decomposed as:

$$\tilde{R}(f) = \mathbb{E}_{(x, \tilde{y})}[\ell(f(x), \tilde{y})] = \sum_{j=1}^K \sum_{i=1}^K T_{ij} \cdot \mathbb{E}_{x|y=i}[\ell(f(x), j)]$$

In matrix form: $\tilde{R} = T^\top R$, where $R$ is the vector of class-conditional risks.

### Key Results

1. **0–1 loss is noise-tolerant under symmetric noise**: If $\eta < (K-1)/K$, then the minimiser of the noisy risk equals the Bayes-optimal classifier. This is because the 0–1 loss satisfies the "sum condition" $\sum_{j} \ell(f(x), j) = \text{const}$.

2. **Cross-entropy is NOT noise-tolerant**: The minimiser of noisy CE is biased away from the Bayes-optimal. The bias increases with $\eta$.

3. **Condition for noise tolerance**: A loss $\ell$ is noise-tolerant under symmetric noise if $\sum_{j=1}^K \ell(f(x), j) = C$ for some constant $C$ and all $f(x)$.

---

## 4. Backward Loss Correction (Patrini et al., 2017)

If we know the transition matrix $T$, we can define a **corrected loss**:

$$\tilde{\ell}_{\text{backward}}(f(x), \tilde{y}) = \sum_{i=1}^K (T^{-1})_{\tilde{y},i} \cdot \ell(f(x), i)$$

**Theorem**: $\mathbb{E}_{\tilde{y}|y}[\tilde{\ell}_{\text{backward}}(f(x), \tilde{y})] = \ell(f(x), y)$.

That is, the backward-corrected loss is an **unbiased estimator** of the clean loss. Training with this loss recovers the same minimiser as training on clean data.

**Requirement**: $T$ must be invertible, which holds if $\eta < 1 - 1/K$.

---

## 5. Generalised Cross-Entropy (Zhang & Sabuncu, 2018)

An alternative approach that doesn't require knowing $T$:

$$\ell_{\text{GCE}}(f(x), y) = \frac{1 - f_y(x)^q}{q}$$

where $f_y(x)$ is the predicted probability for class $y$, and $q \in (0, 1]$.

- **$q \to 0$**: Recovers standard cross-entropy (fast learning, noise-sensitive).
- **$q = 1$**: Recovers Mean Absolute Error (slow learning, noise-robust).
- **$q \approx 0.7$**: Good empirical trade-off.

GCE satisfies a relaxed form of noise tolerance and is more robust than CE without requiring $T$ estimation.

---

## 6. Transition Matrix Estimation via Anchor Points

**Problem**: In practice, $T$ is unknown. Can we estimate it from noisy data alone?

**Anchor-point method** (Patrini et al., 2017):

A sample $x$ is an **anchor point** for class $k$ if $P(y = k \mid x) \approx 1$.

At an anchor point: $P(\tilde{y} = j \mid x) \approx P(\tilde{y} = j \mid y = k) = T_{kj}$

**Algorithm**:
1. Train a model on noisy data.
2. For each class $k$, find examples where the model predicts $P(y=k|x)$ is highest.
3. The noisy-label distribution among these top-confident examples estimates $T[k, :]$.

**Assumptions**: Each class has at least one anchor point in the training set (mild assumption for large datasets).

---

## 7. Why Different Classifiers Respond Differently

### Logistic Regression
- **Linear decision boundary** → limited capacity to memorise noise.
- Acts as an implicit regulariser against label noise.
- Performance degrades gracefully.

### Decision Tree
- **Can shatter the training set** → memorises noisy labels.
- High variance → sensitive to every label flip.
- Shows the steepest accuracy degradation.

### MLP (Neural Network)
- **Medium capacity** with regularisation (dropout, batch norm).
- Can learn complex patterns but also memorise noise if over-trained.
- Benefits most from noise-robust training methods (backward correction, GCE).

---

## 8. References

1. Natarajan, N., Dhillon, I.S., Ravikumar, P.K., and Tewari, A. (2013). "Learning with noisy labels." *NeurIPS*.
2. Patrini, G., Rozza, A., Menon, A.K., Nock, R., and Qu, L. (2017). "Making deep neural networks robust to label noise: A loss correction approach." *CVPR*.
3. Zhang, Z. and Sabuncu, M.R. (2018). "Generalized cross entropy loss for training deep neural networks with noisy labels." *NeurIPS*.
4. Ghosh, A., Kumar, H., and Sastry, P.S. (2017). "Robust loss functions under label noise for deep neural networks." *AAAI*.
5. Han, B. et al. (2018). "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS*.
