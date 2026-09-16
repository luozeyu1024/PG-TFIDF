# Metric definitions

## Classification and open-set inference

For cell (i), the classifier predicts \(\hat{y}_i=\arg\max_c p_{ic}\). Closed-set evaluation reports accuracy, macro F1, weighted F1, multiclass Matthews correlation coefficient (MCC), macro one-vs-rest AUC, and one-vs-rest per-class metrics.

Maximum softmax probability is

\[
\operatorname{MSP}_i=\max_c p_{ic}.
\]

With threshold \(\tau=0.5\), the open-set inference script assigns `unknown` when \(\operatorname{MSP}_i<\tau\). `unknown_rate` is the rejected fraction; `known_acc_on_accepted` is accuracy among accepted known-class cells.

## Clustering

Silhouette values are computed from the original embedding. For cell \(i\),

\[
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}},
\]

where \(a(i)\) is the mean within-label distance and \(b(i)\) is the minimum mean distance to another label. The scaled score and AvgBIO are

\[
\operatorname{ASW}_{\mathrm{scaled}}=\frac{\operatorname{ASW}+1}{2},
\qquad
\operatorname{AvgBIO}=\frac{\operatorname{ASW}_{\mathrm{scaled}}+\operatorname{NMI}+\operatorname{ARI}}{3}.
\]

Louvain partitions a 15-nearest-neighbor connectivity graph. The script scans its resolution grid and retains the partition with maximum NMI against the reference labels. K-means uses \(K\) equal to the number of reference classes, `n_init=10`, and seed 42.

## Prediction agreement

Cohen's kappa between two model label vectors is

\[
\kappa=\frac{p_o-p_e}{1-p_e},
\]

where \(p_o\) is observed agreement and \(p_e\) is agreement expected from the marginal label frequencies. Correctness kappa applies the same equation to \(I(\hat y_i^{(m)}=y_i)\).

## Prior-adjusted biological gain

For nuisance group \(g\), class \(c\), and Laplace smoothing \(\alpha=1\),

\[
\pi_{gc}=\frac{n_{gc}+\alpha}{n_g+\alpha |\mathcal C|}.
\]

The multiclass Brier score, negative log-likelihood, and prior-adjusted gain (PAG) are

\[
\operatorname{Brier}=\frac{1}{N}\sum_i\sum_c(p_{ic}-I[y_i=c])^2,
\]

\[
\operatorname{NLL}_{\mathrm{model}}=-\frac{1}{N}\sum_i\log(p_{i,y_i}+\epsilon),
\]

\[
\operatorname{PAG}=\frac{1}{N}\sum_i\log\frac{p_{i,y_i}+\epsilon}{\pi_{g_i,y_i}+\epsilon}
=\operatorname{NLL}_{\mathrm{prior}}-\operatorname{NLL}_{\mathrm{model}}.
\]

The implementation also reports macro-PAG by averaging class-specific PAG values.

## Marker-signature similarity

RNA markers require adjusted \(P<0.05\), positive log fold change, `pct.1 >= 0.5`, `pct.2 <= 0.3`, and `pct.1 - pct.2 >= 0.2`. ADT markers require adjusted \(P<0.05\) and positive mean difference. Each cluster-method signature contains its first 10 ranked passing markers.

For signatures \(A\) and \(B\),

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

For models \(M\) and \(N\), directional and symmetric best-hit scores are

\[
S_{M\to N}=\frac{1}{|M|}\sum_{A\in M}\max_{B\in N}J(A,B),
\qquad
S_{\mathrm{sym}}=\frac{S_{M\to N}+S_{N\to M}}{2}.
\]

The combined score is the weighted mean of available RNA and ADT scores, with weights 0.5 and 0.5 by default. If one modality is unavailable, the available weight is renormalized.

## Gene-cluster AMI

For every model pair, AMI is computed on the shared gene IDs using `adjusted_mutual_info_score(..., average_method="arithmetic")`. Fewer than 10 shared genes returns a missing value. A zero diagonal is a display convention, not self-agreement estimation.

