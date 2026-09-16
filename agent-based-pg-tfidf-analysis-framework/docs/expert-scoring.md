# Expert scoring and pooling

Three independent experts score Results_1–Results_6 for Dataset81 and
SKCM_GSE115978_aPD1. Results_1–3 belong to Kimi Code and Results_4–6 belong to
Codex.

## Rubric

1. Does the TF-IDF interpretation support the final biological model? `1` supportive, `2` relatively supportive, `3` partially conflicting, `4` not supportive.
2. Does the RNA/ADT marker interpretation support the final biological model? Same `1–4` scale.
3. Does the final model match the mechanism described in the source publication? `Yes/No`.
4. Does the integrated analysis provide an initial biological understanding worth further computational or experimental exploration? `Yes/No`.
5. Are the TF-IDF, marker and final-model interpretations intuitive and clear? `1` clear, `2` relatively clear, `3` not particularly clear, `4` missing or difficult to locate.

## Pooling definition

For dataset $d$, agent family $a$, criterion $c$ and score category $s$, the
plotted count is

$$
n_{dacs}=\sum_{e=1}^{3}\sum_{r=1}^{3}\mathbf{1}(y_{d,a,e,r,c}=s),
$$

and the displayed fraction is

$$
p_{dacs}=\frac{n_{dacs}}{9}.
$$

Each donut therefore pools nine ratings. It does not average ordinal scores
and does not treat the six reports as technical replicates of one deterministic
program. Non-conforming entries are retained in the ambiguity audit rather
than imputed.

## Non-distribution boundary

The expert-scoring script, expert forms, individual and pooled score results,
and agent outputs are not included in the public module. Only the evaluation
design and pooling equations are documented here.
