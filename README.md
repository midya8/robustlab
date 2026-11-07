# RobustLab

A dependency-free adversarial robustness lab for **binary linear classifiers**. Exact norm-ball worst-case margins, attainable perturbation witnesses, hand-derived robust logistic subgradients, and a deliberately visible clean/robust trade-off.

The point is to make the threat set and the limits inspectable. An analytic certificate for a linear score is not a certificate for a neural network, an image preprocessing pipeline, or an unspecified real-world attack.

![Clean and robust accuracy, with a radius sweep](assets/robustness-audit.svg)

## A gate that fails in one direction and passes in the other

Run from the repository root with Python 3.10+:

```sh
python -m robustlab demo
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/render_audit.py --check
```

Two full-batch models train on the same fixed 30 synthetic rows, from zero, for 4,000 steps at learning rate 0.1 with L2 coefficient 0.001. The only changed training setting is the L-infinity perturbation radius: 0 versus 0.30. There is no validation selection, best-seed reporting or test-driven stopping. Reported results use a distinct deterministic 30-row shifted factorial holdout, not independent random samples.

| Training objective | Holdout clean accuracy | Strict robust accuracy at L-infinity radius 0.30 | Robust gate >=75% |
|---|---:|---:|---|
| Ordinary logistic, radius 0 | 100% | 20% | Fail |
| Robust logistic, radius 0.30 | 80% | 80% | Pass |

This fixture was specified directly: balanced labels, a strong coordinate with one of five signs reversed, and a perfectly aligned but small-magnitude coordinate. The ordinary model can use the weak coordinate to resolve the strong coordinate's exceptions. A radius-0.30 adversary can overwhelm that weak coordinate. The robust objective instead mostly uses the strong coordinate, accepting the exceptions. This illustrates a trade-off on this constructed fixture, not a general theorem that robust training must reduce clean accuracy.

The demo returns both models, training settings and sampled loss histories, all train and holdout row-level attacks, holdout radius sweeps, all input rows and gate booleans. Sweep summaries omit duplicated row records; use `audit` at any radius to recover them. Both arms remain visible regardless of gate outcomes. Robust training is fixed-step subgradient descent, not a claim of convergence to a global minimizer.

## What is exact

For score `s(x) = w.x + b`, label `y` in `{-1,+1}`, and a perturbation satisfying `||delta||_p <= epsilon`, the minimum signed margin is:

```text
m_worst = y (w.x + b) - epsilon ||w||_q
1/p + 1/q = 1
```

| Input perturbation ball | Dual weight norm | A returned worst-case perturbation |
|---|---|---|
| L1 | max absolute weight | Put the full budget on the first maximum-absolute-weight coordinate, opposite `y*w` |
| L2 | Euclidean weight norm | `-epsilon*y*w / EuclideanNorm(w)` |
| L-infinity | sum of absolute weights | `-epsilon*y*sign(w)` |

Zero weights get a zero perturbation. At L1 ties, the first maximizing coordinate is chosen deterministically. These witnesses attain the lower margin in real arithmetic. The input domain is **all real coordinates**, without feature boxes or clipping. Bias is not perturbed. The implementation evaluates the identity in floating point; it is not an outward-rounded or interval-arithmetic proof. Near-boundary rounding must not be used as a safety guarantee.

`strict_robust_correct` means the worst signed margin is **strictly positive**. Equality fails conservatively for either label. Prediction itself uses `+1` at a score of zero, so ordinary accuracy and strict margin accuracy can differ at ties. A false strict flag is not automatically a misclassification witness: the attained margin can be exactly zero, where the prediction tie rule matters.

`radius_to_zero_margin` is `max(0, y*s/||w||_q)` for nonzero weights. A clean error has radius zero. With zero weights it is `null` because no perturbation changes the score; `radius_unbounded` distinguishes a strictly correct constant classifier from an incorrect or boundary one. Radius values specify the zero-margin boundary, not an assertion of correctness at that boundary.

## API and strict JSON CLI

```python
from robustlab import audit, fit, objective_gradient

model = {"version": 1, "weights": [3.0, 4.0], "bias": 0.1}
data = {"x": [[0.7, -0.2], [-0.4, 0.3]], "y": [1, -1]}
report = audit(model, data, norm="l2", epsilon=0.4)
trained = fit(data, norm="linf", epsilon=0.3, steps=1000, rate=0.1, l2=0.001)
gradient = objective_gradient(model, data, norm="linf", epsilon=0.3, l2=0.001)
```

```sh
python -m robustlab audit examples/model.json examples/data.json --norm l2 --epsilon 0.4
python -m robustlab fit examples/data.json --norm linf --epsilon 0.3 --steps 1000
```

`audit` returns the normalized model and threat specification, dual norm, per-row scores/predictions/clean and worst margins, flags, radii, perturbations, attacked coordinates and actually attained signed margins. `fit` returns a model inside its result; extract the `model` object to pass to `audit`. It never silently treats the whole training result as a model.

The model requires exactly `version`, `weights`, `bias`; version is integer 1. Data requires exactly `x`, `y`; labels are integer -1 or +1, not booleans or floats. Up to 128 dimensions and 4,096 rows; input numbers are finite with magnitude at most 1,000,000; epsilon and L2 coefficient are nonnegative within that same bound. No implicit rescaling or feature normalization is performed. A radius is measured in the provided coordinate system, so different feature units change its meaning.

Training allows 1..20,000 steps, learning rate in `(0,10]`, and preflights a 20-million row-coordinate-step work cap before evaluating its objective. History includes step zero, checkpoints every `max(1, steps//100)` updates, and the final objective. These are checkpoints, not every update for long runs. Parameter updates outside finite magnitude bounds stop with an error, not a partially returned model.

JSON files are capped at 1 MiB and reject duplicate keys, nonfinite constants, unknown fields and dimensional inconsistencies. CLI errors exit 2 with no partial JSON or traceback. APIs return fresh records and do not mutate the inputs or the global random state. Runtime and tests use only Python's standard library.

## Objective and gradient audits

The loss is the mean of `softplus(-y*s + epsilon*dual_norm(w))` plus `l2/2 * sum(w_j**2)`. The robustness term is **inside** the softplus, not an ordinary norm regularizer pasted outside it. Stable softplus and sigmoid calculations avoid exponent overflow on extreme scores. L2 regularization excludes the bias.

The selected norm subgradient is `sign(w)` for L-infinity input balls, normalized `w` for L2, and one signed maximum coordinate for L1. At zero or tied weights, these are valid subgradients, not claims of differentiability. Fixed learning rates can oscillate near a kink. The robust demo's small near-zero second weight remains reported rather than clipped away or presented as exact sparsity.

35 tests cover:

- All L-infinity corners for 240 small model/label configurations, using independent rational arithmetic; separate L1-vertex references and an L2 Cauchy/10,000-direction circle check.
- Hand-derived gradients against central finite differences for all three norms away from kinks, plus convex subgradient inequalities at zero/tied weights and a corner-enumerated robust-loss reference.
- Label/bias signs, zero weights, strict boundary ties, positive-scaling-invariant radii, monotone radius sweeps, threat-set distinctions and no-clipping witnesses.
- Extreme scores, exact first optimizer update/final objective, deterministic training, work-cap preflight, explosive-update rejection, invalid schemas, dimension/row boundaries, input/RNG isolation and CLI failure behavior.
- Both demo gate directions and every holdout witness. Four figure tests check all four bar coordinates, all 14 sweep points, full unchanged metadata, byte regeneration, input preservation and rejection of altered fits/attacks/fixtures.

The references verify finite test families; they are not a machine-checked proof for arbitrary floating-point inputs. The SVG is generated from the complete unchanged demo, with the full fixture, models, settings, histories and audits embedded as metadata.

## Sources and limits

This is an independent implementation of standard norm duality and robust linear logistic training, following the linear-case discussion in [Kolter and Madry's Adversarial Robustness tutorial](https://adversarial-ml-tutorial.org/adversarial_ml_slides_parts_2_3.pdf) and the logistic inner problem in [On Exact Solutions of the Inner Optimization Problem of Adversarial Robustness, Theorem 2](https://arxiv.org/html/2208.09449v3). Code, fixture, tests and figure are original to this repository; no external dataset or implementation is bundled.

No nonlinear verification, multiclass classifier, perceptual metric, feature-box constraints, adaptive attack suite, distributional guarantee, calibrated uncertainty or production security claim. No paid service, model download, API key, GPU or network access is needed. MIT licensed.
