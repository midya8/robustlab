# Design and audit notes

## All-real norm-ball audits

The original `audit`, `exact_audit` and `fit` paths use a binary affine score with an all-real norm ball. Each labeled row gets its own independent perturbation; the bias is fixed. These paths do not intersect the ball with a box, clip coordinates or add discretization, categorical encoding, normalization or preprocessing. The separate `box_audit` contract is described below. None of these paths is a probability model or neural network.

Hölder's inequality bounds the score change by epsilon times the dual weight norm. The selected perturbations attain that bound: a signed coordinate vertex for L1, the opposite weight direction for L2, and a signed corner for L-infinity. With zero weights the objective is constant and the zero perturbation attains it. Consequently the returned lower margin is an analytic minimum in real arithmetic, rather than a lower bound from an incomplete attack search. Float arithmetic can still round; the API states that caveat in every audit.

Positive signed margins mean correct for both labels. At equality, the implementation makes no strict certificate even though the deterministic +1 score-tie rule can classify a positive label correctly. It reports the actual attained margin and attacked coordinates so a missing strict certificate is not confused with a label-flipping attack.

For nonzero weights, positive-margin clean rows have an open certified-radius interval below margin/dual norm. A row already incorrect or on the boundary gets radius zero. Constant models have no finite zero-margin radius; separate `null` and `radius_unbounded` fields distinguish this from a clean error. No infinities or NaNs are serialized.

## Robust objective, not a post-hoc attack metric

The inner maximum of logistic loss is obtained by minimizing its signed margin, because softplus of the negative margin is decreasing in that margin. Thus each row uses `softplus(epsilon*dual_norm - y*score)`. Norm penalties outside the softplus would solve a different objective. A separate quadratic weight regularizer is explicit and does not include bias.

For a selected subgradient g of the dual norm, `d(loss)/d(w)` is `-sigmoid(-worst_margin)` times `(y*x - epsilon*g)`, plus `l2*w`. The bias derivative is `-sigmoid(-worst_margin)*y`. At nonsmooth points this is a selected subgradient of the convex objective, not a two-sided derivative. The zero vector is chosen for the Euclidean norm at zero, zero coordinates get sign zero for the L1 dual norm, and the first signed maximum coordinate is chosen for the infinity dual norm. All are valid choices.

Training is deterministic fixed-step full-batch descent from zero. It exposes the exact rate, update count, radius, regularizer and checkpoint stride. No convergence, exact optimum or monotonic-loss claim is made; constant steps can oscillate at kinks. The final loss is recomputed at the returned parameters, not the pre-update parameters. Evaluation does not choose a checkpoint, threshold, fixture or seed.

For n rows and d dimensions, an audit costs `O(n*d)` time and output space. Each optimizer update costs `O(n*d)`; memory is `O(n*d)` for validated input plus `O(d)` gradients and at most about 200 checkpoint records. Full audit witnesses are `O(n*d)`. The fit work cap is checked before any objective calculation. The 1-MiB CLI cap is applied by bounded binary reading, rather than a potentially racy file-size-only check.

## Fixed fixture and two gate directions

Training uses all products y in {-1,+1}, a in {.75,.9,1.05,1.2,-1}, b in {.16,.20,.24}, with `x=(y*a,y*b)`. Holdout uses a in {.8,.95,1.1,1.25,-1.05}, b in {.17,.21,.25}. Each split contains 30 rows; they are two supplied grids, not independent random draws or an estimate of population performance.

The same optimizer settings fit two arms; only training epsilon changes from 0 to .3. At holdout epsilon .3, ordinary training has clean/strict-robust accuracy 1.0/.2, while robust training has .8/.8. A predeclared .75 strict-robust gate fails on the ordinary arm and passes on the robust arm. Clean accuracy stays visible; the robust arm does not dominate it. Nothing removes the robust arm's tiny nonzero second weight or conditions publication on a favorable result.

This experiment is a controlled descriptive fixture comparison, not empirical evidence about images or a statistical claim about adversarial training. The radius sweep evaluates the fixed trained models, with no refitting at the holdout radii.

## Independent checks and provenance

The L-infinity reference enumerates all corner perturbations for dimensions 1..4, weights drawn from {-2,0,1}, and both labels: 240 model/label configurations. It uses Fraction arithmetic for the expected minima rather than the production dual-norm function. Separate L1 tests enumerate coordinate vertices; L2 checks use the 3-4-5 identity, perturbation length and 10,000 circle directions. The direction grid is a numerical reference, not an exact general optimizer.

Smooth-point central finite differences cover every weight and bias for every norm at epsilon 0 and .25 with explicit L2 regularization. Nonsmooth cases use the convex supporting-hyperplane inequality, not invalid central-derivative comparisons. Another reference directly maximizes logistic loss over L-infinity corners. Both final demo arms retain full attained-witness audits.

The renderer requires equality with the entire unchanged fixed demo and embeds it as metadata. Tests inspect numeric bar positions and all sweep coordinates, not just labels or an SVG file's existence. Visual inspection remains separate because parsers do not verify readability.

The mathematical problem and exact inner solutions follow [Kolter/Madry's tutorial](https://adversarial-ml-tutorial.org/adversarial_ml_slides_parts_2_3.pdf) and [arXiv:2208.09449, logistic Theorem 2](https://arxiv.org/html/2208.09449v3). This repository supplies its own code, tests, fixture and drawings. It neither incorporates external source text nor claims a new robustness method.

## Exact arithmetic is a separate contract

`exact_audit` preserves the same validated numeric inputs, but evaluates them as exact rationals via `Fraction.from_float`. It does not reinterpret JSON decimal spellings or accept rational-string inputs. Integers within the existing numeric bound convert to exactly representable floats; noninteger floats retain their actual binary ratios. This follows [Python's Fraction documentation](https://docs.python.org/3/library/fractions.html), including its explicit warning that `from_float(0.3)` differs from `3/10`.

L1 and L-infinity dual norms and attaining perturbations use only absolute values, sums, maxima and multiplication of rational values. All signed margins, attacked coordinates and zero-margin radii therefore remain rational. The certificate compares integer numerators with zero; serialized numerator/denominator strings preserve every bit and always have a positive denominator. A chosen L1 maximum coordinate uses the same deterministic first-tie convention, with no float norm used in its exact computation. The returned float audit is a comparison record, not a step in the rational certificate.

The fraction calculation independently constructs the exact score, margin and attack; it checks equality of the recomputed attained margin and the analytic worst margin. Tests independently enumerate vertices/corners instead of calling a production dual-norm helper, and reconstruct attack feasibility and its dot product from serialized fractions. There are 156 model/label/norm configurations in that finite family: dimensions 1..3, three weights per coordinate, two labels, and two supported norms. A subnormal-squared test returns a positive rational even when its float product is zero.

Resource bounds apply before fraction conversion: at most 4,096 row-coordinate products, in addition to the original 128-dimension/4,096-row bounds. Arithmetic/output operation counts are linear in that product count, but integer bit-operation costs grow with operand size; there is no constant-time rational claim. Binary64 inputs bound the operand sizes here, and the tests exercise the smallest subnormal. General L2 directions and margins can be irrational, so this API rejects L2 rather than wrapping a rounded square root in an exact label.

The selected false-positive fixture is the first matching case from the finite ordered grid described in README. This is intentional edge-case discovery, not a representative statistical experiment, and it does not alter the original two trained models or gate results. The exact SVG prints decimal approximations only for readability; its flags are computed from exact integer comparisons and its metadata carries the complete fractions.

This is an exact mathematical audit of the supplied linear function and real-valued threat ball. A separately rounded inference/attack pipeline is another function and is not certified. No broader security claim follows from exact arithmetic.

## Exact box intersection and a primal/dual certificate

The optional `box_audit` API has its own explicit bounds, not a global reinterpretation of `audit` or `exact_audit`. Require a nonempty closed coordinate interval at every coordinate and require every clean row to be inside it, including endpoints. All accepted numeric values become exact binary rationals. L1 and L-infinity are supported; L2 is not.

Let `s_i = -y*sign(w_i)`, `g_i = abs(w_i)`. A minimizer never needs movement opposite `s_i`: replacing such movement by zero preserves box feasibility (the clean point is in the box), uses no more norm budget and cannot worsen the minimization objective. Set zero-weight movement to zero. Thus movement amounts `t_i >= 0` have capacities `c_i = x_i-lower_i` when `s_i=-1`, or `upper_i-x_i` when `s_i=+1`. Minimize margin by maximizing gain `sum(g_i*t_i)`.

For L-infinity, constraints separate as `0 <= t_i <= min(c_i,epsilon)`, so fill every effective capacity. The gain upper bound is the sum of gain times effective capacity, attained by that witness. Equivalently use zero budget multiplier and upper multipliers `g_i`.

For L1, require `sum(t_i) <= epsilon` and `0 <= t_i <= c_i`. Sort coordinates by descending gain, then ascending index; fill capacity in order until budget or all useful capacity is exhausted. This is bounded fractional knapsack with unit movement costs, not a heuristic projection of the old attack. If budget is exhausted at a coordinate, choose its gain for `lambda`; if budget is zero, use the maximum gain; if budget remains after useful capacities fill, use zero. Set `mu_i = max(g_i-lambda,0)`. All are nonnegative and `g_i <= lambda+mu_i`. Therefore any feasible movement satisfies:

```text
sum(g_i*t_i)
  <= lambda*sum(t_i) + sum(mu_i*t_i)
  <= lambda*epsilon + sum(mu_i*c_i)
```

The chosen allocation attains this bound, including ties, a budget exactly equal to total useful capacity and degenerate capacities. The code asserts exact primal/dual equality, exact attained/minimum margin equality, box inclusion and norm feasibility before returning. The serialized certificate includes every allocation, effective upper capacity, budget multiplier, upper multiplier, primal gain, dual bound and zero gap. Returned rational numerators and positive denominators are strings. A consumer can reconstruct the inequalities using integers, without rerunning the sort.

The L1 clipping counterexample is directly specified: weights `(4,1)`, bias `-1/4`, clean point `(1/16,3/4)`, label `+1`, box `[0,1]^2`, epsilon `5/8`. Clean margin is `3/4`. The all-real witness concentrates budget on the first coordinate; clipping it leaves `(0,3/4)` and margin `1/2`. The optimal box attack uses movement `(1/16,9/16)`, lands at `(0,3/16)` and attains `-1/16`. Gain is `13/16`; a dual is `lambda=1`, `mu=(3,0)`, giving `5/8 + 3/16 = 13/16`. A positive candidate attack margin is not a lower bound. For L-infinity, coordinate-wise clipping of the adverse-direction all-real witness is optimal. The demo retains both comparisons and the radius sweep, plus both strict gate directions. At radius 2 the box saturates and unused budget is allowed. The original fitted models, demo and first two figures are unchanged.

An independent test oracle enumerates all sign orthants, each orthant's box corners, and intersections of the L1 budget plane with each box edge. These include the vertices of the orthant box intersected with the budget half-space. It minimizes the original signed affine score directly, not the gain-sorted routine. It checks 468 model/label/norm/budget configurations in dimensions 1..3. Other tests reconstruct the entire primal/dual certificate, witness and score; exercise asymmetric/degenerate/nonbinding boxes, both labels, zero/tied weights, saturated budget, binary/subnormal values, input/RNG isolation, actual 4,096-product execution, preflight ordering and strict CLI failures. Four SVG tests cover all 36 plotted points, six polylines, unchanged full metadata and reproduction. Suite total is 80 tests.

The product cap is checked after ordinary finite validation but before constructing any Fraction. With `n` rows and `d` coordinates, sorting takes `O(d log d)` rational comparisons once; each row takes `O(d)` rational operations, apart from integer bit complexity. Output and retained data take `O(n*d)` rational values. Existing per-value magnitude limits bound integer-part growth but not denominators of very small binary floats. Exact rational arithmetic can still be slower than the unchanged float API. The 4,096-product cap is an operational limit, not a timing guarantee.

No box-constrained training, general LP backend, nonlinear or multiclass verification, L2 box solver, decimal input semantics, calibrated risk or separately rounded execution certificate is claimed. The fractional-knapsack formulation is standard (Duke CS330 lecture 15, section 5.1: https://courses.cs.duke.edu/spring20/compsci330/lecture15scribe.pdf); the threat reduction, code, tests and illustrations here are original implementations.
