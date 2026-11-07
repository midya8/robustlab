"""Exact rational L1/L-infinity audits of the binary float inputs actually stored."""
from fractions import Fraction
from .core import audit, dataset, model, threat


def _q(x):
    return Fraction.from_float(float(x))


def _record(q):
    return {'numerator': str(q.numerator), 'denominator': str(q.denominator)}


def exact_audit(payload, data, *, norm='linf', epsilon=0.0):
    """Rational proof records; the accepted numeric values mean exact binary floats."""
    w, b = model(payload)
    x, y = dataset(data, len(w))
    norm, eps = threat(norm, epsilon)
    if norm not in ('l1', 'linf'):
        raise ValueError('exact rational audit supports only l1 and linf, not l2')
    # Avoid unbounded large-integer arithmetic/output; checked before any Fraction.
    if len(w) * len(x) > 4096:
        raise ValueError('exact audit is capped at 4096 row-coordinate products')
    qw, qb, qe = tuple(_q(t) for t in w), _q(b), _q(eps)
    if norm == 'linf':
        dual = sum(abs(t) for t in qw)
        g = tuple((t > 0) - (t < 0) for t in qw)
    else:
        dual = max(abs(t) for t in qw)
        i = next(i for i,t in enumerate(qw) if abs(t) == dual)
        g = tuple(((t > 0) - (t < 0)) if j == i else 0 for j,t in enumerate(qw))
    floating = audit(payload, data, norm=norm, epsilon=eps)
    rows = []
    for row, label, fr in zip(x, y, floating['rows']):
        qx = tuple(_q(t) for t in row)
        score = sum(a*t for a,t in zip(qw,qx)) + qb
        clean = label * score
        worst = clean - qe*dual
        delta = tuple(-qe*label*t for t in g)
        attacked = tuple(t+d for t,d in zip(qx,delta))
        attained = label * (sum(a*t for a,t in zip(qw,attacked)) + qb)
        if attained != worst:
            raise ArithmeticError('internal witness identity failed')
        radius = max(Fraction(0), clean/dual) if dual else None
        exact_flag = worst > 0
        rows.append({'x': [_record(t) for t in qx], 'label': label,
                     'score': _record(score), 'prediction': 1 if score >= 0 else -1,
                     'clean_signed_margin': _record(clean),
                     'worst_signed_margin': _record(worst),
                     'strict_clean_correct': clean > 0, 'strict_robust_correct': exact_flag,
                     'radius_to_zero_margin': _record(radius) if radius is not None else None,
                     'radius_unbounded': dual == 0 and clean > 0,
                     'delta': [_record(t) for t in delta],
                     'attacked_x': [_record(t) for t in attacked],
                     'attained_signed_margin': _record(attained),
                     'float_worst_signed_margin': fr['worst_signed_margin'],
                     'float_strict_robust_correct': fr['strict_robust_correct'],
                     'float_prediction': fr['prediction'],
                     'strict_flag_disagrees': exact_flag != fr['strict_robust_correct'],
                     'prediction_disagrees': (1 if score >= 0 else -1) != fr['prediction']})
    return {'format': 'robustlab-exact-v1',
            'input_semantics': 'each validated number is its exact stored binary float value, not its decimal spelling',
            'model': {'version': 1, 'weights': [_record(t) for t in qw], 'bias': _record(qb)},
            'threat': {'norm': norm, 'epsilon': _record(qe), 'domain': 'all real coordinates; no box clipping'},
            'dual_weight_norm': _record(dual), 'rows': rows,
            'counts': {'total': len(rows), 'correct': sum(r['prediction'] == r['label'] for r in rows),
                       'strict_clean_correct': sum(r['strict_clean_correct'] for r in rows),
                       'strict_robust_correct': sum(r['strict_robust_correct'] for r in rows),
                       'strict_flag_disagreements': sum(r['strict_flag_disagrees'] for r in rows),
                       'prediction_disagreements': sum(r['prediction_disagrees'] for r in rows)},
            'boundary_rule': 'strict worst signed margin > 0; score tie predicts +1',
            'scope': 'exact rational arithmetic for this stored binary linear model and threat set; not a proof of a separately rounded execution pipeline'}


def exact_demo():
    cases = [
        ('float_false_positive', {'version': 1, 'weights': [.1,.2], 'bias': -.30000000000000004},
         {'x': [[1.1,1.1]], 'y': [1]}, .1),
        ('float_misses_positive_margin', {'version': 1, 'weights': [.1], 'bias': -.020000000000000004},
         {'x': [[.2]], 'y': [-1]}, 0.0),
        ('dyadic_exact_boundary', {'version': 1, 'weights': [.5], 'bias': 0},
         {'x': [[.5]], 'y': [1]}, .5),
    ]
    return {'cases': [{'name': name, 'input_model': m, 'input_data': d,
                       'input_epsilon': e, 'audit': exact_audit(m,d,epsilon=e)}
                      for name,m,d,e in cases],
            'selection': 'first false-positive in a documented finite decimal grid search, a directly constructed cancellation case and a dyadic control; not a frequency estimate',
            'scope': 'numeric boundary fixtures only; the original trained-model demo is unchanged'}
