"""Binary linear robustness under explicit, unbounded-domain norm balls."""
import math

NORMS = ('l1', 'l2', 'linf')


def number(v, name, *, minimum=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(name + ' must be a finite number')
    if abs(v) > 1e6 or not math.isfinite(v) or (minimum is not None and v < minimum):
        raise ValueError(name + ' is outside its finite allowed range')
    return float(v)


def vector(v, name):
    if not isinstance(v, (list, tuple)) or not 1 <= len(v) <= 128:
        raise ValueError(name + ' needs 1..128 coordinates')
    return tuple(number(x, name) for x in v)


def model(payload):
    if not isinstance(payload, dict) or set(payload) != {'version', 'weights', 'bias'}:
        raise ValueError('model requires exactly version, weights, bias')
    if type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('model version must be integer 1')
    w = vector(payload['weights'], 'weights')
    b = number(payload['bias'], 'bias')
    return w, b


def dataset(payload, d):
    if not isinstance(payload, dict) or set(payload) != {'x', 'y'}:
        raise ValueError('data requires exactly x and y')
    x, y = payload['x'], payload['y']
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= 4096:
        raise ValueError('data needs 1..4096 rows')
    if not isinstance(y, (list, tuple)) or len(x) != len(y):
        raise ValueError('x/y lengths must agree')
    rows = tuple(vector(row, 'x') for row in x)
    if any(len(row) != d for row in rows):
        raise ValueError('data dimensions must match weights')
    if any(type(t) is not int or t not in (-1, 1) for t in y):
        raise ValueError('labels must be integer -1 or +1')
    return rows, tuple(y)


def threat(norm, epsilon):
    if norm not in NORMS:
        raise ValueError('norm must be l1, l2 or linf')
    return norm, number(epsilon, 'epsilon', minimum=0)


def _dual(w, norm):
    """Dual norm and a selected subgradient; canonical first max for l1."""
    sign = lambda t: float((t > 0) - (t < 0))
    if norm == 'linf':
        return math.fsum(abs(t) for t in w), tuple(sign(t) for t in w)
    if norm == 'l2':
        n = math.hypot(*w)
        return n, tuple(t / n if n else 0.0 for t in w)
    n = max(abs(t) for t in w)
    i = next(i for i, t in enumerate(w) if abs(t) == n)
    return n, tuple(sign(t) if j == i else 0.0 for j, t in enumerate(w))


def _softplus(t):
    return max(t, 0.0) + math.log1p(math.exp(-abs(t)))


def _negative_sigmoid(t):
    # -sigmoid(-t), stable for arbitrary finite margins.
    if t >= 0:
        z = math.exp(-t)
        return -z / (1 + z)
    return -1 / (1 + math.exp(t))


def _objective(w, b, x, y, norm, eps, l2):
    dual, g = _dual(w, norm)
    loss, gw, gb = [], [0.0] * len(w), []
    for row, label in zip(x, y):
        margin = label * (math.fsum(a*t for a, t in zip(w, row)) + b) - eps * dual
        loss.append(_softplus(-margin))
        scale = _negative_sigmoid(margin) / len(x)
        gb.append(scale * label)
        for j in range(len(w)):
            gw[j] += scale * (label * row[j] - eps * g[j])
    return (math.fsum(loss) / len(x) + .5 * l2 * math.fsum(t*t for t in w),
            tuple(t + l2*a for t, a in zip(gw, w)), math.fsum(gb))


def objective_gradient(payload, data, *, norm='linf', epsilon=0.0, l2=0.0):
    """Robust logistic mean loss + l2/2 ||w||_2^2 and selected subgradient."""
    w, b = model(payload)
    x, y = dataset(data, len(w))
    norm, eps = threat(norm, epsilon)
    l2 = number(l2, 'l2', minimum=0)
    loss, gw, gb = _objective(w, b, x, y, norm, eps, l2)
    return {'loss': loss, 'weights_gradient': list(gw), 'bias_gradient': gb,
            'gradient_kind': 'selected subgradient; ordinary gradient away from norm kinks'}


def audit(payload, data, *, norm='linf', epsilon=0.0):
    """Return exact-in-real-arithmetic worst signed margins and witnesses."""
    w, b = model(payload)
    x, y = dataset(data, len(w))
    norm, eps = threat(norm, epsilon)
    dual, g = _dual(w, norm)
    rows = []
    for row, label in zip(x, y):
        score = math.fsum(a*t for a, t in zip(w, row)) + b
        clean = label * score
        worst = clean - eps * dual
        delta = [-eps * label * t for t in g]
        attacked = [a + t for a, t in zip(row, delta)]
        attained = label * (math.fsum(a*t for a, t in zip(w, attacked)) + b)
        # Zero weights: a strictly positive constant signed margin never reaches zero.
        radius = (max(0.0, clean / dual) if dual else None)
        rows.append({'x': list(row), 'label': label, 'score': score,
                     'prediction': 1 if score >= 0 else -1,
                     'clean_signed_margin': clean, 'worst_signed_margin': worst,
                     'strict_clean_correct': clean > 0,
                     'strict_robust_correct': worst > 0,
                     'radius_to_zero_margin': radius,
                     'radius_unbounded': dual == 0 and clean > 0,
                     'delta': delta, 'attacked_x': attacked,
                     'attained_signed_margin': attained})
    return {'model': {'version': 1, 'weights': list(w), 'bias': b},
            'threat': {'norm': norm, 'epsilon': eps, 'domain': 'all real coordinates; no box clipping'},
            'dual_weight_norm': dual, 'rows': rows,
            'accuracy': sum(r['prediction'] == r['label'] for r in rows) / len(rows),
            'strict_clean_accuracy': sum(r['strict_clean_correct'] for r in rows) / len(rows),
            'strict_robust_accuracy': sum(r['strict_robust_correct'] for r in rows) / len(rows),
            'boundary_rule': 'strict signed margin > 0; equality fails conservatively, even when +1 tie prediction is correct',
            'numerics': 'analytic identity evaluated in float64, not a rounded/interval formal certificate'}


def fit(data, *, norm='linf', epsilon=0.0, l2=0.001, steps=1000, rate=0.1):
    """Fixed-step full-batch subgradient descent from zero, no model selection."""
    if not isinstance(data, dict) or not isinstance(data.get('x'), (list, tuple)) or not data['x']:
        raise ValueError('nonempty data required')
    d = len(vector(data['x'][0], 'x'))
    x, y = dataset(data, d)
    norm, eps = threat(norm, epsilon)
    l2, rate = number(l2, 'l2', minimum=0), number(rate, 'rate', minimum=0)
    if not 0 < rate <= 10:
        raise ValueError('rate must be in (0,10]')
    if type(steps) is not int or not 1 <= steps <= 20000:
        raise ValueError('steps must be integer 1..20000')
    if len(x) * d * steps > 20000000:
        raise ValueError('training exceeds 20 million row-coordinate steps')
    w, b, history = (0.0,) * d, 0.0, []
    for step in range(steps):
        loss, gw, gb = _objective(w, b, x, y, norm, eps, l2)
        if step % max(1, steps // 100) == 0:
            history.append({'step': step, 'loss': loss})
        w = tuple(a - rate*t for a, t in zip(w, gw))
        b -= rate * gb
        if any(not math.isfinite(a) or abs(a) > 1e6 for a in (*w, b)):
            raise ValueError('training left finite parameter bounds; lower rate')
    final, _, _ = _objective(w, b, x, y, norm, eps, l2)
    history.append({'step': steps, 'loss': final})
    return {'model': {'version': 1, 'weights': list(w), 'bias': b},
            'training': {'norm': norm, 'epsilon': eps, 'l2': l2, 'steps': steps, 'rate': rate,
                         'initialization': 'zero', 'history_stride': max(1, steps // 100), 'optimizer': 'fixed-step full-batch subgradient descent'},
            'history': history, 'convergence_claim': False}
