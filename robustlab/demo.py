"""Fixed hand-designed trade-off, no fixture or seed search."""
from .core import audit, fit


def fixture(holdout=False):
    # Factorial, balanced labels; one of five robust-coordinate signs is reversed.
    strong = (0.75, 0.9, 1.05, 1.2, -1.0) if not holdout else (0.8, 0.95, 1.1, 1.25, -1.05)
    weak = (0.16, 0.20, 0.24) if not holdout else (0.17, 0.21, 0.25)
    rows = [(y, [y*a, y*b]) for y in (-1, 1) for a in strong for b in weak]
    return {'x': [x for y, x in rows], 'y': [y for y, x in rows]}


def demo():
    train, test = fixture(), fixture(True)
    arms = {}
    for name, eps in [('clean', 0.0), ('robust', 0.3)]:
        result = fit(train, epsilon=eps, steps=4000, rate=0.1, l2=0.001)
        arms[name] = {'fit': result, 'train': audit(result['model'], train, epsilon=0.3),
                      'holdout': audit(result['model'], test, epsilon=0.3),
                      'sweep': [{'epsilon': e, 'audit': {k:v for k,v in audit(result['model'], test, epsilon=e).items() if k != 'rows'}}
                                for e in (0, .1, .2, .3, .4, .6, 1.0)]}
    clean, robust = arms['clean']['holdout'], arms['robust']['holdout']
    # Honest gate directions: reject the clean arm as robust, accept the robust arm.
    gates = {'clean_passes_75pct_robust_gate': clean['strict_robust_accuracy'] >= .75,
             'robust_passes_75pct_robust_gate': robust['strict_robust_accuracy'] >= .75,
             'clean_has_higher_clean_accuracy': clean['accuracy'] > robust['accuracy']}
    return {'train_data': train, 'holdout_data': test, 'arms': arms, 'gates': gates,
            'scope': '30+30 synthetic factorial rows, deterministic shifted holdout; no statistical generalization or nonlinear-model claim'}
