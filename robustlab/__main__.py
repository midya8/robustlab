import argparse
import json
import os
import sys
from .core import audit, fit
from .demo import demo


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def load(path):
    with open(path, 'rb') as f:
        raw = f.read(1048577)
    if len(raw) > 1048576:
        raise ValueError('JSON files are limited to 1 MiB')
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def main(argv=None):
    p = argparse.ArgumentParser(description='Linear classifier robustness under explicit norm balls')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('demo')
    a = sub.add_parser('audit')
    a.add_argument('model'); a.add_argument('data')
    f = sub.add_parser('fit')
    f.add_argument('data'); f.add_argument('--steps', type=int, default=1000)
    f.add_argument('--rate', type=float, default=.1); f.add_argument('--l2', type=float, default=.001)
    for parser in (a, f):
        parser.add_argument('--norm', choices=('l1', 'l2', 'linf'), default='linf')
        parser.add_argument('--epsilon', type=float, default=0.0)
    args = p.parse_args(argv)
    try:
        if args.command == 'demo':
            out = demo()
        elif args.command == 'audit':
            out = audit(load(args.model), load(args.data), norm=args.norm, epsilon=args.epsilon)
        else:
            out = fit(load(args.data), norm=args.norm, epsilon=args.epsilon,
                      steps=args.steps, rate=args.rate, l2=args.l2)
        text = json.dumps(out, indent=2, allow_nan=False)
    except (OSError, ValueError, TypeError, OverflowError) as e:
        p.exit(2, 'error: ' + str(e) + '\n')
    print(text)


if __name__ == '__main__':
    main()
