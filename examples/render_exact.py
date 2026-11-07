"""Numeric boundary table; no clipped axes, rounding never chooses a certificate."""
import argparse
import html
import json
from pathlib import Path
from fractions import Fraction
from robustlab.exact import exact_demo


def render(payload):
    if payload != exact_demo():
        raise ValueError('requires complete unchanged exact demo')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="590" viewBox="0 0 1120 590">',
         '<rect width="1120" height="590" fill="#101827"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#e7edf8}.muted{fill:#b7c4d9}</style>',
         '<metadata>'+html.escape(json.dumps(payload,sort_keys=True,allow_nan=False))+'</metadata>']
    def text(x,y,s,size=16,fill=None):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}"'+(f' style="fill:{fill}"' if fill else '')+'>'+html.escape(s)+'</text>')
    text(35,43,'RobustLab | when floating-point boundary signs disagree',27)
    text(35,73,'Exact rational audits of stored binary floats. L-infinity balls. Strict worst signed margin > 0.',16,'#b7c4d9')
    for x,t in [(45,'Fixture'),(410,'Float worst margin'),(665,'Rational margin (decimal view)'),(940,'Strict flags')]:text(x,122,t,15)
    labels=['False positive in float audit','Float misses positive margin','Dyadic boundary control']
    for i,case in enumerate(payload['cases']):
        r=case['audit']['rows'][0];q=r['worst_signed_margin'];value=Fraction(int(q['numerator']),int(q['denominator']))
        y=157+79*i
        svg.append(f'<rect x="35" y="{y-15}" width="1050" height="68" rx="7" fill="#1e2a40"/>')
        text(45,y+10,labels[i],17)
        text(45,y+34,'epsilon = '+str(case['input_epsilon']),14,'#b7c4d9')
        for key,x,margin in [('float',410,r['float_worst_signed_margin']),('exact',665,float(value))]:
            flag=r['float_strict_robust_correct'] if key=='float' else r['strict_robust_correct']
            color='#66d7b4' if flag else '#f4a8b1'
            text(x,y+14,f'{margin:+.9e}',18,color)
            svg.append(f'<g data-result="{i}-{key}" data-flag="{str(flag).lower()}" data-margin="{margin}"/>')
        text(940,y+6,'float: '+str(r['float_strict_robust_correct']),15)
        text(940,y+30,'exact: '+str(r['strict_robust_correct']),15)
    text(35,417,'Certificates use integer numerator/denominator comparisons, not the decimal views printed above.',16)
    text(35,446,'Every exact attack is rational, feasible in the stated ball, and attains the exact worst margin.',16)
    text(35,478,'Exact binary inputs are not exact decimal literals. Full fractions, models and witnesses are embedded.',15,'#b7c4d9')
    text(35,507,'Boundary fixtures were selected to expose rounding, not to estimate its frequency. The trained demo is unchanged.',15,'#b7c4d9')
    text(35,536,'L1 and L-infinity only. All-real coordinates, no clipping. This does not certify a separately rounded execution pipeline.',15,'#b7c4d9')
    text(35,568,'Independent implementation using standard norm duality and Python Fraction arithmetic.',14,'#b7c4d9')
    svg.append('</svg>');return '\n'.join(svg)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    path=Path(__file__).resolve().parents[1]/'assets'/'exact-boundary.svg';actual=render(exact_demo())
    if args.check:
        if not path.exists() or path.read_text()!=actual:raise SystemExit('exact figure is stale')
    else:path.write_text(actual)


if __name__=='__main__':main()
