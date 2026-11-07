"""Render only the unchanged fixed demo, with full metadata and no external assets."""
import argparse
import html
import json
from pathlib import Path
from robustlab.demo import demo


def render(payload):
    expected=demo()
    if payload != expected:
        raise ValueError('renderer requires the complete unchanged fixed demo')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="710" viewBox="0 0 1120 710">',
         '<rect width="1120" height="710" fill="#101827"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#e7edf8}.muted{fill:#b7c4d9}.small{font-size:14px}.axis{stroke:#54647b;stroke-width:1}</style>',
         '<metadata>'+html.escape(json.dumps(payload,sort_keys=True,allow_nan=False))+'</metadata>']
    def text(x,y,s,size=16,cls=''):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}">{html.escape(s)}</text>')
    text(36,43,'RobustLab | exact linear threat-set audits',28)
    text(36,73,'Fixed synthetic holdout: 30 rows. L-infinity radius 0.30. Same training rows, optimizer and budget.',16,'muted')
    text(42,116,'Clean vs strict robust accuracy',20)
    for j,(name,arm) in enumerate(payload['arms'].items()):
        x=80+j*215
        for k,(field,color,label) in enumerate([('accuracy','#66c7f2','Clean'),('strict_robust_accuracy','#af94f5','Robust')]):
            value=arm['holdout'][field];bx=x+65*k;top=362-190*value
            svg.append(f'<rect data-bar="{name}-{field}" x="{bx}" y="{top}" width="48" height="{190*value}" fill="{color}" rx="3"/>')
            text(bx,top-10,f'{value:.0%}',17)
            text(bx,386,label,13,'muted')
        text(x,418,name+' training',17)
    svg.append('<path class="axis" d="M60 172V362H455" fill="none"/>')
    text(28,177,'100%',12,'muted');text(39,366,'0%',12,'muted')
    text(535,116,'Strict robust accuracy as the budget grows',20)
    for y,val in [(172,'100%'),(267,'50%'),(362,'0%')]:
        svg.append(f'<path class="axis" d="M575 {y}H1060"/>');text(533,y+5,val,12,'muted')
    for name,color in [('clean','#66c7f2'),('robust','#af94f5')]:
        points=[(575+485*s['epsilon'],362-190*s['audit']['strict_robust_accuracy']) for s in payload['arms'][name]['sweep']]
        svg.append(f'<polyline data-curve="{name}" points="'+ ' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="3"/>')
        for x,y in points:svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
    for e in (0,.2,.4,.6,1):text(570+485*e,386,str(e),12,'muted')
    text(735,418,'L-infinity epsilon',16)
    svg.append('<rect x="36" y="445" width="1048" height="220" rx="10" fill="#1e2a40"/>')
    for y,s in [(477,'Worst signed margin = y (w.x + b) - epsilon ||w||_1. A witness attains it for this linear model.'),
                (508,'The robust arm trades 100% -> 80% clean accuracy for 20% -> 80% strict robust accuracy here.'),
                (539,'Robust gate (>=75%): clean arm FAIL, robust arm PASS. No test-driven model selection.'),
                (578,'A strict margin of zero fails conservatively; prediction uses +1 at a score tie.'),
                (608,'All-real input domain, no box clipping. Analytic identity in float64, not interval-certified arithmetic.'),
                (638,'Independent implementation. Norm-ball semantics: Kolter/Madry tutorial; arXiv:2208.09449, Theorem 2.')]:
        text(55,y,s,15,'muted' if y>=578 else '')
    text(36,694,'Supplied binary linear classifier only. Synthetic fixture, not an image benchmark or a guarantee about neural networks.',14,'muted')
    svg.append('</svg>')
    return '\n'.join(svg)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    path=Path(__file__).resolve().parents[1]/'assets'/'robustness-audit.svg'
    actual=render(demo())
    if args.check:
        if not path.exists() or path.read_text()!=actual:raise SystemExit('figure is stale')
    else:path.write_text(actual)


if __name__=='__main__':main()
