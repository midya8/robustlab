"""A bounded-domain audit, including the deliberately suboptimal clipped L1 attack."""
import argparse
import html
import json
from pathlib import Path
from robustlab.box import box_demo, _decode


SERIES=(('all_real_worst_signed_margin','#f2a7b5','All-real minimum'),
        ('clipped_old_attack_margin','#efc76a','Clipped old attack (candidate)'),
        ('worst_signed_margin','#63d8bc','Box-constrained minimum'))


def render(payload):
    if payload!=box_demo():raise ValueError('requires complete unchanged box demo')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650">',
         '<rect width="1120" height="650" fill="#101827"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>',
         '<metadata>'+html.escape(json.dumps(payload,sort_keys=True,allow_nan=False))+'</metadata>']
    def text(x,y,s,size=15,color=None):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}"'+(f' style="fill:{color}"' if color else '')+'>'+html.escape(s)+'</text>')
    text(35,42,'RobustLab | clipping an attack is not a robustness certificate',26)
    text(35,72,'Linear model w = (4, 1), b = -1/4; x = (1/16, 3/4), label +1; domain [0, 1]^2.',17)
    for k,(key,color,label) in enumerate(SERIES):
        text(35+365*k,106,label,15,color)
    for norm,left in [('l1',80),('linf',650)]:
        text(left,145,'L1 budget' if norm=='l1' else 'L-infinity budget',19)
        def xy(e,v):return left+420*e,450-(v+4.5)/5.5*280
        for v in (-4,-3,-2,-1,0,1):
            _,y=xy(0,v)
            svg.append(f'<path d="M {left} {y} H {left+420}" stroke="#33445e"/>')
            text(left-35,y+5,str(v),13)
        for e in (0,.25,.5,.75,1):
            x,_=xy(e,0);text(x-8,474,str(e),13)
        text(left+160,501,'epsilon',15)
        cases=[c for c in payload['cases'] if c['norm']==norm and c['epsilon']<=1]
        for key,color,label in SERIES:
            pts=[]
            for c in cases:
                value=float(_decode(c['audit']['rows'][0][key]));e=c['epsilon'];x,y=xy(e,value);pts.append(f'{x},{y}')
                svg.append(f'<circle data-norm="{norm}" data-series="{key}" data-epsilon="{e}" data-value="{value}" cx="{x}" cy="{y}" r="4" fill="{color}"/>')
            svg.append(f'<polyline data-norm="{norm}" data-series="{key}" points="'+ ' '.join(pts)+f'" fill="none" stroke="{color}" stroke-width="2"'+(' stroke-dasharray="6 5"' if key=='clipped_old_attack_margin' else '')+'/>')
    text(35,538,'At L1 epsilon = 5/8: clipping the old attack leaves margin +1/2; the true box minimum is -1/16.',17)
    text(35,568,'Greedy allocation moves 1/16 in coordinate 1, then 9/16 in coordinate 2. Exact primal = dual gain = 13/16.',16)
    text(35,597,'L-infinity clipping is optimal here; the two corresponding curves overlap. Both panels share the same axes.',15,'#b7c4d9')
    text(35,627,'Hand-specified synthetic fixture, no prevalence claim. Rational mathematical model, not a rounded execution guarantee.',15,'#b7c4d9')
    svg.append('</svg>');return '\n'.join(svg)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    path=Path(__file__).resolve().parents[1]/'assets'/'box-audit.svg';actual=render(box_demo())
    if a.check:
        if not path.exists() or path.read_text()!=actual:raise SystemExit('box figure is stale')
    else:path.write_text(actual)


if __name__=='__main__':main()
