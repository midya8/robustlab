import copy
import itertools
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from robustlab import box_audit, exact_audit
from robustlab.box import box_demo


def q(r):return F(int(r['numerator']),int(r['denominator']))
def model(w,b=0):return {'version':1,'weights':list(w),'bias':b}
def data(x,y=1):return {'x':[list(x)],'y':[y]}
def bounds(d):return {'lower':[0]*d,'upper':[1]*d}


def reference(w,b,x,y,box,norm,eps):
    """Enumerate all orthant-polytope vertices, not the greedy allocation."""
    d=len(w);values=[]
    for signs in itertools.product((-1,1),repeat=d):
        caps=[(F(x[i])-F(box['lower'][i]) if signs[i]<0 else F(box['upper'][i])-F(x[i])) for i in range(d)]
        if norm=='linf':caps=[min(c,F(eps)) for c in caps]
        points=list(itertools.product(*[(F(0),c) for c in caps]))
        if norm=='l1':
            for j in range(d):
                for ends in itertools.product(*[(F(0),caps[i]) for i in range(d) if i!=j]):
                    v=list(ends);v.insert(j,F(eps)-sum(ends))
                    if 0<=v[j]<=caps[j]:points.append(tuple(v))
        for t in points:
            if norm=='l1' and sum(t)>F(eps):continue
            z=[F(a)+s*v for a,s,v in zip(x,signs,t)]
            values.append(y*(sum(F(a)*v for a,v in zip(w,z))+F(b)))
    return min(values)


class BoxTests(unittest.TestCase):
    def test_independent_all_orthant_vertices(self):
        checked=0
        for d in range(1,4):
            for w in itertools.product((-1,0,2),repeat=d):
                x=[(0,.25,.75)[i] for i in range(d)]
                for y in (-1,1):
                    for norm in ('l1','linf'):
                        for e in (0,.375,2):
                            r=box_audit(model(w,.125),data(x,y),bounds(d),norm=norm,epsilon=e)['rows'][0]
                            self.assertEqual(q(r['worst_signed_margin']),reference(w,.125,x,y,bounds(d),norm,e))
                            checked+=1
        self.assertEqual(checked,468)

    def test_primal_dual_reconstruct(self):
        for norm in ('l1','linf'):
            for y in (-1,1):
                p=box_audit(model([-.1,0,.3],.2),data([.2,.5,.9],y),bounds(3),norm=norm,epsilon=.7)
                r=p['rows'][0];c=r['certificate'];g=[abs(q(t)) for t in p['model']['weights']]
                lam=q(c['budget_multiplier']);mu=list(map(q,c['upper_multipliers']));cap=list(map(q,c['effective_upper']));t=list(map(q,c['allocation']))
                self.assertGreaterEqual(lam,0)
                self.assertTrue(all(v>=0 and a<=lam+v for a,v in zip(g,mu)))
                self.assertTrue(all(0<=v<=a for v,a in zip(t,cap)))
                if norm=='l1':self.assertLessEqual(sum(t),q(p['threat']['epsilon']))
                self.assertEqual(sum(a*v for a,v in zip(g,t)),q(c['primal_gain']))
                self.assertEqual(lam*q(p['threat']['epsilon'])+sum(a*v for a,v in zip(cap,mu)),q(c['dual_gain_bound']))
                self.assertEqual(q(c['gap']),0)
                z=list(map(q,r['attacked_x']));delta=list(map(q,r['delta']));x=list(map(q,r['x']))
                self.assertEqual(z,[a+d for a,d in zip(x,delta)])
                self.assertTrue(all(0<=v<=1 for v in z))
                self.assertEqual(y*(sum(q(a)*v for a,v in zip(p['model']['weights'],z))+q(p['model']['bias'])),q(r['worst_signed_margin']))
                self.assertEqual(q(r['attained_signed_margin']),q(r['worst_signed_margin']))

    def test_clipping_old_l1_witness_is_not_minimum(self):
        r=box_audit(model([4,1],-.25),data([.0625,.75]),bounds(2),norm='l1',epsilon=.625)['rows'][0]
        self.assertEqual(q(r['clipped_old_attack_margin']),F(1,2))
        self.assertEqual(q(r['worst_signed_margin']),F(-1,16))
        self.assertEqual(list(map(q,r['delta'])),[F(-1,16),F(-9,16)])
        self.assertFalse(r['clipped_old_attack_is_optimal']);self.assertFalse(r['strict_robust_correct'])

    def test_linf_clipping_is_optimal(self):
        for c in box_demo()['cases']:
            if c['norm']=='linf':self.assertTrue(c['audit']['rows'][0]['clipped_old_attack_is_optimal'])

    def test_nonnesting_opposite_label_asymmetric_box(self):
        box={'lower':[-2,.125],'upper':[3,.75]}
        for y in (-1,1):
            for norm in ('l1','linf'):
                r=box_audit(model([-3,2],.5),data([-.5,.25],y),box,norm=norm,epsilon=.625)['rows'][0]
                self.assertEqual(q(r['worst_signed_margin']),reference([-3,2],.5,[-.5,.25],y,box,norm,.625))

    def test_degenerate_box_zero_weights(self):
        for w in ([0,0],[2,-1]):
            r=box_audit(model(w,1),data([.25,.75]),{'lower':[.25,.75],'upper':[.25,.75]},norm='l1',epsilon=2)['rows'][0]
            self.assertEqual(q(r['worst_signed_margin']),q(r['clean_signed_margin']))
            self.assertEqual(q(r['certificate']['primal_gain']),0)
            self.assertEqual(list(map(q,r['delta'])),[0,0])

    def test_zero_budget_and_tie_rule(self):
        p=box_audit(model([1],-.5),data([.5]),bounds(1),norm='l1',epsilon=0)
        r=p['rows'][0];self.assertEqual(q(r['worst_signed_margin']),0);self.assertFalse(r['strict_robust_correct']);self.assertEqual(r['prediction'],1)
        self.assertEqual(q(r['certificate']['budget_multiplier']),1)

    def test_budget_slack_and_saturated_box(self):
        r=box_audit(model([2,1]),data([.25,.5]),bounds(2),norm='l1',epsilon=2)['rows'][0]
        self.assertLess(q(r['perturbation_norm']),2);self.assertEqual(q(r['certificate']['budget_multiplier']),0)
        self.assertEqual(list(map(q,r['attacked_x'])),[0,0])

    def test_equal_weight_canonical_first_coordinate(self):
        r=box_audit(model([1,1]),data([1,1]),bounds(2),norm='l1',epsilon=.5)['rows'][0]
        self.assertEqual(list(map(q,r['delta'])),[F(-1,2),0])

    def test_nonbinding_box_matches_all_real_exact(self):
        b={'lower':[-10]*3,'upper':[10]*3};m=model([-.1,.3,.7],.2);d=data([.2,1.1,-.4],-1)
        for norm in ('l1','linf'):
            a=box_audit(m,d,b,norm=norm,epsilon=.3)['rows'][0];r=exact_audit(m,d,norm=norm,epsilon=.3)['rows'][0]
            self.assertEqual(a['worst_signed_margin'],r['worst_signed_margin'])
            self.assertEqual(a['delta'],r['delta'])

    def test_budget_monotonicity_and_lower_bound_order(self):
        for norm in ('l1','linf'):
            vals=[]
            for c in box_demo()['cases']:
                if c['norm']!=norm:continue
                r=c['audit']['rows'][0];v=q(r['worst_signed_margin']);vals.append(v)
                self.assertLessEqual(q(r['all_real_worst_signed_margin']),v)
                self.assertLessEqual(v,q(r['clipped_old_attack_margin']))
            self.assertEqual(vals,sorted(vals,reverse=True))

    def test_binary_semantics_and_subnormal(self):
        p=box_audit(model([.1]),data([.2]),bounds(1),epsilon=.1)
        self.assertEqual(q(p['model']['weights'][0]),F(.1));self.assertNotEqual(q(p['model']['weights'][0]),F(1,10))
        tiny=math.ulp(0.)
        r=box_audit(model([tiny]),data([tiny]),bounds(1),epsilon=0)['rows'][0]
        self.assertEqual(q(r['worst_signed_margin']),F(tiny)**2);self.assertGreater(q(r['worst_signed_margin']),0)

    def test_input_rng_and_return_isolation(self):
        m=model([1,-2]);d=data([.2,.5]);b=bounds(2);saved=copy.deepcopy((m,d,b));rng=random.getstate()
        p=box_audit(m,d,b,norm='l1',epsilon=.5);p['bounds']['lower'][0]['numerator']='bad';p['rows'][0]['x'][0]['numerator']='bad'
        self.assertEqual((m,d,b),saved);self.assertEqual(random.getstate(),rng)
        self.assertNotEqual(box_audit(m,d,b)['bounds']['lower'][0]['numerator'],'bad')

    def test_invalid_bounds(self):
        bad=[None,{},dict(bounds(2),extra=1),{'lower':[0],'upper':[1]}, {'lower':[1,0],'upper':[0,1]}, {'lower':[False,0],'upper':[1,1]}, {'lower':[0,0],'upper':[math.inf,1]}, {'lower':[0,0],'upper':[1e7,1]}, {'lower':'00','upper':[1,1]}]
        for b in bad:
            with self.subTest(b=b),self.assertRaises(ValueError):box_audit(model([1,1]),data([.5,.5]),b)
        for x in ([-.1,.5],[.5,1.1]):
            with self.assertRaises(ValueError):box_audit(model([1,1]),data(x),bounds(2))

    def test_invalid_threat_model_data(self):
        for kw in ({'norm':'l2'},{'norm':'L1'},{'epsilon':True},{'epsilon':-.1},{'epsilon':math.nan}):
            with self.assertRaises(ValueError):box_audit(model([1]),data([.5]),bounds(1),**kw)
        with self.assertRaises(ValueError):box_audit(model([1]),{'x':[[.5]],'y':[True]},bounds(1))

    def test_budget_preflight_before_fractions(self):
        with patch('robustlab.box._q',side_effect=AssertionError('must preflight')):
            with self.assertRaisesRegex(ValueError,'4096'):box_audit(model([1]*128),{'x':[[.5]*128]*33,'y':[1]*33},bounds(128))

    def test_actual_product_cap(self):
        p=box_audit(model([1]*128),{'x':[[.5]*128]*32,'y':[1]*32},bounds(128),norm='l1',epsilon=.25)
        self.assertEqual(len(p['rows']),32);self.assertEqual(q(p['rows'][-1]['worst_signed_margin']),F(255,4))

    def test_demo_both_gate_directions(self):
        p=box_demo();flags=[c['audit']['rows'][0]['strict_robust_correct'] for c in p['cases']]
        self.assertEqual(len(flags),14);self.assertIn(True,flags);self.assertIn(False,flags)

    def test_cli_matches_api(self):
        with tempfile.TemporaryDirectory() as td:
            paths=[]
            for name,obj in [('m',model([4,1],-.25)),('d',data([.0625,.75])),('b',bounds(2))]:
                p=Path(td,name+'.json');p.write_text(json.dumps(obj));paths.append(str(p))
            out=subprocess.run([sys.executable,'-m','robustlab','box-audit',*paths,'--norm','l1','--epsilon','.625'],capture_output=True,text=True)
            self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout),box_audit(model([4,1],-.25),data([.0625,.75]),bounds(2),norm='l1',epsilon=.625))
        out=subprocess.run([sys.executable,'-m','robustlab','box-demo'],capture_output=True,text=True)
        self.assertEqual(json.loads(out.stdout),box_demo())

    def test_cli_bad_files_and_l2_are_errors(self):
        with tempfile.TemporaryDirectory() as td:
            m=Path(td,'m');d=Path(td,'d');b=Path(td,'b')
            m.write_text(json.dumps(model([1])));d.write_text(json.dumps(data([.5])))
            for contents in ('{"lower":[0],"upper":[1],"upper":[1]}','{','{"lower":[0],"upper":[NaN]}',' '*1048577):
                b.write_text(contents)
                out=subprocess.run([sys.executable,'-m','robustlab','box-audit',str(m),str(d),str(b)],capture_output=True,text=True)
                self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'');self.assertNotIn('Traceback',out.stderr)
            b.write_text(json.dumps(bounds(1)))
            for extra in (['--norm','l2'],[]):
                target=str(b) if extra else str(Path(td,'missing'))
                out=subprocess.run([sys.executable,'-m','robustlab','box-audit',str(m),str(d),target,*extra],capture_output=True,text=True)
                self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'')
