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
from robustlab import exact_audit
from robustlab.exact import exact_demo


def m(w,b=0):return {'version':1,'weights':list(w),'bias':b}
def q(r):return F(int(r['numerator']),int(r['denominator']))


class ExactTests(unittest.TestCase):
    def test_independent_corner_minima_float_inputs(self):
        for d in range(1,4):
            for w in itertools.product((-.1,0,.3),repeat=d):
                x=[.2+i*.1 for i in range(d)]
                for y in (-1,1):
                    for norm in ('l1','linf'):
                        eps=.1;b=-.2
                        if norm=='linf':ds=list(itertools.product((-F(eps),F(eps)),repeat=d))
                        else:
                            ds=[]
                            for i in range(d):
                                for sign in (-1,1):
                                    v=[F(0)]*d;v[i]=sign*F(eps);ds.append(v)
                        vals=[y*(sum(F(a)*(F(t)+e) for a,t,e in zip(w,x,delta))+F(b)) for delta in ds]
                        r=exact_audit(m(w,b),{'x':[x],'y':[y]},norm=norm,epsilon=eps)['rows'][0]
                        self.assertEqual(q(r['worst_signed_margin']),min(vals))
                        self.assertEqual(q(r['attained_signed_margin']),min(vals))
                        self.assertEqual(r['strict_robust_correct'],min(vals)>0)

    def test_witness_rational_feasibility_and_score(self):
        for norm in ('l1','linf'):
            p=exact_audit(m([-.1,.3,.7],.2),{'x':[[.2,1.1,-.4]],'y':[-1]},norm=norm,epsilon=.3)
            r=p['rows'][0];delta=list(map(q,r['delta']));x=list(map(q,r['x']))
            length=sum(abs(t) for t in delta) if norm=='l1' else max(abs(t) for t in delta)
            self.assertLessEqual(length,F(.3))
            z=list(map(q,r['attacked_x']));self.assertEqual(z,[a+d for a,d in zip(x,delta)])
            w=list(map(q,p['model']['weights']));b=q(p['model']['bias'])
            self.assertEqual(-1*(sum(a*t for a,t in zip(w,z))+b),q(r['worst_signed_margin']))

    def test_false_positive_float_certificate(self):
        r=exact_demo()['cases'][0]['audit']['rows'][0]
        self.assertGreater(r['float_worst_signed_margin'],0)
        self.assertLess(q(r['worst_signed_margin']),0)
        self.assertTrue(r['float_strict_robust_correct'])
        self.assertFalse(r['strict_robust_correct'])
        self.assertTrue(r['strict_flag_disagrees'])

    def test_float_misses_positive_and_dyadic_control(self):
        p=exact_demo()
        r=p['cases'][1]['audit']['rows'][0]
        self.assertEqual(r['float_worst_signed_margin'],0)
        self.assertGreater(q(r['worst_signed_margin']),0)
        self.assertTrue(r['strict_robust_correct']);self.assertFalse(r['float_strict_robust_correct'])
        r=p['cases'][2]['audit']['rows'][0]
        self.assertEqual(q(r['worst_signed_margin']),0)
        self.assertFalse(r['strict_flag_disagrees'])
        self.assertFalse(r['strict_robust_correct'])

    def test_binary_not_decimal_semantics(self):
        p=exact_audit(m([.1]),{'x':[[.2]],'y':[1]},epsilon=.3)
        self.assertEqual(q(p['model']['weights'][0]),F(.1))
        self.assertNotEqual(q(p['model']['weights'][0]),F(1,10))
        self.assertEqual(q(p['threat']['epsilon']),F(.3))

    def test_prediction_disagrees_under_cancellation(self):
        p=exact_audit(m([.1],-.020000000000000004),{'x':[[.2]],'y':[-1]})
        r=p['rows'][0];self.assertEqual(r['prediction'],-1);self.assertEqual(r['float_prediction'],1)
        self.assertTrue(r['prediction_disagrees'])
        self.assertEqual(p['counts']['prediction_disagreements'],1)

    def test_zero_weights_and_tie_radius(self):
        p=exact_audit(m([0,0],.5),{'x':[[1,2],[1,2]],'y':[1,-1]},epsilon=1e6)
        self.assertTrue(p['rows'][0]['radius_unbounded']);self.assertFalse(p['rows'][1]['radius_unbounded'])
        self.assertIsNone(p['rows'][0]['radius_to_zero_margin'])
        r=exact_audit(m([.5]),{'x':[[.5]],'y':[1]},epsilon=.5)['rows'][0]
        self.assertEqual(q(r['radius_to_zero_margin']),F(.5));self.assertEqual(q(r['worst_signed_margin']),0)

    def test_canonical_l1_tie(self):
        r=exact_audit(m([-.3,.3]),{'x':[[1,1]],'y':[1]},norm='l1',epsilon=.1)['rows'][0]
        self.assertEqual(list(map(q,r['delta'])),[F(.1),0])

    def test_subnormal_exactness(self):
        tiny=math.ulp(0.)
        r=exact_audit(m([tiny]),{'x':[[tiny]],'y':[1]})['rows'][0]
        self.assertEqual(r['float_worst_signed_margin'],0)
        self.assertEqual(q(r['worst_signed_margin']),F(tiny)**2)
        self.assertTrue(r['strict_robust_correct'])
        self.assertEqual(r['worst_signed_margin']['denominator'],str(1<<2148))

    def test_float_cancellation_does_not_change_exact_witness(self):
        p=exact_demo()['cases'][0]['audit'];r=p['rows'][0]
        self.assertEqual(q(r['attained_signed_margin']),q(r['worst_signed_margin']))
        self.assertEqual(p['counts']['strict_flag_disagreements'],1)

    def test_counts_multiple_rows(self):
        p=exact_audit(m([1]),{'x':[[2],[-2],[0]],'y':[1,-1,1]},epsilon=1)
        self.assertEqual(p['counts'],{'total':3,'correct':3,'strict_clean_correct':2,'strict_robust_correct':2,'strict_flag_disagreements':0,'prediction_disagreements':0})

    def test_preflight_before_fraction_or_float_audit(self):
        with patch('robustlab.exact._q',side_effect=AssertionError('must preflight')):
            with self.assertRaisesRegex(ValueError,'4096'):
                exact_audit(m([1]*128),{'x':[[1]*128]*33,'y':[1]*33})

    def test_actual_work_boundary(self):
        p=exact_audit(m([1]*128),{'x':[[1]*128]*32,'y':[1]*32},epsilon=.5)
        self.assertEqual(len(p['rows']),32)
        self.assertEqual(q(p['rows'][0]['worst_signed_margin']),64)

    def test_reject_l2_and_bad_inputs(self):
        for kw in ({'norm':'l2'},{'norm':'l3'},{'epsilon':True},{'epsilon':-1}):
            with self.assertRaises(ValueError):exact_audit(m([1]),{'x':[[1]],'y':[1]},**kw)
        with self.assertRaises(ValueError):exact_audit(m([1]),{'x':[[1]],'y':[True]})

    def test_isolation_and_json_roundtrip(self):
        model=m([.1,.2]);data={'x':[[.2,.3]],'y':[1]};before=copy.deepcopy((model,data));state=random.getstate()
        p=exact_audit(model,data,epsilon=.1)
        self.assertEqual(json.loads(json.dumps(p,allow_nan=False)),p)
        p['model']['weights'][0]['numerator']='99'
        self.assertEqual((model,data),before);self.assertEqual(random.getstate(),state)

    def test_exact_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp=Path(tmp)/'m';dp=Path(tmp)/'d';mp.write_text(json.dumps(m([.1])));dp.write_text(json.dumps({'x':[[.2]],'y':[1]}))
            p=subprocess.run([sys.executable,'-m','robustlab','exact-audit',str(mp),str(dp),'--epsilon','.1'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout),exact_audit(m([.1]),{'x':[[.2]],'y':[1]},epsilon=.1))
            p=subprocess.run([sys.executable,'-m','robustlab','exact-audit',str(mp),str(dp),'--norm','l2'],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertNotIn('Traceback',p.stderr)

    def test_exact_demo_cli(self):
        p=subprocess.run([sys.executable,'-m','robustlab','exact-demo'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),exact_demo())


if __name__=='__main__':unittest.main()
