import copy
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest.mock import patch
from robustlab import audit, fit, objective_gradient
from robustlab.core import _dual, _objective
from robustlab.demo import demo, fixture


def model(w, b=0):
    return {'version': 1, 'weights': list(w), 'bias': b}


class AuditTests(unittest.TestCase):
    def test_linf_independent_fraction_all_corners(self):
        for d in range(1, 5):
            for w in itertools.product((-2, 0, 1), repeat=d):
                for y in (-1, 1):
                    x = [Fraction(i-1, 3) for i in range(d)]
                    eps, b = Fraction(1, 4), Fraction(1, 2)
                    values = [y*(sum(Fraction(a)*(t+e) for a,t,e in zip(w,x,delta))+b)
                              for delta in itertools.product((-eps, eps), repeat=d)]
                    r = audit(model(w,float(b)), {'x': [[float(t) for t in x]], 'y':[y]},
                              epsilon=float(eps))['rows'][0]
                    self.assertAlmostEqual(r['worst_signed_margin'], float(min(values)))
                    self.assertAlmostEqual(r['attained_signed_margin'], float(min(values)))

    def test_l1_independent_vertices(self):
        for w in itertools.product((-2, 0, 1), repeat=3):
            for y in (-1, 1):
                x, eps, b = [1/3, -.2, 2], .7, -.5
                vals = [y*(sum(a*t for a,t in zip(w,x))+b)]
                for j in range(3):
                    for sign in (-1, 1):
                        z = x[:]; z[j] += sign*eps
                        vals.append(y*(sum(a*t for a,t in zip(w,z))+b))
                r = audit(model(w,b), {'x':[x], 'y':[y]}, norm='l1',epsilon=eps)['rows'][0]
                self.assertAlmostEqual(r['worst_signed_margin'],min(vals))
                self.assertAlmostEqual(sum(abs(t) for t in r['delta']), eps if any(w) else 0)

    def test_l2_cauchy_and_circle_reference(self):
        r = audit(model([3,4],.1), {'x':[[.7,-.2]],'y':[-1]}, norm='l2',epsilon=.4)
        row=r['rows'][0]
        self.assertAlmostEqual(r['dual_weight_norm'],5)
        self.assertAlmostEqual(math.hypot(*row['delta']),.4)
        self.assertAlmostEqual(row['attained_signed_margin'],row['worst_signed_margin'])
        samples=[-(3*(.7+.4*math.cos(a*math.tau/10000))+4*(-.2+.4*math.sin(a*math.tau/10000))+.1)
                 for a in range(10000)]
        self.assertLessEqual(row['worst_signed_margin'], min(samples)+1e-12)
        self.assertLess(min(samples)-row['worst_signed_margin'],1e-6)

    def test_label_and_bias(self):
        for n in ('l1','l2','linf'):
            a=audit(model([2,-3],1),{'x':[[1,2],[1,2]],'y':[1,-1]},norm=n,epsilon=.2)
            self.assertAlmostEqual(a['rows'][0]['clean_signed_margin'],-a['rows'][1]['clean_signed_margin'])
            self.assertEqual(a['rows'][0]['delta'],[-t for t in a['rows'][1]['delta']])

    def test_epsilon_zero(self):
        a=audit(model([2,-3]),{'x':[[1,2]],'y':[1]},epsilon=0)
        r=a['rows'][0]
        self.assertEqual(r['clean_signed_margin'],r['worst_signed_margin'])
        self.assertEqual(r['delta'],[0,0])

    def test_tie_prediction_vs_strict_certificate(self):
        a=audit(model([1]),{'x':[[0]],'y':[1]})
        self.assertEqual(a['accuracy'],1)
        self.assertEqual(a['strict_clean_accuracy'],0)
        self.assertEqual(a['strict_robust_accuracy'],0)
        a=audit(model([1]),{'x':[[1]],'y':[1]},epsilon=1)
        self.assertEqual(a['rows'][0]['worst_signed_margin'],0)
        self.assertFalse(a['rows'][0]['strict_robust_correct'])

    def test_zero_weights_positive_constant_unbounded(self):
        a=audit(model([0,0],2),{'x':[[1,3],[1,3]],'y':[1,-1]},epsilon=1e6)
        self.assertEqual(a['rows'][0]['radius_to_zero_margin'],None)
        self.assertTrue(a['rows'][0]['radius_unbounded'])
        self.assertFalse(a['rows'][1]['radius_unbounded'])
        self.assertEqual(a['rows'][0]['delta'],[0,0])

    def test_radius_and_positive_scaling(self):
        data={'x':[[3,1],[-2,1]],'y':[1,-1]}
        for norm in ('l1','l2','linf'):
            a=audit(model([2,-1],.4),data,norm=norm,epsilon=.3)
            b=audit(model([20,-10],4),data,norm=norm,epsilon=.3)
            self.assertEqual(a['strict_robust_accuracy'],b['strict_robust_accuracy'])
            for r,s in zip(a['rows'],b['rows']):
                self.assertAlmostEqual(r['radius_to_zero_margin'],s['radius_to_zero_margin'])
                self.assertAlmostEqual(r['worst_signed_margin']*10,s['worst_signed_margin'])

    def test_certificate_monotonic_with_epsilon(self):
        data=fixture(True)
        for norm in ('l1','l2','linf'):
            vals=[audit(model([1,2]),data,norm=norm,epsilon=e)['strict_robust_accuracy'] for e in (0,.1,.3,1,10)]
            self.assertEqual(vals, sorted(vals,reverse=True))

    def test_norm_threat_sets_distinct(self):
        data={'x':[[2,0]],'y':[1]}
        vals=[audit(model([3,4]),data,norm=n,epsilon=1)['rows'][0]['worst_signed_margin'] for n in ('l1','l2','linf')]
        self.assertEqual(vals,[2,1,-1])

    def test_canonical_tied_l1_attack(self):
        a=audit(model([-2,2]),{'x':[[1,1]],'y':[1]},norm='l1',epsilon=.5)
        self.assertEqual(a['rows'][0]['delta'],[.5,0])

    def test_input_return_and_rng_isolation(self):
        m=model([2,-1]);data=fixture(); orig=copy.deepcopy((m,data)); state=random.getstate()
        a=audit(m,data,epsilon=.1);a['model']['weights'][0]=99;a['rows'][0]['x'][0]=99
        self.assertEqual((m,data),orig);self.assertEqual(state,random.getstate())

    def test_no_box_clipping(self):
        a=audit(model([1]),{'x':[[.1]],'y':[1]},epsilon=1)
        self.assertEqual(a['rows'][0]['attacked_x'],[-.9])

    def test_boundary_dimensions(self):
        a=audit(model([1]*128),{'x':[[1]*128]*4096,'y':[1]*4096},epsilon=.1)
        self.assertEqual(len(a['rows']),4096)
        with self.assertRaises(ValueError):audit(model([1]*129),{'x':[[1]*129],'y':[1]})
        with self.assertRaises(ValueError):audit(model([1]),{'x':[[1]]*4097,'y':[1]*4097})

    def test_validation_models(self):
        for m in ({},model([]),model([True]),model([float('nan')]),model([float('inf')]),model([1e7]),
                  {**model([1]),'version':True},{**model([1]),'extra':0},{**model([1]),'bias':False}):
            with self.assertRaises(ValueError):audit(m,{'x':[[1]],'y':[1]})

    def test_validation_data(self):
        for d in ({},{'x':[],'y':[]},{'x':[[1]],'y':[True]},{'x':[[1]],'y':[1.]},
                  {'x':[[1,2]],'y':[1]},{'x':[[1]],'y':[0]},{'x':[[1]],'y':[]},
                  {'x':[[None]],'y':[1]},{'x':[[1]],'y':[1],'extra':2}):
            with self.assertRaises(ValueError):audit(model([1]),d)

    def test_validation_threat(self):
        for n,e in [('l3',1),('l2',-1),('l1',True),('linf',float('nan'))]:
            with self.assertRaises(ValueError):audit(model([1]),{'x':[[1]],'y':[1]},norm=n,epsilon=e)


class TrainingTests(unittest.TestCase):
    def test_finite_difference_all_norms_away_from_kinks(self):
        data={'x':[[.3,-.2,1.1],[-.7,2,.5],[1,1,-1]],'y':[1,-1,1]}
        w=[.3,-.8,1.2];b=.17;h=1e-5
        for n in ('l1','l2','linf'):
            for e in (0,.25):
                g=objective_gradient(model(w,b),data,norm=n,epsilon=e,l2=.07)
                for j in range(4):
                    wp,wm=w[:],w[:];bp,bm=b,b
                    if j<3:wp[j]+=h;wm[j]-=h
                    else:bp+=h;bm-=h
                    lp=objective_gradient(model(wp,bp),data,norm=n,epsilon=e,l2=.07)['loss']
                    lm=objective_gradient(model(wm,bm),data,norm=n,epsilon=e,l2=.07)['loss']
                    actual=g['weights_gradient'][j] if j<3 else g['bias_gradient']
                    self.assertAlmostEqual(actual,(lp-lm)/(2*h),places=8)

    def test_nonsmooth_subgradient_inequality(self):
        data={'x':[[1,2],[-1,-2]],'y':[1,-1]}
        for n,w in [('linf',[0,.7]),('l1',[.7,-.7]),('l2',[0,0]),('l1',[0,0])]:
            g=objective_gradient(model(w),data,norm=n,epsilon=.3,l2=.1)
            for v in itertools.product((-1,0,1),repeat=2):
                f=objective_gradient(model(v,.2),data,norm=n,epsilon=.3,l2=.1)['loss']
                plane=g['loss']+sum(a*(t-s) for a,t,s in zip(g['weights_gradient'],v,w))+g['bias_gradient']*.2
                self.assertGreaterEqual(f+1e-12,plane)

    def test_exact_robust_loss_from_corner_attacks(self):
        w=[.4,-.7];data={'x':[[1,2],[-.3,.7]],'y':[1,-1]};eps=.2
        terms=[]
        for row,y in zip(data['x'],data['y']):
            terms.append(max(math.log1p(math.exp(-y*(sum(a*(x+d) for a,x,d in zip(w,row,delta))+.1)))
                             for delta in itertools.product((-eps,eps),repeat=2)))
        actual=objective_gradient(model(w,.1),data,epsilon=eps,l2=.05)['loss']
        self.assertAlmostEqual(actual,sum(terms)/2+.025*sum(a*a for a in w))

    def test_extreme_scores_finite(self):
        for y in (-1,1):
            g=objective_gradient(model([1e6],1e6),{'x':[[1e6]],'y':[y]},epsilon=1e6)
            self.assertTrue(math.isfinite(g['loss']))
            self.assertTrue(all(math.isfinite(a) for a in g['weights_gradient']))

    def test_fit_history_exact_final_and_update(self):
        data={'x':[[1],[-1]],'y':[1,-1]}
        f=fit(data,steps=1,rate=.1,l2=.1)
        self.assertEqual(f['model']['weights'],[.05]);self.assertEqual(f['model']['bias'],0)
        self.assertEqual(len(f['history']),2)
        g=objective_gradient(f['model'],data,l2=.1)
        self.assertEqual(g['loss'],f['history'][-1]['loss'])
        self.assertFalse(f['convergence_claim'])

    def test_training_not_test_dependent(self):
        data=fixture();a=fit(data,steps=20);b=fit(copy.deepcopy(data),steps=20)
        self.assertEqual(a,b)
        self.assertLess(a['history'][-1]['loss'],a['history'][0]['loss'])

    def test_work_cap_before_objective(self):
        with patch('robustlab.core._objective',side_effect=AssertionError('must preflight')):
            with self.assertRaises(ValueError):fit({'x':[[1]*128]*4096,'y':[1]*4096},steps=100)

    def test_fit_options(self):
        for kwargs in ({'steps':True},{'steps':0},{'steps':20001},{'rate':0},{'rate':11},{'l2':-1}, {'epsilon':-1}):
            with self.assertRaises(ValueError):fit(fixture(),**kwargs)

    def test_reject_explosive_parameter_update(self):
        with self.assertRaises(ValueError):fit({'x':[[1e6]],'y':[1]},steps=1,rate=10)

    def test_demo_gates_and_full_witnesses(self):
        p=demo()
        self.assertEqual(p['gates'],{'clean_passes_75pct_robust_gate':False,'robust_passes_75pct_robust_gate':True,'clean_has_higher_clean_accuracy':True})
        self.assertEqual(p['arms']['clean']['holdout']['accuracy'],1)
        self.assertEqual(p['arms']['robust']['holdout']['accuracy'],.8)
        self.assertEqual(p['arms']['clean']['holdout']['strict_robust_accuracy'],.2)
        self.assertEqual(p['arms']['robust']['holdout']['strict_robust_accuracy'],.8)
        for a in p['arms'].values():
            for r in a['holdout']['rows']:
                self.assertAlmostEqual(r['worst_signed_margin'],r['attained_signed_margin'])


class CLITests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,'-m','robustlab',*args],capture_output=True,text=True)

    def test_cli_audit_and_fit(self):
        with tempfile.TemporaryDirectory() as t:
            m=os.path.join(t,'m.json');d=os.path.join(t,'d.json')
            with open(m,'w') as f:json.dump(model([1,2]),f)
            with open(d,'w') as f:json.dump(fixture(),f)
            p=self.run_cli('audit',m,d,'--epsilon','.3','--norm','l2')
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),audit(model([1,2]),fixture(),norm='l2',epsilon=.3))
            p=self.run_cli('fit',d,'--steps','2');self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(len(json.loads(p.stdout)['history']),3)

    def test_cli_duplicate_nonfinite_oversized(self):
        with tempfile.TemporaryDirectory() as t:
            m=os.path.join(t,'m');d=os.path.join(t,'d')
            with open(d,'w') as f:json.dump({'x':[[1]],'y':[1]},f)
            for raw in ('{"version":1,"weights":[1],"bias":0,"bias":2}',
                        '{"version":1,"weights":[NaN],"bias":0}', '{', ' '*1048577):
                with open(m,'w') as f:f.write(raw)
                p=self.run_cli('audit',m,d)
                self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertNotIn('Traceback',p.stderr)

    def test_cli_bad_flags_missing_files(self):
        for args in (['audit','missing','missing'],['fit','missing','--epsilon','nan'],[],['audit','a','b','--norm','l3']):
            p=self.run_cli(*args);self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'')

    def test_demo_cli(self):
        p=self.run_cli('demo');self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),demo())


if __name__=='__main__':unittest.main()
