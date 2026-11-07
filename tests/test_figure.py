import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from robustlab.demo import demo

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render',ROOT/'examples'/'render_audit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
NS={'s':'http://www.w3.org/2000/svg'}


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=demo();cls.svg=module.render(cls.payload);cls.tree=ET.fromstring(cls.svg)

    def test_all_four_bars_actual_coordinates(self):
        bars=self.tree.findall('s:rect[@data-bar]',NS)
        self.assertEqual(len(bars),4)
        expected={'clean-accuracy':1.,'clean-strict_robust_accuracy':.2,'robust-accuracy':.8,'robust-strict_robust_accuracy':.8}
        for b in bars:
            v=expected[b.attrib['data-bar']]
            self.assertAlmostEqual(float(b.attrib['height']),190*v)
            self.assertAlmostEqual(float(b.attrib['y']),362-190*v)
        self.assertEqual(self.tree.attrib['viewBox'],'0 0 1120 710')

    def test_all_fourteen_curve_points(self):
        curves=self.tree.findall('s:polyline',NS);self.assertEqual(len(curves),2)
        for c in curves:
            expected=self.payload['arms'][c.attrib['data-curve']]['sweep']
            points=[tuple(map(float,p.split(','))) for p in c.attrib['points'].split()]
            self.assertEqual(len(points),7)
            for (x,y),s in zip(points,expected):
                self.assertAlmostEqual(x,575+485*s['epsilon'])
                self.assertAlmostEqual(y,362-190*s['audit']['strict_robust_accuracy'])
        self.assertEqual(len(self.tree.findall('s:circle',NS)),14)

    def test_metadata_full_demo_and_reproducible(self):
        self.assertEqual(json.loads(self.tree.find('s:metadata',NS).text),self.payload)
        self.assertEqual((ROOT/'assets'/'robustness-audit.svg').read_text(),self.svg)
        original=copy.deepcopy(self.payload);self.assertEqual(module.render(self.payload),self.svg)
        self.assertEqual(self.payload,original)

    def test_reject_changed_fit_witness_or_fixture(self):
        paths=[('arms','clean','fit','model','bias'),('arms','robust','holdout','accuracy'),
               ('train_data','x',0,0),('arms','clean','holdout','rows',0,'delta',0)]
        for path in paths:
            p=copy.deepcopy(self.payload);c=p
            for k in path[:-1]:c=c[k]
            c[path[-1]]=123
            with self.assertRaises(ValueError):module.render(p)


if __name__=='__main__':unittest.main()
