import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from robustlab.exact import exact_demo
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('renderexact',ROOT/'examples'/'render_exact.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
NS={'s':'http://www.w3.org/2000/svg'}


class ExactFigureTests(unittest.TestCase):
    def test_all_six_flags_and_values(self):
        root=ET.fromstring(module.render(exact_demo()));rows=root.findall('s:g[@data-result]',NS)
        self.assertEqual(len(rows),6)
        expected=[True,False,False,True,False,False]
        self.assertEqual([r.attrib['data-flag']=='true' for r in rows],expected)
        self.assertGreater(float(rows[0].attrib['data-margin']),0);self.assertLess(float(rows[1].attrib['data-margin']),0)
        self.assertEqual(float(rows[2].attrib['data-margin']),0);self.assertGreater(float(rows[3].attrib['data-margin']),0)
        self.assertEqual(root.attrib['viewBox'],'0 0 1120 590')

    def test_metadata_reproducibility_and_mutation(self):
        p=exact_demo();before=copy.deepcopy(p);s=module.render(p);root=ET.fromstring(s)
        self.assertEqual(json.loads(root.find('s:metadata',NS).text),p)
        self.assertEqual((ROOT/'assets'/'exact-boundary.svg').read_text(),s)
        self.assertEqual(p,before)

    def test_text_positions_and_scope(self):
        root=ET.fromstring(module.render(exact_demo()));texts=root.findall('s:text',NS)
        for t in texts:
            self.assertGreaterEqual(float(t.attrib['x']),0);self.assertLess(float(t.attrib['x']),1120)
            self.assertGreaterEqual(float(t.attrib['y']),0);self.assertLess(float(t.attrib['y']),590)
        full=' '.join(t.text or '' for t in texts)
        self.assertIn('not exact decimal literals',full)
        self.assertIn('not to estimate its frequency',full)
        self.assertIn('does not certify a separately rounded execution pipeline',full)

    def test_reject_changed_input_margin_and_attack(self):
        for path in [('input_model','bias'),('audit','rows',0,'worst_signed_margin','numerator'),('audit','rows',0,'delta',0,'numerator')]:
            p=exact_demo();c=p['cases'][0]
            for k in path[:-1]:c=c[k]
            c[path[-1]]=123
            with self.assertRaises(ValueError):module.render(p)
