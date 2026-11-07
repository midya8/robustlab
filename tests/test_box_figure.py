import copy
import html
import importlib.util
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from robustlab.box import box_demo, _decode

spec=importlib.util.spec_from_file_location('render_box',Path(__file__).resolve().parents[1]/'examples'/'render_box.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
NS={'s':'http://www.w3.org/2000/svg'}


class BoxFigureTests(unittest.TestCase):
    def test_all_36_points_and_shared_domains(self):
        p=box_demo();root=ET.fromstring(renderer.render(p));points=root.findall('s:circle',NS);self.assertEqual(len(points),36)
        for point in points:
            a=point.attrib;norm=a['data-norm'];e=float(a['data-epsilon']);key=a['data-series']
            c=next(c for c in p['cases'] if c['norm']==norm and c['epsilon']==e)
            v=float(_decode(c['audit']['rows'][0][key]));self.assertEqual(float(a['data-value']),v)
            self.assertEqual(float(a['cx']),(80 if norm=='l1' else 650)+420*e)
            self.assertEqual(float(a['cy']),450-(v+4.5)/5.5*280)
            self.assertTrue(170<=float(a['cy'])<=450)
        self.assertEqual(root.attrib['viewBox'],'0 0 1120 650')

    def test_polylines_match_all_points(self):
        root=ET.fromstring(renderer.render(box_demo()));lines=root.findall('s:polyline',NS);self.assertEqual(len(lines),6)
        for line in lines:
            a=line.attrib;points=[p for p in root.findall('s:circle',NS) if all(p.attrib[k]==a[k] for k in ('data-norm','data-series'))]
            self.assertEqual(a['points'],' '.join(p.attrib['cx']+','+p.attrib['cy'] for p in points))
        self.assertEqual(sum('stroke-dasharray' in p.attrib for p in lines),2)

    def test_full_metadata_reproducible_no_input_mutation(self):
        p=box_demo();saved=copy.deepcopy(p);svg=renderer.render(p)
        self.assertEqual(p,saved);self.assertEqual(svg,renderer.render(p))
        self.assertEqual(json.loads(ET.fromstring(svg).find('s:metadata',NS).text),p)
        self.assertEqual(svg,(Path(__file__).resolve().parents[1]/'assets'/'box-audit.svg').read_text())
        self.assertIn('true box minimum is -1/16',svg);self.assertIn('curves overlap',svg)

    def test_reject_changed_certificate_model_or_selection(self):
        for change in ('certificate','model','selection','bounds'):
            p=box_demo()
            if change=='certificate':p['cases'][0]['audit']['rows'][0]['certificate']['gap']['numerator']='1'
            elif change=='model':p['model']['weights'][0]=5
            elif change=='bounds':p['bounds']['upper'][0]=2
            else:p['selection']='unknown'
            with self.assertRaises(ValueError):renderer.render(p)
