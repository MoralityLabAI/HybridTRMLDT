import json
import tempfile
import unittest
from pathlib import Path
from repro import box, compare, digest, metrics

class ReproTests(unittest.TestCase):
    def test_boxed(self):
        self.assertEqual(box('<think>\\boxed{wrong}</think>Answer \\boxed{\\frac{1}{2}}'), '\\frac{1}{2}')
        self.assertEqual(box('\\boxed{a^{b+c}}'), 'a^{b+c}')
        self.assertIsNone(box('\\boxed{missing'))
        self.assertEqual(box('\\boxed 5\n'), '5')

    def test_metrics(self):
        r = [{'correct': True, 'cut': False, 'prediction': '3', 'tokens': 8},
             {'correct': False, 'cut': True, 'prediction': None, 'tokens': 16}]
        self.assertEqual(metrics(r)['cut_share'], 0.5)

    def test_pairing(self):
        with tempfile.TemporaryDirectory() as d:
            files=[]
            for name, c in [('base', False), ('edited', True)]:
                p=Path(d)/f'{name}.json'
                p.write_text(json.dumps({'roster_sha256': digest(['q']), 'sampling': {'seed': 0}, 'model': name,
                    'records': [{'id':'q', 'sample':0, 'correct':c, 'cut':not c, 'prediction':None if not c else '2', 'tokens':20}]}))
                files.append(str(p))
            class Args: pass
            a=Args(); a.results=files; a.out=str(Path(d)/'report.json')
            compare(a)
            r=json.loads(Path(a.out).read_text())
            self.assertEqual(r['paired'][0]['rescues'], 1)

if __name__ == '__main__':
    unittest.main()
