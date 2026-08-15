from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def _load(path):
    with open(ROOT/path,encoding='utf-8') as f: return json.load(f)

def load_policy(path='config/policy.json'): return _load(path)
def load_catalogue(path='config/catalogue.json'):
    c=_load(path)
    # Runtime-only compatibility view: the physical catalogue remains pedagogical-only.
    c.update(load_policy())
    c['scaffolds']=[]
    for d in c['strategies']:
        x=dict(d); x['phase']=d['phases'][0]; x['role']='pedagogical_coach'; x['permitted']=['ask reflection','invite contribution','surface perspectives']; x['forbidden']=['choose an idea','rank ideas','generate an idea']; c['scaffolds'].append(x)
    return c
