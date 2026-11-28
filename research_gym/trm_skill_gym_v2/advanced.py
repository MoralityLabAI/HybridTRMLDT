"""SKILL-02B/F advanced CPU experiment; no credential or paid API."""
import argparse
import json
from pathlib import Path
import torch
from research_gym.trm_skill_gym import campaign as C
from .runner import SOURCE
from . import subspace, student


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--count",type=int,default=128)
    p.add_argument("--nulls",type=int,default=8)
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1])
    p.add_argument("--skills",nargs="+",choices=("repair","curation","research","stopping"),default=["repair","curation","research","stopping"])
    a=p.parse_args()
    if not 1<=a.count<=1024 or not 1<=a.nulls<=32:
        p.error("bounded count/nulls required")
    torch.set_num_threads(1)
    archive=[json.loads(l) for l in (a.source/"search/archive.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    distill=student.run(archive)
    result={"profile":{"count":a.count,"seeds":a.seeds,"skills":a.skills,"nulls":a.nulls},
            "student":distill,"subspace":subspace.run(a.source,a.skills,a.seeds,a.count,a.nulls)}
    C.write_json(a.out/"advanced.json",result)
    summary={"student":{k:{m:v for m,v in x.items() if m!="predictions"} for k,x in distill["rungs"].items()},
             "latent":{k:{m:round(v["macro"],6) for m,v in x["latent"]["test"].items() if "macro" in v}
                        for k,x in result["subspace"].items()},
             "weight":{k:{m:round(v["macro"],6) for m,v in x["tied_weight"]["test"].items() if "macro" in v}
                        for k,x in result["subspace"].items()}}
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
