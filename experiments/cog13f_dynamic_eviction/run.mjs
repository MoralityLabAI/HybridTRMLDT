import {readFileSync,writeFileSync,mkdirSync} from "node:fs";
import {join,resolve,dirname} from "node:path";
import {fileURLToPath} from "node:url";
import {createHash} from "node:crypto";
import {gzipSync} from "node:zlib";
import {makeCorpus as oldCorpus} from "../cog13e_retention_value/core.mjs";
import {suite} from "./core.mjs";

const out=process.argv[2];if(!out)throw Error("usage: node run.mjs NEW_OUTPUT_DIR");
const dir=resolve(out),here=dirname(fileURLToPath(import.meta.url));
mkdirSync(dir);
const x=suite(true),finger=t=>JSON.stringify({nodes:t.nodes,trace:t.trace});
const older=new Set(oldCorpus().tasks.map(finger));
if(x.tasks.some(t=>older.has(finger(t))))throw Error("old corpus overlap");
const frozen=JSON.parse(readFileSync(join(here,"results.json"),"utf8"));
if(JSON.stringify(x.result)!==JSON.stringify(frozen))throw Error("frozen result mismatch");
if(x.calibrationRuns.length+x.runs.length!==x.result.audit.runs)throw Error("incomplete replay");
if(x.labels.some(g=>g.split!=="train"||g.continuation!=="recent"))throw Error("label custody");
if(new Set(x.labels.map(g=>g.root)).size!==64)throw Error("training roots");
let count=0;
for(const g of x.labels){
 const min=Math.min(...g.rows.map(r=>r.cost));count+=g.rows.length;
 for(const r of g.rows)if(r.regret!==r.cost-min||r.target!==Math.log1p(r.regret/10))throw Error("label arithmetic");
}
if(count!==x.result.model.rows)throw Error("training count");
const hashes={};
const emit=(name,data)=>{writeFileSync(join(dir,name),data,{flag:"wx"});hashes[name]=createHash("sha256").update(data).digest("hex");};
const json=x=>JSON.stringify(x,null,2)+"\n";
const gz=xs=>gzipSync(xs.map(x=>JSON.stringify(x)).join("\n")+"\n");
emit("result.json",json(x.result));
emit("registry.json",json(x.tasks));
emit("eviction_train.jsonl.gz",gz(x.labels));
emit("training_receipts.json",json(x.trainReceipts));
emit("calibration_replay.jsonl.gz",gz(x.calibrationRuns));
emit("heldout_replay.jsonl.gz",gz(x.runs));
const sourceHashes=Object.fromEntries(["PROTOCOL.md","core.mjs","run.mjs","results.json"].map(n=>[n,createHash("sha256").update(readFileSync(join(here,n))).digest("hex")]));
writeFileSync(join(dir,"manifest.json"),json({schema:"cog13f.exports.v1",sourceHashes,hashes,oldCorpusDisjoint:true,trainingCandidates:count,calibrationRuns:x.calibrationRuns.length,evaluationRuns:x.runs.length,selectedLambda:x.result.selected,evidence:x.result.evidence}),{flag:"wx"});
console.log(JSON.stringify({selectedLambda:x.result.selected,gate:x.result.gate,audit:x.result.audit,labels:count},null,2));
