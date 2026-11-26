import {readFileSync,writeFileSync,existsSync} from "node:fs";
import {join,resolve} from "node:path";
import {cases,arms,canon,gates,summarize,check,audit} from "./search.mjs";
const path=resolve(process.argv[2]),seeds=[1701,1709,1723],errors=[],results=[];
for(const seed of seeds){
 try{
  const dir=join(path,"cog13h-seed-"+seed),r=JSON.parse(readFileSync(join(dir,"result.json"),"utf8")),m=JSON.parse(readFileSync(join(dir,"models.json"),"utf8"));
  check(r.seed===seed&&r.trainRows===224&&r.archiveSha256==="d99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea","source/seed registry");
  check(canon(r.registry)===canon(cases())&&r.results.length===112&&new Set(r.results.map(x=>x.case.id+":"+x.arm.name)).size===112,"cohort");
  check(r.results.every(x=>x.seed===seed&&cases().some(c=>canon(c)===canon(x.case))&&arms().some(a=>canon(a)===canon(x.arm))),"arm or root binding");
  check(r.errors.length===0,"producer audit failure");
  check(canon(r.summary)===canon(summarize(r.results))&&canon(r.gates)===canon(gates(r.results)),"summary replay");
  // The native runner already checked raw mailbox/certificates and replayed all
  // ranking scores. Here recompute discrete counts, gate, source, seed registry.
  results.push(r);
 }catch(e){errors.push({seed,error:String(e)});}
}
const complete=results.length===seeds.length&&errors.length===0;
const commonUnique=complete?results[0].gates.unique.filter(id=>results.every(r=>r.gates.unique.includes(id))):[];
const unknowns=results.reduce((s,r)=>s+r.results.filter(x=>["startup_unknown","query_unknown","evidence_error"].includes(x.reason)).length,0);
const gatesPassed={recurrence:complete&&!unknowns&&commonUnique.length>0&&results.every(r=>r.gates.recurrence),compression:complete&&!unknowns&&results.every(r=>r.gates.compression),cache:complete&&!unknowns&&results.every(r=>r.gates.cache)};
const out={schema:"cog13h.aggregate.v1",complete,unknowns,gates:gatesPassed,commonUnique,errors,perSeed:results.map(r=>({seed:r.seed,summary:r.summary,gates:r.gates,retrospective:r.retrospective,fit:r.fit}))};
writeFileSync(join(path,"aggregate.json"),JSON.stringify(out,null,2)+"\n",{flag:"wx"});console.log(JSON.stringify(out,null,2));
if(!complete)process.exitCode=1;
