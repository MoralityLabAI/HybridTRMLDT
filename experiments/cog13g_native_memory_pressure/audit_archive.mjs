// Replay the actual scope archives, bind raw mailbox bytes and compare common goals.
import {readFileSync,writeFileSync,existsSync} from "node:fs";
import {join,resolve} from "node:path";
import {createHash} from "node:crypto";
import {makeCases,arms,audit,check,equal,canonical} from "./core.mjs";
const folder=resolve(process.argv[2]??".");
const read=p=>JSON.parse(readFileSync(p,"utf8"));
const sha=b=>createHash("sha256").update(b).digest("hex");
const cases=makeCases(),configs=arms(),all=[],goalRegistry=new Map(),errors=[];
let comparableGoals=0,rawReplies=0,rawCertificates=0;
const statuses=readFileSync(join(folder,"scope-exit.tsv"),"utf8").trim().split("\n").map(x=>x.split("\t"));
const expected=new Set([2,4,8].flatMap(g=>configs.map(a=>g+":"+a.name)));
check(statuses.length===12,"scope registry");
for(const [gib,name,status] of statuses){
 const key=gib+":"+name;check(expected.delete(key),"duplicate or unregistered scope");
 const dir=join(folder,gib+"gib_"+name),row={capGiB:Number(gib),arm:name,scopeExit:Number(status),audited:false};
 try{
  check(status==="0","scope failure or OOM; partial artifacts retained");
  const r=read(join(dir,"result.json")),reg=read(join(dir,"registry.json")),summary=read(join(dir,"summary.json"));
  check(equal(reg.cases,cases)&&equal(reg.arm,configs.find(a=>a.name===name)),"registry mismatch");
  check(equal(reg.arm,r.arm)&&equal(reg.limit,r.limit)&&r.limit.bytes===Number(gib)*1073741824&&r.limit.cpu===180e6&&r.limit.wallMs===480000,"budget mismatch");
  check(reg.cgroup.max===String(r.limit.bytes)&&reg.cgroup.swap==="0"&&reg.cgroup.quota===reg.cgroup.period,"cgroup mismatch");
  check(!r.error&&!r.oom&&!summary.auditError,"invalid native scope");
  const frozen=readFileSync(join(dir,"events.jsonl"),"utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
  check(equal(frozen,r.events),"incremental log mismatch");
  check(r.roots.length===cases.length&&r.roots.every((x,i)=>equal(x.c,cases[i])),"cohort mismatch");
  const receipt=audit(r);
  for(const e of r.events){
   if(e.kind==="opened"){
    const native=read(join(dir,"session_"+e.session,"response-0.json"));
    check(native.ready===true&&equal(native.goals,e.initial),"startup mailbox mismatch");
   }else if(e.kind==="reply"){
    const d=join(dir,"session_"+e.session),req=read(join(d,"request-"+e.request.id+".json")),rep=read(join(d,"response-"+e.request.id+".json"));
    check(req.id===e.request.id&&equal(req.actions,e.request.actions)&&req.stop===false&&equal(rep,e.reply),"raw request or reply mismatch");
    rawReplies++;
    if(rep.result.accepted){
     const k=cases[e.root].id+":"+canonical(req.actions);
     if(goalRegistry.has(k)){check(equal(goalRegistry.get(k),rep.result.goals),"cross-arm goal mismatch");comparableGoals++;}
     else goalRegistry.set(k,rep.result.goals);
    }
   }else if(e.kind==="certificate"&&e.certificate.folder){
    const c=e.certificate,d=join(dir,c.folder),src=readFileSync(join(d,"Certificate.lean")),log=readFileSync(join(d,"stdout.log"));
    check(sha(src)===c.sourceHash&&sha(log)===c.logSha256,"certificate hash mismatch");
    if(c.accepted){
     const name=cases[e.root].id,source="import Mathlib\nset_option maxHeartbeats 100000000\ntheorem "+name+" "+cases[e.root].statement+" := by\n"+cases[e.root].scripts[0].map(a=>"  "+a+"\n").join("")+"#print axioms "+name+"\n";
     check(src.toString("utf8")===source&&c.process.exitCode===0&&Array.isArray(c.axioms)&&c.axioms.every(x=>["propext","Classical.choice","Quot.sound"].includes(x))&&!/sorryAx|error(?:\([^\n]*\))?:/.test(log.toString("utf8")),"native proof custody");
     rawCertificates++;
    }
   }
  }
  Object.assign(row,summary,{audited:true,eventCount:receipt.events,peakFraction:r.end.peak/r.limit.bytes,
   pressureClass:r.pressureObserved?"observed_pressure":"nonbinding",unknown:r.unknown});
 }catch(e){row.error=String(e);errors.push({key,error:row.error});}
 all.push(row);
}
check(expected.size===0,"missing scope");
const result={schema:"cog13g.aggregate.v1",complete:errors.length===0,rows:all,rawReplies,rawCertificates,comparableGoals,errors,
 comparisonEligible:errors.length===0&&all.every(r=>r.unknown===0),
 pressureObserved:all.some(r=>r.audited&&r.pressureClass==="observed_pressure"),
 note:"Fixed tactic tapes; pressure and measured CPU only, no learned discovery claim. Host cache warming/order may confound. Any unknowns suppress clean comparisons."};
writeFileSync(join(folder,"aggregate.json"),JSON.stringify(result,null,2)+"\n",{flag:"wx"});console.log(JSON.stringify(result,null,2));
if(!result.complete)process.exitCode=1;
