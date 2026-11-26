// Native Lean transport. Requires Linux, GNU time, Node 22 and Lean 4.35.0.
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, openSync, closeSync, readFileSync, writeFileSync, renameSync } from "node:fs";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { MENU,fitAll,readout } from "./model.mjs";
import {check,canon,cases,arms,search,audit,summarize,gates,fixture} from "./search.mjs";

const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const hash=b=>createHash("sha256").update(b).digest("hex");
const json=(p,v)=>writeFileSync(p,JSON.stringify(v,null,2)+"\n",{flag:"wx"});
const text=p=>readFileSync(p,"utf8");
const allowedAxioms=new Set(["propext","Classical.choice","Quot.sound"]);
function assert(x,m){if(!x)throw Error(m);}
function rssTree(pid,seen=new Set()){
 if(!pid||seen.has(pid))return 0;seen.add(pid);
 try{
  const stat=text("/proc/"+pid+"/status"),m=stat.match(/^VmRSS:\s+(\d+)\s+kB/m);
  const children=text("/proc/"+pid+"/task/"+pid+"/children").trim().split(/\s+/).filter(Boolean).map(Number);
  return Number(m?.[1]??0)+children.reduce((s,c)=>s+rssTree(c,seen),0);
 }catch{return 0;}
}
function source(c,actions,probe=false){
 assert(/^cog13h_[0-1]_[0-7]$/.test(c.id),"invalid theorem name");
 assert(actions.every(a=>MENU.includes(a)),"action outside frozen grammar");
 return "import Mathlib\n"+(probe?"import StateCacheH\n":"")+"set_option maxHeartbeats 1000000\n"+
  "theorem "+c.id+" "+c.statement+" := by\n"+
  (probe?"  cog13h_probe\n":actions.map(a=>"  "+a+"\n").join("")+"#print axioms "+c.id+"\n");
}
function atomic(p,v){const tmp=p+".tmp";writeFileSync(tmp,JSON.stringify(v),{flag:"wx"});renameSync(tmp,p);}
function nativeBackend(out,lean,cwd){
 let n=0,phase="initial";const processes=[];
 const now=()=>performance.now();
 function launch(kind,root,folder,file,env={}){
  const log=join(folder,"stdout.log"),timing=join(folder,"time.txt"),fd=openSync(log,"wx");
  const begin=now(),child=spawn("/usr/bin/time",["-f","%U %S %M","-o",timing,lean,file],{cwd,env:{...process.env,LEAN_NUM_THREADS:"1",...env},detached:true,stdio:["ignore",fd,fd]});closeSync(fd);
  const p={kind,root,phase,folder,child,log,timing,begin,peakRssKiB:0,spawnError:null,recorded:false};
  child.on("error",e=>{p.spawnError=String(e);});
  return p;
 }
 function sample(p){const rss=rssTree(p.child.pid);p.peakRssKiB=Math.max(p.peakRssKiB,rss);return rss;}
 function exited(p){return p.child.exitCode!==null||p.child.signalCode!==null||p.spawnError!==null;}
 function terminate(p){if(!exited(p)&&p.child.pid)try{process.kill(-p.child.pid,"SIGKILL");}catch{}}
 async function wait(p,deadline){while(!exited(p)&&now()<deadline){sample(p);await sleep(5);}return exited(p);}
 function record(p){
  if(p.recorded)return;p.recorded=true;
  let userSeconds=null,systemSeconds=null,gnuMaxRssKiB=null;
  if(existsSync(p.timing)){const lines=text(p.timing).trim().split("\n");const m=lines.at(-1)?.match(/^([\d.]+) ([\d.]+) (\d+)$/);if(m){userSeconds=Number(m[1]);systemSeconds=Number(m[2]);gnuMaxRssKiB=Number(m[3]);}}
  processes.push({kind:p.kind,root:p.root,phase:p.phase,folder:p.folder,elapsedMs:now()-p.begin,exitCode:p.child.exitCode,signal:p.child.signalCode,spawnError:p.spawnError,peakSampledRssKiB:p.peakRssKiB,userSeconds,systemSeconds,gnuMaxRssKiB,logSha256:existsSync(p.log)?hash(readFileSync(p.log)):null});
 }
 async function stop(p){if(!exited(p)){terminate(p);await wait(p,now()+500);}record(p);}
 async function readResponse(p,id,deadline){
  const file=join(p.folder,"response-"+id+".json");
  while(now()<deadline){
   sample(p);
   if(existsSync(file))return JSON.parse(text(file));
   if(exited(p))throw Error("Lean exited before response "+id+": "+text(p.log).slice(-1000));
   await sleep(5);
  }throw Error("Lean request timeout "+id);
 }
 return {now,processes,setPhase:x=>{phase=x;},
  open:async(c,deadline,slots)=>{
   const folder=join(out,"proc_"+String(n++).padStart(5,"0"));mkdirSync(folder);
   const file=join(folder,"Probe.lean");writeFileSync(file,source(c,[],true),{flag:"wx"});
   const p=launch("probe_server",c.id,folder,file,{COG13H_MAILBOX:folder,COG13H_CAP:String(slots),COG13H_POLICY:"lru"});
   let number=0,closed=false;
   try{
    const r=await readResponse(p,0,Math.min(deadline,now()+15000));
    assert(r.ready===true&&Array.isArray(r.goals)&&r.goals.length>0&&r.goals.every(x=>typeof x==="string"),"invalid startup");
    return {initial:r.goals,folder,rssKiB:()=>sample(p),
     query:async(prefix,id,limit)=>{
      assert(!closed&&prefix.length>0&&prefix.every(a=>MENU.includes(a)),"invalid query");
      number++;assert(number===id,"probe sequence");atomic(join(folder,"request-"+number+".json"),{id:number,actions:prefix,stop:false});
      const data=await readResponse(p,number,Math.min(limit,now()+5000));
      assert(data.id===number&&typeof data.result?.accepted==="boolean","desynchronized probe response");
      const a=data.result;
      if(a.accepted)assert(Array.isArray(a.goals)&&a.goals.every(x=>typeof x==="string"),"invalid goal response");
      return data;
     },
     close:async()=>{
      if(closed)return;closed=true;
      if(!exited(p)){number++;try{atomic(join(folder,"request-"+number+".json"),{id:number,actions:[],stop:true});}catch{}await wait(p,now()+400);}
      await stop(p);
     }};
   }catch(e){await stop(p);throw e;}
  },
  certify:async(c,prefix,deadline,rep)=>{
   const folder=join(out,"proc_"+String(n++).padStart(5,"0"));mkdirSync(folder);
   const file=join(folder,"Certificate.lean"),src=source(c,prefix,false);writeFileSync(file,src,{flag:"wx"});
   const p=launch("certificate_"+rep,c.id,folder,file);
   const finished=await wait(p,Math.min(deadline,now()+15000));if(!finished)terminate(p);await stop(p);
   const log=text(p.log),name=c.id;
   const noAxioms=log.includes("'"+name+"' does not depend on any axioms");
   const match=log.match(new RegExp("'"+name+"' depends on axioms: \\[([^\\]]*)\\]"));
   const axioms=noAxioms?[]:match?match[1].split(",").map(x=>x.trim()).filter(Boolean):null;
   const accepted=finished&&p.child.exitCode===0&&axioms!==null&&axioms.every(x=>allowedAxioms.has(x))&&!/sorryAx|error(?:\([^\n]*\))?:|declaration uses 'sorry'/.test(log);
   return {accepted,sourceHash:hash(src),axioms,exitCode:p.child.exitCode,timedOut:!finished,sourceFile:file,logSha256:hash(log)};
  }};
}

// Actual archive bytes are verified by prepare.py before this runner starts.
const [mode,outArg,leanArg,cwdArg,inputArg,seedArg]=process.argv.slice(2);
check(outArg&&["--fixture","--native"].includes(mode),"usage: --fixture OUTPUT | --native OUT LEAN MATHLIB INPUT SEED");
if(mode==="--fixture"){
 const rows=Array.from({length:224},(_,i)=>({root:"train"+Math.floor(i/28),split:"train",goals:["P : Prop\nhp : P\n⊢ P ∧ P"],action:MENU[i%16],target:Number(i%16===0||i%16===2)}));
 const result=await fixture(fitAll(rows,1701));json(resolve(outArg),result);console.log(JSON.stringify(result));
}else{
 check(leanArg&&cwdArg&&inputArg&&[1701,1709,1723].includes(Number(seedArg)),"native parameters");
 const seed=Number(seedArg),out=resolve(outArg),input=JSON.parse(text(inputArg));mkdirSync(out);
 check(input.archiveSha256==="d99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea","input archive pin");
 const registry=cases();check(registry.every(c=>!input.oldStatements.includes(c.statement)),"old statement collision");
 const models=fitAll(input.train,seed);
 const retrospective=readout(input.retrospective,models),trainFit=readout(input.fitDiagnostic,models);
 json(join(out,"models.json"),models);json(join(out,"retrospective.json"),{scope:"selection-biased archived observed candidates; not a proof rollout",metrics:retrospective});
 json(join(out,"registry.json"),registry);
 const raw=nativeBackend(out,resolve(leanArg),resolve(cwdArg)),results=[],errors=[],goals=new Map();
 const initialGoals=new Map();
 const backend={now:raw.now,open:async(c,arm,s,deadline)=>raw.open(c,deadline,arm.slots),
  certify:async(c,arm,s,actions,rep,deadline)=>raw.certify(c,actions,deadline,rep)};
 function rawAudit(r){
  const phase=r.case.id+"_"+r.arm.name,procs=raw.processes.filter(p=>p.phase===phase);
  const sessions=procs.filter(p=>p.kind==="probe_server");
  check(sessions.length===1,"probe process denominator");
  const proc=sessions[0],folder=proc.folder;
  if(r.initial){
   const start=JSON.parse(text(join(folder,"response-0.json")));check(start.ready&&canon(start.goals)===canon(r.initial),"raw initial goals");
   if(initialGoals.has(r.case.id))check(canon(initialGoals.get(r.case.id))===canon(r.initial),"root goal differs across arms");else initialGoals.set(r.case.id,r.initial);
  }
  for(const e of r.events.filter(x=>x.id)){
   const req=JSON.parse(text(join(folder,"request-"+e.id+".json")));check(req.id===e.id&&canon(req.actions)===canon(e.actions)&&req.stop===false,"raw request");
   if(!e.reply)continue;
   const res=JSON.parse(text(join(folder,"response-"+e.id+".json")));check(canon(res)===canon(e.reply),"raw reply");
   const k=r.case.id+":"+canon(e.actions);
   const verdict=e.reply.result.accepted?{accepted:true,goals:e.reply.result.goals}:{accepted:false};
   if(goals.has(k))check(canon(goals.get(k))===canon(verdict),"common native verdict or goal differs");else goals.set(k,verdict);
  }
  check(new Set(r.certs.filter(c=>c.sourceFile).map(c=>c.sourceFile)).size===r.certs.filter(c=>c.sourceFile).length,"reused compiler process");
  for(const cert of r.certs){
   if(!cert.sourceFile){check(!cert.accepted,"missing accepted cert");continue;}
   const src=readFileSync(cert.sourceFile),log=readFileSync(join(cert.sourceFile,"..","stdout.log"));
   const p=procs.find(p=>p.folder===join(cert.sourceFile,"..")&&p.kind.startsWith("certificate_"));check(p&&p.root===r.case.id&&p.exitCode===cert.exitCode&&p.logSha256===cert.logSha256,"compiler process binding");
   check(hash(src)===cert.sourceHash&&hash(log)===cert.logSha256&&src.toString("utf8")===source(r.case,r.closedActions,false),"certificate byte custody");
   if(cert.accepted)check(cert.exitCode===0&&Array.isArray(cert.axioms)&&cert.axioms.every(x=>allowedAxioms.has(x))&&!/sorryAx|error(?:\([^\n]*\))?:/.test(log.toString("utf8")),"invalid accepted cert");
  }
 }
 try{
  const all=arms();
  for(let i=0;i<registry.length;i++){
   const c=registry[i],offset=(i+[1701,1709,1723].indexOf(seed))%all.length,order=[...all.slice(offset),...all.slice(0,offset)];
   for(const arm of order){
    const phase=c.id+"_"+arm.name;raw.setPhase(phase);
    const cpu0=process.cpuUsage(),r=await search(c,arm,models,backend,seed),cpu=process.cpuUsage(cpu0);
    const ps=raw.processes.filter(p=>p.phase===phase),complete=ps.filter(p=>p.userSeconds!==null&&p.systemSeconds!==null);
    r.cost={controllerCpuUsec:cpu.user+cpu.system,leanCpuSeconds:complete.reduce((v,p)=>v+p.userSeconds+p.systemSeconds,0),completeProcesses:complete.length,processes:ps.length};
    results.push(r);
    writeFileSync(join(out,phase+".json"),JSON.stringify(r,null,2)+"\n",{flag:"wx"});
    try{check(r.reason!=="evidence_error","search evidence error");audit(r,models);rawAudit(r);}catch(e){errors.push({phase,error:String(e)});}
   }
  }
  check(results.length===112&&new Set(results.map(r=>r.case.id+":"+r.arm.name)).size===112,"registry coverage");
  const result={schema:"cog13h.native.v1",seed,archiveSha256:input.archiveSha256,registry,trainRows:input.train.length,fit:{loop:models.loop.losses,flat:models.flat.losses},fitDiagnostic:trainFit,retrospective,summary:summarize(results),gates:gates(results),errors,results};
  json(join(out,"result.json"),result);
  console.log(JSON.stringify({seed,summary:result.summary,gates:result.gates,errors,retrospective},null,2));
  if(errors.length)process.exitCode=1;
 }catch(e){json(join(out,"blocker.json"),{error:String(e),stack:e.stack});process.exitCode=1;console.error(e);}
 finally{json(join(out,"processes.json"),raw.processes);}
}
