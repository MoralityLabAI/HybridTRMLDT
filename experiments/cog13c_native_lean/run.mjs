// Native Lean transport. Requires Linux, GNU time, Node 22 and Lean 4.35.0.
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, openSync, closeSync, readFileSync, writeFileSync, renameSync } from "node:fs";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { MENU, suite, selftest } from "./core.mjs";

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
 assert(/^cog13c_[0-2]_[0-7]$/.test(c.id),"invalid theorem name");
 assert(actions.every(a=>MENU.includes(a)),"action outside frozen grammar");
 return "import Mathlib\n"+(probe?"import ProbeServer\n":"")+"set_option maxHeartbeats 1000000\n"+
  "theorem "+c.id+" "+c.statement+" := by\n"+
  (probe?"  ram03_probe_server\n":actions.map(a=>"  "+a+"\n").join("")+"#print axioms "+c.id+"\n");
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
  open:async(c,deadline)=>{
   const folder=join(out,"proc_"+String(n++).padStart(5,"0"));mkdirSync(folder);
   const file=join(folder,"Probe.lean");writeFileSync(file,source(c,[],true),{flag:"wx"});
   const p=launch("probe_server",c.id,folder,file,{RAM03_MAILBOX:folder});
   let number=0,closed=false;
   try{
    const r=await readResponse(p,0,Math.min(deadline,now()+15000));
    assert(r.ready===true&&r.state?.accepted===true&&Array.isArray(r.state.goals)&&r.state.goals.length>0&&r.state.goals.every(x=>typeof x==="string"),"invalid startup");
    return {initial:r.state.goals,rssKiB:()=>sample(p),
     query:async(prefix,limit)=>{
      assert(!closed&&prefix.length>0&&prefix.every(a=>MENU.includes(a)),"invalid query");
      number++;atomic(join(folder,"request-"+number+".json"),{id:number,actions:prefix,stop:false});
      const data=await readResponse(p,number,Math.min(limit,now()+5000));
      assert(data.id===number&&typeof data.result?.accepted==="boolean","desynchronized probe response");
      const a=data.result;
      if(a.accepted)assert(Array.isArray(a.goals)&&a.goals.every(x=>typeof x==="string"),"invalid goal response");
      return a;
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
   const accepted=finished&&p.child.exitCode===0&&axioms!==null&&axioms.every(x=>allowedAxioms.has(x))&&!/sorryAx|error:|declaration uses 'sorry'/.test(log);
   return {accepted,sourceHash:hash(src),axioms,exitCode:p.child.exitCode,timedOut:!finished,sourceFile:file,logSha256:hash(log)};
  }};
}
const [mode,output,leanArg,cwdArg]=process.argv.slice(2);
assert(["--fixture","--native"].includes(mode)&&output,"usage: node run.mjs --fixture OUT.json | --native NEW_DIR LEAN MATHLIB_ROOT");
if(mode==="--fixture"){
 const result=await selftest();json(resolve(output),result);console.log(JSON.stringify(result));
}else{
 assert(leanArg&&cwdArg&&process.platform==="linux","native mode requires Linux, Lean and Mathlib root");
 const out=resolve(output);mkdirSync(out);const backend=nativeBackend(out,resolve(leanArg),resolve(cwdArg));
 try{
  const result=await suite(backend);
  json(join(out,"result.json"),result);
  const rows=result.teacher.rows.map(r=>({root:r.root,split:"train",input:{goals:r.goals,candidate:r.action},target:{progress:r.target,closed:r.closed},prefix:r.prefix,source:"native_lean_probe"}));
  writeFileSync(join(out,"tactic_train.jsonl"),rows.map(r=>JSON.stringify(r)).join("\n")+"\n",{flag:"wx"});
  writeFileSync(join(out,"allocator_train.jsonl"),result.allocator.rows.map(r=>JSON.stringify(r)).join("\n")+"\n",{flag:"wx"});
  json(join(out,"task_registry.json"),result.cases);
  const hashes=Object.fromEntries(["result.json","tactic_train.jsonl","allocator_train.jsonl","task_registry.json"].map(f=>[f,hash(readFileSync(join(out,f)))]));
  json(join(out,"export_manifest.json"),{schema:"cog13c.native_exports.v1",hashes,tacticRows:rows.length,allocatorRows:result.allocator.rows.length});
  console.log(JSON.stringify({summary:result.summary,teacherRows:rows.length,allocatorSolved:result.allocator.solved,allocatorCensored:result.allocator.censored},null,2));
 }catch(e){json(join(out,"blocker.json"),{error:String(e),stack:e.stack});process.exitCode=1;console.error(e);}
 finally{
  json(join(out,"processes.json"),backend.processes);
  const totals={};for(const p of backend.processes){const a=totals[p.phase]??={processes:0,completeCPU:0,userSeconds:0,systemSeconds:0,peakProcessRssKiB:0};a.processes++;if(p.userSeconds!==null&&p.systemSeconds!==null){a.completeCPU++;a.userSeconds+=p.userSeconds;a.systemSeconds+=p.systemSeconds;}a.peakProcessRssKiB=Math.max(a.peakProcessRssKiB,p.peakSampledRssKiB,p.gnuMaxRssKiB??0);}
  json(join(out,"process_cost_by_phase.json"),totals);
 }
}
