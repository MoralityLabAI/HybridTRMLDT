// Linux/Node 22 native runner. Every result is grounded in a Lean response.
import {spawn} from "node:child_process";
import {createHash} from "node:crypto";
import {readFileSync,writeFileSync,mkdirSync,existsSync,openSync,closeSync,renameSync} from "node:fs";
import {join,resolve,dirname} from "node:path";
import {performance} from "node:perf_hooks";
import {check,equal,makeCases,arms,audit,fixture,badAction} from "./core.mjs";

const read=p=>readFileSync(p,"utf8");
const json=(p,v)=>writeFileSync(p,JSON.stringify(v,null,2)+"\n",{flag:"wx"});
const sha=x=>createHash("sha256").update(x).digest("hex");
const now=()=>performance.now(),sleep=ms=>new Promise(r=>setTimeout(r,ms));
function cgroup(){
 const match=read("/proc/self/cgroup").split("\n").find(s=>s.startsWith("0::"));
 check(match,"cgroup v2 not found");
 const path=join("/sys/fs/cgroup",match.slice(3));
 const fields={path};
 for(const k of ["memory.max","memory.swap.max","memory.current","memory.peak","memory.events","cpu.stat"]){
  const p=join(path,k);fields[k]=existsSync(p)?read(p).trim():null;
 }
 return fields;
}
function limits(cg,bytes){
 if(bytes>0)check(cg["memory.max"]===String(bytes)&&cg["memory.swap.max"]==="0","expected cgroup cap/swap mismatch: "+JSON.stringify(cg));
}
function rss(pid,seen=new Set()){
 if(!pid||seen.has(pid))return 0;seen.add(pid);
 try{
  const s=read("/proc/"+pid+"/status"),k=Number(s.match(/^VmRSS:\s+(\d+)\s+kB/m)?.[1]??0);
  const children=read("/proc/"+pid+"/task/"+pid+"/children").trim().split(/\s+/).filter(Boolean).map(Number);
  return k+children.reduce((a,c)=>a+rss(c,seen),0);
 }catch{return 0;}
}
function source(c,actions,probe){
 check(/^cog13d_[0-7]$/.test(c.id),"unexpected theorem name");
 return "import Mathlib\n"+(probe?"import StateCacheServer\n":"")+"set_option maxHeartbeats 100000000\n"+
  "theorem "+c.id+" "+c.statement+" := by\n"+
  (probe?"  cog13d_probe\n":actions.map(s=>"  "+s+"\n").join("")+"#print axioms "+c.id+"\n");
}
function atomic(p,v){const t=p+".tmp";writeFileSync(t,JSON.stringify(v),{flag:"wx"});renameSync(t,p);}
function launch(lean,cwd,folder,file,extra={}){
 const log=join(folder,"stdout.log"),time=join(folder,"time.txt"),fd=openSync(log,"wx");
 const start=now(),p=spawn("/usr/bin/time",["-f","%U %S %M","-o",time,lean,file],{
  cwd,env:{...process.env,...extra,LEAN_NUM_THREADS:"1"},detached:true,stdio:["ignore",fd,fd]});
 closeSync(fd);let spawnError=null,peak=0;
 p.on("error",e=>{spawnError=String(e);});
 const exited=()=>p.exitCode!==null||p.signalCode!==null||spawnError!==null;
 const sample=()=>{peak=Math.max(peak,rss(p.pid));};
 const kill=()=>{if(!exited()&&p.pid)try{process.kill(-p.pid,"SIGKILL");}catch{}};
 const wait=async deadline=>{while(!exited()&&now()<deadline){sample();await sleep(4);}return exited();};
 const info=()=>{
  let user=null,system=null,maxRss=null;
  if(existsSync(time)){const m=read(time).trim().split("\n").at(-1)?.match(/^([\d.]+) ([\d.]+) (\d+)$/);if(m){user=Number(m[1]);system=Number(m[2]);maxRss=Number(m[3]);}}
  return {exitCode:p.exitCode,signal:p.signalCode,spawnError,elapsedMs:now()-start,userSeconds:user,systemSeconds:system,gnuMaxRssKiB:maxRss,peakSampledTreeRssKiB:peak,logSha256:sha(readFileSync(log))};
 };
 return {p,log,sample,exited,kill,wait,info};
}
async function response(proc,folder,id,deadline){
 const p=join(folder,"response-"+id+".json");
 while(now()<deadline){
  proc.sample();
  if(existsSync(p))return JSON.parse(read(p));
  if(proc.exited())throw Error("Lean exited before response "+id+": "+read(proc.log).slice(-2000));
  await sleep(4);
 }throw Error("Lean response timeout "+id);
}
async function probe(c,arm,folder,lean,cwd){
 mkdirSync(folder);
 const file=join(folder,"Probe.lean"),src=source(c,[],true);writeFileSync(file,src,{flag:"wx"});
 const proc=launch(lean,cwd,folder,file,{COG13D_MAILBOX:folder,COG13D_CAP:String(arm.cap),COG13D_POLICY:arm.policy});
 const events=[];const started=now();let root=null,error=null,queryWallMs=0;
 try{
  root=await response(proc,folder,0,Math.min(started+30000,started+180000));
  check(root.ready===true&&Array.isArray(root.goals)&&root.goals.length>0,"invalid startup");
  for(const req of c.requests){
   const t=now();atomic(join(folder,"request-"+req.id+".json"),{id:req.id,actions:req.actions,stop:false});
   const e=await response(proc,folder,req.id,Math.min(t+10000,started+180000));
   const elapsed=now()-t;queryWallMs+=elapsed;
   events.push({...e,queryWallMs:elapsed});
  }
  atomic(join(folder,"request-"+(c.requests.length+1)+".json"),{id:c.requests.length+1,actions:[],stop:true});
  check(await proc.wait(now()+5000),"Lean server shutdown timeout");
  check(proc.p.exitCode===0,"Lean speculative server exit failure");
 }catch(e){error=String(e);}
 finally{if(!proc.exited()){proc.kill();await proc.wait(now()+1000);}}
 const receipt={root:c.id,arm,rootResponse:root,events,error,queryWallMs,process:proc.info(),sourceSha256:sha(src)};
 json(join(folder,"receipt.json"),receipt);
 return receipt;
}
async function certify(c,actions,folder,lean,cwd,negative=false){
 mkdirSync(folder);const src=source(c,actions,false),file=join(folder,"Certificate.lean");writeFileSync(file,src,{flag:"wx"});
 const proc=launch(lean,cwd,folder,file);
 const finished=await proc.wait(now()+30000);if(!finished){proc.kill();await proc.wait(now()+1000);}
 const log=read(proc.log),name=c.id;
 const no=log.includes("'"+name+"' does not depend on any axioms");
 const m=log.match(new RegExp("'"+name+"' depends on axioms: \\[([^\\]]*)\\]"));
 const ax=no?[]:m?m[1].split(",").map(s=>s.trim()).filter(Boolean):null;
 const approved=new Set(["propext","Classical.choice","Quot.sound"]);
 const passed=negative?finished&&proc.p.exitCode!==null&&proc.p.exitCode!==0&&log.includes("cog13d_missing_identifier")&&/error(?:\([^\n]*\))?:/.test(log):
  finished&&proc.p.exitCode===0&&ax!==null&&ax.every(x=>approved.has(x))&&!/sorryAx|error(?:\([^\n]*\))?:|declaration uses 'sorry'/.test(log);
 const receipt={passed,negative,sourceSha256:sha(src),axioms:ax,process:proc.info()};
 json(join(folder,"receipt.json"),receipt);return receipt;
}
async function native(out,lean,cwd,bytes){
 mkdirSync(out);const before=cgroup();limits(before,bytes);
 const cases=makeCases(),configs=arms(),summary={},runs=[],certificates=[],errors=[];
 json(join(out,"registry.json"),{cases,arms:configs,cgroup:before,requestedBytes:bytes});
 for(const c of cases){
  let base=null,rootGoals=null;
  for(const arm of configs){
   const r=await probe(c,arm,join(out,c.id+"_"+arm.name),lean,cwd);
   let a=null;
   try{
    check(!r.error,"probe infrastructure: "+r.error);
    check(r.process.exitCode===0,"probe exit");
    if(arm.cap===0){a=audit(c,arm,r.events);base=r.events;rootGoals=r.rootResponse.goals;}
    else {check(base!==null,"missing baseline");check(equal(rootGoals,r.rootResponse.goals),"changed initial goal");a=audit(c,arm,r.events,base);}
   }catch(e){errors.push({root:c.id,arm:arm.name,error:String(e)});}
   runs.push({root:c.id,arm:arm.name,audit:a,queryWallMs:r.queryWallMs,process:r.process,error:r.error});
   const s=summary[arm.name]??={expectedRoots:8,validRoots:0,requests:0,executed:0,restored:0,hits:0,insertions:0,evictions:0,peak:0,queryWallMs:0,processWallMs:0,completeCpuRecords:0,userSeconds:0,systemSeconds:0,peakProcessRssKiB:0};
   if(a){s.validRoots++;for(const k of ["requests","executed","restored","hits","insertions","evictions"])s[k]+=a[k];s.peak=Math.max(s.peak,a.peak);}
   s.queryWallMs+=r.queryWallMs;s.processWallMs+=r.process.elapsedMs;
   if(r.process.userSeconds!==null&&r.process.systemSeconds!==null){s.completeCpuRecords++;s.userSeconds+=r.process.userSeconds;s.systemSeconds+=r.process.systemSeconds;}
   s.peakProcessRssKiB=Math.max(s.peakProcessRssKiB,r.process.gnuMaxRssKiB??0,r.process.peakSampledTreeRssKiB);
  }
  for(let s=0;s<c.scripts.length;s++){
   const pair=[];
   for(let j=0;j<2;j++)pair.push(await certify(c,c.scripts[s],join(out,c.id+"_certificate_"+s+"_"+j),lean,cwd));
   if(!pair.every(x=>x.passed)||pair[0].sourceSha256!==pair[1].sourceSha256)errors.push({root:c.id,script:s,error:"independent compilation failed"});
   certificates.push({root:c.id,script:s,pair});
  }
 }
 const negative=await certify(cases[0],[badAction],join(out,"negative_certificate"),lean,cwd,true);
 if(!negative.passed)errors.push({error:"wrong identifier sentinel failed to reject"});
 const after=cgroup();limits(after,bytes);
 const complete=errors.length===0&&Object.values(summary).every(s=>s.validRoots===8);
 const result={schema:"cog13d.native.v1",requestedBytes:bytes,complete,errors,summary,runs,certificates,negative,cgroupBefore:before,cgroupAfter:after};
 json(join(out,"result.json"),result);console.log(JSON.stringify({complete,errors,summary,cgroupAfter:after},null,2));
 if(!complete)process.exitCode=1;
}
const [mode,output,lean,cwd,limit]=process.argv.slice(2);
check(output&&["--fixture","--native"].includes(mode),"usage: run.mjs --fixture OUT.json | --native NEW_DIR LEAN MATHLIB_ROOT CAP_BYTES (0=uncapped)");
if(mode==="--fixture"){
 const result=fixture();const frozen=JSON.parse(read(join(dirname(new URL(import.meta.url).pathname),"fixture.json")));
 check(equal(result,frozen),"fixture receipt mismatch");json(resolve(output),result);console.log(JSON.stringify(result));
}else{
 check(lean&&cwd&&limit!==undefined&&/^\d+$/.test(limit)&&process.platform==="linux","native Linux configuration required");
 await native(resolve(output),resolve(lean),resolve(cwd),Number(limit));
}
