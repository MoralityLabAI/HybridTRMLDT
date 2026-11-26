// Native cgroup-backed executor. No generated memory ballast.
import {spawn} from "node:child_process";
import {readFileSync,writeFileSync,appendFileSync,mkdirSync,existsSync,openSync,closeSync,renameSync} from "node:fs";
import {createHash} from "node:crypto";
import {join,resolve,dirname} from "node:path";
import {fileURLToPath} from "node:url";
import {performance} from "node:perf_hooks";
import {BAD,check,canonical,equal,makeCases,arms,portfolio,audit,fixture} from "./core.mjs";

const read=p=>readFileSync(p,"utf8"),sha=b=>createHash("sha256").update(b).digest("hex");
const json=(p,v)=>writeFileSync(p,JSON.stringify(v,null,2)+"\n",{flag:"wx"});
const now=()=>performance.now(),sleep=ms=>new Promise(r=>setTimeout(r,ms));
const allowed=new Set(["propext","Classical.choice","Quot.sound"]);
function metrics(s){return Object.fromEntries(s.trim().split("\n").map(x=>x.trim().split(/\s+/)).filter(x=>x.length>=2).map(([k,v])=>[k,Number(v)]));}
function cgroup(bytes){
 const relative=read("/proc/self/cgroup").split("\n").find(x=>x.startsWith("0::"));
 check(relative,"cgroup v2 required");const path=join("/sys/fs/cgroup",relative.slice(3));
 const get=n=>read(join(path,n)).trim();
 const max=get("memory.max"),swap=get("memory.swap.max"),[quota,period]=get("cpu.max").split(/\s+/);
 check(max===String(bytes)&&swap==="0"&&Number(quota)===Number(period)&&Number(period)>0,"hard cgroup or CPUQuota mismatch");
 function sample(){
  const cpu=metrics(get("cpu.stat")),m=metrics(get("memory.stat")),e=metrics(get("memory.events"));
  return {cpu:cpu.usage_usec,current:Number(get("memory.current")),peak:Number(get("memory.peak")),anon:m.anon,file:m.file,maxEvents:e.max,oomEvents:e.oom,oomKills:e.oom_kill};
 }
 return {path,max,swap,quota:Number(quota),period:Number(period),sample};
}
function anonTree(pid,seen=new Set()){
 if(!pid||seen.has(pid))return 0;seen.add(pid);
 try{
  const s=read("/proc/"+pid+"/status"),m=Number(s.match(/^RssAnon:\s+(\d+)\s+kB/m)?.[1]??0)*1024;
  const children=read("/proc/"+pid+"/task/"+pid+"/children").trim().split(/\s+/).filter(Boolean).map(Number);
  return m+children.reduce((a,x)=>a+anonTree(x,seen),0);
 }catch{return 0;}
}
function declaration(c,actions,name,probe=false){
 check(/^[A-Za-z][A-Za-z0-9_]*$/.test(name),"theorem name");
 return "theorem "+name+" "+c.statement+" := by\n"+(probe?"  cog13d_probe\n":actions.map(a=>"  "+a+"\n").join(""))+"#print axioms "+name+"\n";
}
function source(c,actions,probe=false){
 return "import Mathlib\n"+(probe?"import StateCacheServer\n":"")+"set_option maxHeartbeats 100000000\n"+declaration(c,actions,c.id,probe);
}
function axiomList(log,name){
 if(log.includes("'"+name+"' does not depend on any axioms"))return [];
 const m=log.match(new RegExp("'"+name+"' depends on axioms: \\[([^\\]]*)\\]"));
 return m?m[1].split(",").map(x=>x.trim()).filter(Boolean):null;
}
function cleanLog(s){return !/sorryAx|error(?:\([^\n]*\))?:|declaration uses 'sorry'/.test(s);}
function atomic(p,v){const t=p+".tmp";writeFileSync(t,JSON.stringify(v),{flag:"wx"});renameSync(t,p);}
function launch(lean,cwd,folder,file,extra={}){
 const log=join(folder,"stdout.log"),timing=join(folder,"time.txt"),fd=openSync(log,"wx"),t0=now();
 const p=spawn("/usr/bin/time",["-f","%U %S %M","-o",timing,lean,file],{cwd,env:{...process.env,LEAN_NUM_THREADS:"1",...extra},detached:true,stdio:["ignore",fd,fd]});closeSync(fd);
 let spawnError=null;p.on("error",e=>{spawnError=String(e);});
 const exited=()=>p.exitCode!==null||p.signalCode!==null||spawnError!==null;
 const kill=()=>{if(!exited()&&p.pid)try{process.kill(-p.pid,"SIGKILL");}catch{}};
 async function wait(until,within=()=>true){while(!exited()&&now()<until&&within())await sleep(5);return exited();}
 const info=()=>{let userSeconds=null,systemSeconds=null,maxRssKiB=null;if(existsSync(timing)){const m=read(timing).trim().split("\n").at(-1)?.match(/^([\d.]+) ([\d.]+) (\d+)$/);if(m){userSeconds=Number(m[1]);systemSeconds=Number(m[2]);maxRssKiB=Number(m[3]);}}
 return {exitCode:p.exitCode,signal:p.signalCode,spawnError,userSeconds,systemSeconds,maxRssKiB,wallMs:now()-t0,logSha256:sha(readFileSync(log))};};
 return {p,log,exited,kill,wait,info,anon:()=>anonTree(p.pid)};
}
async function reply(proc,folder,id,until,within,onPoll=()=>{}){
 const path=join(folder,"response-"+id+".json");
 while(now()<until&&within()){
  onPoll();
  if(existsSync(path))return JSON.parse(read(path));
  if(proc.exited())throw Error("Lean exited before reply "+id+": "+read(proc.log).slice(-1400));
  await sleep(5);
 }
 throw Error("probe timeout or charged CPU/wall budget");
}
async function stop(proc,grace=500){
 if(!proc.exited()){await proc.wait(now()+grace);if(!proc.exited()){proc.kill();await proc.wait(now()+1000);}}
 check(proc.exited(),"process failed to terminate");
 return proc.info();
}
async function compile(lean,cwd,folder,src,within,deadlineMs=30000){
 mkdirSync(folder);const p=join(folder,"Certificate.lean");writeFileSync(p,src,{flag:"wx"});
 const proc=launch(lean,cwd,folder,p);const finished=await proc.wait(now()+deadlineMs,within);
 if(!finished)proc.kill();
 const info=await stop(proc,0),log=read(proc.log);
 return {finished,sourceHash:sha(src),logSha256:sha(log),process:info,log,folder};
}
function nativeBackend(out,lean,cwd,cg){
 let n=0;
 return {sample:cg.sample,now,event:e=>appendFileSync(join(out,"events.jsonl"),JSON.stringify(e)+"\n"),
 open:async(c,slots,session,within)=>{
  const folder=join(out,"session_"+session);mkdirSync(folder);
  const file=join(folder,"Probe.lean"),src=source(c,[],true);writeFileSync(file,src,{flag:"wx"});
  const before=cg.sample().current;let peak=before,number=0,closed=false;
  const proc=launch(lean,cwd,folder,file,{COG13D_MAILBOX:folder,COG13D_CAP:String(slots),COG13D_POLICY:"lru"});
  try{
   const r=await reply(proc,folder,0,now()+40000,within,()=>{peak=Math.max(peak,cg.sample().current);});
   check(r.ready===true&&Array.isArray(r.goals)&&r.goals.length>0,"invalid native root");
   return {initial:r.goals,peakDelta:Math.max(0,peak-before),anon:proc.anon,
    query:async(req,within)=>{
     check(!closed&&req.id===number+1,"session request sequence");number++;
     atomic(join(folder,"request-"+number+".json"),{id:number,actions:req.actions,stop:false});
     return await reply(proc,folder,number,now()+20000,within);
    },
    close:async()=>{
     if(closed)return null;closed=true;
     if(!proc.exited()){number++;try{atomic(join(folder,"request-"+number+".json"),{id:number,actions:[],stop:true});}catch{}}
     const info=await stop(proc);json(join(folder,"process.json"),{...info,sourceHash:sha(src)});return info;
    }};
  }catch(e){proc.kill();const info=await stop(proc,0);json(join(folder,"startup_failure.json"),{error:String(e),process:info,sourceHash:sha(src)});throw e;}
 },
 certify:async(c,actions,rep,within)=>{
  const folder=join(out,"cert_"+c.id+"_"+rep+"_"+n++),src=source(c,actions),r=await compile(lean,cwd,folder,src,within);
  const axioms=axiomList(r.log,c.id),accepted=r.finished&&r.process.exitCode===0&&axioms!==null&&axioms.every(x=>allowed.has(x))&&cleanLog(r.log);
  const receipt={accepted,sourceHash:r.sourceHash,logSha256:r.logSha256,axioms,folder:folder.split("/").at(-1),process:r.process};
  json(join(folder,"receipt.json"),receipt);return receipt;
 }};
}
async function preflight(out,lean,cwd){
 mkdirSync(out);const cases=makeCases();
 const names=[],decls=[];
 for(const c of cases)for(let j=0;j<2;j++){const name=c.id+"_ref"+j;names.push(name);decls.push(declaration(c,c.scripts[j],name));}
 const src="import Mathlib\nset_option maxHeartbeats 100000000\n"+decls.join("\n");
 const ref=await compile(lean,cwd,join(out,"reference"),src,()=>true,300000);
 const inventories=names.map(name=>({name,axioms:axiomList(ref.log,name)}));
 const neg=await compile(lean,cwd,join(out,"negative"),source(cases[0],[BAD]),()=>true,30000);
 const negative=neg.finished&&neg.process.exitCode!==null&&neg.process.exitCode!==0&&/error(?:\([^\n]*\))?:/.test(neg.log)&&neg.log.includes("cog13d_missing_identifier");
 const passed=ref.finished&&ref.process.exitCode===0&&cleanLog(ref.log)&&inventories.every(r=>r.axioms!==null&&r.axioms.every(x=>allowed.has(x)))&&negative;
 const {log:_,...reference}=ref,{log:__,...negativeProcess}=neg;
 const result={passed,registryHash:sha(canonical(cases)),leanHash:sha(readFileSync(lean)),references:names.length,inventories,reference,negative,negativeProcess};
 json(join(out,"receipt.json"),result);console.log(JSON.stringify({passed,references:names.length,negative}));check(passed,"native reference preflight failed");
}
function verifyCertificateFiles(out,result){
 for(const r of result.roots){
  for(const c of r.certificates){
   if(!c.folder){check(!c.accepted,"accepted missing certificate");continue;}
   const folder=join(out,c.folder),src=readFileSync(join(folder,"Certificate.lean")),log=readFileSync(join(folder,"stdout.log"));
   check(sha(src)===c.sourceHash&&sha(log)===c.logSha256,"certificate bytes changed");
   const expected=source(r.c,r.c.scripts[0]);check(src.toString("utf8")===expected,"certificate source is not registered script");
   if(c.accepted){const axioms=axiomList(log.toString("utf8"),r.c.id);check(c.process.exitCode===0&&axioms!==null&&axioms.every(x=>allowed.has(x))&&cleanLog(log.toString("utf8")),"false native certificate");}
  }
 }
}
async function native(out,lean,cwd,bytes,armName,preflightPath){
 mkdirSync(out);const cases=makeCases(),arm=arms().find(a=>a.name===armName);check(arm,"unknown arm");
 const pre=JSON.parse(read(preflightPath));check(pre.passed&&pre.registryHash===sha(canonical(cases))&&pre.leanHash===sha(readFileSync(lean))&&pre.references===64&&pre.negative,"preflight receipt mismatch");
 const cg=cgroup(bytes),limit={bytes,cpu:180e6,wallMs:480000};
 json(join(out,"registry.json"),{cases,arm,limit,cgroup:{path:cg.path,max:cg.max,swap:cg.swap,quota:cg.quota,period:cg.period},preflightSha256:sha(readFileSync(preflightPath))});
 const result=await portfolio(nativeBackend(out,lean,cwd,cg),arm,cases,limit);
 json(join(out,"result.json"),result);
 let a=null,error=null;
 try{check(!result.error,"execution evidence error: "+result.error);a=audit(result);verifyCertificateFiles(out,result);}catch(e){error=String(e);}
 const summary={capGiB:bytes/1073741824,arm:armName,audit:a,auditError:error,certified:result.certified,cpuSeconds:result.cpu/1e6,wallSeconds:result.wall/1000,perCpuHour:result.perCpuHour,overrunCpuSeconds:result.overrunCpu/1e6,peakBytes:result.end.peak,pressureObserved:result.pressureObserved,oom:result.oom,peakLive:result.peakLive,opens:result.opens,pressureEvictions:result.pressureEvictions,executed:result.executed,restored:result.restored,unknown:result.unknown,replied:result.replied,attempted:result.attempted,stop:result.stop};
 json(join(out,"summary.json"),summary);console.log(JSON.stringify(summary));
 if(error||result.oom)process.exitCode=1;
}
const [mode,out,lean,cwd,cap,arm,pre]=process.argv.slice(2);
check(out&&["--fixture","--preflight","--native"].includes(mode),"mode and new output required");
if(mode==="--fixture"){
 const got=await fixture(),expected=JSON.parse(read(join(dirname(fileURLToPath(import.meta.url)),"fixture.json")));check(equal(got,expected),"fixture receipt changed");json(resolve(out),got);console.log(JSON.stringify(got));
}else if(mode==="--preflight"){check(lean&&cwd,"Lean path and Mathlib root required");await preflight(resolve(out),resolve(lean),resolve(cwd));}
else{check(lean&&cwd&&["2147483648","4294967296","8589934592"].includes(cap)&&arm&&pre,"native parameters");await native(resolve(out),resolve(lean),resolve(cwd),Number(cap),arm,resolve(pre));}
