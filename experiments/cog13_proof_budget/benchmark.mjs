// COG-13: independent portfolio allocation and ledger-cache replay pilot.
// Run with Node 20+: node experiments/cog13_proof_budget/benchmark.mjs TASKS.jsonl.gz OUTPUT.json
import {readFileSync,writeFileSync} from "node:fs";
import {gunzipSync} from "node:zlib";
import {createHash} from "node:crypto";

function requireThat(x,msg){if(!x)throw new Error(msg);}

function canonical(x){if(Array.isArray(x))return "["+x.map(canonical).join(",")+"]";if(x!==null&&typeof x==="object")return "{"+Object.keys(x).sort().map(k=>JSON.stringify(k)+":"+canonical(x[k])).join(",")+"}";return JSON.stringify(x);}

function validateTasks(tasks,hash){
 const counts={},roots=new Set(),worlds=new Set();
 for(const t of tasks){const {task_digest,...rest}=t;requireThat(t.schema_version==="cogentic.ledger.v1"&&hash(canonical(rest))===task_digest,"task digest");requireThat(["train","calibration","ordinary_test","ood_test"].includes(t.split),"split");const w=t.world,key=canonical(w);requireThat(!roots.has(t.root_id)&&!worlds.has(key),"duplicate root/world");roots.add(t.root_id);worlds.add(key);counts[t.split]=(counts[t.split]||0)+1;const a=new Set(w.atoms);requireThat(w.atoms.every(Number.isInteger)&&a.size===w.atoms.length&&a.has(w.goal)&&!w.facts.includes(w.goal)&&w.facts.every(x=>a.has(x))&&w.rules.every(r=>r.premises.length&&r.premises.every(x=>a.has(x))&&a.has(r.conclusion)),"invalid graph");}
 requireThat(tasks.length===256&&Object.values(counts).length===4&&Object.values(counts).every(n=>n===64),"unexpected cohort");return counts;
}

function context(t){
 const w=t.world,near=new Map([[w.goal,0]]),full=new Set([w.goal]);
 for(let depth=1;depth<=2;depth++){let old=new Set(near.keys());for(const r of w.rules)if(old.has(r.conclusion))for(const x of r.premises)if(!near.has(x))near.set(x,depth);}
 let prev=-1;while(prev!==full.size){prev=full.size;for(const r of w.rules)if(full.has(r.conclusion))for(const x of r.premises)full.add(x);}
 return {id:t.root_id,w,near,full};
}

function ready(w,k,i){const r=w.rules[i];return !k.has(r.conclusion)&&r.premises.every(x=>k.has(x));}

function choose(c,k,router){
 let best=-1,score=Infinity;
 c.w.rules.forEach((r,i)=>{if(!ready(c.w,k,i))return;const s=router==="full"?(c.full.has(r.conclusion)?0:1):(c.near.get(r.conclusion)??3);if(s<score){best=i;score=s;}});
 requireThat(best>=0,"unsolved root has no ready rule");return best;
}

function step(c,k,i){
 requireThat(Number.isInteger(i)&&i>=0&&i<c.w.rules.length&&ready(c.w,k,i),"illegal fresh rule");
 const x=c.w.rules[i].conclusion;k.add(x);return x;
}

function features(c,k){
 const w=c.w,unknown=w.rules.filter(r=>!k.has(r.conclusion)),readyN=unknown.filter(r=>r.premises.every(x=>k.has(x))).length;
 const missingGoal=Math.min(...w.rules.filter(r=>r.conclusion===w.goal).map(r=>r.premises.filter(x=>!k.has(x)).length));
 return [1,w.rules.length/64,w.atoms.length/64,(k.size-w.facts.length)/64,(w.atoms.length-k.size)/64,unknown.length/64,readyN/64,missingGoal/4,[...c.near.keys()].filter(x=>!k.has(x)).length/16];
}

function trajectory(c,router){
 const k=new Set(c.w.facts),xs=[],actions=[];
 while(!k.has(c.w.goal)){xs.push(features(c,k));let i=choose(c,k,router);step(c,k,i);actions.push(i);requireThat(actions.length<=c.w.rules.length,"nonterminating policy");}
 return {xs,actions};
}

function fit(train){
 const n=9,A=Array.from({length:n},()=>Array(n+1).fill(0));let labels=0;
 for(const c of train){const {xs,actions}=trajectory(c,"local"),w=1/actions.length;labels+=actions.length;for(let j=0;j<xs.length;j++){const x=xs[j],y=actions.length-j;for(let a=0;a<n;a++){for(let b=0;b<n;b++)A[a][b]+=w*x[a]*x[b];A[a][n]+=w*x[a]*y;}}}
 for(let a=0;a<n;a++)A[a][a]+=0.01;
 for(let a=0;a<n;a++){let p=a;for(let b=a+1;b<n;b++)if(Math.abs(A[b][a])>Math.abs(A[p][a]))p=b;[A[p],A[a]]=[A[a],A[p]];let v=A[a][a];requireThat(Math.abs(v)>1e-12,"singular fit");for(let b=a;b<=n;b++)A[a][b]/=v;for(let j=0;j<n;j++)if(j!==a){let q=A[j][a];for(let b=a;b<=n;b++)A[j][b]-=q*A[a][b];}}
 return {weights:A.map(r=>r[n]),trainingVerifierCalls:labels,trainingRoots:train.length};
}

function cacheBytes(cache,cs){
 return JSON.stringify([...cache].map(([i,k])=>[cs[i].id,[...k].sort((a,b)=>a-b)])).length;
}

function tapeBytes(tapes,cs){
 return JSON.stringify(tapes.map((t,i)=>[cs[i].id,t])).length;
}

function run(cs,policy,capacity,weights,oracleCosts,budgets){
 const max=budgets.at(-1),router=policy==="full_ancestry"?"full":"local",cache=new Map(),tapes=cs.map(()=>[]),done=new Set(),events=[],snaps={};
 let calls=0,replays=0,rr=0,evictions=0,peakCache=0,peakTape=tapeBytes(tapes,cs);
 const measure=()=>{peakCache=Math.max(peakCache,cacheBytes(cache,cs));peakTape=Math.max(peakTape,tapeBytes(tapes,cs));};
 const snap=()=>{if(budgets.includes(calls))snaps[calls]={completed:done.size,calls,replays};};
 const rank=i=>{const c=cs[i],k=cache.get(i)||new Set(c.w.facts);
 if(policy==="static_size")return c.w.rules.length;
 if(policy==="remaining_rules")return c.w.rules.filter(r=>!k.has(r.conclusion)).length;
 if(policy==="ancestry_priority"||policy==="full_ancestry")return [...c.full].filter(x=>!k.has(x)).length;
 if(policy==="clairvoyant")return oracleCosts[i]-tapes[i].length;
 if(policy==="learned")return Math.max(1,features(c,k).reduce((s,x,j)=>s+x*weights[j],0));
 throw Error("unknown policy");};
 while(calls<max&&done.size<cs.length){
  let i;if(policy==="round_robin"){while(done.has(rr))rr=(rr+1)%cs.length;i=rr;rr=(rr+1)%cs.length;}
  else{requireThat(capacity===cs.length,"non-RR scheduler requires resident states");i=cs.map((_,j)=>j).filter(j=>!done.has(j)).sort((a,b)=>rank(a)-rank(b)||a-b)[0];}
  const c=cs[i];let k=cache.get(i);
  if(!k){let victim=null;if(cache.size===capacity){victim=cache.keys().next().value;cache.delete(victim);evictions++;}k=new Set(c.w.facts);cache.set(i,k);events.push({kind:"load",root:i,evicted:victim});measure();
    for(const a of tapes[i]){if(calls===max)break;const fact=step(c,k,a);calls++;replays++;events.push({kind:"replay",root:i,action:a,fact});measure();snap();}
    if(calls===max)break;
  }else{cache.delete(i);cache.set(i,k);}
  let a=choose(c,k,router),fact=step(c,k,a);calls++;tapes[i].push(a);events.push({kind:"new",root:i,action:a,fact});measure();
  if(k.has(c.w.goal)){done.add(i);cache.delete(i);}snap();
 }
 for(const b of budgets)if(!snaps[b])snaps[b]={completed:done.size,calls,replays};
 return {policy,capacity,roots:cs.map(c=>c.id),budget:max,calls,replays,evictions,peakCache,peakTape,completed:done.size,snaps,events};
}

function audit(cs,r,budgets){
 // Independent array-based checker. Reconstruct actual cache contents and
 // require the immutable tape prefix to be replayed before fresh work.
 const cache=new Map(),tapes=cs.map(()=>[]),done=new Set(),snaps={};let calls=0,replays=0,evictions=0,peakCache=0,peakTape=0;
 const measure=()=>{peakCache=Math.max(peakCache,JSON.stringify([...cache].map(([i,s])=>[cs[i].id,[...s.known].sort((a,b)=>a-b)])).length);peakTape=Math.max(peakTape,JSON.stringify(tapes.map((t,i)=>[cs[i].id,t])).length);};
 measure();
 requireThat(canonical(r.roots)===canonical(cs.map(c=>c.id))&&r.budget===budgets.at(-1)&&Number.isInteger(r.capacity)&&r.capacity>0&&r.capacity<=cs.length,"cohort or budget mismatch");
 for(const e of r.events){const i=e.root;requireThat(Number.isInteger(i)&&i>=0&&i<cs.length&&!done.has(i)&&calls<r.budget,"invalid event");const w=cs[i].w;
 if(e.kind==="load"){requireThat(!cache.has(i),"duplicate load");let victim=null;if(cache.size===r.capacity){victim=cache.keys().next().value;cache.delete(victim);evictions++;}requireThat(victim===e.evicted,"wrong eviction");cache.set(i,{known:[...w.facts],cursor:0});measure();continue;}
 const s=cache.get(i);requireThat(s&&["new","replay"].includes(e.kind),"missing cache or event kind");const a=e.action;requireThat(Number.isInteger(a)&&a>=0&&a<w.rules.length,"bad action");const rule=w.rules[a];requireThat(rule.premises.every(p=>s.known.includes(p))&&!s.known.includes(rule.conclusion)&&e.fact===rule.conclusion,"forged fact");
 if(e.kind==="replay"){requireThat(s.cursor<tapes[i].length&&tapes[i][s.cursor]===a,"wrong tape prefix");replays++;}
 else{requireThat(s.cursor===tapes[i].length,"fresh work before restoration");tapes[i].push(a);}
 s.cursor++;s.known.push(rule.conclusion);cache.delete(i);cache.set(i,s);calls++;measure();
 if(s.known.includes(w.goal)){requireThat(e.kind==="new","replayed completed root");done.add(i);cache.delete(i);}
 if(budgets.includes(calls))snaps[calls]={completed:done.size,calls,replays};
 }
 for(const b of budgets)if(!snaps[b])snaps[b]={completed:done.size,calls,replays};
 requireThat(calls===r.calls&&replays===r.replays&&evictions===r.evictions&&done.size===r.completed&&canonical(snaps)===canonical(r.snaps)&&peakCache===r.peakCache&&peakTape===r.peakTape,"receipt mismatch");
 requireThat(calls===r.budget||done.size===cs.length,"premature stop");return calls;
}

function suite(tasks){
 const train=tasks.filter(t=>t.split==="train").map(context),model=fit(train),budgets=[48,96,144],runs=[],summary={},arms=[["round_robin",8],["static_size",8],["remaining_rules",8],["ancestry_priority",8],["learned",8],["full_ancestry",8],["clairvoyant",8],["round_robin",1],["round_robin",2],["round_robin",4]];
 let auditCalls=0,offlineCalls=0,negativeSentinel=false;
 for(const split of ["calibration","ordinary_test","ood_test"]){
  const cs=tasks.filter(t=>t.split===split).map(context),label=cs.map(c=>trajectory(c,"local").actions.length);offlineCalls+=label.reduce((a,b)=>a+b,0);const agg={};summary[split]=agg;
  for(let b=0;b<cs.length;b+=8){const batch=cs.slice(b,b+8),oracle=label.slice(b,b+8);
   for(const [p,cap] of arms){const key=p+"@"+cap,r=run(batch,p,cap,model.weights,p==="clairvoyant"?oracle:null,budgets);auditCalls+=audit(batch,r,budgets);runs.push({split,batch:b/8,...r});
    const a=agg[key]??={completed:[0,0,0],calls:[0,0,0],replays:[0,0,0],evictions:0,maxCacheBytes:0,maxTapeBytes:0,batches:0};a.batches++;budgets.forEach((v,j)=>{a.completed[j]+=r.snaps[v].completed;a.calls[j]+=r.snaps[v].calls;a.replays[j]+=r.snaps[v].replays;});a.evictions+=r.evictions;a.maxCacheBytes=Math.max(a.maxCacheBytes,r.peakCache);a.maxTapeBytes=Math.max(a.maxTapeBytes,r.peakTape);
    if(!negativeSentinel&&r.events.some(e=>e.kind==="new")){const forged=JSON.parse(JSON.stringify(r));forged.events.find(e=>e.kind==="new").fact=-1;try{audit(batch,forged,budgets);}catch{negativeSentinel=true;}}
   }
  }
  for(let j=0;j<budgets.length;j++)for(const [key,a] of Object.entries(agg))if(!key.startsWith("full_ancestry"))requireThat(a.completed[j]<=agg["clairvoyant@8"].completed[j],"clairvoyant completion bound violated");
 }
 requireThat(negativeSentinel,"negative sentinel accepted");
 return {summary:{schema:"cog13.proof_budget.v1",evidence:"public synthetic development; not confirmation",budgetsPerEightRoots:budgets,model,offlineDiagnosticVerifierCalls:offlineCalls,independentReplay:{runs:runs.length,calls:auditCalls,forgedFactRejected:negativeSentinel},splits:summary},runs};
}

const [, , input, output] = process.argv;
requireThat(input && output, "usage: node benchmark.mjs TASKS.jsonl.gz OUTPUT.json");
const hash = v => createHash("sha256").update(v).digest("hex");
const compressed = readFileSync(input);
requireThat(hash(compressed) === "cf8292144cfd37b42c9a8a4d86ac1517ef4c6cb4c26400210d1a840de310a83c", "compressed source SHA mismatch");
const plain = gunzipSync(compressed);
requireThat(hash(plain) === "aef151e3f7de099e9b5d7b51c92ad94aa3bab20a85b4f73b8f4fb5ac8ce7905a", "plain source SHA mismatch");
const tasks = plain.toString("utf8").trim().split("\n").map(JSON.parse);
validateTasks(tasks, hash);
const result = suite(tasks);
writeFileSync(output, JSON.stringify(result, null, 2)+"\n", {flag:"wx"});
console.log(JSON.stringify(result.summary, null, 2));
