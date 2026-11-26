// COG-13B. Node 20+, standard library only. Fresh synthetic development worlds.
import {writeFileSync} from "node:fs";
function insist(v,m){if(!v)throw Error(m);}

function rng32(seed){let s=seed>>>0;return n=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)%n;};}

function shuffled(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=r(i+1);[a[i],a[j]]=[a[j],a[i]];}return a;}

function makeTasks(){
 const seed=2026112603,splits=["train","calibration","ordinary","deeper"],tasks=[],rawSeen=new Set(),seen=new Set();
 for(let s=0;s<4;s++){const r=rng32(seed+s*1000003);for(let i=0;i<64;i++){
  const family=i%4,depth=s===3?6+r(5):2+r(4),facts=Array.from({length:3+r(3)},(_,j)=>j),rules=[],pool=[...facts];let prev=[...facts],n=pool.length;
  const add=(p,c)=>rules.push({premises:[...new Set(p)].sort((a,b)=>a-b),conclusion:c});
  for(let d=0;d<depth;d++){let layer=[];const width=family===0?1:2+r(2);for(let j=0;j<width;j++){let p=[prev[r(prev.length)]];if(family!==0)p.push(...shuffled(pool,r).slice(0,1+r(2)));let c=n++;add(p,c);layer.push(c);if(family===2&&r(3)!==0)add([prev[r(prev.length)],facts[r(facts.length)]],c);}prev=layer;pool.push(...layer);}
  const goal=n++;add(prev,goal);let decoys=[...facts];for(let j=0,m=5+r(5)+(family===3?depth:0);j<m;j++){let c=n++;add(shuffled(decoys,r).slice(0,1+r(3)),c);decoys.push(c);}
  for(let j=0,m=2+r(3);j<m;j++){let unavailable=n++,c=n++;add([facts[r(facts.length)],unavailable],c);}
  let raw={n,facts,rules,goal};const key=JSON.stringify(raw);insist(!rawSeen.has(key),"duplicate unpermuted world");rawSeen.add(key);const perm=shuffled(Array.from({length:n},(_,j)=>j),r);
  const w={n,facts:facts.map(x=>perm[x]).sort((a,b)=>a-b),rules:shuffled(rules.map(z=>({premises:z.premises.map(x=>perm[x]).sort((a,b)=>a-b),conclusion:perm[z.conclusion]})),r),goal:perm[goal]};
  insist(n<128&&w.rules.length<128,"encoding limit");const fingerprint=JSON.stringify(w);insist(!seen.has(fingerprint),"duplicate presented world");seen.add(fingerprint);tasks.push({id:`${s}:${i}`,split:splits[s],family,depth,w});
 }}
 return tasks;
}

function meter(){return {verifier:0,replay:0,ruleVisits:0,membership:0,rank:0,feature:0,dot:0,checkpointRead:0,checkpointWrite:0,tapeRead:0,tapeWrite:0};}

function units(m,w){return w*m.verifier+m.ruleVisits+m.membership+m.rank+m.feature+m.dot+m.checkpointRead+m.checkpointWrite+m.tapeRead+m.tapeWrite;}

function has(k,x,m){m.membership++;return k.has(x);}

function context(t,m,full){
 const w=t.w,near=new Map([[w.goal,0]]);
 for(let d=1;d<=(full?w.n:2);d++){const old=new Set(near.keys());for(const r of w.rules){m.ruleVisits++;if(has(old,r.conclusion,m)){for(const x of r.premises){m.membership++;if(!near.has(x))near.set(x,d);}}}if(near.size===old.size)break;}
 return {w,near,full};
}

function ready(c,k,r,m){return !has(k,r.conclusion,m)&&r.premises.every(p=>has(k,p,m));}

function propose(c,k,m){let best=-1,score=Infinity;for(let i=0;i<c.w.rules.length;i++){const r=c.w.rules[i];m.ruleVisits++;if(!ready(c,k,r,m))continue;m.membership++;const s=c.full?(c.near.has(r.conclusion)?0:1):(c.near.get(r.conclusion)??3);if(s<score){best=i;score=s;}}insist(best>=0,"no ready candidate");return best;}

function apply(c,k,a,m,isReplay){m.verifier++;if(isReplay)m.replay++;insist(Number.isInteger(a)&&a>=0&&a<c.w.rules.length,"invalid action");m.ruleVisits++;const r=c.w.rules[a];insist(ready(c,k,r,m),"invalid proof");k.add(r.conclusion);return r.conclusion;}

function feat(c,k,m){const w=c.w;let u=0,rn=0,mg=Infinity,ng=0;
 for(const r of w.rules){m.ruleVisits++;if(!has(k,r.conclusion,m)){u++;if(r.premises.every(p=>has(k,p,m)))rn++;}if(r.conclusion===w.goal){let miss=0;for(const p of r.premises)if(!has(k,p,m))miss++;mg=Math.min(mg,miss);}}
 for(const a of c.near.keys())if(!has(k,a,m))ng++;m.feature+=9;
 return [1,w.rules.length/64,w.n/64,(k.size-w.facts.length)/64,(w.n-k.size)/64,u/64,rn/64,mg/4,ng/16];
}

function fit(tasks){const m=meter(),n=9,A=Array.from({length:n},()=>Array(n+1).fill(0));
 for(const t of tasks){const c=context(t,m,false),k=new Set(c.w.facts),xs=[];while(!has(k,c.w.goal,m)){xs.push(feat(c,k,m));apply(c,k,propose(c,k,m),m,false);}for(let j=0;j<xs.length;j++){const x=xs[j],y=xs.length-j,w=1/xs.length;for(let a=0;a<n;a++){for(let b=0;b<n;b++)A[a][b]+=w*x[a]*x[b];A[a][n]+=w*x[a]*y;}}}
 for(let a=0;a<n;a++)A[a][a]+=.01;for(let a=0;a<n;a++){let p=a;for(let b=a+1;b<n;b++)if(Math.abs(A[b][a])>Math.abs(A[p][a]))p=b;[A[p],A[a]]=[A[a],A[p]];const v=A[a][a];insist(Math.abs(v)>1e-12,"singular");for(let b=a;b<=n;b++)A[a][b]/=v;for(let j=0;j<n;j++)if(j!==a){const q=A[j][a];for(let b=a;b<=n;b++)A[j][b]-=q*A[a][b];}}
 return {weights:A.map(x=>x[n]),collection:m,roots:tasks.length};
}

function pred(c,k,m,weights){m.dot+=weights.length;return Math.max(1,feat(c,k,m).reduce((a,x,i)=>a+x*weights[i],0));}

function pack(k,n){const a=Array(Math.ceil(n/8)).fill(0);for(const x of k)a[x>>3]|=1<<(x&7);return a;}

function unpack(b,n){insist(b.length===Math.ceil(n/8),"checkpoint size");const k=new Set();for(let i=0;i<n;i++)if(b[i>>3]&(1<<(i&7)))k.add(i);return k;}

function equal(a,b){return JSON.stringify(a)===JSON.stringify(b);}

function run(batch,arm,weights){
 const m=meter(),cs=batch.map(t=>context(t,m,arm.router==="full")),live=new Map(),back=new Map(),tapes=cs.map(()=>[]),done=new Set(),events=[],starts=cs.map(()=>0),est=cs.map(c=>arm.policy==="adaptive"?pred(c,new Set(c.w.facts),m,weights):0);
 const budgets=[48,96,144],costWeights=[10,100,1000],callCurve=budgets.map(()=>0),weighted=Object.fromEntries(costWeights.map(w=>[w,budgets.map(()=>0)]));
 let pointer=0,evictions=0,peakPacked=0,peakLive=0,peakBack=0,peakTape=0,peakPayload=0,started=0,longSolved=0,shortSolved=0;
 const measure=()=>{const l=[...live.keys()].reduce((a,i)=>a+Math.ceil(cs[i].w.n/8),0),b=[...back.values()].reduce((a,x)=>a+x.length,0),t=tapes.reduce((a,x)=>a+x.length+2,0);peakLive=Math.max(peakLive,l);peakBack=Math.max(peakBack,b);peakPacked=Math.max(peakPacked,l+b);peakTape=Math.max(peakTape,t);peakPayload=Math.max(peakPayload,l+b+t);};
 const receipt=(e)=>{measure();events.push({...e,cost:{...m},solved:done.size});for(let j=0;j<budgets.length;j++){if(m.verifier<=budgets[j])callCurve[j]=done.size;for(const w of costWeights)if(units(m,w)<=w*budgets[j])weighted[w][j]=done.size;}};
 const touch=(i,k)=>{live.delete(i);live.set(i,k);};
 receipt({kind:"setup"});
 while(m.verifier<144&&done.size<cs.length){
  let i=-1;
  if(arm.policy==="rr"){for(let j=0;j<cs.length;j++){m.rank++;const a=(pointer+j)%cs.length;if(!done.has(a)){i=a;pointer=(a+1)%cs.length;break;}}}
  else{let best=Infinity;for(let j=0;j<cs.length;j++){m.rank++;if(done.has(j))continue;let s=arm.policy==="static"?cs[j].w.rules.length:est[j]+(live.has(j)||back.has(j)?0:tapes[j].length);if(s<best){i=j;best=s;}}}
  insist(i>=0,"scheduler");const c=cs[i];let k=live.get(i);
  if(!k){let victim=null,saved=null,restored=null;if(live.size===arm.slots){victim=live.keys().next().value;const v=live.get(victim);if(arm.backing==="packed"){saved=pack(v,cs[victim].w.n);back.set(victim,saved);m.checkpointWrite+=saved.length;}live.delete(victim);evictions++;}
   if(back.has(i)){restored=back.get(i);back.delete(i);m.checkpointRead+=restored.length;k=unpack(restored,c.w.n);}else k=new Set(c.w.facts);touch(i,k);receipt({kind:"load",root:i,victim,saved,restored});
   if(!restored){for(const a of tapes[i]){if(m.verifier>=144)break;m.tapeRead++;const fact=apply(c,k,a,m,true);receipt({kind:"replay",root:i,action:a,fact});}}
   if(m.verifier>=144)break;
  }else{touch(i,k);receipt({kind:"select",root:i});}
  let q=arm.quantum==="block"?144:arm.quantum==="adaptive"?Math.max(1,Math.min(8,Math.ceil(est[i]))):arm.quantum;
  for(let j=0;j<q&&m.verifier<144&&!done.has(i);j++){
   const a=propose(c,k,m),fact=apply(c,k,a,m,false);if(tapes[i].length===0)started++;tapes[i].push(a);m.tapeWrite++;starts[i]++;touch(i,k);
   if(has(k,c.w.goal,m)){done.add(i);live.delete(i);back.delete(i);if(c.w.rules.length>=40)longSolved++;else shortSolved++;}
   if(arm.policy==="adaptive"&&!done.has(i))est[i]=pred(c,k,m,weights);
   receipt({kind:"new",root:i,action:a,fact});
  }
 }
 return {arm,roots:batch.map(t=>t.id),calls:m.verifier,replays:m.replay,completed:done.size,started,longSolved,shortSolved,evictions,peakPacked,peakLive,peakBack,peakTape,peakPayload,counts:m,callCurve,weighted,events};
}

function audit(batch,r){
 const live=new Map(),back=new Map(),tapes=batch.map(()=>[]),done=new Set();let calls=0,replays=0,reads=0,writes=0,tapeRead=0,tapeWrite=0,evictions=0,started=0,long=0,short=0,peakPacked=0,peakLive=0,peakBack=0,peakTape=0,peakPayload=0,prev=null;
 const callsCurve=[0,0,0],weighted={10:[0,0,0],100:[0,0,0],1000:[0,0,0]},budgets=[48,96,144];
 insist(equal(r.roots,batch.map(t=>t.id))&&r.events[0].kind==="setup","cohort/setup");
 const bytes=(facts,n)=>Array.from({length:Math.ceil(n/8)},(_,j)=>facts.filter(x=>(x>>3)===j).reduce((a,x)=>a+2**(x%8),0));
 for(let z=0;z<r.events.length;z++){const e=r.events[z],i=e.root;if(z===0){insist(e.kind==="setup","setup");}else{insist(calls<144&&Number.isInteger(i)&&i>=0&&i<batch.length&&!done.has(i),"event binding");const w=batch[i].w;
  if(e.kind==="load"){insist(!live.has(i),"loaded resident");const victim=live.size===r.arm.slots?live.keys().next().value:null;insist(victim===e.victim,"wrong eviction");let saved=null,restored=null;if(victim!==null){let v=live.get(victim);insist(v.cursor===tapes[victim].length,"evicted incomplete restore");if(r.arm.backing==="packed"){saved=bytes(v.known,batch[victim].w.n);back.set(victim,{bits:saved,known:[...v.known]});writes+=saved.length;}live.delete(victim);evictions++;}let s={known:[...w.facts],cursor:0};if(back.has(i)){const b=back.get(i);restored=b.bits;reads+=restored.length;s={known:[...b.known],cursor:tapes[i].length};back.delete(i);}insist(equal(saved,e.saved)&&equal(restored,e.restored),"forged packed checkpoint");live.set(i,s);
  }else if(e.kind==="select"){insist(live.has(i),"selected nonresident");const s=live.get(i);live.delete(i);live.set(i,s);
  }else{insist(["new","replay"].includes(e.kind)&&live.has(i),"step without resident");const s=live.get(i),a=e.action;insist(Number.isInteger(a)&&a>=0&&a<w.rules.length,"action index");const rule=w.rules[a];insist(rule.premises.every(p=>s.known.includes(p))&&!s.known.includes(rule.conclusion)&&rule.conclusion===e.fact,"invalid derivation");
   if(e.kind==="replay"){insist(s.cursor<tapes[i].length&&a===tapes[i][s.cursor],"forged tape");replays++;tapeRead++;}
   else{insist(s.cursor===tapes[i].length,"new before restoration");if(tapes[i].length===0)started++;tapes[i].push(a);tapeWrite++;}
   s.cursor++;s.known.push(rule.conclusion);live.delete(i);live.set(i,s);calls++;if(s.known.includes(w.goal)){insist(e.kind==="new","goal replay");done.add(i);live.delete(i);back.delete(i);if(w.rules.length>=40)long++;else short++;}
  }}
  const l=[...live.keys()].reduce((a,i)=>a+Math.ceil(batch[i].w.n/8),0),b=[...back.values()].reduce((a,x)=>a+x.bits.length,0),t=tapes.reduce((a,x)=>a+x.length+2,0);peakLive=Math.max(peakLive,l);peakBack=Math.max(peakBack,b);peakPacked=Math.max(peakPacked,l+b);peakTape=Math.max(peakTape,t);peakPayload=Math.max(peakPayload,l+b+t);
  const m=e.cost;insist(Object.keys(meter()).every(k=>Number.isSafeInteger(m[k])&&m[k]>=0&&(!prev||m[k]>=prev[k])),"nonmonotone counter");insist(m.verifier===calls&&m.replay===replays&&m.checkpointRead===reads&&m.checkpointWrite===writes&&m.tapeRead===tapeRead&&m.tapeWrite===tapeWrite&&e.solved===done.size,"counter or completion mismatch");
  for(let j=0;j<3;j++){if(calls<=budgets[j])callsCurve[j]=done.size;for(const w of [10,100,1000])if(units(m,w)<=w*budgets[j])weighted[w][j]=done.size;}prev=m;
 }
 insist(calls===144||done.size===8,"premature termination");insist(equal(prev,r.counts)&&calls===r.calls&&replays===r.replays&&done.size===r.completed&&started===r.started&&long===r.longSolved&&short===r.shortSolved&&evictions===r.evictions&&peakPacked===r.peakPacked&&peakLive===r.peakLive&&peakBack===r.peakBack&&peakTape===r.peakTape&&peakPayload===r.peakPayload&&equal(callsCurve,r.callCurve)&&equal(weighted,r.weighted),"terminal receipt");
 return calls;
}

function suite(){
 const tasks=makeTasks(),model=fit(tasks.filter(t=>t.split==="train")),arms=[
  ["rr1_tape", "rr",1,1,"tape","local"],["rr4_tape","rr",1,4,"tape","local"],["rr8_tape","rr",1,8,"tape","local"],["rr1_resident8","rr",8,1,"tape","local"],["static_block","static",1,"block","tape","local"],["rr1_packed","rr",1,1,"packed","local"],["rr4_packed","rr",1,4,"packed","local"],["adaptive_tape","adaptive",1,"adaptive","tape","local"],["adaptive_packed","adaptive",1,"adaptive","packed","local"],["full_static_block","static",1,"block","tape","full"]
 ].map(([name,policy,slots,quantum,backing,router])=>({name,policy,slots,quantum,backing,router}));
 const summary={},runs=[];let audited=0,negativeFact=false,negativeCheckpoint=false,negativeCost=false;
 for(const split of ["calibration","ordinary","deeper"]){let cohort=tasks.filter(t=>t.split===split),s=summary[split]={};
  for(let i=0;i<64;i+=8){let batch=cohort.slice(i,i+8);for(const arm of arms){const r=run(batch,arm,model.weights);audited+=audit(batch,r);runs.push({split,batch:i/8,...r});let x=s[arm.name]??={completed:[0,0,0],weighted:{10:[0,0,0],100:[0,0,0],1000:[0,0,0]},calls:0,replays:0,started:0,longSolved:0,shortSolved:0,longRoots:0,evictions:0,peakPacked:0,peakLive:0,peakBack:0,peakTape:0,peakPayload:0,counts:meter()};
   for(let j=0;j<3;j++){x.completed[j]+=r.callCurve[j];for(const w of [10,100,1000])x.weighted[w][j]+=r.weighted[w][j];}
   for(const k of ["calls","replays","started","longSolved","shortSolved","evictions"])x[k]+=r[k];x.longRoots+=batch.filter(t=>t.w.rules.length>=40).length;for(const k of ["peakPacked","peakLive","peakBack","peakTape","peakPayload"])x[k]=Math.max(x[k],r[k]);for(const k of Object.keys(x.counts))x.counts[k]+=r.counts[k];
   if(!negativeFact){let z=JSON.parse(JSON.stringify(r));const e=z.events.find(e=>e.kind==="new");if(e){e.fact=-1;try{audit(batch,z);}catch{negativeFact=true;}}}
   if(!negativeCheckpoint){let z=JSON.parse(JSON.stringify(r));const e=z.events.find(e=>e.restored);if(e){e.restored[0]^=1;try{audit(batch,z);}catch{negativeCheckpoint=true;}}}
   if(!negativeCost){let z=JSON.parse(JSON.stringify(r));const e=z.events.find(e=>e.kind==="new");if(e){e.cost.verifier--;try{audit(batch,z);}catch{negativeCost=true;}}}
  }}
  // Packed replay uses the same round-robin choices as all-resident, so
  // equality of fresh-call completion curves is expected for the q=1 pair.
  insist(equal(s.rr1_packed.completed,s.rr1_resident8.completed)&&s.rr1_packed.replays===0,"packed vs resident invariant");
  insist(s.static_block.replays===0&&s.adaptive_packed.replays===0,"unexpected restoration");
 }
 insist(negativeFact&&negativeCheckpoint&&negativeCost,"negative sentinel");
 return {summary:{schema:"cog13b.checkpoint_quantum.v1",seed:2026112603,counts:{train:64,calibration:64,ordinary:64,deeper:64},evidence:"fresh public synthetic generator; development only",budgets:[48,96,144],costWeights:[10,100,1000],model,audit:{runs:runs.length,verifierCalls:audited,negativeFact,negativeCheckpoint,negativeCost},splits:summary},runs};
}
const [, , output]=process.argv;
insist(output,"usage: node benchmark.mjs NEW_OUTPUT.json");
const before=process.memoryUsage();
const t0=process.hrtime.bigint();
const result=suite();
const runtime={elapsedMs:Number(process.hrtime.bigint()-t0)/1e6,rssBefore:before.rss,rssAfter:process.memoryUsage().rss,maxRSSKiB:process.resourceUsage().maxRSS,node:process.version,platform:process.platform};
writeFileSync(output,JSON.stringify({...result,runtime},null,2)+"\n",{flag:"wx"});
console.log(JSON.stringify({summary:result.summary,runtime},null,2));
