// COG-13E: train-only marginal retention-value pilot; pure standard JS.
function insist(x,m){if(!x)throw Error(m);}

function same(a,b){return JSON.stringify(a)===JSON.stringify(b);}

function rng32(seed){let x=seed>>>0;return n=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)%n;};}

function shuffled(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=r(i+1);[a[i],a[j]]=[a[j],a[i]];}return a;}

function makeCorpus(){
 const tasks=[],fingerprints=new Set(),counts={train:64,calibration:16,ordinary:32,deeper:32};
 for(const [s,[split,n]] of Object.entries(counts).entries())for(let i=0;i<n;i++){
  const r=rng32(2026112607+s*1000003+i*1009),family=i%4,nodes=[{id:0,parent:null,depth:0,work:0,total:0,bytes:0}],leaves=[];
  const add=(parent,expensive=false)=>{const p=nodes[parent],cost=1+r(expensive?80:15),bytes=24+(p.depth+1)*8+r(24)+(expensive?r(40):0),id=nodes.length;nodes.push({id,parent,depth:p.depth+1,work:cost,total:p.total+cost,bytes});return id;};
  const common=add(0,family===2),groups=Array.from({length:2},()=>add(common,family===1));
  for(let b=0;b<6;b++){let node=groups[b%2],depth=(split==="deeper"?8:4)+r(split==="deeper"?6:4),expensive=family===2?b%3===0:family===1?b%2===0:false;
   while(nodes[node].depth<depth)node=add(node,expensive);leaves.push(node);}
  const path=id=>{let x=id,a=[];while(x){a.unshift(x);x=nodes[x].parent;}return a;};
  const paths=leaves.map(path),trace=[];let current=r(6),hot=r(6);
  for(let t=0;t<60;t++){
   if(family===0){if(t%5===0)current=r(6);}
   else if(family===1){current=(current+1)%6;}
   else if(family===2){if(t%12===0)hot=r(6);current=r(5)<4?hot:r(6);}
   else {if(r(3)===0)current=(current+1+r(5))%6;}
   const p=paths[current];let d;
   if(t%10===0)d=1+r(Math.min(3,p.length));else if(family===1)d=Math.min(p.length,3+Math.floor(t/6)%Math.max(1,p.length-2));else if(family===0)d=Math.min(p.length,2+t%5);else d=1+r(p.length);
   const target=p[d-1],failure=r(9)===0,closed=!failure&&d===p.length&&r(2)===0;
   trace.push({target,outcome:failure?"reject":closed?"closed":"open",failWork:failure?1+r(12):0});
  }
  const fp=JSON.stringify({nodes,trace});insist(!fingerprints.has(fp),"duplicate world");fingerprints.add(fp);
  tasks.push({id:`${split}:${i}`,split,family,nodes,trace});
 }
 return {tasks,counts};
}

function path(t,id){const a=[];while(id){a.unshift(id);id=t.nodes[id].parent;}return a;}

function ancestor(t,a,b){while(t.nodes[b].depth>t.nodes[a].depth)b=t.nodes[b].parent;return a===b;}

function deepest(t,cache,target,skip=-1){let best=0;for(const id of cache.keys())if(id!==skip&&t.nodes[id].depth>t.nodes[best].depth&&ancestor(t,id,target))best=id;return best;}

function cacheSize(cache){let n=0;for(const e of cache.values())n+=e.bytes;return n;}

function marginal(t,cache,id){
 const e=cache.get(id),a=deepest(t,cache,id,id);return Math.max(0,e.work-(cache.get(a)?.work??0));
}

function features(t,cache,id,time,past){
 const e=cache.get(id),recent=past.slice(-12);let matches=0,recentMatches=0,last=-1,desc=0;
 for(let j=0;j<past.length;j++)if(ancestor(t,id,past[j].target)){matches++;last=j;}
 for(const r of recent)if(ancestor(t,id,r.target))recentMatches++;
 for(const other of cache.keys())if(other!==id&&ancestor(t,id,other))desc++;
 const m=marginal(t,cache,id),age=Math.min(24,time-e.last);
 return [1,e.depth/14,e.bytes/192,Math.log1p(e.work)/7,Math.log1p(m)/7,age/24,Math.log1p(e.hits)/4,Math.log1p(matches)/4,recentMatches/12,Math.min(24,time-last)/24,Math.min(8,desc)/8,Math.log1p(100*m/e.bytes)/7,Math.min(24,time-e.born)/24];
}

function benefit(t,cache,id,time,horizon=12){
 let saved=0;
 for(const q of t.trace.slice(time+1,time+1+horizon)){
  const a=deepest(t,cache,q.target),b=deepest(t,cache,q.target,id);
  saved+=(cache.get(a)?.work??0)-(cache.get(b)?.work??0);
 }
 insist(saved>=0,"negative marginal");
 return saved;
}

function dot(a,b){return a.reduce((s,x,i)=>s+x*b[i],0);}

function score(t,cache,id,policy,time,past,model){
 const e=cache.get(id),m=marginal(t,cache,id);
 if(policy==="lru")return e.last;
 if(policy==="lfu")return e.hits;
 if(policy==="depth_hits")return e.depth*(e.hits+1);
 if(policy==="work_density")return m/e.bytes;
 if(policy==="recent_density")return m*(1+past.slice(-12).filter(r=>ancestor(t,id,r.target)).length)/e.bytes;
 if(policy==="learned"||policy==="shuffled")return Math.max(0,dot(model.weights,features(t,cache,id,time,past)));
 if(policy==="oracle12")return benefit(t,cache,id,time)/e.bytes;
 throw Error("unknown policy "+policy);
}

function run(t,cap,policy,model=null,collect=null,record=true){
 const cache=new Map(),past=[],events=[];
 let executed=0,baseline=0,commands=0,savedCommands=0,hits=0,evictions=0,peak=0,written=0,read=0,rankings=0,featureEvals=0,closed=0,rejected=0,open=0;
 for(let time=0;time<t.trace.length;time++){
  const q=t.trace[time],hit=deepest(t,cache,q.target),node=t.nodes[q.target],hitEntry=cache.get(hit),saved=hitEntry?.work??0;
  // The controller observes each successful suffix command's cost only when
  // that command is executed. A restored prefix already has its measured cost.
  let work=saved,attempts=0;
  const suffix=path(t,q.target).slice(hitEntry?.depth??0);
  for(const a of suffix){work+=t.nodes[a].work;attempts++;}
  if(q.outcome==="reject")attempts++;
  insist(work===node.total,"prefix cost mismatch");
  const charged=work-saved+q.failWork;
  executed+=charged;baseline+=work+q.failWork;commands+=attempts;savedCommands+=hitEntry?.depth??0;
  if(hitEntry){hits++;read+=hitEntry.bytes;hitEntry.last=time;hitEntry.hits++;}
  past.push({target:q.target});
  let admitted=false,removed=[],decisions=[];
  if(q.outcome==="closed")closed++;else if(q.outcome==="reject")rejected++;else open++;
  if(cap>0&&q.outcome==="open"&&!cache.has(q.target)&&node.bytes<=cap){
   // Byte count is a declared synthetic payload. It is known after the
   // successful state exists, never used as an estimate of Lean allocation.
   cache.set(q.target,{id:q.target,depth:node.depth,bytes:node.bytes,work,born:time,last:time,hits:0});
   admitted=true;written+=node.bytes;
   while(cacheSize(cache)>cap){
    const list=[...cache.keys()].sort((a,b)=>a-b),scores=[];
    for(const id of list){
     const e=cache.get(id);rankings++;
     if(policy==="learned"||policy==="shuffled")featureEvals++;
     if(collect){const x=features(t,cache,id,time,past),future=benefit(t,cache,id,time);collect.push({root:t.id,split:t.split,policy,cap,time,entry:id,input:x,target:Math.log1p(100*future/e.bytes),futureWork:future,bytes:e.bytes});}
     scores.push(score(t,cache,id,policy,time,past,model));
    }
    let k=0;for(let j=1;j<list.length;j++)if(scores[j]<scores[k]||(scores[j]===scores[k]&&cache.get(list[j]).last<cache.get(list[k]).last))k=j;
    const victim=list[k];if(record)decisions.push({candidates:list,scores,victim});cache.delete(victim);removed.push(victim);evictions++;
   }
  }
  const bytes=cacheSize(cache);peak=Math.max(peak,bytes);
  insist(bytes<=cap,"cap exceeded");
  if(record)events.push({time,target:q.target,outcome:q.outcome,hit,commands:attempts,work:charged,saved,admitted,removed,decisions,after:[...cache.keys()].sort((a,b)=>a-b),bytes});
 }
 return {root:t.id,split:t.split,cap,policy,executed,baseline,commands,savedCommands,hits,evictions,peak,written,read,rankings,featureEvals,closed,rejected,open,events};
}

function fit(groups,shuffle=false){
 const n=13,A=Array.from({length:n},()=>Array(n+1).fill(0));let rows=0,meanTarget=0;
 for(const g of groups){if(!g.length)continue;const rr=rng32(6419+groups.indexOf(g)*127),ys=shuffle?shuffled(g.map(r=>r.target),rr):g.map(r=>r.target),w=1/g.length;
  for(let j=0;j<g.length;j++){const x=g[j].input,y=ys[j];rows++;meanTarget+=y;for(let a=0;a<n;a++){for(let b=0;b<n;b++)A[a][b]+=w*x[a]*x[b];A[a][n]+=w*x[a]*y;}}
 }
 for(let j=0;j<n;j++)A[j][j]+=.15;
 for(let j=0;j<n;j++){let p=j;for(let k=j+1;k<n;k++)if(Math.abs(A[k][j])>Math.abs(A[p][j]))p=k;[A[p],A[j]]=[A[j],A[p]];const z=A[j][j];insist(Math.abs(z)>1e-12,"singular");for(let b=j;b<=n;b++)A[j][b]/=z;for(let k=0;k<n;k++)if(k!==j){let q=A[k][j];for(let b=j;b<=n;b++)A[k][b]-=q*A[j][b];}}
 return {weights:A.map(x=>x[n]),roots:groups.length,rows,meanTarget:meanTarget/rows,shuffledLabels:shuffle};
}

function audit(t,r){
 const known=[],seen=new Map();let work=0,baseline=0,commands=0,savedCommands=0,hits=0,evictions=0,peak=0,written=0,read=0,rankings=0,featureEvals=0,closed=0,rejected=0,open=0;
 const paths=t.nodes.map((_,i)=>path(t,i)),isPrefix=(a,b)=>a.length<=b.length&&a.every((x,i)=>b[i]===x);
 const cumulative=id=>paths[id].reduce((a,x)=>a+t.nodes[x].work,0);
 const size=()=>known.reduce((a,id)=>a+t.nodes[id].bytes,0);
 insist(r.root===t.id&&r.split===t.split&&r.events.length===t.trace.length&&r.cap>=0,"root or trace mismatch");
 for(let i=0;i<t.trace.length;i++){
  const q=t.trace[i],e=r.events[i],p=paths[q.target];
  insist(e.time===i&&e.target===q.target&&e.outcome===q.outcome,"request changed");
  const best=known.filter(a=>isPrefix(paths[a],p)).sort((a,b)=>paths[b].length-paths[a].length)[0]??0;
  const skipped=cumulative(best),raw=cumulative(q.target)+q.failWork,cost=raw-skipped;
  const count=p.length-paths[best].length+Number(q.outcome==="reject");
  insist(e.hit===best&&e.work===cost&&e.saved===skipped&&e.commands===count,"forged execution cost or prefix");
  work+=cost;baseline+=raw;commands+=count;savedCommands+=paths[best].length;
  if(best){hits++;read+=t.nodes[best].bytes;seen.get(best).last=i;seen.get(best).hits++;}
  if(q.outcome==="closed")closed++;else if(q.outcome==="reject")rejected++;else open++;
  const admit=r.cap>0&&q.outcome==="open"&&!known.includes(q.target)&&t.nodes[q.target].bytes<=r.cap;
  insist(e.admitted===admit,"invalid admission");
  if(admit){known.push(q.target);seen.set(q.target,{last:i,hits:0});written+=t.nodes[q.target].bytes;}
  insist(e.removed.length===e.decisions.length,"eviction record mismatch");
  for(let j=0;j<e.removed.length;j++){
   const v=e.removed[j],d=e.decisions[j];
   insist(size()>r.cap&&known.includes(v)&&v===d.victim&&same([...known].sort((a,b)=>a-b),d.candidates)&&d.scores.length===known.length&&d.scores.every(Number.isFinite),"invalid eviction");
   let k=0;for(let z=1;z<d.candidates.length;z++){if(d.scores[z]<d.scores[k]||(d.scores[z]===d.scores[k]&&seen.get(d.candidates[z]).last<seen.get(d.candidates[k]).last))k=z;}
   insist(d.candidates[k]===v,"eviction not score minimum");
   rankings+=known.length;if(["learned","shuffled"].includes(r.policy))featureEvals+=known.length;
   known.splice(known.indexOf(v),1);seen.delete(v);evictions++;
  }
  const bytes=size();peak=Math.max(peak,bytes);insist(bytes<=r.cap&&bytes===e.bytes&&same([...known].sort((a,b)=>a-b),e.after),"cache bytes or state changed");
 }
 const a={executed:work,baseline,commands,savedCommands,hits,evictions,peak,written,read,rankings,featureEvals,closed,rejected,open};
 for(const [k,v] of Object.entries(a))insist(r[k]===v,"terminal counter "+k);
 return r.events.length;
}

function summarize(runs){
 const out={};
 for(const r of runs){const s=out[r.split]??={},cap=s[r.cap]??={},a=cap[r.policy]??={roots:0,work:0,rootReplayWork:0,commands:0,savedCommands:0,hits:0,evictions:0,peak:0,written:0,read:0,rankings:0,featureEvals:0,closed:0,rejected:0,open:0};
  a.roots++;a.work+=r.executed;a.rootReplayWork+=r.baseline;
  for(const k of ["commands","savedCommands","hits","evictions","written","read","rankings","featureEvals","closed","rejected","open"])a[k]+=r[k];a.peak=Math.max(a.peak,r.peak);
 }
 return out;
}

function suite(keepRuns=true){
 const {tasks,counts}=makeCorpus(),caps=[128,256,512],policies=["lru","lfu","depth_hits","work_density","recent_density","learned","shuffled","oracle12"],groups=[];
 let trainingWork=0,trainingRequests=0;const trainReceipts=[];
 for(const t of tasks.filter(t=>t.split==="train")){
  const rows=[];
  for(const cap of caps)for(const p of ["lru","work_density"]){const r=run(t,cap,p,null,rows,false);trainingWork+=r.executed;trainingRequests+=t.trace.length;trainReceipts.push({root:t.id,cap,policy:p,executed:r.executed,requests:t.trace.length});}
  insist(rows.every(x=>x.split==="train"),"training split leak");groups.push(rows);
 }
 const model=fit(groups),sham=fit(groups,true),runs=[],negatives={cost:false,cache:false,admission:false,missing:false};let audited=0;
 for(const t of tasks.filter(t=>t.split!=="train")){
  const base=run(t,0,"lru");audited+=audit(t,base);runs.push(base);
  for(const cap of caps)for(const p of policies){const r=run(t,cap,p,p==="shuffled"?sham:model);audited+=audit(t,r);insist(r.baseline===base.executed&&r.executed<=base.executed,"root replay mismatch");runs.push(r);
   if(t.split==="calibration"&&r.policy==="learned"&&cap===256&&!negatives.cost){
    const corrupt=(k,f)=>{const v=JSON.parse(JSON.stringify(r));f(v);try{audit(t,v);}catch{negatives[k]=true;}};
    corrupt("cost",v=>v.events[0].work++);
    corrupt("cache",v=>v.events[0].bytes++);
    corrupt("admission",v=>v.events[0].admitted=!v.events[0].admitted);
    corrupt("missing",v=>v.events.pop());
   }
  }
 }
 insist(Object.values(negatives).every(Boolean),"negative mutation accepted");
 // Target-label sanity: a cached ancestor whose deeper child covers every
 // matching request has no marginal future value over that static set.
 const t={nodes:[{parent:null,depth:0,total:0},{parent:0,depth:1,total:10},{parent:1,depth:2,total:30}],trace:[{target:2},{target:2}]};
 const c=new Map([[1,{work:10}],[2,{work:30}]]);
 insist(benefit(t,c,1,0)===0&&benefit(t,c,2,0)===20,"overlap marginal invariant");
 const summary={schema:"cog13e.retention.v1",evidence:"generated trace simulator; no native CPU/RAM evidence",seed:2026112607,counts,caps,horizon:12,requestsPerRoot:60,model,shuffledControl:sham,trainingWork,trainingRequests,trainingRows:model.rows,trainingRollouts:trainReceipts.length,audit:{runs:runs.length,requests:audited,negativeMutations:negatives,overlapMarginalInvariant:true},splits:summarize(runs)};
 return {summary,runs:keepRuns?runs:[],trainingRows:keepRuns?groups.flat():[],trainReceipts:keepRuns?trainReceipts:[],tasks:keepRuns?tasks:[]};
}
export {makeCorpus,run,audit,features,benefit,suite};
