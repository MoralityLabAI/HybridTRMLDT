// COG-13F: shared generator/path helpers adapted from COG-13E commit 8c5817d0079e3998f1b9704b662f37b58cc0739c.
function insist(x,m){if(!x)throw Error(m);}

function same(a,b){return JSON.stringify(a)===JSON.stringify(b);}

function rng32(seed){let x=seed>>>0;return n=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)%n;};}

function shuffled(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=r(i+1);[a[i],a[j]]=[a[j],a[i]];}return a;}

function makeCorpus(){
 const tasks=[],fingerprints=new Set(),counts={train:64,calibration:24,ordinary:32,deeper:32};
 for(const [s,[split,n]] of Object.entries(counts).entries())for(let i=0;i<n;i++){
  const r=rng32(2026112619+s*1000003+i*1009),family=i%4,nodes=[{id:0,parent:null,depth:0,work:0,total:0,bytes:0}],leaves=[];
  const add=(parent,expensive=false)=>{const p=nodes[parent],cost=1+r(expensive?80:15),bytes=24+(p.depth+1)*8+r(24)+(expensive?r(40):0),id=nodes.length;nodes.push({id,parent,depth:p.depth+1,work:cost,total:p.total+cost,bytes});return id;};
  const common=add(0,family===2),groups=Array.from({length:2},()=>add(common,family===1));
  for(let b=0;b<6;b++){let node=groups[b%2],depth=(split==="deeper"?11:4)+r(split==="deeper"?6:7),expensive=family===2?b%3===0:family===1?b%2===0:false;
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
  tasks.push({id:`cog13f:${split}:${i}`,split,family,nodes,trace});
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
 return [1,e.depth/16,e.bytes/192,Math.log1p(e.work)/7,Math.log1p(m)/7,age/24,Math.log1p(e.hits)/4,Math.log1p(matches)/4,recentMatches/12,Math.min(24,time-last)/24,Math.min(8,desc)/8,Math.log1p(100*m/e.bytes)/7,Math.min(24,time-e.born)/24];
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


function clone(c){return new Map([...c].map(([k,v])=>[k,{...v}]));}

function meter(){return {work:0,baseline:0,commands:0,savedCommands:0,hits:0,read:0,written:0,rankings:0,featureEvals:0,coverageEvals:0,evictions:0,peak:0,requests:0,closed:0,rejected:0,open:0,lookaheadRollouts:0,lookaheadRequests:0,lookaheadWork:0};}

function recent(t,c,i,past){return marginal(t,c,i)*(1+past.slice(-12).filter(q=>ancestor(t,i,q.target)).length)/c.get(i).bytes;}

function coverage(t,c,i,past){
 let val=0;
 for(const q of past.slice(-12)){
  const a=deepest(t,c,q.target),b=deepest(t,c,q.target,i);
  val+=(c.get(a)?.work??0)-(c.get(b)?.work??0);
 }
 return (marginal(t,c,i)+val)/c.get(i).bytes;
}

function input(t,c,id,time,past,cap){
 const e=c.get(id);
 return [...features(t,c,id,time,past),Math.log1p(100*coverage(t,c,id,past))/7,e.bytes/cap,cap/512,(cacheSize(c)-cap)/cap];
}

function prior(t,c,id,past){return Math.log1p(100*recent(t,c,id,past));}

function argmin(c,ids,scores){
 let k=0;
 for(let j=1;j<ids.length;j++){if(scores[j]<scores[k]||(scores[j]===scores[k]&&c.get(ids[j]).last<c.get(ids[k]).last))k=j;}
 return ids[k];
}

function scoreOne(t,c,id,time,past,cap,policy,model,lambda,m){
 m.rankings++;
 if(policy==="lru")return c.get(id).last;
 if(policy==="work")return marginal(t,c,id)/c.get(id).bytes;
 if(policy==="recent"||(policy==="residual"&&lambda===0)||(policy==="sham"&&lambda===0))return prior(t,c,id,past);
 if(policy==="coverage"){m.coverageEvals++;return coverage(t,c,id,past);}
 if(policy==="residual"||policy==="sham"){
  m.featureEvals++;m.coverageEvals++;
  return prior(t,c,id,past)+lambda*dot(model.weights,input(t,c,id,time,past,cap));
 }
 throw Error("unknown policy "+policy);
}

function finishEviction(t,c,time,past,cap,m){
 while(cacheSize(c)>cap){const ids=[...c.keys()].sort((a,b)=>a-b),scores=ids.map(i=>scoreOne(t,c,i,time,past,cap,"recent",null,0,m));c.delete(argmin(c,ids,scores));m.evictions++;}
}

function counterfactual(t,c,id,time,past,cap,m){
 const cc=clone(c),pp=[...past],local=meter();cc.delete(id);finishEviction(t,cc,time,pp,cap,local);
 const end=Math.min(t.trace.length,time+13);
 for(let j=time+1;j<end;j++)advance(t,cc,j,pp,cap,"recent",null,0,local,null,false);
 m.lookaheadRollouts++;m.lookaheadRequests+=local.requests;m.lookaheadWork+=local.work;
 return local.work;
}

function advance(t,c,time,past,cap,policy,model,lambda,m,collect=null,record=true){
 const q=t.trace[time],node=t.nodes[q.target],hit=deepest(t,c,q.target),old=c.get(hit),saved=old?.work??0;
 let cost=saved,attempts=0;
 for(const id of path(t,q.target).slice(old?.depth??0)){cost+=t.nodes[id].work;attempts++;}
 insist(cost===node.total,"path cost");
 attempts+=Number(q.outcome==="reject");
 const work=cost-saved+q.failWork;
 m.requests++;m.work+=work;m.baseline+=cost+q.failWork;m.commands+=attempts;m.savedCommands+=old?.depth??0;
 if(old){m.hits++;m.read+=old.bytes;old.last=time;old.hits++;}
 past.push({target:q.target});m[q.outcome==="reject"?"rejected":q.outcome]++;
 let admitted=false,decisions=[];
 if(cap>0&&q.outcome==="open"&&!c.has(q.target)&&node.bytes<=cap){
  c.set(q.target,{id:q.target,depth:node.depth,bytes:node.bytes,work:cost,born:time,last:time,hits:0});m.written+=node.bytes;admitted=true;
  let iteration=0;
  while(cacheSize(c)>cap){
   const ids=[...c.keys()].sort((a,b)=>a-b);
   let costs=null;
   if(collect||policy==="oracle")costs=ids.map(id=>counterfactual(t,c,id,time,past,cap,m));
   if(collect){
    insist(t.split==="train","nontraining label");
    const min=Math.min(...costs),rows=ids.map((id,j)=>({entry:id,input:input(t,c,id,time,past,cap),prior:prior(t,c,id,past),cost:costs[j],regret:costs[j]-min,target:Math.log1p((costs[j]-min)/10)}));
    collect.push({root:t.id,split:t.split,cap,behavior:policy,time,iteration,continuation:"recent",horizon:Math.min(12,t.trace.length-time-1),rows});
   }
   let scores;
   if(policy==="oracle"){scores=costs;m.rankings+=ids.length;}
   else scores=ids.map(id=>scoreOne(t,c,id,time,past,cap,policy,model,lambda,m));
   const victim=argmin(c,ids,scores);c.delete(victim);m.evictions++;iteration++;
   if(record)decisions.push({ids,scores,victim,oracleCosts:policy==="oracle"?costs:null});
  }
 }
 const bytes=cacheSize(c);m.peak=Math.max(m.peak,bytes);insist(bytes<=cap,"capacity");
 return {time,target:q.target,outcome:q.outcome,hit,work,commands:attempts,saved,admitted,decisions,after:[...c.keys()].sort((a,b)=>a-b),bytes};
}

function run(t,cap,policy,model=null,lambda=0,collect=null,record=true){
 const cache=new Map(),past=[],m=meter(),events=[];
 for(let j=0;j<t.trace.length;j++){const e=advance(t,cache,j,past,cap,policy,model,lambda,m,collect,record);if(record)events.push(e);}
 return {root:t.id,split:t.split,cap,policy,lambda,m,events};
}

function fit(groups,sham=false){
 const n=17,A=Array.from({length:n},()=>Array(n+1).fill(0)),centroid=x=>x.reduce((s,v)=>s+v,0)/x.length;let rows=0,decisions=0;
 for(let i=0;i<groups.length;i++){
  const g=groups[i],w=1/g.reduce((s,d)=>s+d.rows.length,0),r=rng32(271828+i*157);
  for(const d of g){
   decisions++;const ys=sham?shuffled(d.rows.map(x=>x.target),r):d.rows.map(x=>x.target);
   const residuals=d.rows.map((x,j)=>ys[j]-x.prior),mean=centroid(residuals);
   const means=Array.from({length:n},(_,k)=>centroid(d.rows.map(x=>x.input[k])));
   for(let j=0;j<d.rows.length;j++){rows++;const x=d.rows[j].input.map((v,k)=>v-means[k]),y=residuals[j]-mean;
    for(let a=0;a<n;a++){for(let b=0;b<n;b++)A[a][b]+=w*x[a]*x[b];A[a][n]+=w*x[a]*y;}
   }
  }
 }
 for(let j=0;j<n;j++)A[j][j]+=.2;
 for(let j=0;j<n;j++){let p=j;for(let k=j+1;k<n;k++)if(Math.abs(A[k][j])>Math.abs(A[p][j]))p=k;[A[p],A[j]]=[A[j],A[p]];let z=A[j][j];insist(Math.abs(z)>1e-12,"singular");for(let b=j;b<=n;b++)A[j][b]/=z;for(let k=0;k<n;k++)if(k!==j){let q=A[k][j];for(let b=j;b<=n;b++)A[k][b]-=q*A[j][b];}}
 return {weights:A.map(r=>r[n]),rows,decisions,roots:groups.length,sham};
}

function audit(t,r,model){
 const c=new Map(),past=[],m=meter(),paths=t.nodes.map((_,i)=>path(t,i)),prefix=(a,b)=>a.length<=b.length&&a.every((x,j)=>x===b[j]);
 const sum=id=>paths[id].reduce((a,i)=>a+t.nodes[i].work,0);
 insist(r.root===t.id&&r.split===t.split&&r.events.length===t.trace.length,"registry");
 for(let time=0;time<t.trace.length;time++){
  const q=t.trace[time],e=r.events[time],n=t.nodes[q.target],p=paths[q.target],ids=[...c.keys()];
  const hit=ids.filter(i=>prefix(paths[i],p)).sort((a,b)=>paths[b].length-paths[a].length)[0]??0,old=c.get(hit),saved=sum(hit),work=sum(q.target)-saved+q.failWork,commands=p.length-paths[hit].length+Number(q.outcome==="reject");
  insist(e.time===time&&e.target===q.target&&e.outcome===q.outcome&&e.hit===hit&&e.work===work&&e.commands===commands&&e.saved===saved,"execution receipt");
  m.requests++;m.work+=work;m.baseline+=sum(q.target)+q.failWork;m.commands+=commands;m.savedCommands+=paths[hit].length;
  if(old){old.hits++;old.last=time;m.hits++;m.read+=old.bytes;}
  past.push({target:q.target});m[q.outcome==="reject"?"rejected":q.outcome]++;
  const admit=r.cap>0&&q.outcome==="open"&&!c.has(q.target)&&n.bytes<=r.cap;insist(admit===e.admitted,"admission");
  if(admit){c.set(q.target,{id:q.target,depth:n.depth,bytes:n.bytes,work:sum(q.target),born:time,last:time,hits:0});m.written+=n.bytes;}
  for(const d of e.decisions){
   insist(cacheSize(c)>r.cap&&same(d.ids,[...c.keys()].sort((a,b)=>a-b)),"eviction candidates");
   let scores;
   if(r.policy==="oracle"){scores=d.ids.map(i=>counterfactual(t,c,i,time,past,r.cap,m));m.rankings+=d.ids.length;insist(same(scores,d.oracleCosts),"oracle replay");}
   else scores=d.ids.map(i=>scoreOne(t,c,i,time,past,r.cap,r.policy,model,r.lambda,m));
   insist(same(scores,d.scores)&&d.victim===argmin(c,d.ids,scores),"score or victim forged");
   c.delete(d.victim);m.evictions++;
  }
  const bytes=[...c.values()].reduce((s,x)=>s+x.bytes,0);m.peak=Math.max(m.peak,bytes);
  insist(bytes===e.bytes&&bytes<=r.cap&&same([...c.keys()].sort((a,b)=>a-b),e.after),"cache mismatch");
 }
 insist(same(m,r.m),"terminal receipt");
 return m.requests;
}

function charged(m){return m.work+m.rankings+2*m.featureEvals+m.coverageEvals+.01*(m.read+m.written);}

function addMetrics(a,m){for(const [k,v] of Object.entries(m)){a[k]=k==="peak"?Math.max(a[k]??0,v):(a[k]??0)+v;}return a;}

function summaryOf(runs){
 const out={};
 for(const r of runs){const s=out[r.split]??={},c=s[r.cap]??={},a=c[r.name]??={roots:0,m:meter()};a.roots++;addMetrics(a.m,r.m);}
 for(const s of Object.values(out))for(const c of Object.values(s))for(const a of Object.values(c))a.charged=charged(a.m);
 return out;
}

function pairs(runs,tasks){
 const out=[],map=new Map(tasks.map(t=>[t.id,t.family]));
 for(const split of ["ordinary","deeper"])for(const cap of [128,256,512])for(const [a,b] of [["calibrated","recent"],["calibrated","coverage"],["residual1","sham1"],["residual1","recent"],["coverage","recent"]]){
  const x=runs.filter(r=>r.split===split&&r.cap===cap&&r.name===a),y=new Map(runs.filter(r=>r.split===split&&r.cap===cap&&r.name===b).map(r=>[r.root,r]));
  const rows=x.map(r=>({root:r.root,family:map.get(r.root),x:charged(r.m),y:charged(y.get(r.root).m),rawX:r.m.work,rawY:y.get(r.root).m.work}));
  const pct=(rows,key1,key2)=>100*rows.reduce((s,r)=>s+r[key2]-r[key1],0)/rows.reduce((s,r)=>s+r[key2],0);
  const strata=[0,1,2,3].map(f=>rows.filter(r=>r.family===f)),rand=rng32(47111+cap+split.length*1009+a.length*109+b.length*17),boot=[];
  for(let k=0;k<4000;k++){const sample=[];for(const group of strata)for(let j=0;j<group.length;j++)sample.push(group[rand(group.length)]);boot.push(pct(sample,"x","y"));}
  boot.sort((a,b)=>a-b);
  out.push({split,cap,arm:a,control:b,roots:rows.length,rawSavedPct:pct(rows,"rawX","rawY"),chargedSavedPct:pct(rows,"x","y"),descriptive95:[boot[100],boot[3899]],wins:rows.filter(r=>r.x<r.y).length,ties:rows.filter(r=>r.x===r.y).length,losses:rows.filter(r=>r.x>r.y).length});
 }
 return out;
}

function suite(keep=true){
 const {tasks,counts}=makeCorpus(),caps=[128,256,512],groups=[],trainReceipts=[],training=meter();
 for(const t of tasks.filter(t=>t.split==="train")){
  const rows=[];
  for(const cap of caps)for(const p of ["recent","coverage"]){const r=run(t,cap,p,null,0,rows,false);addMetrics(training,r.m);trainReceipts.push({root:t.id,cap,policy:p,m:r.m});}
  insist(rows.every(d=>d.split==="train"&&d.continuation==="recent"),"training custody");
  groups.push(rows);
 }
 const model=fit(groups),sham=fit(groups,true),calibration=[],calibrationRuns=[],neg={cost:false,score:false,cache:false,missing:false};let audited=0,auditedRuns=0;
 for(const lambda of [0,.25,.5,1]){
  const m=meter();
  for(const t of tasks.filter(t=>t.split==="calibration"))for(const cap of caps){const r=run(t,cap,"residual",model,lambda);audited+=audit(t,r,model);auditedRuns++;addMetrics(m,r.m);if(keep)calibrationRuns.push(r);}
  calibration.push({lambda,m,charged:charged(m)});
 }
 const selected=[...calibration].sort((a,b)=>a.charged-b.charged||a.lambda-b.lambda)[0].lambda;
 const arms=[["lru","lru",0],["work","work",0],["recent","recent",0],["coverage","coverage",0],["residual1","residual",1],["calibrated","residual",selected],["sham1","sham",1],["oracle","oracle",0]];
 const runs=[];
 for(const t of tasks.filter(t=>["ordinary","deeper"].includes(t.split))){
  for(const cap of caps)for(const [name,policy,lambda] of arms){
   const md=policy==="sham"?sham:model,r=run(t,cap,policy,md,lambda);r.name=name;
   audited+=audit(t,r,md);auditedRuns++;runs.push(r);
   if(t.split==="ordinary"&&cap===256&&name==="residual1"&&!neg.cost){
    const mutate=(k,f)=>{const v=JSON.parse(JSON.stringify(r));f(v);try{audit(t,v,md);}catch{neg[k]=true;}};
    mutate("cost",v=>v.events[0].work++);
    mutate("score",v=>v.events.find(e=>e.decisions.length).decisions[0].scores[0]+=1);
    mutate("cache",v=>v.events[0].bytes++);
    mutate("missing",v=>v.events.pop());
   }
  }
 }
 insist(Object.values(neg).every(Boolean),"negative sentinel failed");
 const summary=summaryOf(runs),paired=pairs(runs,tasks),deeper=summary.deeper;
 const gate={beatsBoth: caps.every(c=>deeper[c].calibrated.charged<Math.min(deeper[c].recent.charged,deeper[c].coverage.charged)),beatsSham: caps.every(c=>deeper[c].residual1.charged<deeper[c].sham1.charged)};
 const result={schema:"cog13f.dynamic.v1",seed:2026112619,counts,caps,horizon:12,model,sham,training,trainingRollouts:trainReceipts.length,calibration,selected,summary,paired,gate,audit:{runs:auditedRuns,requests:audited,negative:neg},evidence:"synthetic development, modeled work/bytes only"};
 return {result,runs:keep?runs:[],calibrationRuns,labels:keep?groups.flat():[],tasks:keep?tasks:[],trainReceipts:keep?trainReceipts:[]};
}
export {makeCorpus,suite};
