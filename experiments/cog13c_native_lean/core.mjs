// COG-13C: shared portable scheduling, recurrent scorer, factorized transport.
const MENU=["assumption","rfl","constructor","intro hnew","simp","simp_all","omega","linarith","ring","norm_num","positivity","aesop","exact h","all_goals assumption","all_goals omega","all_goals ring"];
function requireIt(x,m){if(!x)throw Error(m);}

function makeCases(){
 const cases=[],names=["nat_order","rat_ring","finset_union","mixed_logic"];
 for(let s=0;s<3;s++)for(let i=0;i<8;i++){
  const k=3+s*19+i*2,f=i%4,depth=s===2?4+(i%2):1+(i%2),goals=[];
  let binders;
  if(f===0){binders="(a b : ℕ) (h : a ≤ b)";for(let j=0;j<depth;j++)goals.push(`a + ${k+j} ≤ b + ${k+j+1}`);}
  if(f===1){binders="(x : ℚ)";for(let j=0;j<depth;j++){let c=k+j;goals.push(`(x + ${c}) * (x + ${c}) = x * x + ${2*c} * x + ${c*c}`);}}
  if(f===2){binders="(x : ℕ) (s t : Finset ℕ)";for(let j=0;j<depth;j++)goals.push(j%2===0?`(x ∈ s ∪ t ↔ x ∈ s ∨ x ∈ t) ∧ (${k+j} : ℕ) ≤ ${k+j+1}`:`(x ∈ s ∩ t ↔ x ∈ s ∧ x ∈ t) ∧ (${k+j} : ℕ) < ${k+j+2}`);}
  if(f===3){binders="(a b : ℕ) (h : a ≤ b) (P : Prop) (hp : P)";for(let j=0;j<depth;j++)goals.push(j%2===0?`P ∧ a + ${k+j} ≤ b + ${k+j+1}`:`P → a + ${k+j} ≤ b + ${k+j+2}`);}
  const target=goals.map(x=>`(${x})`).join(" ∧ ");
  cases.push({id:`cog13c_${s}_${i}`,split:["train","ordinary","deeper"][s],family:names[f],depth,statement:`${binders} : ${target}`});
 }
 requireIt(new Set(cases.map(c=>c.statement)).size===24,"duplicate statements");return cases;
}

function rng(seed){let x=seed>>>0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};}

function hash(s){let h=2166136261;for(let j=0;j<s.length;j++)h=Math.imul(h^s.charCodeAt(j),16777619);return h>>>0;}

function vector(goals,action){
 const x=Array(32).fill(0),i=MENU.indexOf(action);requireIt(i>=0,"unregistered action");x[i]=1;
 const tokens=goals.join(" ").match(/[\p{L}\p{N}_]+|[^\s]/gu)||[];
 for(const t of tokens){let h=hash(t);x[16+h%13]+=(h&256)?1:-1;}
 for(let j=16;j<29;j++)x[j]/=Math.sqrt(Math.max(1,tokens.length));
 x[29]=Math.min(1,goals.length/8);x[30]=Math.min(1,tokens.length/128);x[31]=1;return x.map(Math.fround);
}

function dot(a,b){let x=0;for(let i=0;i<a.length;i++)x+=a[i]*b[i];return x;}

function zero(n){return Array(n).fill(0);}

function forward(model,x,input=null){
 const h=[zero(model.b.length)],pre=input??model.W.map(w=>dot(w,x));
 for(let t=0;t<3;t++)h.push(model.R.map((r,i)=>Math.tanh(pre[i]+dot(r,h[t])+model.b[i])));
 return {h,score:dot(model.v,h[3])+model.o};
}

function train(rows){
 requireIt(rows.length>=16&&rows.every(r=>r.split==="train"&&MENU.includes(r.action)),"bad training custody");
 const pos=rows.filter(r=>r.target===1).length;requireIt(pos>0&&pos<rows.length,"missing positive/negative supervision");
 const rand=rng(1701),H=12,init=(n)=>Array.from({length:n},()=>0.12*(2*rand()-1));
 const model={W:Array.from({length:H},()=>init(32)),R:Array.from({length:H},()=>init(H)),b:zero(H),v:init(H),o:0};
 const losses=[];
 for(let epoch=0;epoch<20;epoch++){
  let loss=0;
  const idx=rows.map((_,i)=>i);for(let i=idx.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
  for(const j of idx){const r=rows[j],x=vector(r.goals,r.action),f=forward(model,x),p=1/(1+Math.exp(-f.score)),d=p-r.target,eta=.015;
   loss-=r.target*Math.log(Math.max(p,1e-12))+(1-r.target)*Math.log(Math.max(1-p,1e-12));
   let delta=model.v.map(v=>v*d),gW=model.W.map(w=>zero(w.length)),gR=model.R.map(w=>zero(w.length)),gb=zero(H);
   for(let t=3;t>=1;t--){
    const z=delta.map((v,i)=>v*(1-f.h[t][i]**2));for(let i=0;i<H;i++){gb[i]+=z[i];for(let k=0;k<32;k++)gW[i][k]+=z[i]*x[k];for(let k=0;k<H;k++)gR[i][k]+=z[i]*f.h[t-1][k];}
    delta=Array.from({length:H},(_,k)=>model.R.reduce((a,r,i)=>a+r[k]*z[i],0));
   }
   for(let i=0;i<H;i++){model.v[i]-=eta*d*f.h[3][i];model.b[i]-=eta*gb[i];for(let k=0;k<32;k++)model.W[i][k]-=eta*gW[i][k];for(let k=0;k<H;k++)model.R[i][k]-=eta*gR[i][k];}model.o-=eta*d;
  }
  losses.push(loss/rows.length);
 }
 requireIt(losses.every(Number.isFinite),"nonfinite training");
 return {model,losses,positives:pos,rows:rows.length,seed:1701,parameters:553};
}

function factorize(model,rows){
 const V=[],U=[],rand=rng(419);
 const orth=v=>{for(const q of V){const d=dot(q,v);v=v.map((x,j)=>x-d*q[j]);}let n=Math.sqrt(dot(v,v));requireIt(n>1e-12,"singular factor");return v.map(x=>x/n);};
 for(let k=0;k<4;k++){
  let v=orth(Array.from({length:32},()=>rand()-.5));
  for(let t=0;t<48;t++){const u=model.W.map(w=>dot(w,v));v=orth(Array.from({length:32},(_,j)=>model.W.reduce((a,w,i)=>a+w[j]*u[i],0)));}
  V.push(v);U.push(model.W.map(w=>dot(w,v)));
 }
 const scale=V.map(v=>Math.max(1e-6,...rows.map(r=>Math.abs(dot(v,vector(r.goals,r.action)))))/7);
 return {V,U,scale};
}

function transportRank(model,factors,goals,actions,mode){
 const xs=actions.map(a=>vector(goals,a)),dims=mode==="dense"?32:4,bits=mode==="rank4q"?4:32;
 const coords=mode==="dense"?xs:xs.map(x=>factors.V.map(v=>dot(v,x)));
 const count=coords.length,payload=Math.ceil(count*dims*bits/8),packet=new Uint8Array(4+payload);
 packet.set([1,count,dims,bits]);const dv=new DataView(packet.buffer);let clipping=0;
 coords.flat().forEach((x,j)=>{if(bits===32)dv.setFloat32(4+4*j,x,true);else{const s=factors.scale[j%dims],z=Math.round(x/s);if(z< -7||z>7)clipping++;let code=Math.max(-7,Math.min(7,z))+7;packet[4+(j>>1)]|=code<<((j%2)*4);}});
 requireIt(packet[0]===1&&packet[1]===count&&packet[2]===dims&&packet[3]===bits,"packet header");
 const scores=[];
 for(let i=0;i<count;i++){const x=Array.from({length:dims},(_,k)=>{let j=i*dims+k;return bits===32?dv.getFloat32(4+4*j,true):(((packet[4+(j>>1)]>>((j%2)*4))&15)-7)*factors.scale[k];});
  const input=mode==="dense"?null:model.b.map((_,j)=>factors.U.reduce((a,u,k)=>a+u[j]*x[k],0));
  scores.push(forward(model,x,input).score);
 }
 requireIt(scores.every(Number.isFinite),"nonfinite score");
 const order=scores.map((_,i)=>i).sort((a,b)=>scores[b]-scores[a]||a-b);
 return {order,scores,bytes:packet.length+1,clipping,packet:Array.from(packet),reply:order[0]};
}

function utf8Length(s){let n=0;for(const c of s){const v=c.codePointAt(0);n+=v<128?1:v<2048?2:v<65536?3:4;}return n;}
function sameGoals(a,b){return JSON.stringify(a)===JSON.stringify(b);}

function progress(before,r){return r?.accepted===true&&Array.isArray(r.goals)&&!sameGoals(before,r.goals);}

function newRoot(c){return {c,stack:[],initial:null,terminal:false,solved:false,reason:null,attempts:0,events:[],certificates:[],bytes:0,clipping:0,examples:[]};}

function rootFeatures(root){
 const node=root.stack.at(-1),goals=node?.goals??[root.c.statement];
 return [1,Math.min(1,goals.length/8),Math.min(1,goals.join(" ").length/512),Math.min(1,(node?.prefix.length??0)/8),Math.min(1,(node?.tried.length??0)/16)];
}

function estimate(root,allocator){return allocator?Math.max(1,Math.min(32,dot(allocator.weights,rootFeatures(root)))):Math.max(1,root.c.statement.length/50);}

function fitAllocator(rollouts){
 const rows=[];let censored=0,solved=0;
 for(const r of rollouts){if(!r.solved){censored++;continue;}solved++;for(let i=0;i<r.examples.length;i++)rows.push({root:r.c.id,split:"train",features:r.examples[i],remaining:r.examples.length-i,weight:1/r.examples.length});}
 requireIt(solved>=2&&rows.length>=2,"insufficient certified train trajectories for allocator");
 const n=5,A=Array.from({length:n},()=>zero(n+1));for(const r of rows)for(let i=0;i<n;i++){for(let j=0;j<n;j++)A[i][j]+=r.weight*r.features[i]*r.features[j];A[i][n]+=r.weight*r.features[i]*r.remaining;}
 for(let i=0;i<n;i++)A[i][i]+=.05;
 for(let i=0;i<n;i++){let p=i;for(let j=i+1;j<n;j++)if(Math.abs(A[j][i])>Math.abs(A[p][i]))p=j;[A[i],A[p]]=[A[p],A[i]];const d=A[i][i];requireIt(Math.abs(d)>1e-12,"singular allocator");for(let j=i;j<=n;j++)A[i][j]/=d;for(let k=0;k<n;k++)if(k!==i){const q=A[k][i];for(let j=i;j<=n;j++)A[k][j]-=q*A[i][j];}}
 return {weights:A.map(a=>a[n]),rows,solved,censored};
}

async function collect(cases,backend){
 const rows=[],receipts=[];
 for(const c of cases){backend.setPhase?.("teacher_"+c.id);requireIt(c.split==="train","nontraining collector");const start=backend.now(),deadline=start+90000;let s=null,prefix=[],calls=0,done=false,reason="teacher_depth",elapsed=0,prefixCommands=0;
  try{
   s=await backend.open(c,deadline);let goals=s.initial;
   for(let t=0;t<5&&backend.now()<deadline;t++){
    const outcomes=[];
    for(const action of MENU){if(backend.now()>=deadline){reason="deadline";break;}let r;calls++;prefixCommands+=prefix.length;
     try{r=await s.query([...prefix,action],deadline);}catch(e){r={unknown:true,error:String(e)};}
     if(r.unknown){receipts.push({root:c.id,prefix:[...prefix],action,unknown:true,error:r.error,kind:"unknown"});reason="unknown_probe";break;}
     const target=progress(goals,r)?1:0;
     const row={root:c.id,split:"train",goals:[...goals],prefix:[...prefix],action,target,closed:r.accepted===true&&r.goals.length===0,receipt:r};rows.push(row);outcomes.push(row);
    }
    if(reason==="unknown_probe"||reason==="deadline")break;
    const best=outcomes.find(r=>r.closed)||outcomes.find(r=>r.target===1);if(!best){reason="no_progress";break;}
    prefix.push(best.action);goals=best.receipt.goals;
    if(!goals.length){done=true;reason="probe_closed";break;}
   }
  }catch(e){reason="infrastructure:"+String(e);}finally{if(s)await s.close();elapsed=backend.now()-start;}
  receipts.push({root:c.id,calls,prefixCommands,done,reason,elapsedMs:elapsed});
 }
 return {rows,receipts};
}

async function portfolio(cases,arm,backend,fit,fac,allocator,tag){
 backend.setPhase?.(tag);
 const roots=cases.map(newRoot),live=new Map(),events=[],start=backend.now(),deadline=start+arm.wallMs;
 let calls=0,certCalls=0,prefixCommands=0,opens=0,evictions=0,rr=0,peakRssKiB=0,peakMetadataBytes=0,started=0,reason="budget";
 const sample=()=>{const rss=[...live.values()].reduce((a,s)=>a+(s.rssKiB?.()??0),0);peakRssKiB=Math.max(peakRssKiB,rss);const bytes=utf8Length(JSON.stringify(roots.map(r=>({id:r.c.id,stack:r.stack,initial:r.initial}))));peakMetadataBytes=Math.max(peakMetadataBytes,bytes);};
 const close=async(i,kind="close")=>{const s=live.get(i);if(s){sample();live.delete(i);await s.close();events.push({kind,root:roots[i].c.id,elapsedMs:backend.now()-start});}};
 const finish=(r,why)=>{r.terminal=true;r.reason=why;};
 function chooseRoot(){const available=roots.map((r,i)=>i).filter(i=>!roots[i].terminal);
  if(!available.length)return -1;
  if(arm.scheduler==="rr"){for(let j=0;j<roots.length;j++){let i=(rr+j)%roots.length;if(!roots[i].terminal){rr=(i+1)%roots.length;return i;}}}
  const cost=i=>arm.scheduler==="static"?roots[i].c.statement.length:estimate(roots[i],allocator)+(live.has(i)?0:1);
  return available.sort((a,b)=>cost(a)-cost(b)||a-b)[0];
 }
 try{
  while(calls<arm.probes&&backend.now()<deadline){
   const i=chooseRoot();if(i<0){reason="all_terminal";break;}const r=roots[i];let s=live.get(i);
   if(!s){
    if(live.size>=arm.slots){const victim=live.keys().next().value;await close(victim,"evict");evictions++;}
    try{s=await backend.open(r.c,deadline);opens++;}catch(e){finish(r,"open_unknown:"+String(e));continue;}
    if(r.initial===null){started++;r.initial=[...s.initial];r.stack=[{goals:[...s.initial],prefix:[],tried:[]}];}
    else if(!sameGoals(r.initial,s.initial)){await s.close();finish(r,"reopen_root_changed");continue;}
    live.set(i,s);events.push({kind:"open",root:r.c.id,elapsedMs:backend.now()-start});
   }else{live.delete(i);live.set(i,s);}
   sample();
   const q=arm.scheduler==="rr"?1:arm.scheduler==="static"?arm.probes:Math.max(1,Math.min(4,Math.ceil(estimate(r,allocator))));
   for(let k=0;k<q&&calls<arm.probes&&backend.now()<deadline&&!r.terminal;k++){
    let n=r.stack.at(-1);
    while(n&&n.tried.length>=MENU.length){r.stack.pop();n=r.stack.at(-1);}
    if(!n){finish(r,"search_exhausted");break;}
    const options=MENU.filter(a=>!n.tried.includes(a));let action=options[0],transport=null;
    if(arm.router!=="native"){transport=transportRank(fit.model,fac,n.goals,options,arm.router);action=options[transport.order[0]];r.bytes+=transport.bytes;r.clipping+=transport.clipping;}
    r.examples.push(rootFeatures(r));n.tried.push(action);const prefix=[...n.prefix,action];calls++;r.attempts++;prefixCommands+=n.prefix.length;
    const t0=backend.now();let res;
    try{res=await s.query(prefix,deadline);}catch(e){res={unknown:true,error:String(e)};}
    const event={kind:"probe",root:r.c.id,attempt:r.attempts,portfolioCall:calls,before:[...n.goals],prefix,action,options,reply:res,probeMs:backend.now()-t0,elapsedMs:backend.now()-start,transport};r.events.push(event);events.push(event);sample();
    if(res.unknown){finish(r,"probe_unknown");await close(i);break;}
    if(!progress(n.goals,res))continue;
    if(res.goals.length===0){
     await close(i);
     for(let rep=0;rep<2;rep++){
      if(backend.now()>=deadline)break;certCalls++;
      let cert;try{cert=await backend.certify(r.c,prefix,deadline,rep,tag);}catch(e){cert={accepted:false,error:String(e)};}
      r.certificates.push(cert);events.push({kind:"certificate",root:r.c.id,rep,certificate:cert,elapsedMs:backend.now()-start});if(!cert.accepted)break;
     }
     r.solved=r.certificates.length===2&&r.certificates.every(x=>x.accepted===true)&&r.certificates[0].sourceHash===r.certificates[1].sourceHash&&backend.now()<=deadline;
     finish(r,r.solved?"certified":"certificate_rejected_or_budget");break;
    }
    if(prefix.length<10&&res.goals.length<=8&&!r.stack.some(x=>sameGoals(x.goals,res.goals)))r.stack.push({goals:[...res.goals],prefix,tried:[]});
   }
   if(r.terminal)await close(i);
  }
 }finally{sample();for(const i of [...live.keys()])await close(i);}
 for(const r of roots)if(!r.terminal)finish(r,backend.now()>=deadline?"wall_budget":"probe_budget");
 const elapsedMs=backend.now()-start;
 const result={tag,arm,roots,calls,certCalls,prefixCommands,opens,evictions,started,peakRssKiB,peakMetadataBytes,elapsedMs,reason,events,solved:roots.filter(r=>r.solved).length,onlineBytes:roots.reduce((a,r)=>a+r.bytes,0)};
 auditPortfolio(result);return result;
}

function auditPortfolio(r){
 const reg=new Map(r.roots.map(x=>[x.c.id,x])),live=new Set(),calls=new Map(),certs=new Map();let total=0,certTotal=0,prefix=0,evict=0;
 requireIt(r.roots.length===reg.size&&r.calls<=r.arm.probes,"root registry/budget");
 for(const e of r.events){
  requireIt(reg.has(e.root),"unknown root event");
  if(e.kind==="evict"){requireIt(live.has(e.root),"eviction of nonresident");live.delete(e.root);evict++;}
  else if(e.kind==="close"){requireIt(live.has(e.root),"close of nonresident");live.delete(e.root);}
  else if(e.kind==="open"){requireIt(!live.has(e.root)&&live.size<r.arm.slots,"residency cap");live.add(e.root);}
  else if(e.kind==="probe"){
   requireIt(live.has(e.root),"probe without resident process");total++;prefix+=e.prefix.length-1;requireIt(e.portfolioCall===total&&e.attempt===(calls.get(e.root)||0)+1&&e.prefix.at(-1)===e.action&&e.prefix.every(a=>MENU.includes(a))&&e.options.includes(e.action),"invalid probe receipt");calls.set(e.root,e.attempt);
   if(e.transport){requireIt(e.transport.packet.length+1===e.transport.bytes&&e.options[e.transport.reply]===e.action,"transport reply/byte mismatch");}
  }else if(e.kind==="certificate"){certTotal++;const a=certs.get(e.root)||[];requireIt(e.rep===a.length,"certificate order");a.push(e.certificate);certs.set(e.root,a);live.delete(e.root);}
  else throw Error("unknown receipt kind");
 }
 for(const x of r.roots){const a=certs.get(x.c.id)||[];requireIt(x.attempts===(calls.get(x.c.id)||0)&&sameGoals(x.certificates,a),"root receipt mismatch");if(x.solved)requireIt(a.length===2&&a.every(c=>c.accepted===true)&&a[0].sourceHash===a[1].sourceHash,"uncertified success");}
 requireIt(live.size===0&&total===r.calls&&prefix===r.prefixCommands&&certTotal===r.certCalls&&evict===r.evictions&&r.solved===r.roots.filter(x=>x.solved).length,"summary mismatch");
 return total;
}

async function suite(backend){
 const cases=makeCases(),trainCases=cases.filter(c=>c.split==="train"),teacher=await collect(trainCases,backend);
 const fit=train(teacher.rows),fac=factorize(fit.model,teacher.rows);
 const training=[];
 const arm=(name,scheduler,router,slots)=>({name,scheduler,router,slots,probes:32,wallMs:90000});
 for(let i=0;i<trainCases.length;i+=4)training.push(await portfolio(trainCases.slice(i,i+4),arm("train_dense","static","dense",1),backend,fit,fac,null,`train${i/4}`));
 const allocator=fitAllocator(training.flatMap(r=>r.roots));
 const arms=[arm("native_rr1","rr","native",1),arm("native_rr2","rr","native",2),arm("native_block","static","native",1),arm("dense_block","static","dense",1),arm("dense_adaptive","adaptive","dense",1),arm("rank4_adaptive","adaptive","rank4",1),arm("rank4q_adaptive","adaptive","rank4q",1)];
 const runs=[];
 for(const split of ["ordinary","deeper"]){const cohort=cases.filter(c=>c.split===split);for(let i=0;i<cohort.length;i+=4){let b=i/4,order=[...arms.slice(b),...arms.slice(0,b)];for(const a of order)runs.push(await portfolio(cohort.slice(i,i+4),a,backend,fit,fac,allocator,`${split}_${b}_${a.name}`));}}
 const summary={};for(const r of runs){let s=summary[r.roots[0].c.split]??={},v=s[r.arm.name]??={solved:0,roots:0,calls:0,certCalls:0,prefixCommands:0,opens:0,evictions:0,onlineBytes:0,elapsedMs:0,peakRssKiB:0,peakMetadataBytes:0};for(const k of ["solved","calls","certCalls","prefixCommands","opens","evictions","onlineBytes","elapsedMs"])v[k]+=r[k];v.roots+=r.roots.length;v.peakRssKiB=Math.max(v.peakRssKiB,r.peakRssKiB);v.peakMetadataBytes=Math.max(v.peakMetadataBytes,r.peakMetadataBytes);}
 return {schema:"cog13c.native_portfolio.v1",cases,teacher,fit,factors:fac,training,allocator,runs,summary};
}

function fakeBackend(){
 let t=0,opens=0;return {now:()=>t,open:async c=>{t+=5;opens++;return {initial:["synthetic fixture goal"],rssKiB:()=>100,query:async prefix=>{t+=2;const a=prefix.at(-1);return a==="assumption"?{accepted:true,goals:[]}:{accepted:false,error:"fixture rejection"};},close:async()=>{t++;opens--;requireIt(opens>=0,"fixture session underflow");}};},certify:async(c,prefix,deadline,rep)=>{t+=3;return {accepted:prefix.at(-1)==="assumption",sourceHash:c.id+prefix.join(":"),fixture:true,rep};}};
}

async function selftest(){
 const result=await suite(fakeBackend());
 requireIt(result.runs.length===28&&result.teacher.rows.every(r=>r.split==="train")&&result.allocator.rows.every(r=>r.split==="train"),"fixture registry");
 requireIt(result.runs.every(r=>r.roots.every(x=>x.solved)),"fixture completion");
 let bad=JSON.parse(JSON.stringify(result.runs[0]));bad.roots[0].certificates[0].accepted=false;let rejected=false;try{auditPortfolio(bad);}catch{rejected=true;}requireIt(rejected,"forged certificate accepted");
 const g=["test x ≤ y"],a=MENU.slice(0,3);
 const p=transportRank(result.fit.model,result.factors,g,a,"rank4q");requireIt(p.bytes===4+6+1,"q4 packing");
 requireIt(result.fit.model.W.length===12&&result.fit.model.W[0].length===32,"shape");
 return {fixture:true,cases:24,portfolioRuns:28,teacherRows:result.teacher.rows.length,allocatorRows:result.allocator.rows.length,negativeCertificateRejected:rejected,q4ThreeCandidatesBytes:p.bytes,notScientificEvidence:true};
}
export {MENU,makeCases,transportRank,auditPortfolio,suite,selftest};
