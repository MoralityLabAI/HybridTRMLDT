// COG-13G: deterministic portfolio schedule and independent receipt replay.
const BAD="exact cog13d_missing_identifier";
function check(ok,msg){if(!ok)throw Error(msg);}

function canonical(x){if(Array.isArray(x))return "["+x.map(canonical).join(",")+"]";if(x!==null&&typeof x==="object")return "{"+Object.keys(x).sort().map(k=>JSON.stringify(k)+":"+canonical(x[k])).join(",")+"}";return JSON.stringify(x);}

function equal(a,b){return canonical(a)===canonical(b);}

function prefix(a,b){return a.length<=b.length&&a.every((x,i)=>x===b[i]);}

function makeCases(){
 const cases=[];
 for(let i=0;i<32;i++){
  const f=i%4,depth=6+Math.floor(i/8),k=131+11*i;let binders,leaf,solver;
  if(f===0){binders="(a b : ℕ) (h : a ≤ b)";leaf=j=>`a + ${k+j} ≤ b + ${k+j+1}`;solver="omega";}
  if(f===1){binders="(x y : ℚ)";leaf=j=>{const c=k+j;return `(x + y + ${c})^3 = x^3 + 3*x^2*y + 3*x*y^2 + y^3 + ${3*c}*x^2 + ${6*c}*x*y + ${3*c}*y^2 + ${3*c*c}*x + ${3*c*c}*y + ${c*c*c}`;};solver="ring";}
  if(f===2){binders="(x : ℕ) (s t : Finset ℕ)";leaf=j=>j%2===0?"x ∈ s ∪ t ↔ x ∈ s ∨ x ∈ t":"x ∈ s ∩ t ↔ x ∈ s ∧ x ∈ t";solver="simp [Finset.mem_union, Finset.mem_inter]";}
  if(f===3){binders="(P Q : Prop) (hp : P) (hq : Q)";leaf=j=>j%2===0?"P":"Q";solver="assumption";}
  // Give non-arithmetic families a unique visible binder, which does not
  // participate in the proof. IDs and size stay outside any trained policy.
  if(f>=2)binders+=` (tag${i} : ℕ)`;
  const target=Array.from({length:depth},(_,j)=>`(${leaf(j)})`).join(" ∧ ");
  const script=first=>Array.from({length:depth},(_,j)=>j===depth-1?[solver]:[j===0?first:"constructor",solver]).flat();
  const a=script("constructor"),b=script("refine ⟨?_, ?_⟩"),mid=Math.floor(a.length/2),last=a.length-1;
  const tape=[[a.slice(0,2),"open"],[b.slice(0,2),"open"],[a.slice(0,mid),"open"],[b.slice(0,mid),"open"],[[...a.slice(0,mid),BAD],"reject"],[b.slice(0,last),"open"],[a.slice(0,last),"open"],[a,"closed"]].map(([actions,expect],j)=>({index:j,actions,expect}));
  check(a.length<=32&&tape.every(x=>x.actions.length<=32),"script bounds");
  cases.push({id:`cog13g_${i}`,family:["nat_order","rat_ring","finset","prop"][f],depth,statement:`${binders} : ${target}`,scripts:[a,b],tape});
 }
 check(new Set(cases.map(x=>x.statement)).size===32,"duplicate registry");
 return cases;
}

function arms(){return [{name:"replay_lru",slots:0,evict:"lru",window:24},{name:"snapshot_lru",slots:16,evict:"lru",window:24},{name:"snapshot_recent",slots:16,evict:"recent",window:24},{name:"block_snapshot",slots:16,evict:"lru",window:1}];}

function cacheTransition(cache,req,slots,result){
 let hit=null;for(const x of cache)if(prefix(x.actions,req.actions)&&(!hit||x.actions.length>hit.actions.length))hit=x;
 let next=cache.map(e=>({...e,actions:[...e.actions]})),evicted=null,inserted=false;
 if(hit)next=next.filter(x=>!equal(x.actions,hit.actions)).concat([{...hit,touched:req.id,hits:hit.hits+1}]);
 if(slots&&result.accepted&&result.goals.length&&!next.some(x=>equal(x.actions,req.actions))){
  next.push({actions:req.actions,touched:req.id,hits:0});inserted=true;
  if(next.length>slots){const v=[...next].sort((a,b)=>a.touched-b.touched)[0];evicted=v.actions;next=next.filter(x=>!equal(x.actions,v.actions));}
 }
 return {restored:hit?.actions.length??0,inserted,evicted,cache:next,executed:req.actions.length-(hit?.actions.length??0)};
}

function validateReply(cache,req,slots,e){
 check(e.id===req.id&&typeof e.result?.accepted==="boolean","native id/outcome");
 check(e.result.accepted===(req.expect!=="reject"),"unexpected tactic outcome");
 if(e.result.accepted)check(Array.isArray(e.result.goals)&&e.result.goals.every(x=>typeof x==="string")&&Boolean(e.result.goals.length)===(req.expect==="open"),"goal outcome");
 const a=cacheTransition(cache,req,slots,e.result);
 for(const k of Object.keys(a))check(equal(a[k],e[k]),"cache transition "+k);
 return a.cache;
}

function nextChoice(roots,queue,window,next){
 while(queue.length<window&&next<roots.length)queue.push(next++);
 return {index:queue.shift()??null,next};
}

function evictionScore(session,anon){
 const perCommand=session.executed>0?session.queryCPU/session.executed:0;
 // For each observed request on this root, exact deepest retained prefix.
 const reusable=session.recent.reduce((sum,a)=>sum+Math.max(0,...session.cache.filter(x=>prefix(x.actions,a)).map(x=>x.actions.length)),0);
 return (session.startCPU+perCommand*reusable)/Math.max(1048576,anon);
}

function chooseVictim(live,policy,exclude){
 const c=[...live.values()].filter(s=>s.root!==exclude).sort((a,b)=>a.last-b.last||a.root-b.root);
 if(!c.length)return null;
 if(policy==="lru")return c[0];
 return c.sort((a,b)=>evictionScore(a,a.handle.anon())-evictionScore(b,b.handle.anon())||a.last-b.last||a.root-b.root)[0];
}

function sameGoalKey(c,q){return c.id+":"+canonical(q.actions);}

async function portfolio(backend,arm,cases,limit){
 const roots=cases.map(c=>({c,cursor:0,terminal:false,certified:false,reason:null,attempts:0,certificates:[]}));
 const live=new Map(),queue=[],events=[],goals=new Map(),initial=new Map(),start=backend.sample(),wall0=backend.now();
 let next=0,tick=0,opens=0,evictions=0,pressureEvictions=0,attempted=0,replied=0,unknown=0,executed=0,restored=0,peakLive=0,reserve=128*1048576,stop="all_terminal",error=null;
 const spent=()=>backend.sample().cpu-start.cpu;
 const within=()=>spent()<limit.cpu&&backend.now()-wall0<limit.wallMs;
 const emit=(kind,data={})=>{const event={kind,...data,cgroup:backend.sample(),elapsedMs:backend.now()-wall0};events.push(event);backend.event?.(event);return event;};
 const close=async(s,reason)=>{
  live.delete(s.root);const r=await s.handle.close();
  if(reason==="slot"||reason==="before_open"||reason==="before_cert"||reason==="after_reply"){evictions++;if(reason!=="slot")pressureEvictions++;}
  emit("close",{root:s.root,session:s.session,reason,process:r??null});
 };
 const pressure=async(reason,exclude=null,opening=false)=>{
  for(;;){
   const n=backend.sample().current,threshold=(opening?.90:.85)*limit.bytes;
   if(n+(opening?reserve:0)<=threshold)break;
   // Freeze anonymous-memory samples before computing and logging the decision.
   const candidates=[...live.values()].filter(s=>s.root!==exclude).map(s=>({root:s.root,last:s.last,startCPU:s.startCPU,queryCPU:s.queryCPU,executed:s.executed,recent:s.recent,cache:s.cache,anon:s.handle.anon()}));
   if(!candidates.length)break;
   const rank=x=>arm.evict==="lru"?x.last:evictionScore(x,x.anon);
   const victim=[...candidates].sort((a,b)=>rank(a)-rank(b)||a.last-b.last||a.root-b.root)[0];
   const v=live.get(victim.root);
   emit("eviction_decision",{reason,policy:arm.evict,exclude,victim:v.root,candidates,reserve,threshold});
   await close(v,reason);
  }
 };
 try{
  while(within()){
   const choice=nextChoice(roots,queue,arm.window,next);next=choice.next;const i=choice.index;
   if(i===null)break;
   const r=roots[i],q=r.c.tape[r.cursor];check(!r.terminal&&q,"schedule root");
   let s=live.get(i);
   if(!s){
    await pressure("before_open",null,true);
    if(live.size>=arm.window){const v=chooseVictim(live,arm.evict,null);check(v,"slot victim");await close(v,"slot");}
    if(!within()){stop="budget_before_open";break;}
    const before=backend.sample();const session=opens++;emit("open_attempt",{root:i,session,reserve});
    try{
     const handle=await backend.open(r.c,arm.slots,session,within);
     const after=backend.sample();reserve=Math.max(reserve,Math.max(0,after.current-before.current),handle.peakDelta??0);
     if(initial.has(i))check(equal(initial.get(i),handle.initial),"root goals changed on reopen");else initial.set(i,handle.initial);
     s={root:i,session,handle,id:0,cache:[],last:tick,startCPU:after.cpu-before.cpu,queryCPU:0,executed:0,recent:[]};live.set(i,s);peakLive=Math.max(peakLive,live.size);
     emit("opened",{root:i,session,initial:handle.initial,startCPU:s.startCPU,anon:handle.anon(),reserve});
    }catch(e){r.terminal=true;r.reason="startup_unknown";unknown++;emit("unknown",{root:i,session,stage:"open",error:String(e)});if(!within()){stop="budget_open";break;}continue;}
   }
   if(!within()){stop="budget_before_query";break;}
   const req={...q,id:++s.id},cpu0=backend.sample().cpu,t0=backend.now();r.attempts++;attempted++;tick++;
   emit("query",{root:i,session:s.session,request:req});
   let e;
   try{e=await s.handle.query(req,within);}catch(err){unknown++;r.terminal=true;r.reason="probe_unknown";emit("unknown",{root:i,session:s.session,stage:"query",error:String(err)});await close(s,"unknown");continue;}
   const cpu=backend.sample().cpu-cpu0;replied++;
   // Invalid native replies are hard evidence failures, not policy wins.
   s.cache=validateReply(s.cache,req,arm.slots,e);
   if(e.result.accepted){const key=sameGoalKey(r.c,q);if(goals.has(key))check(equal(goals.get(key),e.result.goals),"repeated exact-prefix goal mismatch");else goals.set(key,e.result.goals);}
   s.queryCPU+=cpu;s.executed+=e.executed;s.last=tick;s.recent.push(q.actions);if(s.recent.length>12)s.recent.shift();
   executed+=e.executed;restored+=e.restored;r.cursor++;
   emit("reply",{root:i,session:s.session,request:req,reply:e,queryCPU:cpu,queryWallMs:backend.now()-t0,anon:s.handle.anon()});
   if(q.expect==="closed"){
    await close(s,"closed");r.terminal=true;r.reason="certificate_budget_or_reject";
    await pressure("before_cert",null,true);
    for(let rep=0;rep<2;rep++){
     if(!within())break;emit("cert_attempt",{root:i,rep,actions:q.actions});
     let cert;try{cert=await backend.certify(r.c,q.actions,rep,within);}catch(err){cert={accepted:false,error:String(err)};}
     r.certificates.push(cert);emit("certificate",{root:i,rep,certificate:cert});
     if(!cert.accepted)break;
    }
    const creditClock=backend.sample(),creditWall=backend.now()-wall0;
    if(r.certificates.length===2&&r.certificates.every(x=>x.accepted)&&r.certificates[0].sourceHash===r.certificates[1].sourceHash&&creditClock.cpu-start.cpu<limit.cpu&&creditWall<limit.wallMs){
     r.certified=true;r.reason="certified";emit("credit",{root:i,creditClock,creditWall});
    }
   }else{
    queue.push(i);
    await pressure("after_reply",null,false);
   }
  }
  if(!within()&&stop==="all_terminal")stop="budget";
 }catch(e){error=String(e);stop="evidence_failure";emit("failure",{error});}
 finally{for(const s of [...live.values()])await close(s,"shutdown");}
 for(const r of roots)if(!r.terminal){r.terminal=true;r.reason=stop;}
 const end=backend.sample(),cpu=end.cpu-start.cpu,wall=backend.now()-wall0,certified=roots.filter(r=>r.certified).length;
 return {arm,limit,stop,error,start,end,cpu,wall,certified,perCpuHour:cpu>0?certified*3600e6/cpu:0,opens,evictions,pressureEvictions,peakLive,attempted,replied,unknown,executed,restored,roots,events,pressureObserved:end.peak>=.85*limit.bytes||pressureEvictions>0||end.maxEvents>start.maxEvents||end.oomEvents>start.oomEvents,oom:end.oomEvents>start.oomEvents,overrunCpu:Math.max(0,cpu-limit.cpu)};
}

function audit(result){
 const {arm,limit}=result,roots=result.roots,cache=new Map(),currentSession=new Map(),pending=new Map(),pendingOpen=new Map(),pendingCert=new Map(),cursor=roots.map(()=>0),terminal=new Set(),credited=new Set(),certs=new Map(),initial=new Map(),seenGoals=new Map(),queue=[];let next=0,expected=null,opens=0,replies=0,attempts=0,unknown=0,executed=0,restored=0,evictions=0,pressureEvictions=0,peakLive=0,lastCPU=result.start.cpu,lastWall=0;
 const getRoot=()=>{const q=nextChoice(roots,queue,arm.window,next);next=q.next;return q.index;};
 for(const e of result.events){
  check(e.cgroup.cpu>=lastCPU&&e.elapsedMs>=lastWall,"nonmonotonic clock");lastCPU=e.cgroup.cpu;lastWall=e.elapsedMs;
  if(e.kind==="open_attempt"){
   if(expected===null)expected=getRoot();check(e.root===expected&&!currentSession.has(e.root)&&e.session===opens,"open schedule");opens++;pendingOpen.set(e.session,e.root);
  }else if(e.kind==="opened"){
   check(pendingOpen.get(e.session)===e.root&&currentSession.size<arm.window,"session custody");pendingOpen.delete(e.session);
   check(!initial.has(e.root)||equal(initial.get(e.root),e.initial),"root observation changed");initial.set(e.root,e.initial);
   currentSession.set(e.root,e.session);cache.set(e.session,{entries:[],id:0});peakLive=Math.max(peakLive,currentSession.size);
  }else if(e.kind==="eviction_decision"){
   const candidates=e.candidates,rank=x=>arm.evict==="lru"?x.last:evictionScore(x,x.anon);
   const sorted=[...candidates].sort((a,b)=>rank(a)-rank(b)||a.last-b.last||a.root-b.root);
   check(sorted.length&&e.victim===sorted[0].root&&candidates.every(c=>currentSession.has(c.root)),"eviction decision");
  }else if(e.kind==="close"){
   check(currentSession.get(e.root)===e.session,"closed wrong session");currentSession.delete(e.root);cache.delete(e.session);
   if(["slot","before_open","before_cert","after_reply"].includes(e.reason)){evictions++;if(e.reason!=="slot")pressureEvictions++;}
  }else if(e.kind==="query"){
   if(expected===null)expected=getRoot();check(e.root===expected&&currentSession.get(e.root)===e.session&&!pending.has(e.session),"query schedule");
   const s=cache.get(e.session),q=roots[e.root].c.tape[cursor[e.root]];
   check(equal({...q,id:s.id+1},e.request),"request tape");s.id++;pending.set(e.session,e);attempts++;
  }else if(e.kind==="reply"){
   const p=pending.get(e.session),s=cache.get(e.session);check(p&&p.root===e.root&&equal(p.request,e.request),"reply binding");
   s.entries=validateReply(s.entries,e.request,arm.slots,e.reply);pending.delete(e.session);
   if(e.reply.result.accepted){const key=sameGoalKey(roots[e.root].c,e.request);check(!seenGoals.has(key)||equal(seenGoals.get(key),e.reply.result.goals),"goal mismatch");seenGoals.set(key,e.reply.result.goals);}
   replies++;executed+=e.reply.executed;restored+=e.reply.restored;cursor[e.root]++;expected=null;
   if(e.request.expect==="closed")terminal.add(e.root);else queue.push(e.root);
  }else if(e.kind==="unknown"){
   unknown++;check(e.root===expected,"unknown schedule");
   if(e.stage==="query"){check(pending.has(e.session),"unknown without query");pending.delete(e.session);}else{check(pendingOpen.has(e.session),"unknown without open");pendingOpen.delete(e.session);}
   terminal.add(e.root);expected=null;
  }else if(e.kind==="cert_attempt"){
   check(terminal.has(e.root)&&!pendingCert.has(e.root)&&equal(e.actions,roots[e.root].c.scripts[0])&&e.rep===(certs.get(e.root)?.length??0),"certificate attempt");
   pendingCert.set(e.root,e.rep);
  }else if(e.kind==="certificate"){
   check(pendingCert.get(e.root)===e.rep,"certificate receipt");pendingCert.delete(e.root);
   const a=certs.get(e.root)??[];a.push(e.certificate);certs.set(e.root,a);
  }else if(e.kind==="credit"){
   const a=certs.get(e.root);check(!credited.has(e.root)&&a?.length===2&&a.every(c=>c.accepted)&&a[0].sourceHash===a[1].sourceHash&&e.creditClock.cpu-result.start.cpu<limit.cpu&&e.creditWall<limit.wallMs&&e.creditClock.cpu<=e.cgroup.cpu&&e.creditWall<=e.elapsedMs,"uncertified credit");
   credited.add(e.root);
  }else if(e.kind!=="failure")throw Error("unknown event "+e.kind);
 }
 for(let i=0;i<roots.length;i++)check(roots[i].cursor===cursor[i]&&roots[i].certified===credited.has(i),"root summary");
 check(credited.size===result.certified&&opens===result.opens&&attempts===result.attempted&&replies===result.replied&&unknown===result.unknown&&executed===result.executed&&restored===result.restored&&evictions===result.evictions&&pressureEvictions===result.pressureEvictions&&peakLive===result.peakLive,"terminal counters");
 check(currentSession.size===0&&pendingOpen.size===0&&pending.size===0&&pendingCert.size===0,"unclosed session or request");
 return {events:result.events.length,certified:credited.size,goals:seenGoals.size};
}

function fixtureBackend(bytes){
 let cpu=0,wall=0,peak=0,current=128*1048576,n=0;
 const sample=()=>({cpu,current,peak,anon:current,file:0,maxEvents:0,oomEvents:0});
 const advance=(work,ms)=>{cpu+=work;wall+=ms;peak=Math.max(peak,current);};
 return {sample,now:()=>wall,open:async(c,slots,session)=>{const bytes=120*1048576;current+=bytes;n++;advance(300000,10);let cache=[];return {initial:["fixture "+c.id],peakDelta:bytes,anon:()=>bytes,query:async req=>{let result=req.expect==="reject"?{accepted:false,error:"fixture"}:{accepted:true,goals:req.expect==="closed"?[]:["fixture "+canonical(req.actions)]};let tr=cacheTransition(cache,req,slots,result);cache=tr.cache;advance(1000+tr.executed*2000,2);return {id:req.id,result,...tr};},close:async()=>{current-=bytes;n--;check(n>=0,"fixture process leak");advance(5000,1);return {fixture:true};}};},certify:async(c,a)=>{advance(200000,10);return {accepted:true,sourceHash:c.id+canonical(a),fixture:true};}};
}

async function fixture(){
 const cases=makeCases(),configs=arms(),rows=[],bad={credit:false,work:false,missing:false,eviction:false};
 for(const bytes of [2,4,8].map(x=>x*1073741824))for(const arm of configs){
  const r=await portfolio(fixtureBackend(bytes),arm,cases,{bytes,cpu:180e6,wallMs:480000});check(!r.error&&r.certified===32,"fixture result");const a=audit(r);
  rows.push({capGiB:bytes/1073741824,arm:arm.name,certified:r.certified,opens:r.opens,evictions:r.evictions,pressureEvictions:r.pressureEvictions,executed:r.executed,restored:r.restored,peakLive:r.peakLive,events:a.events});
  if(bytes===2147483648&&arm.name==="snapshot_recent"){
   const mutate=(k,fn)=>{const x=JSON.parse(JSON.stringify(r));fn(x);try{audit(x);}catch{bad[k]=true;}};
   mutate("credit",x=>x.events.find(e=>e.kind==="certificate").certificate.accepted=false);
   mutate("work",x=>x.events.find(e=>e.kind==="reply").reply.executed++);
   mutate("missing",x=>x.events.splice(x.events.findIndex(e=>e.kind==="reply"),1));
   mutate("eviction",x=>x.events.find(e=>e.kind==="eviction_decision").victim=999);
  }
 }
 check(Object.values(bad).every(Boolean),"negative fixture");
 return {schema:"cog13g.fixture.v1",mock:true,rows,negative:bad};
}
export {BAD,check,canonical,equal,makeCases,arms,portfolio,audit,fixture};
