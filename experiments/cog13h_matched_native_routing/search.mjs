import {MENU,rankPolicy} from "./model.mjs";
function check(x,m){if(!x)throw Error(m);}

function same(a,b){return JSON.stringify(a)===JSON.stringify(b);}

function canon(x){if(Array.isArray(x))return "["+x.map(canon).join(",")+"]";if(x!==null&&typeof x==="object")return "{"+Object.keys(x).sort().map(k=>JSON.stringify(k)+":"+canon(x[k])).join(",")+"}";return JSON.stringify(x);}

function cases(){
 const out=[];
 for(let s=0;s<2;s++)for(let i=0;i<8;i++){
  const k=1013+29*i+317*s,f=i%4,d=s?4+(i%2):1+(i%2),gs=[];let binders;
  if(f===0){binders="(a b : ℕ) (h : a ≤ b)";for(let j=0;j<d;j++)gs.push(`a + ${k+j} ≤ b + ${k+j+2}`);}
  if(f===1){binders="(x : ℚ)";for(let j=0;j<d;j++){let n=k+j;gs.push(`(x - ${n}) * (x - ${n}) = x*x - ${2*n}*x + ${n*n}`);}}
  if(f===2){binders="(x : ℕ) (s t : Finset ℕ)";for(let j=0;j<d;j++)gs.push(j%2===0?`(x ∈ t ∪ s ↔ x ∈ t ∨ x ∈ s) ∧ (${k+j} : ℕ) ≤ ${k+j+2}`:`(x ∈ t ∩ s ↔ x ∈ t ∧ x ∈ s) ∧ (${k+j} : ℕ) < ${k+j+3}`);}
  if(f===3){binders="(a b : ℕ) (h : a ≤ b) (P : Prop) (hp : P)";for(let j=0;j<d;j++)gs.push(j%2===0?`P ∧ a + ${k+j} ≤ b + ${k+j+2}`:`P → (a + ${k+j} ≤ b + ${k+j+3}) ∧ P`);}
  out.push({id:`cog13h_${s}_${i}`,split:s?"deeper":"ordinary",family:["nat_order","rat_ring","finset","mixed_logic"][f],depth:d,statement:binders+" : "+gs.map(g=>"("+g+")").join(" ∧ ")});
 }
 check(new Set(out.map(x=>x.statement)).size===16,"registry duplicates");return out;
}

function arms(){return ["frequency","symbolic","flat","loop","loop_rank4","loop_q4","loop_replay"].map(name=>({name,policy:name==="loop_replay"?"loop":name,slots:name==="loop_replay"?0:16}));}

function prefix(a,b){return a.length<=b.length&&a.every((x,i)=>x===b[i]);}

function cacheStep(cache,actions,id,slots,result){
 let hit=null;for(const e of cache)if(prefix(e.actions,actions)&&(!hit||e.actions.length>hit.actions.length))hit=e;
 let next=cache.map(e=>({...e,actions:[...e.actions]})),inserted=false,evicted=null;
 if(hit)next=next.filter(e=>!same(e.actions,hit.actions)).concat([{...hit,touched:id,hits:hit.hits+1}]);
 if(slots&&result.accepted&&result.goals.length&&!next.some(e=>same(e.actions,actions))){
  next.push({actions:[...actions],touched:id,hits:0});inserted=true;
  if(next.length>slots){const v=[...next].sort((a,b)=>a.touched-b.touched)[0];evicted=v.actions;next=next.filter(e=>!same(e.actions,v.actions));}
 }
 return {cache:next,restored:hit?.actions.length??0,executed:actions.length-(hit?.actions.length??0),inserted,evicted};
}

function validate(cache,actions,id,slots,reply){
 check(reply.id===id&&typeof reply.result?.accepted==="boolean","cache reply");
 if(reply.result.accepted)check(Array.isArray(reply.result.goals)&&reply.result.goals.every(x=>typeof x==="string"),"goal type");
 const t=cacheStep(cache,actions,id,slots,reply.result);
 for(const k of Object.keys(t))check(canon(t[k])===canon(reply[k]),"cache mismatch "+k);
 return t.cache;
}

function node(goals,p=[]){return {goals:[...goals],prefix:[...p],tried:[]};}

function nextNode(stack){while(stack.length&&stack.at(-1).tried.length===MENU.length)stack.pop();return stack.at(-1);}

function advance(stack,n,actions,r){
 if(!r.accepted||same(n.goals,r.goals))return false;
 if(!r.goals.length)return true;
 if(actions.length<10&&r.goals.length<=8&&!stack.some(x=>same(x.goals,r.goals)))stack.push(node(r.goals,actions));
 return false;
}

async function search(c,arm,models,backend,seed){
 const start=backend.now(),deadline=start+45000,events=[],certs=[];let session=null,initial=null,cache=[],solved=false,reason="probe_budget",bytes=0,macs=0,executed=0,restored=0,peak=0,closedActions=null,attempts=0,setupBytes=0;
 if(!["frequency","symbolic"].includes(arm.policy))setupBytes=JSON.stringify({model:models[arm.policy==="flat"?"flat":"loop"].model,...(["loop_rank4","loop_q4"].includes(arm.policy)?{factors:models.factors}:{})}).length;
 try{
  session=await backend.open(c,arm,seed,deadline);initial=session.initial;
  check(Array.isArray(initial)&&initial.length>0,"initial goals");
  const stack=[node(initial)];
  while(attempts<32&&backend.now()<deadline){
   const n=nextNode(stack);if(!n){reason="search_exhausted";break;}
   const options=MENU.filter(a=>!n.tried.includes(a)),rank=rankPolicy(arm.policy,n.goals,options,models),action=options[rank.order[0]],actions=[...n.prefix,action];
   n.tried.push(action);attempts++;bytes+=rank.bytes;macs+=rank.macs;
   const t=backend.now();let reply=null,error=null;
   try{reply=await session.query(actions,attempts,deadline);}catch(e){error=String(e);}
   const e={id:attempts,options,before:[...n.goals],actions,action,rank,reply,error,queryMs:backend.now()-t,elapsedMs:backend.now()-start};events.push(e);
   if(!reply){reason="query_unknown";break;}
   cache=validate(cache,actions,attempts,arm.slots,reply);peak=Math.max(peak,cache.length);executed+=reply.executed;restored+=reply.restored;
   if(e.elapsedMs>=45000){reason="deadline";break;}
   if(advance(stack,n,actions,reply.result)){
    closedActions=actions;await session.close();session=null;reason="certificate_rejected_or_budget";
    for(let rep=0;rep<2&&backend.now()<deadline;rep++){
     let cert;try{cert=await backend.certify(c,arm,seed,actions,rep,deadline);}catch(e){cert={accepted:false,error:String(e)};}
     certs.push({...cert,elapsedMs:backend.now()-start});if(!cert.accepted)break;
    }
    solved=certs.length===2&&certs.every(x=>x.accepted&&x.elapsedMs<=45000)&&certs[0].sourceHash===certs[1].sourceHash&&backend.now()<=deadline;
    if(solved)reason="certified";
    break;
   }
  }
  if(reason==="probe_budget"&&backend.now()>=deadline)reason="deadline";
 }catch(e){reason=initial?"evidence_error":"startup_unknown";events.push({kind:"failure",error:String(e),elapsedMs:backend.now()-start});}
 finally{if(session)await session.close();}
 return {case:c,arm,seed,initial,events,certs,solved,reason,attempts,bytes,macs,executed,restored,peak,setupBytes,closedActions,elapsedMs:backend.now()-start};
}

function audit(r,models){
 check(r.attempts<=32&&r.events.filter(x=>x.id).length===r.attempts,"probe budget");
 if(!r.initial){check(!r.solved&&r.attempts===0&&r.reason==="startup_unknown","startup credit");return 0;}
 const stack=[node(r.initial)];let cache=[],executed=0,restored=0,bytes=0,macs=0,peak=0,closed=null,last=0,attempt=0;
 for(const e of r.events){
  if(e.kind==="failure"){check(!r.solved,"failure credit");continue;}
  const n=nextNode(stack);check(n&&!closed&&e.id===++attempt&&e.elapsedMs>=last,"event order");last=e.elapsedMs;
  const options=MENU.filter(a=>!n.tried.includes(a)),rank=rankPolicy(r.arm.policy,n.goals,options,models);
  check(same(e.before,n.goals)&&same(e.options,options)&&canon(e.rank)===canon(rank)&&e.action===options[rank.order[0]]&&same(e.actions,[...n.prefix,e.action]),"policy replay");
  n.tried.push(e.action);bytes+=rank.bytes;macs+=rank.macs;
  if(!e.reply){check(e.error&&r.reason==="query_unknown"&&!r.solved&&attempt===r.attempts,"unknown reply");break;}
  cache=validate(cache,e.actions,e.id,r.arm.slots,e.reply);peak=Math.max(peak,cache.length);executed+=e.reply.executed;restored+=e.reply.restored;
  if(e.elapsedMs<45000&&advance(stack,n,e.actions,e.reply.result))closed=e.actions;
 }
 check(same(closed,r.closedActions)&&executed===r.executed&&restored===r.restored&&bytes===r.bytes&&macs===r.macs&&peak===r.peak,"episode receipt");
 if(r.solved)check(closed&&r.reason==="certified"&&r.certs.length===2&&r.certs.every(c=>c.accepted&&c.elapsedMs<=45000)&&r.certs[0].sourceHash===r.certs[1].sourceHash,"uncertified success");
 return attempt;
}

function summarize(results){
 const summary={};
 for(const r of results){
  const a=(summary[r.case.split]??={})[r.arm.name]??={roots:0,solved:0,unknown:0,attempts:0,bytes:0,macs:0,executed:0,restored:0,elapsedMs:0};
  a.roots++;a.solved+=Number(r.solved);a.unknown+=Number(["startup_unknown","query_unknown","evidence_error"].includes(r.reason));
  for(const k of ["attempts","bytes","macs","executed","restored","elapsedMs"])a[k]+=r[k];
 }
 return summary;
}

function gates(results){
 const find=(id,arm)=>results.find(r=>r.case.id===id&&r.arm.name===arm),deep=cases().filter(c=>c.split==="deeper"),all=cases();
 const d=(a,b)=>deep.reduce((s,c)=>s+Number(find(c.id,a).solved)-Number(find(c.id,b).solved),0);
 const unique=deep.filter(c=>find(c.id,"loop").solved&&!find(c.id,"flat").solved&&!find(c.id,"symbolic").solved).map(c=>c.id);
 const summary=summarize(results),b=summary.deeper.loop,q=summary.deeper.loop_q4;
 return {unique,loopMinusFlat:d("loop","flat"),loopMinusSymbolic:d("loop","symbolic"),recurrence:unique.length>0&&d("loop","flat")>0&&d("loop","symbolic")>0,compression:b.solved>=2&&deep.filter(c=>find(c.id,"loop").solved&&!find(c.id,"loop_q4").solved).length<=1&&b.bytes>=8*q.bytes,cache:all.every(c=>find(c.id,"loop").solved===find(c.id,"loop_replay").solved)&&results.filter(r=>r.arm.name==="loop").reduce((s,r)=>s+r.executed,0)<results.filter(r=>r.arm.name==="loop_replay").reduce((s,r)=>s+r.executed,0)&&results.every(r=>r.reason!=="evidence_error")};
}

async function fixture(models){
 let tick=0;
 const backend={now:()=>tick,open:async(c,arm)=>{tick+=5;let cache=[];return {initial:["P : Prop\nhp : P\n⊢ P ∧ P"],query:async(actions,id)=>{tick+=2;const a=actions.at(-1),r=actions.length===1&&a==="constructor"?{accepted:true,goals:["P : Prop\nhp : P\n⊢ P"]}:actions[0]==="constructor"&&a==="assumption"?{accepted:true,goals:[]}:{accepted:false,error:"fixture rejection"};
 const t=cacheStep(cache,actions,id,arm.slots,r);cache=t.cache;return {id,result:r,...t};},close:async()=>{tick++;}};},certify:async(c,a,s,actions)=>{tick+=3;return {accepted:true,sourceHash:canon(actions),fixture:true};}};
 const rs=[];
 for(const a of arms()){const r=await search(cases()[0],a,models,backend,1701);audit(r,models);rs.push(r);}
 check(rs.every(x=>x.solved),"fixture failed");
 const good=rs.find(x=>x.arm.name==="loop"),bad={...good,events:JSON.parse(JSON.stringify(good.events))};bad.events[0].reply.executed++;
 let rejected=false;try{audit(bad,models);}catch{rejected=true;}check(rejected,"forged cache receipt passed");
 const noCert=JSON.parse(JSON.stringify(good));noCert.certs=[];let rejectedCert=false;try{audit(noCert,models);}catch{rejectedCert=true;}check(rejectedCert,"forged certificate passed");
 return {fixture:true,arms:rs.length,probes:rs.reduce((s,r)=>s+r.attempts,0),negativeCacheRejected:rejected,negativeCertificateRejected:rejectedCert,certifiedFixture:rs.length};
}
export {check,canon,cases,arms,search,audit,summarize,gates,fixture};
