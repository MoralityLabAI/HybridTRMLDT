// Frozen request tapes and independent cache-receipt checker.
const badAction="exact cog13d_missing_identifier";
function check(ok,msg){if(!ok)throw Error(msg);}

function equal(a,b){return JSON.stringify(a)===JSON.stringify(b);}

function prefix(a,b){return a.length<=b.length&&a.every((x,i)=>x===b[i]);}

function makeCases(){
 const rows=[];
 for(let i=0;i<8;i++){
  const f=i%4,depth=4+Math.floor(i/4),k=103+17*i;let binders,leaf,solvers=[];
  if(f===0){binders="(a b : ℕ) (h : a ≤ b)";leaf=j=>`a + ${k+j} ≤ b + ${k+j+1}`;solvers=Array(depth).fill("omega");}
  if(f===1){binders="(x : ℚ)";leaf=j=>`(x + ${k+j}) * (x + ${k+j}) = x * x + ${2*(k+j)} * x + ${(k+j)**2}`;solvers=Array(depth).fill("ring");}
  if(f===2){binders="(x : ℕ) (s t : Finset ℕ)";leaf=j=>j%2===0?"x ∈ s ∪ t ↔ x ∈ s ∨ x ∈ t":"x ∈ s ∩ t ↔ x ∈ s ∧ x ∈ t";solvers=Array(depth).fill("simp [Finset.mem_union, Finset.mem_inter]");}
  if(f===3){binders="(P Q : Prop) (hp : P) (hq : Q)";leaf=j=>j%2===0?"P":"Q";solvers=Array(depth).fill("assumption");}
  const target=Array.from({length:depth},(_,j)=>`(${leaf(j)})`).join(" ∧ ");
  const script=(first)=>solvers.flatMap((s,j)=>j===depth-1?[s]:[j===0?first:"constructor",s]);
  const a=script("constructor"),b=script("refine ⟨?_, ?_⟩"),requests=[];
  const add=(actions,expect)=>requests.push({id:requests.length+1,actions:[...actions],expect});
  for(let d=1;d<a.length;d++){
   const x=a.slice(0,d),y=b.slice(0,d);
   add(x,"open");add(y,"open");add([...x,badAction],"reject");
   if(d>2)add(a.slice(0,d-2),"open");
   add([...y,badAction],"reject");add(x,"open");
  }
  add(a,"closed");add(b,"closed");
  rows.push({id:`cog13d_${i}`,family:["nat_order","rat_ring","finset_membership","propositional"][f],depth,statement:`${binders} : ${target}`,scripts:[a,b],requests});
 }
 check(new Set(rows.map(x=>x.statement)).size===8,"duplicate statements");
 return rows;
}

function arms(){return [{name:"root_replay",cap:0,policy:"lru"},{name:"lru1",cap:1,policy:"lru"},{name:"lru4",cap:4,policy:"lru"},{name:"value4",cap:4,policy:"value"},{name:"lru16",cap:16,policy:"lru"}];}

function choice(cache,actions){let hit=null;for(const e of cache)if(prefix(e.actions,actions)&&(!hit||e.actions.length>hit.actions.length))hit=e;return hit;}

function weight(policy,e){return policy==="value"?e.actions.length*(e.hits+1):e.touched;}

function transition(cache,req,arm,result){
 const selected=choice(cache,req.actions),restored=selected?.actions.length??0;
 let next=cache.map(x=>({...x,actions:[...x.actions]})),inserted=false,evicted=null;
 if(selected){next=next.filter(x=>!equal(x.actions,selected.actions));next.push({actions:[...selected.actions],touched:req.id,hits:selected.hits+1});}
 if(arm.cap&&result.accepted&&result.goals.length&&!next.some(x=>equal(x.actions,req.actions))){
  next.push({actions:[...req.actions],touched:req.id,hits:0});inserted=true;
  if(next.length>arm.cap){let victim=next[0];for(const e of next)if(weight(arm.policy,e)<weight(arm.policy,victim)||(weight(arm.policy,e)===weight(arm.policy,victim)&&e.touched<victim.touched))victim=e;evicted=victim.actions;next=next.filter(x=>!equal(x.actions,victim.actions));}
 }
 return {restored,inserted,evicted,cache:next};
}

function expected(req,result){
 check(typeof result?.accepted==="boolean","missing native result");
 if(req.expect==="reject")check(!result.accepted,"negative tactic accepted");
 else check(result.accepted&&Array.isArray(result.goals)&&(req.expect==="closed"?result.goals.length===0:result.goals.length>0),"unexpected native outcome "+req.expect);
}

function audit(c,arm,events,baseline=null){
 let cache=[],executed=0,restored=0,peak=0,hits=0,insertions=0,evictions=0;
 check(events.length===c.requests.length,"incomplete request tape");
 for(let i=0;i<c.requests.length;i++){
  const req=c.requests[i],e=events[i];check(e.id===req.id,"wrong response id");expected(req,e.result);
  if(baseline){const b=baseline[i];check(b.result.accepted===e.result.accepted&&(!e.result.accepted||equal(b.result.goals,e.result.goals)),"cached result differs from root replay");}
  // Independent expected transition. For this tape all commands preceding the
  // single deliberately failing final command succeed in baseline.
  const t=transition(cache,req,arm,e.result);check(t.restored===e.restored&&t.inserted===e.inserted&&equal(t.evicted,e.evicted)&&equal(t.cache,e.cache),"cache transition mismatch");
  check(e.executed===req.actions.length-t.restored,"attempted-command mismatch");
  if(baseline)check(e.executed+e.restored===baseline[i].executed,"cost conservation");
  cache=t.cache;executed+=e.executed;restored+=e.restored;hits+=Number(e.restored>0);insertions+=Number(e.inserted);evictions+=Number(e.evicted!==null);peak=Math.max(peak,cache.length);check(peak<=arm.cap,"capacity violation");
 }
 return {requests:events.length,executed,restored,hits,insertions,evictions,peak};
}

function fixtureReply(req,arm,cache){
 const result=req.expect==="reject"?{accepted:false,error:"fixture negative"}:{accepted:true,goals:req.expect==="closed"?[]:["fixture open "+req.actions.length]};
 const t=transition(cache,req,arm,result);return {id:req.id,result,...t,executed:req.actions.length-t.restored};
}

function fixture(){
 const cases=makeCases(),configs=arms(),rows={},negative={forgedHit:false,forgedResult:false,forgedCache:false,missingRequest:false};
 for(const c of cases){
  let base=null;
  for(const arm of configs){
   let cache=[],events=[];for(const req of c.requests){const e=fixtureReply(req,arm,cache);events.push(e);cache=e.cache;}
   const a=audit(c,arm,events,base);if(!base)base=events;
   const t=rows[arm.name]??={requests:0,executed:0,restored:0,hits:0,insertions:0,evictions:0,peak:0};
   for(const k of Object.keys(t))if(k==="peak")t[k]=Math.max(t[k],a[k]);else t[k]+=a[k];
   if(c===cases[0]&&arm.name==="lru4"){
    const mutate=(key,fn)=>{let forged=JSON.parse(JSON.stringify(events));fn(forged);try{audit(c,arm,forged,base);}catch{negative[key]=true;}};
    mutate("forgedHit",a=>a[3].restored++);
    mutate("forgedResult",a=>a.find(e=>e.result.accepted&&e.result.goals.length).result.goals=["forged"]);
    mutate("forgedCache",a=>a[0].cache=[]);
    mutate("missingRequest",a=>a.pop());
   }
  }
 }
 check(Object.values(negative).every(Boolean),"negative sentinel escaped");
 for(const a of Object.values(rows))check(a.requests===rows.root_replay.requests&&a.executed+a.restored===rows.root_replay.executed,"aggregate conservation");
 return {schema:"cog13d.fixture.v1",syntheticFixture:true,cases:cases.length,arms:configs.length,requestRows:cases.reduce((n,c)=>n+c.requests.length,0)*configs.length,summary:rows,negative};
}
export {check,equal,makeCases,arms,audit,fixture,badAction};
