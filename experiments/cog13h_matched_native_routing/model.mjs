// COG-13H: archived COG-13C features/codec, independent matched models.
const MENU=["assumption","rfl","constructor","intro hnew","simp","simp_all","omega","linarith","ring","norm_num","positivity","aesop","exact h","all_goals assumption","all_goals omega","all_goals ring"];
function requireIt(x,m){if(!x)throw Error(m);}

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
  scores.push(model.kind==="flat"?flatForward(model,x).score:forward(model,x,input).score);
 }
 requireIt(scores.every(Number.isFinite),"nonfinite score");
 const order=scores.map((_,i)=>i).sort((a,b)=>scores[b]-scores[a]||a-b);
 return {order,scores,bytes:packet.length+1,clipping,packet:Array.from(packet),reply:order[0]};
}


function flatForward(model,x){
 const h=model.W.map((w,i)=>Math.tanh(dot(w,x)+model.b[i]));
 return {h,score:dot(model.v,h)+dot(model.skip,x.slice(0,8))+model.o};
}

function fit(rows,kind,seed){
 requireIt(["flat","loop"].includes(kind)&&[1701,1709,1723].includes(seed),"fit config");
 requireIt(rows.length===224&&new Set(rows.map(r=>r.root)).size===8&&rows.every(r=>r.split==="train"&&MENU.includes(r.action)&&(r.target===0||r.target===1)),"train custody");
 const rand=rng(seed),orderRng=rng(seed^0x8e13a1),H=kind==="loop"?12:16,init=n=>Array.from({length:n},()=>.12*(2*rand()-1));
 const model={kind,W:Array.from({length:H},()=>init(32)),b:zero(H),v:init(H),o:0};
 if(kind==="loop")model.R=Array.from({length:H},()=>init(H));else model.skip=init(8);
 const losses=[],vectors=rows.map(r=>vector(r.goals,r.action)),eta=.015;
 let updates=0;
 for(let epoch=0;epoch<20;epoch++){
  let loss=0;
  const idx=rows.map((_,i)=>i);
  for(let i=idx.length-1;i>0;i--){const j=Math.floor(orderRng()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
  for(const j of idx){
   const x=vectors[j],y=rows[j].target,f=kind==="loop"?forward(model,x):flatForward(model,x),p=1/(1+Math.exp(-f.score)),d=p-y;
   loss-=y*Math.log(Math.max(p,1e-12))+(1-y)*Math.log(Math.max(1-p,1e-12));
   if(kind==="loop"){
    let delta=model.v.map(v=>v*d),gW=model.W.map(w=>zero(w.length)),gR=model.R.map(w=>zero(w.length)),gb=zero(H);
    for(let t=3;t>=1;t--){
     const z=delta.map((v,i)=>v*(1-f.h[t][i]**2));
     for(let i=0;i<H;i++){gb[i]+=z[i];for(let k=0;k<32;k++)gW[i][k]+=z[i]*x[k];for(let k=0;k<H;k++)gR[i][k]+=z[i]*f.h[t-1][k];}
     delta=Array.from({length:H},(_,k)=>model.R.reduce((a,r,i)=>a+r[k]*z[i],0));
    }
    for(let i=0;i<H;i++){model.v[i]-=eta*d*f.h[3][i];model.b[i]-=eta*gb[i];for(let k=0;k<32;k++)model.W[i][k]-=eta*gW[i][k];for(let k=0;k<H;k++)model.R[i][k]-=eta*gR[i][k];}
   }else{
    const dz=model.v.map((v,i)=>v*d*(1-f.h[i]**2));
    for(let i=0;i<H;i++){model.v[i]-=eta*d*f.h[i];model.b[i]-=eta*dz[i];for(let k=0;k<32;k++)model.W[i][k]-=eta*dz[i]*x[k];}
    for(let k=0;k<8;k++)model.skip[k]-=eta*d*x[k];
   }
   model.o-=eta*d;updates++;
  }
  losses.push(loss/rows.length);
 }
 const parameters=model.W.flat().length+model.b.length+model.v.length+1+(kind==="loop"?model.R.flat().length:model.skip.length);
 requireIt(parameters===553&&updates===4480&&losses.every(Number.isFinite),"fit accounting");
 return {model,losses,parameters,updates,seed,kind};
}

function frequency(rows){return Object.fromEntries(MENU.map(a=>{const r=rows.filter(x=>x.action===a);return [a,(1+r.reduce((s,x)=>s+x.target,0))/(2+r.length)];}));}

function target(goal){const i=goal.lastIndexOf("⊢");return (i<0?goal:goal.slice(i+1)).replace(/\s+/g," ").trim();}

function topOps(t){
 let d=0,ops=new Set();for(const c of t){if("([{⟨".includes(c))d++;else if(")]}⟩".includes(c))d=Math.max(0,d-1);else if(d===0&&"∧∨→↔=≤≥<>".includes(c))ops.add(c);}return ops;
}

function symbolicOrder(goals,options,freq){
 const g=goals[0]??"",t=target(g),ops=topOps(t),rank=[];
 const push=(...a)=>{for(const v of a)if(!rank.includes(v))rank.push(v);};
 const local=g.slice(0,Math.max(0,g.lastIndexOf("⊢"))).split("\n").map(l=>l.split(":").slice(1).join(":").replace(/\s+/g," ").trim());
 if(local.includes(t))push("assumption","exact h");
 if(ops.has("→"))push("intro hnew","simp","aesop");
 else if(ops.has("↔"))push("simp","simp_all","aesop","constructor");
 else if(ops.has("∧"))push("constructor","simp","simp_all","aesop");
 else if(ops.has("∨"))push("simp","aesop","constructor");
 else if(/∈|Finset|∪|∩/.test(t))push("simp","simp_all","aesop");
 else if(/ℕ|ℤ/.test(g)&&/[≤≥<>=]/.test(t))push("omega","norm_num","simp","linarith");
 else if(/ℚ|ℝ/.test(g)&&ops.has("="))push("ring","norm_num","linarith","simp");
 else if(ops.has("="))push("rfl","simp","ring","omega","norm_num");
 if(goals.length>1){
  const ts=goals.map(target);
  if(ts.every(x=>/[≤≥<>]/.test(x)&&!/[∧∨→↔]/.test(x))&&goals.every(x=>/ℕ|ℤ/.test(x)))rank.unshift("all_goals omega");
  if(ts.every(x=>x.includes("=")&&!/[∧∨→↔]/.test(x))&&goals.every(x=>/ℚ|ℝ/.test(x)))rank.unshift("all_goals ring");
  if(ts.every(x=>/^[\p{L}\p{N}_]+$/u.test(x)))rank.unshift("all_goals assumption");
 }
 push("assumption","simp","rfl","simp_all","omega","ring","linarith","norm_num","positivity","constructor","intro hnew","exact h","aesop","all_goals assumption","all_goals omega","all_goals ring");
 return options.map((_,i)=>i).sort((a,b)=>rank.indexOf(options[a])-rank.indexOf(options[b])||freq[options[b]]-freq[options[a]]||a-b);
}

function rankPolicy(policy,goals,options,models){
 requireIt(options.length>0&&options.every(a=>MENU.includes(a)),"action menu");
 if(policy==="frequency"){const order=options.map((_,i)=>i).sort((a,b)=>models.freq[options[b]]-models.freq[options[a]]||a-b);return {order,bytes:0,macs:0,packet:null};}
 if(policy==="symbolic")return {order:symbolicOrder(goals,options,models.freq),bytes:0,macs:0,packet:null};
 const kind=policy==="flat"?"flat":"loop",mode=policy==="loop_rank4"?"rank4":policy==="loop_q4"?"rank4q":"dense";
 const m=models[kind].model,p=transportRank(m,models.factors,goals,options,mode),macs=options.length*(kind==="flat"?536:mode==="dense"?828:620);
 return {...p,macs};
}

function fitAll(rows,seed){
 const loop=fit(rows,"loop",seed),flat=fit(rows,"flat",seed),factors=factorize(loop.model,rows);
 return {loop,flat,factors,freq:frequency(rows)};
}

function readout(groups,models){
 const out={};
 for(const g of groups){
  requireIt(g.split!=="train"&&g.actions.length===g.labels.length&&g.labels.every(y=>y===0||y===1),"retrospective input");
  for(const policy of ["frequency","symbolic","flat","loop","loop_rank4","loop_q4"]){
   const a=out[policy]??={groups:0,bothLabels:0,topProgress:0,bothTop:0,pairs:0,orderedPairs:0};
   const r=rankPolicy(policy,g.goals,g.actions,models),y=g.labels;
   a.groups++;a.topProgress+=y[r.order[0]];if(y.includes(0)&&y.includes(1)){a.bothLabels++;a.bothTop+=y[r.order[0]];}
   for(let i=0;i<r.order.length;i++)for(let j=i+1;j<r.order.length;j++)if(y[r.order[i]]!==y[r.order[j]]){a.pairs++;a.orderedPairs+=y[r.order[i]];}
  }
 }
 return out;
}
export {MENU,vector,fitAll,rankPolicy,readout};
