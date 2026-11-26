// Paired, stratified descriptive readout. No retuning or confirmation claim.
function pairedAnalysis(result){
 const pairs=[],taskMap=new Map(result.tasks.map(t=>[t.id,t])),runs=result.runs,scenarios=[];
 const rng=seed=>{let x=seed>>>0;return n=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)%n;};};
 for(const split of ["calibration","ordinary","deeper"])for(const cap of [128,256,512]){
  const learned=runs.filter(r=>r.split===split&&r.cap===cap&&r.policy==="learned");
  for(const policy of ["lru","recent_density","shuffled"]){
   const ctrl=new Map(runs.filter(r=>r.split===split&&r.cap===cap&&r.policy===policy).map(r=>[r.root,r]));
   const rows=learned.map(l=>({root:l.root,family:taskMap.get(l.root).family,learned:l.executed,control:ctrl.get(l.root).executed,featureEvals:l.featureEvals}));
   const gain=rows.reduce((s,x)=>s+x.control-x.learned,0),control=rows.reduce((s,x)=>s+x.control,0),f=rows.reduce((s,x)=>s+x.featureEvals,0);
   const strata=[0,1,2,3].map(f=>rows.filter(x=>x.family===f)),rand=rng(9137+cap+split.length*117+policy.length*997),boot=[];
   for(let j=0;j<4000;j++){let g=0,b=0;for(const group of strata)for(let k=0;k<group.length;k++){const a=group[rand(group.length)];g+=a.control-a.learned;b+=a.control;}boot.push(100*g/b);}
   boot.sort((a,b)=>a-b);
   pairs.push({split,cap,control:policy,roots:rows.length,wins:rows.filter(x=>x.learned<x.control).length,ties:rows.filter(x=>x.learned===x.control).length,losses:rows.filter(x=>x.learned>x.control).length,relativeSavedPct:100*gain/control,descriptiveStratifiedBootstrap95:[boot[100],boot[3899]],additionalFeatureCostBreakEven:gain/f});
  }
  const arms=result.summary.splits[split][cap];
  for(const [name,weight,extra,byte] of [["work_only",0,0,0],["ranking_plus_features",1,2,0],["ranking_features_transfer",1,2,0.01]]){
   const scores=Object.fromEntries(Object.entries(arms).filter(([p])=>p!=="oracle12").map(([p,a])=>[p,a.work+weight*a.rankings+extra*a.featureEvals+byte*(a.read+a.written)]));
   scenarios.push({split,cap,scenario:name,scores,best:Object.keys(scores).sort((a,b)=>scores[a]-scores[b])[0]});
  }
 }
 return {schema:"cog13e.paired.v1",note:"Descriptive paired root-stratified bootstrap (4,000 resamples), no inference gate; costs are declared synthetic sensitivity only.",pairs,scenarios};
}
export {pairedAnalysis};
