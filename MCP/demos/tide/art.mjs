import { ellipse, glow, random, shape } from "../art-utils.mjs";

const w=960,h=460;
const palettes={
  quay:["#233a57","#76566c","#263b4f","#e3b88b",368,115],
  archive:["#253944","#716452","#273941","#ffcb83",620,130],
  tower:["#293756","#79617a","#374354","#eacac0",724,106],
  ferry:["#232c48","#605165","#283746","#dca681",309,120],
  causeway:["#213b52","#74737e","#304957","#f0c58e",714,103],
  arrive:["#505c70","#d39683","#38666b","#ffdd9f",593,128],
  watch:["#3c536b","#9b7d90","#355066","#e9b99b",688,133],
  lost:["#1f2b42","#484857","#252e41","#cfb3a6",337,117],
};

function roof(ctx,x,y,width,height,shade) {
  shape(ctx,[[x,y],[x+width*.24,y-height*.2],[x+width*.88,y-height*.16],[x+width,y],
    [x+width,y+height],[x,y+height]],shade);
  shape(ctx,[[x+width*.24,y-height*.2],[x+width*.88,y-height*.16],[x+width,y],
    [x+width*.7,y+2],[x,y]],"#172a37");
}

function tower(ctx,x,y,width,height,lit) {
  ctx.fillStyle="#273947";ctx.fillRect(x,y,width,height);
  shape(ctx,[[x,y],[x+width*.50,y-width*.75],[x+width,y]],"#162a39");
  shape(ctx,[[x+width*.72,y],[x+width,y],[x+width,y+height],[x+width*.72,y+height]],"#1e2d3a");
  if(lit) glow(ctx,x+width*.51,y+height*.31,width*.98,"#f5b4776b");
  ctx.fillStyle=lit?"#f4cf91":"#354a58";
  ctx.fillRect(x+width*.42,y+height*.21,width*.15,height*.18);
  ctx.fillStyle="#425462";ctx.fillRect(x+width*.23,y+height*.6,width*.11,height*.095);
}

function rock(ctx,x,y,size,dark) {
  shape(ctx,[[x-size*.6,y+size*.25],[x-size*.4,y-size*.3],[x+size*.13,y-size*.44],
    [x+size*.61,y-size*.09],[x+size*.54,y+size*.23]],dark?"#20323d":"#405567");
  shape(ctx,[[x-size*.4,y-size*.3],[x+size*.13,y-size*.44],[x+size*.59,y-size*.09],
    [x,y+size*.03],[x-size*.55,y+size*.02]],dark?"#314958":"#66727b");
}

function lamp(ctx,x,y,scale) {
  glow(ctx,x,y,50*scale,"#f6cb8066");
  ctx.fillStyle="#273340";ctx.fillRect(x-2*scale,y-5*scale,4*scale,42*scale);
  ctx.fillStyle="#f4cb82";ctx.fillRect(x-4*scale,y-6*scale,8*scale,11*scale);
  shape(ctx,[[x-8*scale,y-6*scale],[x,y-12*scale],[x+8*scale,y-6*scale]],"#252d38");
}

function arch(ctx,x,base) {
  ctx.lineWidth=16;ctx.strokeStyle="#304450";
  ctx.beginPath();ctx.moveTo(x-48,base);ctx.lineTo(x-48,base-63);ctx.arc(x,base-63,48,Math.PI,0);
  ctx.lineTo(x+48,base);ctx.stroke();
  ctx.lineWidth=3;ctx.strokeStyle="#77818790";
  ctx.beginPath();ctx.moveTo(x-45,base);ctx.lineTo(x-45,base-63);ctx.arc(x,base-63,45,Math.PI,0);ctx.stroke();
}

export function drawScene(ctx, scene) {
  const [top, horizon, water, sun, moonX, moonY]=palettes[scene]??palettes.quay;
  const rng=random([...scene].reduce((seed,char)=>seed*31+char.charCodeAt(0),37));
  ctx.save();ctx.scale(ctx.canvas.width/w,ctx.canvas.height/h);
  const grad=ctx.createLinearGradient(0,0,0,295);
  grad.addColorStop(0,top);grad.addColorStop(1,horizon);
  ctx.fillStyle=grad;ctx.fillRect(0,0,w,h);
  glow(ctx,moonX,moonY,110,sun+"62");
  ellipse(ctx,moonX,moonY,34,34,sun);
  for(let i=0;i<100;i++){
    const x=rng()*w,y=rng()*234;
    ctx.fillStyle=i%3?"#f1d6c353":"#d1cdcb76";ctx.fillRect(x,y,i%9?1:2,1);
  }
  for(let i=0;i<35;i++){
    const x=rng()*w,y=103+rng()*173;
    ellipse(ctx,x,y,18+rng()*71,.8+rng()*3,"#e2a99123");
  }
  shape(ctx,[[0,270],[83,225],[194,241],[252,193],[327,212],[414,197],[495,237],[591,213],
    [668,197],[732,218],[792,181],[872,203],[960,176],[960,301],[0,301]],"#273d525e");
  shape(ctx,[[0,271],[126,249],[266,270],[371,235],[455,238],[543,252],[640,238],[727,269],[843,230],[960,242],
    [960,310],[0,310]],"#2b3e4e");
  const waves=ctx.createLinearGradient(0,255,0,h);
  waves.addColorStop(0,water);waves.addColorStop(1,"#192d39");
  ctx.fillStyle=waves;ctx.fillRect(0,266,w,h-266);
  for(let i=0;i<245;i++){
    const y=268+rng()*188;
    const x=rng()*w;
    const near=Math.abs(x-moonX)<45+(y-260)*.5;
    ctx.fillStyle=near?sun+"64":i%2?"#8ea7aa53":"#182d406e";
    ctx.fillRect(x,y,3+rng()*(7+(y-259)*.18),i%5?1:2);
  }
  if (scene==="archive"){
    shape(ctx,[[0,66],[286,70],[372,261],[306,460],[0,460]],"#22313b");
    shape(ctx,[[960,54],[718,73],[650,236],[688,460],[960,460]],"#24343d");
    for(let i=0;i<6;i++){
      ctx.fillStyle="#3f4b4d";ctx.fillRect(20,108+i*44,236,3);
      for(let j=0;j<17;j++){
        ctx.fillStyle=["#6c6458","#83715d","#586b62","#935d57"][j%4];
        ctx.fillRect(24+j*13,89+i*44-rng()*8,6+rng()*4,18+rng()*4);
      }
    }
    shape(ctx,[[201,374],[482,322],[669,379],[452,441]],"#5a514b");
    shape(ctx,[[247,367],[423,333],[574,360],[421,401]],"#c7ab81");
    ctx.strokeStyle="#5e6d68";ctx.lineWidth=3;ctx.beginPath();
    ctx.moveTo(279,370);ctx.bezierCurveTo(363,392,432,340,530,369);ctx.stroke();
    for(let i=0;i<4;i++) ellipse(ctx,311+i*49,370+Math.sin(i*2)*10,4,3,i===3?"#764761":"#a8724c");
    lamp(ctx,619,337,1.3);
  } else if(scene==="tower"){
    tower(ctx,102,163,185,310,true);tower(ctx,320,222,54,103,false);
    const bell="#b78c71";shape(ctx,[[151,247],[151,216],[162,189],[219,189],[234,215],[233,247],[252,265],[136,265]],bell);
    ellipse(ctx,194,267,55,9,"#775f60");ellipse(ctx,194,271,15,7,"#d4a17f");
    ctx.strokeStyle="#db987e";ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(242,258);ctx.quadraticCurveTo(288,244,281,228);ctx.stroke();
  } else if(scene==="causeway" || scene==="arrive" || scene==="lost"){
    for(let i=0;i<11;i++){
      const y=h-12-i*15,x=475+i*22+Math.sin(i*.89)*32;
      rock(ctx,x,y,146-i*8,i<3);
    }
    tower(ctx,718,220,38,67,scene!=="lost");
    tower(ctx,778,203,43,82,scene==="arrive");
    roof(ctx,800,243,63,51,"#3b4b58");
    if(scene==="arrive") arch(ctx,762,391); else for(let i=0;i<3;i++) lamp(ctx,505+i*130,241+i%2*13,.6);
  } else {
    const pier=[[0,299],[267,305],[337,339],[309,390],[138,416],[39,452],[0,459]];
    shape(ctx,pier,"#243743");shape(ctx,[[0,302],[265,305],[338,339],[303,356],[197,348],[0,377]],"#56616b");
    for(let i=0;i<11;i++){
      const x=i*25,y=319+i*2;
      ctx.strokeStyle="#334956";ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-31,y+55);ctx.stroke();
    }
    for(let i=0;i<7;i++)roof(ctx,25+i*104,239-i%3*13,63+i%2*16,36+i%2*15,i%2?"#394b58":"#304451");
    tower(ctx,212,166,46,87,false);tower(ctx,741,192,64,103,true);
    lamp(ctx,275,301,1.1);lamp(ctx,101,307,1);
    if(scene==="ferry"||scene==="watch"){
      shape(ctx,[[407,348],[451,331],[586,347],[558,375],[459,369]],"#212c3b");
      shape(ctx,[[419,349],[451,334],[579,348],[554,355],[453,357]],"#56616b");
      ctx.strokeStyle="#997e77";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(500,350);ctx.lineTo(478,290);ctx.stroke();
      shape(ctx,[[479,287],[512,312],[491,321]],"#b7a68d");
    }
  }
  for(let i=0;i<85;i++){
    const x=rng()*w,y=rng()*h;
    ctx.fillStyle="#f3d3b019";ctx.fillRect(x,y,1,1);
  }
  const fog=ctx.createLinearGradient(0,0,0,h);
  fog.addColorStop(0,"transparent");fog.addColorStop(.79,"transparent");fog.addColorStop(1,"#0f1b2655");
  ctx.fillStyle=fog;ctx.fillRect(0,0,w,h);
  ctx.restore();
}
