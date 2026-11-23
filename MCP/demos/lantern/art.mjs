import { ellipse, glow, random, shape } from "../art-utils.mjs";
import { gate, height, walls, width } from "./game.mjs";

const backdrops = new WeakMap();

function backdrop(canvas) {
  if (backdrops.has(canvas)) return backdrops.get(canvas);
  const layer = document.createElement("canvas");
  layer.width = width; layer.height = height;
  const ctx = layer.getContext("2d");
  const rng = random(19041);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#23666a"); gradient.addColorStop(.57, "#133e46"); gradient.addColorStop(1, "#0d242d");
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  for (let i = 0; i < 760; i++) {
    const x = rng() * width; const y = rng() * height;
    ctx.fillStyle = rng() < .5 ? "#9dc5ab31" : "#0a1d3060";
    ctx.fillRect(x, y, 3 + rng() * 27, 1 + rng() * 2);
  }
  const island = [[49,154],[105,122],[146,128],[186,80],[331,56],[385,79],[501,54],[625,64],
    [689,81],[827,74],[885,133],[915,144],[925,354],[876,411],[841,459],[706,471],[636,506],
    [605,493],[538,528],[431,511],[365,527],[313,485],[236,503],[177,464],[108,468],[54,400]];
  shape(ctx, island.map(([x,y]) => [x,y+12]), "#091f29");
  shape(ctx, island.map(([x,y]) => [x,y+6]), "#55605a");
  shape(ctx, island, "#385253");
  ctx.save();
  ctx.beginPath(); island.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.closePath(); ctx.clip();
  for (let x = 52; x < 930; x += 46) {
    for (let y = 57; y < 540; y += 33) {
      const a = x + (rng()-.5) * 13; const b = y + (rng()-.5) * 11;
      shape(ctx, [[a-25,b-13],[a+15,b-16],[a+30,b+4],[a+21,b+17],[a-17,b+14],[a-32,b+5]],
        ["#3c5958","#456363","#314a4c","#395355","#537170"][Math.floor(rng()*5)]);
      if (rng() < .36) {
        ctx.strokeStyle="#21383d"; ctx.beginPath(); ctx.moveTo(a-13,b-10); ctx.lineTo(a,b); ctx.lineTo(a+8,b-2);
        ctx.lineTo(a+15,b+8); ctx.stroke();
      }
    }
  }
  for (let i=0;i<185;i++) {
    const x = 48+rng()*867, y = 65+rng()*465;
    ellipse(ctx,x,y,2+rng()*9,1+rng()*2,rng()<.4?"#91aa7550":"#162f385d");
  }
  ctx.restore();
  for (const w of walls) {
    ellipse(ctx,w.x+w.w/2,w.y+w.h+3,w.w/2+10,12,"#132c35a0");
    shape(ctx,[[w.x,w.y],[w.x+w.w,w.y],[w.x+w.w,w.y+w.h],[w.x,w.y+w.h]],"#26333d");
    shape(ctx,[[w.x,w.y],[w.x+7,w.y-12],[w.x+w.w+7,w.y-12],[w.x+w.w,w.y]],"#708b80");
    for(let x=w.x+5;x<w.x+w.w-4;x+=17) {
      ctx.fillStyle="#4c625e"; ctx.fillRect(x,w.y+5,12,w.h-12);
      ctx.fillStyle="#94a78e"; ctx.fillRect(x+2,w.y+5,6,2);
    }
  }
  for(let i=0;i<130;i++) {
    const x=26+rng()*900,y=rng()<.50?55+rng()*51:464+rng()*66, h=9+rng()*20;
    ctx.strokeStyle=["#bdad66","#496f60","#213941"][Math.floor(rng()*3)]; ctx.lineWidth=1.4;
    ctx.beginPath();ctx.moveTo(x,y);ctx.quadraticCurveTo(x-2,y-h*.65,x+(rng()-.5)*11,y-h);ctx.stroke();
  }
  backdrops.set(canvas, layer);
  return layer;
}

function arch(ctx, x, y, lit) {
  ellipse(ctx,x,y+18,30,11,"#14272f");
  if (lit) glow(ctx,x,y-15,86,"#f7cf6eab");
  ctx.lineWidth=12;ctx.strokeStyle=lit?"#edc474":"#78948b";
  ctx.beginPath();ctx.moveTo(x-24,y+16);ctx.lineTo(x-24,y-13);ctx.arc(x,y-13,24,Math.PI,0);
  ctx.lineTo(x+24,y+16);ctx.stroke();
  ctx.lineWidth=4;ctx.strokeStyle="#375456";ctx.strokeRect(x-30,y+12,60,8);
  for(let i=0;i<4;i++) ellipse(ctx,x-19+i*12,y-20,2,3,lit?"#fff2bd":"#385653");
}

function drawBearer(ctx,p,t) {
  const bob = Math.sin(t*12)*1.5;
  ellipse(ctx,p.x,p.y+12,15,6,"#0b1a229c");
  glow(ctx,p.x+11*p.dx,p.y-5,45,"#f4b856a1");
  shape(ctx,[[p.x-7,p.y-17+bob],[p.x+5,p.y-15+bob],[p.x+9,p.y+7],[p.x-12,p.y+9]],"#8ec8ad");
  shape(ctx,[[p.x-10,p.y+4],[p.x-13,p.y+11],[p.x+8,p.y+9],[p.x+6,p.y+0]],"#2b8882");
  ellipse(ctx,p.x,p.y-16+bob,7,8,"#b5d7b3");
  ctx.fillStyle="#233b3f";ctx.fillRect(p.x+3*p.dx-4,p.y-17+bob,8,5);
  ctx.fillStyle="#f1d38e";ctx.fillRect(p.x+14*p.dx-3,p.y-6,6,9);
  if (p.swing > 0) {
    const facing=Math.atan2(p.dy,p.dx);
    ctx.lineWidth=6;ctx.strokeStyle="#ffe7a9b8";ctx.beginPath();
    ctx.arc(p.x,p.y,39,facing-1.22,facing+1.22);ctx.stroke();
  }
  if (p.dash > 0) {
    ctx.strokeStyle="#b2e3c6b0";ctx.lineWidth=1.5;
    for(let i=0;i<3;i++) {ctx.beginPath();ctx.moveTo(p.x-5*p.dx,p.y-5+i*5);ctx.lineTo(p.x-43*p.dx,p.y-5+i*5-24*p.dy);ctx.stroke();}
  }
}

function drawFoe(ctx,foe,t) {
  const {x,y,kind,radius:r}=foe;
  const tone = kind==="guard"?"#c2817c":kind==="skulk"?"#d3a072":"#a98db9";
  ellipse(ctx,x,y+8,r+4,7,"#152229");
  if (foe.charge>0) {
    ctx.strokeStyle="#f3a776";ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,22+r,0,Math.PI*2);ctx.stroke();
  }
  const bob=Math.sin(t*6+foe.id)*3;
  shape(ctx,[[x-r,y+5],[x-r*.61,y-11+bob],[x-4,y-5+bob],[x+3,y-13+bob],[x+r,y+6],
    [x+5,y+10],[x-7,y+8]],foe.hurt?"#ffe1ad":tone);
  shape(ctx,[[x-6,y-5+bob],[x+7,y-5+bob],[x+5,y+4+bob],[x-4,y+5+bob]],"#3a343f");
  ctx.fillStyle="#fdd48c";ctx.fillRect(x-3,y-2+bob,2,2);ctx.fillRect(x+2,y-2+bob,2,2);
}

export function drawMarsh(ctx,state,t) {
  ctx.save();ctx.scale(ctx.canvas.width/width,ctx.canvas.height/height);
  ctx.drawImage(backdrop(ctx.canvas),0,0);
  arch(ctx,gate.x,gate.y,state.embers>=state.encounter.required);
  for (const drop of state.drops) {
    const bob=Math.sin(t*7+drop.x)*3;
    glow(ctx,drop.x,drop.y+bob,19,"#ffd171bd");
    shape(ctx,[[drop.x,drop.y-8+bob],[drop.x+5,drop.y+bob],[drop.x,drop.y+5+bob],[drop.x-5,drop.y+bob]],"#f6d789");
  }
  for(const foe of state.enemies) if(foe.hp) drawFoe(ctx,foe,t);
  drawBearer(ctx,state.player,t);
  if(state.result!=="running") {
    ctx.fillStyle="#10202aba";ctx.fillRect(0,0,width,height);
    ctx.textAlign="center";ctx.font="48px Georgia";ctx.fillStyle="#f7efdd";
    ctx.fillText(state.result==="won"?"Through the Tide Gate":"The Lantern Went Out",width/2,height/2);
    ctx.font="18px Segoe UI";ctx.fillStyle="#bad3c3";ctx.fillText("Restart run to try again",width/2,height/2+38);
  }
  ctx.restore();
}
