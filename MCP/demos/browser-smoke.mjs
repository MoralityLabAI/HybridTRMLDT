import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const [chrome, target, mode, gameDirectory, goalX, goalY] = process.argv.slice(2);
if (!chrome || !target || (mode && (!['rpg','story','storyworld'].includes(mode) || !gameDirectory))) {
  throw new Error("Usage: node browser-smoke.mjs CHROME CAPTURE_DIRECTORY [rpg|story GAME_DIRECTORY [GOAL_X GOAL_Y] | storyworld GAME_DIRECTORY WORLD_DATA BALANCE_JSON]");
}
const storyworld=mode==="storyworld";
const out=resolve(target);
await mkdir(out,{recursive:true});
const root=mode?resolve(gameDirectory):fileURLToPath(new URL(".",import.meta.url));
const types={".html":"text/html; charset=utf-8",".css":"text/css; charset=utf-8",".mjs":"text/javascript; charset=utf-8",".js":"text/javascript; charset=utf-8",".json":"application/json"};
const server=createServer(async (req,res)=>{
  try{
    const path=decodeURIComponent(new URL(req.url,"http://127.0.0.1").pathname);
    if(mode && path!=="/" && !(storyworld && ["/art.js","/world-data.js"].includes(path))) {res.writeHead(404).end();return;}
    const file=resolve(root,"."+path+(path.endsWith("/")?"index.html":""));
    if(!file.startsWith(root.endsWith(sep)?root:root+sep)) {res.writeHead(403).end();return;}
    const body=await readFile(storyworld && path==="/world-data.js" ? goalX : file);
    const headers={"Content-Type":types[extname(file)]??"application/octet-stream"};
    if(mode) headers["Content-Security-Policy"]="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'";
    if(storyworld) headers["Content-Security-Policy"]="default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'";
    res.writeHead(200,headers).end(body);
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
const app=`http://127.0.0.1:${server.address().port}`;
const profile=await mkdtemp(join(tmpdir(),"pocket-goblin-smoke-"));
const browser=spawn(chrome,["--headless=new","--disable-gpu","--no-first-run","--disable-background-networking",
  "--remote-debugging-port=0",`--user-data-dir=${profile}`,"about:blank"],{stdio:"ignore",windowsHide:true});

async function poll(read,message,attempts=100){
  for(let i=0;i<attempts;i++){
    try{const value=await read();if(value)return value;}catch{}
    await sleep(100);
  }
  throw new Error(message);
}
let socket;
try{
  const port=await poll(async()=>(await readFile(join(profile,"DevToolsActivePort"),"utf8")).split("\n")[0],
    "Chrome remote-debugging port did not appear");
  const tabs=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket=new WebSocket(tabs.find(tab=>tab.type==="page").webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{
    socket.addEventListener("open",resolve,{once:true});socket.addEventListener("error",reject,{once:true});
  });
  let next=0;
  const replies=new Map();
  const failures=[];
  socket.addEventListener("message",event=>{
    const message=JSON.parse(event.data);
    if(message.method==="Runtime.exceptionThrown") failures.push(message.params.exceptionDetails.text);
    const pair=replies.get(message.id);
    if(!pair)return;
    replies.delete(message.id);
    if(message.error) pair.reject(new Error(message.error.message));else pair.resolve(message.result);
  });
  const send=(method,params={})=>new Promise((resolve,reject)=>{
    const id=++next;replies.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));
  });
  const evaluate=async expression=>{
    const value=await send("Runtime.evaluate",{expression,returnByValue:true});
    if(value.exceptionDetails)throw new Error(value.exceptionDetails.text+" "+expression);
    return value.result.value;
  };
  const visit=async(path,ready)=>{
    await send("Page.navigate",{url:app+path});
    await poll(()=>evaluate(ready),`Page ${path} failed to initialize`);
  };
  const screenshot=async file=>{
    const shot=await send("Page.captureScreenshot",{format:"png"});
    await writeFile(join(out,file),Buffer.from(shot.data,"base64"));
  };
  const press=(type,key,code)=>send("Input.dispatchKeyEvent",{type,key,code});
  const click=async id=>assert.equal(await evaluate(
    `(() => { const b=document.querySelector('[data-choice="${id}"]'); if(!b || b.disabled) return false; b.click(); return true; })()`),true);
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride",{width:1280,height:950,deviceScaleFactor:1,mobile:false});
  if(mode){
    await visit("/","typeof window.gameState === 'function' && !!document.querySelector('canvas')");
    const state=()=>evaluate("window.gameState()");
    const art=()=>evaluate("document.querySelector('canvas').toDataURL()");
    const initial=await state();
    const reset=()=>evaluate("document.querySelector('#restart').click()");
    const checks=[];
    await screenshot("start.png");
    if(mode==="rpg"){
      assert.equal(initial.status,"playing");
      assert.ok(Number.isFinite(initial.x) && Number.isFinite(initial.y) && initial.hp>0);
      const hold=async(keys,action)=>{
        for(const [key,code] of keys) await press("keyDown",key,code);
        try { return await action(); }
        finally { for(const [key,code] of keys) await press("keyUp",key,code); }
      };
      const sword=[" ","Space"];
      await reset();
      await hold([["d","KeyD"],sword],()=>poll(async()=> (await state()).hits>initial.hits,
        "Moving toward the first foe and swinging should cause damage"));
      const moved=await state();
      assert.ok(Math.hypot(moved.x-initial.x,moved.y-initial.y)>5,"movement should change player position");
      checks.push("keyboard movement and combat damaged an enemy");
      await screenshot("gameplay.png");
      if(goalX && goalY){
        const navigate=async(axis,target,negative,positive)=>{
          const current=(await state())[axis];
          if(Math.abs(current-target)<14)return;
          const direction=current>target?-1:1;
          await hold([direction<0?negative:positive,sword],()=>poll(async()=>{
            const now=await state();
            return now.status==="won" || direction*(target-now[axis])<14;
          },`Could not reach ${axis}=${target}`));
        };
        await navigate("y",Number(goalY),["w","KeyW"],["s","KeyS"]);
        await navigate("x",Number(goalX),["a","KeyA"],["d","KeyD"]);
        assert.equal((await state()).status,"won");
        await screenshot("victory.png");
        await reset();
        await poll(async()=> (await state()).status==="lost","Idle enemies never caused a loss",210);
        await screenshot("defeat.png");
        checks.push("public controls reached victory and enemy attacks caused defeat");
      }
      await reset();
      assert.equal((await state()).hits,initial.hits);
      assert.equal((await state()).hp,initial.hp);
      checks.push("restart restored initial combat state");
    }else if(storyworld){
      const balance=JSON.parse(await readFile(goalY,"utf8"));
      const snapshots=new Set();
      for(const [destination,path] of Object.entries(balance.scene_witnesses)){
        await reset();
        for(const step of path){
          assert.equal((await state()).scene,step.scene);
          assert.equal(await evaluate("document.querySelectorAll('button[data-option]:not(:disabled)').length"),3);
          await evaluate(`document.querySelector('button[data-option="${step.option}"]').click()`);
          const changed=await state();
          assert.equal(changed.reaction,step.reaction);
          for (const [stat,value] of Object.entries(step.after)) {
            assert.ok(Math.abs(changed.stats[stat]-value)<1e-9,`${stat} differs from the reference rule`);
          }
          await evaluate("document.querySelector('#continue').click()");
        }
        const arrived=await state();
        assert.equal(arrived.scene,destination);
        snapshots.add(await art());
        if(destination in balance.endings){
          assert.equal(arrived.ending,destination);
          assert.equal(await evaluate("document.querySelectorAll('button[data-option]:not(:disabled)').length"),0);
          await screenshot(`${destination}.png`);
        }
      }
      assert.equal(snapshots.size,20,"every reachable encounter should render distinct Canvas pixels");
      await reset();
      assert.deepEqual(await state(),initial);
      checks.push("20 distinct Canvas scenes through player controls", "response formulas and bounded effects match independent replay", "all endings and restart work");
    }else{
      assert.equal(initial.scene,"start");
      assert.ok(!initial.clue);
      assert.equal(await evaluate(`(() => { const b=document.querySelector('[data-choice="cross"]'); return !!b && (b.disabled || b.getAttribute('aria-disabled')==='true'); })()`),true);
      checks.push("crossing is gated before study");
      await click("rush");
      assert.equal((await state()).ending,"bad");
      await sleep(120);
      const badArt=await art();
      await screenshot("bad-ending.png");
      await reset();
      assert.equal((await state()).scene,"start");
      assert.ok(!(await state()).clue);
      await click("study");
      assert.ok((await state()).clue);
      assert.notEqual((await state()).scene,"start");
      await click("cross");
      assert.equal((await state()).ending,"good");
      await sleep(120);
      assert.notEqual(await art(),badArt,"ending choices should change the Canvas scene");
      await screenshot("good-ending.png");
      checks.push("public controls reached both endings with distinct Canvas art and reset the clue");
    }
    assert.deepEqual(failures,[]);
    const receipt={status:"verified",game:mode,title:await evaluate("document.title"),checks};
    await writeFile(join(out,"browser-receipt.json"),JSON.stringify(receipt,null,2)+"\n");
    console.log(JSON.stringify(receipt));
  }else{
  await visit("/",`document.querySelector('h1')?.textContent.includes('field tests') && document.querySelectorAll('.project').length===2`);
  await screenshot("field-tests.png");
  await visit("/lantern/",`document.querySelector('#hp')?.textContent==='7'`);
  const initial=await evaluate(`document.getElementById('arena').getContext('2d').getImageData(92,253,120,55).data.join(',')`);
  await press("keyDown","d","KeyD");await sleep(460);await press("keyUp","d","KeyD");
  const moved=await evaluate(`document.getElementById('arena').getContext('2d').getImageData(92,253,120,55).data.join(',')`);
  assert.notEqual(initial,moved);
  await press("keyDown"," ","Space");await sleep(400);await press("keyUp"," ","Space");
  await screenshot("lantern-below.png");
  await visit("/tide/",`document.querySelector('#scene-title')?.textContent==='A harbor holding its breath'`);
  await screenshot("hour-between-tides.png");
  await click("causeway");
  assert.equal(await evaluate(`document.querySelector('[data-choice="true-crossing"]').disabled`),true);
  await click("back");await click("archive");await click("study");await click("tower");await click("listen");
  await click("causeway");await click("true-crossing");
  assert.equal(await evaluate("document.getElementById('scene-title').textContent"),"The keeper’s window");
  await screenshot("tide-ending.png");
  await send("Emulation.setDeviceMetricsOverride",{width:412,height:825,deviceScaleFactor:1,mobile:true,screenOrientation:{angle:0,type:"portraitPrimary"}});
  await visit("/tide/",`document.querySelector('#scene-title')?.textContent==='A harbor holding its breath'`);
  assert.equal(await evaluate("document.documentElement.scrollWidth <= innerWidth"),true);
  await screenshot("tide-mobile.png");
  await send("Emulation.setDeviceMetricsOverride",{width:1280,height:950,deviceScaleFactor:1,mobile:false});
  const variants=[];
  for(const arm of ["assisted","wide"]){
    for(const project of ["lantern","tide"]){
      let data;
      try{data=JSON.parse(await readFile(join(root,"variants",`${project}-${arm}.json`),"utf8"));}
      catch(error){if(error.code==="ENOENT")continue;throw error;}
      if(project==="lantern"){
        await visit(`/lantern/?variant=${arm}`,`document.querySelector('.side .kicker')?.textContent===${JSON.stringify(data.name)}`);
        assert.equal(await evaluate("document.querySelector('#hp').textContent"),"6");
        assert.equal(await evaluate("document.querySelector('#threat').textContent"),"4 sentries");
      }else{
        await visit(`/tide/?variant=${arm}`,`document.querySelector('#scene-title')?.textContent==='A harbor holding its breath'`);
        await click("causeway");
        assert.equal(await evaluate(`document.querySelector('[data-choice=\"drowned-observatory\"]')?.disabled`),true);
        await click("back");await click("archive");await click("study");await click("tower");await click("listen");
        await click("causeway");await click("drowned-observatory");await click("shutter");
        assert.equal(await evaluate("document.getElementById('scene-title').textContent"),data.ending.title);
      }
      variants.push(`${project}-${arm}`);
      await screenshot(`${project}-${arm}.png`);
    }
  }
  assert.deepEqual(failures,[]);
  console.log(JSON.stringify({status:"verified",checks:["project navigation","canvas input","locked story route","evidence-based ending","mobile viewport"],
    captures:out,variants}));
  }
}finally{
  socket?.close();browser.kill();server.close();
}
