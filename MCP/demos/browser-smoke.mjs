import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const [chrome, target] = process.argv.slice(2);
if (!chrome || !target) throw new Error("Usage: node browser-smoke.mjs /path/to/chrome /capture-directory");
const out=resolve(target);
await mkdir(out,{recursive:true});
const root=fileURLToPath(new URL(".",import.meta.url));
const types={".html":"text/html; charset=utf-8",".css":"text/css; charset=utf-8",".mjs":"text/javascript; charset=utf-8",".json":"application/json"};
const server=createServer(async (req,res)=>{
  try{
    const path=decodeURIComponent(new URL(req.url,"http://127.0.0.1").pathname);
    const file=resolve(root,"."+path+(path.endsWith("/")?"index.html":""));
    if(!file.startsWith(root.endsWith(sep)?root:root+sep)) {res.writeHead(403).end();return;}
    const body=await readFile(file);
    res.writeHead(200,{"Content-Type":types[extname(file)]??"application/octet-stream"}).end(body);
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
const app=`http://127.0.0.1:${server.address().port}`;
const profile=await mkdtemp(join(tmpdir(),"pocket-goblin-smoke-"));
const browser=spawn(chrome,["--headless=new","--disable-gpu","--no-first-run","--disable-background-networking",
  "--remote-debugging-port=0",`--user-data-dir=${profile}`,"about:blank"],{stdio:"ignore",windowsHide:true});

async function poll(read,message){
  for(let i=0;i<100;i++){
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
}finally{
  socket?.close();browser.kill();server.close();
}
