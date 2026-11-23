import { drawScene } from "./art.mjs";
import { begin, choose, facts, options, scenes } from "./world.mjs";
import { assessPassage, loadVariant, worldWithPassage } from "../variants.mjs";

const byId = id => document.getElementById(id);
const canvas=byId("story-art");
const ctx=canvas.getContext("2d");
const passage=await loadVariant("tide",assessPassage);
const world=passage?worldWithPassage(passage):scenes;
let state=begin();

function render(focus=false) {
  const scene=world[state.node];
  drawScene(ctx,scene.art);
  canvas.setAttribute("aria-label",`Illustration of ${scene.title.toLowerCase()}`);
  byId("scene-place").textContent=scene.place;
  byId("scene-title").textContent=scene.title;
  byId("scene-text").textContent=scene.text;
  const buttons=options(world,state).map(choice=>{
    const button=document.createElement("button");
    button.className="choice";
    button.disabled=!choice.available;
    button.dataset.choice=choice.id;
    const label=document.createElement("span");
    label.textContent=choice.label;
    const detail=document.createElement("small");
    detail.textContent=choice.available?"→":choice.lock;
    if(choice.available) detail.setAttribute("aria-hidden","true");
    button.append(label,detail);
    button.addEventListener("click",()=>{
      state=choose(world,state,choice.id);
      render(true);
    });
    return button;
  });
  if(!buttons.length) {
    const again=document.createElement("button");
    again.className="choice";again.textContent="Return to Low Quay";
    again.addEventListener("click",reset);buttons.push(again);
  }
  byId("choices").replaceChildren(...buttons);
  byId("inventory").replaceChildren(...Object.entries(facts).map(([id,name])=>{
    const item=document.createElement("li");
    const known=state.held.includes(id);
    item.classList.toggle("known",known);
    item.textContent=known?name:`Unknown · ${name}`;
    return item;
  }));
  byId("trail").replaceChildren(...state.trail.slice(-7).map(node=>{
    const item=document.createElement("li");item.textContent=world[node].title;return item;
  }));
  if(focus) byId("scene-title").focus();
}

function reset() { state=begin();render(true); }
byId("restart").addEventListener("click",reset);
render();
