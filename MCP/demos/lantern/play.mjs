import { createGame, step } from "./game.mjs";
import { drawMarsh } from "./art.mjs";
import { assessPatrol, loadVariant } from "../variants.mjs";

const keys = new Set();
const bound = new Set(["w","a","s","d","arrowup","arrowleft","arrowdown","arrowright","shift"," ","e"]);
const canvas = document.querySelector("#arena");
const ctx = canvas.getContext("2d");
const hp = document.querySelector("#hp");
const bar = document.querySelector("#hp-bar");
const embers = document.querySelector("#embers");
const threats = document.querySelector("#threat");
const objective = document.querySelector("#objective");
const encounter = await loadVariant("lantern", assessPatrol);
let state = createGame(encounter ?? undefined);
if (encounter) document.querySelector(".side .kicker").textContent = encounter.name;
let last = 0;

window.addEventListener("keydown", e => {
  const key=e.key.toLowerCase();
  if (!bound.has(key) || e.target.closest?.("button, a, input")) return;
  e.preventDefault(); keys.add(key);
});
window.addEventListener("keyup",e => keys.delete(e.key.toLowerCase()));
window.addEventListener("blur",() => keys.clear());

for(const button of document.querySelectorAll("[data-key]")) {
  button.addEventListener("pointerdown",event => {
    event.preventDefault();
    button.setPointerCapture(event.pointerId);
    keys.add(button.dataset.key);
  });
  for(const event of ["pointerup","pointercancel","lostpointercapture"]) {
    button.addEventListener(event,() => keys.delete(button.dataset.key));
  }
}
document.querySelector("#restart").addEventListener("click",event => {
  state=createGame(encounter ?? undefined);keys.clear();event.currentTarget.blur();
});

function setText(node,text) { if(node.textContent!==text) node.textContent=text; }
function tick(ms) {
  const dt=last?(ms-last)/1000:0;
  last=ms;
  step(state,{
    up:keys.has("w")||keys.has("arrowup"), down:keys.has("s")||keys.has("arrowdown"),
    left:keys.has("a")||keys.has("arrowleft"), right:keys.has("d")||keys.has("arrowright"),
    attack:keys.has(" "), dash:keys.has("shift"), open:keys.has("e"),
  },dt);
  drawMarsh(ctx,state,state.t);
  setText(hp,String(state.player.hp));
  bar.style.width=`${100*state.player.hp/state.encounter.vigor}%`;
  setText(embers,`Embers ${state.embers}/${state.encounter.required}`);
  setText(threats,`${state.enemies.filter(foe=>foe.hp).length} sentries`);
  setText(objective,state.notice);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
