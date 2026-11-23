import { createGame, openGround } from "./lantern/game.mjs";
import { begin, choose, scenes } from "./tide/world.mjs";

const profile = (checks, detail = {}) => ({
  checks, passed: Object.values(checks).filter(Boolean).length, assigned: Object.keys(checks).length,
  verified: Object.values(checks).every(Boolean), ...detail,
});
const keys = (value, shape) => value && typeof value === "object" && !Array.isArray(value)
  && Object.keys(value).sort().join(",") === [...shape].sort().join(",");
const text = (value, min, max) => typeof value === "string" && value.trim() === value
  && value.length >= min && value.length <= max && !/[<>\x00-\x1f]/u.test(value);
const between = (n, low, high) => Number.isInteger(n) && low <= n && n <= high;
const zone = (foe, kind, x0, x1, y0, y1) => foe?.kind === kind
  && between(foe.x, x0, x1) && between(foe.y, y0, y1);

export function assessPatrol(candidate) {
  const foeList = Array.isArray(candidate?.foes) ? candidate.foes : [];
  const wellFormed = keys(candidate, ["name","required","vigor","foes"]) && foeList.length === 4
    && foeList.every(foe => keys(foe, ["kind","x","y"]) && ["guard","skulk","wisp"].includes(foe.kind)
      && Number.isInteger(foe.x) && Number.isInteger(foe.y));
  const checks = {
    schema: !!wellFormed,
    name: text(candidate?.name, 4, 42),
    pacing: candidate?.required === 3 && candidate?.vigor === 6,
    nearWisp: foeList.some(foe => zone(foe, "wisp", 180, 250, 215, 315)),
    gardenSkulk: foeList.some(foe => zone(foe, "skulk", 500, 591, 155, 245)),
    southWisp: foeList.some(foe => zone(foe, "wisp", 493, 609, 359, 447)),
    gateGuard: foeList.some(foe => zone(foe, "guard", 751, 815, 230, 336)),
    safeSpawns: !!wellFormed && createGame(candidate).enemies.every(foe => openGround(foe.x, foe.y, foe.radius))
      && foeList.every((foe, index) => foeList.slice(index + 1).every(other => Math.hypot(foe.x - other.x, foe.y - other.y) >= 80)),
  };
  return profile(checks);
}

const passageKeys=["entryLabel","entryRequires","interior","ending","returnLabel","forwardLabel"];
function validScene(scene) {
  return keys(scene, ["title","text","palette"]) && text(scene.title, 4, 52) && text(scene.text, 45, 280);
}

export function assessPassage(candidate) {
  const checks={
    schema: !!keys(candidate,passageKeys) && validScene(candidate?.interior) && validScene(candidate?.ending),
    entryLabel: text(candidate?.entryLabel, 6, 65),
    clues: JSON.stringify(candidate?.entryRequires) === '["lens","bell"]',
    sceneArt: candidate?.interior?.palette === "archive" && candidate?.ending?.palette === "arrive",
    proseClues: /\blens\b/i.test(candidate?.interior?.text ?? "") && /\bbell\b/i.test(candidate?.interior?.text ?? "")
      && /\bdawn\b/i.test(candidate?.ending?.text ?? ""),
    choices: text(candidate?.returnLabel, 6, 65) && text(candidate?.forwardLabel, 6, 65)
      && /\bcauseway\b/i.test(candidate.returnLabel) && /\bshutter\b/i.test(candidate.forwardLabel),
  };
  if(Object.values(checks).every(Boolean)) {
    const world=buildPassage(candidate);
    const quay=begin();
    const bare=choose(world,quay,"causeway");
    const enter=state=>choose(world,state,"drowned-observatory");
    const both={...bare,held:["lens","bell"]};
    const room=enter(both);
    checks.runtime=enter(bare)===bare && enter({...bare,held:["seal"]}).node==="causeway"
      && room.node==="submerged" && choose(world,room,"return").node==="causeway"
      && choose(world,room,"shutter").node==="observatory-ending"
      && world["observatory-ending"].choices.length===0 && world.causeway.choices.length===scenes.causeway.choices.length+1;
  } else checks.runtime=false;
  return profile(checks);
}

function buildPassage(passage) {
  return {
    ...scenes,
    causeway: {...scenes.causeway, choices:[
      ...scenes.causeway.choices.slice(0,-1),
      {id:"drowned-observatory", label:passage.entryLabel, to:"submerged",
        requires:passage.entryRequires, lock:"Find the chart and the bell beat"},
      scenes.causeway.choices.at(-1),
    ]},
    submerged: {place:"The Drowned Observatory", title:passage.interior.title,
      text:passage.interior.text, art:passage.interior.palette, choices:[
        {id:"return", label:passage.returnLabel, to:"causeway"},
        {id:"shutter", label:passage.forwardLabel, to:"observatory-ending"},
      ]},
    "observatory-ending": {place:"The Observatory · Dawn", title:passage.ending.title,
      text:passage.ending.text, art:passage.ending.palette, choices:[]},
  };
}

export function worldWithPassage(candidate) {
  if(!assessPassage(candidate).verified) throw new Error("The story variant failed its contract");
  return buildPassage(candidate);
}

export async function loadVariant(project, assess) {
  const arm=new URLSearchParams(location.search).get("variant");
  if(!["assisted","wide"].includes(arm)) return null;
  try {
    const response=await fetch(new URL(`./variants/${project}-${arm}.json`,import.meta.url));
    if(!response.ok) return null;
    const data=await response.json();
    return assess(data).verified?data:null;
  } catch {
    return null;
  }
}
