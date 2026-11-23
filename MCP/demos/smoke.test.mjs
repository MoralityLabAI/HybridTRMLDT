import assert from "node:assert/strict";
import { test } from "node:test";
import { createGame, gate, openGround, step } from "./lantern/game.mjs";
import { begin, choose, options, scenes } from "./tide/world.mjs";
import { assessPassage, assessPatrol, worldWithPassage } from "./variants.mjs";

test("the marsh encounter starts on walkable ground and the gate requires embers", () => {
  const game = createGame();
  assert.ok(game.enemies.every(foe => openGround(foe.x, foe.y, foe.radius)));
  Object.assign(game.player, { x: gate.x, y: gate.y });
  step(game, { open: true }, .01);
  assert.equal(game.result, "running");
  assert.match(game.notice, /dark/);
  game.embers = game.encounter.required;
  step(game, { open: true }, .01);
  assert.equal(game.result, "won");
});

test("a forward strike knocks out a wisp and its ember can be collected", () => {
  const game = createGame();
  const wisp = game.enemies[0];
  Object.assign(wisp, { x: game.player.x + 41, y: game.player.y });
  step(game, { attack: true }, .01);
  assert.equal(wisp.hp, 0);
  assert.equal(game.drops.length, 1);
  for (let frame = 0; frame < 10; frame += 1) step(game, { right: true }, .016);
  assert.equal(game.embers, 1);
  const before = game.player.x;
  step(game, { dash: true, right: true }, .05);
  assert.ok(game.player.x - before > 20);
});

test("the crossing requires independent lens and bell clues", () => {
  const quay = begin();
  const bare = choose(scenes, quay, "causeway");
  assert.equal(choose(scenes, bare, "true-crossing"), bare);
  let state = choose(scenes, choose(scenes, quay, "archive"), "study");
  assert.deepEqual(state.held, ["lens"]);
  state = choose(scenes, choose(scenes, state, "tower"), "listen");
  state = choose(scenes, state, "causeway");
  assert.ok(options(scenes, state).find(choice => choice.id === "true-crossing").available);
  assert.equal(choose(scenes, state, "true-crossing").node, "arrive");
});

test("the seal route and unverified reflection are separate endings", () => {
  let state = choose(scenes, begin(), "ferry");
  state = choose(scenes, state, "borrow");
  state = choose(scenes, state, "causeway");
  assert.equal(choose(scenes, state, "sealed-crossing").node, "watch");
  assert.equal(choose(scenes, state, "risk").node, "lost");
  for (const scene of Object.values(scenes)) {
    assert.ok(scene.choices.every(choice => choice.to in scenes));
  }
});

test("a verified alternate patrol can be played and blocked spawns are rejected", () => {
  const patrol={name:"The Salt Patrol", required:3, vigor:6, foes:[
    {kind:"wisp",x:210,y:250}, {kind:"skulk",x:529,y:200},
    {kind:"wisp",x:549,y:401}, {kind:"guard",x:782,y:280},
  ]};
  assert.equal(assessPatrol(patrol).verified,true);
  assert.equal(createGame(patrol).player.hp,6);
  patrol.foes[0].y=175;
  assert.equal(assessPatrol(patrol).verified,false);
});

test("a verified alternate passage leaves the base world unchanged", () => {
  const passage={entryLabel:"Enter the drowned observatory", entryRequires:["lens","bell"],
    interior:{title:"The room below",text:"The bell goes quiet as the lens divides the amber light across the room.",palette:"archive"},
    ending:{title:"The narrow window",text:"Dawn rises across the chart before the keeper takes the lantern away.",palette:"arrive"},
    returnLabel:"Return to the causeway",forwardLabel:"Open the shutter"};
  assert.equal(assessPassage(passage).verified,true);
  assert.ok(worldWithPassage(passage)["observatory-ending"]);
  assert.equal(scenes["observatory-ending"],undefined);
  assert.equal(assessPassage({...passage,entryRequires:["seal"]}).verified,false);
});
