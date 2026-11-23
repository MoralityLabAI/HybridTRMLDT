import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createGame, gate, openGround, step, width } from "./lantern/game.mjs";

const cell = 16;
const columns = Math.ceil(width / cell) + 1;
const surrounding = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const cellAt = (x, y) => y * columns + x;
const pointAt = (x, y) => ({x:x * cell, y:y * cell});

function waypoint(start, target) {
  const x = Math.round(start.x / cell);
  const y = Math.round(start.y / cell);
  const origin = cellAt(x, y);
  const goal = cellAt(Math.round(target.x / cell), Math.round(target.y / cell));
  const queue = [[x, y]];
  const parents = new Map([[origin, null]]);
  let best = origin;
  let nearest = distance(pointAt(x, y), target);
  for (let index = 0; index < queue.length; index++) {
    const [col, row] = queue[index];
    const current = cellAt(col, row);
    const gap = distance(pointAt(col, row), target);
    if (gap < nearest) [best, nearest] = [current, gap];
    if (current === goal) break;
    for (const [dx, dy] of surrounding) {
      const nextX = col + dx;
      const nextY = row + dy;
      const id = cellAt(nextX, nextY);
      if (parents.has(id) || !openGround(nextX * cell, nextY * cell, 13)) continue;
      parents.set(id, current);
      queue.push([nextX, nextY]);
    }
  }
  if (best === origin) return target;
  while (parents.get(best) !== origin) best = parents.get(best);
  return pointAt(best % columns, Math.floor(best / columns));
}

function play(encounter) {
  const game = createGame(encounter);
  for (let frame = 0; frame < 60 * 75 && game.result === "running"; frame++) {
    const player = game.player;
    const nearby = (a, b) => distance(a, player) - distance(b, player);
    const foe = game.enemies.filter(enemy => enemy.hp).sort(nearby)[0];
    const drop = [...game.drops].sort(nearby)[0];
    const target = game.embers >= game.encounter.required ? gate : drop ?? foe;
    assert.ok(target, "the encounter has a reachable objective");
    const attack = target === foe && distance(target, player) < 67;
    const heading = attack || distance(target, player) < 31 ? target : waypoint(player, target);
    const retreat = attack && player.nextSwing > .06 ? -1 : 1;
    const dx = (heading.x - player.x) * retreat;
    const dy = (heading.y - player.y) * retreat;
    step(game, {right:dx > 4, left:dx < -4, down:dy > 4, up:dy < -4, attack, open:target === gate,
      dash:target === gate && distance(target, player) > 100 && distance(player, heading) > 10}, 1 / 60);
  }
  assert.equal(game.result, "won");
  return {name: game.encounter.name, result: game.result, embers: game.embers, vigor: game.player.hp, seconds: game.t};
}

const variants = await readFile(new URL("./variants/lantern-assisted.json", import.meta.url), "utf8");
console.log(JSON.stringify({method:"public game controls only", playthroughs:[play(), play(JSON.parse(variants))]}));
