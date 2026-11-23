export const width = 960;
export const height = 580;
export const gate = { x: 856, y: 275, radius: 43 };
export const walls = [
  {x: 204, y: 150, w: 118, h: 38}, {x: 253, y: 375, w: 120, h: 38},
  {x: 426, y: 209, w: 48, h: 132}, {x: 622, y: 116, w: 109, h: 38},
  {x: 633, y: 389, w: 106, h: 37},
];

const variants = {
  wisp: { hp: 1, speed: 113, radius: 10, damage: 1, windup: .34 },
  guard: { hp: 3, speed: 75, radius: 15, damage: 2, windup: .58 },
  skulk: { hp: 2, speed: 143, radius: 12, damage: 1, windup: .40 },
};
export const baseEncounter = {
  name: "The Saltfen", required: 3, vigor: 7,
  foes: [
    {kind: "wisp", x: 311, y: 255}, {kind: "guard", x: 582, y: 285},
    {kind: "skulk", x: 510, y: 407}, {kind: "wisp", x: 716, y: 214},
    {kind: "guard", x: 770, y: 361},
  ],
};

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const overlapsWall = (x, y, r, wall) => {
  const nearX = Math.max(wall.x, Math.min(x, wall.x + wall.w));
  const nearY = Math.max(wall.y, Math.min(y, wall.y + wall.h));
  return Math.hypot(x - nearX, y - nearY) < r;
};
export function openGround(x, y, r = 12) {
  return x - r > 65 && x + r < 905 && y - r > 74 && y + r < 513
    && !walls.some(wall => overlapsWall(x, y, r, wall));
}

export function createGame(encounter = baseEncounter) {
  return {
    encounter, result: "running", t: 0, embers: 0, drops: [],
    notice: "Recover embers from the sentries.",
    player: { x: 117, y: 280, radius: 12, dx: 1, dy: 0, hp: encounter.vigor, dash: 0, dashCooldown: 0,
      guard: 0, swing: 0, nextSwing: 0 },
    enemies: encounter.foes.map((foe, id) => ({ ...variants[foe.kind], ...foe, id, charge: 0, recovery: .2, hurt: 0 })),
  };
}

function march(actor, x, y) {
  if (openGround(actor.x + x, actor.y, actor.radius)) actor.x += x;
  if (openGround(actor.x, actor.y + y, actor.radius)) actor.y += y;
}

function strike(state) {
  const p = state.player;
  p.swing = .23;
  p.nextSwing = .40;
  for (const foe of state.enemies) {
    const dx = foe.x - p.x;
    const dy = foe.y - p.y;
    const range = distance(foe, p);
    if (!foe.hp || range > 60 + foe.radius || (range > 20 && dx * p.dx + dy * p.dy < -5)) continue;
    foe.hp -= 1;
    foe.hurt = .27;
    foe.charge = 0;
    foe.recovery = .42;
    if (foe.hp) march(foe, p.dx * 17, p.dy * 17);
    else state.drops.push({x: foe.x, y: foe.y});
  }
}

function moveEnemy(state, foe, dt) {
  const p = state.player;
  foe.hurt = Math.max(0, foe.hurt - dt);
  foe.recovery = Math.max(0, foe.recovery - dt);
  const range = distance(p, foe);
  if (foe.charge > 0) {
    foe.charge -= dt;
    if (foe.charge <= 0) {
      foe.recovery = .84;
      if (range < 38 + foe.radius && p.guard <= 0) {
        p.hp = Math.max(0, p.hp - foe.damage);
        p.guard = .65;
        state.notice = p.hp ? "You were hit. Dash through the wind-up." : "The marsh swallowed the lantern.";
      }
    }
    return;
  }
  if (range < 30 + foe.radius && foe.recovery <= 0) {
    foe.charge = foe.windup;
    return;
  }
  if (range > 330 || range < 23 + foe.radius || foe.hurt > 0) return;
  const travel = foe.speed * dt;
  const dx = (p.x - foe.x) / range;
  const dy = (p.y - foe.y) / range;
  const x = foe.x;
  const y = foe.y;
  march(foe, dx * travel, dy * travel);
  if (foe.x === x && foe.y === y) {
    const tangent = foe.id % 2 ? -1 : 1;
    march(foe, -dy * travel * tangent, dx * travel * tangent);
  }
}

export function step(state, input, dt) {
  if (state.result !== "running") return state;
  dt = Math.max(0, Math.min(dt, .05));
  state.t += dt;
  const p = state.player;
  for (const timer of ["dash", "dashCooldown", "guard", "swing", "nextSwing"]) {
    p[timer] = Math.max(0, p[timer] - dt);
  }
  const dx = (input.right ? 1 : 0) - (input.left ? 1 : 0);
  const dy = (input.down ? 1 : 0) - (input.up ? 1 : 0);
  const len = Math.hypot(dx, dy);
  if (len && !p.dash) [p.dx, p.dy] = [dx / len, dy / len];
  if (input.dash && !p.dashCooldown) {
    p.dash = .18;
    p.dashCooldown = .76;
    p.guard = Math.max(p.guard, .24);
  }
  if (p.dash || len) {
    const speed = p.dash ? 500 : 205;
    march(p, p.dx * speed * dt, p.dy * speed * dt);
  }
  if (input.attack && !p.nextSwing) strike(state);
  for (const foe of state.enemies) if (foe.hp) moveEnemy(state, foe, dt);
  state.drops = state.drops.filter(drop => {
    if (distance(drop, p) > 24) return true;
    state.embers += 1;
    if (state.embers === state.encounter.required) {
      p.hp = Math.min(state.encounter.vigor, p.hp + 2);
      state.notice = "Gate lit. Head east and open the arch.";
    }
    return false;
  });
  if (!p.hp) state.result = "lost";
  else if (input.open && distance(gate, p) < gate.radius) {
    if (state.embers >= state.encounter.required) {
      state.result = "won";
      state.notice = "The tide gate opens. You carried the lantern through.";
    } else state.notice = "The gate is dark. Recover more embers.";
  }
  return state;
}
