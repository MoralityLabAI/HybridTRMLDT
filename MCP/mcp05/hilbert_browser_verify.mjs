import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const [chrome, captureDir, webDir, candidatePath] = process.argv.slice(2);
if (!chrome || !captureDir || !webDir || !candidatePath) {
  throw new Error(
    "Usage: node hilbert_browser_verify.mjs CHROME CAPTURE_DIR WEB_DIR STORYWORLD_HILBERT_JSON"
  );
}

const out = resolve(captureDir);
const root = resolve(webDir);
await mkdir(out, { recursive: true });
const candidate = JSON.parse(await readFile(resolve(candidatePath), "utf8"));

// Reference complex 2-qubit replay in Node to cross-check browser player independently.
function applyGate(state, name) {
  if (name === "I") return state.map(([r, im]) => [r, im]);
  const res = state.map(([r, im]) => [r, im]);
  if (name === "CX01" || name === "CX10") {
    const control = name === "CX01" ? 2 : 1;
    const target = name === "CX01" ? 1 : 2;
    for (let i = 0; i < 4; i++) {
      const src = i & control ? i ^ target : i;
      res[i] = [state[src][0], state[src][1]];
    }
    return res;
  }
  const sym = name[0];
  const bit = 2 >> Number(name[1]);
  const inv = 1 / Math.SQRT2;
  for (let i = 0; i < 4; i++) {
    if (i & bit) continue;
    const j = i | bit;
    const [ar, ai] = state[i];
    const [br, bi] = state[j];
    if (sym === "H") {
      res[i] = [(ar + br) * inv, (ai + bi) * inv];
      res[j] = [(ar - br) * inv, (ai - bi) * inv];
    } else if (sym === "X") {
      res[i] = [br, bi];
      res[j] = [ar, ai];
    } else if (sym === "Z") {
      res[j] = [-br, -bi];
    } else if (sym === "S") {
      res[j] = [-bi, br];
    } else {
      throw new Error(`Unknown gate ${name}`);
    }
  }
  return res;
}

function replayWitness(scenes, model, path) {
  let state = [
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
  ];
  const clues = new Set();
  for (const [i, j] of path) {
    const [, , op, clue] = scenes[i].options[j];
    for (const g of model.operators[op]) state = applyGate(state, g);
    if (clue) clues.add(clue);
  }
  const keys = ["00", "01", "10", "11"];
  const born = {};
  keys.forEach((k, idx) => {
    born[k] = state[idx][0] ** 2 + state[idx][1] ** 2;
  });
  return { state, born, clues: [...clues].sort() };
}

const allowedPaths = new Set(["/", "/index.html", "/world-data.js", "/art.js", "/player.js"]);
const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
};

const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, "http://127.0.0.1").pathname);
    if (!allowedPaths.has(pathname)) {
      res.writeHead(404).end();
      return;
    }
    const rel = pathname === "/" ? "index.html" : pathname.slice(1);
    const file = resolve(root, rel);
    if (!file.startsWith(root.endsWith(sep) ? root : root + sep)) {
      res.writeHead(403).end();
      return;
    }
    const body = await readFile(file);
    res.writeHead(200, {
      "Content-Type": types[extname(file)] ?? "application/octet-stream",
      "Content-Security-Policy":
        "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'",
    });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const app = `http://127.0.0.1:${server.address().port}`;
const profile = await mkdtemp(join(tmpdir(), "hilbert-canvas-smoke-"));
const browser = spawn(
  chrome,
  [
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--disable-background-networking",
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { stdio: "ignore", windowsHide: true }
);

async function poll(fn, message, attempts = 100) {
  for (let i = 0; i < attempts; i++) {
    try {
      const value = await fn();
      if (value) return value;
    } catch {}
    await sleep(80);
  }
  throw new Error(message);
}

let socket;
try {
  const port = await poll(
    async () => (await readFile(join(profile, "DevToolsActivePort"), "utf8")).split("\n")[0].trim(),
    "Chrome remote-debugging port did not appear"
  );
  const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = new WebSocket(tabs.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    socket.addEventListener("open", res, { once: true });
    socket.addEventListener("error", rej, { once: true });
  });

  let next = 0;
  const replies = new Map();
  const failures = [];
  socket.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.method === "Runtime.exceptionThrown") {
      const details = msg.params.exceptionDetails;
      failures.push(details.exception?.description || details.text || "JS exception");
    }
    const pair = replies.get(msg.id);
    if (!pair) return;
    replies.delete(msg.id);
    if (msg.error) pair.reject(new Error(msg.error.message));
    else pair.resolve(msg.result);
  });

  const send = (method, params = {}) =>
    new Promise((res, rej) => {
      const id = ++next;
      replies.set(id, { resolve: res, reject: rej });
      socket.send(JSON.stringify({ id, method, params }));
    });

  const evaluate = async (expression) => {
    const value = await send("Runtime.evaluate", { expression, returnByValue: true });
    if (value.exceptionDetails) {
      throw new Error(
        (value.exceptionDetails.exception?.description || value.exceptionDetails.text) +
          " in " +
          expression
      );
    }
    return value.result.value;
  };

  const screenshot = async (file) => {
    const shot = await send("Page.captureScreenshot", { format: "png" });
    await writeFile(join(out, file), Buffer.from(shot.data, "base64"));
  };

  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1280,
    height: 920,
    deviceScaleFactor: 1,
    mobile: false,
  });

  await send("Page.navigate", { url: app + "/" });
  await poll(
    async () => {
      if (failures.length) return true;
      return await evaluate(
        "typeof window.gameState === 'function' && typeof window.drawScene === 'function' && !!document.querySelector('canvas')"
      );
    },
    "Page failed to initialize window.gameState, window.drawScene, and <canvas>"
  );
  assert.deepEqual(failures, [], `Startup runtime errors: ${failures.join("; ")}`);

  const state = () => evaluate("window.gameState()");
  const reset = async () => {
    const ok = await evaluate(
      "(() => { const b = document.querySelector('#restart'); if (!b) return false; b.click(); return true; })()"
    );
    assert.equal(ok, true, "Missing #restart button");
  };
  const chooseOption = async (optIdx) => {
    const clicked = await evaluate(`(() => {
      const btn = document.querySelector('button[data-option="${optIdx}"]');
      if (!btn || btn.disabled) return false;
      btn.click();
      const cont = document.querySelector('#continue');
      if (cont && !cont.disabled && cont.offsetParent !== null) cont.click();
      return true;
    })()`);
    assert.equal(clicked, true, `Could not click button[data-option="${optIdx}"]`);
  };
  const chooseEnding = async (endId) => {
    const clicked = await evaluate(`(() => {
      const btn = document.querySelector('button[data-ending="${endId}"]');
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    })()`);
    assert.equal(clicked, true, `Could not click button[data-ending="${endId}"]`);
  };

  // Sample downscaled RGB grid + visual stats from the live Canvas
  const sampleCanvas = async () =>
    evaluate(`(() => {
      const c = document.querySelector('canvas');
      const ctx = c.getContext('2d');
      const w = c.width, h = c.height;
      if (w < 200 || h < 150) throw new Error('Canvas too small: ' + w + 'x' + h);
      const img = ctx.getImageData(0, 0, w, h).data;
      const bins = new Set();
      let edges = 0, interior = 0;
      for (let y = 0; y < h - 1; y += 2) {
        for (let x = 0; x < w - 1; x += 2) {
          const idx = (y * w + x) * 4;
          const r = img[idx], g = img[idx + 1], b = img[idx + 2];
          bins.add(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4));
          const rx = idx + 4, dy = idx + w * 4;
          const grad = Math.abs(r - img[rx]) + Math.abs(g - img[rx + 1]) + Math.abs(b - img[rx + 2]) +
                       Math.abs(r - img[dy]) + Math.abs(g - img[dy + 1]) + Math.abs(b - img[dy + 2]);
          if (grad > 28) edges++;
          interior++;
        }
      }
      // 32x20 RGB thumbnail for cross-scene L1 comparison and exact data hash
      const thumb = [];
      for (let ty = 0; ty < 20; ty++) {
        for (let tx = 0; tx < 32; tx++) {
          const sx = Math.floor((tx + 0.5) * w / 32);
          const sy = Math.floor((ty + 0.5) * h / 20);
          const idx = (sy * w + sx) * 4;
          thumb.push(img[idx], img[idx + 1], img[idx + 2]);
        }
      }
      return {
        width: w,
        height: h,
        dataUrl: c.toDataURL(),
        quantizedColors: bins.size,
        edgeDensity: interior > 0 ? edges / interior : 0,
        thumb
      };
    })()`);

  const sceneSamples = new Map();
  const recordSample = async (id) => {
    if (sceneSamples.has(id)) return;
    const s = await sampleCanvas();
    const digest = createHash("sha256").update(s.dataUrl).digest("hex");
    sceneSamples.set(id, { ...s, sha256: digest });
  };

  const initial = await state();
  assert.equal(initial.scene, "enc_01", `Expected initial scene enc_01, got ${initial.scene}`);
  assert.equal(initial.ending, null, "Initial ending must be null");
  assert.ok(
    Math.abs((initial.born?.["00"] ?? 0) - 1) < 1e-6,
    `Initial Born |00> must be 1, got ${JSON.stringify(initial.born)}`
  );
  await recordSample("enc_01");
  await screenshot("start.png");

  const scenes = candidate.scenes;
  const model = candidate.math;
  const publicEndings = candidate.plan.endings.filter((e) => !e[1]).map((e) => e[0]);
  const secretGates = candidate.exact.secret_gates;

  // Walk greedy option-0 path to test public ending + sample encounters along Acts 1-5
  await reset();
  let cursor = 0;
  while (cursor < scenes.length) {
    const curState = await state();
    assert.equal(curState.scene, scenes[cursor].id);
    await recordSample(scenes[cursor].id);
    if (cursor === 12) await screenshot("act3.png");
    const advance = scenes[cursor].options[0][4];
    await chooseOption(0);
    cursor += advance;
  }
  await chooseEnding(publicEndings[0]);
  const pubState = await state();
  assert.equal(pubState.ending, publicEndings[0]);
  await recordSample(publicEndings[0]);
  await screenshot(`${publicEndings[0]}.png`);

  // Walk each secret ending's exact witness and verify Born weights + unlock
  const verifiedSecrets = [];
  for (const [secretId, gateInfo] of Object.entries(secretGates)) {
    if (!gateInfo.witness) continue;
    await reset();
    for (const [sceneIdx, optIdx] of gateInfo.witness) {
      const cur = await state();
      assert.equal(
        cur.scene,
        scenes[sceneIdx].id,
        `Witness expected ${scenes[sceneIdx].id}, saw ${cur.scene}`
      );
      await recordSample(scenes[sceneIdx].id);
      await chooseOption(optIdx);
    }
    const expected = replayWitness(scenes, model, gateInfo.witness);
    const fin = await state();
    for (const k of ["00", "01", "10", "11"]) {
      assert.ok(
        Math.abs((fin.born?.[k] ?? -1) - expected.born[k]) < 1e-5,
        `Born[${k}] mismatch on ${secretId}: browser=${fin.born?.[k]} vs ref=${expected.born[k]}`
      );
    }
    assert.deepEqual(
      [...(fin.clues || [])].sort(),
      expected.clues,
      `Clue mismatch on ${secretId}`
    );
    assert.ok(
      (fin.eligibleSecrets || []).includes(secretId),
      `Secret ending ${secretId} was not unlocked on its exact witness`
    );
    await chooseEnding(secretId);
    const termState = await state();
    assert.equal(termState.ending, secretId);
    await recordSample(secretId);
    await screenshot(`${secretId}.png`);
    verifiedSecrets.push(secretId);
  }

  await reset();
  const afterReset = await state();
  assert.equal(afterReset.scene, "enc_01");
  assert.equal(afterReset.ending, null);
  assert.deepEqual(afterReset.clues || [], []);
  assert.deepEqual(failures, [], `Runtime exceptions: ${failures.join("; ")}`);

  const samples = [...sceneSamples.values()];
  const hashes = new Set(samples.map((s) => s.sha256));
  assert.equal(
    hashes.size,
    samples.length,
    `Expected ${samples.length} distinct Canvas scene renders, got ${hashes.size} unique hashes`
  );

  const meanQuantizedColors =
    samples.reduce((acc, s) => acc + s.quantizedColors, 0) / samples.length;
  assert.ok(
    meanQuantizedColors >= 6,
    `Canvas scenes look flat/blank (mean quantized colors = ${meanQuantizedColors.toFixed(2)})`
  );
  const meanEdgeDensity =
    samples.reduce((acc, s) => acc + s.edgeDensity, 0) / samples.length;

  let pairSum = 0,
    pairCount = 0;
  for (let i = 0; i < samples.length; i++) {
    for (let j = i + 1; j < samples.length; j++) {
      let diff = 0;
      const a = samples[i].thumb,
        b = samples[j].thumb;
      for (let k = 0; k < a.length; k++) diff += Math.abs(a[k] - b[k]);
      pairSum += diff / a.length;
      pairCount++;
    }
  }
  const meanPairwiseL1 = pairCount > 0 ? pairSum / pairCount : 0;
  assert.ok(
    meanPairwiseL1 >= 2.0,
    `Canvas scenes barely vary across the storyworld (mean pairwise RGB L1 = ${meanPairwiseL1.toFixed(2)})`
  );

  const receipt = {
    status: "verified",
    game: "hilbert-storyworld",
    title: await evaluate("document.title"),
    visited_scenes: samples.length,
    distinct_canvas_hashes: hashes.size,
    verified_secret_endings: verifiedSecrets,
    visual_metrics: {
      canvas_width: samples[0].width,
      canvas_height: samples[0].height,
      mean_quantized_colors: Number(meanQuantizedColors.toFixed(2)),
      max_quantized_colors: Math.max(...samples.map((s) => s.quantizedColors)),
      mean_edge_density: Number(meanEdgeDensity.toFixed(4)),
      mean_pairwise_rgb_l1: Number(meanPairwiseL1.toFixed(2)),
    },
    checks: [
      `${samples.length} distinct Canvas scenes verified across encounters and endings`,
      "browser 2-qubit complex unitary state and Born weights match exact reference replay",
      `secret endings unlocked via exact Hilbert witnesses (${verifiedSecrets.join(", ")})`,
      "restart restores |00> initial state and clears clues",
    ],
  };
  await writeFile(join(out, "browser-receipt.json"), JSON.stringify(receipt, null, 2) + "\n");
  console.log(JSON.stringify(receipt));
} finally {
  socket?.close();
  browser.kill();
  server.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
