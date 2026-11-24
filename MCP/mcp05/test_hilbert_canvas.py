"""Unit and headless-Chrome verification tests for the Hilbert Canvas art benchmark."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .creative_bench import save
from .hilbert_canvas_bench import (
    all_scene_ids,
    check_hilbert_art_map,
    default_chrome,
    default_node,
    verify_assembled_bundle,
)
from .hilbert_storyworld import check_algebra, exact_routes, simulate


def make_fixture_candidate() -> dict:
    cast = [
        ["bellkeeper", "Sera", "Guard the cracked belltower", "Clipped lantern-watch cadence"],
        ["archivist", "Tomas", "Catalog the silence", "Measured ink-stained clauses"],
        ["ferryman", "Nio", "Hear the river echo again", "Low tidal proverbs"],
    ]
    acts = [
        ["First Silence", "Belltower", "Midnight passes unspoken", "The clapper is intact"],
        ["Echo Market", "Cistern", "Rumors traded in jars", "Sound pooled below stone"],
        ["Phase Steps", "Bridge", "Watchmen seal the arches", "Order of strikes shifts tone"],
        ["Tidal Vault", "Sluice", "Water rises toward bronze", "Two keys interfere"],
        ["Dawn Peal", "Belfry", "City gathers in fog", "Truth of the muted bell"],
    ]
    endings_plan = [
        ["end_public_a", False, "Restore a civic chime", "public"],
        ["end_public_b", False, "Keep the city hushed", "public"],
        ["end_public_c", False, "Melt the bronze", "public"],
        ["end_secret_echo", True, "Phase-locked river peal", "c0+c1"],
        ["end_secret_glass", True, "Interference vault harmony", "c2+c3"],
    ]
    plan = {"title": "The Unrung Belfry", "premise": "A river city lost the sound of its night bell.",
            "cast": cast, "acts": acts, "endings": endings_plan}
    model = {
        "basis": {"00": "Still air", "01": "Cistern hum", "10": "Tower tension", "11": "Split resonance"},
        "clues": {f"c{i}": f"Clue {i}" for i in range(6)},
        "operators": {
            "idle": ["I"],
            "mix0": ["H0"],
            "mix1": ["H1"],
            "phase0": ["S0"],
            "phase1": ["S1"],
            "link": ["CX01"],
        },
        "secrets": {
            "end_secret_echo": {"requires": ["c0", "c1"], "projectors": [["Y0+", 0.9, 1.0], ["00", 0.0, 0.6]]},
            "end_secret_glass": {"requires": ["c2", "c3"], "projectors": [["Y1+", 0.9, 1.0], ["00", 0.0, 0.6]]},
        },
    }
    scenes = []
    for idx in range(30):
        sid = f"enc_{idx + 1:02d}"
        shortcut = (idx + 1) % 6 == 3
        if idx == 0:
            opts = [
                ["Wait by lantern", "Still air.", "idle", None, 1],
                ["Strike rim", "The rim hums.", "mix0", "c0", 1],
                ["Tap sluice", "Water answers.", "mix1", "c2", 1],
            ]
        elif idx == 1:
            opts = [
                ["Keep watch", "Quiet steps.", "idle", None, 1],
                ["Twist phase", "A quarter-turn ring.", "phase0", "c1", 1],
                ["Hold bronze", "Cold resonance.", "phase1", "c3", 1],
            ]
        else:
            opts = [
                ["Walk onward", f"Step {idx}a.", "idle", None, 1],
                ["Listen closely", f"Step {idx}b.", "idle", None, 2 if shortcut else 1],
                ["Compare notes", f"Step {idx}c.", "link" if idx == 10 else "idle", None, 1],
            ]
        scenes.append({"id": sid, "title": f"Scene {idx + 1}", "text": f"Unique scene prose {idx + 1}.", "options": opts})
    revised_cast = [[r[0], "Changed belief", "Late line", "Visible cost"] for r in cast]
    terminals = [[e[0], f"Title {e[0]}", f"Final prose for {e[0]}."] for e in endings_plan]
    exact = exact_routes(scenes, model)
    balance = simulate(scenes, model, [e[0] for e in endings_plan if not e[1]])
    return {
        "plan": plan,
        "cast": revised_cast,
        "scenes": scenes,
        "endings": terminals,
        "math": model,
        "algebra": check_algebra(model, plan),
        "balance": balance,
        "exact": exact,
    }


class HilbertCanvasBenchTests(unittest.TestCase):
    def test_art_map_and_headless_chrome_bundle_verification(self):
        candidate = make_fixture_candidate()
        ids = all_scene_ids(candidate)
        landmarks = ["Belltower", "Cistern", "ArchBridge", "SluiceGate", "FogPlaza"]
        palettes = [
            ("#141b2d", "#242f49", "#f4c97a"),
            ("#1e162b", "#312447", "#d98a6c"),
            ("#0f262a", "#1c3d42", "#7de2d1"),
            ("#2b1d19", "#463029", "#f2a65a"),
            ("#192338", "#2d3b55", "#9dc4ff"),
            ("#23192d", "#3d2c4e", "#e8b4f8"),
            ("#10221c", "#1f3c32", "#a8e6cf"),
            ("#2d2418", "#4a3c28", "#ffd166"),
            ("#1a1a2e", "#16213e", "#e94560"),
        ]
        raw_map = {}
        cast_ids = [r[0] for r in candidate["plan"]["cast"]]
        for idx, sid in enumerate(ids):
            sky, ground, light = palettes[idx % len(palettes)]
            raw_map[sid] = [
                sky, ground, light,
                18 + (idx * 17) % 68,
                24 + (idx * 13) % 58,
                35 + (idx * 11) % 50,
                landmarks[idx % len(landmarks)],
                [cast_ids[idx % 3], cast_ids[(idx + 1) % 3]],
            ]
        checked = check_hilbert_art_map(raw_map, candidate)
        self.assertEqual(len(checked), 35)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand_path = root / "candidate.json"
            map_path = root / "art-map.json"
            html_path = root / "index.html"
            ren_path = root / "renderer.js"
            play_path = root / "player.js"
            save(cand_path, candidate)
            save(map_path, checked)
            html_path.write_text(
                "<!doctype html><html><head><meta charset='utf-8'><title>The Unrung Belfry</title></head>"
                "<body><h1 id='scene-title'></h1><canvas id='art' width='640' height='400'></canvas>"
                "<div id='scene-text'></div><div id='reaction'></div><div id='quantum-readout'></div>"
                "<div id='choices'></div><button id='restart'>Restart</button>"
                "<script src='world-data.js'></script><script src='art.js'></script><script src='player.js'></script>"
                "</body></html>\n",
                encoding="utf-8",
            )
            ren_path.write_text(
                """window.drawScene = function(canvas, scene, cast, qstate) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const spec = window.HILBERT_ART[scene.id] || ['#141b2d','#242f49','#f4c97a',50,50,50,'Belltower',[]];
  const [sky, ground, light, xp, yp, sp, landmark, actors] = spec;
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, sky); g.addColorStop(0.65, ground); g.addColorStop(1, '#090b12');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  const cx = xp * W / 100, cy = yp * H / 100, sc = sp / 100;
  const rg = ctx.createRadialGradient(cx, cy, 8, cx, cy, 180 * sc + 40);
  rg.addColorStop(0, light); rg.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = rg; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = ground; ctx.fillRect(0, H * 0.72, W, H * 0.28);
  ctx.strokeStyle = light; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(cx, cy, 42 * sc + 12, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = '#111522';
  if (landmark === 'Belltower') { ctx.fillRect(cx - 28 * sc, cy - 80 * sc, 56 * sc, 130 * sc); }
  else if (landmark === 'Cistern') { ctx.beginPath(); ctx.ellipse(cx, cy + 20, 70 * sc, 28 * sc, 0, 0, Math.PI * 2); ctx.fill(); }
  else { ctx.fillRect(cx - 45 * sc, cy - 35 * sc, 90 * sc, 70 * sc); }
  (actors || []).forEach((a, idx) => {
    const ax = 120 + idx * 110 + (xp % 25);
    ctx.fillStyle = idx % 2 ? light : '#d8e2dc';
    ctx.beginPath(); ctx.arc(ax, H * 0.68, 12, 0, Math.PI * 2); ctx.fill();
    ctx.fillRect(ax - 9, H * 0.68 + 12, 18, 38);
  });
};
""",
                encoding="utf-8",
            )
            play_path.write_text(
                """(function() {
  const W = window.HILBERT_WORLD;
  let idx = 0, clues = new Set(), ending = null, amp = [[1,0],[0,0],[0,0],[0,0]];
  function gate(st, name) {
    if (name === 'I') return st;
    const out = st.map(p => [p[0], p[1]]);
    if (name === 'CX01' || name === 'CX10') {
      const c = name === 'CX01' ? 2 : 1, t = name === 'CX01' ? 1 : 2;
      for (let i = 0; i < 4; i++) out[i] = st[(i & c) ? (i ^ t) : i].slice();
      return out;
    }
    const sym = name[0], bit = name[1] === '0' ? 2 : 1, s2 = Math.SQRT2;
    for (let i = 0; i < 4; i++) {
      if (i & bit) continue;
      const j = i | bit, [ar, ai] = st[i], [br, bi] = st[j];
      if (sym === 'H') { out[i] = [(ar+br)/s2, (ai+bi)/s2]; out[j] = [(ar-br)/s2, (ai-bi)/s2]; }
      else if (sym === 'X') { out[i] = [br, bi]; out[j] = [ar, ai]; }
      else if (sym === 'Z') { out[j] = [-br, -bi]; }
      else if (sym === 'S') { out[j] = [-bi, br]; }
    }
    return out;
  }
  function prob(st, P) {
    const bIdx = ['00','01','10','11'].indexOf(P);
    if (bIdx >= 0) return st[bIdx][0]**2 + st[bIdx][1]**2;
    const axis = P[0], bit = P[1] === '0' ? 2 : 1, sign = P[2];
    let sum = 0;
    if (axis === 'Z') {
      for (let i = 0; i < 4; i++) if (Boolean(i & bit) === (sign === '-')) sum += st[i][0]**2 + st[i][1]**2;
      return sum;
    }
    const s = sign === '+' ? 1 : -1;
    for (let i = 0; i < 4; i++) {
      if (i & bit) continue;
      const [ar, ai] = st[i], [br, bi] = st[i | bit];
      if (axis === 'X') sum += ((ar + s*br)**2 + (ai + s*bi)**2) / 2;
      else sum += ((ar + s*bi)**2 + (ai - s*br)**2) / 2;
    }
    return sum;
  }
  function eligibleSecrets() {
    return Object.entries(W.math.secrets).filter(([, rule]) =>
      rule.requires.every(c => clues.has(c)) &&
      rule.projectors.every(([P, lo, hi]) => { const p = prob(amp, P); return p >= lo - 1e-9 && p <= hi + 1e-9; })
    ).map(([k]) => k).sort();
  }
  window.gameState = function() {
    const born = {'00': prob(amp,'00'), '01': prob(amp,'01'), '10': prob(amp,'10'), '11': prob(amp,'11')};
    return {
      scene: ending ? ending : (idx < W.scenes.length ? W.scenes[idx].id : 'enc_finale'),
      index: idx, born, clues: Array.from(clues).sort(), eligibleSecrets: eligibleSecrets(), ending
    };
  };
  function render() {
    const titleEl = document.getElementById('scene-title');
    const textEl = document.getElementById('scene-text');
    const choicesEl = document.getElementById('choices');
    choicesEl.innerHTML = '';
    let sceneObj;
    if (ending) {
      sceneObj = W.endings.find(e => e.id === ending);
      titleEl.textContent = sceneObj.title; textEl.textContent = sceneObj.text;
    } else if (idx < W.scenes.length) {
      sceneObj = W.scenes[idx];
      titleEl.textContent = sceneObj.title; textEl.textContent = sceneObj.text;
      sceneObj.options.forEach((opt, j) => {
        const b = document.createElement('button');
        b.setAttribute('data-option', String(j)); b.textContent = opt[0];
        b.onclick = () => {
          for (const g of W.math.operators[opt[2]]) amp = gate(amp, g);
          if (opt[3]) clues.add(opt[3]);
          document.getElementById('reaction').textContent = opt[1];
          idx += opt[4]; render();
        };
        choicesEl.appendChild(b);
      });
    } else {
      sceneObj = {id: W.scenes[W.scenes.length - 1].id, title: 'Finale', text: 'Choose the fate of the bell.'};
      titleEl.textContent = sceneObj.title; textEl.textContent = sceneObj.text;
      const elig = new Set(eligibleSecrets());
      W.endings.forEach(e => {
        if (!e.secret || elig.has(e.id)) {
          const b = document.createElement('button');
          b.setAttribute('data-ending', e.id); b.textContent = e.title;
          b.onclick = () => { ending = e.id; render(); };
          choicesEl.appendChild(b);
        }
      });
    }
    window.drawScene(document.getElementById('art'), sceneObj, W.cast, window.gameState());
  }
  document.getElementById('restart').onclick = () => {
    idx = 0; clues = new Set(); ending = null; amp = [[1,0],[0,0],[0,0],[0,0]]; render();
  };
  render();
})();
""",
                encoding="utf-8",
            )
            receipt = verify_assembled_bundle(
                cand_path, map_path, html_path, ren_path, play_path,
                root / "preview", root / "browser", default_node(), default_chrome(),
            )
            self.assertEqual(receipt["status"], "verified")
            self.assertEqual(sorted(receipt["verified_secret_endings"]), ["end_secret_echo", "end_secret_glass"])
            self.assertGreaterEqual(receipt["visited_scenes"], 25)
            self.assertEqual(receipt["distinct_canvas_hashes"], receipt["visited_scenes"])


if __name__ == "__main__":
    unittest.main()
