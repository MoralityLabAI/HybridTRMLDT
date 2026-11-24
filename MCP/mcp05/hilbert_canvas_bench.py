"""Model-authored Canvas art and Hilbert browser player benchmark stages."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "MCP.mcp05"

from .context import sha
from .creative_bench import save
from .inky_goose_game import ROOT, read_json, replay_writes
from .inky_storyworld import require, run_phase

DEFAULT_NODE = Path.home() / "AppData/Local/Programs/nodejs/node.exe"
DEFAULT_CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
VERIFY_MJS = Path(__file__).resolve().with_name("hilbert_browser_verify.mjs")


def default_node() -> Path:
    if DEFAULT_NODE.is_file():
        return DEFAULT_NODE
    found = shutil.which("node")
    require(bool(found), "node executable not found")
    return Path(found)


def default_chrome() -> Path:
    if DEFAULT_CHROME.is_file():
        return DEFAULT_CHROME
    found = shutil.which("chrome")
    require(bool(found), "chrome executable not found")
    return Path(found)


def all_scene_ids(candidate: dict) -> list[str]:
    return [*(s["id"] for s in candidate["scenes"]), *(e[0] for e in candidate["endings"])]


def check_hilbert_art_map(source, candidate: dict) -> dict:
    if isinstance(source, list):
        require(all(isinstance(r, list) and len(r) == 9 and isinstance(r[0], str) for r in source),
                "array art map rows must be [sceneId, sky, ground, light, x, y, scale, landmark, actors]")
        scenes = {r[0]: r[1:] for r in source}
    elif isinstance(source, dict):
        scenes = source.get("scenes", source)
    else:
        raise ValueError("art map must be a JSON object mapping scene IDs to 8-element arrays")
    expected = all_scene_ids(candidate)
    require(isinstance(scenes, dict) and set(scenes) == set(expected),
            f"art map keys must match all {len(expected)} scene and ending IDs: missing={sorted(set(expected)-set(scenes))}")
    cast_ids = {row[0] for row in candidate["plan"]["cast"]}
    normalized = {}
    for sid in expected:
        row = scenes[sid]
        require(isinstance(row, list) and len(row) == 8,
                f"{sid}: expected [skyHex, groundHex, lightHex, xPct, yPct, scalePct, landmark, actorIds]")
        colors, nums, landmark, actors = row[:3], row[3:6], row[6], row[7]
        require(all(isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c.strip()) for c in colors),
                f"{sid}: sky, ground, light must be #RRGGBB hex strings")
        require(all(type(n) in (int, float) and math.isfinite(n) and 1 <= n <= 100 for n in nums),
                f"{sid}: x, y, scale must be numbers in [1, 100]")
        require(isinstance(landmark, str) and landmark.strip(), f"{sid}: landmark must be non-empty string")
        require(isinstance(actors, list) and all(isinstance(a, str) for a in actors) and set(actors) <= cast_ids,
                f"{sid}: actorIds must be a subset of {sorted(cast_ids)}")
        normalized[sid] = [*(c.strip().lower() for c in colors), *(float(n) if isinstance(n, float) else int(n) for n in nums),
                           landmark.strip(), list(actors)]
    require(len({json.dumps(r[:7], sort_keys=True) for r in normalized.values()}) == len(expected),
            "every encounter and ending needs a distinct [sky, ground, light, x, y, scale, landmark] composition")
    landmarks = {r[6].casefold() for r in normalized.values()}
    require(len(landmarks) >= 4, "use at least 4 distinct recurring architectural/setting landmarks across the world")
    palettes = {(r[0], r[1], r[2]) for r in normalized.values()}
    require(len(palettes) >= 8, "use at least 8 distinct three-color atmospheric palettes across acts and endings")
    return normalized


def build_world_data_js(candidate: dict, art_map: dict) -> str:
    plan = candidate["plan"]
    late_by_id = {row[0]: row for row in candidate["cast"]}
    cast = [
        {
            "id": row[0],
            "name": row[1],
            "desire": row[2],
            "voice": row[3],
            "belief": late_by_id[row[0]][1],
            "late_line": late_by_id[row[0]][2],
            "loss": late_by_id[row[0]][3],
        }
        for row in plan["cast"]
    ]
    secret_flags = {row[0]: bool(row[1]) for row in plan["endings"]}
    endings = [
        {"id": row[0], "secret": secret_flags[row[0]], "title": row[1], "text": row[2]}
        for row in candidate["endings"]
    ]
    world = {
        "title": plan["title"],
        "premise": plan["premise"],
        "acts": plan["acts"],
        "cast": cast,
        "scenes": candidate["scenes"],
        "endings": endings,
        "math": candidate["math"],
    }
    return (
        "window.HILBERT_WORLD = " + json.dumps(world, ensure_ascii=False, separators=(",", ":")) + ";\n"
        "window.HILBERT_ART = " + json.dumps(art_map, ensure_ascii=False, separators=(",", ":")) + ";\n"
    )


class _DomNestingCheck(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, str | None]] = []
        self.restart_inside_choices = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        el_id = attr_map.get("id")
        if el_id == "restart" and any(pid == "choices" for _, pid in self.stack):
            self.restart_inside_choices = True
        if tag not in ("meta", "link", "br", "hr", "img", "input"):
            self.stack.append((tag, el_id))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def check_html_page(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    low = html.lower()
    require("<canvas" in low, "index.html must include a <canvas> element")
    for token in ("world-data.js", "art.js", "player.js"):
        require(token in low, f"index.html must load local script {token}")
    for dom_id in ("scene-title", "scene-text", "choices", "restart"):
        require(dom_id in html, f"index.html must include element id {dom_id}")
    require("http://" not in low and "https://" not in low, "index.html must not reference remote URLs")
    checker = _DomNestingCheck()
    checker.feed(html)
    require(not checker.restart_inside_choices, "button#restart must be OUTSIDE #choices so clearing #choices does not remove #restart")
    return html


def check_renderer_js(path: Path, node: Path) -> str:
    source = path.read_text(encoding="utf-8")
    require("drawScene" in source, "renderer.js must define window.drawScene")
    require("window.HILBERT_ART =" not in source and "window.HILBERT_ART=" not in source,
            "do not overwrite window.HILBERT_ART inside renderer.js; read window.HILBERT_ART[scene.id]")
    require("http://" not in source.lower() and "https://" not in source.lower(),
            "renderer.js must be procedural Canvas 2D with no external URLs")
    proc = subprocess.run([str(node), "--check", str(path)], capture_output=True, text=True, timeout=15)
    require(proc.returncode == 0, f"node --check failed on renderer.js: {proc.stderr.strip()}")
    return source


def verify_assembled_bundle(
    candidate_path: Path,
    art_map_path: Path,
    html_path: Path,
    renderer_path: Path,
    player_path: Path,
    preview_dir: Path,
    browser_dir: Path,
    node: Path,
    chrome: Path,
) -> dict:
    proc = subprocess.run([str(node), "--check", str(player_path)], capture_output=True, text=True, timeout=15)
    require(proc.returncode == 0, f"node --check failed on player.js: {proc.stderr.strip()}")
    check_html_page(html_path)
    check_renderer_js(renderer_path, node)
    candidate = read_json(candidate_path)
    art_map = check_hilbert_art_map(read_json(art_map_path), candidate)
    preview_dir.mkdir(parents=True, exist_ok=True)
    browser_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "world-data.js").write_text(build_world_data_js(candidate, art_map), encoding="utf-8", newline="\n")
    shutil.copyfile(html_path, preview_dir / "index.html")
    shutil.copyfile(renderer_path, preview_dir / "art.js")
    shutil.copyfile(player_path, preview_dir / "player.js")
    smoke = subprocess.run(
        [str(node), str(VERIFY_MJS), str(chrome), str(browser_dir), str(preview_dir), str(candidate_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if smoke.returncode != 0:
        detail = (smoke.stderr or smoke.stdout).strip()[-900:]
        raise ValueError(f"Browser verification failed: {detail}")
    return read_json(browser_dir / "browser-receipt.json")


def next_stage_attempt(arm: Path, stage: str) -> Path:
    return arm / f"{stage}-attempt-{1 + len(list(arm.glob(f'{stage}-attempt-*'))):02d}"


def author_canvas_edition(
    arm: Path,
    candidate: dict,
    gdk: Path,
    *,
    node: Path | None = None,
    chrome: Path | None = None,
    context_limit: int = 16384,
) -> dict:
    receipt_path = arm / "canvas-edition.json"
    if receipt_path.is_file():
        return read_json(receipt_path)
    node = (node or default_node()).resolve(strict=True)
    chrome = (chrome or default_chrome()).resolve(strict=True)
    candidate_path = arm / "candidate.json"
    if not candidate_path.is_file():
        save(candidate_path, candidate)
    script = str(Path(__file__).resolve())

    # 1. Model authors 35-scene art-map.json
    art_receipt_path = arm / "canvas-art-map.json"
    if art_receipt_path.is_file():
        art_data = read_json(art_receipt_path)
        art_map = art_data["value"]
        art_map_file = arm / art_data["phase"] / "workspace/result.json"
    else:
        cast_summary = [f"{r[0]} ({r[1]}: {r[3][:70]})" for r in candidate["plan"]["cast"]]
        scene_roster = [f"{s['id']}:{s['title']}" for s in candidate["scenes"]] + [
            f"{e[0]}:{e[1]}" for e in candidate["endings"]
        ]
        acts_summary = [f"Act {i+1} {a[0]} @ {a[1]}" for i, a in enumerate(candidate["plan"]["acts"])]
        art_brief = (
            f"Create result.json via create_file. Design the visual Canvas composition map for all 35 scenes "
            f"(30 encounters + 5 endings) of your Hilbert storyworld {candidate['plan']['title']!r}.\n"
            "Write one JSON object mapping every exact ID below to an 8-item array:\n"
            "[skyHex, groundHex, lightHex, xPct, yPct, scalePct, landmarkName, actorIds]\n"
            "Exact JSON shape example: {\"enc_01\":[\"#1f2a44\",\"#2e3a59\",\"#f5d76e\",50,45,60,\"Bell Tower\",[\"c1\",\"c2\"]], ...}\n"
            "- skyHex, groundHex, lightHex: 6-digit #RRGGBB hex strings with atmospheric contrast across acts and endings.\n"
            "- xPct, yPct, scalePct: integers in 15..90 positioning and scaling the focal landmark/composition.\n"
            "- landmarkName: choose 5 to 8 recurring setting/bell landmarks and vary them across scenes.\n"
            f"- actorIds: array of 1 to 3 cast IDs present in that scene from {[r[0] for r in candidate['plan']['cast']]}.\n"
            "Every single scene ID must have a distinct composition, at least 4 distinct landmarks and 8 distinct color palettes.\n"
            f"Cast: {json.dumps(cast_summary, ensure_ascii=False)}\n"
            f"Acts: {json.dumps(acts_summary, ensure_ascii=False)}\n"
            f"Required 35 IDs: {json.dumps(scene_roster, ensure_ascii=False)}\n"
            "One create_file action, compact JSON, no commentary."
        )
        feedback = ""
        for _ in range(3):
            phase = next_stage_attempt(arm, "canvas-art-map")
            out_file = phase / "workspace/result.json"
            check_cmd = [sys.executable, "-B", script, "check-art-map", str(out_file), "--candidate", str(candidate_path)]
            run_phase(phase, gdk, art_brief + feedback, [out_file.name], check_cmd,
                      max_tokens=min(4200, context_limit - 1900), iterations=1, context=context_limit, allow_basename=True, compact_writes=True)
            try:
                require(out_file.is_file(), "Goose did not receive result.json")
                art_map = check_hilbert_art_map(read_json(out_file), candidate)
                proof = replay_writes(phase, [out_file.name], require_verified=False)
            except (KeyError, TypeError, ValueError) as err:
                save(phase / "validation-error.json", {"error": str(err)})
                feedback = f"\nPrevious art map rejected: {err}. Write complete valid result.json with all 35 IDs."
                continue
            save(art_receipt_path, {"phase": phase.name, "value": art_map, "authorship": proof})
            art_map_file = out_file
            break
        else:
            raise ValueError(f"{arm.name}/canvas-art-map failed: {feedback.strip()}")

    # Persist normalized art-map file for bundle assembly
    normalized_map_path = arm / "art-map.normalized.json"
    save(normalized_map_path, art_map)

    # 2. Model authors index.html
    page_receipt_path = arm / "canvas-page.json"
    if page_receipt_path.is_file():
        page_info = read_json(page_receipt_path)
        html_file = arm / page_info["phase"] / "workspace/index.html"
    else:
        page_brief = (
            f"Call create_file now with exact path index.html for the playable browser edition of {candidate['plan']['title']!r}.\n"
            "CRITICAL: Keep the ENTIRE file under 55 lines total (<style> MUST be under 18 compact lines — do NOT write long CSS variables or hundreds of rules, or you will run out of tokens before <body>!).\n"
            "Write clean HTML5 markup and minimal dark/parchment CSS (no inline application JS, no external/remote URLs).\n"
            "Include inside <body>:\n"
            "- Heading and <canvas id=\"art\" width=\"640\" height=\"400\"></canvas>.\n"
            "- Elements with exact IDs: scene-title, scene-text, reaction, quantum-readout, choices, and OUTSIDE #choices a separate <button id=\"restart\">Restart</button>.\n"
            "- At the end of <body>, load local scripts in this exact order: world-data.js, art.js, player.js.\n"
            "One create_file action only, no commentary."
        )
        feedback = ""
        for _ in range(4):
            phase = next_stage_attempt(arm, "canvas-page")
            out_file = phase / "workspace/index.html"
            check_cmd = [sys.executable, "-B", script, "check-page", str(out_file)]
            run_phase(phase, gdk, page_brief + feedback, [out_file.name], check_cmd,
                      max_tokens=min(3600, context_limit - 1200), iterations=1, context=context_limit, allow_basename=True, compact_writes=True)
            try:
                require(out_file.is_file(), "Goose did not receive index.html")
                check_html_page(out_file)
                proof = replay_writes(phase, [out_file.name], require_verified=False)
            except (OSError, ValueError) as err:
                save(phase / "validation-error.json", {"error": str(err)})
                feedback = f"\nPrevious index.html rejected: {err}. Fix and write complete index.html."
                continue
            save(page_receipt_path, {"phase": phase.name, "authorship": proof})
            html_file = out_file
            break
        else:
            raise ValueError(f"{arm.name}/canvas-page failed: {feedback.strip()}")

    # 3. Model authors renderer.js (art.js)
    renderer_receipt_path = arm / "canvas-renderer.json"
    landmarks = sorted({row[6] for row in art_map.values()})
    cast_designs = [f"{r[0]} ({r[1]}): {r[2]}; {r[3]}" for r in candidate["plan"]["cast"]]
    renderer_brief = (
        f"Call create_file now with exact path renderer.js for your fable {candidate['plan']['title']!r}.\n"
        "Define window.drawScene = function(canvas, scene, cast, qstate) { ... }.\n"
        "At runtime window.HILBERT_ART[scene.id] already supplies your authored array:\n"
        "  [skyColor, groundColor, lightColor, xPercent, yPercent, scalePercent, landmarkName, actorIds]\n"
        "Do NOT assign or overwrite window.HILBERT_ART inside renderer.js.\n"
        "Inside drawScene:\n"
        "1. Paint a rich sky-to-ground linear/radial gradient, horizon silhouette, atmospheric haze/stars, and lighting glow using skyColor, groundColor, lightColor, and xPercent/yPercent.\n"
        "2. Draw distinct procedural architectural/prop geometry for each of your named landmarks scaled by scalePercent/100 around (xPercent*W/100, yPercent*H/100), with fallback geometry for any landmark.\n"
        "3. Draw recognizable multi-shape character silhouettes/figures (head, cloak/coat, lantern/bell prop) for each actorId in actorIds.\n"
        "4. Draw a subtle phase/bell interference ring or wave motif driven by qstate (e.g. qstate && qstate.born, qstate && qstate.clues) and scene.id so every scene and ending has a distinct, colorful composition.\n"
        f"Your landmarks: {json.dumps(landmarks, ensure_ascii=False)}\n"
        f"Your cast: {json.dumps(cast_designs, ensure_ascii=False)}\n"
        "Runtime argument types: scene is {id: string (e.g. 'enc_01', 'enc_finale'), title: string} "
        "(compute integer seed h by summing scene.id.charCodeAt(i)*(i+1) — NEVER write scene.id * number which is NaN, "
        "and ALWAYS provide a fallback 8-item tuple if window.HILBERT_ART[scene.id] is undefined for 'enc_finale'!), "
        "cast is an array of {id, name} objects (do NOT call string methods like c.split on cast items), "
        "and qstate is {born: object, clues: string[]} (use (qstate && Array.isArray(qstate.clues) ? qstate.clues.length : 0)).\n"
        "Group landmarks by keyword/substring into 5-6 procedural architectural styles plus seed-h variation "
        "(do NOT write 30 separate landmark functions!), use skyColor, groundColor, lightColor, and valid rgba()/hsla() strings directly, "
        "pass ctx explicitly to any helpers, and avoid Math.random() so each of the 33+ scenes renders a unique deterministic hash.\n"
        "CRITICAL: Keep renderer.js concise (65 to 95 lines total, no long comments) so create_file closes cleanly! "
        "One create_file action only, no commentary."
    )
    if renderer_receipt_path.is_file():
        ren_info = read_json(renderer_receipt_path)
        renderer_file = arm / ren_info["phase"] / "workspace/renderer.js"
    else:
        feedback = ""
        for _ in range(3):
            phase = next_stage_attempt(arm, "canvas-renderer")
            out_file = phase / "workspace/renderer.js"
            check_cmd = [sys.executable, "-B", script, "check-renderer", str(out_file), "--node", str(node)]
            run_phase(phase, gdk, renderer_brief + feedback, [out_file.name], check_cmd,
                      max_tokens=min(4200, context_limit - 1600), iterations=1, context=context_limit, allow_basename=True, compact_writes=True)
            try:
                require(out_file.is_file(), "Goose did not receive renderer.js")
                check_renderer_js(out_file, node)
                proof = replay_writes(phase, [out_file.name], require_verified=False)
            except (OSError, ValueError) as err:
                save(phase / "validation-error.json", {"error": str(err)})
                feedback = f"\nPrevious renderer.js rejected: {err}. Write concise valid JavaScript under 135 lines."
                continue
            save(renderer_receipt_path, {"phase": phase.name, "authorship": proof})
            renderer_file = out_file
            break
        else:
            raise ValueError(f"{arm.name}/canvas-renderer failed: {feedback.strip()}")

    # 4. Model authors player.js and passes full headless Chrome verification
    player_receipt_path = arm / "canvas-player.json"
    player_brief = (
        f"Call create_file now with exact path \"player.js\" (write ONLY pure JavaScript, NO HTML tags!) "
        f"to run the interactive Hilbert storyworld {candidate['plan']['title']!r}.\n"
        "CRITICAL: window.HILBERT_WORLD and window.HILBERT_ART are ALREADY loaded by world-data.js before player.js runs — "
        "NEVER embed fallback scenes, cast, endings, or math objects inside player.js! Keep player.js concise (~105 lines).\n"
        "Exact runtime data shapes in window.HILBERT_WORLD:\n"
        "- HILBERT_WORLD.scenes: array of 30 objects {id, title, text, options: [[label, reply, operatorName, clueOrNull, advanceInt], ...]}.\n"
        "- HILBERT_WORLD.endings: array of 5 objects {id, secret, title, text}.\n"
        f"- HILBERT_WORLD.math.operators (plain object mapping operatorName -> array of gate strings!): {json.dumps(candidate['math']['operators'])}\n"
        f"- HILBERT_WORLD.math.secrets (plain object keyed by secret ending ID, NOT an array — iterate with Object.keys(HILBERT_WORLD.math.secrets)!): {json.dumps(candidate['math']['secrets'])}\n"
        "DOM element IDs in index.html: #art (canvas), #scene-title, #scene-text, #reaction, #quantum-readout, #choices (div), and #restart (button outside #choices).\n"
        "Exact quantum & UI rules you MUST implement accurately:\n"
        "1. State: sceneIndex = 0, clues = new Set(), ending = null, amp = [[1,0],[0,0],[0,0],[0,0]] (0='00',1='01',2='10',3='11').\n"
        "2. Applying operatorName: look up const seq = HILBERT_WORLD.math.operators[operatorName] || []; for each gate string g of seq:\n"
        "   - if g==='I': continue.\n"
        "   - if g==='CX01': swap amp[2] and amp[3]. if g==='CX10': swap amp[1] and amp[3].\n"
        "   - else let sym=g[0], bit=(g[1]==='0'?2:1); loop i=0..3 where (i & bit)===0, let j=i|bit, [ar,ai]=amp[i], [br,bi]=amp[j]:\n"
        "     * sym==='H': amp[i]=[(ar+br)/Math.SQRT2,(ai+bi)/Math.SQRT2]; amp[j]=[(ar-br)/Math.SQRT2,(ai-bi)/Math.SQRT2];\n"
        "     * sym==='X': amp[i]=[br,bi]; amp[j]=[ar,ai];\n"
        "     * sym==='Z': amp[j]=[-br,-bi];\n"
        "     * sym==='S': amp[j]=[-bi,br];\n"
        "3. Projector probability prob(amp, P) (note P has 3 chars like 'X0+', 'Y1-', 'Z0+', or 2 chars '00'..'11'!):\n"
        "   - if P==='00'||P==='01'||P==='10'||P==='11': idx=['00','01','10','11'].indexOf(P); return amp[idx][0]**2 + amp[idx][1]**2;\n"
        "   - else let axis=P[0], bit=(P[1]==='0'?2:1), sign=P[2], sum=0;\n"
        "     * if axis==='Z': for i=0..3 if (Boolean(i&bit)===(sign==='-')) sum += amp[i][0]**2 + amp[i][1]**2; return sum;\n"
        "     * if axis==='X': let s=(sign==='+'?1:-1); for i=0..3 if ((i&bit)===0) { let j=i|bit; sum += ((amp[i][0]+s*amp[j][0])**2 + (amp[i][1]+s*amp[j][1])**2)/2; } return sum;\n"
        "     * if axis==='Y': let s=(sign==='+'?1:-1); for i=0..3 if ((i&bit)===0) { let j=i|bit; sum += ((amp[i][0]+s*amp[j][1])**2 + (amp[i][1]-s*amp[j][0])**2)/2; } return sum;\n"
        "4. getEligibleSecrets(): Object.keys(HILBERT_WORLD.math.secrets).filter(id => { const r = HILBERT_WORLD.math.secrets[id]; return r.requires.every(c => clues.has(c)) && r.projectors.every(([P,lo,hi]) => { const v = prob(amp,P); return v >= lo - 1e-9 && v <= hi + 1e-9; }); }).sort().\n"
        "5. Expose window.gameState = function() returning {scene: ending ? ending : (sceneIndex < HILBERT_WORLD.scenes.length ? HILBERT_WORLD.scenes[sceneIndex].id : 'enc_finale'), index: sceneIndex, born: {'00':prob(amp,'00'),'01':prob(amp,'01'),'10':prob(amp,'10'),'11':prob(amp,'11')}, clues: Array.from(clues).sort(), eligibleSecrets: getEligibleSecrets(), ending}.\n"
        "6. UI render() (use let/const inside for-loops so click handlers capture each option/ending correctly, never var!):\n"
        "   - If !ending && sceneIndex < HILBERT_WORLD.scenes.length: let sc = HILBERT_WORLD.scenes[sceneIndex]; set #scene-title and #scene-text; clear #choices and append 3 buttons with EXACT attribute data-option=\"0\", data-option=\"1\", data-option=\"2\" (NEVER data-choice!). On click of j: apply opt[2], if (opt[3]) clues.add(opt[3]), set #reaction to opt[1], sceneIndex += opt[4], call render(). Call window.drawScene(document.getElementById('art'), sc, HILBERT_WORLD.cast, window.gameState()).\n"
        "   - If !ending && sceneIndex >= HILBERT_WORLD.scenes.length: set #scene-title and #scene-text to finale prompt; let elig = getEligibleSecrets(); clear #choices and for each e of HILBERT_WORLD.endings where (!e.secret || elig.includes(e.id)), append <button data-ending=\"...\"> (e.id). On click: ending = e.id; render(). Call window.drawScene(document.getElementById('art'), {id:'enc_finale', title:'Finale'}, HILBERT_WORLD.cast, window.gameState()).\n"
        "   - If ending: find endObj in HILBERT_WORLD.endings; set #scene-title to endObj.title, #scene-text to endObj.text, clear #choices (no buttons), and call window.drawScene(document.getElementById('art'), endObj, HILBERT_WORLD.cast, window.gameState()).\n"
        "   - Wire document.getElementById('restart').addEventListener('click', ...) to reset sceneIndex=0, amp=[[1,0],[0,0],[0,0],[0,0]], clues=new Set(), ending=null, and call render().\n"
        "Emit one create_file action with path \"player.js\" and pure JS content only."
    )

    feedback = ""
    for prev_phase in reversed(sorted(arm.glob("canvas-player-attempt-*"))):
        prev_player = prev_phase / "workspace/player.js"
        if not prev_player.is_file():
            continue
        try:
            preview_dir = prev_phase / "preview"
            browser_dir = prev_phase / "browser"
            browser_receipt = verify_assembled_bundle(
                candidate_path, normalized_map_path, html_file, renderer_file, prev_player,
                preview_dir, browser_dir, node, chrome,
            )
            player_proof = replay_writes(prev_phase, [prev_player.name], require_verified=False)
            save(player_receipt_path, {"phase": prev_phase.name, "authorship": player_proof, "browser": browser_receipt})
            break
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    if not player_receipt_path.is_file():
        for attempt_idx in range(4):
            phase = next_stage_attempt(arm, "canvas-player")
            out_file = phase / "workspace/player.js"
            preview_dir = phase / "preview"
            browser_dir = phase / "browser"
            check_cmd = [
                sys.executable, "-B", script, "verify-bundle",
                "--candidate", str(candidate_path),
                "--art-map", str(normalized_map_path),
                "--html", str(html_file),
                "--renderer", str(renderer_file),
                "--player", str(out_file),
                "--preview", str(preview_dir),
                "--browser", str(browser_dir),
                "--node", str(node),
                "--chrome", str(chrome),
            ]
            run_phase(phase, gdk, player_brief + feedback, [out_file.name], check_cmd,
                      max_tokens=min(5000, context_limit - 2560), iterations=1, context=min(8192, context_limit), allow_basename=True, compact_writes=True)
            try:
                require(out_file.is_file(), "Goose did not receive player.js")
                browser_receipt = verify_assembled_bundle(
                    candidate_path, normalized_map_path, html_file, renderer_file, out_file,
                    preview_dir, browser_dir, node, chrome,
                )
                player_proof = replay_writes(phase, [out_file.name], require_verified=False)
                save(player_receipt_path, {"phase": phase.name, "authorship": player_proof, "browser": browser_receipt})
                break
            except (OSError, ValueError) as err:
                err_str = str(err)
                save(phase / "validation-error.json", {"error": err_str})
                # If failure specifically points to drawScene or Canvas visual diversity, let the model repair renderer.js first!
                if attempt_idx < 2 and any(tok in err_str for tok in ("drawScene", "quantized colors", "pairwise RGB L1", "distinct Canvas scene")):
                    rphase = next_stage_attempt(arm, "canvas-renderer-repair")
                    rout = rphase / "workspace/renderer.js"
                    rcheck = [sys.executable, "-B", script, "check-renderer", str(rout), "--node", str(node)]
                    rbrief = (
                        renderer_brief
                        + f"\n\nBrowser Canvas diagnostic on your previous renderer.js:\n{err_str}\n"
                        "Make sure every scene uses window.HILBERT_ART[scene.id] colors, positions, scales, and landmarks "
                        "plus scene.id-specific geometry AND sets window.drawScene = drawScene so every scene renders a distinct, colorful Canvas composition."
                    )
                    run_phase(rphase, gdk, rbrief, [rout.name], rcheck,
                              max_tokens=min(4200, context_limit - 1800), iterations=1, context=min(8192, context_limit), allow_basename=True, compact_writes=True)
                    if rout.is_file():
                        check_renderer_js(rout, node)
                        rproof = replay_writes(rphase, [rout.name], require_verified=False)
                        save(renderer_receipt_path, {"phase": rphase.name, "authorship": rproof})
                        renderer_file = rout
                        if out_file.is_file():
                            try:
                                browser_receipt = verify_assembled_bundle(
                                    candidate_path, normalized_map_path, html_file, renderer_file, out_file,
                                    preview_dir, browser_dir, node, chrome,
                                )
                                player_proof = replay_writes(phase, [out_file.name], require_verified=False)
                                save(player_receipt_path, {"phase": phase.name, "authorship": player_proof, "browser": browser_receipt})
                                break
                            except (OSError, ValueError) as r_err:
                                err_str = str(r_err)
                                save(phase / "validation-error.json", {"error": err_str})
                feedback = f"\nBrowser verification diagnostic on previous attempt:\n{err_str}\nWrite complete fixed player.js."
        else:
            raise ValueError(f"{arm.name}/canvas-player failed: {feedback.strip()}")

    # Publish verified web edition + screenshots into arm / export
    export_dir = arm / "export"
    web_out = export_dir / "web"
    shots_out = export_dir / "screenshots"
    web_out.mkdir(parents=True, exist_ok=True)
    shots_out.mkdir(parents=True, exist_ok=True)
    for fname in ("index.html", "world-data.js", "art.js", "player.js"):
        shutil.copyfile(preview_dir / fname, web_out / fname)
    shot_hashes = {}
    for png in sorted(browser_dir.glob("*.png")):
        shutil.copyfile(png, shots_out / png.name)
        shot_hashes[png.name] = sha(png.read_bytes())
    shutil.copyfile(browser_dir / "browser-receipt.json", export_dir / "browser-receipt.json")

    renderer_text = renderer_file.read_text(encoding="utf-8")
    edition_summary = {
        "status": "verified",
        "art_map_phase": read_json(art_receipt_path)["phase"],
        "page_phase": read_json(page_receipt_path)["phase"],
        "renderer_phase": read_json(renderer_receipt_path)["phase"],
        "player_phase": read_json(player_receipt_path)["phase"],
        "art_map_stats": {
            "scenes_mapped": len(art_map),
            "distinct_landmarks": len({r[6].casefold() for r in art_map.values()}),
            "landmarks": sorted({r[6] for r in art_map.values()}),
            "distinct_palettes": len({(r[0], r[1], r[2]) for r in art_map.values()}),
        },
        "renderer_stats": {
            "bytes": len(renderer_text.encode("utf-8")),
            "lines": len(renderer_text.splitlines()),
            "canvas_ops": sum(renderer_text.count(op) for op in (
                "beginPath", "arc", "lineTo", "quadraticCurveTo", "bezierCurveTo",
                "fillRect", "strokeRect", "ellipse", "createLinearGradient", "createRadialGradient",
            )),
        },
        "browser": browser_receipt,
        "screenshots_sha256": shot_hashes,
        "web_artifacts_sha256": {
            fname: sha((web_out / fname).read_bytes())
            for fname in ("index.html", "world-data.js", "art.js", "player.js")
        },
        "authorship": {
            "art_map": read_json(art_receipt_path)["authorship"],
            "index_html": read_json(page_receipt_path)["authorship"],
            "renderer_js": read_json(renderer_receipt_path)["authorship"],
            "player_js": read_json(player_receipt_path)["authorship"],
        },
    }
    if (export_dir / "manifest.json").is_file():
        manifest = read_json(export_dir / "manifest.json")
        manifest["canvas_edition"] = {
            "browser_status": browser_receipt["status"],
            "visited_scenes": browser_receipt["visited_scenes"],
            "visual_metrics": browser_receipt["visual_metrics"],
            "web_artifacts_sha256": edition_summary["web_artifacts_sha256"],
            "screenshots_sha256": shot_hashes,
        }
        save(export_dir / "manifest.json", manifest)
    save(receipt_path, edition_summary)
    return edition_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_map = sub.add_parser("check-art-map")
    p_map.add_argument("path", type=Path)
    p_map.add_argument("--candidate", type=Path, required=True)

    p_page = sub.add_parser("check-page")
    p_page.add_argument("path", type=Path)

    p_ren = sub.add_parser("check-renderer")
    p_ren.add_argument("path", type=Path)
    p_ren.add_argument("--node", type=Path, default=None)

    p_ver = sub.add_parser("verify-bundle")
    p_ver.add_argument("--candidate", type=Path, required=True)
    p_ver.add_argument("--art-map", type=Path, required=True)
    p_ver.add_argument("--html", type=Path, required=True)
    p_ver.add_argument("--renderer", type=Path, required=True)
    p_ver.add_argument("--player", type=Path, required=True)
    p_ver.add_argument("--preview", type=Path, required=True)
    p_ver.add_argument("--browser", type=Path, required=True)
    p_ver.add_argument("--node", type=Path, default=None)
    p_ver.add_argument("--chrome", type=Path, default=None)

    args = parser.parse_args()
    if args.cmd == "check-art-map":
        res = check_hilbert_art_map(read_json(args.path), read_json(args.candidate))
        print(f"Verified {len(res)} distinct Canvas compositions")
    elif args.cmd == "check-page":
        check_html_page(args.path)
        print("Verified index.html structure")
    elif args.cmd == "check-renderer":
        check_renderer_js(args.path, args.node or default_node())
        print("Verified renderer.js syntax and Canvas contract")
    elif args.cmd == "verify-bundle":
        receipt = verify_assembled_bundle(
            args.candidate, args.art_map, args.html, args.renderer, args.player,
            args.preview, args.browser, args.node or default_node(), args.chrome or default_chrome(),
        )
        print(json.dumps(receipt))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Canvas check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
