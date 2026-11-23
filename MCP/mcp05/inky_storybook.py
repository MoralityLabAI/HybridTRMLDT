"""Ask Inkling to illustrate a picture-book edition of its verified storyworld."""
from __future__ import annotations

import argparse
import json
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
from .inky_goose_game import events, read_json, replay_writes
from .inky_storyworld import apply_source_replacements, browser_command, require, run_phase

SOURCE = "storybook-scene.js"
STYLE = "storybook-finish.txt"
EDITION = "storybook-export"
BRIEF = """Reillustrate your original fable The Hat That Held the Moon as a whimsical mid-century
picture book in the spirit of Dr. Seuss. The owner specifically wants the visual treatment.
Make a real composition change: a warm paper ground, big irregular ink contours, impossible curving
silhouettes, punchy spot color, and readable figures that occupy the frame. Use playful perspective
and draw the story's original floppy star-stitched hat as a recurring motif where it fits. Pip, Mara,
Old Fen, and the Moon are your own characters. Keep their identities. Invent your own drawings.

Use the create_file tool to write storybook-scene.js. Start with function drawScene(canvas, scene, cast).
This function will replace your old drawScene inside the original Canvas module. It owns the full painting.
Already available in that scope:
  var INKY_ART, keyed by scene.id:
    [skyColor, groundColor, lightColor, xPercent, yPercent, scalePercent, landmarkName, actorIds].
  drawLandmark(ctx, landmarkName, canvasWidth, canvasHeight, scale).
  drawActor(ctx, actorId, x, y, scale, t), with the feet at y. It draws your four original silhouettes.
  rr(ctx,x,y,w,h,radius) starts a rounded-rectangle path, and shade(hex,amount) returns a CSS color.
Call the existing drawActor and drawLandmark for the figures and landmarks. Reuse those drawings directly.
Give each scene its own composition with the actual scene data. Spend this pass on the illustrated paper
background, bolder framing, and a little original hat motif. Three or four simple drawn forms can carry
the look. After drawScene, add a compact style element for matching paper and ink on the page. Keep all
the controls working. One file of roughly 45 lines, including style. Call run_verifier after creating it.
"""


def read_style(path):
    data, landmark, css = re.split(r"^---[ \t]*$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return {"data": data.strip(), "landmark": landmark.strip(), "css": css.strip()}


def check_ink(path):
    css = path.read_text(encoding="utf-8")
    for name in ("parchment", "parchment-dim", "silver", "amber-soft"):
        values = re.findall(rf"--{name}\s*:\s*#([0-9a-fA-F]{{6}})\s*;", css)
        require(len(values) == 1 and max(bytes.fromhex(values[0])) <= 0x66,
                f"--{name} is text on a cream background and needs dark ink")
    return css


def recover_style(wire):
    """Read a complete model-authored file string from a malformed tool envelope."""
    decoder = json.JSONDecoder()
    recovered = []
    for event in events(wire):
        if event["event"] != "response" or event["status"] != 200:
            continue
        reply = json.loads(event["body"])["choices"][0]
        if reply["finish_reason"] != "stop":
            continue
        text = reply["message"]["content"]
        for match in re.finditer(r'"content"\s*:\s*(?=")', text):
            try:
                payload, _ = decoder.raw_decode(text[match.end():])
                style = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if isinstance(style, dict) and set(style) == {"data", "landmark", "css"}:
                recovered.append((payload, style, sha(event["body"].encode("utf-8"))))
    require(len(recovered) == 1, "expected exactly one complete style file in the provider replies")
    return recovered[0]


def build(base, source, edition, style=None):
    art = (base / "art.js").read_text(encoding="utf-8")
    start = art.index("  function drawScene(")
    end = art.index("  function shade(", start)
    authored = source.read_bytes().decode("utf-8")
    require(authored.lstrip().startswith("function drawScene(canvas, scene, cast)"),
            "start your source with function drawScene(canvas, scene, cast)")
    if style is not None:
        drawing, css = style["data"], style["css"]
        require(isinstance(css, str) and css.strip() and "</style" not in css.lower(),
                "provide stylesheet content without a closing HTML tag")
        authored = apply_source_replacements(authored, {"replacements": [
            {"old": next(line for line in authored.splitlines() if "var data = INKY_ART[scene.id]" in line),
             "new": drawing, "count": 1},
            {"old": next(line for line in authored.splitlines() if "if (data.landmarkName) drawLandmark" in line),
             "new": style["landmark"], "count": 1},
        ]})
    composed = apply_source_replacements(art, {"replacements": [
        {"old": art[start:end], "new": authored + "\n", "count": 1},
    ]})
    edition.mkdir(exist_ok=True)
    for name in ("index.html", "player.js", "world-data.js", "balance.json"):
        shutil.copyfile(base / name, edition / name)
    if style is not None:
        markup = (base / "index.html").read_text(encoding="utf-8")
        page = apply_source_replacements(markup, {"replacements": [
            {"old": "</head>", "new": f"<style>\n{css}\n</style>\n</head>", "count": 1},
        ]})
        (edition / "index.html").write_bytes(page.encode("utf-8"))
    (edition / "art.js").write_bytes(composed.encode("utf-8"))


def verify(base, source, output, captures, node, chrome, style=None):
    build(base, source, output, style)
    subprocess.run([str(node), "--check", str(output / "art.js")], check=True, timeout=10)
    subprocess.run(browser_command(output, output, captures, node, chrome), check=True, timeout=40)


def publish(root, base, phase, authorship):
    preview = phase / "preview"
    out = root / EDITION
    shutil.copytree(preview, out, dirs_exist_ok=True)
    save(out / "manifest.json", {
        "storyworld": "The Hat That Held the Moon",
        "source_phase": phase.name,
        "base_manifest_sha256": sha((base / "manifest.json").read_bytes()),
        "authorship": authorship,
        "browser": read_json(phase / "browser/browser-receipt.json"),
        "artifacts": {item.name: sha(item.read_bytes()) for item in sorted(preview.iterdir())},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--gdk", type=Path)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--revise", type=Path, help="prior Goose stage containing the renderer to revise")
    parser.add_argument("--recover", type=Path, help="provider wire log for exact recovery of a style payload")
    args = parser.parse_args()
    root = args.world.resolve(strict=True)
    base = root / "export"
    node, chrome = args.node.resolve(strict=True), args.chrome.resolve(strict=True)
    if args.recover:
        if not args.revise:
            parser.error("style recovery needs --revise")
        renderer = args.revise / "workspace" / SOURCE
        replay = replay_writes(args.revise, [SOURCE], require_verified=False)
        payload, style, response_sha = recover_style(args.recover)
        phase = args.recover.parent / "recovered"
        phase.mkdir(exist_ok=True)
        (phase / "storybook-style.json").write_bytes(payload.encode("utf-8"))
        authorship = {args.revise.name: replay, "recovered_style": {
            "provider_reply_sha256": response_sha, "style_file_sha256": sha(payload.encode("utf-8")),
            "provider_trace_sha256": sha(args.recover.read_bytes()), "tool_write_completed": False,
        }}
        ink = args.verify
        if args.gdk:
            prefix = "26-storybook-ink"
            attempts = [root / prefix, *sorted(root.glob(f"{prefix}-retry-*"))]
            phase = root / f"{prefix}-retry-{len(attempts):02}" if attempts[0].exists() else attempts[0]
            ink = phase / "workspace/storybook-ink.css"
            command = [sys.executable, "-B", str(Path(__file__).resolve()), str(root), "--node", str(node),
                       "--chrome", str(chrome), "--revise", str(args.revise), "--recover", str(args.recover),
                       "--verify", str(ink)]
            brief = """Create storybook-ink.css. All FOUR variables below are text ink on a cream page.
In :root, assign dark six-digit hex colors to --parchment, --parchment-dim, --silver and --amber-soft.
Every RGB channel must be between hex 00 and 66, including on --parchment. Pale values were unreadable.
The inherited .choice-btn and #continue gradients also used these variables. Override their backgrounds
explicitly with LIGHT paper and warm gold respectively, so dark button labels remain readable. Also set
dark ink color on .cast-list .visual and .colophon. One short CSS file, one create_file action.
"""
            run_phase(phase, args.gdk.resolve(strict=True), brief, [ink.name], command,
                      max_tokens=420, iterations=1, context=8192, allow_basename=True)
            authorship[phase.name] = replay_writes(phase, [ink.name], require_verified=False)
        if ink:
            phase = ink.parent.parent
            style["css"] += "\n" + check_ink(ink)
        verify(base, renderer, phase / "preview", phase / "browser", node, chrome, style)
        if not args.verify:
            publish(root, base, phase, authorship)
        return
    if args.verify:
        phase = args.verify.parent.parent
        style = read_style(args.verify) if args.revise else None
        source = (args.revise / "workspace" / SOURCE) if args.revise else args.verify
        verify(base, source, phase / "preview", phase / "browser", node, chrome, style)
        return
    if not args.gdk:
        parser.error("authoring needs --gdk")
    prefix = "25-storybook-style" if args.revise else "24-storybook-art"
    phases = [root / prefix, *sorted(root.glob(f"{prefix}-retry-*"))]
    phase = root / f"{prefix}-retry-{len(phases):02}" if phases[0].exists() else phases[0]
    source = phase / "workspace" / (STYLE if args.revise else SOURCE)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), str(root),
               "--node", str(node), "--chrome", str(chrome), "--verify", str(source)]
    if args.revise:
        original = args.revise / "workspace" / SOURCE
        brief = """Use create_file to write storybook-finish.txt as plain text with three sections. Put a line
containing only --- between each section, no other tags. Keep your file under 1100 characters.
First section is JavaScript for your drawScene. Read the array INKY_ART[scene.id]. Its indices 0 to 7
are sky, ground, light, x, y, scale, landmark, actors. Define the object var data consumed by your drawing.
Use these exact keys: skyColor, groundColor, lightColor, xPercent, yPercent, scalePercent, landmarkName, actorIds.
Use a warm paper sky color. All scene coordinates, landmarks and actors must use the actual array.
Second section is one JavaScript drawLandmark(ctx, data.landmarkName, W, H, scale) call.
The scale argument needs a multiplier, and data.scalePercent is a percentage.
Third section is short CSS with cream paper on body and .panel, and ink colors for the headings and
variables --parchment, --parchment-dim, --silver, --amber-soft. Use your own palette.
One create_file tool call, plain text contents.
"""
        command.extend(("--revise", str(args.revise.resolve(strict=True))))
    else:
        original, brief = source, BRIEF
    run_phase(phase, args.gdk.resolve(strict=True), brief, [source.name], command,
              max_tokens=900 if args.revise else 3600, iterations=1 if args.revise else 2,
              context=8192, allow_basename=True, compact_writes=True)
    authorship = replay_writes(phase, [source.name], require_verified=False)
    browser = phase / "browser/browser-receipt.json"
    preview = phase / "preview"
    if not authorship["solver_verified"]:
        verify(base, original, preview, browser.parent, node, chrome, read_style(source) if args.revise else None)
    authorship = {phase.name: authorship}
    if args.revise:
        authorship[args.revise.name] = replay_writes(args.revise, [SOURCE], require_verified=False)
    publish(root, base, phase, authorship)


if __name__ == "__main__":
    main()
