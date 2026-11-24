"""Render the Pocket Goblin yellowpaper from frozen local measurements."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "measured.json").read_text(encoding="utf-8"))
FONT_DIR = Path(os.environ.get("YELLOWPAPER_FONT_DIR", "C:/Windows/Fonts"))
for name, file in {
    "Body": "segoeui.ttf",
    "Bold": "segoeuib.ttf",
    "Display": "georgia.ttf",
    "Mono": "consola.ttf",
}.items():
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
for name, receipt in DATA["screenshots"].items():
    assert hashlib.sha256((ROOT / "assets" / name).read_bytes()).hexdigest() == receipt["sha256"]

W, H = 612, 792
TOTAL_PAGES = 8
PAPER = "#FFF7DA"
AMBER = "#F5C947"
INK = "#142431"
MUTED = "#435257"
TEAL = "#11766D"
AQUA = "#8BE1CC"
LILAC = "#CDBDF2"
RUST = "#AE3F28"
PEACH = "#FCAA7C"
RAIL = "#D2B66A"
WHITE = "#FFFBEA"
NAVY2 = "#203749"
GRID = "#2D4759"
MARGIN = 44
SPAN = W - 2 * MARGIN

FILE = ROOT / "output/pdf/Pocket-Goblin-Yellowpaper.pdf"
FILE.parent.mkdir(parents=True, exist_ok=True)
c = Canvas(str(FILE), pagesize=(W, H), pageCompression=1, invariant=1)
c.setTitle("Pocket Goblin | The cost of context")
c.setAuthor("Pocket Goblin research")


def rect(x, y, w, h, fill, stroke=None, radius=0):
    c.setFillColor(HexColor(fill))
    if stroke:
        c.setStrokeColor(HexColor(stroke))
        c.setLineWidth(0.65)
    draw = c.roundRect if radius else c.rect
    args = (x, y, w, h, radius) if radius else (x, y, w, h)
    draw(*args, fill=1, stroke=int(stroke is not None))


def line(x, y, xx, yy, color=RAIL, weight=0.7):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(weight)
    c.line(x, y, xx, yy)


def label(s, x, y, size=10, font="Body", color=INK, align="left"):
    c.setFont(font, size)
    c.setFillColor(HexColor(color))
    func = {"left": c.drawString, "center": c.drawCentredString, "right": c.drawRightString}[align]
    func(x, y, s)


def copy(s, x, top, w, size=11, leading=16, color=INK, font="Body", floor=0):
    style = ParagraphStyle(
        "body", fontName=font, fontSize=size, leading=leading, textColor=HexColor(color)
    )
    p = Paragraph(escape(s), style)
    _, height = p.wrap(w, H)
    assert top - height >= floor, f"Paragraph at {top} crosses {floor}: {s}"
    p.drawOn(c, x, top - height)
    return top - height


def chip(s, x, y, fill=INK, text=WHITE):
    width = pdfmetrics.stringWidth(s, "Mono", 9) + 18
    rect(x, y, width, 21, fill, radius=3)
    label(s, x + 9, y + 6.5, 9, "Mono", text)


def photo(name, box, x, y, w, h):
    with Image.open(ROOT / "assets" / name) as original:
        cropped = original.crop(box) if box else original.copy()
        ratio = w / h
        if cropped.width / cropped.height > ratio:
            trim = (cropped.width - cropped.height * ratio) / 2
            cropped = cropped.crop((round(trim), 0, round(cropped.width - trim), cropped.height))
        else:
            trim = (cropped.height - cropped.width / ratio) / 2
            cropped = cropped.crop((0, round(trim), cropped.width, round(cropped.height - trim)))
        c.drawImage(ImageReader(cropped), x, y, w, h)


def page(number, section, bg=PAPER, fg=INK):
    rect(0, 0, W, H, bg)
    line(MARGIN, 740, W - MARGIN, 740, fg, 0.6)
    label("POCKET GOBLIN / YELLOWPAPER", MARGIN, 754, 9, "Mono", fg)
    label(section.upper(), W - MARGIN, 754, 9, "Mono", fg, "right")
    line(MARGIN, 43, W - MARGIN, 43, fg, 0.55)
    label("FIELD TESTS   /   24 NOV 2026", MARGIN, 27, 8, "Mono", fg)
    label(f"{number:02d} / {TOTAL_PAGES:02d}", W - MARGIN, 27, 8, "Mono", fg, "right")
    c.bookmarkPage(f"p{number}")
    c.addOutlineEntry(section, f"p{number}", 0, False)


def lookup(game, arm):
    return next(r for r in DATA["primary"] if r["project"] == game and r["arm"] == arm)


def total(r):
    return r["retrieval_seconds"] + r["gdk_seconds"]


def ratio(game):
    return total(lookup(game, "wide")) / total(lookup(game, "assisted"))


def polygon(points, color, outline=INK):
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()
    c.setLineWidth(0.7)
    c.setStrokeColor(HexColor(outline))
    c.setFillColor(HexColor(color))
    c.drawPath(path, fill=1, stroke=1)


def tile(x, y, w, h, colors):
    front, top, side = colors
    dx, dy = 20, 13
    polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)], top, NAVY2)
    polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], side, NAVY2)
    polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], front, NAVY2)


def arrow(points, color=AQUA, weight=1.7):
    for a, b in zip(points, points[1:]):
        line(*a, *b, color, weight)
    ax, ay = points[-2]
    bx, by = points[-1]
    theta = math.atan2(by - ay, bx - ax)
    for angle in (theta + 0.53, theta - 0.53):
        line(bx, by, bx - math.cos(angle) * 8, by - math.sin(angle) * 8, color, weight)


def heading(title, subtitle, *, color=INK, secondary=MUTED):
    label(title, MARGIN, 685, 38, "Display", color)
    copy(subtitle, MARGIN, 661, SPAN, color=secondary, size=11.2, leading=15.5, floor=614)


# Page 1: Cover
page(1, "Creative retrieval pilot", AMBER)
label("The cost of", MARGIN - 3, 657, 65, "Display")
label("context", MARGIN - 3, 578, 81, "Display")
copy(
    "Pocket Goblin puts Inky Mini, Bonsai 2 27B, and a recurrent MCP router to work on Canvas games and illustrated Hilbert storyworlds.",
    MARGIN,
    548,
    495,
    size=14.5,
    leading=20.5,
    floor=490,
)
rect(MARGIN, 393, SPAN, 91, INK, radius=5)
for i, (num, caption) in enumerate(
    [
        (f"{ratio('lantern'):.2f}×", "RPG attempt speedup"),
        (f"{ratio('tide'):.2f}×", "Story attempt speedup"),
        ("76–79%", "Fewer input tokens"),
    ]
):
    x = 62 + i * 174
    label(num, x, 432, 32, "Bold", AMBER if i == 0 else AQUA if i == 1 else LILAC)
    label(caption, x, 410, 10, "Body", WHITE)
    if i < 2:
        line(x + 153, 409, x + 153, 468, GRID)
photo("lantern-assisted.png", (54, 216, 907, 728), 44, 201, 258, 166)
photo("tide-assisted.png", (54, 216, 890, 616), 310, 201, 258, 166)
label("Lantern Below", 44, 179, 15, "Display")
label("The Hour Between Tides", 310, 179, 15, "Display")
label("Saltglass Patrol", 44, 160, 10, "Body")
label("Drowned Observatory", 310, 160, 10, "Body")
copy(
    "Four cold matched pilot calls plus full 20-scene and 30-encounter illustrated storyworld conveyors with browser-verified 2D Canvas artwork.",
    MARGIN,
    124,
    SPAN,
    size=11,
    leading=15,
    floor=59,
)
c.showPage()

# Page 2: Architecture
page(2, "System layout", INK, WHITE)
heading(
    "From shards to Inky",
    "Pocket Goblin assembles the evidence before an answer-only Goose GDK call. The arrows show how source text moves through the observed assisted path.",
    color=WHITE,
    secondary=WHITE,
)
rect(44, 315, 524, 283, NAVY2, radius=7)
for x in range(72, 540, 22):
    for y in range(342, 577, 22):
        c.setFillColor(HexColor(GRID))
        c.circle(x, y, 0.62, stroke=0, fill=1)
for x, label_text, tint in [
    (64, "MCP SOURCES", AQUA),
    (244, "CONTROLLER MESH", LILAC),
    (428, "MODEL SOLVER", AMBER),
]:
    label(label_text, x, 570, 9, "Mono", tint)
for y in (420, 434):
    tile(64, y, 113, 40, ("#126459", AQUA, TEAL))
tile(64, 448, 113, 56, ("#126459", AQUA, TEAL))
tile(241, 433, 108, 77, ("#7655A3", LILAC, "#50396F"))
tile(423, 433, 103, 77, ("#855514", AMBER, "#673B16"))
label("MCP", 76, 483, 15, "Bold", WHITE)
label("Workspace +", 76, 470, 9.5, "Body", WHITE)
label("federation", 76, 457, 9.5, "Body", WHITE)
label("SetTRM", 253, 480, 16, "Bold", WHITE)
label("Recurrent", 253, 464, 9.5, "Body", WHITE)
label("shard ranker", 253, 450, 9.5, "Body", WHITE)
label("Inky Mini", 433, 480, 15, "Bold", WHITE)
label("Local Q4", 433, 464, 9.5, "Body", WHITE)
label("Goose GDK", 433, 450, 9.5, "Body", WHITE)
arrow([(194, 471), (232, 471)])
arrow([(369, 471), (414, 471)])
label("6 chunks", 393, 488, 8, "Mono", AQUA, "center")
copy("Exact vocabulary checks and pinned repository shards", 65, 406, 136, size=9.5, leading=12.5, color=WHITE, floor=360)
copy("Local brief search plus selected federation snippets", 241, 406, 135, size=9.5, leading=12.5, color=WHITE, floor=360)
copy("One structured reply sent to the game contract verifier", 423, 406, 126, size=9.5, leading=12.5, color=WHITE, floor=360)
line(63, 356, 550, 356, GRID)
label("TRACE ARCHIVE", 65, 333, 9, "Mono", AQUA)
label("Tool requests, source receipts, model wire, and verifier outcomes", 165, 333, 9, "Body", WHITE)
chip("WIDE CONTROL", 44, 276, PEACH, INK)
copy(
    "The wide path read the full local project slice and queried four named shards, selecting 13 chunks. It used the same Q4 solver, 448-token completion cap and project verifier.",
    44,
    263,
    SPAN,
    size=11,
    leading=16,
    color=WHITE,
    floor=205,
)
chip("WHAT THE MESH DID", 44, 177, LILAC, INK)
copy(
    "The learned SetTRM ranked the federation shards. Workspace search supplied the briefs. Both assisted queries selected LoopedTransformers and DataGoblin. Long storyworlds extend this into SWMD continuity cards.",
    44,
    163,
    SPAN,
    size=11,
    leading=16,
    color=WHITE,
    floor=88,
)
c.showPage()

# Page 3: Measured pilot timing
page(3, "Raw timings")
heading(
    "Where the time went",
    "Measured attempt time, including retrieval. Each bar uses the same seconds scale. Values come from the local GDK and provider receipts.",
)
segments = [("Retrieval", TEAL), ("Prefill", AMBER), ("Decode", "#9967CA"), ("Other", "#8B9BA0")]
for i, (name, color) in enumerate(segments):
    x = 44 + i * 132
    rect(x, 593, 11, 11, color)
    label(name, x + 17, 593, 10)
scale = 2.25
x0 = 144
for tick in (0, 40, 80, 120, 160):
    x = x0 + tick * scale
    line(x, 428, x, 568, RAIL, 0.55)
    label(str(tick), x, 411, 8.5, "Mono", MUTED, "center")
for game, name, y0 in [("lantern", "RPG", 530), ("tide", "STORY", 462)]:
    for arm, desc, y in [("assisted", "Assisted", y0), ("wide", "Wide", y0 - 25)]:
        row = lookup(game, arm)
        label(name if arm == "assisted" else "", 44, y + 3, 9, "Mono")
        label(desc, 93, y + 3, 9)
        p = row["prefill_ms"] / 1000
        d = row["decode_ms"] / 1000
        values = (row["retrieval_seconds"], p, d, row["gdk_seconds"] - p - d)
        x = x0
        for value, (_, color) in zip(values, segments):
            rect(x, y, value * scale, 13, color)
            x += value * scale
        label(f"{total(row):.3f} s", x + 7, y + 2.5, 9, "Bold")
label("Seconds", 568, 411, 8.5, "Mono", MUTED, "right")
rect(44, 335, SPAN, 57, INK, radius=5)
for x, game, name in [(56, "lantern", "RPG"), (320, "tide", "STORY")]:
    a, b = lookup(game, "assisted"), lookup(game, "wide")
    label(f"{name}    {ratio(game):.2f}× overall", x, 370, 13, "Bold", WHITE)
    label(
        f"{total(b) - total(a):.3f} s saved   ·   {b['prefill_ms'] / a['prefill_ms']:.2f}× prefill",
        x,
        350,
        10,
        "Body",
        AQUA,
    )

label("Raw receipt values in seconds", 44, 303, 18, "Display")
columns = [44, 73, 194, 266, 331, 398, 473, 567]
for x, name in zip(columns, ["#", "Trial", "Input", "Retrieve", "Prefill", "Decode", "GDK", "Total"]):
    label(name, x, 277, 9, "Bold", INK, "left" if x <= 73 else "right")
line(44, 269, 568, 269, INK)
for i, row in enumerate(DATA["primary"]):
    y = 250 - i * 25
    if i % 2 == 0:
        rect(44, y - 7, 524, 21, "#F9EBC1")
    label(str(i + 1), 44, y, 9, "Mono")
    label(("RPG" if row["project"] == "lantern" else "Story") + " / " + row["arm"], 73, y, 9)
    vals = [
        f"{row['input_tokens']:,}",
        f"{row['retrieval_seconds']:.3f}",
        f"{row['prefill_ms'] / 1000:.3f}",
        f"{row['decode_ms'] / 1000:.3f}",
        f"{row['gdk_seconds']:.3f}",
        f"{total(row):.3f}",
    ]
    for x, val in zip(columns[2:], vals):
        label(val, x, y, 9.5, "Mono", INK, "right")
copy(
    "Total = retrieval + GDK wall time. Prefill and decode are measured inside GDK time. Other is the GDK residual. Speedup = wide total ÷ assisted total. Calculations use the unrounded receipts. These are attempt times, including failed attempts.",
    44,
    132,
    SPAN,
    size=10,
    leading=14,
    floor=58,
)
c.showPage()

# Page 4: Pilot game captures
page(4, "Game captures")
heading(
    "Two playable worlds",
    "Browser captures of the verified pilot variants. The host games supply the controls and procedural Canvas artwork. Inky supplied patrol data and branching scene text.",
)
photo("lantern-assisted.png", (54, 216, 1226, 781), 44, 397, 524, 216)
chip("8 / 8 CHECKS", 478, 365, AQUA, INK)
label("Lantern Below", 44, 371, 20, "Display")
copy(
    "Saltglass Patrol passed its encounter contract on the first assisted attempt. A legal-control playthrough collected three embers and opened the gate.",
    44,
    358,
    SPAN,
    size=10.5,
    leading=14,
    floor=318,
)
photo("tide-assisted.png", (54, 215, 1211, 616), 44, 130, 524, 183)
chip("7 / 7 AFTER FEEDBACK", 449, 101, LILAC, INK)
label("The Hour Between Tides", 44, 106, 18, "Display")
label("Drowned Observatory · browser-verified ending", 44, 85, 9.5, "Body", MUTED)
c.showPage()

# Page 5: Illustrated Inky Storyworlds Gallery
SW = DATA["storyworlds"]
ch = SW["cat_hat"]
hi = SW["hilbert_inky"]
hb = SW["hilbert_bonsai"]

page(5, "Illustrated storyworlds")
heading(
    "Two Inky storyworlds",
    "Multi-stage Goose conveyor fables authored locally by Inky Mini, each shipping model-written 2D Canvas artwork and a headless-Chrome verified browser player.",
)
photo("cat-hat-start.png", (50, 232, 1220, 930), 44, 448, 256, 156)
photo("cat-hat-share.png", (50, 232, 1220, 930), 312, 448, 256, 156)
label("The Hat That Held the Moon", 44, 424, 18, "Display")
chip("20 SCENES · 4 ENDINGS", 408, 419, AQUA, INK)
copy(
    f"Picture-book Canvas edition: {ch['scenes']} scenes, {ch['choices']} choices, {ch['reactions']} village reactions and {ch['endings']} endings across {ch['totals']['turns']} timed turns ({ch['totals']['prompt_in']:,} prompt tokens in, {ch['totals']['completion_out']:,} out). Left: opening alley. Right: meadow share ending.",
    44,
    412,
    SPAN,
    size=9.7,
    leading=13.2,
    floor=360,
)
line(44, 353, 568, 353, RAIL, 0.6)
photo("hilbert-inky-start.png", (150, 20, 1125, 660), 44, 185, 256, 156)
photo("hilbert-inky-secret.png", (150, 20, 1125, 660), 312, 185, 256, 156)
label(hi["title"], 44, 161, 18, "Display")
chip("32 / 32 CANVAS HASHES", 408, 156, LILAC, INK)
copy(
    f"30-encounter 2-qubit Hilbert fable ({hi['scene_words']:,} words, {hi['options']} options, {hi['landmarks']} landmarks, {hi['mean_colors']:.1f} mean quantized colors). Left: Act I bell tower and ket readout. Right: secret ending The Bell as a Question ({hi['secret_question_pct']:.2f}% Born access, gate-swap verified).",
    44,
    149,
    SPAN,
    size=9.7,
    leading=13.2,
    floor=58,
)
c.showPage()

# Page 6: Storyworld Conveyor Turns, TPS, and Prompt In/Out Statistics
page(6, "Conveyor telemetry")
heading(
    "Turns, TPS and tokens",
    "Measured provider turns, prompt tokens in, completion tokens out, and prefill/decode tok/s across Inky's storyworlds and the 16K–32K planning ladder.",
)

comb_turns = ch["totals"]["turns"] + hi["totals"]["http_200_timed_turns"]
comb_in = ch["totals"]["prompt_in"] + hi["totals"]["total_prompt_tokens_in"]
comb_un = ch["totals"]["uncached_in"] + hi["totals"]["total_uncached_prompt_tokens"]
comb_ca = ch["totals"]["cached_in"] + hi["totals"]["total_cached_prompt_tokens"]
comb_out = ch["totals"]["completion_out"] + hi["totals"]["total_completion_tokens_out"]
comb_p_s = (ch["totals"]["uncached_in"] / ch["totals"]["prefill_tps"]) + (
    hi["totals"]["total_uncached_prompt_tokens"] / hi["totals"]["aggregate_prefill_tps"]
)
comb_d_s = (ch["totals"]["completion_out"] / ch["totals"]["decode_tps"]) + (
    hi["totals"]["total_completion_tokens_out"] / hi["totals"]["aggregate_decode_tps"]
)
comb_ptps = comb_un / comb_p_s
comb_dtps = comb_out / comb_d_s

rect(44, 544, SPAN, 60, INK, radius=5)
label(f"{comb_turns}", 58, 576, 25, "Bold", AMBER)
label("Inky HTTP-200 turns", 58, 556, 9.2, "Body", WHITE)
line(196, 554, 196, 594, GRID)
label(f"{comb_in / 1000:.1f}K / {comb_out / 1000:.1f}K", 210, 576, 21, "Bold", AQUA)
label("Prompt in / completion out", 210, 556, 9.2, "Body", WHITE)
line(402, 554, 402, 594, GRID)
label(f"{comb_ptps:.1f} / {comb_dtps:.1f} t/s", 416, 576, 21, "Bold", LILAC)
label("Prefill / decode tok/s", 416, 556, 9.2, "Body", WHITE)

label("Inky Mini storyworld and Canvas phase breakdown", 44, 518, 15.5, "Display")
t1_cols = [44, 248, 314, 372, 432, 498, 568]
for x, htxt in zip(t1_cols, ["Campaign / Phase group", "Turns", "Prompt in", "Cached", "Out", "Prefill", "Decode"]):
    label(htxt, x, 497, 8.6, "Bold", INK, "left" if x == 44 else "right")
line(44, 490, 568, 490, INK)

phase_rows = [
    ("Hat Moon / Story & SWMD cards", ch["story_cards"]),
    ("Hat Moon / Canvas & player", ch["canvas_edition"]),
    ("Night Bell / 16K long exposures", hi["long_exposures"]),
    ("Night Bell / 30-scene conveyor", hi["bounded_conveyor"]),
    ("Night Bell / 4-file Canvas edition", hi["canvas_stages"]),
    (
        "Combined Inky storyworlds",
        {
            "turns": comb_turns,
            "prompt_in": comb_in,
            "cached_in": comb_ca,
            "completion_out": comb_out,
            "prefill_tps": comb_ptps,
            "decode_tps": comb_dtps,
        },
    ),
]
for i, (rname, rdata) in enumerate(phase_rows):
    y = 474 - i * 19.5
    is_tot = i == len(phase_rows) - 1
    if is_tot:
        line(44, y + 12, 568, y + 12, INK, 0.55)
        rect(44, y - 5, 524, 17, "#F5DF98")
    elif i % 2 == 0:
        rect(44, y - 5, 524, 17, "#F9EBC1")
    fnt = "Bold" if is_tot else "Body"
    mfnt = "Bold" if is_tot else "Mono"
    label(rname, 44, y, 8.6, fnt)
    vals = [
        f"{rdata['turns']}",
        f"{rdata['prompt_in']:,}",
        f"{rdata['cached_in']:,}",
        f"{rdata['completion_out']:,}",
        f"{rdata['prefill_tps']:.1f}",
        f"{rdata['decode_tps']:.1f}",
    ]
    for x, val in zip(t1_cols[1:], vals):
        label(val, x, y, 8.6, mfnt, INK, "right")

label("Cold long-context planning ladder (16K–32K)", 44, 334, 15.5, "Display")
t2_cols = [44, 248, 298, 358, 418, 482, 568]
for x, htxt in zip(t2_cols, ["Arm / Model", "Input", "Out", "Prefill", "Decode", "Wall s", "Direct plan"]):
    label(htxt, x, 313, 8.6, "Bold", INK, "left" if x == 44 else "right")
line(44, 306, 568, 306, INK)

ladder_rows = []
for arm_k, short_k in [("16384", "16K"), ("24576", "24K"), ("32768", "32K")]:
    iv = hi["sized_planning"][arm_k]
    bv = hb["sized_planning"][arm_k]
    ladder_rows.append((f"{short_k} / Inky Mini (IQ4_NL, 2L)", iv, "Repair"))
    ladder_rows.append((f"{short_k} / Bonsai 2 27B (Q4_0)", bv, "Authored"))

for i, (mname, r, verdict_txt) in enumerate(ladder_rows):
    y = 290 - i * 19.5
    if i % 2 == 0:
        rect(44, y - 5, 524, 17, "#F9EBC1")
    label(mname, 44, y, 8.5, "Body")
    vals = [
        f"{r['prompt_tokens']:,}",
        f"{r['completion_tokens']:,}",
        f"{r['prefill_tps']:.1f}",
        f"{r['decode_tps']:.1f}",
        f"{r['wall_seconds']:.1f}",
        verdict_txt,
    ]
    for j, (x, val) in enumerate(zip(t2_cols[1:], vals)):
        col = TEAL if (j == 5 and verdict_txt == "Authored") else RUST if (j == 5) else INK
        label(val, x, y, 8.5, "Bold" if j == 5 else "Mono", col, "right")

copy(
    "Prefill tok/s divides uncached prompt tokens by provider prompt_ms; decode tok/s divides completion tokens by predicted_ms. Every Hilbert benchmark turn verified exact /apply-template + /tokenize preflight parity. At 16K and 24K, Bonsai 2 27B Q4_0 completed cold planning 30.7× and 10.7× faster than Inky Mini.",
    44,
    172,
    SPAN,
    size=9.4,
    leading=13,
    floor=55,
)
c.showPage()

# Page 7: Matched Hilbert Storyworld + Canvas Comparison (Inky Mini vs. Bonsai 2 27B Q4_0)
page(7, "Matched Hilbert comparison")
heading(
    "Inky Mini vs. Bonsai 2 27B",
    "Matched 16K Hilbert storyworld and procedural 2D Canvas benchmark under the identical federation packet, Goose tool contract, and browser verifier.",
)
photo("hilbert-bonsai-start.png", (10, 8, 675, 458), 44, 426, 256, 174)
photo("hilbert-bonsai-secret.png", (10, 8, 675, 458), 312, 426, 256, 174)
label(hb["title"], 44, 398, 18, "Display")
chip("BONSAI 2 27B · Q4_0", 412, 393, AMBER, INK)
copy(
    f"Converted from ternary PTQ1_0 to unrotated Q4_0 (0 trit involution mismatches across 26.87B folded weights), Bonsai 2 27B passed {hb['first_attempt_passes']}/{hb['total_stages']} conveyor stages on attempt 1 and verified {hb['distinct_hashes']}/{hb['visited_scenes']} distinct Canvas hashes (left: Act I tower; right: secret ending end_mix).",
    44,
    386,
    SPAN,
    size=9.6,
    leading=13.2,
    floor=340,
)

label("Matched 16K conveyor + Canvas scorecard", 44, 322, 15.5, "Display")
t3_cols = [44, 418, 568]
for x, htxt in zip(t3_cols, ["Metric", "Inky Mini (IQ4_NL)", "Bonsai 2 27B (Q4_0)"]):
    label(htxt, x, 301, 8.8, "Bold", INK, "left" if x == 44 else "right")
line(44, 294, 568, 294, INK)

it = hi["totals"]
bt = hb["totals"]
score_rows = [
    ("Cold 16K / 24K / 32K direct plans", "0 / 3", "3 / 3"),
    ("First-attempt stage passes (16K)", f"{hi['first_attempt_passes']} / {hi['total_stages']}", f"{hb['first_attempt_passes']} / {hb['total_stages']}"),
    ("Timed HTTP-200 Goose turns", f"{it['http_200_timed_turns']}", f"{bt['http_200_timed_turns']}"),
    ("Total prompt tokens in (uncached)", f"{it['total_prompt_tokens_in']:,} ({it['total_uncached_prompt_tokens']:,})", f"{bt['total_prompt_tokens_in']:,} ({bt['total_uncached_prompt_tokens']:,})"),
    ("Total completion tokens out", f"{it['total_completion_tokens_out']:,}", f"{bt['total_completion_tokens_out']:,}"),
    ("Aggregate prefill / decode tok/s", f"{it['aggregate_prefill_tps']:.2f} / {it['aggregate_decode_tps']:.2f}", f"{bt['aggregate_prefill_tps']:.2f} / {bt['aggregate_decode_tps']:.2f}"),
    ("Cumulative provider wall time", f"{it['total_provider_wall_seconds']:.1f} s", f"{bt['total_provider_wall_seconds']:.1f} s"),
    ("Storyworld scenes / words / endings", f"{hi['encounters']} / {hi['scene_words']:,} / 5", f"{hb['encounters']} / {hb['scene_words']:,} / 5"),
    ("Distinct Canvas hashes / mean colors", f"{hi['distinct_hashes']}/{hi['visited_scenes']} · {hi['mean_colors']:.1f}", f"{hb['distinct_hashes']}/{hb['visited_scenes']} · {hb['mean_colors']:.1f}"),
    ("Swap-sensitive secret endings verified", "2 / 2 (verified)", "2 / 2 (verified)"),
]
for i, (mtxt, iv_txt, bv_txt) in enumerate(score_rows):
    y = 278 - i * 19.5
    if i % 2 == 0:
        rect(44, y - 5, 524, 17, "#F9EBC1")
    label(mtxt, 44, y, 8.6, "Body")
    label(iv_txt, t3_cols[1], y, 8.6, "Mono", INK, "right")
    label(bv_txt, t3_cols[2], y, 8.6, "Bold" if i in (0, 1, 5, 6) else "Mono", TEAL if i in (0, 1) else INK, "right")

c.showPage()

# Page 8: Methods and provenance
page(8, "Methods and receipts")
heading(
    "What the suite supports",
    "Synthesis across the four-call creative retrieval pilot, two long-form Inky storyworlds, and the matched Bonsai 2 27B Q4_0 Hilbert replication.",
)
rect(44, 492, 524, 114, "#F9E8AF", radius=5)
label("Context discipline vs. model scale", 58, 584, 18, "Display")
copy(
    "Flooding Inky Mini (2 loops) with 16K–32K raw retrieval packets stalled direct plan authoring 3/3 times, yet bounded Pocket Goblin SWMD cards (<=3.5K tokens) let the same small model ship 20-scene and 30-encounter worlds with 32/32 distinct Canvas hashes. Unrotated Bonsai 2 27B Q4_0 recovered 3/3 cold long-context plans and first-pass conveyor reliability.",
    58,
    569,
    495,
    size=9.9,
    leading=13.8,
    floor=512,
)
story = total(lookup("tide", "assisted")) + sum(map(total, DATA["followups"]))
label(
    f"Pilot story verified in {story:.1f}s (3 calls)   ·   30-scene Hilbert worlds verified in headless Chrome",
    58,
    503,
    9.6,
    "Bold",
    TEAL,
)

label("Interpretation", 44, 463, 20, "Display")
copy(
    "The pilot and Hilbert benchmarks jointly show why Pocket Goblin separates retrieval routing from bounded stage cards. Small looped checkpoints are highly sensitive to unrouted prefill width and multi-file output caps, whereas single-file Goose create_file stages with deterministic geometry, algebra, and DOM verifiers converge reliably.",
    44,
    448,
    SPAN,
    size=10.3,
    leading=14.5,
    floor=378,
)

label("Reproduce and audit", 44, 352, 20, "Display")
copy(
    "Hardware: RTX 5080 Laptop GPU (16 GB VRAM), deterministic seed 523, temperature 0. Every file in the exported browser editions was replayed from recorded Goose create_file tool calls, validated by SWMD/SweepWeave topology checks, and exercised in headless Chrome across all public and non-commuting secret routes.",
    44,
    337,
    SPAN,
    size=10.0,
    leading=14.0,
    floor=255,
)

for y, title, value in [
    (225, "PILOT SOURCE COMMIT", DATA["plan"]["project_commit"]),
    (191, "FEDERATION INDEX REVISION", DATA["plan"]["index_revision"]),
    (157, "HILBERT PACKET SHA-256", SW["reference_packet_sha256"]),
    (123, "INKY HILBERT + CANVAS RECEIPT SHA-256", hi["receipt_sha256"]),
    (89, "BONSAI 2 27B Q4_0 HILBERT + CANVAS RECEIPT SHA-256", hb["receipt_sha256"]),
]:
    label(title, 44, y, 8.0, "Mono", TEAL)
    label(value, 44, y - 13, 8.8, "Mono")

c.showPage()
c.save()
print(FILE)
