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
for name, file in {"Body": "segoeui.ttf", "Bold": "segoeuib.ttf", "Display": "georgia.ttf",
                   "Mono": "consola.ttf"}.items():
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
for name, receipt in DATA["screenshots"].items():
    assert hashlib.sha256((ROOT / "assets" / name).read_bytes()).hexdigest() == receipt["sha256"]

W, H = 612, 792
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
        c.setLineWidth(.65)
    draw = c.roundRect if radius else c.rect
    args = (x, y, w, h, radius) if radius else (x, y, w, h)
    draw(*args, fill=1, stroke=int(stroke is not None))

def line(x, y, xx, yy, color=RAIL, weight=.7):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(weight)
    c.line(x, y, xx, yy)

def label(s, x, y, size=10, font="Body", color=INK, align="left"):
    c.setFont(font, size)
    c.setFillColor(HexColor(color))
    func = {"left": c.drawString, "center": c.drawCentredString, "right": c.drawRightString}[align]
    func(x, y, s)

def copy(s, x, top, w, size=11, leading=16, color=INK, font="Body", floor=0):
    style = ParagraphStyle("body", fontName=font, fontSize=size, leading=leading, textColor=HexColor(color))
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
        cropped = original.crop(box)
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
    line(MARGIN, 740, W - MARGIN, 740, fg, .6)
    label("POCKET GOBLIN / YELLOWPAPER", MARGIN, 754, 9, "Mono", fg)
    label(section.upper(), W - MARGIN, 754, 9, "Mono", fg, "right")
    line(MARGIN, 43, W - MARGIN, 43, fg, .55)
    label("FIELD TESTS   /   23 NOV 2026", MARGIN, 27, 8, "Mono", fg)
    label(f"{number:02d} / 05", W - MARGIN, 27, 8, "Mono", fg, "right")
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
    c.setLineWidth(.7)
    c.setStrokeColor(HexColor(outline))
    c.setFillColor(HexColor(color))
    c.drawPath(path, fill=1, stroke=1)

def tile(x, y, w, h, colors):
    front, top, side = colors
    dx, dy = 20, 13
    polygon([(x,y+h),(x+dx,y+h+dy),(x+w+dx,y+h+dy),(x+w,y+h)], top, NAVY2)
    polygon([(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)], side, NAVY2)
    polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)], front, NAVY2)

def arrow(points, color=AQUA, weight=1.7):
    for a, b in zip(points, points[1:]):
        line(*a, *b, color, weight)
    ax, ay = points[-2]
    bx, by = points[-1]
    theta = math.atan2(by - ay, bx - ax)
    for angle in (theta + .53, theta - .53):
        line(bx, by, bx - math.cos(angle) * 8, by - math.sin(angle) * 8, color, weight)

def heading(title, subtitle, *, color=INK, secondary=MUTED):
    label(title, MARGIN, 685, 40, "Display", color)
    copy(subtitle, MARGIN, 661, SPAN, color=secondary, size=11.5, leading=16, floor=615)

# Cover
page(1, "Creative retrieval pilot", AMBER)
label("The cost of", MARGIN - 3, 657, 65, "Display")
label("context", MARGIN - 3, 578, 81, "Display")
copy("Pocket Goblin puts Inky Mini and a recurrent MCP router to work on a Canvas action RPG and an illustrated branching story.", MARGIN, 548, 480, size=15, leading=21, floor=493)
rect(MARGIN, 393, SPAN, 91, INK, radius=5)
for i, (num, caption) in enumerate([(f"{ratio('lantern'):.2f}×", "RPG attempt speedup"),
                                    (f"{ratio('tide'):.2f}×", "Story attempt speedup"),
                                    ("76–79%", "Fewer input tokens")]):
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
copy("Four cold matched calls across two public development tasks. The speedups include retrieval and GDK request time. The playable story came from two separate verifier-guided revisions.", MARGIN, 124, SPAN, size=11, leading=15, floor=59)
c.showPage()

# Architecture
page(2, "System layout", INK, WHITE)
heading("From shards to Inky", "Pocket Goblin assembles the evidence before an answer-only Goose GDK call. The arrows show how source text moves through the observed assisted path.", color=WHITE, secondary=WHITE)
rect(44, 315, 524, 283, NAVY2, radius=7)
for x in range(72, 540, 22):
    for y in range(342, 577, 22):
        c.setFillColor(HexColor(GRID))
        c.circle(x, y, .62, stroke=0, fill=1)
for x, label_text, tint in [(64, "MCP SOURCES", AQUA), (244, "CONTROLLER MESH", LILAC), (428, "MODEL SOLVER", AMBER)]:
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
arrow([(194,471),(232,471)])
arrow([(369,471),(414,471)])
label("6 chunks", 393, 488, 8, "Mono", AQUA, "center")
copy("Exact vocabulary checks and pinned repository shards", 65, 406, 136, size=9.5, leading=12.5, color=WHITE, floor=360)
copy("Local brief search plus selected federation snippets", 241, 406, 135, size=9.5, leading=12.5, color=WHITE, floor=360)
copy("One structured reply sent to the game contract verifier", 423, 406, 126, size=9.5, leading=12.5, color=WHITE, floor=360)
line(63, 356, 550, 356, GRID)
label("TRACE ARCHIVE", 65, 333, 9, "Mono", AQUA)
label("Tool requests, source receipts, model wire, and verifier outcomes", 165, 333, 9, "Body", WHITE)
chip("WIDE CONTROL", 44, 276, PEACH, INK)
copy("The wide path read the full local project slice and queried four named shards, selecting 13 chunks. It used the same Q4 solver, 448-token completion cap and project verifier.", 44, 263, SPAN, size=11, leading=16, color=WHITE, floor=205)
chip("WHAT THE MESH DID", 44, 177, LILAC, INK)
copy("The learned SetTRM ranked the federation shards. Workspace search supplied the briefs. Both assisted queries selected LoopedTransformers and DataGoblin. The diagram depicts logical roles in the run, including local retrieval and verification.", 44, 163, SPAN, size=11, leading=16, color=WHITE, floor=88)
c.showPage()

# Measured timing
page(3, "Raw timings")
heading("Where the time went", "Measured attempt time, including retrieval. Each bar uses the same seconds scale. Values come from the local GDK and provider receipts.")
segments = [("Retrieval", TEAL), ("Prefill", AMBER), ("Decode", "#9967CA"), ("Other", "#8B9BA0")]
for i,(name,color) in enumerate(segments):
    x=44+i*132
    rect(x, 593, 11, 11, color)
    label(name, x+17, 593, 10)
scale=2.25
x0=144
for tick in (0, 40, 80, 120, 160):
    x=x0+tick*scale
    line(x, 428, x, 568, RAIL, .55)
    label(str(tick), x, 411, 8.5, "Mono", MUTED, "center")
for game, name, y0 in [("lantern","RPG",530),("tide","STORY",462)]:
    for arm, desc, y in [("assisted","Assisted",y0),("wide","Wide",y0-25)]:
        row=lookup(game,arm)
        label(name if arm=="assisted" else "",44,y+3,9,"Mono")
        label(desc,93,y+3,9)
        p=row["prefill_ms"]/1000
        d=row["decode_ms"]/1000
        values=(row["retrieval_seconds"],p,d,row["gdk_seconds"]-p-d)
        x=x0
        for value,(_,color) in zip(values,segments):
            rect(x,y,value*scale,13,color)
            x += value*scale
        label(f"{total(row):.3f} s",x+7,y+2.5,9,"Bold")
label("Seconds", 568, 411, 8.5, "Mono", MUTED, "right")
rect(44, 335, SPAN, 57, INK, radius=5)
for x, game, name in [(56,"lantern","RPG"),(320,"tide","STORY")]:
    a,b=lookup(game,"assisted"),lookup(game,"wide")
    label(f"{name}    {ratio(game):.2f}× overall",x,370,13,"Bold",WHITE)
    label(f"{total(b)-total(a):.3f} s saved   ·   {b['prefill_ms']/a['prefill_ms']:.2f}× prefill",x,350,10,"Body",AQUA)

label("Raw receipt values in seconds",44,303,18,"Display")
columns=[44,73,194,266,331,398,473,567]
for x, name in zip(columns, ["#", "Trial", "Input", "Retrieve", "Prefill", "Decode", "GDK", "Total"]):
    label(name,x,277,9,"Bold",INK,"left" if x<=73 else "right")
line(44,269,568,269,INK)
for i,row in enumerate(DATA["primary"]):
    y=250-i*25
    if i%2==0:
        rect(44,y-7,524,21,"#F9EBC1")
    label(str(i+1),44,y,9,"Mono")
    label(("RPG" if row["project"]=="lantern" else "Story")+" / "+row["arm"],73,y,9)
    vals=[f"{row['input_tokens']:,}",f"{row['retrieval_seconds']:.3f}",f"{row['prefill_ms']/1000:.3f}",f"{row['decode_ms']/1000:.3f}",f"{row['gdk_seconds']:.3f}",f"{total(row):.3f}"]
    for x,val in zip(columns[2:],vals):
        label(val,x,y,9.5,"Mono",INK,"right")
copy("Total = retrieval + GDK wall time. Prefill and decode are measured inside GDK time. Other is the GDK residual. Speedup = wide total ÷ assisted total. Calculations use the unrounded receipts. These are attempt times, including failed attempts.",44,132,SPAN,size=10,leading=14,floor=58)
c.showPage()

# Game captures
page(4, "Game captures")
heading("Two playable worlds", "Browser captures of the verified variants. The host games supply the controls and procedural Canvas artwork. Inky supplied patrol data and branching scene text.")
photo("lantern-assisted.png",(54,216,1226,781),44,397,524,216)
chip("8 / 8 CHECKS", 478, 365, AQUA, INK)
label("Lantern Below",44,371,20,"Display")
copy("Saltglass Patrol passed its encounter contract on the first assisted attempt. A legal-control playthrough collected three embers and opened the gate.",44,358,SPAN,size=10.5,leading=14,floor=318)
photo("tide-assisted.png",(54,215,1211,616),44,130,524,183)
chip("7 / 7 AFTER FEEDBACK", 449, 101, LILAC, INK)
label("The Hour Between Tides",44,106,18,"Display")
label("Drowned Observatory · browser-verified ending",44,85,9.5,"Body",MUTED)
c.showPage()

# Methods and provenance
page(5, "Methods and receipts")
heading("What the pilot supports", "A local development comparison covering one RPG encounter task and one branching story task. The four primary trials ran assisted, wide, assisted, wide.")
rect(44,482,524,125,"#F9E8AF",radius=5)
label("The outputs",58,582,19,"Display")
copy("The wide RPG reply omitted JSON keys. The wide story reply spent its output cap on control markers. The assisted story met five of seven checks. Its initial brief omitted the verifier's 280-character limit, and the scene ran to 293 characters.",58,567,495,size=10.5,leading=14.5,floor=513)
story=total(lookup("tide","assisted"))+sum(map(total,DATA["followups"]))
label(f"Story time through its verified revision    {story:.3f} s across three attempts",58,499,10.3,"Bold",TEAL)
label("Interpretation",44,449,22,"Display")
copy("The study measures the combined packet policy, source selection and token budget. Isolating the recurrent ranker's contribution requires a matched short-packet control. Two tasks and one primary call per arm give no estimate of run-to-run variance. Story feedback calls are reported separately from the primary speedups.",44,434,SPAN,size=11,leading=15.5,floor=365)
label("Reproduce and audit",44,335,22,"Display")
copy("Pinned runtime: RTX 5080 Laptop GPU, Inkling Mini IQ4_NL, 10 CPU MoE layers, and a 100,096-token slot. All four primary requests were cold with zero cached input tokens and a 448-token output cap.",44,320,SPAN,size=10.5,leading=14.5,floor=274)
copy("Six complete Inky exchanges across two project groups include both MCP wire journals, selected chunk IDs, provider time and verifier outcomes. Two candidates passed. The export labels the traces as public development data and leaves training admission unset.",44,261,SPAN,size=10.5,leading=14.5,floor=211)
for y,title,value in [
    (185,"SOURCE COMMIT",DATA["plan"]["project_commit"]),
    (151,"INDEX REVISION",DATA["plan"]["index_revision"]),
    (117,"NORMALIZED TRACE SHA-256",DATA["runtime"]["normalized_records_sha256"]),
    (83,"ARCHIVE MANIFEST SHA-256",DATA["archive_sha256"]),
]:
    label(title,44,y,8.2,"Mono",TEAL)
    label(value,44,y-14,9,"Mono")
c.showPage()
c.save()
print(FILE)
