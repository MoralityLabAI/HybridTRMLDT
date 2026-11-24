# The cost of context — Pocket Goblin Yellowpaper

This yellowpaper documents the Pocket Goblin Field Tests creative retrieval pilot, Inky Mini's two illustrated browser storyworlds (*The Hat That Held the Moon* and *The City That Lost Its Night Bell*), turn/TPS/prompt-in/completion-out telemetry across the `16K` / `24K` / `32K` long-context ladder, and the matched head-to-head replication on Bonsai 2 27B (`Q4_0` unrotated ternary).

- Pages 1–4 cover the four cold matched calls of the creative retrieval pilot ([upstream report](../../MCP/demos/RESULTS.md)).
- Page 5 presents cropped headless-Chrome Canvas captures from Inky Mini's 20-scene picture-book fable and 30-encounter 2-qubit Hilbert storyworld.
- Page 6 reports exact provider turns, uncached/cached prompt tokens in, completion tokens out, and prefill/decode `tok/s` across storyworld stages and `16K`–`32K` cold planning turns.
- Page 7 compares Inky Mini (`IQ4_NL`, 2 loops) against Bonsai 2 27B (`Q4_0` unrotated) on the identical `16K` Hilbert Storyworld + Canvas benchmark ([comparison report](../../MCP/results/mcp05/HILBERT_STORYWORLD_CANVAS_COMPARISON.md)).
- Page 8 summarizes methodology and pins SHA-256 receipts for all artifacts and screenshots in `measured.json`.

Regenerate the PDF with Python, Pillow, and ReportLab (default fonts read from `C:/Windows/Fonts`, override via `YELLOWPAPER_FONT_DIR`):

```sh
python reports/pocket-goblin-yellowpaper/render.py
```
