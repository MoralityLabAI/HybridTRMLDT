# The cost of context

This yellowpaper documents the Pocket Goblin Field Tests creative retrieval pilot. The raw measurements come from the six Inky exchanges in the campaign archive. The four matched calls drive the comparison table, and the two verifier-guided revisions contribute only to the story time through validation. The upstream [benchmark report](../../MCP/demos/RESULTS.md) describes the tasks and outcomes.

The artwork images are cropped for layout from the archived Chrome captures. Their source filenames and hashes appear in `measured.json` alongside the timings and artifact identifiers. The architecture drawing is generated as PDF vectors.

Regenerate the PDF with Python, Pillow and ReportLab. The default font directory uses the Windows Georgia, Segoe UI, and Consolas TrueType files. Set `YELLOWPAPER_FONT_DIR` to another directory containing those fonts if needed.

```sh
python reports/pocket-goblin-yellowpaper/render.py
```
