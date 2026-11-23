# Fully generated Inky games

Each trial starts with an empty game directory. Goose GDK presents the task brief and the local [Inky web game skill](../../../.agents/skills/inky-web-game/SKILL.md). The skill contains design and verification guidance. Inky writes the game document, including its HTML, CSS, JavaScript and procedural Canvas art.

Run the generator from the repository root. It saves the entire provider exchange, the MCP wire journal, the brief, the skill, and a receipt that records the UTF-8 byte range of the extracted game in the raw provider reply. Browser screenshots and verifier code live outside the game document.

```sh
python -m MCP.mcp05.inky_web_game --gdk /path/to/goblin-gdk --out /path/to/run
node MCP/demos/browser-smoke.mjs /path/to/chrome /path/to/captures rpg /path/to/run/rpg/attempt-01
node MCP/demos/browser-smoke.mjs /path/to/chrome /path/to/captures story /path/to/run/story/attempt-01
```

Goose's desktop and CLI use the standard `.agents/skills` layout. The Pocket Goblin GDK runner attaches the same local skill text explicitly and records it in the request. [Goose Agent Skills documentation](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/context-engineering/using-skills.md)
