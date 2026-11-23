---
name: inky-web-game
description: Create original, self-contained Canvas games in Goose when every playable source byte must come from Inky and be independently replayable.
---

# Inky web game

Design the player verbs, success and loss paths, visible feedback, and one distinctive procedural visual motif before writing. Keep simulation state independent of drawing. Use a complete standalone HTML document with inline CSS, JavaScript, and Canvas art. Return the finished document beginning with its doctype and ending with the closing html tag. Keep the scope compact enough to finish the entire document.

The brief supplies requirements and test hooks. Inky authors the layout, wording, rules, state transitions, and every visual. Write all playable code and artwork in the reply. Use no base project, downloaded files, library, external URL, or asset generation service. A current task may deliberately start from an empty directory.

Expose only a read-only gameState() snapshot for verification. Ensure the same keyboard or visible controls that a player uses are enough to finish the game. Make locked choices visibly distinct, make all outcomes reachable, and make restart reset the actual state. Favor a clear layout with one readable objective and controls near the playfield.

When revising from browser feedback, return the complete replacement document. The host extracts the original HTML bytes from the provider reply and tests the page on loopback with network access blocked. Pass receipts depend on browser observations. Reported or assumed behavior inside the reply is insufficient.

Workflow reference: [OpenAI Game Studio](https://github.com/openai/plugins/blob/main/plugins/game-studio/skills/game-studio/SKILL.md) and [OpenAI Game Playtest](https://github.com/openai/plugins/blob/main/plugins/game-studio/skills/game-playtest/SKILL.md). These inform the design and verification process. The game source is generated separately.
