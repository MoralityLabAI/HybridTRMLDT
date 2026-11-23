import { drawMarsh } from "./lantern/art.mjs";
import { createGame } from "./lantern/game.mjs";
import { drawScene } from "./tide/art.mjs";

const lantern = document.querySelector('[data-preview="lantern"]');
drawMarsh(lantern.getContext("2d"), createGame(), 0);
const tide = document.querySelector('[data-preview="tide"]');
drawScene(tide.getContext("2d"), "quay");
