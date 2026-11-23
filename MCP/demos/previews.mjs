import { drawMarsh } from "./lantern/art.mjs";
import { createGame } from "./lantern/game.mjs";
import { drawScene } from "./tide/art.mjs";

const lantern = document.querySelector('[data-preview="lantern"]');
drawMarsh(lantern.getContext("2d"), createGame(), 0);
const tide = document.querySelector('[data-preview="tide"]');
drawScene(tide.getContext("2d"), "quay");

const release = await fetch(new URL("./variants/catalog.json", import.meta.url)).catch(() => null);
if (release?.ok) {
  const variants = await release.json();
  const links = variants.filter(item => ["lantern","tide"].includes(item.project)
    && ["assisted","wide"].includes(item.arm)).map(item => {
    const link = document.createElement("a");
    link.className = "control";
    link.href = "./" + item.project + "/?variant=" + item.arm;
    link.textContent = item.label + " →";
    return link;
  });
  if (links.length) {
    document.getElementById("variant-links").replaceChildren(...links);
    document.getElementById("verified-variants").hidden = false;
  }
}
