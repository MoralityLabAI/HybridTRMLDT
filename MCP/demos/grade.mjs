import { readFileSync } from "node:fs";
import { assessPassage, assessPatrol } from "./variants.mjs";

const [project, file] = process.argv.slice(2);
const grade = {lantern:assessPatrol,tide:assessPassage}[project];
if (!grade || !file) throw new Error("Usage: node grade.mjs lantern|tide candidate.json");
console.log(JSON.stringify(grade(JSON.parse(readFileSync(file,"utf8")))));
