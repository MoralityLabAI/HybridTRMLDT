# Field Tests · Creative retrieval pilot

Inky authored playable data for two original Canvas projects. Lantern Below asks for an alternate enemy patrol. The Hour Between Tides asks for a clue-gated observatory passage and original ending prose. The host-built projects supply the controls, scene engine and procedural artwork. Each model variant is judged against the same runtime contract for its project.

The benchmark used the existing Inkling Q4 checkpoint on the RTX with 10 CPU MoE layers and a 100,096-token slot. The four fresh requests alternated assisted, wide, assisted, wide. Each had the same 448-token completion allowance and answer-only mode. Both project briefs are public development inputs.

| Call | Input | Selected chunks | Prefill | Retrieval + GDK | Contract result |
| --- | ---: | ---: | ---: | ---: | --- |
| Lantern, assisted | 2,448 | 6 | 27.27 s | 36.31 s | 8/8, verified |
| Tide, wide | 11,921 | 13 | 135.70 s | 159.83 s | Invalid JSON |
| Tide, assisted | 2,823 | 6 | 32.66 s | 49.05 s | 5/7, rejected |
| Lantern, wide | 11,389 | 13 | 124.08 s | 131.05 s | Invalid JSON |

The assisted packets reduced input by 78.5% on Lantern and 76.3% on Tide. The Lantern patrol passed all eight checks, including the walkable spawn and separation rules. Its wide reply supplied spawn positions but omitted the required keys before two numeric values, leaving invalid JSON. The wide Tide reply exhausted its completion budget on control markers. The assisted Tide reply met the gate, artwork, prose and action requirements but used 293 characters for the interior scene. The verifier’s limit was 280 characters, while the original brief asked for short prose without specifying a number.

The assisted calls used the live SetTRM router. It selected LoopedTransformers and DataGoblin in both cases, alongside relevant local project briefs from workspace search. The wide calls queried four named repositories and paged through the full local project evidence. These paired pilot results show a token and latency difference for the complete packet policies. They do not separate the learned ranker’s effect from packet size, workspace retrieval or source selection.

## Playable model variants

The unmodified assisted Lantern candidate is available as **Saltglass Patrol**. It passed 8/8 and completed a simulated run from spawn to the open gate using only movement, attack, dash and interaction inputs.

The first Tide feedback round brought the description within the length limit but dropped the required lens and bell clues. The second revision kept both clues while meeting the length limit. It passed 7/7 and is available as **Drowned Observatory**. Both revisions are separate from the four matched calls. The playable asset bytes match the final verified candidate.

## Reproducibility

The source and initial runner are pinned at `eb02038fc8f1ea95266ba4aee44463f306215543`. The local campaign keeps each request, reply, both MCP wire journals, selected chunk digests, GDK preflight, provider timings and verifier verdict. The derived training records group both treatments and any revision under the same project root. They retain unsuccessful outputs and mark public development provenance.

The export contains six Inky exchanges across two project groups, with two verified candidates. The normalized `training-records.jsonl` SHA-256 is `490d0361f8c2ac4f05baa3c5cced6948e2e7f648c1922d1eaca4af2ee2626e9f`. Its companion receipt binds the original artifact index and Q4 runtime receipt.
