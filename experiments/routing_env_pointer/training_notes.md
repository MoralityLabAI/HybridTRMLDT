# Routing Training Notes

Task: environment pointer routing from prompt text to local environment ID.

Models:

- `ldt`: explicit token lattice router. Tokens refine candidate environment sets.
- `trm`: dependency-light lexical TRM analogue, mirroring Tesseract's TF-IDF router objective.
- `hybrid`: LDT candidate filtering followed by TRM scoring inside the surviving candidate set.

Data:

- data root: `D:\Research_Engine\tesseract_persistent\data`
- max records per env: `80`
- train examples: `402`
- test examples: `173`
- envs: `alphabet_sort, arc_challenge, arc_easy, gsm8k, intellect_3_logic, intellect_3_math, mbpp, wiki_search`

This run does not train QLoRA adapters or use VPD. It isolates router behavior.

- `ldt` accuracy=0.538 correct=93/173 abstained=15 avg_candidates=0.00
- `trm` accuracy=0.815 correct=141/173 abstained=0 avg_candidates=0.00
- `hybrid` accuracy=0.815 correct=141/173 abstained=0 avg_candidates=2.09
