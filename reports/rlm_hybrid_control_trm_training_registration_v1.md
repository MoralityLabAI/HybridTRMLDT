# RLM Hybrid ControlTRM Training Registration v1

The deterministic long-context corpus and ControlTRM training protocol were frozen before training. The corpus contains 96 train, 32 calibration, and 24 evaluation tasks across four equally represented families. Every transcript has exactly 640 lines and split group IDs have zero overlap.

Three gym ControlTRM analogues will train on CPU with seeds `211`, `223`, and `227`, latent width 48, four tied recurrent visits, 200 fixed steps, batch size 16, AdamW learning rate `0.008`, gradient clipping at `1.0`, and checkpoints every 50 steps. The final-step checkpoint is selected without evaluation-dependent early stopping.

Evaluation task proposals may be materialized for later replay, but evaluation accuracy is not computed or used before the final hybrid protocol is registered. These models are gym recurrent proposers, not the official TinyRecursiveModels implementation.
