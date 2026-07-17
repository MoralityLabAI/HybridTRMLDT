# Measured Qwen Stalk Bridge v1

This sidecar records the first preregistered measured-model bridge into the controller mesh. It does not modify
the frozen Fable ZIP.

The source is a target-blind Qwen3.5-0.8B layer-23 rank-one restriction graph backed by 16 activation-chunk hashes
and separate construction/validation measurements. It is joined to each controller stability sheaf through a
Kronecker-sum external product, avoiding arbitrary pairing between Qwen prompts and controller episodes.

The combined predictor reaches `rho=+0.7523`, compared with `+0.5688` for controller-only spectra and `+0.6402`
for categorical labels. Its gain over controller-only spectra has a positive paired lower bound. The registered
bridge nevertheless fails: the categorical interval crosses zero, top-16 uplift is lower than both baselines, and
typed Qwen-weight shuffles average `rho=+0.7582`, yielding N0 `p=0.9380`.

The external-product feature transform is useful, but the exact measured Qwen restrictions have no demonstrated
incremental value. This is a qualified negative result and does not validate Qwen-informed controller routing.

Full math, provenance, and receipts are in `docs/controller_mesh_measured_stalk_bridge.md` and
`reports/checkpoint_controller_mesh_measured_stalk_bridge_v1.md`. Independent external audit is pending.
