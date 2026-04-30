# Program

Build a mostly-binary inference artifact and a faster compatible kernel implementation
for `prism-ml/Bonsai-1.7B-gguf`.

Current backend note: this Python/Numpy scorer is a scaffold smoke test. Reward
launch should use the pinned Prism/llama.cpp scorer path so miner-editable Python
cannot own timing or correctness.

Hard gates:

- shape valid
- non-binary rescue fraction `<= 10%`
- reproducible artifact hash on identical reruns
- heldout quality floor
- runtime output must match the validator reference path

Survivors are Pareto-ranked on:

- `heldout_ppl` (lower is better)
- `speedup`

New best acceptance uses a `0.02` heldout-PPL resolution in nats. Runtime speedup
decides between submissions that remain inside the incumbent's PPL band.
