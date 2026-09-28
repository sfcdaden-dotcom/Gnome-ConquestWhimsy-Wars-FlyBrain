# Fly brain

An experiment: a CPU player for Gnome Conquest driven by a simulated fruit fly
brain. One fly per player; the gnomes are its limbs.

**Status:** the brain loads and runs. It is not wired into the game yet.

## What's here

| Path | What it is |
|---|---|
| `data/Completeness_783.csv` | Every neuron in the adult *Drosophila* brain (138,639), FlyWire public release v783 |
| `data/Connectivity_783.parquet` | Every connection between them (15.1M connections, 54.5M synapses), signed excitatory/inhibitory |
| `vendor/shiu2024/` | The leaky integrate-and-fire whole-brain model from Shiu et al., *Nature* 2024, vendored unmodified (MIT). `UPSTREAM_COMMIT` pins the source |
| `smoke_test.py` | Loads the connectome, stimulates the sugar-taste neurons, prints what fires downstream |
| `requirements.txt` | Python dependencies |

Source: <https://github.com/philshiu/Drosophila_brain_model>. The connectome
comes from the FlyWire project (<https://flywire.ai>); credit FlyWire and
Dorkenwald et al. 2024 / Schlegel et al. 2024 if this ships anywhere public.

## Setup

```bash
python3 -m venv flybrain/.venv
flybrain/.venv/bin/pip install -r flybrain/requirements.txt
flybrain/.venv/bin/python flybrain/smoke_test.py
```

Expected output, roughly:

```
Connectome: 138,639 neurons, 15,091,983 connections, 54,492,922 synapses
Sugar neurons found in v783: 20/21
Simulated 200 ms of fly brain in ~20s
337 neurons spiked; 317 of them downstream of the stimulus
```

The smoke test uses Brian2's pure-numpy backend so no C++ compiler is needed.
Installing a compiler and switching `prefs.codegen.target` to `"cython"` is
much faster.

## Performance note

About 20 seconds per 200 ms of brain time, on the full brain, with the numpy
backend. That is far too slow for a CPU turn, which is why the plan is to:

1. Pick the sensory inputs and motor/drive outputs the game needs.
2. Extract just the subcircuit connecting them (a few thousand neurons).
3. Simulate that subcircuit in TypeScript inside the game.

## Plan

1. `src/engine/ai/flyBrain.ts`: a stub brain that outputs three drives
   (hunger → EXPAND, fear → DEFEND/SURVIVE, aggression → PRESSURE/FINISH) and
   feeds them into the existing objective layer as a new `'fly'` difficulty.
2. Mushroom-body learning: reward on garden capture and won fights, punishment
   on gnome deaths and Home attacks, with learned weights kept in `AiMemory`.
3. Replace the stub with the extracted connectome subcircuit.
