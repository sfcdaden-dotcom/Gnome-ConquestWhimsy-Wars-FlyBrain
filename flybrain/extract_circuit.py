"""Cut the Fly's game circuit out of the whole-brain connectome.

The game drives four sensory channels and reads four motor/command channels:

    input    neurons (FlyWire v783, annotations v2.1.0)      game meaning
    sugar    LB3 gustatory, sugar/water subclass             food on offer
    bitter   LB1a / LB1b gustatory, bitter subclass          losses, scarcity
    loom     LPLC2 + LC4 visual projection (looming)         enemies closing in
    object   LC9 + LC31 visual projection (object detection) a target in reach

    output   neurons                                         urge
    feed     MN9 (proboscis motor neuron)                    feed: harvest, plant, grab gardens
    escape   DNp01, the Giant Fiber                          escape: defend, hold, avoid fights
    approach DNp09 (object approach / forward walking)       approach: advance, attack
    retreat  MDN / DNp50 (moonwalker, backward walking)      retreat: pull back

It runs the Shiu et al. (2024) whole-brain leaky integrate-and-fire model on
all 138,639 neurons under a spread of stimulation patterns and records every
neuron that fired in ANY of them. In this model a neuron that never spikes has
no effect on anything, so for these stimuli that set reproduces the full
brain; new patterns that recruit neurons outside it would differ.

Writes:
  circuit/response.csv       output rates (Hz) for every stimulation pattern
  circuit/active.npy         whole-brain indices of every neuron that fired
  circuit/channels.json      whole-brain indices of each channel's neurons

prune_circuit.py then cuts the (much smaller) game circuit from these.

    flybrain/.venv/bin/python flybrain/extract_circuit.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from brian2 import Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses, mV, ms, prefs, seed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "shiu2024"))
from model import default_params as P  # noqa: E402

prefs.codegen.target = "cython"

DATA = HERE / "data"
OUT = HERE / "circuit"
T_RUN = 300 * ms
SEEDS = [1, 2]
LEVELS = [50, 100, 200]  # Hz
MN9_V630 = "720575940660219265"  # the Shiu et al. MN9, unchanged in v783


def channel_ids(ann: pd.DataFrame) -> dict[str, list[str]]:
    ct, hb, sub = ann.cell_type, ann.hemibrain_type, ann.cell_sub_class
    mn9_type = ann.loc[ann.root_id == MN9_V630, "cell_type"].iloc[0]
    mn9 = ann.loc[(ann.root_id == MN9_V630) | ((ct == mn9_type) & (mn9_type != "") & (ann.super_class == "motor")), "root_id"]
    return {
        "sugar": ann.loc[(ct == "LB3") & (sub == "sugar/water"), "root_id"].tolist(),
        "bitter": ann.loc[ct.isin(["LB1a", "LB1b"]) & (sub == "bitter"), "root_id"].tolist(),
        "loom": ann.loc[hb.isin(["LPLC2", "LC4"]), "root_id"].tolist(),
        # The strongest visual inputs to DNp09 within two synapses (see README).
        "object": ann.loc[hb.isin(["LC9", "LC31"]), "root_id"].tolist(),
        "feed": mn9.tolist(),
        "escape": ann.loc[(ct == "DNp01"), "root_id"].tolist(),
        "approach": ann.loc[(hb == "DNp09"), "root_id"].tolist(),
        "retreat": ann.loc[(hb == "MDN"), "root_id"].tolist(),
    }


INPUTS = ["sugar", "bitter", "loom", "object"]
OUTPUTS = ["feed", "escape", "approach", "retreat"]


def patterns() -> list[dict[str, int]]:
    out = [{c: 0 for c in INPUTS}]
    for c in INPUTS:
        for r in LEVELS:
            out.append({**{k: 0 for k in INPUTS}, c: r})
    for a, b in itertools.combinations(INPUTS, 2):
        out.append({**{k: 0 for k in INPUTS}, a: 100, b: 100})
    out.append({c: 100 for c in INPUTS})
    out.append({c: 200 for c in INPUTS})
    return out


def main() -> None:
    t0 = time.time()
    comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(DATA / "Connectivity_783.parquet")
    ann = pd.read_csv(DATA / "flywire_annotations_v2.1.0.tsv", sep="\t", dtype=str).fillna("")
    idx = {str(f): i for i, f in enumerate(comp.index)}
    chans = {k: [idx[f] for f in v if f in idx] for k, v in channel_ids(ann).items()}
    for k, v in chans.items():
        print(f"{k:9s} {len(v)} neurons")
        if not v:
            sys.exit(f"channel {k} is empty")

    # The whole brain, exactly as model.create_model builds it.
    neu = NeuronGroup(len(comp), P["eqs"], method="linear", threshold=P["eq_th"], reset=P["eq_rst"],
                      refractory="rfc", namespace=P, name="neurons")
    neu.v = P["v_0"]
    neu.g = 0 * mV
    neu.rfc = P["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"], name="synapses")
    syn.connect(i=con.Presynaptic_Index.values, j=con.Postsynaptic_Index.values)
    syn.w = con["Excitatory x Connectivity"].values * P["w_syn"]

    # One Poisson source per stimulated neuron, one-to-one, strong enough that
    # every input event is a spike (as model.poi does with f_poi).
    stim = [i for c in INPUTS for i in chans[c]]
    src = PoissonGroup(len(stim), rates=0 * Hz, name="stim")
    drive = Synapses(src, neu, on_pre="v += w_poi", namespace={"w_poi": P["w_syn"] * P["f_poi"]}, name="drive")
    drive.connect(i=np.arange(len(stim)), j=np.array(stim))
    neu.rfc[np.array(stim)] = 0 * ms
    mon = SpikeMonitor(neu, name="spikes")
    net = Network(neu, syn, src, drive, mon)
    net.store()
    print(f"built in {time.time() - t0:.0f}s")

    active = np.zeros(len(comp), dtype=bool)
    rows = []
    for pat in patterns():
        for s in SEEDS:
            net.restore()
            seed(s)
            rates = np.concatenate([np.full(len(chans[c]), pat[c]) for c in INPUTS]) * Hz
            src.rates = rates
            net.run(T_RUN)
            counts = np.bincount(np.asarray(mon.i[:]), minlength=len(comp))
            active |= counts > 0
            row = {**{f"in_{c}": pat[c] for c in INPUTS}, "seed": s}
            for o in OUTPUTS:
                row[f"out_{o}"] = counts[chans[o]].mean() / float(T_RUN / (1000 * ms))
            rows.append(row)
            print(" ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    OUT.mkdir(exist_ok=True)
    resp = pd.DataFrame(rows)
    resp.to_csv(OUT / "response.csv", index=False)

    # Every neuron that ever fired, plus all channel neurons: the raw material
    # prune_circuit.py cuts the game circuit from.
    for c in INPUTS + OUTPUTS:
        active[chans[c]] = True
    np.save(OUT / "active.npy", np.flatnonzero(active))
    (OUT / "channels.json").write_text(json.dumps({c: [int(i) for i in chans[c]] for c in INPUTS + OUTPUTS}))
    print(f"{int(active.sum()):,} of {len(comp):,} neurons fired in some pattern; {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
