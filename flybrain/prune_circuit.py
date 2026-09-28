"""Cut the Fly's game circuit down from the neurons extract_circuit.py saw fire.

Keeps a neuron only if it lies on a short path — at most MAX_PATH synapses —
from some input channel to some output channel, through neurons that fired,
and keeps only connections of at least MIN_WEIGHT synapses (either sign).
Then re-simulates the pruned circuit with the original Brian2 model under the
same stimulation patterns, and compares its output rates with the whole brain.

    flybrain/.venv/bin/python flybrain/prune_circuit.py            # try several cuts
    flybrain/.venv/bin/python flybrain/prune_circuit.py --export 3 --min-weight 3

Writes circuit/validation.csv, and with --export circuit/flyCircuit.json:
  neurons, flywireIds, channels (local indices), params, and pre/post/weight
  (signed synapse counts), plus the whole-brain response table it reproduces.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from brian2 import Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses, mV, ms, prefs, seed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "shiu2024"))
from extract_circuit import INPUTS, OUTPUTS, SEEDS, T_RUN, patterns  # noqa: E402
from model import default_params as P  # noqa: E402

prefs.codegen.target = "cython"
CIRCUIT = HERE / "circuit"


def distances(n: int, adj: list[list[int]], sources: list[int]) -> np.ndarray:
    dist = np.full(n, 10**9)
    q = deque(sources)
    for s in sources:
        dist[s] = 0
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] > dist[u] + 1:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def prune(con: pd.DataFrame, active: np.ndarray, chans: dict, max_path: int) -> np.ndarray:
    """`con` is already filtered to the connections being kept."""
    local = {g: l for l, g in enumerate(active)}
    e = con[con.Presynaptic_Index.isin(local) & con.Postsynaptic_Index.isin(local)]
    pre = e.Presynaptic_Index.map(local).values
    post = e.Postsynaptic_Index.map(local).values
    fwd = [[] for _ in active]
    bwd = [[] for _ in active]
    for a, b in zip(pre, post):
        fwd[a].append(b)
        bwd[b].append(a)
    d_in = distances(len(active), fwd, [local[i] for c in INPUTS for i in chans[c]])
    d_out = distances(len(active), bwd, [local[i] for c in OUTPUTS for i in chans[c]])
    keep = active[d_in + d_out <= max_path]
    channel_neurons = [i for c in INPUTS + OUTPUTS for i in chans[c]]
    return np.union1d(keep, channel_neurons)


def simulate(con: pd.DataFrame, keep: np.ndarray, chans: dict) -> pd.DataFrame:
    local = {g: l for l, g in enumerate(keep)}
    e = con[con.Presynaptic_Index.isin(local) & con.Postsynaptic_Index.isin(local)]
    neu = NeuronGroup(len(keep), P["eqs"], method="linear", threshold=P["eq_th"], reset=P["eq_rst"],
                      refractory="rfc", namespace=P)
    neu.v = P["v_0"]
    neu.g = 0 * mV
    neu.rfc = P["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"])
    syn.connect(i=e.Presynaptic_Index.map(local).values, j=e.Postsynaptic_Index.map(local).values)
    syn.w = e["Excitatory x Connectivity"].values * P["w_syn"]
    stim = [local[i] for c in INPUTS for i in chans[c]]
    src = PoissonGroup(len(stim), rates=0 * Hz)
    drive = Synapses(src, neu, on_pre="v += w_poi", namespace={"w_poi": P["w_syn"] * P["f_poi"]})
    drive.connect(i=np.arange(len(stim)), j=np.array(stim))
    neu.rfc[np.array(stim)] = 0 * ms
    mon = SpikeMonitor(neu)
    net = Network(neu, syn, src, drive, mon)
    net.store()
    rows = []
    for pat in patterns():
        for s in SEEDS:
            net.restore()
            seed(s)
            src.rates = np.concatenate([np.full(len(chans[c]), pat[c]) for c in INPUTS]) * Hz
            net.run(T_RUN)
            counts = np.bincount(np.asarray(mon.i[:]), minlength=len(keep))
            row = {**{f"in_{c}": pat[c] for c in INPUTS}, "seed": s}
            for o in OUTPUTS:
                row[f"out_{o}"] = counts[[local[i] for i in chans[o]]].mean() / float(T_RUN / (1000 * ms))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=int, help="write the circuit cut at this max path length")
    ap.add_argument("--min-weight", type=int, nargs="*", default=[1], help="drop connections weaker than this")
    args = ap.parse_args()

    all_con = pd.read_parquet(HERE / "data" / "Connectivity_783.parquet")
    comp = pd.read_csv(HERE / "data" / "Completeness_783.csv", index_col=0)
    active = np.load(CIRCUIT / "active.npy")
    chans = json.loads((CIRCUIT / "channels.json").read_text())
    full = pd.read_csv(CIRCUIT / "response.csv")
    outs = [f"out_{o}" for o in OUTPUTS]

    report = []
    cuts = [(p, w) for p in ([args.export] if args.export else [3, 4]) for w in args.min_weight]
    for max_path, min_w in cuts:
        con = all_con[all_con.Connectivity >= min_w]
        keep = prune(con, active, chans, max_path)
        n_edges = int((con.Presynaptic_Index.isin(keep) & con.Postsynaptic_Index.isin(keep)).sum())
        cut = simulate(con, keep, chans)
        err = (cut[outs].values - full[outs].values)
        line = {"max_path": max_path, "min_weight": min_w, "neurons": len(keep), "connections": n_edges,
                "mean_abs_err_hz": float(np.abs(err).mean()), "max_abs_err_hz": float(np.abs(err).max())}
        report.append(line)
        print(line, flush=True)

        if args.export:
            local = {g: l for l, g in enumerate(keep)}
            e = con[con.Presynaptic_Index.isin(local) & con.Postsynaptic_Index.isin(local)]
            circuit = {
                "source": "FlyWire v783 connectome (Dorkenwald et al. 2024; Schlegel et al. 2024), annotations "
                          "v2.1.0, model after Shiu et al. 2024; cut by flybrain/extract_circuit.py + "
                          f"prune_circuit.py --export {max_path} --min-weight {min_w}",
                "params": {"v0": -52, "vReset": -52, "vThreshold": -45, "tMembrane": 20, "tSynapse": 5,
                           "tRefractory": 2.2, "tDelay": 1.8, "wSynapse": 0.275, "wInput": 0.275 * 250},
                "neurons": len(keep),
                "flywireIds": [str(comp.index[g]) for g in keep],
                "channels": {c: [local[i] for i in chans[c]] for c in INPUTS + OUTPUTS},
                "pre": [local[i] for i in e.Presynaptic_Index],
                "post": [local[i] for i in e.Postsynaptic_Index],
                "weight": e["Excitatory x Connectivity"].astype(int).tolist(),
                "validation": line,
                "response": full.groupby([f"in_{c}" for c in INPUTS], sort=False)[outs].mean().reset_index()
                .to_dict(orient="records"),
            }
            (CIRCUIT / "flyCircuit.json").write_text(json.dumps(circuit, separators=(",", ":")))
            cut.to_csv(CIRCUIT / "response_pruned.csv", index=False)
    pd.DataFrame(report).to_csv(CIRCUIT / "validation.csv", index=False)


if __name__ == "__main__":
    main()
