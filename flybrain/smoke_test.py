"""Smoke test: load the FlyWire v783 connectome and run one short simulation.

Stimulates the fly's sugar-taste neurons ("food!") for a brief trial and
reports which neurons respond most strongly. This proves the data and the
Brian2 simulator are installed and wired together; it is not game code yet.

    flybrain/.venv/bin/python flybrain/smoke_test.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
from brian2 import ms, prefs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "shiu2024"))
from model import default_params, run_trial  # noqa: E402

prefs.codegen.target = "numpy"  # no C++ compiler needed; slower but portable

COMP = HERE / "data" / "Completeness_783.csv"
CON = HERE / "data" / "Connectivity_783.parquet"

# Sugar-sensing gustatory neurons from the upstream example notebook. They are
# FlyWire v630 IDs; most survive unchanged into v783, and the ones that were
# renumbered are skipped (the script reports how many were found).
SUGAR_783 = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]


def main() -> None:
    comp = pd.read_csv(COMP, index_col=0)
    con = pd.read_parquet(CON)
    print(f"Connectome: {len(comp):,} neurons, {len(con):,} connections, "
          f"{int(con['Connectivity'].sum()):,} synapses")

    flyid2i = {f: i for i, f in enumerate(comp.index)}
    i2flyid = {i: f for f, i in flyid2i.items()}
    exc = [flyid2i[f] for f in SUGAR_783 if f in flyid2i]
    print(f"Sugar neurons found in v783: {len(exc)}/{len(SUGAR_783)}")
    if not exc:
        sys.exit("No stimulus neurons found; check the ID list against this data version.")

    params = dict(default_params, t_run=200 * ms)
    t0 = time.time()
    # run_trial returns {brian index: spike times} for every neuron.
    spikes = {i: t for i, t in run_trial(exc, [], [], COMP, CON, params).items() if len(t)}
    print(f"Simulated 200 ms of fly brain in {time.time() - t0:.1f}s")

    rates = sorted(((len(t) / 0.2, i2flyid[i]) for i, t in spikes.items()), reverse=True)
    stim = set(exc)
    downstream = [(r, f) for r, f in rates if flyid2i[f] not in stim]
    print(f"{len(rates):,} neurons spiked; {len(downstream):,} of them downstream of the stimulus")
    for r, f in downstream[:10]:
        print(f"  {f}  {r:6.1f} Hz")


if __name__ == "__main__":
    main()
