"""Per-archetype HI Pearson breakdown — RSSM/Dreamer thin wrapper.

Imports RSSMWorldModel so torch.load can deserialize the checkpoint, then
delegates everything else to the AR-LSTM `per_archetype_diag.py` machinery —
the eval pipeline is model-agnostic once the model class is importable.

Usage:
    python baselines/rssm/per_archetype_rssm.py \\
        --ckpt /path/to/rssm_*_epoch_10_object.ckpt \\
        --hdf5 /home/lthil/.stable_worldmodel/scenario4_test_lewm.h5 \\
        --label RSSM_te --seq-lens 1 10 \\
        --out-csv eval_results/rssm_s4/per_archetype.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_WM_ROOT))

# Critical: imports RSSMWorldModel into the namespace so that pickle/torch.load
# can find it when deserializing the saved object.
from baselines.rssm.model import RSSMWorldModel  # noqa: F401, E402

# Delegate the full eval pipeline to the AR-LSTM script — it's model-agnostic.
from baselines.ar_lstm.per_archetype_diag import main  # noqa: E402


if __name__ == "__main__":
    main()
