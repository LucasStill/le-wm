"""counterfactual_fidelity_rssm.py — counterfactual fidelity for the RSSM/Dreamer baseline.

Mirrors `counterfactual_fidelity.py` but uses RSSM's `dynamics.observe()` /
`dynamics.img_step()` API for the rollout instead of JEPA's predictor.

The simulator-side machinery (EpisodeReplayer, CounterfactualSpec) and the
HI-probe machinery (eval_sweep.train_probe, eval_sweep.encode_observations)
are reused as-is — both are model-agnostic.

Usage:
    python counterfactual_fidelity_rssm.py \\
        --ckpts RSSM:/path/to/rssm_*_epoch_10_object.ckpt \\
        --sim_h5 /home/lthil/.stable_worldmodel/turbosens2_train_sensors.h5 \\
        --eval_h5 /home/lthil/.stable_worldmodel/turbosens2_test.h5 \\
        --out_dir eval_results/counterfactual_rssm \\
        --n_episodes 30 --horizon 100 --actions 0 1 2 3 4 5 6 \\
        --branch_fracs 0.25 0.5 0.75
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# le-wm root
sys.path.insert(0, "/home/lthil/thesis/le-wm")
import eval_sweep                          # noqa: E402

# RSSM model class needs to be importable for torch.load to deserialize.
from baselines.rssm.model import RSSMWorldModel  # noqa: F401, E402

# Simulator
sys.path.insert(0, "/home/lthil/thesis/rl_opendeck_simulator")
from scenarios.turbosens2.replay import EpisodeReplayer, CounterfactualSpec  # noqa: E402

# Reuse the differential math from the JEPA script
from counterfactual_fidelity import compute_differentials  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# RSSM rollout — replaces model.predict() / action_encoder logic.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rssm_rollout_HI(model, probe, scaler, sensors_history: np.ndarray,
                    hist_actions: np.ndarray, override_action: int,
                    horizon: int, device) -> np.ndarray:
    """Roll the RSSM forward and decode each predicted feat to HI via the probe.

    Args:
        sensors_history: (H, 11, 16) raw sensor frames before branch_t.
        hist_actions:    (H,) actual actions taken DURING history.
        override_action: action index to use for ALL future steps.
        horizon:         number of future steps to predict.

    Returns predicted HI: (horizon, 10).
    """
    H = sensors_history.shape[0]
    obs = torch.as_tensor(sensors_history.reshape(H, -1)).float().unsqueeze(0).to(device)  # (1, H, 176)
    a_ctx = torch.as_tensor(hist_actions).long().unsqueeze(0).unsqueeze(-1).to(device)     # (1, H, 1)
    a_ctx_oh = model._one_hot_actions(a_ctx)                                                # (1, H, A)

    # Encode + observe → final posterior state.
    embed = model.encoder(obs.reshape(-1, obs.shape[-1])).reshape(1, H, -1)
    is_first = torch.zeros(1, H, dtype=torch.bool, device=device)
    is_first[0, 0] = True
    post, _ = model.dynamics.observe(embed, a_ctx_oh, is_first)
    state = {k: v[:, -1] for k, v in post.items()}

    # Imagine forward with the override action repeated.
    feats = []
    a_override = torch.full((1,), int(override_action), dtype=torch.long, device=device)
    a_override_oh = F.one_hot(a_override.clamp(0, model.num_actions - 1),
                              num_classes=model.num_actions).float()  # (1, A)
    for _ in range(horizon):
        state = model.dynamics.img_step(state, a_override_oh)
        feats.append(model.dynamics.get_feat(state).squeeze(0).cpu().numpy())

    rollout = np.stack(feats, axis=0)  # (horizon, feat_dim)

    # Decode via probe (seq_len=1)
    rollout_scaled = scaler.transform(rollout)
    Xw = rollout_scaled.reshape(rollout.shape[0], 1, rollout.shape[1])
    probe.eval()
    Xw_t = torch.as_tensor(Xw, dtype=torch.float32, device=device)
    preds = probe(Xw_t).cpu().numpy()  # (horizon, 10)
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Eval — same shape as counterfactual_fidelity.evaluate_one_ckpt but with the
# RSSM rollout substituted in.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one_ckpt(ckpt_name: str, ckpt_path: Path, args, replayer,
                      tr_data, te_data, hist_window: int, episodes: list,
                      device) -> dict:
    print(f"\n{'='*60}\nLoading {ckpt_name}: {ckpt_path}")
    model = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if hasattr(model, "module"):
        model = model.module
    model.eval().to(device)

    print(f"  encoding train + test embeddings…")
    Z_tr = eval_sweep.encode_observations(model, tr_data["obs"], "sensor", device)
    Z_te = eval_sweep.encode_observations(model, te_data["obs"], "sensor", device)
    print(f"  Z_tr={Z_tr.shape}  Z_te={Z_te.shape}")

    print(f"  training task-1 probe (TransformerProbe, seq_len=1)…")
    _preds, _gt, probe, scaler = eval_sweep.train_probe(
        Z_tr, tr_data["hi"], Z_te, te_data["hi"],
        tr_data["ep_ids"], te_data["ep_ids"],
        n_outputs=10, device=device, task_name=f"{ckpt_name}_cf_probe",
        seq_len=1,
    )
    probe.eval()

    fh_lewm = h5py.File("/home/lthil/.stable_worldmodel/turbosens2_train.h5", "r")
    ep_offsets = fh_lewm["ep_offset"][:]

    results = []
    for ep_idx in episodes:
        full_ep = replayer.replay_full(int(ep_idx))
        T_ep = full_ep.T

        for frac in args.branch_fracs:
            branch_t = int(T_ep * frac)
            if branch_t < hist_window or branch_t + args.horizon >= T_ep:
                continue

            abs_start = int(ep_offsets[ep_idx]) + branch_t - hist_window
            sensors_history = fh_lewm["pixels"][abs_start : abs_start + hist_window][:]  # (H, 11, 16)
            hist_actions    = full_ep.action_idx[branch_t - hist_window : branch_t]      # (H,)

            for action in args.actions:
                spec = CounterfactualSpec(
                    branch_t=branch_t, override_action=int(action),
                    horizon=args.horizon,
                )
                cf      = replayer.counterfactual(int(ep_idx), spec)
                sim_HI  = cf.s
                sim_len = sim_HI.shape[0]

                try:
                    model_HI = rssm_rollout_HI(
                        model, probe, scaler,
                        sensors_history, hist_actions,
                        int(action), sim_len, device,
                    )
                except Exception as e:
                    print(f"    ep={ep_idx} branch_frac={frac} action={action} "
                          f"ROLLOUT FAILED: {e}")
                    continue

                rmse_per_step = np.sqrt(((sim_HI - model_HI) ** 2).mean(axis=1))
                results.append({
                    "ckpt":          ckpt_name,
                    "ep_idx":        int(ep_idx),
                    "branch_t":      branch_t,
                    "branch_frac":   frac,
                    "action":        int(action),
                    "horizon":       sim_len,
                    "sim_HI":        sim_HI.tolist(),
                    "model_HI":      model_HI.tolist(),
                    "rmse_per_step": rmse_per_step.tolist(),
                })
        print(f"  ep={ep_idx} done across {len(args.branch_fracs)} branches × "
              f"{len(args.actions)} actions")

    fh_lewm.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"ckpt": ckpt_name, "ckpt_path": str(ckpt_path), "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name:path pairs, e.g. RSSM:/path/to/ckpt")
    ap.add_argument("--sim_h5",       required=True)
    ap.add_argument("--eval_h5",      required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--n_episodes",   type=int, default=30)
    ap.add_argument("--horizon",      type=int, default=100)
    ap.add_argument("--actions",      type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--branch_fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--history",      type=int, default=16)
    ap.add_argument("--seed",         type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval data {args.eval_h5}…")
    tr_data, te_data, *_ = eval_sweep.load_dataset(args.eval_h5, encoder_type="sensor")

    print(f"Opening simulator {args.sim_h5}…")
    replayer = EpisodeReplayer(args.sim_h5)
    n_eps_total = replayer.f["ep_meta/seed"].shape[0]
    chosen = sorted(
        np.random.choice(n_eps_total, size=args.n_episodes, replace=False).tolist()
    )
    print(f"  chose {len(chosen)} episodes from {n_eps_total}")

    all_ckpt_results = []
    for spec in args.ckpts:
        if ":" not in spec:
            print(f"!!! Skipping malformed --ckpts entry: {spec}")
            continue
        name, path = spec.split(":", 1)
        path = Path(path)
        if not path.exists():
            print(f"!!! Skipping {name}: {path} does not exist")
            continue
        blob = evaluate_one_ckpt(
            name, path, args, replayer, tr_data, te_data,
            args.history, chosen, device,
        )
        all_ckpt_results.append(blob)
        (out_dir / f"results_{name}.json").write_text(json.dumps(blob, indent=2))

    diffs = compute_differentials(all_ckpt_results)
    summary = {
        "n_episodes":   args.n_episodes,
        "horizon":      args.horizon,
        "actions":      args.actions,
        "branch_fracs": args.branch_fracs,
        "ckpts":        [b["ckpt"] for b in all_ckpt_results],
        "differentials": diffs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll done. Saved per-ckpt JSONs and summary.json to {out_dir}/")
    print(f"  ckpts evaluated: {[b['ckpt'] for b in all_ckpt_results]}")
    print(f"  total rollouts: {sum(len(b['results']) for b in all_ckpt_results)}")


if __name__ == "__main__":
    main()
