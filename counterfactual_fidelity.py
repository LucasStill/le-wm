"""counterfactual_fidelity.py — online-simulator demo for the dataset paper.

For trained world models, measure how predictions match the simulator's
actual response to alternative actions.

Two metrics:
  1. Absolute RMSE(τ) per (ckpt, action). Both sim and model are
     conditioned on action so this is partly degenerate, but the
     ckpt-vs-ckpt comparison is informative.
  2. Differential: |Δmodel(a, 0) - Δsim(a, 0)| at each τ. The genuine
     "did the model learn action causality" test — does the world
     model's response-to-action-change match the simulator's?

Supports multiple ckpts, multiple actions, multiple branch positions.

Usage:
  python counterfactual_fidelity.py \\
      --ckpts E2:/path/to/E2_ep10.ckpt E1:/path/to/E1_ep10.ckpt \\
      --sim_h5 /home/lthil/.stable_worldmodel/scenario4_train_sensors.h5 \\
      --eval_h5 /home/lthil/.stable_worldmodel/scenario4_test_lewm.h5 \\
      --out_dir eval_results/counterfactual \\
      --n_episodes 30 --horizon 100 --actions 0 1 2 3 4 5 6 \\
      --branch_fracs 0.25 0.5 0.75
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import h5py

# Make simulator importable
sys.path.insert(0, "/home/lthil/thesis/rl_opendeck_simulator")
from scenarios.scenario4.replay import EpisodeReplayer, CounterfactualSpec

# Reuse eval_sweep machinery for probe training
sys.path.insert(0, "/home/lthil/thesis/le-wm")
import eval_sweep


def model_rollout_HI(model, probe, scaler, sensors_history: np.ndarray,
                     hist_actions: np.ndarray, override_action: int,
                     horizon: int, device) -> np.ndarray:
    """Roll out world model, decode each predicted z to HI via the probe.

    Args:
        sensors_history: (H, 11, 16) raw sensor frames before branch_t.
        hist_actions:    (H,) actual actions taken DURING history (from dataset).
        override_action: scalar action index to use for ALL future steps.
        horizon:         number of future steps to predict.

    Returns predicted HI: (horizon, 10).
    """
    H = sensors_history.shape[0]
    # 1. Encode history → (H, D)
    info = {
        "pixels": torch.as_tensor(sensors_history).unsqueeze(0).float().to(device),
        "action": torch.as_tensor(hist_actions).unsqueeze(0).long().to(device),
    }
    out = model.encode(info)
    emb = out["emb"]  # (1, H, D)
    actions = info["action"]  # (1, H)

    # 2. Autoregressive predict
    HS = model.wm.history_size if hasattr(model, "wm") else H  # context window for predictor
    HS = min(HS, H)
    rollout_embs = []
    for t in range(horizon):
        emb_ctx = emb[:, -HS:]                      # (1, HS, D)
        act_ctx = actions[:, -HS:]                  # (1, HS)
        act_emb = model.action_encoder(act_ctx)     # (1, HS, A_emb)
        pred = model.predict(emb_ctx, act_emb)      # (1, HS, D)
        next_emb = pred[:, -1:, :]                  # (1, 1, D)
        rollout_embs.append(next_emb)
        emb = torch.cat([emb, next_emb], dim=1)
        next_act = torch.full((1, 1), override_action, dtype=torch.long, device=device)
        actions = torch.cat([actions, next_act], dim=1)

    rollout = torch.cat(rollout_embs, dim=1)[0].detach().cpu().numpy()  # (horizon, D)

    # 3. Decode via probe (seq_len=1)
    rollout_scaled = scaler.transform(rollout)
    Xw = rollout_scaled.reshape(rollout.shape[0], 1, rollout.shape[1])
    probe.eval()
    with torch.no_grad():
        Xw_t = torch.as_tensor(Xw, dtype=torch.float32, device=device)
        preds = probe(Xw_t).cpu().numpy()  # (horizon, 10)
    return preds


def evaluate_one_ckpt(ckpt_name: str, ckpt_path: Path, args, replayer,
                      tr_data, te_data, hist_window: int, episodes: list,
                      device) -> dict:
    """Returns dict with per-(episode, branch_t, action) results."""
    print(f"\n{'='*60}\nLoading {ckpt_name}: {ckpt_path}")
    model = torch.load(str(ckpt_path), map_location=device, weights_only=False).eval().to(device)

    print(f"  encoding train embeddings…")
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

    fh_lewm = h5py.File("/home/lthil/.stable_worldmodel/scenario4_train_lewm.h5", "r")
    ep_offsets = fh_lewm["ep_offset"][:]

    results = []
    for ep_idx in episodes:
        full_ep = replayer.replay_full(int(ep_idx))
        T_ep = full_ep.T

        for frac in args.branch_fracs:
            branch_t = int(T_ep * frac)
            if branch_t < hist_window or branch_t + args.horizon >= T_ep:
                continue

            # Pull sensor history + action history
            abs_start = int(ep_offsets[ep_idx]) + branch_t - hist_window
            sensors_history = fh_lewm["pixels"][abs_start : abs_start + hist_window][:]  # (H,11,16)
            hist_actions = full_ep.action_idx[branch_t - hist_window : branch_t]  # (H,)

            for action in args.actions:
                # Simulator counterfactual
                spec = CounterfactualSpec(branch_t=branch_t, override_action=int(action),
                                          horizon=args.horizon)
                cf = replayer.counterfactual(int(ep_idx), spec)
                sim_HI = cf.s  # (horizon, 10)
                sim_len = sim_HI.shape[0]

                # World-model prediction
                try:
                    model_HI = model_rollout_HI(
                        model, probe, scaler,
                        sensors_history, hist_actions,
                        int(action), sim_len, device,
                    )
                except Exception as e:
                    print(f"    ep={ep_idx} branch_frac={frac} action={action} ROLLOUT FAILED: {e}")
                    continue

                # Per-step RMSE per HI dim, summed
                rmse_per_step = np.sqrt(((sim_HI - model_HI) ** 2).mean(axis=1))
                results.append({
                    "ckpt": ckpt_name,
                    "ep_idx": int(ep_idx),
                    "branch_t": branch_t,
                    "branch_frac": frac,
                    "action": int(action),
                    "horizon": sim_len,
                    "sim_HI": sim_HI.tolist(),
                    "model_HI": model_HI.tolist(),
                    "rmse_per_step": rmse_per_step.tolist(),
                })
        print(f"  ep={ep_idx} done across {len(args.branch_fracs)} branches × {len(args.actions)} actions")

    fh_lewm.close()
    del model
    torch.cuda.empty_cache()
    return {"ckpt": ckpt_name, "ckpt_path": str(ckpt_path), "results": results}


def compute_differentials(all_results: list) -> dict:
    """For each (ckpt, episode, branch_t), compute Δ(action_a, action_0) for
    sim and model, then |Δmodel - Δsim| as the differential metric."""
    out = {}
    for ckpt_blob in all_results:
        ckpt = ckpt_blob["ckpt"]
        # Group by (ep, branch_t)
        groups = {}
        for r in ckpt_blob["results"]:
            key = (r["ep_idx"], r["branch_t"])
            groups.setdefault(key, {})[r["action"]] = r
        # For each group, compute differentials
        diffs = []
        for (ep, bt), action_results in groups.items():
            if 0 not in action_results:
                continue
            base_sim = np.array(action_results[0]["sim_HI"])
            base_mod = np.array(action_results[0]["model_HI"])
            for a, r in action_results.items():
                if a == 0:
                    continue
                d_sim = np.array(r["sim_HI"]) - base_sim       # (horizon, 10)
                d_mod = np.array(r["model_HI"]) - base_mod
                # |Δmodel - Δsim| → Frobenius per step
                diff_per_step = np.sqrt(((d_mod - d_sim) ** 2).mean(axis=1))
                diffs.append({
                    "ep_idx": ep, "branch_t": bt, "action": a,
                    "diff_per_step": diff_per_step.tolist(),
                })
        out[ckpt] = diffs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name:path pairs, e.g. E2:/path/to/ckpt E1:/path/to/ckpt")
    ap.add_argument("--sim_h5", required=True)
    ap.add_argument("--eval_h5", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--actions", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--branch_fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--history", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval data {args.eval_h5}…")
    tr_data, te_data, *_ = eval_sweep.load_dataset(args.eval_h5, encoder_type="sensor")

    print(f"Opening simulator {args.sim_h5}…")
    replayer = EpisodeReplayer(args.sim_h5)
    n_eps_total = replayer.f["ep_meta/seed"].shape[0]
    chosen = sorted(np.random.choice(n_eps_total, size=args.n_episodes, replace=False).tolist())
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
        blob = evaluate_one_ckpt(name, path, args, replayer, tr_data, te_data,
                                 args.history, chosen, device)
        all_ckpt_results.append(blob)
        # Save incrementally (in case later ckpts crash)
        (out_dir / f"results_{name}.json").write_text(json.dumps(blob, indent=2))

    diffs = compute_differentials(all_ckpt_results)
    summary = {
        "n_episodes": args.n_episodes,
        "horizon": args.horizon,
        "actions": args.actions,
        "branch_fracs": args.branch_fracs,
        "ckpts": [b["ckpt"] for b in all_ckpt_results],
        "differentials": diffs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll done. Saved per-ckpt JSONs and summary.json to {out_dir}/")
    print(f"  ckpts evaluated: {[b['ckpt'] for b in all_ckpt_results]}")
    print(f"  total rollouts: {sum(len(b['results']) for b in all_ckpt_results)}")


if __name__ == "__main__":
    main()
