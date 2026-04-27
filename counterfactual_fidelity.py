"""counterfactual_fidelity.py — online-simulator demo for the dataset paper.

For a trained world model, measure how its predictions match the simulator's
actual response to alternative actions. Two metrics:

  1. Absolute RMSE(τ) per action — sanity (noisy because both are
     conditioned on the same action; doesn't fully discriminate).
  2. Differential: |Δmodel(a, 0) - Δsim(a, 0)| at each τ. This is the
     genuine "did the world model learn action causality" test — measures
     whether the world model's response-to-action-change matches the
     simulator's.

Usage:
  python counterfactual_fidelity.py \
      --ckpt /home/lthil/.stable_worldmodel/scenario4_sensor_w1_H32_S1_P4/lewm_s4_sensor_w1_H32_S1_P4_epoch_10_object.ckpt \
      --sim_h5 /home/lthil/.stable_worldmodel/scenario4_train_sensors.h5 \
      --eval_h5 /home/lthil/.stable_worldmodel/scenario4_test_lewm.h5 \
      --out_dir eval_results/counterfactual_E2 \
      --n_episodes 10 --horizon 100 --actions 0 1 5 6
"""
import argparse, json, sys, os
from pathlib import Path
import numpy as np
import torch

# Make simulator importable
sys.path.insert(0, "/home/lthil/thesis/rl_opendeck_simulator")
from scenarios.scenario4.replay import EpisodeReplayer, CounterfactualSpec

# Reuse eval_sweep machinery for probe training
sys.path.insert(0, "/home/lthil/thesis/le-wm")
import eval_sweep


def encode_window(model, sensors_window: torch.Tensor, device) -> torch.Tensor:
    """sensors_window: (T_raw, 11, 16) numpy or tensor → (T_out, d_model) emb tensor.

    Build an info dict and call model.encode for the sensor path.
    """
    if not torch.is_tensor(sensors_window):
        sensors_window = torch.as_tensor(sensors_window)
    info = {"pixels": sensors_window.unsqueeze(0).float().to(device)}  # (1, T_raw, 11, 16)
    info["action"] = torch.zeros(1, sensors_window.shape[0], dtype=torch.long, device=device)
    out = model.encode(info)
    return out["emb"][0]  # (T_out, d_model)


def model_rollout_HI(model, probe, scaler, sensors_history: np.ndarray,
                     action_sequence: np.ndarray, device) -> np.ndarray:
    """sensors_history: (H, 11, 16) raw frames before branch_t.
    action_sequence:    (horizon,) int actions to roll out.
    Returns predicted HI: (horizon, 10).
    """
    H = sensors_history.shape[0]
    horizon = len(action_sequence)
    # encode history → embeddings (H, D)
    init_emb = encode_window(model, sensors_history, device)  # (H, D)
    emb = init_emb.unsqueeze(0)  # (1, H, D)
    # actions: prepend H zeros (these are the "history" actions used by encoder; values don't matter for sensor encode), then the override sequence
    fut_actions = torch.as_tensor(action_sequence, dtype=torch.long, device=device)
    # autoregressive predict step-by-step
    HS = model.wm.history_size if hasattr(model, "wm") else 16
    HS = min(HS, emb.shape[1])
    rollout_embs = []
    for t in range(horizon):
        # last HS embeddings as context
        emb_ctx = emb[:, -HS:]  # (1, HS, D)
        # action embedding: model.action_encoder expects (1, HS) ints
        # We need HS actions: history actions (don't matter for sensor) + the new override
        # Build action sequence: take last (HS-1) zeros + current override
        act_ctx = torch.cat([
            torch.zeros(1, HS - 1, dtype=torch.long, device=device),
            fut_actions[t:t+1].unsqueeze(0),
        ], dim=1)
        act_emb = model.action_encoder(act_ctx)  # (1, HS, A_emb)
        pred = model.predict(emb_ctx, act_emb)   # (1, HS, D)
        next_emb = pred[:, -1:, :]                # (1, 1, D)
        rollout_embs.append(next_emb)
        emb = torch.cat([emb, next_emb], dim=1)

    rollout = torch.cat(rollout_embs, dim=1)[0]  # (horizon, D)
    # Apply probe via the same windowing pipeline as eval_sweep (seq_len=1)
    rollout_np = rollout.detach().cpu().numpy()
    rollout_scaled = scaler.transform(rollout_np)  # (horizon, D)
    Xw = rollout_scaled.reshape(rollout.shape[0], 1, rollout.shape[1])
    probe.eval()
    with torch.no_grad():
        Xw_t = torch.as_tensor(Xw, dtype=torch.float32, device=device)
        preds = probe(Xw_t).cpu().numpy()  # (horizon, 10)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim_h5", required=True, help="*_sensors.h5 path with seeds")
    ap.add_argument("--eval_h5", required=True, help="*_lewm.h5 for probe training")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--actions", type=int, nargs="+", default=[0, 1, 5, 6],
                    help="action indices to test as override (e.g., 0 1 5 6)")
    ap.add_argument("--branch_frac", type=float, default=0.5,
                    help="branch_t = floor(ep_T * branch_frac) — keep mid-episode")
    ap.add_argument("--history", type=int, default=16,
                    help="how many raw frames before branch_t to feed encoder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.ckpt}…")
    model = torch.load(args.ckpt, map_location=device, weights_only=False).eval().to(device)

    print(f"Loading eval data {args.eval_h5}…")
    tr_data, te_data, hi_names, archetype_names, *_ = eval_sweep.load_dataset(
        args.eval_h5, encoder_type="sensor",
    )

    print("Encoding observations through model.encoder…")
    Z_tr = eval_sweep.encode_observations(model, tr_data["obs"], "sensor", device)
    Z_te = eval_sweep.encode_observations(model, te_data["obs"], "sensor", device)

    print("Training probe (TransformerProbe, seq_len=1)…")
    preds_te, gt_te, probe, scaler = eval_sweep.train_probe(
        Z_tr, tr_data["hi"], Z_te, te_data["hi"],
        tr_data["ep_ids"], te_data["ep_ids"],
        n_outputs=10, device=device, task_name="cf_probe",
        seq_len=1,
    )
    probe.eval()

    print(f"Opening simulator {args.sim_h5}…")
    replayer = EpisodeReplayer(args.sim_h5)
    n_eps_total = replayer.f["ep_meta/seed"].shape[0]
    chosen = np.random.choice(n_eps_total, size=args.n_episodes, replace=False)

    print(f"Running {args.n_episodes} episodes × {len(args.actions)} actions × horizon {args.horizon}…")
    results = []
    for ep_idx in chosen:
        ep = replayer.replay_full(int(ep_idx))
        T = ep.T
        branch_t = int(T * args.branch_frac)
        if branch_t < args.history or branch_t + args.horizon >= T:
            continue
        # Extract history sensor frames (raw) — replayer gives `ep.s` (states, 10-dim) NOT raw sensors.
        # We need raw sensor obs from the lewm h5 (or sensors h5).
        # Fall back: read pixels from sim_h5 (it has them in the same shape under 'pixels' or similar).
        # For now, try the lewm h5 train file directly.
        import h5py
        with h5py.File("/home/lthil/.stable_worldmodel/scenario4_train_lewm.h5", "r") as fh:
            ep_offset = int(fh["ep_offset"][ep_idx])
            # absolute timestep = ep_offset + branch_t - history
            abs_start = ep_offset + branch_t - args.history
            sensors_history = fh["pixels"][abs_start : abs_start + args.history][:]  # (H, 11, 16)

        for action in args.actions:
            # Simulator counterfactual
            spec = CounterfactualSpec(branch_t=branch_t, override_action=int(action),
                                       horizon=args.horizon)
            cf = replayer.counterfactual(int(ep_idx), spec)
            sim_HI = cf.s  # (horizon, 10)

            # World-model prediction
            try:
                action_seq = np.full(args.horizon, action, dtype=np.int64)
                model_HI = model_rollout_HI(
                    model, probe, scaler, sensors_history, action_seq, device,
                )
            except Exception as e:
                print(f"  ep={ep_idx} action={action} ROLLOUT FAILED: {e}")
                continue

            # Per-step RMSE per HI dim
            rmse_per_step = np.sqrt(((sim_HI[:len(model_HI)] - model_HI) ** 2).mean(axis=1))
            results.append({
                "ep_idx": int(ep_idx),
                "branch_t": branch_t,
                "action": int(action),
                "sim_HI": sim_HI[:len(model_HI)].tolist(),
                "model_HI": model_HI.tolist(),
                "rmse_per_step": rmse_per_step.tolist(),
            })
        print(f"  ep={ep_idx} done ({len(args.actions)} actions)")

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps({
        "ckpt": args.ckpt, "n_episodes": args.n_episodes,
        "horizon": args.horizon, "actions": args.actions,
        "results": results,
    }, indent=2))
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
