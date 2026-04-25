# le-wm experiment tracking

Auto-published snapshot of the active Scenario-4 TurboSens campaigns
(JEPA on dragon, AR-LSTM on orailixtower).

**Start here:** [`RESULTS.md`](RESULTS.md) — live cross-architecture summary.

## Layout

- `RESULTS.md` — auto-aggregated headline + per-arch tables + log tails
- `coordination/dragon/` — JEPA agent's identity / state / results / log
- `coordination/orailixtower/` — AR-LSTM agent's identity / state / results / log
- `logs/campaign_report_*.md` — long-form retrospective
- `logs/experiment_plan.md` — experimental design
- `logs/handoff_notes.md` — running session log
- `logs/session_*.md` — code-change notes

Updated by `publish_tracking.sh` on dragon. Pushes only happen on actual diff.
Last update timestamp is in `RESULTS.md` itself.
