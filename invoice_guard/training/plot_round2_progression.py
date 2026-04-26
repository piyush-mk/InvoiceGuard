from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "invoice_guard" / "outputs" / "training_runs"
JOB_DIR = (
    ROOT
    / "invoice_guard"
    / "outputs"
    / "job_reports"
    / "69ed3f79d2c8bd8662bce8da_artifacts"
)
BASELINE_PATH = (
    ROOT / "invoice_guard" / "outputs" / "baseline_scores" / "local_baseline_qwen3_4b.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6.2),
            "figure.dpi": 170,
            "axes.facecolor": "#f8fbff",
            "savefig.facecolor": "#ffffff",
            "axes.edgecolor": "#a8bddf",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#c2d3ee",
            "grid.color": "#d7e3f5",
            "grid.alpha": 0.75,
            "grid.linestyle": "-",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    out = RUN_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def _extract_sft_eval(metrics_path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    rows = _read_jsonl(metrics_path)
    eval_rows = [r for r in rows if "eval/avg_grader_score" in r]
    epochs = [int(r["step"]) for r in eval_rows]
    scores = [float(r["eval/avg_grader_score"]) for r in eval_rows]
    success = [float(r["eval/success_rate"]) for r in eval_rows]
    steps = [float(r["eval/avg_steps"]) for r in eval_rows]
    return epochs, scores, success, steps


def _extract_sft_train_loss(metrics_path: Path) -> tuple[list[int], list[float]]:
    rows = _read_jsonl(metrics_path)
    train_rows = [r for r in rows if "train/loss" in r]
    return [int(r["step"]) for r in train_rows], [float(r["train/loss"]) for r in train_rows]


def _extract_grpo_eval(metrics_path: Path) -> tuple[list[str], list[float], list[float], list[float]]:
    rows = _read_jsonl(metrics_path)
    labels = ["init", "iter1", "iter2", "iter3"]
    score_keys = [
        "eval/init/avg_grader_score",
        "eval/iter1/avg_grader_score",
        "eval/iter2/avg_grader_score",
        "eval/iter3/avg_grader_score",
    ]
    success_keys = [
        "eval/init/success_rate",
        "eval/iter1/success_rate",
        "eval/iter2/success_rate",
        "eval/iter3/success_rate",
    ]
    steps_keys = [
        "eval/init/avg_steps",
        "eval/iter1/avg_steps",
        "eval/iter2/avg_steps",
        "eval/iter3/avg_steps",
    ]

    scores, success, steps = [], [], []
    for s_key, u_key, st_key in zip(score_keys, success_keys, steps_keys):
        row = next(r for r in rows if s_key in r)
        scores.append(float(row[s_key]))
        success.append(float(row[u_key]))
        steps.append(float(row[st_key]))
    return labels, scores, success, steps


def _extract_grpo_train(metrics_path: Path) -> dict[str, list[float]]:
    rows = _read_jsonl(metrics_path)
    train_rows = [r for r in rows if "train/task_id" in r]
    return {
        "x": list(range(1, len(train_rows) + 1)),
        "reward": [float(r["train/group_reward_mean"]) for r in train_rows],
        "grader": [float(r["train/group_grader_mean"]) for r in train_rows],
        "success": [float(r["train/group_success_rate"]) for r in train_rows],
        "loss": [float(r["train/loss"]) for r in train_rows],
        "pg_loss": [float(r["train/pg_loss"]) for r in train_rows],
        "kl_loss": [float(r["train/kl_loss"]) for r in train_rows],
    }


def _rolling(values: list[float], window: int = 5) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def main() -> None:
    _setup_style()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _read_json(BASELINE_PATH)
    baseline_score = float(baseline["summary"]["avg_score"])
    baseline_success = float(baseline["summary"]["success_rate"])
    baseline_steps = float(baseline["summary"]["avg_steps"])

    sft_v5c_epochs, sft_v5c_scores, sft_v5c_success, sft_v5c_steps = _extract_sft_eval(
        RUN_DIR / "sft_v5c_sft_metrics.jsonl"
    )
    sft_v5d_epochs, sft_v5d_scores, sft_v5d_success, sft_v5d_steps = _extract_sft_eval(
        RUN_DIR / "sft_v5d_sft_metrics.jsonl"
    )
    sft_v5c_train_x, sft_v5c_train_loss = _extract_sft_train_loss(
        RUN_DIR / "sft_v5c_sft_metrics.jsonl"
    )
    sft_v5d_train_x, sft_v5d_train_loss = _extract_sft_train_loss(
        RUN_DIR / "sft_v5d_sft_metrics.jsonl"
    )

    grpo_labels, grpo_scores, grpo_success, grpo_steps = _extract_grpo_eval(
        JOB_DIR / "metrics.jsonl"
    )
    grpo_train = _extract_grpo_train(JOB_DIR / "metrics.jsonl")

    # 1) Stage progression: grader score
    fig, ax = plt.subplots()
    ax.axhline(
        baseline_score,
        color="#d14c4c",
        linestyle="--",
        linewidth=2.0,
        label=f"Local baseline ({baseline_score:.3f})",
    )
    ax.plot(
        sft_v5c_epochs,
        sft_v5c_scores,
        color="#2b7de9",
        marker="o",
        linewidth=2.2,
        label="SFT v5c eval score",
    )
    ax.plot(
        sft_v5d_epochs,
        sft_v5d_scores,
        color="#10a37f",
        marker="o",
        linewidth=2.2,
        label="SFT v5d eval score",
    )
    ax.plot(
        [0, 1, 2, 3],
        grpo_scores,
        color="#7a4cff",
        marker="D",
        linewidth=2.6,
        label="GRPO (warm-start from SFT v5d-best)",
    )
    ax.set_title("Score Progression: Baseline → SFT → Warm-Started GRPO")
    ax.set_xlabel("Training progress index (SFT=epoch, GRPO=iteration)")
    ax.set_ylabel("Holdout average grader score")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    _save(fig, "round2_progression_eval_score.png")

    # 2) Stage progression: success and steps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1, ax2 = axes
    ax1.axhline(baseline_success, color="#d14c4c", linestyle="--", linewidth=2.0, label="Local baseline")
    ax1.plot(sft_v5c_epochs, sft_v5c_success, color="#2b7de9", marker="o", linewidth=2.0, label="SFT v5c")
    ax1.plot(sft_v5d_epochs, sft_v5d_success, color="#10a37f", marker="o", linewidth=2.0, label="SFT v5d")
    ax1.plot([0, 1, 2, 3], grpo_success, color="#7a4cff", marker="D", linewidth=2.4, label="GRPO")
    ax1.set_title("Success Rate Progression")
    ax1.set_xlabel("Progress index")
    ax1.set_ylabel("Success rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="lower right")

    ax2.axhline(baseline_steps, color="#d14c4c", linestyle="--", linewidth=2.0, label="Local baseline")
    ax2.plot(sft_v5c_epochs, sft_v5c_steps, color="#2b7de9", marker="o", linewidth=2.0, label="SFT v5c")
    ax2.plot(sft_v5d_epochs, sft_v5d_steps, color="#10a37f", marker="o", linewidth=2.0, label="SFT v5d")
    ax2.plot([0, 1, 2, 3], grpo_steps, color="#7a4cff", marker="D", linewidth=2.4, label="GRPO")
    ax2.set_title("Average Steps to Resolution")
    ax2.set_xlabel("Progress index")
    ax2.set_ylabel("Steps")
    ax2.set_ylim(0.0, max(12.5, baseline_steps + 0.5))
    ax2.legend(loc="upper right")
    _save(fig, "round2_progression_success_steps.png")

    # 3) SFT training loss curves
    fig, ax = plt.subplots()
    ax.plot(
        sft_v5c_train_x,
        sft_v5c_train_loss,
        color="#2b7de9",
        linewidth=1.8,
        alpha=0.85,
        label="SFT v5c train loss",
    )
    ax.plot(
        sft_v5d_train_x,
        sft_v5d_train_loss,
        color="#10a37f",
        linewidth=1.8,
        alpha=0.85,
        label="SFT v5d train loss",
    )
    ax.set_yscale("log")
    ax.set_title("SFT Train Loss Curves (Log Scale)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (log)")
    ax.legend(loc="upper right")
    _save(fig, "round2_sft_training_loss_log.png")

    # 4) GRPO optimization signals
    fig, ax = plt.subplots()
    x = grpo_train["x"]
    ax.plot(x, grpo_train["reward"], color="#7a4cff", alpha=0.30, linewidth=1.2)
    ax.plot(x, _rolling(grpo_train["reward"], 5), color="#7a4cff", linewidth=2.3, label="Group reward mean (smoothed)")
    ax.plot(x, grpo_train["grader"], color="#00a6c8", alpha=0.28, linewidth=1.1)
    ax.plot(x, _rolling(grpo_train["grader"], 5), color="#00a6c8", linewidth=2.3, label="Group grader mean (smoothed)")
    ax.plot(x, _rolling(grpo_train["success"], 5), color="#1e8e3e", linewidth=2.3, label="Group success rate (smoothed)")
    ax.set_title("GRPO Training Signals Across Task Updates")
    ax.set_xlabel("GRPO task update step")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 2.4)
    ax.legend(loc="lower right")
    _save(fig, "round2_grpo_training_signals.png")

    # 5) GRPO loss decomposition
    fig, ax = plt.subplots()
    ax.plot(x, np.abs(grpo_train["loss"]), color="#ff8a00", alpha=0.25, linewidth=1.0)
    ax.plot(x, _rolling(np.abs(grpo_train["loss"]).tolist(), 5), color="#ff8a00", linewidth=2.2, label="|Total loss| (smoothed)")
    ax.plot(x, _rolling(np.abs(grpo_train["pg_loss"]).tolist(), 5), color="#d14c4c", linewidth=2.2, label="|Policy loss| (smoothed)")
    ax.plot(x, _rolling(np.abs(grpo_train["kl_loss"]).tolist(), 5), color="#7f8ea3", linewidth=2.2, label="|KL loss| (smoothed)")
    ax.set_yscale("log")
    ax.set_title("GRPO Loss Components (Absolute, Log Scale)")
    ax.set_xlabel("GRPO task update step")
    ax.set_ylabel("Absolute loss (log)")
    ax.legend(loc="upper right")
    _save(fig, "round2_grpo_loss_components_log.png")

    # 6) Stage comparison summary bars
    best_sft = max(max(sft_v5c_scores), max(sft_v5d_scores))
    best_grpo = max(grpo_scores)
    final_grpo = grpo_scores[-1]
    stage_names = [
        "Local baseline",
        "Best SFT",
        "GRPO init",
        "GRPO best",
        "GRPO final",
    ]
    scores = [baseline_score, best_sft, grpo_scores[0], best_grpo, final_grpo]
    success_vals = [baseline_success, max(max(sft_v5c_success), max(sft_v5d_success)), grpo_success[0], max(grpo_success), grpo_success[-1]]
    steps_vals = [baseline_steps, min(min(sft_v5c_steps), min(sft_v5d_steps)), grpo_steps[0], min(grpo_steps), grpo_steps[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    colors = ["#d14c4c", "#2b7de9", "#7a4cff", "#5b5ce2", "#6e4ccf"]
    axes[0].bar(stage_names, scores, color=colors)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Avg Grader Score")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(scores):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(stage_names, success_vals, color=colors)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Success Rate")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(success_vals):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(stage_names, steps_vals, color=colors)
    axes[2].set_ylim(0.0, max(12.5, max(steps_vals) + 0.8))
    axes[2].set_title("Avg Steps")
    axes[2].tick_params(axis="x", rotation=20)
    for i, v in enumerate(steps_vals):
        axes[2].text(i, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Round 2 Stage Snapshot: Baseline vs SFT vs Warm-Started GRPO", y=1.03, fontsize=15, fontweight="bold")
    _save(fig, "round2_stage_comparison.png")


if __name__ == "__main__":
    main()
