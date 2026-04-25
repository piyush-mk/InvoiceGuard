"""
Agentic rollout helper for InvoiceGuard.

Drives the local OpenEnv environment with a Hugging Face causal LM (instead
of an OpenAI client). Reuses the SAME prompt/parse helpers as `inference.py`
so trajectories collected here are byte-identical in IO to what the OpenAI
baseline sees.

Returns a `Trajectory` describing every (prompt, action) pair plus the
per-step env reward and the terminal grader score. The trainer uses this to
compute group-relative advantages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import torch

# Hackathon code is laid out flat: `invoice_guard` is on sys.path at runtime.
from inference import (  # type: ignore
    SYSTEM_PROMPT,
    build_action,
    build_observation_prompt,
    parse_llm_response,
    strip_think_blocks,
)
from models import TaskID  # type: ignore

if TYPE_CHECKING:
    from server.invoice_guard_environment import InvoiceGuardEnvironment


@dataclass
class TrajectoryStep:
    """One agent decision inside an episode."""
    prompt_text: str          # full chat prompt fed to the LM (after template)
    completion_text: str      # raw LM completion (action JSON)
    prompt_ids: torch.Tensor  # token ids for prompt (1D, long)
    completion_ids: torch.Tensor  # token ids for completion (1D, long)
    reward: float             # per-step env reward returned by env.step()


@dataclass
class Trajectory:
    """A full episode."""
    task_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    cumulative_reward: float = 0.0
    grader_score: float = 0.0
    terminal_decision: Optional[str] = None
    success: bool = False

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def _render_chat_prompt(tokenizer, messages: List[dict]) -> str:
    """Apply the model's chat template, leaving the assistant turn open."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.no_grad()
def rollout_episode(
    model,
    tokenizer,
    env: "InvoiceGuardEnvironment",
    task_id: TaskID,
    *,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 384,
    max_prompt_tokens: int = 2048,
    device: Optional[torch.device] = None,
) -> Trajectory:
    """
    Run one full episode against the local env using `model` as the policy.

    Sampling is stochastic on purpose: GRPO needs intra-group variance.
    """
    device = device or next(model.parameters()).device
    obs = env.reset(task_id=task_id.value)

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    traj = Trajectory(task_id=task_id.value)

    while not obs.done:
        user_msg = build_observation_prompt(obs, is_first=(traj.n_steps == 0))
        messages.append({"role": "user", "content": user_msg})

        prompt_text = _render_chat_prompt(tokenizer, messages)
        prompt_enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)
        prompt_ids = prompt_enc.input_ids[0]

        gen = model.generate(
            **prompt_enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion_ids = gen[0, prompt_ids.shape[0]:]
        # Decode WITHOUT skipping special tokens so <think>...</think> tags
        # are preserved for our regex. Then strip think blocks, then remove
        # remaining special tokens (EOS, chat markers, etc.).
        raw_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        cleaned = strip_think_blocks(raw_text)
        for tok in tokenizer.all_special_tokens:
            cleaned = cleaned.replace(tok, "")
        completion_text = cleaned.strip()
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if traj.n_steps < 2:
            print(f"[rollout-diag] task={task_id.value} step={traj.n_steps} "
                  f"gen_tokens={len(completion_ids)} "
                  f"raw_text={repr(raw_text[:300])} "
                  f"completion_text={repr(completion_text[:200])}", flush=True)

        messages.append({"role": "assistant", "content": completion_text})

        params = parse_llm_response(completion_text)
        action = build_action(params)
        obs = env.step(action)
        reward = float(obs.reward or 0.0)

        traj.steps.append(
            TrajectoryStep(
                prompt_text=prompt_text,
                completion_text=completion_text,
                prompt_ids=prompt_ids.detach().cpu(),
                completion_ids=completion_ids.detach().cpu(),
                reward=reward,
            )
        )

    grader_data = obs.metadata.get("grader_result", {}) if hasattr(obs, "metadata") else {}
    traj.grader_score = float(grader_data.get("score", 0.0)) if isinstance(grader_data, dict) else 0.0
    traj.cumulative_reward = float(getattr(env.state, "cumulative_reward", 0.0))
    traj.success = traj.grader_score >= 0.5
    if traj.steps:
        last_params = parse_llm_response(traj.steps[-1].completion_text)
        traj.terminal_decision = last_params.get("final_decision")

    return traj
