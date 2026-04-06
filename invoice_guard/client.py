"""InvoiceGuard Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        InvoiceGuardAction,
        InvoiceGuardObservation,
        InvoiceGuardState,
    )
except (ImportError, ModuleNotFoundError):
    from models import (
        InvoiceGuardAction,
        InvoiceGuardObservation,
        InvoiceGuardState,
    )


class InvoiceGuardEnv(
    EnvClient[InvoiceGuardAction, InvoiceGuardObservation, InvoiceGuardState]
):
    """
    Client for the InvoiceGuard Environment.

    Example:
        >>> async with InvoiceGuardEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset(task_id="task_1_clean_match")
        ...     print(result.observation.invoice_summary)
    """

    def _step_payload(self, action: InvoiceGuardAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[InvoiceGuardObservation]:
        obs_data = payload.get("observation", {})
        observation = InvoiceGuardObservation(
            case_id=obs_data.get("case_id", ""),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            invoice_summary=obs_data.get("invoice_summary", ""),
            goal=obs_data.get("goal", ""),
            available_actions=obs_data.get("available_actions", []),
            revealed_documents=obs_data.get("revealed_documents", []),
            findings=obs_data.get("findings", []),
            remaining_steps=obs_data.get("remaining_steps", 0),
            last_action_result=obs_data.get("last_action_result", ""),
            last_action_error=obs_data.get("last_action_error", False),
            warnings=obs_data.get("warnings", []),
            grader_result=obs_data.get("grader_result", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> InvoiceGuardState:
        return InvoiceGuardState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", ""),
            case_id=payload.get("case_id", ""),
            actions_taken=payload.get("actions_taken", []),
            documents_revealed=payload.get("documents_revealed", []),
            findings_collected=payload.get("findings_collected", []),
            is_finalized=payload.get("is_finalized", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
