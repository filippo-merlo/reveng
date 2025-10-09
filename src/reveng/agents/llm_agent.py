import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from minigrid.minigrid_env import MiniGridEnv

from reveng.agents.agent_abc import Agent
from reveng.agents.llm_templates import ActionResponse
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)
from reveng.llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


class LLMAgent(Agent, BaseLLMInterface):
    """An agent that uses a Language Model to select actions in a MiniGrid environment."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            temperature: Temperature for the model (keep at 0.0 for consistent action selection).
            name: Optional name for the agent. Defaults to class name.
        """
        Agent.__init__(self, name)
        BaseLLMInterface.__init__(
            self,
            model_name=model_name,
            temperature=temperature,
            template_path=Path(__file__).parent / "prompt_templates" / "maze.j2",
        )
        # Cost tracking
        self.total_cost = 0.0
        self.call_count = 0

    def select_action(
        self, env: MiniGridEnv, return_logprobs: bool = False, **kwargs: Any
    ) -> Tuple[int, dict]:
        """
        Select an action using the LLM based on the current environment state.

        Args:
            env: The environment to interact with
            **kwargs: Additional arguments (ignored by this agent)

        Returns:
            The selected action and a dictionary with related metadata.
        """
        base_env = getattr(env, "unwrapped", env)
        agent_pos = (
            tuple(base_env.agent_pos) if hasattr(base_env, "agent_pos") else "unknown"
        )

        logger.info(
            f"Getting action from LLM model {self.model_name} for position {agent_pos}"
        )
        prompt = self._generate_action_query_prompt(env)

        try:
            extra_kwargs = {}
            response_format = ActionResponse
            if return_logprobs:
                extra_kwargs["logprobs"] = True
                extra_kwargs["top_logprobs"] = 5
                extra_kwargs["allowed_openai_params"] = ["logprobs", "top_logprobs"]
                response_format = None

            response, cost, raw_response = self._make_completion_request(
                prompt,
                response_format=response_format,
                **extra_kwargs,
            )

            if response_format is None:
                # TODO add retry if validation fails?
                action_response = ActionResponse.model_validate_json(response)
            else:
                action_response = response

            # Track costs
            self.total_cost += cost
            self.call_count += 1

            if return_logprobs:
                logprobs = raw_response.choices[0].logprobs

            logger.info(
                f"LLM selected action {action_response.action}: {action_response.confidence} (cost: ${cost:.6f}, total: ${self.total_cost:.6f})"
            )
            return action_response.action.value, {
                "agents_name": self.name,
                "llm_response": action_response.action,
                "confidence": action_response.confidence,
                "call_cost": cost,
                "total_cost": self.total_cost,
                "call_count": self.call_count,
                "logprobs": logprobs if logprobs else None,
            }

        except Exception as e:
            logger.error(f"Error getting action from LLM: {e}")
            raise

    def _get_text_observation(self, env: MiniGridEnv) -> str:
        """Get text observation of the environment using existing wrapper."""
        # Wrap env temporarily to get text observation
        text_env = FullObservabilityTextWrapper(env)
        return text_env.observation(None)

    def _generate_action_query_prompt(self, env: MiniGridEnv) -> str:
        """Generate a prompt for the LLM to select an action."""
        # Get text observation using existing wrapper
        obs_text = self._get_text_observation(env)

        prompt = self.render_template(
            obs_text=obs_text,
        )
        return prompt

    def reset(self) -> None:
        """Reset the agent's internal state and cost tracking."""
        super().reset()
        self.total_cost = 0.0
        self.call_count = 0

    def get_cost_summary(self) -> dict:
        """Get a summary of costs incurred by this agent.

        Returns:
            Dictionary with total_cost, call_count, and avg_cost_per_call
        """
        avg_cost = self.total_cost / self.call_count if self.call_count > 0 else 0.0
        return {
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "avg_cost_per_call": avg_cost,
        }
