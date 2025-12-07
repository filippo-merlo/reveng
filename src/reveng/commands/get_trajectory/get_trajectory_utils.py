"""Utility functions for trajectory generation, token processing, and LLM completion handling."""

import re
import json
import traceback
import logging
from typing import Any
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_random_exponential
from litellm import completion, completion_cost

from reveng.agents.alpha_start_agent import AlphaStarAgent
from reveng.agents.llm_agent import LLMAgent
from reveng.datatypes import Step, Trajectory
from reveng.trajectory_generator.trajectory_generator import generate_one_trajectory
from reveng.environment_generator.wrappers.text_obs_wrapper import FullObservabilityTextWrapper

logger = logging.getLogger(__file__)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=5, max=120),
    reraise=True,
)
def completion_with_retry(**generation_kwargs) -> tuple[Any, float]:
    """Call LLM completion API with automatic retry on failure.

    Wraps the litellm completion API with exponential backoff retry logic
    to handle transient failures. Also calculates the cost of the completion.

    Args:
        **generation_kwargs: Keyword arguments passed to litellm.completion().

    Returns:
        tuple[Any, float]: A tuple of (response object, completion cost in USD).

    Raises:
        Exception: Re-raises the last exception after 5 retry attempts.
    """
    try:
        response = completion(**generation_kwargs)
    except Exception as e:
        logger.error(f"Model request failed: {e}\n{traceback.format_exc()}")
        raise
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(f"Failed to calculate cost: {e}")
        cost = 0.0
    return response, cost


def parse_action(model_output: str) -> tuple[int, str]:
    """Parse the model's text output to extract the action.

    Expects the model output to be a JSON string containing an "action" key
    with one of the directional values: LEFT, RIGHT, UP, DOWN.

    Args:
        model_output: The raw text output from the model.

    Returns:
        tuple[int, str]: A tuple of (action_id, action_name) where:
            - action_id: Integer action code (0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, -1=invalid)
            - action_name: String action name or "N/A" if parsing failed.
    """
    try:
        action_dic = json.loads(model_output)
        action = action_dic["action"]
        if action == "LEFT":
            return 0, action
        elif action == "RIGHT":
            return 1, action
        elif action == "UP":
            return 2, action
        elif action == "DOWN":
            return 3, action
        else:
            return -1, "N/A"
    except:
        return -1, "N/A"


def get_astar_distance(env, observation):
    """Calculate the optimal path length using A* algorithm.

    Runs the A* pathfinding algorithm on a cloned environment to determine
    the optimal number of steps required to reach the goal from the current state.

    Args:
        env: The environment instance.
        observation: Current observation from the environment.

    Returns:
        int: The number of steps in the optimal A* path to the goal.
    """
    astar = AlphaStarAgent()
    clone_env = deepcopy(env)
    trajectory = generate_one_trajectory(
        env=clone_env,
        observation=observation,
        info={},
        agent=astar,
        max_steps_per_trajectory=env.unwrapped.width**2,
    )
    return len(trajectory.steps)


def to_dic_list(txt, tokenizer, groups = ["prompt"], start_idx = 0):
    """Convert text to a list of token dictionaries with metadata.

    Tokenizes the input text and creates a structured representation where each token
    has an ID, token string, token ID, and group labels for categorization.

    Args:
        txt: The text to tokenize.
        tokenizer: HuggingFace tokenizer instance.
        groups: List of group labels to assign to all tokens (default: ["prompt"]).
        start_idx: Starting index for token IDs (default: 0).

    Returns:
        list[dict]: List of token dictionaries, each containing:
            - id: Sequential token index
            - token: Token string representation
            - token_id: Vocabulary ID of the token
            - token_groups: List of group labels for this token
    """
    tokens = tokenizer.tokenize(txt)
    ids = tokenizer.encode(txt)

    out = []
    for i, (t, id) in enumerate(zip(tokens, ids), start=start_idx):
        out.append({
            "id": i,
            "token": t,
            "token_id": id,
            "token_groups": deepcopy(groups)
        })
    return out


def annotate_output_tokens(model_name: str, output_tokens):
    """Annotate output tokens with model-specific group labels.

    Adds semantic group labels to output tokens based on model-specific formatting
    and special tokens. Currently supports the GPT-OSS-20B model format with
    structured output channels (analysis, final, etc.).

    Args:
        model_name: Name of the model in format "provider/model_id".
        output_tokens: List of token dictionaries from to_dic_list().

    Returns:
        list[dict]: The same token list with updated token_groups containing
            labels like 'template', 'analysis', 'final', 'action' based on
            the model's output structure.

    Raises:
        NotImplementedError: If the model is not supported for annotation.
    """
    if "openai/gpt-oss-20b" in model_name:
        template_special_tokens = {'<|channel|>', '<|message|>', '<|end|>', '<|start|>', '<|return|>'}
    
        in_template_mode = True
        current_section = None
        
        for token in output_tokens:
            token_str = token['token']
            
            if token_str in template_special_tokens:
                token['token_groups'].append('template')
                
                if token_str == '<|end|>':
                    in_template_mode = True
                elif token_str == '<|message|>':
                    in_template_mode = False
                elif token_str == '<|return|>':
                    current_section = None
            
            elif in_template_mode:
                # Template mode: tokens like 'analysis', 'final', 'assistant'
                token['token_groups'].append('template')
                
                # Track which section we're about to enter
                if token_str in ['analysis', 'final']:
                    current_section = token_str
            
            else:
                # Content mode
                if current_section:
                    token['token_groups'].append(current_section)
                
                # Check for action words (case-insensitive, ignoring Ġ and Ċ)
                clean_token = token_str.replace('Ġ', '').replace('Ċ', '')
                if re.match(r'^(up|down|left|right)$', clean_token, re.IGNORECASE):
                    token['token_groups'].append('action')
        
        return output_tokens
    else:
        raise NotImplementedError(f"The selected model {model_name} is not supported for output annotation.")


def generate_trajectory(
    env: FullObservabilityTextWrapper,
    agent: LLMAgent,
    max_steps_per_trajectory: int,
    generation_kwargs: dict = {},
    metadata: dict = {},
    verbose: bool = False
):
    """Generate a complete agent trajectory in the environment.

    Runs an LLM agent through the environment, collecting observations, actions,
    rewards, and model outputs at each step. Also computes A* optimal distance
    and records agent/goal positions.

    Args:
        env: The wrapped environment instance with text observations.
        agent: The LLM agent to generate actions.
        max_steps_per_trajectory: Maximum number of steps before truncating.
        generation_kwargs: Dictionary of parameters for the LLM completion call
            (e.g., temperature, top_p, max_tokens, top_logprobs).
        metadata: Additional metadata to include (currently unused).
        verbose: If True, log detailed information during generation.

    Returns:
        Trajectory: A Trajectory object containing:
            - steps: List of Step objects with observations, actions, rewards, and metadata
            - final_reward: Total accumulated reward
            - traj_metadata: Dictionary with start/goal positions and A* distance
    """
    generation_kwargs["allowed_openai_params"] = list(generation_kwargs.keys())

    steps: list[Step] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    observation, _ = env.reset()

    traj_metadata = {}
    start_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
    goal_pos = tuple(int(x) for x in env.unwrapped.goal_pos)
    traj_metadata["agent_start_coordinates"] = start_pos
    traj_metadata["goal_coordinates"] = goal_pos
    traj_metadata["astar_distance"] = get_astar_distance(env, observation)

    while not (terminated or truncated):
        if (
            max_steps_per_trajectory is not None
            and step_count >= max_steps_per_trajectory
        ):
            break
        
        unwrapped_env = getattr(env, "unwrapped", env)
        prompt = agent._generate_action_query_prompt(unwrapped_env)
        if verbose:
            logger.info(f"Step {step_count}")
            logger.info(agent._get_text_observation(env))
        full_output, cost = completion_with_retry(
            model=agent.model_name,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            **generation_kwargs
        )
        final_output = full_output.choices[0].message.content
        action, action_name = parse_action(final_output)
        logprobs_serialized = agent._finalize_cost_and_logprobs(cost, full_output, generation_kwargs.get("top_logprobs") is not None)

        if verbose:
            print("Output text:", final_output)
            print("Predicted action:", action_name)

        metadata = agent._build_base_metadata(action, cost, logprobs_serialized)
        metadata["action"] = action_name
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        steps.append(
            Step(
                observation=str(observation),
                action=action,
                reward=float(reward),
                metadata=metadata,
                note=None,
            )
        )

        observation = next_obs
        step_count += 1
    traj_obj = Trajectory(
        steps=steps,
        final_reward=total_reward,
        traj_metadata=traj_metadata,
    )
    return traj_obj