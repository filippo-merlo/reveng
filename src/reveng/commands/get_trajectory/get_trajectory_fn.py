"""Generate and save agent trajectories in navigation environments with detailed token-level analysis."""

import numpy as np
import json
import logging
from typing import Literal
from transformers import AutoTokenizer, PreTrainedTokenizer
from reveng.agents.llm_agent import LLMAgent
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import FullObservabilityTextWrapper

from reveng.commands.get_trajectory.get_trajectory_utils import generate_trajectory, to_dic_list, annotate_output_tokens
from reveng.commands.get_trajectory.compact_json_encoder import CompactJSONEncoder

logger = logging.getLogger(__file__)

def get_trajectory(
    grid_size: int = 5,
    grid_complexity: float = 0.0,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    observation_placeholders: list[str] = ["grid_state"],
    output_path: str = "get_trajectory_example_output.json",
    verbose: bool = False
):
    """Generate an agent trajectory in a 2D navigation environment and save detailed results to JSON.

    Creates a Simple2D navigation environment, runs an LLM agent to generate a trajectory,
    and saves comprehensive information including grid parameters, model parameters, prompt
    template with token-level analysis, and trajectory steps with token probabilities.

    Args:
        grid_size: Size of the square grid environment.
        grid_complexity: Complexity level of obstacles in the grid (higher = more obstacles).
        max_steps_per_trajectory: Maximum number of steps to generate in the trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_name: Name of the model in format "provider/model_id".
        observation_placeholders: List of placeholder names in the prompt template.
        output_path: Path to save the output JSON file.
        verbose: If True, print detailed logging during trajectory generation.

    Returns:
        None. Results are saved to the specified output_path.

    The output JSON structure follows the format expected by the trace viewer: https://github.com/SPAR-Telos/interp/tree/trace-viewer
        - grid_params: Grid configuration (size, complexity, start/goal positions, A* distance, legend)
        - model_params: Model configuration (name, provider, sampling parameters, seed)
        - prompt: Prompt template with token-level annotations (prefix, suffix, placeholder tokens)
        - steps: List of trajectory steps, each containing:
            - step_id: Step number
            - grid_state: Grid visualization as list of strings
            - grid_state_tokens: Tokenized grid state with annotations
            - prompt_suffix_tokens: Tokenized prompt suffix
            - agent_action: Action taken by the agent
            - output_text: Model's generated output text
            - output_tokens: Tokenized output with probabilities and annotations
    """
    base_env = FullObservabilityTextWrapper(
        Simple2DNavigationEnv(size=grid_size, complexity=grid_complexity)
    )
    agent = LLMAgent(model_name)
    model_id = "/".join(model_name.split("/")[1:])
    provider = model_name.split("/")[0]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    traj = generate_trajectory(
        env=base_env,
        agent=agent,
        max_steps_per_trajectory=max_steps_per_trajectory,
        generation_kwargs={
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
            "seed": seed
        },
        verbose=verbose
    )

    grid_params = {}

    grid_params["grid_width"] = grid_size
    grid_params["grid_height"] = grid_size
    grid_params["grid_complexity"] = grid_complexity
    grid_params["fully_observable"] = True
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata["agent_start_coordinates"]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = base_env.grid_cells

    grid_symbols = [cell["symbol"] for cell in base_env.grid_cells.values()]

    model_params = {}

    model_params["model_id"] = model_id
    model_params["provider"] = provider
    model_params["interface"] = "litellm"
    model_params["n_interactions_in_context"] = 0
    model_params["max_tokens"] = max_tokens
    model_params["max_steps_per_trajectory"] = max_steps_per_trajectory
    model_params["temperature"] = temperature
    model_params["reasoning_effort"] = reasoning_effort
    model_params["top_p"] = top_p
    model_params["top_logprobs"] = top_logprobs
    model_params["seed"] = seed

    prompt = {}

    render_kwargs = {"grid_state": "{{grid_state}}"}
    template = agent._template.render(**render_kwargs)
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)

    # TODO: Handle multiple observation placeholders if needed (for now, we default to [0] only)
    observation_placeholder = "{{" + observation_placeholders[0] + "}}"
    prefix, suffix = formatted_template.split(observation_placeholder)
    raw_prefix, raw_suffix = template.split(observation_placeholder)
    prompt["prompt_template"] = formatted_template
    prompt["prompt_template_n_tokens"] = len(template_tokens)
    prompt["prompt_prefix_tokens"] = to_dic_list(prefix, tokenizer)
    raw_prefix_tokens = to_dic_list(raw_prefix, tokenizer)
    start_raw_prefix_idx = len(prompt["prompt_prefix_tokens"]) - len(raw_prefix_tokens)
    for i in range(start_raw_prefix_idx):
        prompt["prompt_prefix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_prefix_n_tokens"] = len(prompt["prompt_prefix_tokens"])
    prompt["prompt_placeholder_tokens"] = to_dic_list(observation_placeholder, tokenizer, groups=["prompt", "placeholder"])
    prompt["prompt_placeholder_n_tokens"] = len(prompt["prompt_placeholder_tokens"])
    prompt["prompt_suffix_tokens"] = to_dic_list(suffix, tokenizer)
    raw_suffix_tokens = to_dic_list(raw_suffix, tokenizer)
    start_raw_suffix_idx = len(prompt["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
    for i in range(len(prompt["prompt_suffix_tokens"]) - start_raw_suffix_idx, len(prompt["prompt_suffix_tokens"]) - 1):
        prompt["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    steps = []

    for step_id, traj_step in enumerate(traj.steps):
        step_dic = {}
        step_dic["step_id"] = step_id
        step_dic["grid_state"] = traj_step.observation.split("\n")
        step_dic["grid_state_tokens"] = to_dic_list(traj_step.observation, tokenizer, groups=["prompt", "grid_state"])
        step_dic["grid_state_n_tokens"] = len(step_dic["grid_state_tokens"])

        for i, t in enumerate(step_dic["grid_state_tokens"]):
            if any(sym in t["token"] for sym in grid_symbols):
                step_dic["grid_state_tokens"][i]["token_groups"] += ["grid_tile"]
        
        step_dic["prompt_suffix_tokens"] = prompt["prompt_suffix_tokens"]
        step_dic["prompt_suffix_n_tokens"] = len(step_dic["prompt_suffix_tokens"])
        step_dic["agent_action"] = traj_step.metadata["action"]

        out_tokens = [t["token"] for t in traj_step.metadata["logprobs"]]
        step_dic["output_text"] = tokenizer.convert_tokens_to_string(out_tokens)
        step_dic["output_tokens"] = to_dic_list(step_dic["output_text"], tokenizer, groups=["output"])
        step_dic["output_n_tokens"] = len(step_dic["output_tokens"])
        step_dic["output_tokens"] = annotate_output_tokens(model_name, step_dic["output_tokens"])
        
        for i, t in enumerate(step_dic["output_tokens"]):
            if "top_logprobs" not in traj_step.metadata["logprobs"][i] or "template" in t["token_groups"]:
                continue
            curr_probs = {}
            for logprob_dic in traj_step.metadata["logprobs"][i]["top_logprobs"]:
                curr_probs[logprob_dic["token"]] = np.round(np.exp(logprob_dic["logprob"]), 4)
            step_dic["output_tokens"][i]["probabilities"] = curr_probs

        steps.append(step_dic)


    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt,
        "steps": steps
    }
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)
