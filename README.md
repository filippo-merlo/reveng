# A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents

*Raghu Arghal, Fade Chen, Niall Dalton, Evgenii Kortukov, Calum McNamara, Angelos Nalmpantis, Moksh Nirvaan, Gabriele Sarti, Mario Giulianelli*

[![Paper](https://img.shields.io/badge/arXiv-2602.08964-b31b1b.svg)](https://arxiv.org/abs/2602.08964)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://www.python.org/)

> **Abstract:** Understanding whether and how language model agents pursue goals is essential for ensuring the safety of AI systems deployed to act autonomously in the world. In this work, we study goal-directedness in a language model agent, GPT-OSS-20B, as it navigates procedurally generated 2D grid environments. We operationalize goal-directedness behaviourally--through the optimality of an agent's actions and through its robustness to environment perturbations--and representationally--by probing the agent's internal activations for evidence of structured spatial knowledge. Our behavioural evaluation reveals that GPT-OSS-20B generally acts as a goal-directed agent, navigating towards the goal across a range of grid sizes with above-chance optimality. Representationally, linear and MLP probes trained on the agent's residual stream activations at intermediate layers uncover internal representations that partially encode the spatial layout of the environment, including the positions of walls, the goal, and the agent itself. Taken together, our results indicate that GPT-OSS-20B can act as a goal-directed agent through reliance on internal representations that partially but non-trivially encode the spatial features of its environment.

## Installation

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SPAR-Telos/reveng
cd reveng
uv sync
```

Copy the example env file and fill in your API key for the model provider:

```bash
cp .env.example .env
# then edit .env and set TOGETHER_AI_API_KEY=...
```

## Trajectory Collection

Trajectories for the **Key-Door** environment were collected with:

```bash
reveng-cli get_trajectories_key_door_env \
    --rooms-per-side-options 2 \
    --add-door-key-options True \
    --model-names "together_ai/openai/gpt-oss-20b" \
    --output-dir "./trajectories_key_door" \
    --num-examples 100 \
    --max-workers 2 \
    --reasoning-effort "medium"
```

Trajectories for the **Key-NoDoor** environment were collected with:

```bash
reveng-cli get_trajectories_key_door_env \
    --rooms-per-side-options 2 \
    --add-door-key-options True \
    --remove-door-from-env-options True \
    --model-names "together_ai/openai/gpt-oss-20b" \
    --output-dir "./trajectories_no_door" \
    --num-examples 100 \
    --max-workers 10 \
    --reasoning-effort "medium"
```

To list all available CLI subcommands and options:

```bash
reveng-cli --help
```

## Repository Layout

All source lives under `src/reveng/`:

- `agents/` — agent implementations: `llm_agent.py` (the LLM-driven agent and prompt templates), plus `random_agent.py` and `alpha_start_agent.py` baselines, sharing the `agent_abc.py` interface.
- `environment_generator/` — procedurally generated MiniGrid environments (`rooms_minigrid.py`, `key_minigrid.py`, `coin_minigrid.py`, `custom_minigrid.py`), grid transformations and perturbation setbacks (`env_transformations.py`, `setbacks.py`), plotting (`env_plots.py`), and observation `wrappers/` for text, RGB, and fog-of-war views.
- `templates/` — Jinja2 prompt templates for full / partial observability, instrumental-goal variants, and the coin-task ablations (only-legend, no-mechanisms, reward).
- `trajectory_generator/` — core rollout loop (`trajectory_generator.py`) and perturbation logic (`perturbations.py`, `perturbation_helpers.py`).
- `commands/` — `reveng-cli` entry points (`cli.py`) that drive trajectory collection and counterfactual grid generation; `get_trajectory/` holds the trajectory-collection runners and the rate limiter used for parallel API calls.
- `experiments/` — experiment runners (instrumental goals, coin task, full / partial / iso-difficulty policy experiments, policy elicitation), dataset generation, and per-experiment result directories with plotting scripts.
- `analysis/` — post-hoc analysis grouped by topic: `behavioural_analysis/` (trajectory optimality, model comparisons, uncertainty), `decoded_grids_analysis/` (probe-decoded grid metrics and LaTeX table generation), and `key_door_env_analysis/` (key/door and 2-path environment breakdowns and plots).
- `datatypes.py`, `llm_interface.py` — shared dataclasses (trajectories, steps) and the LiteLLM-based completion wrapper.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

See [LICENSE](LICENSE).

## Citation

```bibtex
@article{arghal-etal-2026-behavioural,
    title={A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents},
    author={Raghu Arghal and Fade Chen and Niall Dalton and Evgenii Kortukov and Calum McNamara and Angelos Nalmpantis and Moksh Nirvaan and Gabriele Sarti and Mario Giulianelli},
    year={2026},
    journal={arXiv preprint arXiv:2602.08964},
    url={https://arxiv.org/abs/2602.08964}
}
```
