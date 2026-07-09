# reveng

**A Behavioural and Representational Evaluation of Goal-Directedness**

Studies whether `gpt-oss-20b` acts as a goal-directed agent in 2D grid navigation tasks, via behavioral robustness tests and linear/MLP probes on residual stream activations.

## Commands

```bash
# Install (requires Python ≥3.12, uses uv)
make install          # uv sync

# Run tests
make test             # pytest tests/

# Lint / format
make lint             # ruff + mypy
make format           # black + isort

# CLI — collect trajectories
reveng-cli get_trajectories [args]
reveng-cli get_trajectories_multiple_per_grid [args]
reveng-cli get_trajectory [args]

# Key-door / counterfactual variants
reveng-cli get_trajectories_key_door [args]
reveng-cli generate_counterfactual_grids [args]
```

## Architecture

```
src/reveng/
├── datatypes.py               # Step, Trajectory, Action enum, JSON encoder
├── llm_interface.py           # LiteLLM wrapper (multi-provider, cost tracking, logprobs)
├── agents/
│   ├── agent_abc.py           # Agent base class (select_action / update / reset)
│   ├── llm_agent.py           # LLM agent — Jinja2 prompting, nnsight attention capture, MXFP4 weights
│   ├── random_agent.py
│   └── alpha_start_agent.py   # A*-optimal baseline
├── environment_generator/
│   ├── rooms_minigrid.py      # Base grid worlds, wall density d∈[0,1]
│   ├── key_minigrid.py        # Key-door rooms variant
│   ├── coin_minigrid.py       # Implicit goal variant
│   ├── env_transformations.py # Iso-difficulty transforms: Rotate/Reflect/Transpose/StartGoalSwap
│   ├── setbacks.py            # Environment perturbations for robustness tests
│   └── wrappers/              # Text, RGB, fog-of-war observation wrappers
├── templates/                 # Jinja2 prompt templates (full/partial observability variants)
├── trajectory_generator/
│   └── trajectory_generator.py  # Rollout loop → JSON trajectory files
├── experiments/
│   ├── full_policy_experiments.py
│   ├── iso_difficulty_policy_experiments.py
│   ├── instrumental_goals_experiments.py
│   ├── policy_elicitation.py
│   └── generate_dataset*.py
├── analysis/
│   ├── behavioural_analysis/  # Optimality metrics, model comparisons
│   ├── decoded_grids_analysis/ # Probe-decoded spatial representations, LaTeX tables
│   └── key_door_env_analysis/
└── commands/cli.py            # Tyro-based CLI entry point
```

## Core data types

```python
Action       # Enum: LEFT, RIGHT, UP, DOWN
Step         # observation, action, reward, metadata, agent_pos
Trajectory   # list[Step] + final_reward; serializes to JSON
```

Trajectories are saved as JSON and are the canonical intermediate format between data collection and analysis.

## Experiment workflow

1. **Generate grids** — procedural MiniGrid environments (size 7–15, wall density 0–1)
2. **Collect trajectories** — `reveng-cli get_trajectories` runs LLM agent via LiteLLM API; saves JSON
3. **Behavioral evaluation** — measure action optimality vs. A*; test iso-difficulty transforms
4. **Representational analysis** — train linear/MLP probes on residual stream activations decoded from `nnsight` hooks; decode wall positions, goal, agent position (cognitive maps)
5. **Analysis scripts** — `src/reveng/analysis/` produces LaTeX tables and figures

## LLM agent notes

- Uses `nnsight` to hook into `gpt-oss-20b` (HF model) and capture attention weights
- Model loaded with MXFP4-quantized weights
- Prompts are Jinja2 templates in `src/reveng/templates/`; select template via full/partial observability flag
- LiteLLM handles multi-provider routing (OpenAI-compat, Together, Fireworks)

## Tests

```
tests/
├── test_generate_trajectory_reset.py   # Trajectory generation + agent reset
├── test_goal_exists.py                 # Grid validity checks
├── test_is_solvable.py                 # Solvability of procedural grids
└── test_generate_counterfactual_grids.py
```

## Environment variables

From `.env.example` — copy to `.env`:
- `HF_TOKEN` — HuggingFace token for model access
- API keys for LiteLLM providers (Together, Fireworks, etc.)
