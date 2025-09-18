# Contributing to telos-reveng

## Prerequisites

- Python 3.10 or higher (recommended: 3.12)
- [uv package manager](https://docs.astral.sh/uv/) (install via the official installer, e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh` and follow the prompts)

## Development Setup

1. **Install uv** following the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2. **Clone the repository**:
   ```bash
   git clone https://github.com/SPAR-Telos/reveng
   cd reveng
   ```

3. **Create the virtual environment and install project dependencies**:
   ```bash
   uv sync
   ```

   This will:
   - Resolve dependencies declared in `pyproject.toml` / `uv.lock`
   - Create a `.venv` folder at the project root (managed by uv)

4. **Activate the environment (optional)**:
   ```bash
   source .venv/bin/activate
   ```

5. **Install pre-commit hooks**:
   ```bash
   uvx pre-commit install
   ```

`uvx` is shorthand for `uv tool run`; it installs and caches development tools without modifying the project's dependency lists.

## Editor Configuration

### VS Code Setup with Ruff

To configure VS Code to format your Python code on save using Ruff via `uvx ruff format`:

1. **Install Ruff and uvx**:
   ```bash
   uv tool install ruff@latest
   ```

2. **Install the Ruff VS Code Extension**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Ruff" and install the extension by `charliermarsh`

3. **Configure VS Code Settings**:
   Open your VS Code settings (Ctrl+Shift+P → "Preferences: Open Settings (JSON)") and add:
   ```json
   {
     "[python]": {
       "editor.formatOnSave": true,
       "editor.defaultFormatter": "charliermarsh.ruff",
       "editor.codeActionsOnSave": {
         "source.fixAll.ruff": "always",
         "source.organizeImports.ruff": "always"
       }
     },
     "ruff.path": ["uvx", "ruff"]
   }
   ```

This configuration will automatically format your Python code and organize imports every time you save a file, using Ruff via `uvx ruff format`.

## Project Structure

```
├── src/                     # Main package source code
├── tests/                   # Test files
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Locked dependency resolution managed by uv
├── pre-commit-config.yaml   # Pre-commit hooks configuration
├── Makefile                 # Legacy helper targets (optional)
└── README.md                # Project overview
```

## Development Workflow

### Common Commands

Use `uv sync` to keep the local environment in sync with the lockfile and `uvx` for project tooling:

```bash
uv sync --locked            # Install project dependencies exactly as pinned in uv.lock
uvx pytest -n auto -vv      # Run the full test suite
uvx ruff check .            # Run linting without fixing
uvx ruff format .           # Apply formatting
uvx pre-commit run --all-files  # Execute every pre-commit hook locally
uv run python -m reveng      # Execute the package entry point (example)
```

### Running Scripts with uv

- `uv run python path/to/script.py`: Run a script that lives inside the repository using the project environment.
- `uv run -m package.module`: Execute a module as if with `python -m`.
- `uv tool run <command>` / `uvx <command>`: Invoke ad-hoc tooling (for example, `uvx rich-cli tree`).
- Add `--` to forward arguments to the script, e.g. `uv run python scripts/train.py --epochs 10`.

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Ruff**: For linting and code formatting (configured in `pyproject.toml`)
- **Pre-commit hooks**: Automatically run checks before commits
- **Pytest**: For running tests with parallel execution support

### Testing

Run tests using:
```bash
uvx pytest -n auto -vv
```

This runs `pytest` with:
- Parallel execution (`-n auto`)
- Verbose output (`-vv`)
- (Optional) configuration from `pyproject.toml` if present

Test markers available:
- `slow`: For time-intensive tests
- `require_cuda_gpu`: For tests requiring CUDA GPU

### Code Style

The project follows these style guidelines:
- Line length: 119 characters
- Python 3.10+ syntax
- Google-style docstrings
- Import sorting with isort

Before committing, run:
```bash
uvx ruff format .
uvx ruff check .
```

### Pre-commit Hooks

Install the hooks with `uvx pre-commit install`. They will:
- Check and fix trailing whitespace
- Validate TOML files
- Check for merge conflicts
- Run Ruff formatting and linting
- Run tests
- Clean temporary files

## Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality in the `tests/` directory

4. **Run the test suite**:
   ```bash
   uvx pytest -n auto -vv
   ```

5. **Check code style**:
   ```bash
   uvx ruff check .
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   ```
   Pre-commit hooks will automatically run and may modify files.

7. **Push and create a pull request**

## Dependencies

### Core Dependencies
- `transformers>=4.42.0`: Hugging Face transformers library
- `nnsight>=0.5.3`: Neural network interpretability toolkit
- `torch>=2.0.1`: PyTorch deep learning framework
- `typer>=0.17.4`: CLI framework for building command-line interfaces
- `toml>=0.10.2`: TOML file parsing

### Optional Dependencies
- `data`: For data processing (`datasets`, `pandas`)
- `lint`: For development tools (`pytest`, `ruff`, `pre-commit`)
- `notebook`: For Jupyter notebook support (`ipykernel`, `ipywidgets`)

### Adding New Dependencies

1. Activate the environment if needed: `source .venv/bin/activate`.
2. Add the package with uv:
   ```bash
   uv add <package-name>
   ```
   Use `--dev` to add development-only dependencies, and `--group <name>` to target an optional dependency group.
3. Regenerate the lockfile to capture the new requirement:
   ```bash
   uv lock
   ```
4. Re-sync the environment so the dependencies are installed locally:
   ```bash
   uv sync
   ```
5. Commit the updated `pyproject.toml` and `uv.lock` together.

## Package Management

To update dependencies:

1. **Modify `pyproject.toml`** with new dependencies or version constraints
2. **Refresh the lockfile**:
   ```bash
   uv lock --upgrade
   ```
   Use `uv lock --upgrade-package <name>` for targeted upgrades.
3. **Install the updated dependencies locally**:
   ```bash
   uv sync
   ```
4. Commit both `pyproject.toml` and `uv.lock`.

## Makefile

The repository retains a Makefile with convenience wrappers around the `uv` commands above. These are optional but can speed up repeated tasks:

- `make uv-download`: Download and install uv, then create the `.venv` environment.
- `make install`: Activate the environment and install runtime dependencies (no extras) via `uv pip`.
- `make install-dev`: Install development extras, refresh pre-commit hooks, and install the package in editable mode.
- `make test`: Run the pytest suite using the virtual environment's interpreter.
- `make check-style` / `make fix-style`: Run Ruff in check or fix mode through the project interpreter.
- `make clean`: Remove build artifacts and Python bytecode caches.
- `make update-deps`: Rebuild the requirements export files if you need them (these are generated from `pyproject.toml`).

All Makefile commands assume the `.venv` created by `uv sync` and `uv` itself are already available.
