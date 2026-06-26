# Contributing to Agentic Clustering & Auto-Labeling

Thank you for your interest in contributing! This project welcomes contributions of all kinds — bug reports, feature ideas, documentation improvements, and code.

## Getting Started

1. **Fork** the repository and **clone** your fork:

   ```bash
   git clone https://github.com/<your-username>/Agentic_Labelling.git
   cd Agentic_Labelling
   ```

2. **Set up the environment:**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Configure your API key** in a local `.env` file (never commit it):

   ```bash
   LLM_API_KEY=sk-...
   ```

## Making Changes

1. Create a feature branch:

   ```bash
   git checkout -b feature/short-description
   ```

2. Make your changes, keeping them focused and minimal.
3. Run the pipeline / relevant experiments to confirm nothing breaks:

   ```bash
   python run_pipeline.py --no-ui
   ```

4. Commit with a clear message and open a pull request against `main`.

## Pull Request Guidelines

- Describe **what** the change does and **why**.
- Keep PRs scoped to a single concern.
- Do not include secrets, API keys, large data files, or generated artifacts (`outputs/`, `data/processed/`, recordings).
- Update the README or docs when behavior changes.

## Reporting Issues

When filing a bug, please include:

- A short description of the problem.
- Steps to reproduce.
- Expected vs. actual behavior.
- Environment details (OS, Python version).

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
