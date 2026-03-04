# AI Analyst (Python + Machine Learning)

This project provides a ready-to-run **AI Analyst** script that trains a machine-learning model on a CSV dataset and returns key insights.

## What it does

- Loads a CSV file.
- Detects if your target is a **classification** or **regression** problem.
- Builds a preprocessing + Random Forest ML pipeline.
- Trains/test-splits data automatically.
- Prints model performance metrics.
- Prints top influential features using permutation importance.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python ai_analyst.py <path_to_csv> <target_column>
```

Example:

```bash
python ai_analyst.py data/iris.csv species
```

Optional:

```bash
python ai_analyst.py data/sales.csv revenue --test-size 0.25
```

## Output

The script prints:

- Task type (classification/regression)
- Main quality metrics (Accuracy + report OR MAE/R²)
- Top 10 influential features
