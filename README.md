````markdown
# blackjack_predictor

Prediction for Blackjack

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Reproducible setup

1. Install dependencies with `uv sync`.
2. Either place `blkjckhands.csv` in the repository root or `data/raw/`, or set one of these environment variables so the pipeline can download it from Google Drive:
	- `BLACKJACK_GDRIVE_URL`
	- `BLACKJACK_GDRIVE_FILE_ID`
3. Preprocess the data with `uv run invoke preprocess-data`.
4. Train the model with `uv run invoke train`.

Example PowerShell setup:

```powershell
$env:BLACKJACK_GDRIVE_FILE_ID = "your-google-drive-file-id"
uv run invoke preprocess-data
uv run invoke train
```

````
