````markdown
# blackjack_predictor

Prediction for Blackjack

## Project structure

The directory structure of the project looks like this:
```txt
├── .devcontainer/                 # Dev container setup scripts
├── .dvc/                          # DVC configuration
├── .github/                       # GitHub workflows, agents, and prompts
│   ├── workflows/
│   ├── agents/
│   ├── dependabot_pip.yaml
│   └── dependabot_uv.yaml
├── configs/                       # Configuration files
├── data/                          # Raw and processed datasets
│   ├── processed/
│   └── raw/
├── dockerfiles/                   # Service and training Dockerfiles/scripts
│   ├── api_uv.dockerfile
│   ├── backend.dockerfile
│   ├── evaluate_uv.dockerfile
│   ├── frontend.dockerfile
│   ├── train_uv.dockerfile
│   ├── evaluate.sh
│   └── train.sh
├── docs/                          # MkDocs documentation
│   ├── mkdocs.yaml
│   ├── README.md
│   └── source/
│       └── index.md
├── gcloud/                        # Cloud Build, Vertex AI, and monitoring config
│   ├── monitoring/
│   │   └── alert_policy.json
│   └── vertex/
│       ├── blackjack_train_custom_job.yaml
│       ├── cloudbuild.yaml
│       └── scripts/
├── models/                        # Exported and trained model artifacts
├── notebooks/                     # Jupyter notebooks
├── outputs/                       # Hydra/experiment outputs
├── reports/                       # Reports and generated figures
│   └── figures/
├── samples/                       # Example frontend/backend apps
├── src/                           # Python package source
│   └── blackjack_predictor/
│       ├── api/
│       ├── data_/
│       ├── helpers/
│       ├── models/
│       ├── monitoring/
│       ├── performance/
│       ├── evaluate.py
│       ├── run_sweep.py
│       ├── tasks.py
│       └── train.py
├── tests/                         # Unit, integration, and performance tests
│   ├── performancetests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_onnx_alignment.py
│   ├── test_preprocessing.py
│   ├── test_production.py
│   └── test_training_procedure.py
├── Dockerfile
├── LICENSE
├── README.md                      # Project README
├── data.dvc
├── main.py
├── policy.json
├── pyproject.toml                 # Python project metadata
├── tasks.py                       # Invoke task entrypoints
└── uv.lock
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


## GCP training and deployment

The repository includes a Cloud Build pipeline and a Vertex AI custom job definition for reproducible cloud training.

1. Build and deploy the training/API containers:

```bash
gcloud builds submit --config gcloud/vertex/cloudbuild.yaml
```

2. Launch training on Vertex AI:

```bash
bash gcloud/vertex/scripts/submit_vertex_training.sh
```

3. Verify the deployed Cloud Run API:

```bash
bash gcloud/vertex/scripts/verify_cloud_run.sh
```

The GitHub workflow `.github/workflows/gcp_train_deploy.yaml` runs the same path: it builds both containers, deploys the
FastAPI service to Cloud Run, verifies `/health`, uploads the verification output, and submits the Vertex AI custom job.

````
