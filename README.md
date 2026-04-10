# XGBoost from Scratch for Wildfire Prediction

This repository contains a complete implementation of the **XGBoost algorithm from scratch**, originally developed in the context of the *Advanced Machine Learning* course taught at ENSAE. 

The project has now transitioned into an **MLOps pipeline**, focusing not only on understanding the theoretical foundations of gradient boosting, but also on encapsulating the resulting model into a deployable infrastructure.

## Project Objectives

The main objectives of this project are twofold:

**1. Machine Learning Core:**
- To implement the XGBoost algorithm from scratch, without relying on existing ML libraries.
- To closely follow the original mathematical formulation (second-order optimization, regularization, greedy split selection, and approximate split finding).
- To apply the model to the US wildfire dataset and evaluate its predictive performance against standard implementations.

**2. MLOps & Industrialization:**
- To structure the project following the Data Science Cookiecutter standard.
- To separate code from data by hosting datasets on an **S3 MinIO** bucket.
- To track experiments and model registries using **MLflow**.
- To containerize the application and serve predictions via a **FastAPI** interface.
- To automate deployment on a **Kubernetes** cluster using a **GitOps** approach (ArgoCD).

## Repository Structure

The repository follows standard MLOps practices to ensure reproducibility and clean collaboration:

```text
.
├── data/
├── k8s/
├── requirements.txt
├── notebooks/
│   └── main.ipynb
└── src/
    ├── api/
    ├── data/
    └── models/
```

## Dataset

The experiments are based on the **US Wildfire Dataset (2014–2025)**, publicly available on [Kaggle](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset). It combines wildfire ignition records with meteorological and environmental variables derived from the GRIDMET climatology database.

**Important:** To keep the repository lightweight, the aggregated dataset (`df_aggregated.csv`) is hosted on our S3 MinIO bucket. It is automatically fetched by the scripts located in `src/data/` using the credentials defined in your `.env` file.

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
```

### 2. Environment Variables
Create your local environment file:
```bash
cp .env.example .env
```
Open `.env` and fill in your specific credentials (S3 keys, MLflow URIs) provided by the team administrator.

### 3. Code Quality (Ruff)
This project enforces strict code formatting using **Ruff**. Before committing any Python code, run the following commands at the root of the project:
```bash
ruff check . --fix   # Fixes imports and minor errors
ruff format .        # Formats the code strictly
```

## Git Workflow & Branching Convention

To avoid conflicts and maintain a clean history, **pushing directly to the `main` branch is strictly disabled**. All changes must go through a Pull Request (PR) and require at least one approval.

**Branch Naming Convention:**
Please name your branches using the following format: `type/firstname-action`
- `feat/` : New feature (e.g., `feat/alice-mlflow-integration`)
- `fix/` : Bug fix (e.g., `fix/bob-s3-connection`)
- `docs/` : Documentation updates (e.g., `docs/chloe-update-readme`)
- `chore/` : Maintenance or configuration (e.g., `chore/david-docker-setup`)
- `refactor/`: Code reorganization without changing functionality

## Methodology & References

Our implementation is primarily inspired by the original XGBoost paper:
> **Tianqi Chen and Carlos Guestrin** > *XGBoost: A Scalable Tree Boosting System*

A detailed description of the theoretical foundations, implementation choices, and experimental results can be found in our accompanying academic report:
- [Overleaf Report Link](https://overleaf.enst.fr/project/6901e9ac9912eb0d35ba0d1d)

This document includes a rigorous derivation of the XGBoost objective, an analysis of the dataset, and a comparison with existing models from the literature.