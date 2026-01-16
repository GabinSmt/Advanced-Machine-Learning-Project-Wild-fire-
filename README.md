# XGBoost from Scratch for Wildfire Prediction

This repository contains a complete implementation of the **XGBoost algorithm from scratch**, developed in the context of the *Advanced Machine Learning* course taught at ENSAE. The project focuses on understanding the theoretical foundations of gradient boosting with decision trees and applying the resulting model to a real-world, large-scale dataset.


## Dataset

The experiments conducted in this project are based on the **US Wildfire Dataset (2014–2025)**, publicly available on Kaggle:

- https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset

This dataset combines wildfire ignition records with meteorological and environmental variables derived from the GRIDMET climatology database. It provides a challenging and realistic benchmark for evaluating tree-based machine learning models.

The dataset `df_aggregated.csv` was created using the script `df_aggregated.py`, which processes and aggregates the raw data originating from the aforementioned dataset. This preprocessing step includes data cleaning, aggregation, and formatting in order to obtain a dataset suitable for model training and evaluation.

## Repository Structure

- `xgboost_from_scratch.py` contains the full implementation of our XGBoost model developed from scratch.
- `use_case.ipynb` is the core of the project. This notebook includes the complete data processing pipeline, the application of our custom XGBoost model to the wildfire dataset, and a comparative analysis with existing machine learning models.

## Project Objectives

The main objectives of this project are:

- To **implement the XGBoost algorithm from scratch**, without relying on existing machine learning libraries.
- To closely follow the original mathematical formulation of XGBoost, including second-order optimization, regularization, greedy split selection, and approximate split finding.
- To **apply the implemented model to the US wildfire dataset** and evaluate its predictive performance.
- To **compare the performance of our custom implementation** with existing machine learning models and reference implementations.

## Methodology

Our implementation is primarily inspired by the original XGBoost paper:

> **Tianqi Chen and Carlos Guestrin**  
> *XGBoost: A Scalable Tree Boosting System*

The algorithm is built around three main components:
- **Node**: representing individual decision points in a tree,
- **XGBoostTree**: constructing a regression tree using gradients and Hessians,
- **XGBoostClassifier**: implementing the full gradient boosting procedure.

Special attention is given to regularization, greedy split finding, and the use of approximate algorithms such as the **weighted quantile sketch**, in order to remain faithful to the original method.

## Documentation and References

A detailed description of the theoretical foundations, implementation choices, and experimental results can be found in our accompanying report:

- https://overleaf.enst.fr/project/6901e9ac9912eb0d35ba0d1d

This document includes:
- A rigorous derivation of the XGBoost objective,
- An analysis of the dataset and feature correlations,
- A comparison with existing models from the literature.
