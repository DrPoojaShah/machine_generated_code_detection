# Machine-Generated Code Detection

An end-to-end machine learning project for classifying source-code snippets as **human-written (`0`)** or **machine-generated (`1`)**. The workflow covers exploratory data analysis, handcrafted structural features, TF-IDF text representation, baseline and advanced models, language-level evaluation, explainability, robustness analysis, and submission generation.

## Project Overview

This project was completed for **AIT 626** by **Team 7**. It investigates whether statistical, lexical, and structural patterns in source code can distinguish human-written code from AI-generated code.

### Main objectives

- Explore class, programming-language, generator, and code-length distributions.
- Extract structural characteristics from source-code snippets.
- Combine TF-IDF features with handcrafted numerical features.
- Compare Logistic Regression, Linear SVM, Naive Bayes, XGBoost, and a voting ensemble.
- Evaluate performance across programming languages and code generators.
- Analyze model errors, calibration, explainability, and adversarial robustness.
- Generate predictions for an unlabeled test set.

## Pipeline

1. Data loading and validation
2. Exploratory data analysis
3. Handcrafted feature engineering
4. TF-IDF vectorization
5. Baseline model training
6. Combined-feature modeling
7. XGBoost and voting ensemble
8. Per-language and per-generator analysis
9. PCA, t-SNE, feature-weight, and error visualizations
10. Submission generation and code-snippet prediction demo

## Feature Engineering

The project extracts 14 structural features from each code snippet:

- Comment ratio
- Average line length
- Blank-line ratio
- Vocabulary richness
- Maximum indentation depth
- Function count
- Docstring presence
- Try-block count
- Type-hint count
- Print/log statement count
- Magic-number count
- Keyword diversity
- Unique identifier ratio
- Special-character rate

These features are combined with TF-IDF representations using a sparse feature matrix.

## Models

- Logistic Regression
- Linear Support Vector Machine
- Multinomial Naive Bayes
- XGBoost
- Soft Voting Ensemble using calibrated SVM and Logistic Regression

The notebook compares accuracy, macro precision, macro recall, and macro F1 score on the validation set.

## Research Questions

1. Which handcrafted features best discriminate human-written and AI-generated code?
2. Is machine-generated code detection language-agnostic or language-specific?
3. How vulnerable is the detector to simple obfuscation attacks?
4. Which AI generators are hardest to detect, and why?

## Repository Structure

```text
machine-generated-code-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── AIT626_final_project_colab_export.py
├── src/
│   └── feature_engineering.py
└── results/
    └── README.md
```

## Dataset

The pipeline expects the following Parquet files:

```text
train.parquet
validation.parquet
test.parquet
```

Expected columns include:

- `code`
- `label` for training and validation data
- `language`
- `generator`
- `id` or `ID` for test data

The dataset is not included in this repository. Place authorized copies in the `data/` directory and update file paths as needed.

## Installation

```bash
git clone https://github.com/DrPoojaShah/machine-generated-code-detection.git
cd machine-generated-code-detection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The uploaded project was developed in Google Colab. Open or convert the script in `notebooks/` into a notebook, update the dataset paths, and run the sections in order.

For local execution, remove or comment out Colab-specific commands such as:

- `!pip install ...`
- `drive.mount(...)`
- `files.upload()`
- `files.download(...)`

## Generated Outputs

The workflow can generate:

- Class-distribution charts
- Language and generator distributions
- Code-length distributions
- Top TF-IDF terms by class
- Feature-correlation heatmaps
- Human-vs-AI radar charts
- XGBoost feature importance
- Model-comparison charts
- Confusion matrices
- Per-language accuracy plots
- PCA and t-SNE projections
- Error-analysis plots
- Calibration diagrams
- `submission.csv`

## Key Technologies

Python, Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Matplotlib, Seaborn, TF-IDF, NLP, Google Colab

## Future Improvements

- Refactor the full notebook into reusable training and inference modules.
- Add automated hyperparameter tuning.
- Evaluate transformer-based code representations.
- Add stronger adversarial transformations.
- Track experiments with MLflow or Weights & Biases.
- Add unit tests and continuous integration.
- Package the trained pipeline as a web application or API.

## Authors

**Team 7 — AIT 626 Final Project**

Repository maintained by **Dr. Pooja Shah**.
