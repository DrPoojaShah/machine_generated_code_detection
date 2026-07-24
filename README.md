# 🤖 Machine-Generated Code Detection

An end-to-end **Machine Learning** project for detecting **AI-generated** and **human-written** source code using **Natural Language Processing (NLP)**, **TF-IDF feature extraction**, handcrafted structural code features, and ensemble machine learning models.

---

# 📖 Project Overview

This project was developed as part of **AIT 626** (Machine Learning) by **Team 7**. The objective is to investigate whether lexical, statistical, and structural characteristics of source code can accurately distinguish **human-written** code from **AI-generated** code across multiple programming languages and AI code generators.

The project implements a complete machine learning workflow including exploratory data analysis, feature engineering, model development, evaluation, explainability, robustness analysis, and prediction generation.

---

# ✨ Project Highlights

- End-to-end machine learning pipeline for AI-generated code detection.
- Combined **TF-IDF text representations** with handcrafted structural code features.
- Compared multiple machine learning models including Logistic Regression, Linear SVM, Naïve Bayes, XGBoost, and a Voting Ensemble.
- Performed exploratory data analysis, explainability, robustness analysis, calibration analysis, and error analysis.
- Generated predictions for unseen source-code snippets.

---

# 🎯 Main Objectives

- Explore class, programming-language, generator, and code-length distributions.
- Extract handcrafted structural characteristics from source-code snippets.
- Combine TF-IDF features with numerical code features.
- Compare multiple machine learning algorithms.
- Evaluate performance across different programming languages and AI generators.
- Analyze prediction errors and model robustness.
- Generate predictions for unseen source-code snippets.

---

# ⚙️ Machine Learning Pipeline

1. Data Loading and Validation
2. Exploratory Data Analysis (EDA)
3. Structural Feature Engineering
4. TF-IDF Feature Extraction
5. Baseline Machine Learning Models
6. Combined Feature Modeling
7. XGBoost Training
8. Soft Voting Ensemble
9. Explainability & Visualization
10. Prediction Generation

---

# 🧠 Feature Engineering

The project extracts **14 handcrafted structural features** from each source-code snippet, including:

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

These handcrafted features are combined with **TF-IDF representations** to build a richer feature space for machine learning models.

---

# 🤖 Machine Learning Models

The following models were implemented and compared:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Multinomial Naïve Bayes
- XGBoost
- Soft Voting Ensemble (Calibrated SVM + Logistic Regression)

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

---

# ❓ Research Questions

1. Which handcrafted structural features best distinguish AI-generated code from human-written code?
2. Is AI-generated code detection language-independent or language-specific?
3. How robust is the model against simple code obfuscation?
4. Which AI code generators are most difficult to detect?

---

# 📂 Repository Structure

```text
machine_generated_code_detection/
│
├── data/
├── figures/
├── notebooks/
│   ├── AIT626_Machine_Generated_Code_Detection.ipynb
│   └── AIT626_final_project_colab_export.py
├── results/
├── src/
│   └── feature_engineering.py
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

# 📊 Dataset

The project expects the following Parquet datasets:

```text
train.parquet
validation.parquet
test.parquet
```

Required columns include:

- code
- label
- language
- generator
- id

> **Note:** The dataset is **not included** in this repository due to licensing and course restrictions.

---

# 💻 Installation

```bash
git clone https://github.com/DrPoojaShah/machine_generated_code_detection.git

cd machine_generated_code_detection

python -m venv .venv

source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

# ▶️ Usage

The project was originally developed in **Google Colab**.

To execute locally:

1. Place the datasets inside the `data/` folder.
2. Update the dataset paths.
3. Install the required packages.
4. Run the notebook or Python scripts in sequence.

Remove or comment out Google Colab-specific commands such as:

- `drive.mount()`
- `files.upload()`
- `files.download()`

---

# 📈 Generated Outputs

The workflow generates:

- Class Distribution
- Language Distribution
- Generator Distribution
- Code Length Analysis
- TF-IDF Feature Analysis
- Correlation Heatmaps
- Human vs AI Radar Charts
- XGBoost Feature Importance
- Model Comparison Charts
- Confusion Matrices
- PCA Visualization
- t-SNE Visualization
- Error Analysis
- Calibration Curves
- submission.csv

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SciPy
- Matplotlib
- Seaborn
- TF-IDF
- Natural Language Processing (NLP)
- Google Colab

---

# 🚀 Future Improvements

- Convert the notebook into a modular Python package.
- Perform automated hyperparameter optimization.
- Evaluate transformer-based code embeddings.
- Improve adversarial robustness.
- Track experiments using MLflow or Weights & Biases.
- Add automated unit tests.
- Deploy the trained model as a web application or REST API.

---

# 👩‍💻 Authors

**Team 7 — AIT 626 Machine Learning Final Project**

Repository maintained by **Dr. Pooja Shah**