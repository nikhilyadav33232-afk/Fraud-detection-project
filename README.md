# 🛡️ PS-02 — Real-Time Fraud & Anomaly Detection

> **AI-powered transaction fraud detection system with real-time risk analysis, anomaly detection, explainable decisions, and a Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Academic%20Project-success)

---

## 📌 Overview

**PS-02 — Real-Time Fraud & Anomaly Detection** is a machine-learning based fraud detection system designed to analyze financial transactions and identify potentially fraudulent or anomalous activity.

The system combines:

* 🌲 **Random Forest** for supervised fraud classification
* 🌳 **Extra Trees** for model comparison
* 🚨 **Isolation Forest** for anomaly detection
* 📊 **Risk scoring** for transaction-level assessment
* 🧠 **Rule-based explanations** for suspicious transactions
* ⚡ **Inference latency measurement**
* 🖥️ **Streamlit dashboard** for real-time interaction

The project is designed as a simple and understandable implementation suitable for an **academic project, demonstration, and viva presentation**.

---

# ✨ Key Features

### 🔍 Fraud Classification

The system predicts whether a transaction is:

```text
0 → Normal
1 → Fraud
```

The primary classification model is **Random Forest Classifier**.

---

### 🚨 Anomaly Detection

An additional **Isolation Forest** model detects transactions that appear unusual compared with the learned transaction patterns.

This provides a second layer of detection beyond supervised fraud classification.

---

### 📊 Fraud Probability

For every new transaction, the system calculates a fraud probability.

Example:

```text
Fraud Probability: 94.62%
Risk Level: HIGH
Decision: BLOCK / REVIEW
```

---

### 🚦 Risk Classification

Transactions are categorized into three risk levels:

| Fraud Probability | Risk      |
| ----------------- | --------- |
| `< 40%`           | 🟢 LOW    |
| `40% – 74%`       | 🟡 MEDIUM |
| `≥ 75%`           | 🔴 HIGH   |

---

### 🧠 Explainable Fraud Detection

The system provides simple explanations based on suspicious transaction signals.

For example:

```text
High transaction amount
High transaction frequency
Previous fraud history
Unusual geographic distance
```

This makes the model output easier to understand during analysis and presentation.

---

### ⚡ Real-Time Processing

The system processes transactions individually and measures inference latency.

Example:

```text
Processing Latency: 1.24 ms
```

This demonstrates the ability to perform fast transaction-level scoring.

---

### 📈 Feature Importance

Random Forest feature importance is displayed to understand which transaction attributes contribute most to the model's decisions.

The main features are:

* Transaction Amount
* Transactions in Last 1 Hour
* Previous Fraud History
* Geographic Distance

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Transaction Input  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Feature Processing  │
                    │                      │
                    │ Amount               │
                    │ Transactions_1H      │
                    │ Previous_Fraud       │
                    │ Distance_KM          │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
      ┌──────────────────┐          ┌──────────────────┐
      │  Random Forest   │          │ Isolation Forest │
      │ Fraud Classifier │          │ Anomaly Detector │
      └────────┬─────────┘          └────────┬─────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   Risk Assessment   │
                   ├─────────────────────┤
                   │ Fraud Probability   │
                   │ Risk Level          │
                   │ Anomaly Status      │
                   │ Decision            │
                   │ Explanation         │
                   │ Latency             │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Streamlit Dashboard│
                   └─────────────────────┘
```

---

# 📂 Project Structure

```text
PS02-Fraud-Detection/
│
├── app.py
│
├── fraud_detection_model.pkl
│
├── fraud_detection.ipynb
│
├── requirements.txt
│
├── README.md
│
└── screenshots/
    └── dashboard.png
```

> The exact files can vary depending on how the project is organized locally.

---

# 🧰 Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Core programming language             |
| Pandas           | Data processing                       |
| NumPy            | Numerical operations                  |
| Scikit-learn     | Machine learning                      |
| Random Forest    | Fraud classification                  |
| Extra Trees      | Model comparison                      |
| Isolation Forest | Anomaly detection                     |
| Joblib           | Model serialization                   |
| Streamlit        | Web UI/dashboard                      |
| Jupyter Notebook | Model development and experimentation |

---

# 📊 Dataset

For this academic implementation, a small manually created transaction dataset is used.

The dataset contains the following features:

```text
Amount
Transactions_1H
Previous_Fraud
Distance_KM
Fraud
```

Example:

| Amount | Transactions/1H | Previous Fraud | Distance | Fraud |
| -----: | --------------: | -------------: | -------: | ----: |
|    100 |               1 |              0 |        2 |     0 |
|   5000 |              10 |              1 |      500 |     1 |
|    250 |               1 |              0 |        6 |     0 |
|  15000 |              18 |              1 |      900 |     1 |

This dataset is intentionally small so that the complete project remains easy to understand and demonstrate.

---

# 🤖 Machine Learning Models

## 1. Random Forest

Random Forest is the primary supervised classification model.

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    class_weight="balanced",
    random_state=21
)
```

### Why Random Forest?

* Works well with tabular data
* Handles nonlinear relationships
* Provides probability estimates
* Provides feature importance
* Simple to explain

---

## 2. Extra Trees

Extra Trees is used as a comparison model.

```python
ExtraTreesClassifier(
    n_estimators=150,
    max_depth=6,
    class_weight="balanced",
    random_state=21
)
```

This allows the project to compare two tree-based ensemble approaches.

---

## 3. Isolation Forest

Isolation Forest is used for unsupervised anomaly detection.

```python
IsolationForest(
    n_estimators=100,
    contamination=0.20,
    random_state=21
)
```

It provides an additional signal indicating whether a transaction appears anomalous.

---

# ⚙️ How the Detection Works

For every incoming transaction:

### Step 1 — Receive transaction

```text
Amount
Transactions in 1 Hour
Previous Fraud
Distance
```

### Step 2 — Random Forest prediction

The classifier predicts:

```text
Normal / Fraud
```

and calculates:

```text
Fraud Probability
```

### Step 3 — Anomaly detection

Isolation Forest determines:

```text
NORMAL
or
ANOMALOUS
```

### Step 4 — Risk calculation

The fraud probability is converted into:

```text
LOW
MEDIUM
HIGH
```

### Step 5 — Final decision

The system produces:

```text
ALLOW
```

or

```text
BLOCK / REVIEW
```

### Step 6 — Explanation

Suspicious signals are displayed to the user.

### Step 7 — Latency

The processing time is measured in milliseconds.

---

# 🖥️ Streamlit Dashboard

The project includes a simple interactive dashboard.

The user can enter:

```text
Transaction Amount
Transactions in Last 1 Hour
Previous Fraud History
Distance in KM
```

Then click:

```text
🔍 ANALYZE TRANSACTION
```

The dashboard returns:

```text
Fraud Probability
Risk Level
Decision
Anomaly Status
Explanation
Processing Latency
```

It also displays:

* Recent transaction history
* Feature importance
* Training dataset
* System information

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PS02-Fraud-Detection.git
```

Move into the project directory:

```bash
cd PS02-Fraud-Detection
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

Run:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

Typical local address:

```text
http://localhost:8501
```

---

# 📓 Run the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
fraud_detection.ipynb
```

Run the cells sequentially.

---

# 🧪 Example Transaction

### Normal Transaction

```text
Amount: 120
Transactions/1H: 2
Previous Fraud: No
Distance: 5 KM
```

Possible result:

```text
Risk Level: LOW
Decision: ALLOW
```

---

### Suspicious Transaction

```text
Amount: 9000
Transactions/1H: 14
Previous Fraud: Yes
Distance: 700 KM
```

Possible result:

```text
Risk Level: HIGH
Decision: BLOCK / REVIEW
Anomaly: ANOMALOUS
```

---

# 📈 Evaluation Metrics

The notebook evaluates the classification model using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The system also measures:

* Transaction inference latency
* Feature importance

---

# 🎯 PS-02 Requirement Mapping

| Requirement         | Implementation                 |
| ------------------- | ------------------------------ |
| Fraud Detection     | ✅ Random Forest                |
| Class Imbalance     | ✅ `class_weight="balanced"`    |
| Anomaly Detection   | ✅ Isolation Forest             |
| Real-Time Scoring   | ✅ Transaction-level prediction |
| Risk Classification | ✅ Low / Medium / High          |
| Explainability      | ✅ Rule-based explanation       |
| Latency Measurement | ✅ Millisecond inference time   |
| Feature Importance  | ✅ Random Forest                |
| Interactive UI      | ✅ Streamlit                    |

### Current simplified implementation

The current version is intentionally simple. Advanced streaming features such as:

* Online learning with `partial_fit()`
* Advanced concept-drift detection
* Dynamic model retraining
* Graph-based transaction relationships
* PR-AUC evaluation

can be added as future extensions.

---

# 🔮 Future Improvements

The project can be extended into a production-oriented streaming architecture.

### Possible improvements

* Apache Kafka for transaction streaming
* Apache Flink / Spark Streaming
* Online learning with `partial_fit()`
* Advanced concept drift detection
* Graph-based fraud detection
* Real-time databases
* User/device/IP relationship analysis
* PR-AUC and ROC-AUC monitoring
* Model monitoring and retraining
* Authentication and role-based access
* Cloud deployment

---

# ⚠️ Disclaimer

This project is an **academic/demo implementation**.

The dataset is small and manually generated, so the model's performance should not be interpreted as real-world financial fraud detection accuracy.

For production use, a much larger and representative transaction dataset, proper validation, privacy controls, security measures, model monitoring, and domain-specific fraud rules would be required.

---

# 👨‍💻 Project

**Project:** PS-02 — Real-Time Algorithmic Fraud & Anomaly Detection in Streaming Data

**Category:** Machine Learning / Fraud Detection / Anomaly Detection

**Interface:** Streamlit

**Language:** Python

---

## ⭐ If you found this project useful

Give the repository a ⭐ and feel free to explore, improve, and extend the project.

**Built for learning, experimentation, and academic demonstration.**


### 🚨 Anomaly Detection

An additional **Isolation Forest** model detects transactions that appear unusual compared with the l
