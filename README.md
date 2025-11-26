This project implements an end-to-end multivariate time-series forecasting workflow using a sequence model (LSTM) and SHAP-based interpretability.
A synthetic industrial sensor dataset is programmatically generated to include trend, multiple seasonal behaviors, and noise. Sequence windows are used to forecast one-step-ahead sensor readings.
The LSTM architecture undergoes systematic hyperparameter tuning (layers, hidden units, learning rate, batch size) and is evaluated using MAE, RMSE and MAPE.
SHAP explainability identifies which sensors and which time steps contribute most to the forecast, offering transparent feature and temporal importance.
# 📈 Advanced Time Series Forecasting with Deep Learning & Explainability (LSTM + SHAP)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository implements an **end-to-end multivariate time-series forecasting pipeline** using a sequence neural network (LSTM) combined with **SHAP explainability**. The project includes realistic synthetic industrial sensor data generation, model training, hyperparameter tuning, performance evaluation, and interpretability analysis.

---

## 🏭 Dataset Description

Instead of raw industrial data with missing entries, a dataset is **programmatically generated** to mimic real-world sensor characteristics:

- 5 sensor signals (`sensor_1` to `sensor_5`)
- 2000+ time steps
- Clear **trend**, **multiple seasonalities (30, 50, 70)**, and **Gaussian noise**
- Target variable: **one-step-ahead forecast of `sensor_1`**

This approach ensures strong realism for modelling, while avoiding irrelevant data quality noise.

---

## 🧠 Model Architecture (LSTM)

A **many-to-one stacked LSTM** network is used for forecasting:


### 🔧 Sequence Windowing & Batching

The dataset is transformed using overlapping windows of **30 past readings → 1 future prediction**, and split in time order:

- 70% train
- 15% validation
- 15% test

---

## 🔍 Hyperparameter Optimization

A grid search is performed over:

| Parameter | Values |
|-----------|--------|
| LSTM Layers | 1, 2 |
| Hidden Units | 32, 64 |
| Learning Rate | 0.001, 0.0003 |
| Batch Size | 32, 64 |

The best configuration is selected using the **lowest validation MAE** and re-trained on `train + validation`.

> 📌 After running the notebook, replace the placeholder below with your real parameters:

**Best Configuration Found:**

---

## 📊 Model Evaluation (Test Set)

After final training, the model is evaluated using:

- **MAE – Mean Absolute Error**
- **RMSE – Root Mean Squared Error**
- **MAPE – Mean Absolute Percentage Error**

> 📌 Replace placeholders after running:


📌 Visualization (Required in Report):

- True vs Predicted Curve 🌟

> _Add your screenshot here_

---

## 🧾 Explainability: SHAP Insights

To understand **which sensors and which past time steps matter most**, SHAP (DeepExplainer) is applied:

📌 **Feature Importance**
Determines which sensors contributed most to the forecast.
> Expected result: `sensor_1` highest, correlated sensors also contribute.

📌 **Time-Step Importance**
Shows which positions in the 30-step window dominate the prediction.
> Expected result: Recent past values (last 5–10 steps) weigh more.

📍 _Add your plots:_

---

## 💻 How to Run

### 🔽 Install Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib shap
python main_lstm_time_series_project.py
📁 root
├── 📄 main_lstm_time_series_project.py   # Full pipeline script
├── 📔 TimeSeries_LSTM_SHAP.ipynb         # Notebook version
├── 📄 README.md                          # Project documentation
└── 📊 /screenshots                       # (Add your result screenshots)
