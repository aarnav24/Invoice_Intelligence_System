# 📦 Vendor Invoice Intelligence System

An end-to-end data analytics and machine learning system to **predict freight costs** and **detect high-risk vendor invoices**, enabling smarter financial decision-making and reducing manual workload.

---

## 🚀 Overview

This project integrates **SQL, Python, Machine Learning, and Streamlit** to build a complete analytics pipeline that:

- 📊 Forecasts freight costs using regression models  
- 🚨 Flags abnormal invoices requiring manual approval  
- ⚙️ Automates data processing and model inference  
- 📈 Provides an interactive dashboard for real-time predictions  

---

## 🧠 Key Features

### 🚛 Freight Cost Prediction
- Built a regression model (**R² ≈ 97%**) using Linear Regression  
- Identified **invoice value as the primary driver** of freight cost  
- Enables accurate cost estimation for budgeting and vendor negotiations  

---

### 🚨 Invoice Risk Detection
- Developed a classification model (**F1 ≈ 0.95**) using Random Forest  
- Flags invoices based on:
  - Invoice vs purchase mismatch  
  - Receiving delays  
  - Freight anomalies  
- Supports automated prioritization of invoices for manual review  

---

### 📊 Data Analysis & SQL
- Performed **multi-table SQL joins** across vendor invoices and purchase datasets  
- Engineered features such as:
  - Invoice mismatch ratio  
  - Receiving delay  
  - Aggregated purchase metrics  
- Validated insights using **statistical testing (t-test)**  

---

### ⚙️ ML Pipeline
- Modular pipeline for:
  - Data preprocessing  
  - Feature scaling  
  - Model training and evaluation  
  - Inference and prediction  
- Ensured consistency between training and inference  

---

### 🎨 Interactive Dashboard
- Built using **Streamlit**
- Features:
  - Real-time freight prediction  
  - Invoice risk classification  
  - Risk confidence score (`predict_proba`)  
- Designed for business users (finance/operations teams)

---

## 🛠️ Tech Stack

- **Languages:** Python, SQL  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Visualization/UI:** Streamlit  
- **Database:** SQLite  
- **Version Control:** Git, GitHub  

---

## 📁 Project Structure

```bash
Invoice_Intelligence_System/
│
├── src/                            # Core source code
│   ├── freight_cost_prediction/
│   │   ├── data_preprocessing.py
│   │   ├── modelling_evaluation.py
│   │   ├── train.py
│   │
│   ├── invoice_flagging/
│   │   ├── data_preprocessing.py
│   │   ├── modelling_evaluation.py
│   │   ├── train.py
│   │
│   ├── inference/
│   │   ├── predict_freight.py
│   │   ├── predict_invoice_flag.py
│
├── notebooks/                      # EDA & experimentation
│   ├── ingestion_db.ipynb
│   ├── Predicting Freight Cost.ipynb
│   ├── Invoice Flagging.ipynb
│
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/                         # (ignored - generated after training)
├── data/                           # (ignored - datasets not included)
├── logs/                           # (ignored)
├── venv/                           # (ignored)

Note: Data, model artifacts, logs, and virtual environment are excluded from the repository using .gitignore. Run training scripts to generate models locally.
''
---

## ⚡ How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/aarnav24/Invoice_Intelligence_System.git
cd Invoice_Intelligence_System

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Train Models (Optional)
python src/freight_cost_prediction/train.py
python src/invoice_flagging/train.py

### 4️⃣ Run Dashboard
streamlit run app.py
