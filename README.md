# 🏭 Predictive Maintenance Dashboard  

[![CI](https://github.com/your-username/predictive-maintenance-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/guilherme-leite-eng/predictive-maintenance-dashboard/actions)  

---

## 📌 Project Overview  
Unplanned equipment failures are one of the biggest cost drivers in manufacturing.  
This project delivers a **Predictive Maintenance Machine Learning pipeline** + **interactive Streamlit dashboard** that:  

- Predicts machine failures **before they happen** 🛠️  
- Supports multiple **failure modes** (tool wear, overheating, etc.)  
- Provides **cost-aware decision support** (false alarms vs downtime)  
- Offers **real-time what-if analysis** for engineers and operators  

Built as part of my **Big Data in Industry subject in my postgraduate program in Advanced Industrial Automations**, this is structured as a **real production-grade project** (not just a notebook).  

---

## 🚀 Features  

✅ **End-to-End ML Pipeline**  
- Data preprocessing (scaling, encoding, stratified split)  
- Balanced Random Forest model with evaluation  
- Model artifact saving (`joblib`) for reproducibility  

✅ **Interactive Dashboard (Streamlit)**  
- Key KPIs: Accuracy, Precision, Recall, F1  
- Confusion Matrix heatmap  
- Cost-based failure simulation (business impact)  
- Single prediction input (manual machine parameters)  
- Batch CSV upload for scoring entire datasets  
- SHAP explainability plots (why the model made a prediction)  

✅ **Deployment-Ready**  
- Clean repo structure (`src/`, `dashboard/`, `.github/`)  
- CI/CD with GitHub Actions (lint + smoke test)  
- Dockerfile for containerization  
- Live demo on Streamlit Cloud  

---

## 📂 Repository Structure  
predictive-maintenance-dashboard/
│
├── dashboard/ # Streamlit app
│ └── app.py
│
├── src/ # Training pipeline
│ └── train.py
│
├── data/ # Sample dataset (tiny subset for demo)
│ └── predictive_maintenance.csv
│
├── artifacts/ # Trained model (.joblib)
│
├── .github/workflows/ # CI pipeline
│ └── ci.yml
│
├── requirements.txt # Python dependencies
├── Dockerfile # Container setup
├── README.md # This file
├── LICENSE # MIT License
└── .gitignore

## ⚙️ Tech Stack  

- **Python 3.10**  
- **Pandas / NumPy / Scikit-Learn** → ML pipeline  
- **Streamlit + Plotly** → dashboard & visualization  
- **SHAP** → explainability  
- **GitHub Actions** → CI/CD  
- **Docker** → containerization  

---

## 🔬 Training & Evaluation  

To retrain the model:  
### Run training
python src/train.py

### Model artifact saved at: artifacts/pm_pipeline.joblib
The script prints out a classification report (Precision, Recall, F1) and updates the trained model in artifacts/
---
## 📊 Business Value

- **Reduce downtime** → early detection prevents catastrophic failures.
- **Optimize maintenance scheduling** → avoid unnecessary service.
- **Cut costs** → balance between false alarms and missed failures.
- **Engineer explainability** → SHAP values highlight why the model flagged risk.
This mirrors industrial practices at Siemens, Bosch, GE, Porsche, and other leaders in predictive analytics.

## Local Setup

git clone https://github.com/your-username/predictive-maintenance-dashboard.git
cd predictive-maintenance-dashboard
pip install -r requirements.txt

streamlit run dashboard/app.py

## License
MIT License — free to use, adapt, and share.
