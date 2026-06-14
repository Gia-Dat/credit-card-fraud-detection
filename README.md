# Credit Card Fraud Detection

An end-to-end Machine Learning and MLOps project designed to build, deploy, and monitor a robust credit card fraud detection system.

---

## 📌 Project Overview
The primary objective of this project is to train and deploy a machine learning model capable of accurately classifying transactions as either legitimate or fraudulent. 

### 🎯 Purpose
This repository serves as a comprehensive, hands-on learning project to bridge the gap between traditional machine learning workflows and production-ready MLOps practices. The roadmap is divided into two distinct phases.
clea
### 🚀 Key Benefits & Value Add
* **Financial Risk Mitigation:** Minimizes monetary losses by identifying and blocking fraudulent activities in real-time before transactions clear.
* **Proactive Model Health Tracking:** Implements advanced monitoring to detect data drift and performance degradation early, preventing stale models from making bad predictions in production.
* **Scalable & Reproducible Infrastructure:** Utilizes Infrastructure as Code (IaC) and containerization, allowing the entire ecosystem to be destroyed, rebuilt, or scaled instantly across cloud environments.
* **Production-Grade Code Quality:** Enforces CI/CD pipelines, automated testing, and strict linting to ensure that codebase updates are reliable, secure, and deployment-ready without manual intervention.

---

## 🗺️ Project Roadmap

### Phase 1: Traditional Machine Learning
This phase focuses on the core data science workflow, moving from raw data to a containerized, reproducible script.
* **Data Collection & Ingestion:** Sourcing and loading the transaction dataset.
* **Exploratory Data Analysis (EDA):** Identifying patterns, correlation, and class imbalances.
* **Data Preprocessing:** Feature engineering, handling missing values, and scaling.
* **Model Training & Hyperparameter Tuning:** Evaluating multiple algorithms to find the optimal model.
* **Productionization:** Refactoring Jupyter notebooks into clean, modular Python scripts.
* **Environment Management:** Ensuring reproducibility using virtual environments (e.g., Virtualenv, Poetry, or Conda).
* **Containerization:** Packaging the application using Docker.

### Phase 2: MLOps & Production Engineering
This phase focuses on scaling, automating, deploying, and monitoring the model using industry-standard DevOps and MLOps tools.
* **Cloud Infrastructure:** Developing on AWS and managing resources using Infrastructure as Code (IaC) via **Terraform**.
* **Experiment Tracking & Model Registry:** Logging parameters, metrics, and artifacts using **MLflow**.
* **CI/CD Pipeline:** Automating testing and deployment workflows.
* **Code Quality & Automation:** * Implementing **Linters** and **Code Formatters** (e.g., Flake8, Black).
    * Using **Pre-commit hooks** to enforce code quality before commits.
    * Streamlining tasks with **Makefiles** and `make` commands.
* **Testing:** Writing robust **Unit Tests** and **Integration Tests** (e.g., using PyTest).
* **Deployment:** Serving the model as a production API.
* **Model Monitoring:** Tracking data drift and performance over time using **Evidently AI** and visualizing metrics via a **Grafana** dashboard.

---

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-Learn, Pandas, NumPy
* **MLOps & DevOps:** MLflow, Docker, Terraform, AWS
* **Monitoring:** Evidently AI, Grafana
* **CI/CD & Automation:** GitHub Actions, Pre-commit, Make

## ⚙️ How to Run the Project