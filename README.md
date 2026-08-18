# IIoT Predictive Edge Engine

An enterprise real-time IIoT network monitoring and predictive edge computing dashboard. The application uses a hybrid machine learning pipeline combining **unsupervised clustering (DBSCAN/K-Means)** for dynamic network state identification and **supervised regression tree ensembles (XGBoost)** to forecast network latency and dynamically trigger Quality of Service (QoS) mitigations and high-availability hardware failovers.

---

## 🛠️ Project Structure

- **`src/`**
  - [data_pipeline.py](file:///Users/deva135mac/Desktop/ECE_PBL_Project/src/data_pipeline.py): Simulates industrial sensor traffic logs, handles forward-fill missing values, and performs Min-Max feature scaling.
  - [clustering_engine.py](file:///Users/deva135mac/Desktop/ECE_PBL_Project/src/clustering_engine.py): Runs unsupervised density profiling to isolate anomalies and assign cluster state signatures.
  - [ensemble_training.py](file:///Users/deva135mac/Desktop/ECE_PBL_Project/src/ensemble_training.py): Trains the low-overhead XGBoost latency predictor on labeled edge telemetry.
  - [app.py](file:///Users/deva135mac/Desktop/ECE_PBL_Project/src/app.py): Streamlit operations center dashboard with glassmorphism HUD cards, live SVGs, and real-time alerts.
- **`tests/`**
  - [test_pipeline.py](file:///Users/deva135mac/Desktop/ECE_PBL_Project/tests/test_pipeline.py): Python pytest suite verifying each step of the pipeline.
- [requirements.txt](file:///Users/deva135mac/Desktop/ECE_PBL_Project/requirements.txt): Complete dependency list.

---

## ⚙️ Setup Instructions

### 1. Initialize Virtual Environment
Initialize a local Python 3 virtual environment in the project root:
```bash
python3 -m venv .venv
```

### 2. Install Dependencies
Activate the virtual environment (optional) or run pip directly to install the required libraries:
```bash
.venv/bin/pip install -r requirements.txt
```

### 3. Run Pipeline Tests
Ensure that the entire ML telemetry pipeline executes correctly:
```bash
.venv/bin/pytest tests/test_pipeline.py
```

### 4. Launch the Dashboard
Run the Streamlit app server:
```bash
.venv/bin/streamlit run src/app.py --server.port 8505
```
Open your browser and navigate to `http://localhost:8505` to view the running dashboard.
