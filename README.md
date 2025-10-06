# 🏭 Procurement Prediction Engine

## 📘 Overview
The **Procurement Prediction Engine** is a machine learning pipeline designed to generate **plant-item level procurement forecasts** for items ordered across multiple plants.  
The system leverages **XGBoost Regressor** to predict future procurement quantities based on historical purchase data, enabling data-driven inventory and supply chain planning.

---

## ⚙️ Pipeline Structure
The project consists of **four Python modules** that form a sequential data processing and modeling pipeline:

1. **`data_ingestion.py`**  
   - Collects and consolidates raw procurement and material data from multiple sources (e.g., CSV, SQL, API).  
   - Performs schema validation and stores the ingested dataset for downstream processing.

2. **`data_prep.py`**  
   - Cleans, transforms, and enriches the ingested data.  
   - Handles missing values, encodes categorical variables, and performs feature engineering.  
   - Outputs a model-ready dataset for training.

3. **`training.py`**  
   - Trains an **XGBoost Regressor** model on the prepared dataset.  
   - Tunes hyperparameters and evaluates model performance using metrics such as RMSE and MAE.  
   - Saves the trained model and related artifacts for inference.

4. **`forecast.py`**  
   - Loads the trained model and generates **forecasted procurement quantities** at the **plant-item level**.  
   - Outputs predictions in a structured format (e.g., CSV or database table) for business use.

---

## 🧠 Model Details
- **Algorithm:** XGBoost Regressor  
- **Objective:** Predict future procurement quantities per item and plant  
- **Features:** Historical demand, lead times, item category, seasonality indicators, and other relevant operational parameters  
- **Target Variable:** Future procurement quantity  

---

## 🧩 Key Features
- End-to-end modular pipeline  
- Scalable and reusable design  
- Automated feature preparation and model training  
- Configurable forecasting horizon  
- Easy integration with enterprise procurement systems  

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/procurement-prediction-engine.git
   cd procurement-prediction-engine
   ```

2. **Set up environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**
   ```bash
   python data_ingestion.py
   python data_prep.py
   python training.py
   python forecast.py
   ```

4. **View the forecasts**
   - The output forecast file will be generated in `/outputs/forecast_results.csv`

---

## 📊 Example Output
| Plant | Item | Forecast_Date | Forecast_Quantity |
|:------|:-----|:---------------|:------------------|
| P001  | I100 | 2025-10-15     | 320               |
| P002  | I200 | 2025-10-15     | 185               |
| P001  | I300 | 2025-10-15     | 420               |

---

## 📁 Project Structure
```
procurement-prediction-engine/
│
├── data_ingestion.py
├── data_prep.py
├── training.py
├── forecast.py
├── requirements.txt
└── README.md
```

---

## 🧩 Dependencies
- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- matplotlib (optional for visualization)

---

## 🧱 Future Enhancements
- Integrate real-time data ingestion from ERP systems  
- Deploy forecasting as an API or microservice  
- Add model retraining automation  
- Include explainability module for procurement decisions  
