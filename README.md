# Heart Disease Prediction System (AIC354, Assignment 2, Task 2)

Predicts whether a patient has heart disease (1) or not (0) from 4 inputs:
`age`, `thalach` (max heart rate), `exang` (exercise-induced angina, 0/1), `oldpeak` (ST depression).
Model: Support Vector Classifier with feature scaling (no label encoding is needed, all inputs are numeric).
Data: Cleveland Heart Disease data, UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/45/heart+disease).

## Files
| File | What it is |
|---|---|
| `Heart_Disease_Prediction.ipynb` | Complete notebook: load, EDA, preprocess, train, evaluate, predict, feedback |
| `processed_cleveland.csv` | Input data (UCI `processed.cleveland.data` + a header row) |
| `heart-disease-sample-data.csv` | The 4 selected inputs + output |
| `training-data.csv`, `testing-data.csv` | 80% / 20% stratified split |
| `model-predictions.csv` | Predictions on the testing data |
| `heart_disease_model.pkl` | Trained model (scaler + SVC), loads without retraining |
| `Heart_Disease_Report.pdf` | The notebook exported as PDF |
| `requirements-notebook.txt` | Package versions for running the notebook |
| `app.py` | Streamlit web app that loads the saved model |
| `requirements.txt`, `runtime.txt` | Deployment files for Streamlit Community Cloud |

## Run it
```
python -m venv venv
venv\Scripts\activate        # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements-notebook.txt
jupyter notebook             # open Heart_Disease_Prediction.ipynb, then Kernel -> Restart & Run All
```

## Use the saved model without retraining
```python
import pickle, pandas as pd
model = pickle.load(open("heart_disease_model.pkl", "rb"))
patient = pd.DataFrame({"age": [58.0], "thalach": [120.0], "exang": [1.0], "oldpeak": [2.0]})
print(model.predict(patient)[0])   # 1 = heart disease, 0 = no heart disease
```

Educational project, not a medical diagnosis tool.

## Run the web app locally
```
pip install -r requirements.txt
streamlit run app.py
```
