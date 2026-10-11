"""
Heart Disease Prediction System (Streamlit app)
AIC354, Assignment 2, Task 2

Loads the trained model (StandardScaler + Support Vector Classifier) saved by
Heart_Disease_Prediction.ipynb and predicts whether a patient has heart disease
from four inputs: age, thalach, exang, oldpeak.

Educational project. Not a medical diagnosis tool.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"

# Same order as the training data (see Step 4 of the notebook).
INPUT_ATTRIBUTES = ["age", "thalach", "exang", "oldpeak"]

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")


@st.cache_resource
def load_model():
    """Load the saved model once (no retraining happens here)."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


st.title("Heart Disease Prediction System")
st.caption(
    "Support Vector Classifier trained on the Cleveland Heart Disease data "
    "(UCI Machine Learning Repository, 303 patients)."
)
st.warning("Educational model. It is not a medical diagnosis tool.")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file 'heart_disease_model.pkl' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()

st.subheader("Enter the patient's details")

age = st.slider("Age (years)", min_value=29, max_value=77, value=58, step=1)
thalach = st.slider(
    "Maximum heart rate achieved during the exercise test (beats per minute)",
    min_value=71, max_value=202, value=120, step=1,
)
exang_label = st.radio(
    "Exercise-induced angina (chest pain during exercise)",
    options=["No", "Yes"],
    horizontal=True,
)
oldpeak = st.slider(
    "ST depression induced by exercise relative to rest (from the ECG)",
    min_value=0.0, max_value=6.2, value=2.0, step=0.1,
)

if st.button("Predict", type="primary"):
    # Same four columns, same order, same numeric format as the training data.
    user_input = pd.DataFrame(
        {
            "age": [float(age)],
            "thalach": [float(thalach)],
            "exang": [1.0 if exang_label == "Yes" else 0.0],
            "oldpeak": [float(oldpeak)],
        }
    )[INPUT_ATTRIBUTES]

    predicted_disease = int(model.predict(user_input)[0])

    st.subheader("Prediction")
    if predicted_disease == 1:
        st.error("HEART DISEASE PRESENT")
    else:
        st.success("NO HEART DISEASE")

    with st.expander("Feature vector sent to the model"):
        st.dataframe(user_input, hide_index=True)

with st.expander("About the inputs"):
    st.markdown(
        """
| Input | Meaning | Range |
|---|---|---|
| age | Age of the patient in years | 29 to 77 |
| thalach | Maximum heart rate achieved during the exercise test | 71 to 202 |
| exang | Exercise-induced angina (0 = No, 1 = Yes) | 0 or 1 |
| oldpeak | ST depression induced by exercise relative to rest | 0.0 to 6.2 |

**Output:** 0 = no heart disease, 1 = heart disease present.
        """
    )
