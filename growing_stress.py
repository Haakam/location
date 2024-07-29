import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function to take inputs and make predictions
def predict(features):
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)
    return prediction

# Streamlit app
st.title("Mental Health Prediction App")

# Define the input features
gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.selectbox("Occupation", ["Corporate", "Housewife", "Others", "Student"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
treatment = st.selectbox("Treatment", ["Yes", "No"])
social_weakness = st.selectbox("Social Weakness", ["Yes", "No"])
coping_struggles = st.selectbox("Coping Struggles", ["Yes", "No"])
mood_swings_le = st.slider("Mood Swings Level", 0, 10)
days_indoors_le = st.slider("Days Spent Indoors", 0, 30)
work_interest_le = st.slider("Work Interest Level", 0, 10)

# Prepare the feature dictionary
features = {
    'Gender_Male': 1 if gender == "Male" else 0,
    'Occupation_Corporate': 1 if occupation == "Corporate" else 0,
    'Occupation_Housewife': 1 if occupation == "Housewife" else 0,
    'Occupation_Others': 1 if occupation == "Others" else 0,
    'Occupation_Student': 1 if occupation == "Student" else 0,
    'self_employed_Yes': 1 if self_employed == "Yes" else 0,
    'family_history_Yes': 1 if family_history == "Yes" else 0,
    'treatment_Yes': 1 if treatment == "Yes" else 0,
    'Social_Weakness_No': 1 if social_weakness == "No" else 0,
    'Social_Weakness_Yes': 1 if social_weakness == "Yes" else 0,
    'Coping_Struggles_Yes': 1 if coping_struggles == "Yes" else 0,
    'Mood_Swings_le': mood_swings_le,
    'Days_Indoors_le': days_indoors_le,
    'Work_Interest_le': work_interest_le
}
# Display the prediction
if st.button("Predict"):
    prediction = predict(features)
    if prediction[0] == 0:
        outcome = "No Stress"
    elif prediction[0] == 1:
        outcome = "Stress"
    elif prediction[0] == 2:
        outcome = "Maybe"
    else:
        outcome = "Unknown"  # In case the prediction is not 0, 1, or 2

    st.write(f"The predicted outcome is: {outcome}")


