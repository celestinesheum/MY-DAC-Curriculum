import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance
import pickle

st.set_page_config(layout="centered", page_title="Mortality Likelihood App")

st.title("Mortality Likelihood Prediction App")

# Define cached loaders
@st.cache_resource
def load_model():
    model = joblib.load('best_xgb_modelb.pkl')
    return model

@st.cache_resource
def load_threshold():
    with open('best_threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return threshold

best_xgb_model = load_model()
best_threshold = load_threshold()

def xgboost_predict(model, X):
    # The model should predict probabilities
    y_probs = model.predict_proba(X)[:, 1]
    pred = (y_probs > best_threshold).astype(int)
    return pred

# Encoding dictionaries
sex_map = {'Male': 1, 'Female': 0}
age_map = {
    '18 to 24': 0, '25 to 29': 1, '30 to 34': 2, '35 to 39': 3,
    '40 to 44': 4, '45 to 49': 5, '50 to 54': 6, '55 to 59': 7,
    '60 to 64': 8, '65 to 69': 9, '70 to 74': 10, '75 to 79': 11,
    '80 or older': 12
}
race_map = {
    'White': 0,
    'Black': 1,
    'Multiracial': 2,
    'Others': 3,
    'Hispanic': 4
}
smoker_map = {
    'never smoked': 0,
    'former smoker': 1,
    'current smoker - now smokes some days': 2,
    'current smoker - now smokes every day': 3
}
last_checkup_map = {
    'within past year (less than 12 months)': 0,
    'within past 2 years (1-2 yrs)': 1,
    'within past 5 years (2-5 yrs)': 2,
    '5 or more years ago': 3
}
yes_no_map = {'Yes': 1, 'No': 0}
removed_teeth_map = {
    'none of them': 0,
    '1 to 5': 1,
    '6 or more, but not all': 2,
    'all': 3
}
ecig_map = {
    'never used e-cigarettes in my entire life': 0,
    'not at all (right now)': 1,
    'use them some days': 2,
    'use them every day': 3
}

def classify_bmi(bmi):
    # Returns numeric category consistent with training
    if bmi < 18.5:
        return 0
    elif 18.5 <= bmi < 25:
        return 1
    elif 25 <= bmi < 30:
        return 2
    else:
        return 3

def classify_sleep_hours(hours):
    # Returns numeric category consistent with training
    if hours < 7:
        return 0
    elif 7 <= hours <= 8:
        return 1
    else:
        return 2

with st.expander("About this Demo"):
    st.write("""
    This application predicts an individual's likelihood of death using an XGBoost model 
    trained on a penalty-encoded dataset. The logic for generating a '1' label uses the 
    thresholds/comorbidity logic without standard scaling.
    """)

tab_inputs, tab_predict, tab_explain = st.tabs(["User Inputs", "Prediction", "Feature Importance"])

with tab_inputs:
    st.subheader("Enter your details")
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Select Sex", ['Male', 'Female'])
        age_category = st.selectbox("Select Age Category", [
            '18 to 24','25 to 29','30 to 34','35 to 39',
            '40 to 44','45 to 49','50 to 54','55 to 59',
            '60 to 64','65 to 69','70 to 74','75 to 79',
            '80 or older'
        ])
        race_ethnicity = st.selectbox("Select Race/Ethnicity", 
            ['White', 'Black', 'Multiracial', 'Others', 'Hispanic']
        )
        bmi = st.number_input("Enter BMI", min_value=10.0, max_value=50.0, value=25.0)
        sleep_hours = st.slider("Sleep Hours per Day", min_value=0, max_value=24, value=7)

    with col2:
        smoker_status_str = st.selectbox("Smoking Status", [
            'never smoked', 'former smoker', 
            'current smoker - now smokes some days', 
            'current smoker - now smokes every day'
        ])
        physical_health_days = st.slider("Poor Physical Health (days, past month)", 0, 30, 0)
        mental_health_days = st.slider("Poor Mental Health (days, past month)", 0, 30, 0)
        last_checkup_time_str = st.selectbox("Last Checkup", [
            'within past year (less than 12 months)',
            'within past 2 years (1-2 yrs)',
            'within past 5 years (2-5 yrs)',
            '5 or more years ago'
        ])
        physical_activities_str = st.selectbox("Physical Activities?", ['Yes', 'No'])

    removed_teeth_str = st.selectbox("Teeth Removed", 
        ['none of them', '1 to 5', '6 or more, but not all', 'all']
    )
    ecigarette_usage_str = st.selectbox("E-cigarette Usage", [
        'never used e-cigarettes in my entire life', 
        'not at all (right now)', 
        'use them some days', 
        'use them every day'
    ])
    chest_scan_str = st.selectbox("Had a chest scan?", ['Yes', 'No'])
    alcohol_drinkers_str = st.selectbox("Drink alcohol?", ['Yes', 'No'])
    hiv_testing_str = st.selectbox("Had HIV testing?", ['Yes', 'No'])
    fluvax_last12_str = st.selectbox("Flu shot last 12 months?", ['Yes', 'No'])
    pneumovax_ever_str = st.selectbox("Pneumonia vaccine ever?", ['Yes', 'No'])

with tab_predict:
    st.subheader("Predict Mortality Risk")
    if st.button("Predict Mortality"):
        with st.spinner('Predicting...'):
            # Convert inputs
            Sex_numeric = sex_map[sex]
            AgeCategory_numeric = age_map[age_category]
            RaceEthnicityCategory_numeric = race_map[race_ethnicity]
            SmokerStatus = smoker_map[smoker_status_str]
            LastCheckupTime = last_checkup_map[last_checkup_time_str]
            PhysicalActivities = yes_no_map[physical_activities_str]
            RemovedTeeth = removed_teeth_map[removed_teeth_str]
            ECigaretteUsage = ecig_map[ecigarette_usage_str]
            ChestScan = yes_no_map[chest_scan_str]
            AlcoholDrinkers = yes_no_map[alcohol_drinkers_str]
            HIVTesting = yes_no_map[hiv_testing_str]
            FluVaxLast12 = yes_no_map[fluvax_last12_str]
            PneumoVaxEver = yes_no_map[pneumovax_ever_str]

            CalculatedBMI_Classification_numeric = classify_bmi(bmi)
            SleepHours_Classification_numeric = classify_sleep_hours(sleep_hours)

            PhysicalHealthDays = physical_health_days
            MentalHealthDays = mental_health_days

            # Ensure column order matches training
            feature_order = [
                'Sex_numeric', 'AgeCategory_numeric', 'RaceEthnicityCategory_numeric',
                'PhysicalHealthDays', 'MentalHealthDays', 'LastCheckupTime',
                'PhysicalActivities', 'RemovedTeeth', 'SleepHours_Classification_numeric',
                'CalculatedBMI_Classification_numeric', 'SmokerStatus', 'ECigaretteUsage',
                'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver'
            ]

            input_data = pd.DataFrame([{
                'Sex_numeric': Sex_numeric,
                'AgeCategory_numeric': AgeCategory_numeric,
                'RaceEthnicityCategory_numeric': RaceEthnicityCategory_numeric,
                'PhysicalHealthDays': PhysicalHealthDays,
                'MentalHealthDays': MentalHealthDays,
                'LastCheckupTime': LastCheckupTime,
                'PhysicalActivities': PhysicalActivities,
                'RemovedTeeth': RemovedTeeth,
                'SleepHours_Classification_numeric': SleepHours_Classification_numeric,
                'CalculatedBMI_Classification_numeric': CalculatedBMI_Classification_numeric,
                'SmokerStatus': SmokerStatus,
                'ECigaretteUsage': ECigaretteUsage,
                'ChestScan': ChestScan,
                'AlcoholDrinkers': AlcoholDrinkers,
                'HIVTesting': HIVTesting,
                'FluVaxLast12': FluVaxLast12,
                'PneumoVaxEver': PneumoVaxEver
            }])[feature_order]

            prediction = xgboost_predict(best_xgb_model, input_data)
        
        result = "Above Average Risk of Death" if prediction[0] == 1 else "Below Average Risk of Death"
        st.write(f"**Prediction: {result}**")

with tab_explain:
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_importance(best_xgb_model, ax=ax, importance_type='gain')
    st.pyplot(fig)
    st.markdown("Features with higher gain are more influential in the model's decision process.")
