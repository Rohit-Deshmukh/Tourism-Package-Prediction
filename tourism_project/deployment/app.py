
import os
import joblib
import pandas as pd
import streamlit as st


# 1. Page Configuration
st.set_page_config(
    page_title="Holiday Package Classifier",
    layout="centered" 
)


#Model Loading
curr_dir = os.path.dirname(__file__)
file_path = os.path.join(curr_dir, "best_tourism_model_V1.joblib")

try:
    classifier = joblib.load(file_path)
except Exception:
    classifier = None


#Header & Introduction
st.title("Holiday Package Classifier")
st.write(
    "Welcome to the customer conversion tool. Use the forms below to input prospect details "
    "and predict their likelihood of converting on our premium wellness travel packages."
)
st.divider()


#User Inputs
st.subheader("Client Profile")
col_a, col_b = st.columns(2)

with col_a:
    client_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    client_gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    marital_stat = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    income_amt = st.number_input("Monthly Income (₹)", min_value=0.0, value=35000.0, step=2500.0)
    
with col_b:
    job_role = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])
    corp_title = st.selectbox("Corporate Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    urban_tier = st.radio("City Tier Level", [1, 2, 3], horizontal=True)

st.write("")
st.subheader("Pitch & Engagement")
col_c, col_d = st.columns(2)

with col_c:
    reach_out_method = st.selectbox("Acquisition Channel", ["Self Enquiry", "Company Invited"])
    pitch_mins = st.slider("Pitch Duration (Mins)", 0.0, 90.0, 10.0, 1.0)
    follow_up_cnt = st.number_input("Prior Follow-ups", 0, 10, 2)
    satisfaction_rating = st.slider("Pitch Rating (1-5)", 1, 5, 4)
    pkg_offered = st.selectbox("Package Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    
with col_d:
    travelers_count = st.number_input("Total Travelers", 1, 10, 2)
    kids_count = st.number_input("Children Included", 0, 5, 0)
    trips_per_yr = st.number_input("Annual Vacations", 0, 20, 2)
    hotel_stars = st.selectbox("Hotel Preference (Stars)", [3.0, 4.0, 5.0])
    has_passport = st.checkbox("Holds Valid Passport?")
    owns_vehicle = st.checkbox("Owns a Car?")


# 5. Data Transformation
# Encoders organized cleanly as constants
MAP_CONTACT = {"Company Invited": 0, "Self Enquiry": 1}
MAP_JOB = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
MAP_GENDER = {"Female": 0, "Male": 1}
MAP_PKG = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
MAP_STATUS = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
MAP_ROLE = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

# Constructing the inference dataframe
features_df = pd.DataFrame([{
    'Age': client_age,
    'TypeofContact': MAP_CONTACT[reach_out_method],
    'CityTier': urban_tier,
    'DurationOfPitch': pitch_mins,
    'Occupation': MAP_JOB[job_role],
    'Gender': MAP_GENDER[client_gender],
    'NumberOfPersonVisiting': travelers_count,
    'NumberOfFollowups': follow_up_cnt,
    'ProductPitched': MAP_PKG[pkg_offered],
    'PreferredPropertyStar': hotel_stars,
    'MaritalStatus': MAP_STATUS[marital_stat],
    'NumberOfTrips': trips_per_yr,
    'Passport': 1 if has_passport else 0,
    'PitchSatisfactionScore': satisfaction_rating,
    'OwnCar': 1 if owns_vehicle else 0,
    'NumberOfChildrenVisiting': kids_count,
    'Designation': MAP_ROLE[corp_title],
    'MonthlyIncome': income_amt
}])


#Prediction
st.write("") 
if st.button("Predict Conversion", type="primary", use_container_width=True):
    
    if not classifier:
        st.error("System Error: Model weights not found. Ensure 'best_tourism_model_V1.joblib' is in the directory.")
    else:
        with st.spinner("Analyzing profile..."):
            
            # Execute inference
            label_pred = classifier.predict(features_df)[0]

            st.markdown("### Analysis Complete")
            
            # Text-only UI routing based on prediction
            if label_pred == 1:
                st.markdown("#### High Conversion Potential")
                st.write(f"This prospect strongly matches the profile of a converted customer.")
                
                
            else:
                st.markdown("#### Low Conversion Potential")
                st.write(f"This prospect is currently unlikely to purchase the wellness package.")
                


#Footer
st.write("")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
    "Visit with Us - Wellness Tourism Package Classifier<br>"
    "Powered by XGBoost & Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
