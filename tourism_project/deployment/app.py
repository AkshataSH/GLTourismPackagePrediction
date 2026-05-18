import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# -----------------------------
# Download and load model from Hugging Face Hub
# -----------------------------
model_path = hf_hub_download(
    repo_id="akshatash/TourismPkgPrediction",
    filename="best_tourism_model_v1.joblib",
)
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction App")
st.write(
    """
This application predicts whether a customer is likely to purchase a tourism package.
Please enter the customer details below to get a prediction.
"""
)

# -----------------------------
# User input
# -----------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch", min_value=0, max_value=150, value=15, step=1)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=3, step=1)
numberoffollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3, step=1)
productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
preferredpropertystar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
numberoftrips = st.number_input("Number of Trips", min_value=0, max_value=30, value=2, step=1)
passport = st.selectbox("Passport", [0, 1])
pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
owncar = st.selectbox("Own Car", [0, 1])
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=1, step=1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthlyincome = st.number_input("Monthly Income", min_value=0, max_value=200000, value=25000, step=1000)

input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": typeofcontact,
            "CityTier": citytier,
            "DurationOfPitch": durationofpitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": numberofpersonvisiting,
            "NumberOfFollowups": numberoffollowups,
            "ProductPitched": productpitched,
            "PreferredPropertyStar": preferredpropertystar,
            "MaritalStatus": maritalstatus,
            "NumberOfTrips": numberoftrips,
            "Passport": passport,
            "PitchSatisfactionScore": pitchsatisfactionscore,
            "OwnCar": owncar,
            "NumberOfChildrenVisiting": numberofchildrenvisiting,
            "Designation": designation,
            "MonthlyIncome": monthlyincome,
        }
    ]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success("The customer is likely to purchase the tourism package.")
    else:
        st.warning("The customer is not likely to purchase the tourism package.")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"Purchase Probability: **{probability:.2%}**")

