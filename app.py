import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the generalized model
general_model = joblib.load("general_cleanliness_space_types_model.joblib")

# Load the list of features used during training
with open("list_of_space_type_features_used_during_training.txt", "r") as file:
    features_used_during_training = [line.strip() for line in file]


# Create a function for making predictions for a specific record
def predict_for_location(new_record, threshold=0.5):
    # Create a DataFrame for the new record
    new_df = pd.DataFrame([new_record])

    # Reorder the columns to match the order during training
    new_df_encoded = new_df.reindex(columns=features_used_during_training, fill_value=0)

    # Map the Traffic Volumes
    new_df_encoded["Traffic_Volume_High"] = np.where(
        new_df["Traffic_Volume"] == "High", 1, 0
    )
    new_df_encoded["Traffic_Volume_Medium"] = np.where(
        new_df["Traffic_Volume"] == "Medium", 1, 0
    )
    new_df_encoded["Traffic_Volume_Low"] = np.where(
        new_df["Traffic_Volume"] == "Low", 1, 0
    )

    # Map the Space Locations
    new_df_encoded["Location_Type_Conf Rm"] = np.where(
        new_df["Location_Type"] == "Conf Rm", 1, 0
    )
    new_df_encoded["Location_Type_Office"] = np.where(
        new_df["Location_Type"] == "Office", 1, 0
    )
    new_df_encoded["Location_Type_Other"] = np.where(
        new_df["Location_Type"] == "Other", 1, 0
    )
    new_df_encoded["Location_Type_Restroom"] = np.where(
        new_df["Location_Type"] == "Restroom", 1, 0
    )
    new_df_encoded["Location_Type_Stairs"] = np.where(
        new_df["Location_Type"] == "Stairs", 1, 0
    )

    # Convert the timestamp to a Pandas datetime object
    new_df_encoded["Hour_Of_Day"] = new_df["Timestamp"].dt.hour

    new_df_encoded["Day_Of_Week"] = pd.to_datetime(new_df["Timestamp"]).dt.day_of_week

    new_df_encoded["Interaction_Term"] = np.maximum(
        0.1,
        new_df_encoded["Time_Since_Last_Cleaning"]
        * new_df_encoded["Cleanliness_Score"]
        * new_df_encoded["Hour_Of_Day"]
        * new_df_encoded["Time_Since_Last_Inspection"],
    )

    # Scale the Interaction_Term using Min-Max scaling
    scaler = MinMaxScaler()
    new_df_encoded["Interaction_Term"] = scaler.fit_transform(
        new_df_encoded[["Interaction_Term"]]
    )

    # Predict cleanliness label for the new record using the specified threshold
    specific_location_prediction_prob = general_model.predict_proba(new_df_encoded)[
        :, 1
    ][0]
    specific_location_prediction = (
        "Clean" if specific_location_prediction_prob >= threshold else "Not Clean"
    )

    print(
        f'Traffic_Volume_Low: {new_df_encoded["Traffic_Volume_Low"][0]}, Traffic_Volume_Medium: {new_df_encoded["Traffic_Volume_Medium"][0]}, Traffic_Volume_High: {new_df_encoded["Traffic_Volume_High"][0]}'
    )
    print(
        f'Location_Type_Conf Rm: {new_df_encoded["Location_Type_Conf Rm"][0]}, Location_Type_Office: {new_df_encoded["Location_Type_Office"][0]}, Location_Type_Other: {new_df_encoded["Location_Type_Other"][0]}, Location_Type_Restroom: {new_df_encoded["Location_Type_Restroom"][0]}, Location_Type_Stairs: {new_df_encoded["Location_Type_Stairs"][0]}'
    )

    print(f'Hour_Of_Day: {new_df_encoded["Hour_Of_Day"][0]}')
    print(f'Day_Of_Week: {new_df_encoded["Day_Of_Week"][0]}')

    return specific_location_prediction


# Streamlit app
title_text = "<span style= 'font-size: 3em;'>T</span><span style='font-weight: bold; font-size: 3em; font-style: italic;'>A</span><span style= 'font-size: 3em;'>N</span><span style='font-weight: bold; font-size: 3em; font-style: italic;'>I</span><span style= 'font-size: 3em;'>S Recommends</span>"
st.markdown(title_text, unsafe_allow_html=True)

# Input form for the user in the sidebar
with st.sidebar:
    st.header("Input Parameters")
    timestamp_date = st.date_input("Date", pd.to_datetime("today"))
    timestamp_time = st.slider("Hour of Day", 0, 23, 10)
    timestamp = pd.to_datetime(f"{timestamp_date} {timestamp_time}:00:00")

    time_since_last_cleaning = st.slider("Time Since Last Cleaning", 0, 10, 3)
    cleanliness_score = st.slider("Cleanliness Score", 0, 100, 91)
    location_type = st.selectbox(
        "Location Type", ["Restroom", "Office", "Stairs", "Conf Rm", "Other"]
    )
    traffic_volume = st.selectbox("Traffic Volume", ["Low", "Medium", "High"])
    time_since_last_inspection = st.slider("Time Since Last Inspection", 0, 10, 2)

    # Add a slider for the threshold
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5)

# Create a dictionary for the new record
new_record = {
    "Timestamp": timestamp,
    "Time_Since_Last_Cleaning": time_since_last_cleaning,
    "Cleanliness_Score": cleanliness_score,
    "Location_Type": location_type,
    "Traffic_Volume": traffic_volume,
    "Time_Since_Last_Inspection": time_since_last_inspection,
}

# Make a prediction based on user input and the specified threshold
predicted_label = predict_for_location(new_record, threshold)

# Display the prediction result
if predicted_label == "Not Clean":
    st.error("TANIS Recommends to Clean this space.")
else:
    st.success("TANIS Recommends this Space does not need to be cleaned.")
