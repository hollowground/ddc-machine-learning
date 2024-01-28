import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from the CSV file
df = pd.read_csv("cleanliness_data_space_types.csv")

# Extract hour of the day and day of the week from the timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour_Of_Day"] = df["Timestamp"].dt.hour
df["Day_Of_Week"] = df["Timestamp"].dt.dayofweek

# Add the next scheduled clean datetime
df["Next_Scheduled_Clean"] = df["Timestamp"] + pd.to_timedelta(
    df["Time_Since_Last_Cleaning"], unit="H"
)

# Add an inspection datetime based on the next scheduled clean
df["Inspection_Datetime"] = (
    df["Timestamp"]
    - pd.to_timedelta(df["Time_Since_Last_Cleaning"], unit="H")
    + pd.to_timedelta(2, unit="H")
)

# Add an additional column called time since last inspection
df["Time_Since_Last_Inspection"] = df["Timestamp"] - df["Inspection_Datetime"]

# Convert time since last inspection to hours
df["Time_Since_Last_Inspection"] = (
    df["Time_Since_Last_Inspection"].dt.total_seconds() / 3600
)

# One-hot encode the 'Location_Type' and 'Traffic_Volume' columns
df_encoded = pd.get_dummies(df, columns=["Location_Type", "Traffic_Volume"])

# Create interaction term with a minimum value of 0.1
df_encoded["Interaction_Term"] = np.maximum(
    0.1,
    df_encoded["Time_Since_Last_Cleaning"]
    * df_encoded["Cleanliness_Score"]
    * df_encoded["Hour_Of_Day"]
    * df_encoded["Time_Since_Last_Inspection"],
)

df["Interaction_Term"] = df_encoded["Interaction_Term"]

# Scale the Interaction_Term using Min-Max scaling
scaler = MinMaxScaler()
df_encoded["Interaction_Term"] = scaler.fit_transform(df_encoded[["Interaction_Term"]])

# Features (X) and target variable (y) for training
X_train = df_encoded[
    [
        "Time_Since_Last_Cleaning",
        "Time_Since_Last_Inspection",  # Add the new column
        "Cleanliness_Score",
        "Interaction_Term",
        "Hour_Of_Day",
        "Day_Of_Week",
    ]
    + list(df_encoded.filter(regex="Location_Type_"))
    + list(df_encoded.filter(regex="Traffic_Volume_"))
]
y_train = df_encoded["Cleanliness_Label"]

# Save the list of features to a text file
with open("list_of_space_type_features_used_during_training.txt", "w") as file:
    features_used_during_training = list(X_train.columns)
    for feature in features_used_during_training:
        file.write(f"{feature}\n")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Print the correlation matrix
correlation_matrix = X_train.corr()

# Find features with the highest positive correlation
max_corr_positive = correlation_matrix.abs().unstack().sort_values(ascending=False)
pairs_high_corr_positive = max_corr_positive[max_corr_positive < 1].head(
    5
)  # Display top 5
print("\nTop Positive Correlations:")
print(pairs_high_corr_positive)

# Find features with the highest negative correlation
max_corr_negative = -(correlation_matrix.abs().unstack().sort_values(ascending=False))
pairs_high_corr_negative = max_corr_negative[max_corr_negative < -0.5].head(
    5
)  # Display top 5
print("\nTop Negative Correlations:")
print(pairs_high_corr_negative)

# Display a heatmap for the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Initialize the logistic regression model
# general_model = LogisticRegression(solver="sag", max_iter=1000)
general_model = LogisticRegression(max_iter=1000)

# Train the general model
general_model.fit(X_train, y_train)
joblib.dump(general_model, "general_cleanliness_space_types_model.joblib")

# Convert 'Clean' and 'Not Clean' to binary labels (0 and 1)
y_test_binary = (y_test == "Not Clean").astype(int)

# Predictions on the test set with adjusted threshold
adjusted_threshold = 0.7  # Experiment with different threshold values
predictions = (general_model.predict_proba(X_test)[:, 1] > adjusted_threshold).astype(
    int
)

# Evaluate the general model
accuracy = accuracy_score(y_test_binary, predictions)
print("General Model Accuracy:", accuracy)

# Classification report for the general model
print("\nClassification Report for General Model:")
print(classification_report(y_test_binary, predictions))

# After fitting the general model, you can access feature importance (if applicable)
feature_importance = general_model.coef_[0]

# Print or analyze feature importance for the general model
for feature, importance in zip(features_used_during_training, feature_importance):
    print(f"{feature}: {importance}")
