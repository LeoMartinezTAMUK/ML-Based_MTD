# Malware Detection (MD) Scoring Script
# This script is used for testing the trained machine learning model on new actual (possibly real-time) data

import pickle
import numpy as np

# Load the saved model
with open('malwareDetection.pkl', 'rb') as f: # rb stands for read binary
    loaded_model = pickle.load(f)

# Assuming X_new is your new data for prediction (The Real Data is X_new)
num_samples = 10  # Number of samples (adjust as needed)
num_features = 74  # Number of features (adjust as needed)
X_new = np.zeros((num_samples, num_features))  # Placeholder values are zeros 

# Make predictions using the loaded model (The model is expecting a 2D array (such as an Excel Spreadsheet)
predictions = loaded_model.predict(X_new) # Based on the model, it will generate new predictions for the data it is provided

# The model will make predictions based on both its previous trainig plus the information from the new data
print(predictions)
