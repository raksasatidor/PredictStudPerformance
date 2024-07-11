import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Custom class or function imports (if any)
# from your_module import CustomClass

# Load the trained model
model_filename = 'svm_model2.pkl'

try:
    model = joblib.load(model_filename)
except AttributeError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Label encoders for the categorical features (based on the training data)
label_encoders = {
    'school': {'GP': 0, 'MS': 1},
    'address': {'R': 0, 'U': 1},
    'Mjob': {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'higher': {'no': 0, 'yes': 1}
}

# Dummy scaler data (mean and std from the training dataset)
scaler_means = [0.5, 17, 0.5, 2.5, 2.5, 2, 2.5, 2.5, 0.5, 0.5, 5, 10, 10, 1.5, 2.5]
scaler_stds = [0.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.1, 0.8, 0.7, 0.5, 8.5, 5.5, 5.5, 0.7, 0.8]

# Define the input fields for the selected features
st.title('Student Performance Prediction')

# Creating input fields for each feature
school = st.selectbox('School', ['GP', 'MS'])
school = label_encoders['school'][school]

age = st.selectbox('Age', list(range(15, 23)))

address = st.selectbox('Address', ['Rural', 'Urban'])
address = label_encoders['address'][address[0].upper()]

Medu = st.selectbox('Mother\'s Education', ['None', 'Primary Education', '5th to 9th', 'Secondary Education', 'Higher Education'])
Medu = ['None', 'Primary Education', '5th to 9th', 'Secondary Education', 'Higher Education'].index(Medu)

Fedu = st.selectbox('Father\'s Education', ['None', 'Primary Education', '5th to 9th', 'Secondary Education', 'Higher Education'])
Fedu = ['None', 'Primary Education', '5th to 9th', 'Secondary Education', 'Higher Education'].index(Fedu)

Mjob = st.selectbox('Mother\'s Job', ['at_home', 'health', 'other', 'services', 'teacher'])
Mjob = label_encoders['Mjob'][Mjob]

traveltime = st.selectbox('Travel Time', [1, 2, 3, 4])
studytime = st.selectbox('Study Time', [1, 2, 3, 4])
failures = st.selectbox('Failures', [0, 1, 2, 3])

higher = st.selectbox('Higher Education', ['No', 'Yes'])
higher = label_encoders['higher'][higher.lower()]

absences = st.selectbox('Absences', list(range(0, 94)))

G1 = st.selectbox('Early-year exam Grades (G1)', list(range(0, 21)))
G2 = st.selectbox('Mid-year exam grades (G2)', list(range(0, 21)))

Dalc = st.selectbox('Weekday Alcohol consumption', list(range(1, 6)))
Walc = st.selectbox('Weekend Alcohol consumption', list(range(1, 6)))

# Create a submit button
if st.button('Submit'):
    # Predict the result
    input_features = np.array([school, age, address, Medu, Fedu, Mjob, traveltime, studytime, failures, higher, absences, G1, G2, Dalc, Walc]).reshape(1, -1)
    
    # Normalize the input features using the dummy mean and std from training
    input_features = (input_features - scaler_means) / scaler_stds
    
    try:
        prediction = model.predict(input_features)
        # Map prediction to category
        prediction_label = {0: 'Weak', 1: 'Good', 2: 'Excellent'}
        predicted_category = prediction_label[prediction[0]]

        st.write(f'Your student performance is: **{predicted_category}**')
    except Exception as e:
        st.error(f"Error making prediction: {e}")
