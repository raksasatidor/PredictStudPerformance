import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("svm_model2.pkl")

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

    
    prediction = model.predict(input_features)

    # Map prediction to category
    prediction_label = {0: 'Weak', 1: 'Good', 2: 'Excellent'}
    predicted_category = prediction_label[prediction[0]]

    st.write(f'Your student performance is: **{predicted_category}**')
